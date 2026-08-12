"""Genera candidatos de productos nuevos con costo computacional controlado."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
DEFAULT_OUTPUT = PROCESSED_DIR / "candidatos_nuevos.parquet"

MIN_CANDIDATOS = 20
TOP_PASILLOS = 3
TOP_DEPARTAMENTOS = 2
PRODUCTOS_POR_PASILLO = 30
PRODUCTOS_POR_DEPARTAMENTO = 40
PRODUCTOS_GLOBALES = 100
PRODUCTOS_RESPALDO = 1000
PRESELECCION_PASILLO = 35
PRESELECCION_DEPARTAMENTO = 25
PRESELECCION_GLOBAL = 40


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def validar_entradas() -> dict[str, Path]:
    paths = {
        "usuarios": PROCESSED_DIR / "usuarios.parquet",
        "productos": PROCESSED_DIR / "productos.parquet",
        "interacciones": PROCESSED_DIR / "interacciones.parquet",
    }
    faltantes = [str(p) for p in paths.values() if not p.exists()]
    if faltantes:
        raise FileNotFoundError("Faltan tablas procesadas:\n- " + "\n- ".join(faltantes))
    return paths


def tamanio_directorio(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def formato_bytes(n: int) -> str:
    unidades = ["B", "KiB", "MiB", "GiB", "TiB"]
    valor = float(n)
    for unidad in unidades:
        if valor < 1024 or unidad == unidades[-1]:
            return f"{valor:.2f} {unidad}"
        valor /= 1024
    return f"{valor:.2f} TiB"


def ejecutar_etapa(con: duckdb.DuckDBPyConnection, nombre: str, sql: str, contar: str | None = None) -> None:
    inicio = time.perf_counter()
    con.execute(sql)
    filas = con.execute(f"SELECT COUNT(*) FROM {contar}").fetchone()[0] if contar else None
    detalle = f" | {filas:,} filas" if filas is not None else ""
    print(f"   [OK] {nombre}: {time.perf_counter() - inicio:.1f} s{detalle}", flush=True)


def configurar(con: duckdb.DuckDBPyConnection, memory_limit: str, threads: int, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit = '{memory_limit.replace(chr(39), chr(39) * 2)}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order = false")


def crear_fuentes(con: duckdb.DuckDBPyConnection, paths: dict[str, Path], max_usuarios: int | None) -> int:
    limite = f"LIMIT {max_usuarios}" if max_usuarios else ""
    ejecutar_etapa(
        con,
        "selección de usuarios",
        f"""CREATE TEMP TABLE usuarios_trabajo AS
        SELECT * FROM read_parquet('{sql_path(paths['usuarios'])}')
        ORDER BY user_id {limite}""",
        "usuarios_trabajo",
    )
    ejecutar_etapa(
        con,
        "interacciones del universo",
        f"""CREATE TEMP TABLE interacciones_trabajo AS
        SELECT i.*
        FROM read_parquet('{sql_path(paths['interacciones'])}') i
        SEMI JOIN usuarios_trabajo u USING (user_id)""",
        "interacciones_trabajo",
    )
    con.execute(
        f"CREATE VIEW productos AS SELECT * FROM read_parquet('{sql_path(paths['productos'])}')"
    )
    return con.execute("SELECT COUNT(*) FROM usuarios_trabajo").fetchone()[0]


def construir(con: duckdb.DuckDBPyConnection, min_candidatos: int) -> None:
    ejecutar_etapa(
        con,
        "ranking acotado de productos",
        f"""CREATE TEMP TABLE productos_rank AS
        WITH s AS (
          SELECT product_id, aisle_id, department_id,
                 .70 * LN(1 + cantidad_compras) / NULLIF(MAX(LN(1 + cantidad_compras)) OVER (), 0)
                 + .30 * COALESCE(reorder_rate_producto, 0) score_producto
          FROM productos
        )
        SELECT *,
          ROW_NUMBER() OVER (PARTITION BY aisle_id ORDER BY score_producto DESC, product_id) rank_pasillo,
          ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY score_producto DESC, product_id) rank_departamento,
          ROW_NUMBER() OVER (ORDER BY score_producto DESC, product_id) rank_global
        FROM s
        QUALIFY rank_pasillo <= {PRODUCTOS_POR_PASILLO}
             OR rank_departamento <= {PRODUCTOS_POR_DEPARTAMENTO}
             OR rank_global <= {PRODUCTOS_RESPALDO}""",
        "productos_rank",
    )
    ejecutar_etapa(
        con,
        "afinidades de usuarios",
        f"""CREATE TEMP TABLE afinidades AS
        WITH base AS (
          SELECT user_id, aisle_id, department_id, SUM(freq_usuario_producto) compras
          FROM interacciones_trabajo GROUP BY 1,2,3
        ), pa AS (
          SELECT user_id, aisle_id, SUM(compras) compras,
                 SUM(compras) / SUM(SUM(compras)) OVER (PARTITION BY user_id) afinidad,
                 ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY SUM(compras) DESC, aisle_id) pos
          FROM base GROUP BY 1,2 QUALIFY pos <= {TOP_PASILLOS}
        ), de AS (
          SELECT user_id, department_id, SUM(compras) compras,
                 SUM(compras) / SUM(SUM(compras)) OVER (PARTITION BY user_id) afinidad,
                 ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY SUM(compras) DESC, department_id) pos
          FROM base GROUP BY 1,2 QUALIFY pos <= {TOP_DEPARTAMENTOS}
        )
        SELECT user_id, aisle_id, NULL::INTEGER department_id, afinidad, 'pasillo' fuente FROM pa
        UNION ALL
        SELECT user_id, NULL, department_id, afinidad, 'departamento' FROM de""",
        "afinidades",
    )
    ejecutar_etapa(
        con,
        "preselección por fuente",
        f"""CREATE TEMP TABLE candidatos_pre AS
        WITH pasillo AS (
          SELECT a.user_id, p.product_id, .60*a.afinidad + .40*p.score_producto score, 'pasillo' fuente
          FROM afinidades a JOIN productos_rank p ON a.aisle_id=p.aisle_id
          WHERE a.fuente='pasillo' AND p.rank_pasillo <= {PRODUCTOS_POR_PASILLO}
          QUALIFY ROW_NUMBER() OVER (PARTITION BY a.user_id ORDER BY score DESC, p.product_id) <= {PRESELECCION_PASILLO}
        ), depto AS (
          SELECT a.user_id, p.product_id, .50*a.afinidad + .50*p.score_producto score, 'departamento' fuente
          FROM afinidades a JOIN productos_rank p ON a.department_id=p.department_id
          WHERE a.fuente='departamento' AND p.rank_departamento <= {PRODUCTOS_POR_DEPARTAMENTO}
          QUALIFY ROW_NUMBER() OVER (PARTITION BY a.user_id ORDER BY score DESC, p.product_id) <= {PRESELECCION_DEPARTAMENTO}
        ), globales AS (
          SELECT u.user_id, p.product_id, .35*p.score_producto score, 'global' fuente
          FROM usuarios_trabajo u CROSS JOIN productos_rank p
          WHERE p.rank_global <= {PRODUCTOS_GLOBALES}
          QUALIFY ROW_NUMBER() OVER (PARTITION BY u.user_id ORDER BY score DESC, p.product_id) <= {PRESELECCION_GLOBAL}
        )
        SELECT * FROM pasillo UNION ALL SELECT * FROM depto UNION ALL SELECT * FROM globales""",
        "candidatos_pre",
    )
    ejecutar_etapa(
        con,
        "exclusión, deduplicación y Top-N",
        f"""CREATE TEMP TABLE candidatos_finales AS
        WITH nuevos AS (
          SELECT c.* FROM candidatos_pre c
          ANTI JOIN interacciones_trabajo i ON c.user_id=i.user_id AND c.product_id=i.product_id
        ), unicos AS (
          SELECT user_id, product_id, MAX(score) score_candidate,
                 STRING_AGG(DISTINCT fuente, '+' ORDER BY fuente) fuentes
          FROM nuevos GROUP BY 1,2
        ), ranked AS (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score_candidate DESC, product_id) candidate_rank
          FROM unicos
        ), cupos AS (
          SELECT u.user_id, {min_candidatos} + GREATEST(10-u.productos_distintos, 0) objetivo,
                 COUNT(r.product_id) FILTER (WHERE r.candidate_rank <= {min_candidatos} + GREATEST(10-u.productos_distintos, 0)) disponibles
          FROM usuarios_trabajo u LEFT JOIN ranked r USING (user_id) GROUP BY 1,2
        ), respaldo AS (
          SELECT c.user_id, p.product_id, .20*p.score_producto score_candidate, 'respaldo' fuentes
          FROM cupos c CROSS JOIN productos_rank p
          ANTI JOIN interacciones_trabajo i ON c.user_id=i.user_id AND p.product_id=i.product_id
          ANTI JOIN unicos x ON c.user_id=x.user_id AND p.product_id=x.product_id
          WHERE c.disponibles < c.objetivo AND p.rank_global <= {PRODUCTOS_RESPALDO}
          QUALIFY ROW_NUMBER() OVER (PARTITION BY c.user_id ORDER BY p.score_producto DESC, p.product_id) <= c.objetivo-c.disponibles
        ), completos AS (
          SELECT user_id,product_id,score_candidate,fuentes FROM unicos
          UNION ALL SELECT user_id,product_id,score_candidate,fuentes FROM respaldo
        ), rank_final AS (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score_candidate DESC, product_id) candidate_rank
          FROM completos
        )
        SELECT r.user_id, r.product_id, CAST(r.candidate_rank AS INTEGER) candidate_rank,
               r.score_candidate, r.fuentes
        FROM rank_final r JOIN usuarios_trabajo u USING (user_id)
        WHERE r.candidate_rank <= {min_candidatos} + GREATEST(10-u.productos_distintos, 0)""",
        "candidatos_finales",
    )


def validar(con: duckdb.DuckDBPyConnection, min_candidatos: int) -> dict[str, int]:
    fila = con.execute("""SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT (user_id,product_id)),
        COUNT(*) FILTER (WHERE user_id IS NULL OR product_id IS NULL OR score_candidate IS NULL OR candidate_rank<1)
        FROM candidatos_finales""").fetchone()
    comprados = con.execute("""SELECT COUNT(*) FROM candidatos_finales c JOIN interacciones_trabajo i
        ON c.user_id=i.user_id AND c.product_id=i.product_id""").fetchone()[0]
    incompletos = con.execute(f"""WITH n AS (SELECT user_id,COUNT(*) cantidad FROM candidatos_finales GROUP BY 1)
        SELECT COUNT(*) FROM usuarios_trabajo u LEFT JOIN n USING(user_id)
        WHERE COALESCE(cantidad,0) < {min_candidatos} + GREATEST(10-u.productos_distintos,0)""").fetchone()[0]
    if fila[0] != fila[2] or fila[3] or comprados or incompletos:
        raise ValueError(f"Validación fallida: duplicados={fila[0]-fila[2]}, nulos/rank={fila[3]}, comprados={comprados}, incompletos={incompletos}")
    return {"filas": fila[0], "usuarios": fila[1], "productos": con.execute("SELECT COUNT(DISTINCT product_id) FROM candidatos_finales").fetchone()[0]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera candidatos nuevos de forma acotada.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-candidatos", type=int, default=MIN_CANDIDATOS)
    p.add_argument("--max-usuarios", type=int, default=None, help="Piloto determinista con los primeros N usuarios.")
    p.add_argument("--memory-limit", default="4GB", help="Límite de memoria de DuckDB, por ejemplo 4GB.")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-dir", type=Path, default=INTERIM_DIR / "duckdb_candidates_temp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_candidatos < 1 or args.threads < 1 or (args.max_usuarios is not None and args.max_usuarios < 1):
        raise ValueError("min-candidatos, threads y max-usuarios deben ser positivos")
    paths = validar_entradas()
    salida = args.output.resolve()
    parcial = salida.with_name(salida.stem + ".partial" + salida.suffix)
    parcial.unlink(missing_ok=True)
    inicio = time.perf_counter()
    print(f"Iniciando generación | memoria={args.memory_limit} | hilos={args.threads}", flush=True)
    try:
        with duckdb.connect() as con:
            configurar(con, args.memory_limit, args.threads, args.temp_dir)
            n_usuarios = crear_fuentes(con, paths, args.max_usuarios)
            construir(con, args.min_candidatos)
            metricas = validar(con, args.min_candidatos)
            ejecutar_etapa(con, "escritura Parquet", f"COPY candidatos_finales TO '{sql_path(parcial)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        parcial.replace(salida)
    except BaseException:
        parcial.unlink(missing_ok=True)
        raise
    total = time.perf_counter() - inicio
    temp_bytes = tamanio_directorio(args.temp_dir)
    print(f"[OK] {metricas['filas']:,} candidatos | {metricas['usuarios']:,}/{n_usuarios:,} usuarios | {metricas['productos']:,} productos")
    print(f"Costo observado: {total:.1f} s | temporal restante/pico no disponible: {formato_bytes(temp_bytes)} | salida: {formato_bytes(salida.stat().st_size)}")
    print(f"Archivo generado: {salida}")
    if args.max_usuarios:
        print("PILOTO: no usar esta salida como tabla final.")
    if args.temp_dir.exists():
        shutil.rmtree(args.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
