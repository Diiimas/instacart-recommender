"""Evaluacion reproducible del baseline y los candidatos nuevos de Instacart.

No mezcla ``score_candidate`` con frecuencia o recencia: son escalas distintas.
El sistema combinado usa una regla interpretable: primero recomienda el Top-K
historico y solo completa lugares faltantes con candidatos nuevos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("reports/candidate_metrics.csv"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def validate_inputs(processed_dir: Path) -> dict[str, Path]:
    names = ["catalogo", "usuarios", "interacciones", "targets_train", "candidatos_nuevos"]
    paths = {name: processed_dir / f"{name}.parquet" for name in names}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos requeridos:\n- " + "\n- ".join(missing))
    return paths


def build_views(con: duckdb.DuckDBPyConnection, paths: dict[str, Path], k: int) -> None:
    for name, path in paths.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{sql_path(path)}')")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE usuarios_eval AS
        SELECT DISTINCT user_id FROM targets_train;

        CREATE OR REPLACE TEMP TABLE targets AS
        SELECT DISTINCT user_id, product_id FROM targets_train;

        CREATE OR REPLACE TEMP TABLE targets_nuevos AS
        SELECT t.user_id, t.product_id
        FROM targets t
        ANTI JOIN interacciones i USING (user_id, product_id);

        CREATE OR REPLACE TEMP TABLE baseline AS
        SELECT user_id, product_id, baseline_rank AS rank
        FROM (
            SELECT i.user_id, i.product_id,
                   row_number() OVER (
                       PARTITION BY i.user_id
                       ORDER BY i.freq_usuario_producto DESC,
                                i.recencia_usuario_producto ASC,
                                i.ultima_orden_producto DESC,
                                i.product_id
                   ) AS baseline_rank
            FROM interacciones i
            SEMI JOIN usuarios_eval u USING (user_id)
        ) ranked
        WHERE baseline_rank <= {k};

        CREATE OR REPLACE TEMP TABLE nuevos_topk AS
        SELECT c.user_id, c.product_id, c.candidate_rank AS rank
        FROM candidatos_nuevos c
        SEMI JOIN usuarios_eval u USING (user_id)
        WHERE c.candidate_rank <= {k};

        CREATE OR REPLACE TEMP TABLE combinado AS
        WITH cupos AS (
            SELECT u.user_id, {k} - count(b.product_id) AS lugares
            FROM usuarios_eval u
            LEFT JOIN baseline b USING (user_id)
            GROUP BY u.user_id
        ), relleno AS (
            SELECT c.user_id, c.product_id,
                   {k} - cp.lugares + row_number() OVER (
                       PARTITION BY c.user_id ORDER BY c.candidate_rank, c.product_id
                   ) AS rank
            FROM candidatos_nuevos c
            JOIN cupos cp USING (user_id)
            WHERE cp.lugares > 0
            QUALIFY row_number() OVER (
                PARTITION BY c.user_id ORDER BY c.candidate_rank, c.product_id
            ) <= cp.lugares
        )
        SELECT * FROM baseline
        UNION ALL
        SELECT * FROM relleno;
        """
    )


def metric_query(con: duckdb.DuckDBPyConnection, k: int) -> list[tuple]:
    return con.execute(
        f"""
        WITH sistemas AS (
            SELECT 'baseline_historico' AS sistema, user_id, product_id, rank FROM baseline
            UNION ALL
            SELECT 'candidatos_nuevos_top{k}', user_id, product_id, rank FROM nuevos_topk
            UNION ALL
            SELECT 'baseline_mas_relleno', user_id, product_id, rank FROM combinado
        ), por_usuario AS (
            SELECT
                s.sistema,
                u.user_id,
                count(t.product_id) AS aciertos_usuario,
                (SELECT count(*) FROM targets tx WHERE tx.user_id = u.user_id) AS objetivos_usuario
            FROM (SELECT DISTINCT sistema FROM sistemas) s
            CROSS JOIN usuarios_eval u
            LEFT JOIN sistemas rec ON rec.sistema = s.sistema AND rec.user_id = u.user_id
            LEFT JOIN targets t ON t.user_id = rec.user_id AND t.product_id = rec.product_id
            GROUP BY s.sistema, u.user_id
        ), macro AS (
            SELECT sistema,
                   avg(aciertos_usuario::DOUBLE / NULLIF(objetivos_usuario, 0)) AS recall_macro
            FROM por_usuario GROUP BY sistema
        ), resumen AS (
            SELECT
                s.sistema,
                count(*) AS recomendaciones,
                count(DISTINCT s.user_id) AS usuarios_con_recomendaciones,
                count(DISTINCT s.product_id) AS productos_recomendados,
                count(t.product_id) AS aciertos,
                count(DISTINCT CASE WHEN t.product_id IS NOT NULL THEN s.user_id END) AS usuarios_con_acierto
            FROM sistemas s
            LEFT JOIN targets t USING (user_id, product_id)
            GROUP BY s.sistema
        ), completos AS (
            SELECT sistema, count(*) FILTER (WHERE n >= {k}) AS usuarios_topk_completo
            FROM (
                SELECT sistema, user_id, count(*) AS n
                FROM sistemas GROUP BY sistema, user_id
            ) x GROUP BY sistema
        ), constantes AS (
            SELECT
                (SELECT count(*) FROM usuarios_eval) AS usuarios_evaluados,
                (SELECT count(*) FROM targets) AS objetivos,
                (SELECT count(*) FROM catalogo) AS productos_catalogo
        )
        SELECT
            r.sistema,
            r.recomendaciones,
            r.usuarios_con_recomendaciones,
            c.usuarios_topk_completo,
            r.productos_recomendados,
            r.aciertos,
            round(r.aciertos::DOUBLE / NULLIF(kon.objetivos, 0), 6) AS recall_micro,
            round(m.recall_macro, 6) AS recall_macro,
            round(r.usuarios_con_acierto::DOUBLE / NULLIF(kon.usuarios_evaluados, 0), 6) AS hit_rate,
            round(r.productos_recomendados::DOUBLE / NULLIF(kon.productos_catalogo, 0), 6) AS cobertura_catalogo,
            round(c.usuarios_topk_completo::DOUBLE / NULLIF(kon.usuarios_evaluados, 0), 6) AS usuarios_topk_completo_pct
        FROM resumen r
        JOIN completos c USING (sistema)
        JOIN macro m USING (sistema)
        CROSS JOIN constantes kon
        ORDER BY CASE r.sistema
            WHEN 'baseline_historico' THEN 1
            WHEN 'candidatos_nuevos_top{k}' THEN 2
            ELSE 3 END;
        """
    ).fetchall()


def discovery_query(con: duckdb.DuckDBPyConnection, k: int) -> tuple:
    return con.execute(
        f"""
        WITH resumen AS (
            SELECT
                count(*) AS objetivos_nuevos,
                count(c.product_id) AS encontrados_pool,
                count(c.product_id) FILTER (WHERE c.candidate_rank <= {k}) AS encontrados_topk,
                count(DISTINCT tn.user_id) AS usuarios_con_objetivo_nuevo,
                count(DISTINCT CASE WHEN c.product_id IS NOT NULL THEN tn.user_id END) AS usuarios_hit_pool,
                count(DISTINCT CASE WHEN c.candidate_rank <= {k} THEN tn.user_id END) AS usuarios_hit_topk
            FROM targets_nuevos tn
            LEFT JOIN candidatos_nuevos c USING (user_id, product_id)
        )
        SELECT *,
            round(encontrados_pool::DOUBLE / NULLIF(objetivos_nuevos, 0), 6) AS recall_pool_nuevos,
            round(encontrados_topk::DOUBLE / NULLIF(objetivos_nuevos, 0), 6) AS recall_topk_nuevos,
            round(usuarios_hit_pool::DOUBLE / NULLIF(usuarios_con_objetivo_nuevo, 0), 6) AS hit_rate_pool_nuevos,
            round(usuarios_hit_topk::DOUBLE / NULLIF(usuarios_con_objetivo_nuevo, 0), 6) AS hit_rate_topk_nuevos
        FROM resumen;
        """
    ).fetchone()


def print_table(headers: list[str], rows: list[tuple]) -> None:
    widths = [len(h) for h in headers]
    text_rows = [[str(v) for v in row] for row in rows]
    for row in text_rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in text_rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k debe ser mayor que cero")
    paths = validate_inputs(args.processed_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with duckdb.connect() as con:
        con.execute(f"SET memory_limit = '{args.memory_limit}'")
        con.execute(f"SET threads = {args.threads}")
        print(f"Iniciando evaluacion | K={args.k} | memoria={args.memory_limit} | hilos={args.threads}")
        build_views(con, paths, args.k)
        rows = metric_query(con, args.k)
        headers = [d[0] for d in con.description]

        discovery = discovery_query(con, args.k)
        discovery_headers = [d[0] for d in con.description]

    print("\nMetricas de recomendacion")
    print_table(headers, rows)
    print("\nDescubrimiento de productos realmente nuevos")
    print_table(discovery_headers, [discovery])

    # CSV compacto y portable, sin agregar pandas/pyarrow al proyecto.
    import csv
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    discovery_output = args.output.with_name(f"{args.output.stem}_nuevos{args.output.suffix}")
    with discovery_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(discovery_headers)
        writer.writerow(discovery)

    print(f"\n[OK] Evaluacion terminada en {time.perf_counter() - started:.1f} s")
    print(f"Resultados principales: {args.output.resolve()}")
    print(f"Resultados de productos nuevos: {discovery_output.resolve()}")
    print("Nota: 'baseline_mas_relleno' conserva el ranking historico y usa candidatos nuevos solo para completar Top-K.")


if __name__ == "__main__":
    main()
