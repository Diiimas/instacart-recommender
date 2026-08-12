"""
Construcción reproducible de la base analítica de Instacart.

El pipeline:
- Lee los datos originales desde data/raw mediante DuckDB.
- Conserva intactos los archivos originales.
- Materializa las tablas analíticas en data/processed.
- Mantiene separado el historial prior del target train.
"""

from pathlib import Path

from load_data import create_connection


# Ruta raíz del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpeta donde se guardarán las salidas del pipeline
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_output_directory() -> None:
    """
    Crea la carpeta de salida si todavía no existe.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_catalog(connection) -> Path:
    """
    Construye el catálogo legible de productos.

    Integra:
    - products
    - aisles
    - departments

    Returns
    -------
    Path
        Ruta del archivo Parquet generado.
    """
    output_path = PROCESSED_DIR / "catalogo.parquet"

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE catalogo_analitico AS
        SELECT
            CAST(p.product_id AS INTEGER) AS product_id,
            p.product_name,
            CAST(p.aisle_id AS INTEGER) AS aisle_id,
            a.aisle,
            CAST(p.department_id AS INTEGER) AS department_id,
            d.department
        FROM products AS p
        INNER JOIN aisles AS a
            ON p.aisle_id = a.aisle_id
        INNER JOIN departments AS d
            ON p.department_id = d.department_id
        ORDER BY p.product_id;
        """
    )

    catalog_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM catalogo_analitico;
        """
    ).fetchone()[0]

    product_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM products;
        """
    ).fetchone()[0]

    if catalog_rows != product_rows:
        raise ValueError(
            "El catálogo generado no conserva todos los productos: "
            f"{catalog_rows:,} de {product_rows:,}."
        )

    output_path_sql = output_path.as_posix().replace("'", "''")

    connection.execute(
        f"""
        COPY catalogo_analitico
        TO '{output_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    print(
        f"[OK] catalogo.parquet: "
        f"{catalog_rows:,} productos"
    )

    return output_path

def build_train_targets(connection) -> Path:
    """
    Construye el target de evaluación a partir de las órdenes train.

    Cada fila representa un producto presente en la última orden
    revelada de un usuario evaluable. Esta tabla se mantiene separada
    del historial prior para evitar fuga de información.

    Returns
    -------
    Path
        Ruta del archivo Parquet generado.
    """
    output_path = PROCESSED_DIR / "targets_train.parquet"

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE targets_train_analitico AS
        SELECT
            CAST(o.user_id AS INTEGER) AS user_id,
            CAST(op.order_id AS INTEGER) AS order_id,
            CAST(op.product_id AS INTEGER) AS product_id,
            CAST(op.add_to_cart_order AS INTEGER) AS add_to_cart_order,
            CAST(op.reordered AS TINYINT) AS reordered
        FROM order_products_train AS op
        INNER JOIN orders AS o
            ON op.order_id = o.order_id
        WHERE o.eval_set = 'train'
        ORDER BY
            o.user_id,
            op.add_to_cart_order;
        """
    )

    target_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM targets_train_analitico;
        """
    ).fetchone()[0]

    source_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM order_products_train;
        """
    ).fetchone()[0]

    target_users = connection.execute(
        """
        SELECT COUNT(DISTINCT user_id)
        FROM targets_train_analitico;
        """
    ).fetchone()[0]

    train_users = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE eval_set = 'train';
        """
    ).fetchone()[0]

    if target_rows != source_rows:
        raise ValueError(
            "El target generado no conserva todas las interacciones train: "
            f"{target_rows:,} de {source_rows:,}."
        )

    if target_users != train_users:
        raise ValueError(
            "La cantidad de usuarios del target no coincide con las "
            "órdenes train: "
            f"{target_users:,} de {train_users:,}."
        )

    output_path_sql = output_path.as_posix().replace("'", "''")

    connection.execute(
        f"""
        COPY targets_train_analitico
        TO '{output_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    print(
        f"[OK] targets_train.parquet: "
        f"{target_rows:,} productos objetivo de "
        f"{target_users:,} usuarios"
    )

    return output_path

def build_product_profiles(connection) -> Path:
    """
    Construye el perfil analítico de cada producto usando solo el historial prior.

    Las órdenes train no participan en estas features para evitar fuga
    de información.

    Returns
    -------
    Path
        Ruta del archivo Parquet generado.
    """
    output_path = PROCESSED_DIR / "productos.parquet"

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE productos_analitico AS
        SELECT
            CAST(c.product_id AS INTEGER) AS product_id,
            c.product_name,
            CAST(c.aisle_id AS INTEGER) AS aisle_id,
            c.aisle,
            CAST(c.department_id AS INTEGER) AS department_id,
            c.department,

            CAST(COUNT(op.order_id) AS BIGINT) AS cantidad_compras,

            CAST(
                COUNT(DISTINCT o.user_id)
                AS INTEGER
            ) AS cantidad_usuarios,

            CAST(
                COALESCE(AVG(op.reordered), 0)
                AS DOUBLE
            ) AS reorder_rate_producto,

            CAST(
                AVG(op.add_to_cart_order)
                AS DOUBLE
            ) AS add_to_cart_order_promedio

        FROM catalogo_analitico AS c

        LEFT JOIN order_products_prior AS op
            ON c.product_id = op.product_id

        LEFT JOIN orders AS o
            ON op.order_id = o.order_id
           AND o.eval_set = 'prior'

        GROUP BY
            c.product_id,
            c.product_name,
            c.aisle_id,
            c.aisle,
            c.department_id,
            c.department

        ORDER BY
            c.product_id;
        """
    )

    profile_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM productos_analitico;
        """
    ).fetchone()[0]

    catalog_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM catalogo_analitico;
        """
    ).fetchone()[0]

    total_purchases = connection.execute(
        """
        SELECT SUM(cantidad_compras)
        FROM productos_analitico;
        """
    ).fetchone()[0]

    prior_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM order_products_prior;
        """
    ).fetchone()[0]

    if profile_rows != catalog_rows:
        raise ValueError(
            "El perfil de productos no conserva todo el catálogo: "
            f"{profile_rows:,} de {catalog_rows:,} productos."
        )

    if total_purchases != prior_rows:
        raise ValueError(
            "El perfil de productos no conserva todas las compras prior: "
            f"{total_purchases:,} de {prior_rows:,}."
        )

    output_path_sql = output_path.as_posix().replace("'", "''")

    connection.execute(
        f"""
        COPY productos_analitico
        TO '{output_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    print(
        f"[OK] productos.parquet: "
        f"{profile_rows:,} productos y "
        f"{total_purchases:,} compras históricas"
    )

    return output_path

def build_user_profiles(connection) -> Path:
    """
    Construye el perfil analítico de cada usuario usando solo su historial prior.

    Incluye volumen de actividad, diversidad de productos, recompra,
    hábitos temporales, reglas especiales del dataset y segmento.

    Returns
    -------
    Path
        Ruta del archivo Parquet generado.
    """
    output_path = PROCESSED_DIR / "usuarios.parquet"

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE usuarios_analitico AS

        WITH ordenes_usuario AS (
            SELECT
                user_id,
                COUNT(*) AS cantidad_ordenes_historicas,
                MAX(order_number) AS ultima_order_number,
                MODE(order_dow) AS dow_habitual,
                MODE(order_hour_of_day) AS hora_habitual,
                MEDIAN(days_since_prior_order) FILTER (
                    WHERE days_since_prior_order IS NOT NULL
                ) AS mediana_dias_entre_ordenes_historicas,
                SUM(
                    CASE
                        WHEN days_since_prior_order IS NULL THEN 1
                        ELSE 0
                    END
                ) AS cantidad_primeras_ordenes,
                SUM(
                    CASE
                        WHEN days_since_prior_order = 30 THEN 1
                        ELSE 0
                    END
                ) AS cantidad_intervalos_censurados
            FROM orders
            WHERE eval_set = 'prior'
            GROUP BY user_id
        ),

        compras_usuario AS (
            SELECT
                o.user_id,
                COUNT(*) AS cantidad_compras,
                COUNT(DISTINCT op.product_id) AS productos_distintos,
                AVG(op.reordered) AS reorder_rate_usuario,
                AVG(op.add_to_cart_order) AS posicion_media_carrito
            FROM orders AS o
            INNER JOIN order_products_prior AS op
                ON o.order_id = op.order_id
            WHERE o.eval_set = 'prior'
            GROUP BY o.user_id
        )

        SELECT
            CAST(ou.user_id AS INTEGER) AS user_id,

            CAST(
                ou.cantidad_ordenes_historicas
                AS INTEGER
            ) AS cantidad_ordenes_historicas,

            CAST(
                ou.ultima_order_number
                AS INTEGER
            ) AS ultima_order_number,

            CAST(
                cu.cantidad_compras
                AS BIGINT
            ) AS cantidad_compras,

            CAST(
                cu.productos_distintos
                AS INTEGER
            ) AS productos_distintos,

            CAST(
                cu.reorder_rate_usuario
                AS DOUBLE
            ) AS reorder_rate_usuario,

            CAST(
                cu.posicion_media_carrito
                AS DOUBLE
            ) AS posicion_media_carrito,

            CAST(ou.dow_habitual AS TINYINT) AS dow_habitual,
            CAST(ou.hora_habitual AS TINYINT) AS hora_habitual,

            CAST(
                ou.mediana_dias_entre_ordenes_historicas
                AS DOUBLE
            ) AS mediana_dias_entre_ordenes_historicas,

            CAST(
                CASE
                    WHEN ou.cantidad_primeras_ordenes > 0 THEN 1
                    ELSE 0
                END
                AS TINYINT
            ) AS tiene_primera_orden,

            CAST(
                CASE
                    WHEN ou.cantidad_intervalos_censurados > 0 THEN 1
                    ELSE 0
                END
                AS TINYINT
            ) AS tiene_intervalo_censurado_30,

            CAST(
                ou.cantidad_intervalos_censurados
                AS INTEGER
            ) AS cantidad_intervalos_censurados_30,

            CASE
                WHEN ou.cantidad_ordenes_historicas <= 5
                    THEN 'nuevo'
                WHEN ou.cantidad_ordenes_historicas <= 15
                    THEN 'medio'
                ELSE 'heavy'
            END AS segmento_usuario

        FROM ordenes_usuario AS ou

        INNER JOIN compras_usuario AS cu
            ON ou.user_id = cu.user_id

        ORDER BY ou.user_id;
        """
    )

    profile_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM usuarios_analitico;
        """
    ).fetchone()[0]

    historical_users = connection.execute(
        """
        SELECT COUNT(DISTINCT user_id)
        FROM orders
        WHERE eval_set = 'prior';
        """
    ).fetchone()[0]

    total_profile_purchases = connection.execute(
        """
        SELECT SUM(cantidad_compras)
        FROM usuarios_analitico;
        """
    ).fetchone()[0]

    prior_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM order_products_prior;
        """
    ).fetchone()[0]

    users_with_first_order = connection.execute(
        """
        SELECT COUNT(*)
        FROM usuarios_analitico
        WHERE tiene_primera_orden = 1;
        """
    ).fetchone()[0]

    invalid_median_intervals = connection.execute(
        """
        SELECT COUNT(*)
        FROM usuarios_analitico
        WHERE mediana_dias_entre_ordenes_historicas < 0;
        """
    ).fetchone()[0]

    unexpected_null_medians = connection.execute(
        """
        SELECT COUNT(*)
        FROM usuarios_analitico
        WHERE
            (cantidad_ordenes_historicas = 1
             AND mediana_dias_entre_ordenes_historicas IS NOT NULL)
            OR
            (cantidad_ordenes_historicas > 1
             AND mediana_dias_entre_ordenes_historicas IS NULL);
        """
    ).fetchone()[0]

    if profile_rows != historical_users:
        raise ValueError(
            "El perfil no conserva todos los usuarios históricos: "
            f"{profile_rows:,} de {historical_users:,}."
        )

    if total_profile_purchases != prior_rows:
        raise ValueError(
            "El perfil no conserva todas las compras prior: "
            f"{total_profile_purchases:,} de {prior_rows:,}."
        )

    if users_with_first_order != historical_users:
        raise ValueError(
            "La regla de primera orden no se cumple para todos los usuarios: "
            f"{users_with_first_order:,} de {historical_users:,}."
        )

    if invalid_median_intervals != 0:
        raise ValueError(
            "Se encontraron medianas negativas entre órdenes históricas: "
            f"{invalid_median_intervals:,}."
        )

    if unexpected_null_medians != 0:
        raise ValueError(
            "La nulabilidad de la mediana entre órdenes no coincide con "
            "la cantidad de órdenes históricas: "
            f"{unexpected_null_medians:,} usuarios inconsistentes."
        )

    output_path_sql = output_path.as_posix().replace("'", "''")

    connection.execute(
        f"""
        COPY usuarios_analitico
        TO '{output_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    print(
        f"[OK] usuarios.parquet: "
        f"{profile_rows:,} usuarios y "
        f"{total_profile_purchases:,} compras históricas"
    )

    segment_counts = connection.execute(
        """
        SELECT
            segmento_usuario,
            COUNT(*) AS cantidad
        FROM usuarios_analitico
        GROUP BY segmento_usuario
        ORDER BY
            CASE segmento_usuario
                WHEN 'nuevo' THEN 1
                WHEN 'medio' THEN 2
                ELSE 3
            END;
        """
    ).fetchall()

    for segment, count in segment_counts:
        print(f"     - {segment}: {count:,} usuarios")

    return output_path

def build_user_product_interactions(connection) -> Path:
    """
    Construye la base analítica a nivel usuario-producto.

    Agrupa exclusivamente las compras del historial prior. Cada fila
    representa un producto que un usuario compró al menos una vez.

    La recencia se expresa como la cantidad de órdenes transcurridas
    entre la última orden histórica del usuario y la última orden
    en la que compró el producto.

    Returns
    -------
    Path
        Ruta del archivo Parquet generado.
    """
    output_path = PROCESSED_DIR / "interacciones.parquet"

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE interacciones_analitico AS

        WITH ordenes_con_tiempo AS (
            SELECT
                user_id,
                order_id,
                order_number,
                order_dow,
                order_hour_of_day,
                SUM(COALESCE(days_since_prior_order, 0)) OVER (
                    PARTITION BY user_id
                    ORDER BY order_number
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS dias_acumulados_registrados
            FROM orders
            WHERE eval_set = 'prior'
        ),

        historial AS (
            SELECT
                o.user_id,
                op.product_id,
                o.order_number,
                o.order_dow,
                o.order_hour_of_day,
                o.dias_acumulados_registrados,
                op.add_to_cart_order
            FROM ordenes_con_tiempo AS o
            INNER JOIN order_products_prior AS op
                ON o.order_id = op.order_id
        ),

        ultima_orden_usuario AS (
            SELECT
                user_id,
                MAX(order_number) AS ultima_order_number_usuario,
                ARG_MAX(
                    dias_acumulados_registrados,
                    order_number
                ) AS dias_acumulados_ultima_orden
            FROM historial
            GROUP BY user_id
        ),

        agregados AS (
            SELECT
                h.user_id,
                h.product_id,
                COUNT(*) AS freq_usuario_producto,
                MAX(h.order_number) AS ultima_orden_producto,
                ARG_MAX(
                    h.dias_acumulados_registrados,
                    h.order_number
                ) AS dias_acumulados_ultima_compra,
                AVG(h.add_to_cart_order) AS add_to_cart_order_promedio,
                ARG_MAX(h.order_dow, h.order_number) AS ultima_order_dow,
                ARG_MAX(
                    h.order_hour_of_day,
                    h.order_number
                ) AS ultima_order_hour
            FROM historial AS h
            GROUP BY
                h.user_id,
                h.product_id
        )

        SELECT
            CAST(a.user_id AS INTEGER) AS user_id,
            CAST(a.product_id AS INTEGER) AS product_id,

            CAST(
                a.freq_usuario_producto
                AS INTEGER
            ) AS freq_usuario_producto,

            CAST(
                u.ultima_order_number_usuario
                - a.ultima_orden_producto
                AS INTEGER
            ) AS recencia_usuario_producto,

            CAST(
                u.dias_acumulados_ultima_orden
                - a.dias_acumulados_ultima_compra
                AS INTEGER
            ) AS dias_registrados_desde_ultima_compra,

            CAST(
                a.ultima_orden_producto
                AS INTEGER
            ) AS ultima_orden_producto,

            CAST(
                a.add_to_cart_order_promedio
                AS DOUBLE
            ) AS add_to_cart_order_promedio,

            CAST(a.ultima_order_dow AS TINYINT) AS ultima_order_dow,
            CAST(a.ultima_order_hour AS TINYINT) AS ultima_order_hour,

            CAST(c.aisle_id AS INTEGER) AS aisle_id,
            CAST(c.department_id AS INTEGER) AS department_id

        FROM agregados AS a

        INNER JOIN ultima_orden_usuario AS u
            ON a.user_id = u.user_id

        INNER JOIN catalogo_analitico AS c
            ON a.product_id = c.product_id

        ORDER BY
            a.user_id,
            a.product_id;
        """
    )

    interaction_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM interacciones_analitico;
        """
    ).fetchone()[0]

    total_frequency = connection.execute(
        """
        SELECT SUM(freq_usuario_producto)
        FROM interacciones_analitico;
        """
    ).fetchone()[0]

    prior_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM order_products_prior;
        """
    ).fetchone()[0]

    duplicate_keys = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                user_id,
                product_id
            FROM interacciones_analitico
            GROUP BY
                user_id,
                product_id
            HAVING COUNT(*) > 1
        );
        """
    ).fetchone()[0]

    invalid_recency = connection.execute(
        """
        SELECT COUNT(*)
        FROM interacciones_analitico
        WHERE recencia_usuario_producto < 0;
        """
    ).fetchone()[0]

    invalid_days_recency = connection.execute(
        """
        SELECT COUNT(*)
        FROM interacciones_analitico
        WHERE dias_registrados_desde_ultima_compra < 0;
        """
    ).fetchone()[0]

    invalid_latest_purchase_days = connection.execute(
        """
        SELECT COUNT(*)
        FROM interacciones_analitico
        WHERE recencia_usuario_producto = 0
          AND dias_registrados_desde_ultima_compra != 0;
        """
    ).fetchone()[0]

    if total_frequency != prior_rows:
        raise ValueError(
            "Las frecuencias usuario-producto no conservan todas las "
            "compras prior: "
            f"{total_frequency:,} de {prior_rows:,}."
        )

    if duplicate_keys != 0:
        raise ValueError(
            "La base contiene claves usuario-producto duplicadas: "
            f"{duplicate_keys:,}."
        )

    if invalid_recency != 0:
        raise ValueError(
            "Se encontraron valores negativos de recencia: "
            f"{invalid_recency:,}."
        )

    if invalid_days_recency != 0:
        raise ValueError(
            "Se encontraron valores negativos de días registrados desde "
            f"la última compra: {invalid_days_recency:,}."
        )

    if invalid_latest_purchase_days != 0:
        raise ValueError(
            "Hay productos de la última orden histórica cuya recencia "
            "temporal no es cero: "
            f"{invalid_latest_purchase_days:,}."
        )

    output_path_sql = output_path.as_posix().replace("'", "''")

    connection.execute(
        f"""
        COPY interacciones_analitico
        TO '{output_path_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    print(
        f"[OK] interacciones.parquet: "
        f"{interaction_rows:,} pares usuario-producto que representan "
        f"{total_frequency:,} compras históricas"
    )

    return output_path

def validate_source_quality(connection) -> None:
    """
    Ejecuta controles de calidad sobre las fuentes originales.

    El pipeline se detiene si encuentra:
    - Productos huérfanos en el historial.
    - Pares order_id-product_id duplicados.
    - Una cantidad incorrecta de primeras órdenes.
    """
    orphan_products = connection.execute(
        """
        SELECT COUNT(*)
        FROM order_products_prior AS op
        LEFT JOIN products AS p
            ON op.product_id = p.product_id
        WHERE p.product_id IS NULL;
        """
    ).fetchone()[0]

    duplicate_order_products = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                order_id,
                product_id
            FROM order_products_prior
            GROUP BY
                order_id,
                product_id
            HAVING COUNT(*) > 1
        );
        """
    ).fetchone()[0]

    null_intervals = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE days_since_prior_order IS NULL;
        """
    ).fetchone()[0]

    total_users = connection.execute(
        """
        SELECT COUNT(DISTINCT user_id)
        FROM orders;
        """
    ).fetchone()[0]

    invalid_order_numbers = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE order_number < 1
           OR order_number > 100;
        """
    ).fetchone()[0]

    if orphan_products != 0:
        raise ValueError(
            "Se encontraron productos huérfanos en el historial: "
            f"{orphan_products:,}."
        )

    if duplicate_order_products != 0:
        raise ValueError(
            "Se encontraron pares order_id-product_id duplicados: "
            f"{duplicate_order_products:,}."
        )

    if null_intervals != total_users:
        raise ValueError(
            "La cantidad de intervalos nulos no coincide con la cantidad "
            "de usuarios: "
            f"{null_intervals:,} nulos para {total_users:,} usuarios."
        )

    if invalid_order_numbers != 0:
        raise ValueError(
            "Se encontraron números de orden fuera del rango 1-100: "
            f"{invalid_order_numbers:,}."
        )

    print("[OK] Productos huérfanos en el historial: 0")
    print("[OK] Pares order_id-product_id duplicados: 0")
    print(
        f"[OK] Primeras órdenes: "
        f"{null_intervals:,} nulos para {total_users:,} usuarios"
    )
    print("[OK] order_number dentro del rango documentado 1-100")

def main() -> None:
    """
    Ejecuta la construcción completa de la base analítica.
    """
    print("Iniciando construcción de la base analítica...")

    prepare_output_directory()
    connection = create_connection()

    try:
        print("\n0. Controles de calidad de las fuentes")
        validate_source_quality(connection)

        print("\n1. Construcción del catálogo")
        catalog_path = build_catalog(connection)

        print("\n2. Construcción del target de evaluación")
        targets_path = build_train_targets(connection)

        print("\n3. Construcción del perfil de productos")
        products_path = build_product_profiles(connection)

        print("\n4. Construcción del perfil de usuarios")
        users_path = build_user_profiles(connection)

        print("\n5. Construcción de interacciones usuario-producto")
        interactions_path = build_user_product_interactions(connection)

    finally:
        connection.close()

    print("\nPipeline finalizado correctamente.")
    print("Archivos generados:")
    print(f"- {catalog_path}")
    print(f"- {targets_path}")
    print(f"- {products_path}")
    print(f"- {users_path}")
    print(f"- {interactions_path}")
    print("Los archivos originales no fueron modificados.")


if __name__ == "__main__":
    main()
