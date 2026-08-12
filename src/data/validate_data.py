"""
Validaciones básicas de calidad para los datos originales de Instacart.

Esta primera etapa comprueba:
- Presencia de las columnas requeridas.
- Valores fuera de los rangos esperados.
- Valores nulos en columnas críticas.

No modifica los archivos originales.
"""

from load_data import create_connection


# Columnas mínimas esperadas en cada tabla
REQUIRED_COLUMNS = {
    "aisles": {
        "aisle_id",
        "aisle",
    },
    "departments": {
        "department_id",
        "department",
    },
    "order_products_prior": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    },
    "order_products_train": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    },
    "orders": {
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    },
    "products": {
        "product_id",
        "product_name",
        "aisle_id",
        "department_id",
    },
}


def validate_required_columns(connection) -> list[str]:
    """
    Comprueba que cada tabla contenga las columnas requeridas.

    Returns
    -------
    list[str]
        Lista de errores encontrados.
    """
    errors = []

    print("\n1. Validación de columnas")

    for table_name, required_columns in REQUIRED_COLUMNS.items():
        table_info = connection.execute(
            f"DESCRIBE {table_name}"
        ).fetchall()

        existing_columns = {row[0] for row in table_info}
        missing_columns = required_columns - existing_columns

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            errors.append(
                f"{table_name}: faltan las columnas {missing_text}"
            )
            print(f"[ERROR] {table_name}: faltan {missing_text}")
        else:
            print(f"[OK] {table_name}")

    return errors


def validate_orders(connection) -> list[str]:
    """
    Valida valores críticos y rangos de la tabla orders.
    """
    errors = []

    print("\n2. Validación de orders")

    result = connection.execute(
        """
        SELECT
            COUNT(*) AS total_rows,

            SUM(
                CASE
                    WHEN order_id IS NULL OR user_id IS NULL
                    THEN 1 ELSE 0
                END
            ) AS null_keys,

            SUM(
                CASE
                    WHEN eval_set IS NULL
                         OR eval_set NOT IN ('prior', 'train', 'test')
                    THEN 1 ELSE 0
                END
            ) AS invalid_eval_set,

            SUM(
                CASE
                    WHEN order_number IS NULL OR order_number < 1
                    THEN 1 ELSE 0
                END
            ) AS invalid_order_number,

            SUM(
                CASE
                    WHEN order_dow IS NULL
                         OR order_dow < 0
                         OR order_dow > 6
                    THEN 1 ELSE 0
                END
            ) AS invalid_order_dow,

            SUM(
                CASE
                    WHEN order_hour_of_day IS NULL
                         OR TRY_CAST(order_hour_of_day AS INTEGER) IS NULL
                         OR TRY_CAST(order_hour_of_day AS INTEGER) < 0
                         OR TRY_CAST(order_hour_of_day AS INTEGER) > 23
                    THEN 1 ELSE 0
                END
            ) AS invalid_order_hour,

            SUM(
                CASE
                    WHEN days_since_prior_order < 0
                         OR days_since_prior_order > 30
                    THEN 1 ELSE 0
                END
            ) AS invalid_days_since_prior

        FROM orders;
        """
    ).fetchone()

    labels = [
        ("Filas totales", result[0], False),
        ("Claves nulas", result[1], True),
        ("eval_set inválido", result[2], True),
        ("order_number inválido", result[3], True),
        ("order_dow fuera de 0-6", result[4], True),
        ("order_hour_of_day fuera de 0-23", result[5], True),
        ("days_since_prior_order fuera de 0-30", result[6], True),
    ]

    for label, value, is_validation in labels:
        if not is_validation:
            print(f"[INFO] {label}: {value:,}")
        elif value == 0:
            print(f"[OK] {label}: 0")
        else:
            print(f"[ERROR] {label}: {value:,}")
            errors.append(f"orders - {label}: {value}")

    return errors


def validate_order_products(
    connection,
    table_name: str,
    section_number: int,
) -> list[str]:
    """
    Valida una tabla de productos incluidos en pedidos.
    """
    errors = []

    print(f"\n{section_number}. Validación de {table_name}")

    result = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,

            SUM(
                CASE
                    WHEN order_id IS NULL OR product_id IS NULL
                    THEN 1 ELSE 0
                END
            ) AS null_keys,

            SUM(
                CASE
                    WHEN add_to_cart_order IS NULL
                         OR add_to_cart_order < 1
                    THEN 1 ELSE 0
                END
            ) AS invalid_cart_order,

            SUM(
                CASE
                    WHEN reordered IS NULL
                         OR reordered NOT IN (0, 1)
                    THEN 1 ELSE 0
                END
            ) AS invalid_reordered

        FROM {table_name};
        """
    ).fetchone()

    labels = [
        ("Filas totales", result[0], False),
        ("Claves nulas", result[1], True),
        ("add_to_cart_order inválido", result[2], True),
        ("reordered distinto de 0 o 1", result[3], True),
    ]

    for label, value, is_validation in labels:
        if not is_validation:
            print(f"[INFO] {label}: {value:,}")
        elif value == 0:
            print(f"[OK] {label}: 0")
        else:
            print(f"[ERROR] {label}: {value:,}")
            errors.append(f"{table_name} - {label}: {value}")

    return errors


def validate_primary_keys(connection) -> list[str]:
    """
    Comprueba la unicidad de las claves principales y compuestas.
    """
    errors = []

    print("\n5. Validación de claves duplicadas")

    key_definitions = {
        "aisles": ["aisle_id"],
        "departments": ["department_id"],
        "products": ["product_id"],
        "orders": ["order_id"],
        "order_products_prior": ["order_id", "product_id"],
        "order_products_train": ["order_id", "product_id"],
    }

    for table_name, key_columns in key_definitions.items():
        columns_text = ", ".join(key_columns)

        duplicate_rows = connection.execute(
            f"""
            SELECT COALESCE(SUM(row_count - 1), 0)
            FROM (
                SELECT COUNT(*) AS row_count
                FROM {table_name}
                GROUP BY {columns_text}
                HAVING COUNT(*) > 1
            ) AS duplicated_keys;
            """
        ).fetchone()[0]

        if duplicate_rows == 0:
            print(f"[OK] {table_name}: 0 filas duplicadas por clave")
        else:
            print(
                f"[ERROR] {table_name}: "
                f"{duplicate_rows:,} filas duplicadas por clave"
            )
            errors.append(
                f"{table_name} - filas duplicadas por clave: "
                f"{duplicate_rows}"
            )

    return errors


def validate_relationships(connection) -> list[str]:
    """
    Comprueba las relaciones entre productos, categorías y pedidos.
    """
    errors = []

    print("\n6. Validación de relaciones entre tablas")

    relationship_queries = {
        "Productos con aisle_id huérfano": """
            SELECT COUNT(*)
            FROM products AS p
            LEFT JOIN aisles AS a
                ON p.aisle_id = a.aisle_id
            WHERE a.aisle_id IS NULL;
        """,
        "Productos con department_id huérfano": """
            SELECT COUNT(*)
            FROM products AS p
            LEFT JOIN departments AS d
                ON p.department_id = d.department_id
            WHERE d.department_id IS NULL;
        """,
        "Filas prior con product_id huérfano": """
            SELECT COUNT(*)
            FROM order_products_prior AS op
            LEFT JOIN products AS p
                ON op.product_id = p.product_id
            WHERE p.product_id IS NULL;
        """,
        "Filas train con product_id huérfano": """
            SELECT COUNT(*)
            FROM order_products_train AS op
            LEFT JOIN products AS p
                ON op.product_id = p.product_id
            WHERE p.product_id IS NULL;
        """,
        "Filas prior sin pedido prior correspondiente": """
            SELECT COUNT(*)
            FROM order_products_prior AS op
            LEFT JOIN orders AS o
                ON op.order_id = o.order_id
               AND o.eval_set = 'prior'
            WHERE o.order_id IS NULL;
        """,
        "Filas train sin pedido train correspondiente": """
            SELECT COUNT(*)
            FROM order_products_train AS op
            LEFT JOIN orders AS o
                ON op.order_id = o.order_id
               AND o.eval_set = 'train'
            WHERE o.order_id IS NULL;
        """,
    }

    for label, query in relationship_queries.items():
        invalid_rows = connection.execute(query).fetchone()[0]

        if invalid_rows == 0:
            print(f"[OK] {label}: 0")
        else:
            print(f"[ERROR] {label}: {invalid_rows:,}")
            errors.append(f"{label}: {invalid_rows}")

    return errors


def main() -> None:
    """Ejecuta todas las validaciones de calidad."""
    print("Iniciando validación básica de los datos...")

    connection = create_connection()
    errors = []

    try:
        errors.extend(validate_required_columns(connection))

        # Solo valida el contenido si están todas las columnas requeridas.
        if not errors:
            errors.extend(validate_orders(connection))

            errors.extend(
                validate_order_products(
                    connection,
                    "order_products_prior",
                    3,
                )
            )

            errors.extend(
                validate_order_products(
                    connection,
                    "order_products_train",
                    4,
                )
            )

            errors.extend(validate_primary_keys(connection))
            errors.extend(validate_relationships(connection))

    finally:
        connection.close()

    print("\nResumen de validación")

    if errors:
        print(f"Se encontraron {len(errors)} problema(s):")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("Todas las validaciones fueron superadas.")
    print("Los archivos originales no fueron modificados.")


if __name__ == "__main__":
    main()