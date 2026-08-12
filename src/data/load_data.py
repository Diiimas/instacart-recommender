"""
Carga de los archivos originales de Instacart mediante DuckDB.

Este módulo:
- Localiza automáticamente la raíz del repositorio.
- Comprueba que existan los seis CSV requeridos.
- Crea vistas de DuckDB sin modificar los archivos originales.
- Evita cargar el dataset completo en memoria.
"""

from pathlib import Path

import duckdb


# Rutas principales del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Nombre de la vista en DuckDB: nombre del archivo
REQUIRED_FILES = {
    "aisles": "aisles.csv",
    "departments": "departments.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "orders": "orders.csv",
    "products": "products.csv",
}


def get_raw_file_paths() -> dict[str, Path]:
    """
    Comprueba que estén presentes todos los archivos requeridos.

    Returns
    -------
    dict[str, Path]
        Diccionario con el nombre lógico de cada tabla y su ruta.

    Raises
    ------
    FileNotFoundError
        Si falta uno o más archivos.
    """
    file_paths = {
        table_name: RAW_DATA_DIR / file_name
        for table_name, file_name in REQUIRED_FILES.items()
    }

    missing_files = [
        path.name
        for path in file_paths.values()
        if not path.is_file()
    ]

    if missing_files:
        missing_text = "\n- ".join(missing_files)

        raise FileNotFoundError(
            "Faltan los siguientes archivos en data/raw:\n"
            f"- {missing_text}"
        )

    return file_paths


def create_connection() -> duckdb.DuckDBPyConnection:
    """
    Crea una conexión en memoria y registra los CSV como vistas.

    Los archivos se consultan directamente desde el disco y no se
    copian ni se modifican.
    """
    connection = duckdb.connect(database=":memory:")
    file_paths = get_raw_file_paths()

    for table_name, file_path in file_paths.items():
        safe_path = file_path.as_posix().replace("'", "''")

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT *
            FROM read_csv_auto(
                '{safe_path}',
                header = true
            );
            """
        )

    return connection


def main() -> None:
    """Verifica los archivos y muestra las vistas creadas."""
    print(f"Directorio de datos: {RAW_DATA_DIR}")

    connection = create_connection()

    tables = connection.execute("SHOW TABLES").fetchall()

    print("\nArchivos encontrados y vistas creadas correctamente:")

    for (table_name,) in tables:
        print(f"- {table_name}")

    connection.close()

    print("\nCarga inicial finalizada sin modificar los datos originales.")


if __name__ == "__main__":
    main()