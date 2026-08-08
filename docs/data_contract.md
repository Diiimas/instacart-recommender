# Contrato de datos

## Propósito

Este documento define las entradas esperadas y las salidas iniciales del pipeline de datos. Su objetivo es permitir que el EDA, la ingeniería de características y el modelado trabajen sobre estructuras reproducibles.

## Datos de entrada

Los archivos originales deben ubicarse en `data/raw/`:

| Archivo | Granularidad |
|---|---|
| `orders.csv` | Un registro por pedido |
| `order_products__prior.csv` | Un registro por producto incluido en un pedido histórico |
| `order_products__train.csv` | Un registro por producto incluido en un pedido objetivo |
| `products.csv` | Un registro por producto |
| `aisles.csv` | Un registro por pasillo |
| `departments.csv` | Un registro por departamento |

Los archivos originales son de solo lectura: el pipeline no debe modificarlos ni sobrescribirlos.

## Claves principales

| Tabla | Clave |
|---|---|
| `orders` | `order_id` |
| `order_products__prior` | `order_id`, `product_id` |
| `order_products__train` | `order_id`, `product_id` |
| `products` | `product_id` |
| `aisles` | `aisle_id` |
| `departments` | `department_id` |

## Relaciones esperadas

- `orders.order_id` se relaciona con `order_products__prior.order_id`.
- `orders.order_id` se relaciona con `order_products__train.order_id`.
- `order_products__prior.product_id` y `order_products__train.product_id` se relacionan con `products.product_id`.
- `products.aisle_id` se relaciona con `aisles.aisle_id`.
- `products.department_id` se relaciona con `departments.department_id`.

## Reglas de integridad iniciales

El pipeline debe verificar:

- presencia de los seis archivos;
- nombres de columnas obligatorias;
- tipos de datos compatibles;
- unicidad de las claves principales;
- ausencia de claves huérfanas en las relaciones;
- valores nulos en campos críticos;
- correspondencia entre `eval_set` y las tablas `prior` y `train`.

La validación debe informar los errores sin modificar los datos originales.

## Separación entre historial y objetivo

Como protocolo inicial:

- `order_products__prior` se utiliza para construir el historial;
- `order_products__train` se reserva como verdad de referencia para evaluación;
- no se pueden construir variables usando productos del pedido objetivo;
- baseline y modelos personalizados deben evaluarse sobre los mismos usuarios y pedidos objetivo.

Esta decisión debe confirmarse con el equipo antes de la evaluación definitiva.

## Salidas previstas

El pipeline podrá generar en `data/interim/`:

- tablas validadas;
- agregaciones por usuario;
- agregaciones por producto;
- historial usuario-producto;
- controles de calidad.

Las tablas finales para modelado se guardarán en `data/processed/`.

Los nombres, columnas y formatos definitivos de las salidas se documentarán cuando el equipo confirme los requerimientos del modelo.

## Convenciones

- Los nombres de archivos y columnas se mantienen en `snake_case`.
- Los datos crudos nunca se sobrescriben.
- Toda salida debe poder regenerarse mediante código.
- Los CSV crudos y los datos procesados no se versionan en GitHub.
- Los cambios en este contrato se realizan mediante Pull Request.