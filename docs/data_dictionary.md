# Diccionario de datos procesados

## Objetivo

Este documento describe las tablas analíticas generadas por
`src/data/build_dataset.py`.

El pipeline utiliza DuckDB para procesar los archivos originales de Instacart
sin modificarlos. Las variables predictoras se construyen exclusivamente con
las órdenes del conjunto `prior`, mientras que las órdenes `train` se mantienen
separadas como target de evaluación para evitar fuga de información.

---

## catalogo.parquet

Catálogo completo y legible de productos.

**Granularidad:** una fila por producto.

**Cantidad de filas:** 49.688.

| Columna | Tipo | Descripción |
|---|---|---|
| `product_id` | INTEGER | Identificador único del producto. |
| `product_name` | VARCHAR | Nombre del producto. |
| `aisle_id` | INTEGER | Identificador del pasillo al que pertenece. |
| `aisle` | VARCHAR | Nombre del pasillo. |
| `department_id` | INTEGER | Identificador del departamento al que pertenece. |
| `department` | VARCHAR | Nombre del departamento. |

---

## targets_train.parquet

Productos incluidos en la última orden revelada de los usuarios evaluables.

Esta tabla se utiliza únicamente como verdad conocida para evaluar las
recomendaciones. No participa en la construcción de las variables predictoras.

**Granularidad:** una fila por producto incluido en una orden `train`.

**Cantidad de filas:** 1.384.617.

**Cantidad de usuarios:** 131.209.

| Columna | Tipo | Descripción |
|---|---|---|
| `user_id` | INTEGER | Identificador del usuario evaluable. |
| `order_id` | INTEGER | Identificador de su orden objetivo. |
| `product_id` | INTEGER | Producto presente en la orden objetivo. |
| `add_to_cart_order` | INTEGER | Posición en la que el producto fue agregado al carrito. |
| `reordered` | TINYINT | Indica si el usuario ya había comprado el producto anteriormente. |

---

## productos.parquet

Perfil agregado de cada producto, calculado exclusivamente con las compras
históricas del conjunto `prior`.

**Granularidad:** una fila por producto.

**Cantidad de filas:** 49.688.

| Columna | Tipo | Descripción |
|---|---|---|
| `product_id` | INTEGER | Identificador único del producto. |
| `product_name` | VARCHAR | Nombre del producto. |
| `aisle_id` | INTEGER | Identificador del pasillo. |
| `aisle` | VARCHAR | Nombre del pasillo. |
| `department_id` | INTEGER | Identificador del departamento. |
| `department` | VARCHAR | Nombre del departamento. |
| `cantidad_compras` | BIGINT | Cantidad total de compras históricas del producto. |
| `cantidad_usuarios` | INTEGER | Cantidad de usuarios distintos que compraron el producto. |
| `reorder_rate_producto` | DOUBLE | Proporción de compras del producto marcadas como recompra. |
| `add_to_cart_order_promedio` | DOUBLE | Posición promedio del producto dentro del carrito. |

---

## usuarios.parquet

Perfil de comportamiento de cada usuario, calculado exclusivamente con sus
órdenes históricas del conjunto `prior`.

**Granularidad:** una fila por usuario.

**Cantidad de filas:** 206.209.

| Columna | Tipo | Descripción |
|---|---|---|
| `user_id` | INTEGER | Identificador único del usuario. |
| `cantidad_ordenes_historicas` | INTEGER | Cantidad de órdenes `prior` realizadas por el usuario. |
| `ultima_order_number` | INTEGER | Número de la última orden incluida en el historial disponible. |
| `cantidad_compras` | BIGINT | Cantidad total de productos comprados en el historial. |
| `productos_distintos` | INTEGER | Cantidad de productos diferentes comprados. |
| `reorder_rate_usuario` | DOUBLE | Proporción de compras marcadas como recompra. |
| `posicion_media_carrito` | DOUBLE | Posición promedio de sus productos dentro del carrito. |
| `dow_habitual` | TINYINT | Día de la semana más frecuente para realizar pedidos. |
| `hora_habitual` | TINYINT | Hora del día más frecuente para realizar pedidos. |
| `tiene_primera_orden` | TINYINT | Indica si se identificó la primera orden del usuario mediante un intervalo nulo. |
| `tiene_intervalo_censurado_30` | TINYINT | Indica si el usuario presenta al menos un intervalo registrado con el límite de 30 días. |
| `cantidad_intervalos_censurados_30` | INTEGER | Cantidad de intervalos entre órdenes registrados como 30 días. |
| `segmento_usuario` | VARCHAR | Segmento según cantidad de órdenes históricas: `nuevo`, `medio` o `heavy`. |

### Segmentación de usuarios

| Segmento | Regla | Usuarios |
|---|---:|---:|
| `nuevo` | Hasta 5 órdenes históricas | 59.741 |
| `medio` | Entre 6 y 15 órdenes históricas | 81.172 |
| `heavy` | Más de 15 órdenes históricas | 65.296 |

---

## interacciones.parquet

Tabla analítica principal para generar candidatos y entrenar modelos de
recomendación personalizados.

**Granularidad:** una fila por combinación única de usuario y producto.

**Cantidad de filas:** 13.307.953.

**Compras históricas representadas:** 32.434.489.

| Columna | Tipo | Descripción |
|---|---|---|
| `user_id` | INTEGER | Identificador del usuario. |
| `product_id` | INTEGER | Identificador del producto comprado. |
| `freq_usuario_producto` | INTEGER | Cantidad de veces que el usuario compró el producto. |
| `recencia_usuario_producto` | INTEGER | Cantidad de órdenes transcurridas entre la última compra del producto y la última orden histórica del usuario. Un valor de 0 indica que apareció en su orden histórica más reciente. |
| `ultima_orden_producto` | INTEGER | Número de la última orden histórica en la que el usuario compró el producto. |
| `add_to_cart_order_promedio` | DOUBLE | Posición promedio del producto en los carritos del usuario. |
| `ultima_order_dow` | TINYINT | Día de la semana de la última compra histórica del producto. |
| `ultima_order_hour` | TINYINT | Hora del día de la última compra histórica del producto. |
| `aisle_id` | INTEGER | Identificador del pasillo del producto. |
| `department_id` | INTEGER | Identificador del departamento del producto. |

---

## Reglas metodológicas

- Los archivos originales de `data/raw/` no son modificados.
- Las variables predictoras se calculan únicamente con órdenes `prior`.
- Las órdenes `train` se conservan separadas como target de evaluación.
- Las órdenes `test` no se emplean para calcular métricas porque su contenido
  real no está revelado.
- `days_since_prior_order = NULL` identifica la primera orden de cada usuario.
- El valor 30 en `days_since_prior_order` se interpreta como un intervalo
  censurado por el límite del dataset.
- La recencia se mide en número de órdenes y no en días calendario.
- Los archivos Parquet se regeneran ejecutando:

```powershell
python .\src\data\build_dataset.py