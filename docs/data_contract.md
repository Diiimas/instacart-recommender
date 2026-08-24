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

- `order_products__prior` se utiliza para construir el historial;
- `order_products__train` se reserva como verdad de referencia para evaluación;
- no se pueden construir variables usando los **productos** del pedido objetivo;
- baseline y modelos personalizados deben evaluarse sobre los mismos usuarios y pedidos objetivo.

### Qué sí se puede usar del pedido objetivo

El **contexto** del pedido objetivo no es fuga y sí se usa: cuándo ocurre el pedido, es decir su día de la semana, su hora y cuántos días pasaron desde la compra anterior.

La distinción es entre *cuándo* compra y *qué* compra. En producción, cuando el sistema tiene que recomendar, la persona ya está armando el carrito: sabemos el día y la hora, y sabemos hace cuánto fue su última compra. Lo que no sabemos es qué va a poner adentro.

Usar el contenido del pedido objetivo sí sería fuga y está prohibido.

## Protocolo de evaluación

### Tamaño de la recomendación

**K = 10 productos por cliente.**

El valor se midió contra ocho tamaños distintos, de 5 a 30. El óptimo por F1 es exactamente 10, con un máximo plano entre 8 y 12. Lo sostiene la distribución: solo el 18,4 % de las órdenes tiene más de 10 productos repetidos, así que para el resto agrandar la lista no mueve el techo.

Reproducible con `python src/models/optimizar_n.py`.

### Comparación entre sistemas

Todos los sistemas que se comparen entre sí deben medirse con:

- los mismos usuarios de validación;
- el mismo corte temporal;
- el mismo K;
- el mismo código de evaluación, que es `src/evaluation/metrics.py`.

Ningún sistema se mide con su propia implementación de las métricas. Una tabla comparativa cuyas filas salgan de corridas distintas no es válida.

### Desempate explícito

**Todo ranking debe declarar un criterio de desempate. Nunca puede quedar librado al orden de las filas.**

Esta regla no es preventiva: en el Sprint 1 el baseline de recompra empataba en sus dos criterios de orden para dos tercios de las filas, y el desempate lo terminaba decidiendo el orden del archivo. Ese orden no era neutral, porque una operación previa había dejado arriba las filas de la orden objetivo. El baseline se llevaba así 1,3 puntos de recall que no había ganado.

El criterio de desempate debe además ser **neutral respecto de la variable objetivo**. Hoy se usa `product_id` ascendente, que es arbitrario pero no aporta señal.

### Métricas oficiales

Las calcula `src/evaluation/metrics.py`. Ninguna se reemplaza ni se descarta.

| Métrica | Denominador | Para qué |
|---|---|---|
| `hit_rate` | usuarios evaluados | **KPI principal.** A qué proporción de clientes le acertamos al menos un producto |
| `recall` | productos del pedido real, promediado por usuario | Qué parte de la compra capturamos |
| `recall_micro` | productos objetivo totales | La misma lectura, ponderada por volumen |
| `precision` | K | De lo que mostramos, qué proporción acertó |
| `f1` | — | Media armónica de precision y recall |
| `cobertura` | catálogo | Qué proporción del catálogo llega a recomendarse |
| `lift` | métrica del baseline de popularidad | Cuánto mejora sobre recomendar lo más vendido |

Convenciones:

- Los promedios son **macro**: se calcula la métrica por usuario y después se promedia entre usuarios, de modo que cada persona pesa igual. `recall_micro` acompaña como lectura de volumen agregado.
- `precision` divide por K y no por la cantidad de recomendaciones entregadas: si el sistema devuelve menos de K, el hueco se paga, porque en la interfaz real queda vacío igual.
- Un usuario sin recomendaciones cuenta como cero aciertos y no se saltea. Saltearlo premiaría al sistema por no contestar.

### Métricas de novedad

La recompra y el descubrimiento son problemas distintos y se miden por separado. **No se promedian entre sí ni se reportan en la misma escala**: anticipar lo que alguien nunca compró es mucho más difícil, y mezclarlas esconde las dos cosas.

Un producto es **nuevo para un usuario** si no está en *su* historial. Es relativo a la persona, no al catálogo.

| Métrica | Se promedia sobre | Para qué |
|---|---|---|
| `novedad_ofrecida` | todos los usuarios | Cuántos de los K lugares dedicamos a lo desconocido. Es una decisión de diseño, no un resultado |
| `precision_novedad` | los que recibieron algo nuevo | De lo nuevo que ofrecimos, cuánto acertó |
| `recall_novedad` | los que compraron algo nuevo | De lo nuevo que compró, cuánto anticipamos |
| `hit_rate_novedad` | los que compraron algo nuevo | A cuántos les acertamos al menos un producto nuevo |

Las tres últimas se promedian sobre subconjuntos distintos de usuarios, y cada una reporta su propio `n` junto al valor. A quien no compró nada nuevo no hay contra qué medirlo, y contarlo como cero inventaría un fracaso que no existió.

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