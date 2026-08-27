# Análisis de resultados y métricas de evaluación

**Data Analyst · Anastasia Ganderats · Sprint 2 (modelo final)**

Lectura de los resultados del sistema final y cómo nombramos la evaluación. Se apoya en la corrida
final de Julieta (protocolo, modelos y sistema de dos bloques) y en el análisis de features y
segmentos del EDA.

> **Universos (importante):** la comparación oficial entre modelos (LightGBM vs. baselines) y los
> resultados del sistema final se calculan sobre los **26.243 usuarios de validación**. El análisis de
> features, segmentos y cobertura del catálogo del EDA usa los **131.209 usuarios evaluables**. Son
> universos distintos, a propósito, y no se mezclan.

## 1. El sistema final: dos bloques

El sistema no devuelve una sola lista, sino dos, en lugares distintos de la pantalla:

- **Bloque principal (recompra):** hasta 10 productos que el cliente vuelve a comprar. Es la parte
  fuerte.
- **Bloque de sugerencias (novedad):** hasta 5 / 2 / 1 productos nuevos según el segmento (nuevo /
  medio / heavy). Es descubrimiento.

Cada bloque se mide con su propia vara. La clave de todo el análisis es no mezclarlos: la recompra y
el descubrimiento son necesidades distintas y se evalúan aparte.

## 2. Bloque principal: qué tan bien predecimos la recompra

Comparación oficial sobre los 26.243 usuarios de validación, mismo protocolo y mismo corte K=10.

| Modelo | Hit Rate@10 | Recall@10 macro | Cobertura | Lift |
|---|---:|---:|---:|---:|
| Popularidad (baseline) | 46,1 % | 0,070 | 0,0 % | 1,0x |
| Recompra personal | 85,5 % | 0,327 | 44,9 % | 3,66x |
| Heurística | 85,5 % | 0,339 | 43,3 % | 3,82x |
| Regresión logística | 86,8 % | 0,348 | 36,9 % | 3,96x |
| **LightGBM (elegido)** | **87,2 %** | **0,356** | 39,0 % | **4,07x** |

**Lectura:** el Hit Rate pasa de 46 % (popularidad) a 87 % (LightGBM), casi el doble. El bloque
principal le acierta al menos un producto habitual a **87 de cada 100 clientes** (79.401 aciertos
totales). El lift de 4,07x dice que captura la recompra cuatro veces mejor que recomendar lo popular.

## 3. Bloque de sugerencias: qué tan bien descubrimos

El bloque de novedad suma **1.964 aciertos de descubrimiento sin resignar ninguno de recompra**,
porque va aparte del principal. Rinde distinto según el segmento, y por eso su tamaño cambia:

| Segmento | Recompra (Hit Rate) | Sugerencias (hasta) | Acierto de novedad | Aciertos nuevos |
|---|---:|---:|---:|---:|
| Nuevos | 83,0 % | 5 | 16,5 % | 1.247 |
| Medios | 87,2 % | 2 | 6,6 % | 573 |
| Heavy | 91,1 % | 1 | 2,3 % | 144 |

**Lectura:** la recompra sube con la antigüedad (al heavy le acertamos casi siempre), pero el
descubrimiento va al revés, rinde mucho más en los nuevos (16,5 % contra 2,3 %). Tiene sentido: el
cliente nuevo todavía está explorando. Por eso el bloque de sugerencias es más grande justo donde más
se necesita.

> **Los tamaños son máximos, no fijos.** El bloque principal puede traer menos de 10 productos cuando
> el cliente tiene poco historial (le pasa al 6 % de los usuarios, y al 14 % de los nuevos), y las
> sugerencias también pueden venir incompletas. En la interfaz se comunica como **hasta 10** y **hasta
> 5 / 2 / 1**.

## 4. Por qué la cobertura NO valida descubrimiento

La recompra cubre el **71,9 % del catálogo** sobre los evaluables (39 % en validación), pero tiene
descubrimiento cero por construcción: nunca recomienda algo fuera del historial del cliente. Cubre
mucho solo porque a cada persona le da sus propios productos. La cobertura mide **amplitud del catálogo
tocado**, no capacidad de sorprender. Usarla para la historia de descubrimiento nos engañaría. El
descubrimiento se mide con su propio acierto de novedad, sobre los targets que el cliente nunca compró
(hay 555.793 objetivos nuevos entre 107.008 clientes).

## 5. Cheat-sheet de métricas (para no mezclar nombres)

| Métrica | Qué mide | Ojo |
|---|---|---|
| Hit Rate@10 | % de clientes con al menos 1 acierto | **NO es Recall**. En popularidad, Recall 7 % vs Hit Rate 46 % son los mismos aciertos con distinto nombre |
| Recall@10 macro | promedio del recall por cliente (todos pesan igual) | queda por encima del micro |
| Recall@10 micro | aciertos totales / productos totales (quien compra más pesa más) | 0,286 en el modelo elegido |
| Cobertura de catálogo | % del catálogo recomendado | mide amplitud, **no** descubrimiento |
| Acierto de novedad | Hit Rate solo sobre targets nuevos | esta sí es la métrica de descubrimiento |

**Regla para los gráficos:** el título dice la métrica exacta y su definición. Nunca "Recall" a
secas; si mostramos una con el nombre de la otra, nos equivocamos por hasta seis veces.

## 6. El límite es el dato

Después de elegir LightGBM probamos cinco caminos para exprimirlo más: seis variables nuevas, objetivo
de ranking (lambdarank), más datos de entrenamiento, hiperparámetros con Optuna y el tamaño del
carrito. Ninguno movió la aguja. Dos ejemplos claros: sumar seis variables dejó el Hit Rate en 87,20 %
contra 87,22 %, y cuadruplicar los datos de entrenamiento (de 25 % a 100 %) dejó el Recall clavado en
~0,355.

**Conclusión que respaldan estos experimentos:** el límite no es el modelo, es el dato. Para crecer
hace falta información que el historial de compras no tiene, como precio o promociones, y el dataset de
Instacart no la trae. Saberlo evita perder tiempo afinando un modelo que ya tocó su techo.

## 7. Control cualitativo (hecho)

En el dashboard, el explorador muestra para cualquiera de los 26.243 clientes de validación sus dos
bloques con lo que acertó de verdad (columna del archivo de recomendaciones). Revisado con casos de
los tres segmentos: las recomendaciones tienen sentido de negocio, y se ve con transparencia dónde el
sistema acierta y dónde llega a su techo (por ejemplo, un cliente nuevo cuyo próximo pedido es casi
todo nuevo).

## 8. Limitaciones y próximos pasos

- El dataset es de **recompra**: su naturaleza dificulta el descubrimiento, y el generador de novedad
  concentra sus sugerencias en pocos productos distintos. Ahí está el margen de mejora.
- **Reglas de asociación (notebook 05):** con lift aparecen complementos accionables (vinos con vinos,
  lavandería con limpieza, tofu con congelados veganos). Son la base para diversificar el bloque de
  sugerencias más allá del generador actual.
- **Enriquecer el dato:** sumar precio y promociones es lo que destraba el descubrimiento, según los
  cinco experimentos de la sección 6.
- El protocolo (split temporal, sin fuga) y las métricas están unificadas en el módulo común de
  evaluación.
