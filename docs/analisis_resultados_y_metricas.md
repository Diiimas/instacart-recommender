# Análisis de resultados y propuesta de métricas de evaluación

**Data Analyst · Anastasia Ganderats · Daily 3**

Propuesta para acordar entre todos cómo leemos y nombramos la evaluación. Se apoya en los
resultados que ya midieron Julieta (protocolo + baselines) y Dimas (candidatos).

## 1. Lectura de los resultados del baseline (recompra personal)

| Métrica | Valor | Lectura |
|---|---|---|
| Hit Rate@10 | 85,5 % | 85 de cada 100 usuarios reciben al menos 1 acierto |
| Recall@10 macro | 0,330 | el cliente promedio recibe el 33 % de su lista objetivo |
| Recall@10 micro | 0,261 | capturamos el 26 % de todos los productos vendidos |
| Popularidad (Recall / Hit Rate) | 7 % / 45,8 % | mismos aciertos, distinto nombre |

**Interpretación:** el baseline de recompra es fortísimo en repetición y difícil de superar ahí
—exactamente lo que anticipaba el EDA (sesgo de popularidad + recompra alta)—. El margen de un
modelo que rankee mejor está en **heavy y medio** (la recompra captura el 51 % de lo alcanzable en
heavy vs. 71 % en nuevos), ~93.000 de los 131.209 evaluables. A los **nuevos** los destraba el
**descubrimiento** (candidatos), no un mejor ranking.

## 2. Por qué la cobertura NO valida "descubrimiento"

La recompra personal cubre el **71,9 % del catálogo** pero tiene **descubrimiento cero por
construcción**: nunca recomienda algo fuera del historial del usuario. Cubre mucho solo porque cada
persona recibe *sus* productos. Conclusión: la cobertura mide **amplitud del catálogo tocado**, no
capacidad de descubrimiento. Usarla para la historia de descubrimiento nos engaña.

## 3. Propuesta: evaluar con DOS lentes, no una

**Lente A — Repetición** (¿acertamos lo habitual?): sobre **todos** los targets.
- Recall@10 (macro y micro) y Hit Rate@10 → ya medidos.

**Lente B — Descubrimiento** (¿acertamos lo NUEVO relevante?): **solo** sobre los targets que el
usuario nunca compró (hay 555.793).
- **Recall@10 de novedad** y **Hit Rate@10 de novedad** → Dimas ya los computa (3,25 % / 14,24 %). Solo hay que nombrarlos como *la* métrica de descubrimiento.
- **Tasa de novedad**: % de las recomendaciones que son nuevas para el usuario (recompra = 0 %; candidatos > 0 %). Mide cuánto empuja el modelo más allá del historial.
- **Diversidad de la novedad**: nº de productos distintos usados en las recomendaciones nuevas (Dimas: el top-10 de candidatos usa solo 531 productos → baja). Es "cobertura", pero medida solo sobre lo nuevo, que es donde sí importa.

La cobertura de catálogo se mantiene como indicador de amplitud, con la aclaración explícita de que
**no** es descubrimiento.

## 4. Métrica principal por historia de usuario (a acordar)

| Historia de usuario | KPI principal | Acompañamiento |
|---|---|---|
| "Recomprar rápido lo habitual" | Hit Rate@10 (85,5 %) | Recall@10 macro/micro |
| "Descubrir productos nuevos relevantes" | Hit Rate@10 de novedad (14,24 %) | Tasa de novedad + Diversidad de novedad |

Así cada historia se mide con lo que de verdad la valida, y no confundimos "cubre mucho" con
"descubre bien".

## 5. Cheat-sheet de métricas (para el dashboard — no mezclar nombres)

| Métrica | Qué mide | Baseline | Ojo |
|---|---|---:|---|
| Recall@10 macro | promedio del recall por persona (todos pesan igual) | 0,330 | 7 puntos por encima del micro |
| Recall@10 micro | aciertos totales / productos totales (quien compra más pesa más) | 0,261 | — |
| Hit Rate@10 | % de usuarios con ≥1 acierto | 0,855 | **NO es Recall**: en popularidad, Recall 7 % vs Hit Rate 45,8 % = mismos aciertos, 6× de diferencia |
| Cobertura de catálogo | % del catálogo recomendado | 71,9 % | mide amplitud, **no** descubrimiento |
| Recall@10 de novedad | recall solo sobre targets nuevos | 3,25 % | esta sí es la métrica de descubrimiento |

**Regla para los gráficos:** el título dice la métrica exacta y su definición. Nunca "Recall" a
secas — si mostramos uno con el nombre del otro, nos equivocamos por hasta 6×.

## 6. Mi control cualitativo (siguiente paso)

Tomar 3–5 usuarios ejemplo (1 nuevo, 1 medio, 1 heavy) y mostrar su Top-10 de recompra vs. los
candidatos, marcando si el target real cayó adentro. Valida que las recomendaciones tienen sentido
de negocio, no solo que la métrica da bien.
