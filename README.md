# Instacart Recommender

Sistema de recomendación de productos basado en el historial de compras de Instacart.

Proyecto final colaborativo de Ciencia de Datos desarrollado mediante metodología Scrum.

## Objetivo

Construir un sistema capaz de generar diez recomendaciones de productos para cada usuario a partir de su historial de compras, comparándolo con un baseline reproducible de popularidad.

El proyecto busca combinar:

- análisis exploratorio de datos;
- procesamiento reproducible;
- ingeniería de características;
- recomendación personalizada;
- evaluación con métricas comunes;
- interpretación de resultados desde una perspectiva de negocio.

## Equipo

| Integrante | Rol principal |
|---|---|
| Dimas Giménez | Data Engineer y líder técnico |
| Anastasia | Análisis exploratorio de datos |
| Julieta | Modelado y evaluación |
| Leonardo | Negocio y documentación |

Los roles representan áreas de liderazgo. La integración y revisión del producto son responsabilidad de todo el equipo.

## Datos

El proyecto utiliza el conjunto de datos público de Instacart, compuesto por seis tablas:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

Los archivos originales deben ubicarse localmente en `data/raw/` y no se almacenan en el repositorio debido a su volumen.

## Enfoque inicial

La arquitectura conserva la granularidad de las tablas originales. No se construye una única tabla monolítica con todos los archivos.

Como hipótesis inicial del Sprint 1:

- `prior` representa el historial disponible;
- `train` se reserva como objetivo para evaluación;
- ninguna variable puede utilizar información del pedido objetivo;
- el baseline y el modelo personalizado deben evaluarse sobre los mismos usuarios, targets y métricas;
- se generarán recomendaciones `Top 10`.

Estas decisiones deberán mantenerse documentadas y actualizarse si el equipo modifica el protocolo de evaluación.

## Métricas mínimas propuestas

- `Precision@10`
- `Recall@10`
- cobertura de catálogo
- proporción de usuarios con diez recomendaciones válidas

Las métricas definitivas serán confirmadas por el equipo antes de evaluar los modelos.

## Estructura del repositorio

```text
instacart-recommender/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── tests/
├── docs/
│   └── decisions/
├── README.md
├── SETUP.md
└── requirements.txt