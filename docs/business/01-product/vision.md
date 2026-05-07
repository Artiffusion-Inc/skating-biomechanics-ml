# Product Vision & Value Proposition

## Mission

AI-тренер по фигурному катанию — анализ видео и рекомендации на русском языке.

SkateLab — SaaS-платформа для фигуристов и тренеров, которая анализирует видео тренировок с помощью ML, измеряет биомеханические параметры элементов и даёт персонализированные рекомендации для улучшения техники.

## Problem Statement

Фигуристы и тренеры сталкиваются с тремя ключевыми проблемами:

1. **Нет объективных данных.** Оценка техники основана на субъективном восприятии тренера. Нет числовых метрик: высота прыжка, скорость вращения, качество приземления.
2. **Дорого.** Биомеханический анализ доступен только национальным сборным (Omega — 14 камер вокруг катка, закрытый софт). Индивидуальные фигуристы и малые клубы не имеют доступа.
3. **Языковой барьер.** Существующие решения — на английском. Русскоязычные фигуристы (Россия, Казахстан, Беларусь, Латвия — ключевые рынки) не получают анализа на родном языке.

## Value Proposition

**Для фигуристов:** Загрузил видео → получил объективные метрики и рекомендации на русском. Сравнил себя с эталоном. Отследил прогресс.

**Для тренеров:** Дашборд учеников, объективные данные для корректировки техники, трекинг прогресса во времени.

**Для клубов:** Масштабирование качества тренерской работы, единая система аналитики.

### Unique Differentiators

| Дифференциатор | Что это | Почему важно |
|---------------|---------|--------------|
| OOFSkate proxy features | Оценка качества через кинематику тела, не через лезвие конька | Работает с обычного телефона, не нужны 14 камер или IMU-датчики |
| H3.6M 17kp 3D | Стандартный формат скелета, совместимый с исследованиями | Легко интегрировать новые модели из academia |
| CoM trajectory | Центр масс вместо времени полёта | Устраняет 60% ошибку в измерении высоты прыжка |
| Русский язык | Полная локализация: UI, рекомендации, метрики | Единственный продукт с native русским |
| Хореограф-планировщик | ISU element DB + CSP solver + music analysis | Визуальное планирование программы с учётом музыки |
| Anatomical Re-ID | Идентификация фигуриста по биометрии, не по одежде | Не путает людей в одинаковой чёрной форме на льду |

## Product Components

### Core ML Pipeline

```
Video → RTMO (rtmlib, CUDA, COCO 17kp)
  → COCO→H3.6M conversion → GapFiller → Smoothing (One-Euro, Numba JIT)
  → Phase Detection (CoM-based, adaptive sigma)
  → Biomechanics Metrics (airtime, height, knee angles, rotation, landing quality)
  → DTW alignment vs reference → GOE proxy score
  → Rule-based Recommender → Russian Text Report
```

### SaaS Platform

| Компонент | Описание |
|-----------|----------|
| Upload | Chunked S3 multipart upload, presigned URLs для R2 |
| Sessions | CRUD с метриками, персистенция в Postgres |
| Metrics Registry | 12+ биомеханических метрик, русские лейблы |
| Progress Dashboard | Тренды, PR трекер, диагностика (5 правил) |
| Coach Dashboard | Ученики, сессии, диагностика, управление связями |
| Choreography Planner | ISU elements + CSP solver + music analysis + SVG rink renderer |
| i18n | next-intl, русский + английский |

### Tech Stack

| Слой | Технология |
|------|-----------|
| ML Pipeline | Python, rtmlib, onnxruntime-gpu, scipy, numba |
| Backend | FastAPI, SQLAlchemy, Alembic, arq + Valkey |
| Frontend | Next.js 16, React, Tailwind, shadcn/ui, Recharts, three.js |
| Storage | Cloudflare R2 (S3-compatible), Postgres |
| Remote GPU | Vast.ai Serverless |
| Task Runner | go-task |

## Current Status

**MVP 100% complete** (ROADMAP.md — single source of truth).

- 65+ файлов кода, 5341+ нод в графе знаний
- Pipeline: ~12с для 14.5с видео (GPU, frame_skip=8)
- 279+ тестов
- GPU-only inference (CPU запрещён)

## Key Constraints

- **GPU-only.** CPU inference запрещена. `device='cuda'`.
- **Backend не импортирует ML internals.** arq worker может импортировать ML типы, но не вызывать pipeline напрямую.
- **Тяжёлый ML → GPU** (локальный или Vast.ai Serverless).
