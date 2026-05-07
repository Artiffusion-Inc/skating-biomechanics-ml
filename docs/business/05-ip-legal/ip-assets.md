# Intellectual Property & Legal

## IP Assets

### 1. Proprietary Software

| Актив | Статус | Защита |
|-------|--------|--------|
| ML Pipeline (pose estimation, analysis, recommender) | Proprietary code | Copyright (авторское право) |
| OOFSkate proxy features implementation | Proprietary | Trade secret + copyright |
| Anatomical Re-ID algorithm | Proprietary | Trade secret |
| Choreography CSP solver | Proprietary | Copyright |
| CoM-based phase detection | Proprietary | Trade secret |
| Russian recommendation engine | Proprietary | Copyright |

### 2. Datasets (Curated)

| Датасет | Происхождение | Права | Статус |
|---------|-------------|-------|--------|
| SkatingVerse (28K видео) | Public dataset | Research use | Скачан |
| Figure-Skating-Classification (5168 seq) | HuggingFace | CC/Research | Скачан |
| MCFS (2668 segments) | GitHub | Research | Скачан |
| AthletePose3D (1.3M frames) | GitHub | Research | Скачан |
| Reference database (эталоны) | Self-built | Proprietary | В разработке |

> **Риск:** Датасеты research-only → нужен чек лицензий для коммерческого использования. Некоторые могут требовать отдельного соглашения.

### 3. Models

| Модель | Происхождение | Лицензия | Коммерческое использование |
|--------|-------------|---------|--------------------------|
| RTMO/Body (COCO 17kp) | rtmlib (OpenMMLab) | Apache 2.0 | ✅ Да |
| MotionAGFormer | AthletePose3D (Nagoya U) | ? | ❓ Нужна проверка |
| TCPFormer | AthletePose3D | ? | ❓ Нужна проверка |
| ONNX Runtime | Microsoft | MIT | ✅ Да |
| YOLO26-Pose | Ultralytics | AGPL-3.0 | ⚠️ Требует open-source производных |

### 4. Trade Secrets

- Адаптивные пороги phase detection (sigma-based)
- Биометрические пропорции для Re-ID
- Правила рекомендаций (jump_rules, three_turn_rules)
- Идеальные диапазоны элементов (element_defs)

## Legal Structure

### Required

| Документ | Статус | Примечание |
|----------|--------|-----------|
| Terms of Service | ❌ Не создан | Нужен перед пилотом |
| Privacy Policy | ❌ Не создан | GDPR/152-ФЗ compliance |
| Data Processing Agreement | ❌ Не создан | Для B2B клиентов |
| Cookie Policy | ❌ Не создан | EU users |

### Data Privacy

| Регламент | Требования | Статус |
|-----------|-----------|--------|
| 152-ФЗ (Россия) | Персональные данные → серверы в РФ | ❌ Нет (R2/CF — США/EU) |
| GDPR (EU) | Consent, data portability, right to be forgotten | ❌ Нет |
| COPPA (дети) | Parental consent для <13 | ❓ Нужен (родители — сегмент B2) |

> **Критичный риск:** 152-ФЗ требует хранение персональных данных россиян на серверах в РФ. R2/Cloudflare — США/EU. Решение: российский хостинг для БД + R2 для медиа (персональные данные ≠ видео).

### IP Risks

| Риск | Вероятность | Влияние | Mitigation |
|------|------------|---------|-----------|
| AGPL-3.0 (YOLO26-Pose) | Средняя | Высокое | Перейти на RTMO (Apache 2.0) полностью |
| Dataset licensing | Высокая | Высокое | Проверить все лицензии, получить коммерческие |
| Patent troll (ML methods) | Низкая | Высокое | Prior art documentation |
| Employee IP (open-source contribs) | Средняя | Среднее | CLA для контрибьюторов |

### Competitive IP Position

**Moat:** Не патенты, а:
1. **Data network effect** — больше пользователей → больше эталонов → лучше рекомендации
2. **Russian language lock-in** — нет конкурентов с native русским
3. **Domain expertise** — специфичные знания о фигурном катании (ISU rules, element definitions, biomechanics)

**Что НЕ стоит патентовать:**
- ML pipeline (быстро устареет, 18 мес на патент)
- Алгоритмы (patent → раскрытие → копирование)

**Что стоит защищать:**
- Бренд SkateLab → товарный знак
- Дизайн → авторское право
- Dataset эталонов → trade secret