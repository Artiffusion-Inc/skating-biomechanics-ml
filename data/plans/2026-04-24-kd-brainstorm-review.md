# Knowledge Distillation: MogaNet-B → YOLO26-Pose — Синтез-отчёт 4 агентов

**Дата:** 2026-04-24
**Контекст:** Обзор текущей реализации KD (v33) перед полным обучением 210 epochs
**Статус:** КРИТИЧЕСКИЕ БАГИ ОБНАРУЖЕНЫ — обучение остановлено

---

## 1. Executive Summary

Текущая реализация KD (v33) содержит **3 showstopper-бага** и **фундаментальную архитектурную ошибку**: feature distillation на замороженном backbone бессмысленна, т.к. градиенты не могут обновить веса. Кроме того, валидатор упадёт на epoch 6+ из-за несоответствия размерности loss_items (9 элементов при train, 6 при val). FeatureAdapter никогда не обучался (параметры не в optimizer). TeacherHeatmapLoader может вернуть короче batch → crash в KL divergence.

**Приоритет действий:**
1. **НЕМЕДЛЕННО** — остановить v33 (crash неизбежен на epoch 6+)
2. **ПАРАЛЛЕЛЬНО** — разработать исправленную архитектуру v34
3. **ПОСЛЕ** — smoke test 10 epochs ($3), потом full run

**Главное решение:** либо unfreeze backbone (рискованно для 264K данных), либо убрать feature distillation и оставить только logit (heatmap) KD с online teacher inference вместо offline LMDB.

---

## 2. Critical Issues (ранжированы по серьёзности)

| # | Severity | Issue | Source | Impact |
|---|----------|-------|--------|--------|
| 1 | **P0 CRASH** | Validation crash: `kd_loss` возвращает 6 items при `model.training=False`, validator ожидает 9 | Agent 2 (BUG-3) | Crash на epoch 6+ |
| 2 | **P0 CRASH** | FeatureAdapter params не в optimizer — веса никогда не обновляются | Agent 2 (BUG-4) | Feature distillation = random noise |
| 3 | **P0 CRASH** | TeacherHeatmapLoader batch size mismatch при отсутствующих LMDB ключах | Agent 2 (BUG-8) | Crash в KL computation |
| 4 | **P0 DESIGN** | Feature distillation на замороженном backbone — градиенты не проходят | Agent 1, 3 | Feature KD бесполезен |
| 5 | **P1 DESIGN** | FeatureAdapter направлен наоборот (teacher→student вместо student→teacher) | Agent 1 | Адаптер не соответствует MMPose |
| 6 | **P1 DESIGN** | Offline heatmaps нарушают correspondences с augmentation | Agent 3 | DWPose использует online teacher |
| 7 | **P1 DATA** | COCO 10% = 2.1% данных — катастрофический forgetting | Agent 3 | Деградация на non-skating данных |
| 8 | **P2 PERF** | Sigma усредняется по ВСЕМ anchors (8400+) включая background | Agent 2 | Размытый сигнал |
| 9 | **P2 PERF** | Double forward pass для feature extraction (2x backbone cost) | Agent 2 | Удвоение времени обучения |
| 10 | **P2 CODE** | Dead code: `kpts_perm.view()` на строке 655 — результат отбрасывается | Agent 2 | Мусор |
| 11 | **P2 CODE** | Hardcoded `model.model[:10]` — backbone slice может быть неверен для YOLO26 | Agent 2 | Feature mismatch |
| 12 | **P2 CODE** | `restore_model` никогда не вызывается — resource leak в TeacherFeatureLoader | Agent 2 | Утечка памяти |
| 13 | **P3 CODE** | O(n) fallback key matching в TeacherFeatureLoader | Agent 2 | Медленный lookup |
| 14 | **P3 CODE** | Дублирование `keypoints_to_heatmap` с разными encoding | Agent 2 | Несогласованность |

---

## 3. Consensus Findings (все 4 агента согласны)

### 3.1. Замороженный backbone убивает feature distillation
**Согласие: 4/4** — Все агенты подтверждают: feature MSE loss не может изменить веса замороженного backbone. Это фундаментальное противоречие с подходом DWPose, где teacher и student обучаются end-to-end.

### 3.2. KL divergence корректен (не MSE)
**Согласие: 3/4** — Agent 1, 2, 3 подтверждают: DWPose использует KL divergence для logit loss. Обзор плана (plan review), утверждавший "MSE вместо KL", был **ошибочным**.

### 3.3. Offline heatmaps — слабое место
**Согласие: 3/4** — Teacher heatmaps, предгенерированные на unaugmented crops, не соответствуют augmented обучающим данным. DWPose запускает teacher online. Это различие создаёт domain gap.

### 3.4. 264K skating-only = catastrophic forgetting
**Согласие: 3/4** — Без достаточного количества COCO данных модель потеряет обобщающую способность. Текущие 5,659 COCO изображений (2.1%) недостаточны. Рекомендуется 10-20% COCO ratio.

### 3.5. Бюджет позволяет ablation study
**Согласие: 4/4** — Потрачено ~$71, осталось ~$79. Этого достаточно для smoke test + no-KD baseline + feature-only KD ablation.

---

## 4. Conflicts Between Agents

| Topic | Agent A | Agent B | Resolution |
|-------|---------|---------|------------|
| **KL vs MSE для logit loss** | Agent 3 рекомендует MSE (6,912-dim KL noisy) | Agent 1 подтверждает KL как в DWPose | **Использовать KL** — DWPose paper авторитетнее эвристики. Но нормировать на foreground pixels. |
| **Unfreeze backbone** | Agent 3: unfreeze epoch 21-210 | Agent 1: не упоминает | **Двухстадийный подход**: freeze 1-20, unfreeze 21-210 с малым LR. Согласуется с Agent 4 (Option B). |
| **Online vs offline teacher** | Agent 3: online inference | Agent 2: offline LMDB уже работает | **Компромисс**: онлайн для logit KD, оффлайн для feature KD (если оставляем). |
| **Target KD gain** | Agent 4: +0.03 до +0.10 AP | Agent 1: не оценивает | **Консервативно: +0.03 AP**. DWPose получил +1.7 AP на COCO-WholeBody, но мы fine-tune на skating-specific данных. |
| **FeatureAdapter direction** | Agent 1: teacher→student неправильно | Agent 2: не комментирует направление | **Agent 1 прав** — MMPose проецирует student→teacher. Но при frozen backbone это всё равно мёртвый код. |

---

## 5. Immediate Action Items (v33 running — WILL CRASH)

### 5.1. НЕМЕДЛЕННО: Остановить v33
v33 **гарантированно упадёт** на первой же валидации после epoch 6. Не тратить GPU бюджет.

### 5.2. Исправить 3 P0 бага перед перезапуском

**BUG-3 (Validation crash):**
```python
# Проблема: kd_loss возвращает 6 items при model.training=False
# Validator инициализирован self.loss = torch.zeros(9) из train loss_items
# 9 += 6 → shape mismatch

# Решение: kd_loss всегда возвращать одинаковое число items
# Либо: validator знать о KD mode и ожидать разное число
```

**BUG-4 (FeatureAdapter not in optimizer):**
```python
# Проблема: Adapter создаётся в первом forward pass (лениво)
# Optimizer уже построен до первого forward
# Adapter weights никогда не обновляются

# Решение: инициализировать Adapter до build_optimizer()
# Либо: rebuild optimizer после первого forward
```

**BUG-8 (Batch size mismatch):**
```python
# Проблема: если LMDB key отсутствует, возвращается короче batch
# KL(s, t) требует одинаковую размерность

# Решение: падбить или фильтровать перед KL
# Лучше: генерировать heatmaps для ВСЕХ изображений
```

### 5.3. Решение по архитектуре
Перед перезапуском: выбрать одну из двух стратегий:
- **Вариант A:** Убрать feature distillation, оставить только logit KD
- **Вариант B:** Unfreeze backbone + двухстадийное обучение

---

## 6. Architecture Redesign (v34)

### 6.1. Вариант A: Logit-only KD (рекомендуется для быстрого результата)

```
Architecture:
  Student (YOLO26n-pose, sigma head) ← online MogaNet-B teacher
  Loss: L_total = L_pose + α × KL(student_heatmap || teacher_heatmap)

Changes from v33:
  - REMOVE FeatureAdapter (мертвый код)
  - REMOVE feature distillation loss
  - REMOVE TeacherFeatureLoader (HDF5 4GB больше не нужен)
  - CHANGE TeacherHeatmapLoader → online inference
  - FIX batch size handling в heatmap loader
  - FIX kd_loss return value consistency
  - ADD COCO до 10-20% ratio
```

**Плюсы:** Простота, нет backbone issues, экономит 4GB памяти, 2x быстрее (нет double forward).
**Минусы:** Нет feature-level supervision, потенциально ниже gain.

### 6.2. Вариант B: Two-stage с unfreeze (рекомендуется для максимального gain)

```
Stage 1 (epoch 1-20): Frozen backbone
  Loss: L_total = L_pose + α × KL(student_heatmap || teacher_heatmap)
  - No feature distillation (frozen = pointless)

Stage 2 (epoch 21-210): Unfrozen backbone
  Loss: L_total = L_pose + α × KL + γ × FeatureMSE(student_proj → teacher_feat)
  - FeatureAdapter направлен student→teacher (как MMPose)
  - Adapter params В optimizer
  - Малый LR для backbone (1/10 от head LR)
  - Weight decay schedule
```

**Плюсы:** Feature distillation реально работает, ближе к DWPose paper.
**Минусы:** Риск overfitting на 264K данных, сложнее реализация, дороже.

### 6.3. Общие изменения для обоих вариантов

| Изменение | Причина | Сложность |
|-----------|---------|-----------|
| Online teacher inference | Augmentation correspondence | Средняя |
| COCO ratio 10-20% | Prevent catastrophic forgetting | Лёгкая |
| Sigma только по foreground anchors | Убрать background noise | Лёгкая |
| KL нормировка на foreground | Убрать background dominance | Лёгкая |
| Удалить dead code | Чистота | Тривиальная |
| Fix backbone slice для YOLO26 | Корректный feature extraction | Лёгкая |

---

## 7. Evaluation Framework

### 7.1. Metrics

| Metric | Target | Source |
|--------|--------|--------|
| mAP50-95 на FineFS val | ≥ 0.55 (min), ≥ 0.65 (good) | Agent 4 |
| mAP50-95 на AP3D val | ≥ 0.96 (teacher baseline) | Agent 4 |
| KD gain (Δ AP vs no-KD) | +0.03 to +0.10 | Agent 4 |
| Training stability | Нет NaN/Inf loss | Agent 2 |
| Validation consistency | Нет crash | Agent 2 |

### 7.2. Evaluation frequency
- Каждый epoch: training loss, val loss (без crash!)
- Каждые 10 epochs: mAP50-95 на FineFS val subset (5K images)
- После обучения: полный mAP50-95 на FineFS val (58K) + AP3D val (21K)

### 7.3. Required scripts

```bash
# 1. Smoke test (каждый вариант)
yolo train model=yolo26n-pose.pt data=skating_full.yaml epochs=10 batch=32

# 2. No-KD baseline
yolo train model=yolo26n-pose.pt data=skating_full.yaml epochs=50 batch=32

# 3. KD training (v34)
python distill_trainer.py --config stage3_distill.yaml

# 4. Evaluation
yolo val model=runs/pose/train/weights/best.pt data=skating_full.yaml

# 5. Comparison table
python compare_results.py --baseline runs/baseline --kd runs/kd_v34
```

---

## 8. Risk Register

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | Frozen backbone слишком ограничительный | 60% | Средний | Вариант B: unfreeze на epoch 21 |
| R2 | Corrupted teacher heatmaps (sigmoid vs clamp) | 30% | **КРИТИЧЕСКИЙ** | Верифицировать peaks ~0.7 перед обучением |
| R3 | Top-down vs bottom-up domain gap | 50% | Средний | Смешанный data augmentation |
| R4 | Catastrophic forgetting (мало COCO) | 70% | Высокий | Увеличить COCO до 20% |
| R5 | Online teacher inference слишком медленный | 40% | Средний | Batch teacher inference, кеширование |
| R6 | Sigma head случайный в начале → noisy KL | 80% | Низкий | Warmup: α=0 первые 5 epochs |
| R7 | 264K недостаточно для unfreezing | 50% | Высокий | Dropout, weight decay, early stopping |
| R8 | FineFS pseudo-labels низкого качества | 40% | Средний | Cross-validate с AP3D |
| R9 | Budget exceeded | 20% | Средний | Smoke test перед full run |
| R10 | YOLO26 architecture changes (backbone slice) | 30% | Средний | Верифицировать `model.model[:10]` |

---

## 9. Budget & Ablation Plan

### 9.1. Текущий бюджет

| Item | Spent | Remaining |
|------|-------|-----------|
| Total budget | ~$71 | ~$79 |

### 9.2. Ablation Plan (Option B, $33 total)

| Ablation | Cost | Duration | Priority | Что тестирует |
|----------|------|----------|----------|---------------|
| **A1: Smoke test 10 epochs** | $3 | ~2h | **КРИТИЧЕСКАЯ** | Нет crash, loss decreasing |
| **A2: No-KD baseline 50 epochs** | $15 | ~12h | **ВЫСОКАЯ** | Upper bound без KD |
| **A3: Feature-only KD 50 epochs** | $15 | ~18h | **СРЕДНЯЯ** | Ценность feature distillation |

### 9.3. Full training budget (после ablation)

| Scenario | Cost | Duration | Условие |
|----------|------|----------|---------|
| Вариант A (logit-only KD, 210 epochs) | ~$46 | ~36h | A1+A2 pass |
| Вариант B (two-stage, 210 epochs) | ~$56 | ~44h | A1+A2 pass, A3 показывает gain |
| Direct fine-tuning fallback | ~$15 | ~12h | KD не работает |

### 9.4. Бюджетная хронология

```
Phase 1: Smoke + Ablation ($33)
  ├── A1 smoke test ($3, 2h)
  ├── A2 no-KD baseline ($15, 12h)
  └── A3 feature-only KD ($15, 18h) ← только если A2 pass

Phase 2: Full training ($46-56)
  └── Выбранный вариант 210 epochs

Total: $79-89 (в пределах бюджета)
```

---

## 10. Fallback Strategies

| # | Strategy | Cost | Expected Gain | When to trigger |
|---|----------|------|---------------|-----------------|
| B1 | Direct fine-tuning без KD | $15 | baseline (0.517 AP) | KD не даёт gain после A1-A3 |
| B2 | Same-family distillation YOLO26l→n | $30 | +0.02-0.05 AP | Cross-family gap слишком большой |
| B3 | Self-distillation SDPose-style | $20 | +0.01-0.03 AP | Feature distillation бесполезна |
| B4 | Pseudo-labeling из SkatingVerse | $0 | +0.05-0.15 AP (potential) | Нужны больше данных, а не KD |

**Рекомендация:** Начать с B1 как absolute fallback. B4 (pseudo-labeling) — самый перспективный путь к реальному улучшению, т.к. 28K реальных видео с соревнований >> teacher distillation.

---

## 11. Competitive Landscape (SOTA 2025-2026)

| Method | AP (COCO) | Type | Relevance |
|--------|-----------|------|-----------|
| DWPose (RTMDet-B + RTMPose-B) | 66.5 | Teacher-Student | Наша target архитектура |
| YOLO26n-pose | 57.2 | Bottom-up | Наш student |
| MogaNet-B (teacher) | ~68 (pose) | Top-down | Наш teacher |
| SDPose (self-distillation) | ~64 | Self-KD | Fallback B3 |
| Pose3DM (3D lifter) | 37.9mm MPJPE | 3D | Monitoring |
| DeepGlint (SV 1st place) | 95.73% | Detection | Reference |

**Ключевой insight от Agent 4:** DWPose teacher→student gain на COCO-WholeBody: +1.7 AP (64.8→66.5). Мы обучаемся на skating-specific данных, где gain может быть ниже из-за domain gap. Реалистичная цель: +0.03 AP.

---

## 12. Appendix: Per-Agent Key Findings

### Agent 1: KD Theory & Method Expert

**3 HIGH отклонения от DWPose paper:**
1. Frozen backbone убивает feature distillation — DWPose обучает end-to-end
2. SimCC vs Heatmap mismatch — DWPose distills 1D SimCC, мы синтезируем 2D Gaussians
3. Adapter direction reversed — MMPose проецирует student→teacher

**Позитивное:** KL divergence корректен (plan review был ошибочным), alpha=0.00005 близко к MMPose (0.00007), student heatmap использует GT keypoints.

### Agent 2: Architecture & Code Review Expert

**3 CRITICAL бага:**
1. Validation crash на epoch 6+ (loss items shape mismatch)
2. FeatureAdapter params не в optimizer (feature KD = noise)
3. TeacherHeatmapLoader batch size mismatch (crash в KL)

**Дополнительное:** Dead code, sigma по всем anchors (8400+), double forward pass, hardcoded backbone slice, resource leak, дублированный heatmap encoding.

### Agent 3: Data Pipeline & Loss Function Expert

**P0 findings:**
1. Feature distillation на frozen backbone — бесполезна
2. Нет COCO данных = catastrophic forgetting
3. KL на 6,912-dim noisy (background dominance)
4. Offline heatmaps нарушают augmentation correspondence

**Рекомендации:** Online teacher inference, двухстадийный unfreeze (1-20 frozen, 21-210 unfrozen), MSE вместо KL (отклонено — см. раздел 4).

### Agent 4: Evaluation & Strategy Expert

**Ключевые числа:**
- DWPose gain: +1.7 AP
- YOLO26n baseline: 0.572 AP
- MogaNet-B teacher: 0.962 AP (AP3D)
- Target KD gain: +0.03 to +0.10 AP

**Ablation plan:** Smoke test ($3) → No-KD baseline ($15) → Feature-only KD ($15). Budget: $33 ablation + $46-56 full training = $79-89 total.

**Risk register:** R1 frozen backbone (60%), R2 corrupted heatmaps (30%, CRITICAL), R3 domain gap (50%). Fallback: direct fine-tuning ($15), same-family KD ($30), pseudo-labeling ($0).

---

## Заключение

Текущая реализация KD (v33) **неготова к обучению**. 3 P0 бага вызовут crash. Фундаментальная проблема — feature distillation на frozen backbone — делает половину KD pipeline бесполезной.

**Рекомендуемый путь:**
1. Остановить v33
2. Реализовать Вариант A (logit-only KD) с online teacher
3. Smoke test A1 ($3)
4. Если A1 pass → full training ($46)
5. Если A1 fail → fallback B1 (direct fine-tuning, $15)

Это минимизирует риски и бюджет при максимальной информативности.
