# WeakAL+AutoWS: Hybrid Active Learning + Weak Supervision for Text Classification

**ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА**

**Автор:** Мигурский Иван Феликсович
**Научный руководитель:** Пивоваров Дмитрий Евгеньевич

---

## О проекте

Исследование гибридного подхода, объединяющего **активное обучение (AL)** и **weak supervision (WS)** для классификации текстовых обращений пользователей. Основная идея: слабые эвристики дают массу данных, а человек размечает только те примеры, где модель сомневается больше всего.

**Гипотеза:** гибрид AL+WS достигает точности, сопоставимой с чистым активным обучением, при меньшем числе ручных меток — но только если точность слабых правил (WS-acc) ≥ 85%.

---

## Архитектура

```
Сырые тексты
    │
    ▼
┌─────────────────────────┐
│  Weak Supervision блок   │
│  6 Labeling Functions →  │
│  Dawid-Skene агрегация   │
│  → WS-метки              │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Классификатор (RF/LR)  │
│  Обучен на WS + человек  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Active Learning блок    │
│  Выбор неуверенных       │
│  примеров → оракул       │
└───────────┬─────────────┘
            │
            └────► цикл повторяется
```

Пайплайн **модульный**: любую стратегию AL, агрегатор WS или технику улучшения можно заменить без изменения остальных компонентов.

---

## Структура проекта

```
weakal_pipeline/
├── __init__.py                  # Экспорт ключевых классов
├── __main__.py                  # CLI: python -m weakal_pipeline
├── config.py                    # PipelineConfig + ExperimentConfig
├── data.py                      # Загрузка датасетов, TF-IDF, сплит
├── pyproject.toml               # Метаданные проекта
│
├── active_learning/
│   └── __init__.py              # ActiveLearner, 6 стратегий запросов
│
├── weak_supervision/
│   ├── __init__.py              # WeakSupervisor, 6 LF, Dawid-Skene, WeakCertainty
│   └── enhanced_ws.py           # 13 классов улучшения WS (T1–T14)
│
├── pipeline/
│   ├── __init__.py              # HybridPipeline, ALOnlyPipeline, WSOnlyPipeline
│   └── enhanced_hybrid.py       # 16 расширенных пайплайнов (T1–T14, Combo, Flood)
│
├── experiments/
│   ├── __init__.py              # run_experiment(), run_comparison()
│   └── ws_comparison.py         # Сравнение техник WS (отдельный CLI)
│
└── visualization/
    └── __init__.py              # Графики matplotlib + LaTeX-таблицы
```

---

## Ключевые компоненты

### Активное обучение — 6 стратегий

| Стратегия | Описание |
|---|---|
| `RANDOM` | Случайный выбор (базлайн) |
| `UNCERTAINTY_LEAST_CONFIDENT` | Наименьшая уверенность |
| `UNCERTAINTY_MARGIN` | Минимальная разница между топ-2 классами |
| `UNCERTAINTY_ENTROPY` | Максимальная энтропия |
| `BADGE` | K-means++ на градиентных эмбеддингах (разнообразие + неуверенность) |
| `COST_SENSITIVE` | Взвешивание по обратной частоте класса |

### Weak Supervision — 6 Labeling Functions

| LF | Метод |
|---|---|
| `NaiveBayesLF` | Наивный Байес на TF-IDF |
| `SVMLF` | SVM с линейным ядром |
| `RandomForestLF` | Случайный лес |
| `KNNLF` | k-ближайших соседей |
| `LogisticRegressionLF` | Логистическая регрессия |
| `KeywordLF` | Совпадение с TF-IDF-ключевыми словами |

Агрегация голосов: **Dawid-Skene** (EM-алгоритм) или **мажоритарное голосование**.

### 14 техник улучшения WS (T1–T14)

| Группа | Техники | Суть |
|---|---|---|
| Калибровка | T3 (Platt), T7 (Isotonic) | Исправление вероятностей LF |
| Фильтрация | T5 (Unanimous), T6 (Entropy) | Отбрасывание ненадёжных меток |
| Устойчивое обучение | T1 (Weighted), T2 (Verification) | Понижение веса / проверка WS-меток |
| Расширение | T4 (Pseudo-Label), T14 (Self-Training) | Авторазметка уверенных примеров |
| Альт. стратегии | T9 (BADGE), T11 (Cost-Sensitive), T12 (Adaptive) | Улучшенный выбор примеров |
| Структурная | T10 (Curriculum / Label Propagation) | Графовое распространение меток |
| Альт. LF | T8 (BERT LF) | SentenceTransformer как 7-я функция |
| Альт. агрегация | T13 (FlyingSquid) | Триплетная агрегация вместо Dawid-Skene |

### Пайплайны

| Пайплайн | Описание |
|---|---|
| `ALOnlyPipeline` | Чистое активное обучение (базлайн) |
| `WSOnlyPipeline` | Чистый weak supervision (без AL-цикла) |
| `HybridPipeline` | AL + WS + WeakCertainty (основной метод) |
| `T1–T14 Pipeline` | Гибрид + конкретная техника улучшения |
| `ComboPipeline` | T1+T2+T3+T6 комбинированно |
| `WSAdaptiveFloodPipeline` | Адаптивный Flood-режим (увеличение потока WS-меток) |

---

## Датасеты

Проект поддерживает 13 датасетов из HuggingFace. Ключевые для исследования:

| Датасет | Классы | Примеры | Домен |
|---|---|---|---|
| `customer_tickets` | 4 | ~4 900 | IT-обращения |
| `bitext_ecommerce` | 13 | ~10 000 | E-commerce интенты |
| `rakuten_amazon` | 15 | ~12 000 | Товары / отзывы |
| `cfpb_complaints` | 41 | ~8 000 | Финансовые жалобы |
| `hp_tickets` | 27 | ~3 500 | IT-поддержка |

Дополнительные: `banking77` (77 классов), `clinc150` (150 классов), `bitext_banking`, `bitext_insurance`, `bitext_mortgage`, `bitext_wealth`, `bitext_travel`, `bitext_customer_support`.

Все датасеты загружаются автоматически с HuggingFace и кэшируются в `.cache/data/`.

---

## Установка

```bash
# Клонирование
git clone https://github.com/<username>/weakal_pipeline.git
cd weakal_pipeline

# Установка зависимостей
pip install numpy pandas scikit-learn scipy datasets matplotlib

# Опционально — для T8 (BERT LF) и T13 (FlyingSquid)
pip install sentence-transformers
```

Требуется **Python ≥ 3.10**.

---

## Использование

### Быстрый старт

```bash
# Запуск на одном датасете
python -m weakal_pipeline --dataset customer_tickets

# Быстрый тест (500 примеров)
python -m weakal_pipeline --dataset customer_tickets --quick

# Только гибридный режим
python -m weakal_pipeline --dataset customer_tickets --mode hybrid

# Все 5 ключевых датасетов
python -m weakal_pipeline --key-datasets
```

### Сравнение техник WS

```bash
python -m weakal_pipeline.run_ws_comparison --dataset customer_tickets
python -m weakal_pipeline.run_ws_comparison --dataset customer_tickets --quick
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--dataset` | `customer_tickets` | Название датасета |
| `--mode` | `all` | Режим: `baseline`, `al_only`, `ws_only`, `hybrid`, `random_labels`, `all` |
| `--budget` | `300` | Макс. число ручных меток |
| `--repeats` | `5` | Число повторов с разными seed |
| `--quick` | `False` | Укороченный запуск (500 примеров) |
| `--classifier` | `rf` | Классификатор: `rf`, `lr`, `svm` |
| `--all-datasets` | — | Запуск на всех 13 датасетах |
| `--key-datasets` | — | Запуск на 5 ключевых датасетах |

### Программный запуск

```python
from weakal_pipeline import PipelineConfig, ExperimentConfig, run_experiment

config = PipelineConfig(
    dataset_name="customer_tickets",
    query_strategy="uncertainty_entropy",
    max_human_labels=200,
    batch_size=10,
)

result = run_experiment(ExperimentConfig(
    name="my_experiment",
    mode="hybrid",
    config=config,
    n_repeats=5,
))

print(result.summary())
```

---

## Результаты

### Основные выводы

1. **Гибрид AL+WS** достигает точности, близкой к чистому AL, только при WS-acc ≥ 85%. При WS-acc < 85% слабые метки вредят.

2. **Isotonic-калибровка (T7)** — лучшая техника улучшения: +5% accuracy и снижение разброса в 5 раз по сравнению с Platt-калибровкой (T3).

3. **Фильтрация единогласием (T5)** — высокая точность, но низкое покрытие (многие примеры отбрасываются).

4. **Flood-режим** увеличивает поток WS-меток в 3–5×. Работает без потери качества только при WS-acc > 95% на лёгких датасетах (hp_tickets: 97.8% → 97.8%). На остальных — падение 2–7%.

5. **Практическое правило**: сначала оцените WS-acc на проверочной выборке. Если ≥ 90% — используйте гибрид. Если < 85% — только AL.

### Сводная таблица (customer_tickets, 5-seed)

| Пайплайн | Accuracy | F1 Macro | WS-acc | Экономия меток |
|---|---|---|---|---|
| AL Only | 0.752 ± 0.016 | 0.645 ± 0.048 | — | 0% |
| T7 Isotonic | 0.722 ± 0.012 | 0.578 ± 0.030 | 0.739 | 24.5% |
| T11 Cost-Sensitive | 0.716 ± 0.021 | 0.579 ± 0.042 | 0.709 | 30.6% |
| T5 Unanimous | 0.712 ± 0.021 | 0.571 ± 0.036 | 0.736 | 24.3% |
| Original Hybrid | 0.676 ± 0.063 | 0.538 ± 0.078 | 0.689 | 18.4% |

### Кросс-датасетный анализ

| Датасет | Сложность | AL-only | Hybrid | Вердикт |
|---|---|---|---|---|
| hp_tickets | Лёгкий | 97.8% | 97.8% | WS помогает |
| rakuten_amazon | Средний | 98.3% | 96.9% | WS почти не хуже |
| customer_tickets | Средний | 75.0% | 75.0% | WS нейтрален |
| cfpb_complaints | Сложный | 40.1% | 23.9% | WS вреден |

---

## Конфигурация

Вся конфигурация через `PipelineConfig` в `config.py`. Ключевые параметры:

```python
PipelineConfig(
    # Данные
    dataset_name="customer_tickets",
    max_samples=None,            # Без ограничения
    test_size=0.2,

    # TF-IDF
    max_features=5000,
    ngram_range=(1, 2),

    # Active Learning
    query_strategy="uncertainty_least_confident",
    batch_size=10,
    initial_per_class=2,
    max_human_labels=300,

    # Weak Supervision
    label_model="dawid_skene",
    lf_confidence_threshold=0.7,

    # Hybrid
    ws_accuracy_threshold=0.6,
    ws_confidence_filter=0.8,
    weak_cert_alpha=0.9,

    # Классификатор
    classifier_type="rf",        # rf | lr | svm
)
```

---

## Визуализация

Модуль `visualization` автоматически строит:

- Кривые Accuracy / F1 от числа ручных меток
- Сравнительные столбчатые диаграммы пайплайнов
- Стековые графики вклада WS-меток
- LaTeX-таблицы для статей

```python
from weakal_pipeline.visualization import generate_all_plots

generate_all_plots(results, output_dir="./plots")
```

---

## Зависимости

| Пакет | Версия | Назначение |
|---|---|---|
| `numpy` | ≥1.24 | Массивы, линейная алгебра |
| `pandas` | ≥2.0 | Работа с датасетами |
| `scikit-learn` | ≥1.3 | Классификаторы, метрики, TF-IDF |
| `scipy` | ≥1.11 | Разреженные матрицы |
| `datasets` | ≥2.14 | Загрузка HuggingFace |
| `matplotlib` | ≥3.7 | Графики |
| `sentence-transformers` | ≥2.2 | T8: BERT LF (опционально) |
