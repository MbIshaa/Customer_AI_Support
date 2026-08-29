import json

import pandas as pd
import plotly.express as px
import streamlit as st
from ollama import chat


# ----------------------------------------------------------
# НАСТРОЙКИ СТРАНИЦЫ
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Customer Insights",
    page_icon="📊",
    layout="wide",
)


# ----------------------------------------------------------
# ЗАГРУЗКА ДАННЫХ
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/enriched_support_requests.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


df = load_data()


# ----------------------------------------------------------
# РУССКИЕ НАЗВАНИЯ
# ----------------------------------------------------------

COLUMN_LABELS = {
    "date": "Дата",
    "category": "Категория",
    "theme": "Тема",
    "text": "Текст обращения",
    "summary": "Краткая проблема",
    "severity": "Серьёзность",
    "sentiment": "Тональность",
    "churn_risk": "Риск ухода",
    "recommendation": "Рекомендация",
}

SENTIMENT_LABELS = {
    "negative": "Негативная",
    "neutral": "Нейтральная",
    "positive": "Позитивная",
}

CHURN_LABELS = {
    "low": "Низкий",
    "medium": "Средний",
    "high": "Высокий",
}

CHURN_VALUES_BY_LABEL = {
    value: key
    for key, value in CHURN_LABELS.items()
}

CATEGORIES = sorted(df["category"].dropna().unique())


# ----------------------------------------------------------
# ДАТЫ
# ----------------------------------------------------------

MIN_DATE = df["date"].min().date()
MAX_DATE = df["date"].max().date()


def apply_date_range(data, selected_period):
    if not selected_period:
        return data

    if isinstance(selected_period, (list, tuple)):
        if len(selected_period) == 2:
            start_date, end_date = selected_period
        elif len(selected_period) == 1:
            start_date = end_date = selected_period[0]
        else:
            return data
    else:
        start_date = end_date = selected_period

    start_ts = pd.Timestamp(start_date)
    end_ts = (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    return data[
        (data["date"] >= start_ts)
        & (data["date"] <= end_ts)
    ]


def render_context_summary(data):
    """
    Короткая сводка по текущей выборке.
    Автоматически меняется вместе с фильтрами.
    """
    if data.empty:
        return

    start_date = data["date"].min().strftime("%d.%m.%Y")
    end_date = data["date"].max().strftime("%d.%m.%Y")
    category_count = data["category"].nunique()

    st.markdown(
        (
            f"**{len(data):,} обращений** · "
            f"{start_date} — {end_date} · "
            f"**{category_count} категорий**"
        )
    )


def render_tab_context(key_prefix):
    """Отдельные фильтры для конкретной вкладки."""
    section_header("Контекст анализа")

    with st.container(border=True):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(
            [1.45, 1.0, 1.0, 1.45]
        )

        with filter_col1:
            selected_categories = st.multiselect(
                "Категории",
                CATEGORIES,
                default=CATEGORIES,
                key=f"{key_prefix}_categories",
            )

        with filter_col2:
            selected_severity = st.multiselect(
                "Серьёзность",
                [1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5],
                key=f"{key_prefix}_severity",
            )

        with filter_col3:
            churn_labels = [
                "Низкий",
                "Средний",
                "Высокий",
            ]

            selected_churn_labels = st.multiselect(
                "Риск ухода",
                churn_labels,
                default=churn_labels,
                key=f"{key_prefix}_churn",
            )

            selected_churn = [
                CHURN_VALUES_BY_LABEL[label]
                for label in selected_churn_labels
            ]

        with filter_col4:
            selected_period = st.date_input(
                "Период",
                value=(MIN_DATE, MAX_DATE),
                min_value=MIN_DATE,
                max_value=MAX_DATE,
                format="DD.MM.YYYY",
                key=f"{key_prefix}_period",
            )

    result = df[
        (df["category"].isin(selected_categories))
        & (df["severity"].isin(selected_severity))
        & (df["churn_risk"].isin(selected_churn))
    ].copy()

    result = apply_date_range(
        result,
        selected_period,
    )

    return result


# ----------------------------------------------------------
# ВИЗУАЛЬНЫЙ СТИЛЬ
# ----------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1480px;
            padding-top: 4.5rem;
            padding-bottom: 3rem;
        }

        .app-hero {
            padding: 1.35rem 1.55rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 18px;
            background:
                linear-gradient(
                    135deg,
                    rgba(90, 110, 255, 0.10),
                    rgba(90, 110, 255, 0.02)
                );
            margin-bottom: 1rem;
        }

        .app-eyebrow {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.65;
            margin-bottom: 0.35rem;
        }

        .app-title {
            font-size: 2.05rem;
            line-height: 1.15;
            font-weight: 760;
            margin: 0;
        }

        .app-subtitle {
            font-size: 1rem;
            opacity: 0.72;
            margin-top: 0.5rem;
            margin-bottom: 0.9rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .app-chip {
            display: inline-block;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(128, 128, 128, 0.08);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .journey {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            margin-bottom: 1rem;
        }

        .journey-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.60;
            margin-bottom: 0.45rem;
        }

        .journey-flow {
            font-size: 1.02rem;
            font-weight: 700;
        }

        .section-heading {
            margin-top: 0.25rem;
            margin-bottom: 0.2rem;
            font-size: 1.25rem;
            font-weight: 720;
        }

        .section-caption {
            opacity: 0.66;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        .muted-note {
            opacity: 0.68;
            font-size: 0.86rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(128, 128, 128, 0.035);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
            border-color: rgba(128, 128, 128, 0.20);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.65rem;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 3.8rem;
            padding-left: 1.45rem;
            padding-right: 1.45rem;
            border-radius: 12px 12px 0 0;
            font-weight: 750;
        }

        .stTabs [data-baseweb="tab"] p {
            font-size: 1.22rem !important;
            line-height: 1.2 !important;
            font-weight: 750 !important;
        }

        .stTabs [aria-selected="true"] {
            border-bottom-width: 3px !important;
        }

        .ai-plan-shell {
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(128, 128, 128, 0.20);
            background:
                linear-gradient(
                    135deg,
                    rgba(90, 110, 255, 0.08),
                    rgba(128, 128, 128, 0.02)
                );
            margin-bottom: 1rem;
        }

        .ai-plan-title {
            font-size: 1.12rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .ai-plan-subtitle {
            font-size: 0.92rem;
            opacity: 0.72;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 3.25rem;
            border-radius: 11px;
            font-size: 1.02rem;
            font-weight: 720;
            padding: 0.75rem 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def polish_figure(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=20, b=12),
        hoverlabel=dict(namelength=-1),
    )
    return fig


# ----------------------------------------------------------
# ПРИОРИТЕТЫ
# ----------------------------------------------------------

def build_priority_table(data, limit=None):
    table = (
        data.groupby(["category", "summary"])
        .agg(
            requests=("id", "count"),
            avg_severity=("severity", "mean"),
            critical=("severity", lambda x: (x == 5).sum()),
            high_churn=("churn_risk", lambda x: (x == "high").sum()),
        )
        .reset_index()
    )

    table["priority_score"] = (
        table["requests"]
        + table["avg_severity"] * 5
        + table["critical"] * 3
        + table["high_churn"] * 3
    )

    table = table.sort_values(
        "priority_score",
        ascending=False,
    )

    table["avg_severity"] = table["avg_severity"].round(2)
    table["priority_score"] = table["priority_score"].round(1)

    if limit is not None:
        return table.head(limit)

    return table


# ----------------------------------------------------------
# AI-РЕКОМЕНДАЦИИ
# ----------------------------------------------------------

def get_top_values(data, column, limit=3):
    if column not in data.columns:
        return []

    values = (
        data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        ~values.str.lower().isin(
            [
                "",
                "nan",
                "none",
                "null",
                "unknown",
                "неизвестно",
            ]
        )
    ]

    return (
        values
        .value_counts()
        .head(limit)
        .index
        .tolist()
    )


def get_representative_ticket_examples(
    data,
    category,
    summary,
    limit=3,
):
    """
    Даёт Qwen несколько конкретных примеров обращений,
    чтобы рекомендации опирались не только на агрегаты.
    """
    subset = data[
        (data["category"] == category)
        & (data["summary"] == summary)
    ].copy()

    if subset.empty:
        return []

    subset = subset.sort_values(
        ["severity", "date"],
        ascending=[False, False],
    )

    examples = []
    seen = set()

    for value in subset["text"].dropna():
        value = " ".join(str(value).split())

        if not value:
            continue

        normalized = value.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        if len(value) > 220:
            value = value[:217].rstrip() + "..."

        examples.append(value)

        if len(examples) >= limit:
            break

    return examples


def get_top_existing_recommendations(
    data,
    category,
    summary,
    limit=2,
):
    subset = data[
        (data["category"] == category)
        & (data["summary"] == summary)
    ]

    return get_top_values(
        subset,
        "recommendation",
        limit=limit,
    )


def build_recommendation_prompt(
    data,
    priority_table,
    limit=3,
):
    rows = []

    for position, (_, row) in enumerate(
        priority_table.head(limit).iterrows(),
        start=1,
    ):
        subset = data[
            (data["category"] == row["category"])
            & (data["summary"] == row["summary"])
        ]

        existing = get_top_existing_recommendations(
            data,
            row["category"],
            row["summary"],
            limit=3,
        )

        themes = get_top_values(
            subset,
            "theme",
            limit=3,
        )

        examples = get_representative_ticket_examples(
            data,
            row["category"],
            row["summary"],
            limit=3,
        )

        examples_text = (
            " | ".join(
                f"«{example}»"
                for example in examples
            )
            if examples
            else "нет примеров"
        )

        rows.append(
            (
                f"ПРОБЛЕМА {position}\n"
                f"Категория: {row['category']}\n"
                f"Проблема: {row['summary']}\n"
                f"Обращений: {int(row['requests'])}\n"
                f"Средняя серьёзность: "
                f"{float(row['avg_severity']):.2f}\n"
                f"Критических: {int(row['critical'])}\n"
                f"Высокий риск ухода: "
                f"{int(row['high_churn'])}\n"
                f"Частые темы: "
                f"{'; '.join(themes) if themes else 'нет данных'}\n"
                f"Примеры обращений: {examples_text}\n"
                f"Рекомендации из обращений: "
                f"{'; '.join(existing) if existing else 'нет'}"
            )
        )

    return "\n\n".join(rows)


def parse_management_recommendations(raw_answer, count=3):
    raw_answer = (raw_answer or "").strip()

    if not raw_answer:
        return {}

    raw_answer = (
        raw_answer
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        payload = json.loads(raw_answer)
    except json.JSONDecodeError:
        return {}

    items = payload.get("recommendations", [])
    result = {}

    banned_generic = {
        "проверить систему",
        "исправить проблему",
        "улучшить качество",
        "проверить качество",
        "проверить доставку",
        "разобраться в проблеме",
    }

    def normalize_action(value):
        return " ".join(
            str(value)
            .strip()
            .lower()
            .replace("ё", "е")
            .split()
        )

    def is_specific(action):
        normalized = normalize_action(action)

        if not normalized:
            return False

        if normalized in banned_generic:
            return False

        if len(normalized.split()) < 5:
            return False

        return True

    def too_similar(first, second):
        words_1 = set(normalize_action(first).split())
        words_2 = set(normalize_action(second).split())

        if not words_1 or not words_2:
            return True

        overlap = len(words_1 & words_2)
        union = len(words_1 | words_2)

        return (overlap / union) >= 0.55

    for item in items:
        try:
            number = int(item.get("id"))
        except (TypeError, ValueError):
            continue

        immediate = str(
            item.get("immediate_action", "")
        ).strip()

        process = str(
            item.get("process_action", "")
        ).strip()

        control = str(
            item.get("control_action", "")
        ).strip()

        if not (1 <= number <= count):
            continue

        if not (
            is_specific(immediate)
            and is_specific(process)
            and is_specific(control)
        ):
            continue

        if (
            too_similar(immediate, process)
            or too_similar(immediate, control)
            or too_similar(process, control)
        ):
            continue

        forbidden_fragments = (
            "выручк",
            "прибыл",
            "маржин",
            "конверси",
            "бюджет",
            "sla",
            "средний чек",
        )

        combined = " ".join(
            [immediate, process, control]
        ).lower()

        if any(
            fragment in combined
            for fragment in forbidden_fragments
        ):
            continue

        result[number] = {
            "immediate_action": immediate,
            "process_action": process,
            "control_action": control,
        }

    return result


def fallback_actions(
    data,
    category,
    summary,
):
    existing = get_top_existing_recommendations(
        data,
        category,
        summary,
        limit=2,
    )

    if len(existing) >= 2:
        return {
            "immediate_action": existing[0],
            "process_action": existing[1],
            "control_action": (
                "Сравнить число обращений, критических случаев "
                "и высокий риск ухода по этой проблеме после изменения."
            ),
        }

    if len(existing) == 1:
        return {
            "immediate_action": existing[0],
            "process_action": (
                "Проанализировать повторяющиеся обращения этой группы "
                "и закрепить изменение процесса после проверки сценария."
            ),
            "control_action": (
                "Сравнить число обращений, критических случаев "
                "и высокий риск ухода по этой проблеме после изменения."
            ),
        }

    return {
        "immediate_action": (
            "Разобрать последние обращения этой группы "
            "и выделить повторяющийся сценарий проблемы."
        ),
        "process_action": (
            "После проверки сценария изменить соответствующий процесс "
            "и отслеживать повторные обращения."
        ),
        "control_action": (
            "Сравнить число обращений, критических случаев "
            "и высокий риск ухода по этой проблеме после изменения."
        ),
    }


def render_business_recommendations(
    data,
    priority_table,
    actions_by_number,
):
    for position, (_, row) in enumerate(
        priority_table.head(3).iterrows(),
        start=1,
    ):
        actions = actions_by_number.get(position)

        if not actions:
            actions = fallback_actions(
                data,
                row["category"],
                row["summary"],
            )

        with st.container(border=True):
            st.caption(
                f"ПЛАН ДЕЙСТВИЙ · ПРИОРИТЕТ #{position}"
            )

            st.markdown(
                f"#### {row['category']} — {row['summary']}"
            )

            st.markdown(
                "**Почему важно**  \n"
                f"{int(row['requests'])} обращений · "
                f"средняя серьёзность "
                f"{float(row['avg_severity']):.2f} · "
                f"{int(row['critical'])} критических · "
                f"{int(row['high_churn'])} с высоким риском ухода"
            )

            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:
                st.markdown("**1. Что сделать сейчас**")
                st.write(actions["immediate_action"])

            with action_col2:
                st.markdown("**2. Что изменить в процессе**")
                st.write(actions["process_action"])

            with action_col3:
                st.markdown("**3. Как проверить результат**")
                st.write(actions["control_action"])

            st.markdown(
                "**Базовая точка для контроля:** "
                f"{int(row['requests'])} обращений · "
                f"{int(row['critical'])} критических · "
                f"{int(row['high_churn'])} с высоким риском ухода."
            )


# ----------------------------------------------------------
# AI-АНАЛИТИК: ПОДГОТОВКА КОНТЕКСТА
# ----------------------------------------------------------

MONTH_STEMS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "май": 5,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}

CATEGORY_ALIASES = {
    "Доставка": ["доставк"],
    "Оплата": ["оплат"],
    "Поддержка": ["поддержк"],
    "Возвраты": ["возврат"],
    "Качество товара": ["качеств товара", "качество"],
    "Мобильное приложение": ["мобильн", "приложен"],
    "Личный кабинет": ["личн кабинет", "кабинет"],
    "Бонусная программа": ["бонус"],
    "Цена": ["цен", "скидк", "промокод"],
    "Документы": ["документ", "чек", "счет", "счёт"],
}

UNSUPPORTED_METRICS = {
    "выручк": "выручке",
    "доход": "доходах",
    "прибыл": "прибыли",
    "конверси": "конверсии",
    "марж": "маржинальности",
    "оборот": "обороте",
    "себестоим": "себестоимости",
    "средний чек": "среднем чеке",
    "среднего чека": "среднем чеке",
    "количество заказов": "количестве заказов",
    "число заказов": "количестве заказов",
    "уникальных клиентов": "количестве уникальных клиентов",
    "число клиентов": "количестве уникальных клиентов",
}


def find_selected_month(data, question_lower):
    month_number = None

    for stem, number in MONTH_STEMS.items():
        if stem in question_lower:
            month_number = number
            break

    if month_number is None:
        return None

    available_months = sorted(
        data["month"].dropna().unique()
    )

    candidates = [
        month
        for month in available_months
        if month.endswith(f"-{month_number:02d}")
    ]

    return candidates[-1] if candidates else None


def find_selected_category(question_lower):
    for category, aliases in CATEGORY_ALIASES.items():
        if any(
            alias in question_lower
            for alias in aliases
        ):
            return category

    return None


def find_unsupported_metric(question):
    question_lower = question.lower()

    for keyword, metric_name in UNSUPPORTED_METRICS.items():
        if keyword in question_lower:
            return metric_name

    return None


def build_question_context(data, question):
    question_lower = question.lower()
    selected_month = find_selected_month(
        data,
        question_lower,
    )
    selected_category = find_selected_category(
        question_lower
    )

    if selected_month and selected_category:
        category_data = data[
            data["category"] == selected_category
        ]

        category_monthly = (
            category_data.groupby("month")
            .size()
            .reindex(
                sorted(data["month"].unique()),
                fill_value=0,
            )
        )

        subset = data[
            (data["month"] == selected_month)
            & (data["category"] == selected_category)
        ]

        top_problems = (
            subset["summary"]
            .value_counts()
            .head(6)
        )

        critical_count = int(
            (subset["severity"] == 5).sum()
        )
        high_churn_count = int(
            (subset["churn_risk"] == "high").sum()
        )
        avg_severity = (
            float(subset["severity"].mean())
            if len(subset)
            else 0.0
        )

        months = list(category_monthly.index)
        current_value = int(
            category_monthly.get(
                selected_month,
                0,
            )
        )
        yearly_average = float(
            category_monthly.mean()
        )

        context = f"""
ВОПРОС О КАТЕГОРИИ: {selected_category}
ВЫБРАННЫЙ МЕСЯЦ: {selected_month}

Динамика обращений этой категории:
{category_monthly.to_string()}

В выбранном месяце:
- обращений: {current_value}
- критических: {critical_count}
- высокий риск ухода: {high_churn_count}
- средняя серьёзность: {avg_severity:.2f}
- среднее число обращений этой категории за месяц: {yearly_average:.1f}

Основные проблемы:
{top_problems.to_string()}
"""

        if selected_month in months:
            current_index = months.index(
                selected_month
            )

            if current_index > 0:
                previous_month = months[
                    current_index - 1
                ]
                previous_value = int(
                    category_monthly.loc[
                        previous_month
                    ]
                )

                if previous_value > 0:
                    change_percent = (
                        (current_value - previous_value)
                        / previous_value
                        * 100
                    )

                    context += f"""

Сравнение, рассчитанное Python:
- предыдущий месяц: {previous_month}
- обращений в предыдущем месяце: {previous_value}
- изменение: {change_percent:+.1f}%
"""

        return context

    if selected_category:
        category_data = data[
            data["category"] == selected_category
        ]

        category_monthly = (
            category_data.groupby("month")
            .size()
            .reindex(
                sorted(data["month"].unique()),
                fill_value=0,
            )
        )

        top_problems = (
            category_data["summary"]
            .value_counts()
            .head(8)
        )

        return f"""
КАТЕГОРИЯ: {selected_category}
Всего обращений категории: {len(category_data)}
Критических: {int((category_data["severity"] == 5).sum())}
Высокий риск ухода: {int((category_data["churn_risk"] == "high").sum())}

Динамика:
{category_monthly.to_string()}

Основные проблемы:
{top_problems.to_string()}
"""

    if selected_month:
        month_data = data[
            data["month"] == selected_month
        ]

        month_categories = (
            month_data["category"]
            .value_counts()
            .head(10)
        )

        month_problems = (
            month_data["summary"]
            .value_counts()
            .head(8)
        )

        return f"""
МЕСЯЦ: {selected_month}
Всего обращений: {len(month_data)}
Критических: {int((month_data["severity"] == 5).sum())}
Высокий риск ухода: {int((month_data["churn_risk"] == "high").sum())}

Категории:
{month_categories.to_string()}

Основные проблемы:
{month_problems.to_string()}
"""

    top_categories = (
        data["category"]
        .value_counts()
        .head(10)
    )

    top_problems = (
        data["summary"]
        .value_counts()
        .head(10)
    )

    churn_by_category = (
        data[
            data["churn_risk"] == "high"
        ]["category"]
        .value_counts()
        .head(5)
    )

    priority_context = build_priority_table(
        data,
        limit=8,
    )

    return f"""
ОБЩИЕ ПОКАЗАТЕЛИ
Всего обращений: {len(data)}
Критических: {int((data["severity"] == 5).sum())}
Высокий риск ухода: {int((data["churn_risk"] == "high").sum())}
Негативных: {int((data["sentiment"] == "negative").sum())}

ТОП КАТЕГОРИЙ
{top_categories.to_string()}

ТОП ПРОБЛЕМ
{top_problems.to_string()}

ВЫСОКИЙ РИСК УХОДА ПО КАТЕГОРИЯМ
{churn_by_category.to_string()}

ПРИОРИТЕТНЫЕ ПРОБЛЕМЫ
{priority_context[[
    "category",
    "summary",
    "requests",
    "avg_severity",
    "critical",
    "high_churn",
    "priority_score",
]].to_string(index=False)}
"""


def clean_llm_answer(answer):
    answer = (answer or "").strip()

    if not answer:
        return (
            "Не удалось получить содержательный "
            "ответ от модели."
        )

    markers = [
        "Вывод:",
        "**Вывод**",
        "### Вывод",
    ]

    positions = [
        answer.find(marker)
        for marker in markers
        if answer.find(marker) >= 0
    ]

    if positions:
        answer = answer[min(positions):].strip()

    bad_phrases = (
        "хорошо,",
        "давайте посмотрим",
        "давай посмотрим",
        "давайте разбер",
        "давай разбер",
        "мне нужно проанализ",
        "мне нужно разобраться",
        "сначала посмотрю",
        "сначала я посмотрю",
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in answer.split("\n\n")
        if paragraph.strip()
    ]

    while paragraphs and any(
        phrase in paragraphs[0].lower()
        for phrase in bad_phrases
    ):
        paragraphs.pop(0)

    cleaned = "\n\n".join(
        paragraphs
    ).strip()

    return cleaned or answer


# ----------------------------------------------------------
# ВКЛАДКИ — ПУТЬ РУКОВОДИТЕЛЯ
# ----------------------------------------------------------

(
    tab_charts,
    tab_risks,
    tab_ai,
    tab_requests,
) = st.tabs(
    [
        "1 · 📈 Графики",
        "2 · ⚠️ Риски",
        "3 · 🤖 AI-аналитик",
        "4 · 🔎 Все обращения",
    ]
)


# ==========================================================
# 1. ГРАФИКИ
# ==========================================================

with tab_charts:

    filtered_df = render_tab_context(
        "charts"
    )

    render_context_summary(filtered_df)
    st.divider()

    if filtered_df.empty:
        st.info(
            "По выбранным фильтрам нет обращений."
        )

    else:
        section_header(
            "Общая картина",
            (
                "Сначала руководитель оценивает объём, динамику "
                "и структуру клиентских обращений."
            ),
        )

        total = len(filtered_df)
        critical = int(
            (filtered_df["severity"] == 5).sum()
        )
        high_churn = int(
            (
                filtered_df["churn_risk"]
                == "high"
            ).sum()
        )
        negative = int(
            (
                filtered_df["sentiment"]
                == "negative"
            ).sum()
        )

        critical_percent = (
            critical / total * 100
            if total
            else 0
        )
        churn_percent = (
            high_churn / total * 100
            if total
            else 0
        )
        negative_percent = (
            negative / total * 100
            if total
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Обращения",
            f"{total:,}",
        )
        col2.metric(
            "Критические",
            f"{critical} ({critical_percent:.1f}%)",
        )
        col3.metric(
            "Высокий риск ухода",
            f"{high_churn} ({churn_percent:.1f}%)",
        )
        col4.metric(
            "Негативная тональность",
            f"{negative_percent:.1f}%",
        )

        st.divider()

        section_header(
            "Динамика обращений",
            "Как меняется нагрузка на поддержку во времени.",
        )

        monthly = (
            filtered_df.groupby("month")
            .size()
            .reset_index(name="requests")
        )

        fig_monthly = px.line(
            monthly,
            x="month",
            y="requests",
            markers=True,
            labels={
                "month": "Месяц",
                "requests": "Количество обращений",
            },
        )

        fig_monthly.update_layout(
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title=None,
        )

        polish_figure(
            fig_monthly,
            height=355,
        )

        st.plotly_chart(
            fig_monthly,
            width="stretch",
            config={
                "displayModeBar": False
            },
        )

        left, right = st.columns(2)

        with left:
            section_header(
                "Категории обращений",
                "Какие направления создают основную нагрузку.",
            )

            category_data = (
                filtered_df["category"]
                .value_counts()
                .reset_index()
            )
            category_data.columns = [
                "category",
                "requests",
            ]

            fig_categories = px.bar(
                category_data,
                x="requests",
                y="category",
                orientation="h",
                labels={
                    "requests": "Обращения",
                    "category": "Категория",
                },
            )

            fig_categories.update_layout(
                xaxis_title=None,
                yaxis_title=None,
            )
            fig_categories.update_yaxes(
                categoryorder="total ascending"
            )

            polish_figure(
                fig_categories,
                height=405,
            )

            st.plotly_chart(
                fig_categories,
                width="stretch",
                config={
                    "displayModeBar": False
                },
            )

        with right:
            section_header(
                "Риск ухода клиентов",
                "Как распределены обращения по churn risk.",
            )

            churn_data = (
                filtered_df["churn_risk"]
                .value_counts()
                .reset_index()
            )
            churn_data.columns = [
                "risk",
                "requests",
            ]

            churn_data["risk"] = (
                churn_data["risk"]
                .map(CHURN_LABELS)
                .fillna(churn_data["risk"])
            )

            fig_churn = px.bar(
                churn_data,
                x="risk",
                y="requests",
                labels={
                    "risk": "Риск ухода",
                    "requests": "Обращения",
                },
            )

            fig_churn.update_layout(
                xaxis_title=None,
                yaxis_title=None,
            )

            polish_figure(
                fig_churn,
                height=405,
            )

            st.plotly_chart(
                fig_churn,
                width="stretch",
                config={
                    "displayModeBar": False
                },
            )


# ==========================================================
# 2. РИСКИ
# ==========================================================

with tab_risks:

    filtered_df = render_tab_context(
        "risks"
    )

    render_context_summary(filtered_df)
    st.divider()

    if filtered_df.empty:
        st.info(
            "По выбранным фильтрам нет обращений."
        )

    else:
        section_header(
            "Возможные риски",
            (
                "После общей картины руководитель смотрит, "
                "какие проблемы требуют первоочередного внимания."
            ),
        )

        priority = build_priority_table(
            filtered_df
        )

        priority_display = priority[
            [
                "category",
                "summary",
                "requests",
                "avg_severity",
                "critical",
                "high_churn",
                "priority_score",
            ]
        ].head(15).rename(
            columns={
                "category": "Категория",
                "summary": "Проблема",
                "requests": "Обращения",
                "avg_severity": "Средняя серьёзность",
                "critical": "Критические",
                "high_churn": "Высокий риск ухода",
                "priority_score": "Приоритет",
            }
        )

        max_priority = (
            float(
                priority_display[
                    "Приоритет"
                ].max()
            )
            if not priority_display.empty
            else 1.0
        )

        st.dataframe(
            priority_display,
            width="stretch",
            hide_index=True,
            height=430,
            column_config={
                "Средняя серьёзность":
                    st.column_config.NumberColumn(
                        "Средняя серьёзность",
                        format="%.2f",
                    ),
                "Приоритет":
                    st.column_config.ProgressColumn(
                        "Приоритет",
                        help=(
                            "Чем выше значение, тем выше "
                            "управленческий приоритет проблемы."
                        ),
                        min_value=0.0,
                        max_value=max_priority,
                        format="%.1f",
                    ),
            },
        )

        st.divider()

        section_header(
            "Критические обращения",
            (
                "Конкретные случаи с максимальной серьёзностью "
                "для оперативного разбора."
            ),
        )

        critical_df = filtered_df[
            filtered_df["severity"] == 5
        ][
            [
                "date",
                "category",
                "theme",
                "text",
                "summary",
                "churn_risk",
                "recommendation",
            ]
        ].sort_values(
            "date",
            ascending=False,
        )

        critical_display = (
            critical_df.head(30).copy()
        )

        if "churn_risk" in critical_display.columns:
            critical_display["churn_risk"] = (
                critical_display[
                    "churn_risk"
                ]
                .map(CHURN_LABELS)
                .fillna(
                    critical_display[
                        "churn_risk"
                    ]
                )
            )

        critical_display = (
            critical_display.rename(
                columns=COLUMN_LABELS
            )
        )

        st.dataframe(
            critical_display,
            width="stretch",
            hide_index=True,
            height=470,
            column_config={
                "Дата":
                    st.column_config.DatetimeColumn(
                        "Дата",
                        format="DD.MM.YYYY",
                    ),
                "Текст обращения":
                    st.column_config.TextColumn(
                        "Текст обращения",
                        width="large",
                    ),
                "Рекомендация":
                    st.column_config.TextColumn(
                        "Рекомендация",
                        width="large",
                    ),
            },
        )


# ==========================================================
# 3. AI-АНАЛИТИК
# ==========================================================

with tab_ai:

    filtered_df = render_tab_context(
        "ai"
    )

    render_context_summary(filtered_df)
    st.divider()

    if filtered_df.empty:
        st.info(
            "По выбранным фильтрам нет обращений."
        )

    else:
        section_header(
            "AI-план действий",
        )

        priority = build_priority_table(
            filtered_df
        )
        top3 = priority.head(3)

        # --------------------------------------------------
        # TOP-3 В ОДНУ СТРОКУ
        # --------------------------------------------------

        priority_cols = st.columns(3)

        for position, (col, (_, row)) in enumerate(
            zip(priority_cols, top3.iterrows()),
            start=1,
        ):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"### #{position}"
                    )

                    st.markdown(
                        f"**{row['category']}**"
                    )

                    st.write(
                        row["summary"]
                    )

                    metric_1, metric_2 = st.columns(2)

                    metric_1.metric(
                        "Обращения",
                        int(row["requests"]),
                    )

                    metric_2.metric(
                        "Критические",
                        int(row["critical"]),
                    )

                    st.markdown(
                        (
                            f"**Серьёзность:** "
                            f"{float(row['avg_severity']):.2f}"
                        )
                    )

                    st.markdown(
                        (
                            f"**Высокий риск ухода:** "
                            f"{int(row['high_churn'])}"
                        )
                    )

        st.write("")

        # --------------------------------------------------
        # CTA — СФОРМИРОВАТЬ ПЛАН
        # --------------------------------------------------

        button_col_left, button_col_center, button_col_right = st.columns(
            [1.1, 2.8, 1.1]
        )

        with button_col_center:
            generate_plan = st.button(
                "✨ Сформировать AI-план действий",
                type="primary",
                key="generate_management_plan",
                width="stretch",
            )

        # --------------------------------------------------
        # ГЕНЕРАЦИЯ ПЛАНА
        # --------------------------------------------------

        if generate_plan:
            short_context = (
                build_recommendation_prompt(
                    filtered_df,
                    priority,
                    limit=3,
                )
            )

            actions_by_number = {}

            try:
                with st.spinner(
                    "Qwen формирует управленческий план..."
                ):
                    rec_response = chat(
                        model="qwen3:4b",
                        format="json",
                        messages=[
                            {
                                "role": "system",
                                "content": """
/no_think

Ты AI-консультант руководителя коммерческого блока.

Твоя задача — не пересчитывать данные, а превратить уже рассчитанные
Python факты в конкретный план действий.

Python уже:
- применил фильтры;
- посчитал показатели;
- определил TOP-3 проблемы;
- передал частые темы;
- передал реальные примеры обращений клиентов;
- передал рекомендации, уже содержащиеся в данных.

Для каждой проблемы верни ТРИ РАЗНЫХ действия:

1. immediate_action
Что команда должна сделать сейчас с текущей проблемой.

2. process_action
Какое изменение процесса, продукта или контроля стоит проверить,
чтобы такие обращения повторялись реже.

3. control_action
Как проверить эффект, используя ТОЛЬКО имеющиеся показатели:
число обращений, критические обращения и высокий риск ухода.

КРИТИЧЕСКИЕ ПРАВИЛА:
- опирайся только на переданный контекст;
- реальные примеры обращений важнее общих предположений;
- не придумывай техническую первопричину, если её нет в данных;
- не придумывай выручку, прибыль, SLA, бюджет, конверсию,
  средний чек и другие отсутствующие показатели;
- не придумывай числовые цели и сроки;
- не советуй просто «проверить систему», «разобраться» или
  «улучшить качество» без указания конкретного объекта действия;
- immediate_action, process_action и control_action
  не должны повторять друг друга;
- каждое действие — одна короткая фраза, примерно 8–22 слова;
- начинай действие с глагола;
- никаких вступлений, выводов и рассуждений.

Верни ТОЛЬКО JSON:

{
  "recommendations": [
    {
      "id": 1,
      "immediate_action": "конкретное действие",
      "process_action": "конкретное изменение процесса",
      "control_action": "как проверить результат"
    },
    {
      "id": 2,
      "immediate_action": "конкретное действие",
      "process_action": "конкретное изменение процесса",
      "control_action": "как проверить результат"
    },
    {
      "id": 3,
      "immediate_action": "конкретное действие",
      "process_action": "конкретное изменение процесса",
      "control_action": "как проверить результат"
    }
  ]
}
""",
                            },
                            {
                                "role": "user",
                                "content": f"""
TOP-3:

{short_context}

/no_think

Верни только JSON.
""",
                            },
                        ],
                        think=False,
                        keep_alive="30m",
                        options={
                            "temperature": 0.1,
                            "num_predict": 520,
                            "num_ctx": 4096,
                        },
                    )

                actions_by_number = (
                    parse_management_recommendations(
                        rec_response.message.content,
                        count=3,
                    )
                )

            except Exception as error:
                st.warning(
                    "Qwen не вернул корректный план. "
                    "Использую безопасные рекомендации "
                    "из размеченных обращений."
                )
                st.caption(str(error))

            if len(actions_by_number) < 3:
                st.info(
                    "Часть рекомендаций Qwen не прошла "
                    "проверку качества. Для них использован fallback."
                )

            st.divider()

            section_header(
                "Рекомендуемый план действий",
            )

            render_business_recommendations(
                filtered_df,
                priority,
                actions_by_number,
            )


# ==========================================================
# 4. ВСЕ ОБРАЩЕНИЯ
# ==========================================================

with tab_requests:

    section_header(
        "Контекст анализа",
    )

    with st.container(border=True):
        search_query = st.text_input(
            "Поиск по ключевым словам",
            placeholder=(
                "Например: двойное списание, "
                "курьер, возврат денег..."
            ),
            key="requests_search",
        )

        request_filter1, request_filter2, request_filter3, request_filter4 = (
            st.columns(
                [1.45, 1.0, 1.0, 1.45]
            )
        )

        with request_filter1:
            request_categories = st.multiselect(
                "Категории",
                CATEGORIES,
                default=CATEGORIES,
                key="requests_categories",
            )

        with request_filter2:
            request_severity = st.multiselect(
                "Серьёзность",
                [1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5],
                key="requests_severity",
            )

        with request_filter3:
            request_churn_labels = st.multiselect(
                "Риск ухода",
                [
                    "Низкий",
                    "Средний",
                    "Высокий",
                ],
                default=[
                    "Низкий",
                    "Средний",
                    "Высокий",
                ],
                key="requests_churn",
            )

            request_churn = [
                CHURN_VALUES_BY_LABEL[label]
                for label in request_churn_labels
            ]

        with request_filter4:
            requests_period = st.date_input(
                "Период",
                value=(MIN_DATE, MAX_DATE),
                min_value=MIN_DATE,
                max_value=MAX_DATE,
                format="DD.MM.YYYY",
                key="requests_period",
            )

    st.divider()

    section_header(
        "Все обращения",
    )

    requests_df = df[
        (df["category"].isin(request_categories))
        & (df["severity"].isin(request_severity))
        & (df["churn_risk"].isin(request_churn))
    ].copy()

    requests_df = apply_date_range(
        requests_df,
        requests_period,
    )

    if search_query.strip():
        query = (
            search_query
            .strip()
            .lower()
        )

        searchable_columns = [
            "category",
            "theme",
            "text",
            "summary",
            "recommendation",
        ]

        search_mask = pd.Series(
            False,
            index=requests_df.index,
        )

        for column in searchable_columns:
            if column in requests_df.columns:
                search_mask = (
                    search_mask
                    | requests_df[column]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        query,
                        regex=False,
                    )
                )

        requests_df = requests_df[
            search_mask
        ]

    requests_df = requests_df.sort_values(
        "date",
        ascending=False,
    )

    render_context_summary(requests_df)

    if search_query.strip():
        st.write(
            f'Поиск: «{search_query.strip()}»'
        )

    if requests_df.empty:
        st.info(
            "Ничего не найдено. Попробуйте другое "
            "ключевое слово или измените фильтры."
        )

    else:
        display_columns = [
            "date",
            "category",
            "theme",
            "text",
            "summary",
            "severity",
            "sentiment",
            "churn_risk",
            "recommendation",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in requests_df.columns
        ]

        requests_display = (
            requests_df[
                display_columns
            ].copy()
        )

        if "sentiment" in requests_display.columns:
            requests_display["sentiment"] = (
                requests_display[
                    "sentiment"
                ]
                .map(SENTIMENT_LABELS)
                .fillna(
                    requests_display[
                        "sentiment"
                    ]
                )
            )

        if "churn_risk" in requests_display.columns:
            requests_display["churn_risk"] = (
                requests_display[
                    "churn_risk"
                ]
                .map(CHURN_LABELS)
                .fillna(
                    requests_display[
                        "churn_risk"
                    ]
                )
            )

        requests_display = (
            requests_display.rename(
                columns=COLUMN_LABELS
            )
        )

        st.dataframe(
            requests_display,
            width="stretch",
            hide_index=True,
            height=650,
            column_config={
                "Дата":
                    st.column_config.DatetimeColumn(
                        "Дата",
                        format="DD.MM.YYYY",
                    ),
                "Текст обращения":
                    st.column_config.TextColumn(
                        "Текст обращения",
                        width="large",
                    ),
                "Рекомендация":
                    st.column_config.TextColumn(
                        "Рекомендация",
                        width="large",
                    ),
            },
        )

        csv_data = (
            requests_display
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            "Скачать выборку CSV",
            data=csv_data,
            file_name="filtered_support_requests.csv",
            mime="text/csv",
            width="stretch",
        )
