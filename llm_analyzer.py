import time
from pathlib import Path
from typing import Literal

import pandas as pd
from ollama import chat
from pydantic import BaseModel, Field


class ComplaintAnalysis(BaseModel):
    summary: str

    sentiment: Literal[
        "positive",
        "neutral",
        "negative"
    ]

    severity: int = Field(ge=1, le=5)

    churn_risk: Literal[
        "low",
        "medium",
        "high"
    ]

    root_cause: str | None = None

    recommendation: str


INPUT_FILE = "data/support_requests.csv"
CACHE_FILE = "data/llm_unique_analysis.csv"
OUTPUT_FILE = "data/enriched_support_requests.csv"


df = pd.read_csv(INPUT_FILE)


# Берём только уникальные комбинации текста и категории
unique_requests = (
    df[
        ["category", "theme", "text"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


print(f"Всего обращений: {len(df)}")
print(f"Уникальных текстов для LLM: {len(unique_requests)}")


# Если анализ запускался раньше —
# загружаем уже обработанные обращения
if Path(CACHE_FILE).exists():
    results_df = pd.read_csv(CACHE_FILE)

    processed_keys = set(
        zip(
            results_df["category"],
            results_df["theme"],
            results_df["text"]
        )
    )

    print(
        f"Уже обработано ранее: {len(results_df)}"
    )

else:
    results_df = pd.DataFrame()

    processed_keys = set()


results = []


for index, row in unique_requests.iterrows():

    key = (
        row["category"],
        row["theme"],
        row["text"]
    )

    # Не анализируем повторно
    if key in processed_keys:
        continue


    prompt = f"""
Категория обращения:
{row["category"]}

Тема:
{row["theme"]}

Текст клиента:
{row["text"]}
"""


    try:

        response = chat(
            model="qwen3:4b",

            messages=[
                {
                    "role": "system",
                    "content": """
Ты аналитик клиентской поддержки крупной компании.

Категория обращения уже известна.
Не пытайся определять её заново.

Твоя задача — понять содержание текста клиента.

Правила:

1. summary:
Создай короткое название проблемы для бизнес-аналитики.

Требования:
- 3-7 слов
- без слов "клиент", "пользователь", "человек"
- без пересказа всей ситуации
- использовать деловой стиль

Примеры:

Плохо:
"Клиент не может оплатить заказ банковской картой"

Хорошо:
"Ошибка оплаты банковской картой"

Плохо:
"Клиент хочет вернуть товар, но не понимает как"

Хорошо:
"Сложность оформления возврата"

Плохо:
"Курьер сообщил, что не может найти мой адрес"

Хорошо:
"Проблема поиска адреса доставки"

2. sentiment:
positive, neutral или negative.

3. severity:
1 — вопрос или небольшое неудобство
2 — небольшая проблема
3 — существенная проблема
4 — серьёзная проблема
5 — критическая проблема

Если клиент прямо говорит, что уйдёт,
перестанет покупать, обратится в суд
или проблема связана с серьёзной
финансовой потерей — severity 5.

4. churn_risk:
low, medium или high.

Если клиент говорит,
что больше не будет пользоваться компанией,
churn_risk = high.

5. root_cause:
Указывай причину только тогда,
когда она прямо следует из текста.

Если причина неизвестна,
верни null.

Не выдумывай технические причины.

6. recommendation:
Одно короткое и конкретное действие
для бизнеса.
""",
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            format=ComplaintAnalysis.model_json_schema(),

            think=False,

            options={
                "temperature": 0
            },
        )


        analysis = ComplaintAnalysis.model_validate_json(
            response.message.content
        )


        result = {
            "category": row["category"],
            "theme": row["theme"],
            "text": row["text"],

            "summary": analysis.summary,
            "sentiment": analysis.sentiment,
            "severity": analysis.severity,
            "churn_risk": analysis.churn_risk,
            "root_cause": analysis.root_cause,
            "recommendation": analysis.recommendation,
        }


        results.append(result)


        print(
            f"[{index + 1}/{len(unique_requests)}] "
            f"{row['category']} → {analysis.summary}"
        )


        # Сохраняем каждые 10 результатов
        if len(results) >= 10:

            batch_df = pd.DataFrame(results)

            if Path(CACHE_FILE).exists():

                old_df = pd.read_csv(CACHE_FILE)

                batch_df = pd.concat(
                    [old_df, batch_df],
                    ignore_index=True
                )


            batch_df.to_csv(
                CACHE_FILE,
                index=False,
                encoding="utf-8-sig"
            )


            results = []


    except Exception as error:

        print(
            f"Ошибка при обработке строки "
            f"{index + 1}: {error}"
        )

        time.sleep(2)


# Сохраняем остаток
if results:

    batch_df = pd.DataFrame(results)

    if Path(CACHE_FILE).exists():

        old_df = pd.read_csv(CACHE_FILE)

        batch_df = pd.concat(
            [old_df, batch_df],
            ignore_index=True
        )


    batch_df.to_csv(
        CACHE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# Загружаем весь LLM-анализ
analysis_df = pd.read_csv(CACHE_FILE)


# Присоединяем результаты обратно
# к исходным 1000 обращениям
enriched_df = df.merge(
    analysis_df,
    on=[
        "category",
        "theme",
        "text"
    ],
    how="left"
)


enriched_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)

print(
    f"Создан файл: {OUTPUT_FILE}"
)

print(
    f"Обращений в итоговой базе: "
    f"{len(enriched_df)}"
)

print("\nРаспределение severity:")

print(
    enriched_df["severity"]
    .value_counts()
    .sort_index()
)

print("\nРаспределение churn risk:")

print(
    enriched_df["churn_risk"]
    .value_counts()
)