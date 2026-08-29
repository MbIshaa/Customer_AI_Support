import pandas as pd


INPUT_FILE = "data/enriched_support_requests.csv"


df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)


# ============================================================
# ОБЩИЕ KPI
# ============================================================

total_requests = len(df)

critical_requests = len(
    df[df["severity"] == 5]
)

high_churn = len(
    df[df["churn_risk"] == "high"]
)

negative_requests = len(
    df[df["sentiment"] == "negative"]
)


print("=" * 70)
print("КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ")
print("=" * 70)

print(f"Всего обращений: {total_requests}")

print(
    f"Критических обращений: "
    f"{critical_requests} "
    f"({critical_requests / total_requests:.1%})"
)

print(
    f"Высокий риск ухода: "
    f"{high_churn} "
    f"({high_churn / total_requests:.1%})"
)

print(
    f"Негативных обращений: "
    f"{negative_requests} "
    f"({negative_requests / total_requests:.1%})"
)


# ============================================================
# ТОП КАТЕГОРИЙ
# ============================================================

print("\n" + "=" * 70)
print("ТОП КАТЕГОРИЙ")
print("=" * 70)

top_categories = (
    df["category"]
    .value_counts()
    .head(10)
)

print(top_categories)


# ============================================================
# ТОП КОНКРЕТНЫХ ПРОБЛЕМ
# ============================================================

print("\n" + "=" * 70)
print("ТОП ПРОБЛЕМ")
print("=" * 70)

top_problems = (
    df["summary"]
    .value_counts()
    .head(15)
)

print(top_problems)


# ============================================================
# ГДЕ БОЛЬШЕ ВСЕГО КРИТИЧЕСКИХ ОБРАЩЕНИЙ
# ============================================================

print("\n" + "=" * 70)
print("КРИТИЧЕСКИЕ ОБРАЩЕНИЯ ПО КАТЕГОРИЯМ")
print("=" * 70)

critical_by_category = (
    df[df["severity"] == 5]
    ["category"]
    .value_counts()
)

print(critical_by_category)


# ============================================================
# РИСК УХОДА ПО КАТЕГОРИЯМ
# ============================================================

print("\n" + "=" * 70)
print("ВЫСОКИЙ РИСК УХОДА ПО КАТЕГОРИЯМ")
print("=" * 70)

churn_by_category = (
    df[df["churn_risk"] == "high"]
    ["category"]
    .value_counts()
)

print(churn_by_category)


# ============================================================
# ДИНАМИКА ПО МЕСЯЦАМ
# ============================================================

print("\n" + "=" * 70)
print("ОБРАЩЕНИЯ ПО МЕСЯЦАМ")
print("=" * 70)

monthly = (
    df.groupby("month")
    .size()
)

print(monthly)


# ============================================================
# КРИТИЧЕСКИЕ ОБРАЩЕНИЯ ПО МЕСЯЦАМ
# ============================================================

print("\n" + "=" * 70)
print("КРИТИЧЕСКИЕ ОБРАЩЕНИЯ ПО МЕСЯЦАМ")
print("=" * 70)

critical_monthly = (
    df[df["severity"] == 5]
    .groupby("month")
    .size()
)

print(critical_monthly)


# ============================================================
# HIGH CHURN ПО МЕСЯЦАМ
# ============================================================

print("\n" + "=" * 70)
print("HIGH CHURN ПО МЕСЯЦАМ")
print("=" * 70)

churn_monthly = (
    df[df["churn_risk"] == "high"]
    .groupby("month")
    .size()
)

print(churn_monthly)


# ============================================================
# ПРИОРИТЕТНЫЕ ПРОБЛЕМЫ
# ============================================================

print("\n" + "=" * 70)
print("ПРИОРИТЕТНЫЕ ПРОБЛЕМЫ")
print("=" * 70)

priority = (
    df.groupby(
        ["category", "summary"]
    )
    .agg(
        requests=("id", "count"),
        avg_severity=("severity", "mean"),
        critical=("severity", lambda x: (x == 5).sum()),
        high_churn=("churn_risk", lambda x: (x == "high").sum()),
    )
    .reset_index()
)


# Простой индекс приоритета
priority["priority_score"] = (
    priority["requests"] * 1
    + priority["avg_severity"] * 5
    + priority["critical"] * 3
    + priority["high_churn"] * 3
)


priority = priority.sort_values(
    "priority_score",
    ascending=False
)


print(
    priority[
        [
            "category",
            "summary",
            "requests",
            "avg_severity",
            "critical",
            "high_churn",
            "priority_score",
        ]
    ].head(15).to_string(index=False)
)


priority.to_csv(
    "data/problem_priorities.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nСоздан файл: "
    "data/problem_priorities.csv"
)