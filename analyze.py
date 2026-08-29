import pandas as pd


# Загружаем данные
df = pd.read_csv("data/support_requests.csv")

# Преобразуем дату из текста в настоящий тип даты
df["date"] = pd.to_datetime(df["date"])

# Создаем месяц для аналитики
df["month"] = df["date"].dt.to_period("M").astype(str)


print("=" * 60)
print("ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 60)

print(f"Всего обращений: {len(df)}")
print(f"Период: {df['date'].min().date()} — {df['date'].max().date()}")
print(f"Количество категорий: {df['category'].nunique()}")


print("\n" + "=" * 60)
print("ОБРАЩЕНИЯ ПО КАТЕГОРИЯМ")
print("=" * 60)

category_counts = df["category"].value_counts()

print(category_counts)


print("\n" + "=" * 60)
print("ОБРАЩЕНИЯ ПО МЕСЯЦАМ")
print("=" * 60)

monthly_counts = df.groupby("month").size()

print(monthly_counts)


print("\n" + "=" * 60)
print("КАТЕГОРИИ ПО МЕСЯЦАМ")
print("=" * 60)

monthly_categories = pd.crosstab(
    df["month"],
    df["category"]
)

print(monthly_categories)


print("\n" + "=" * 60)
print("САМАЯ ЧАСТАЯ ПРОБЛЕМА КАЖДОГО МЕСЯЦА")
print("=" * 60)

for month, group in df.groupby("month"):
    top_category = group["category"].value_counts().index[0]
    count = group["category"].value_counts().iloc[0]

    print(
        f"{month}: {top_category} — {count} обращений"
    )


# Сохраняем таблицу для дальнейшего использования
monthly_categories.to_csv(
    "data/monthly_categories.csv",
    encoding="utf-8-sig"
)

print("\nСоздан файл: data/monthly_categories.csv")