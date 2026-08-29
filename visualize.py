import pandas as pd
import plotly.express as px


# Загружаем обращения
df = pd.read_csv("data/support_requests.csv")

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)


# Считаем количество обращений
monthly = (
    df.groupby(["month", "category"])
    .size()
    .reset_index(name="requests")
)


# Пока показываем самые показательные категории
important_categories = [
    "Доставка",
    "Оплата",
    "Мобильное приложение",
    "Поддержка",
]

chart_data = monthly[
    monthly["category"].isin(important_categories)
]


fig = px.line(
    chart_data,
    x="month",
    y="requests",
    color="category",
    markers=True,
    title="Динамика клиентских обращений",
    labels={
        "month": "Месяц",
        "requests": "Количество обращений",
        "category": "Категория",
    },
)

fig.show()