import streamlit as st
import pandas as pd

st.set_page_config(page_title="Titanic — Задание 15", layout="centered")

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv"
    df = pd.read_csv(url, index_col="PassengerId")
    return df

def compute_survivors(df: pd.DataFrame, age_min: int, age_max: int) -> pd.DataFrame:
    """Возвращает таблицу вида: Pclass | Количество выживших мужчин
    Фильтры: пол=male, Survived=1, age_min <= Age <= age_max.
    Пропуски возраста исключаются естественным образом (NaN не проходит сравнения)."""
    filtered = df[
        (df["Sex"] == "male") &
        (df["Survived"] == 1) &
        (df["Age"] >= age_min) &
        (df["Age"] <= age_max)
    ]
    result = (
        filtered.groupby("Pclass")
        .size()
        .reset_index(name="Количество выживших мужчин")
        .sort_values("Pclass", ascending=True)
        .reset_index(drop=True)
    )
    return result

# --- загрузка данных ---
df = load_data()

st.title("Задание 15 — Titanic")
st.markdown(
    "Подсчитать **количество выживших мужчин по каждому классу обслуживания**, "
    "указав диапазон возрастов (от … и до …)."
)

# --- элементы управления ---
age_min = st.number_input("Возраст от:", min_value=0, max_value=100, value=0, step=1)
age_max = st.number_input("Возраст до:", min_value=0, max_value=100, value=80, step=1)

# --- вычисление и вывод ---
result = compute_survivors(df, age_min, age_max)
st.subheader("Результат")
st.dataframe(result)
st.caption("Класс 1 — люкс, 2 — средний, 3 — эконом.")
