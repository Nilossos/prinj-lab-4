import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit_app import compute_survivors

def make_df():
    # тестовый датафрейм, покрывающий разные случаи:
    # - разные классы обслуживания
    # - разные возраста
    # - мужчины и женщины
    # - выжившие и невыжившие
    return pd.DataFrame([
        {"Pclass": 1, "Sex": "male",   "Age": 18, "Survived": 1},
        {"Pclass": 1, "Sex": "male",   "Age": 40, "Survived": 1},
        {"Pclass": 1, "Sex": "male",   "Age": 111, "Survived": 1},
        {"Pclass": 2, "Sex": "male",   "Age": 30, "Survived": 1},
        {"Pclass": 2, "Sex": "male",   "Age": 30, "Survived": 0},
        {"Pclass": 3, "Sex": "male",   "Age": 15, "Survived": 1},
        {"Pclass": 3, "Sex": "male",   "Age": 50, "Survived": 1},
        {"Pclass": 1, "Sex": "female", "Age": 22, "Survived": 1},
        {"Pclass": 3, "Sex": "female", "Age": 35, "Survived": 1},
        {"Pclass": 2, "Sex": "male",   "Age": pd.NA, "Survived": 1},
    ])

# Базовый тест на подсчет выживших мужчин по классам обслуживания
def test_basic_counts_by_class():
    df = make_df()
    res = compute_survivors(df, age_min=0, age_max=60)
    expected = pd.DataFrame({
        "Pclass": [1, 2, 3],
        "Количество выживших мужчин": [2, 1, 2],
    })
    assert_frame_equal(res.reset_index(drop=True), expected)

# Тест на включение граничных значений возраста
def test_inclusive_age_bounds():
    df = pd.DataFrame([
        {"Pclass": 1, "Sex": "male", "Age": 10, "Survived": 1},  # = min
        {"Pclass": 1, "Sex": "male", "Age": 20, "Survived": 1},  # = max
        {"Pclass": 1, "Sex": "male", "Age": 15, "Survived": 1},  # внутри
        {"Pclass": 1, "Sex": "male", "Age": 9,  "Survived": 1},  # ниже min
        {"Pclass": 1, "Sex": "male", "Age": 21, "Survived": 1},  # выше max
    ])
    res = compute_survivors(df, age_min=10, age_max=20)
    expected = pd.DataFrame({
        "Pclass": [1],
        "Количество выживших мужчин": [3],
    })
    assert_frame_equal(res.reset_index(drop=True), expected)

# Тест на случай, когда нет выживших мужчин в заданном возрастном диапазоне
def test_no_matches_returns_empty():
    df = make_df()
    res = compute_survivors(df, age_min=90, age_max=99)
    expected = pd.DataFrame(columns=["Pclass", "Количество выживших мужчин"])
    res = res.astype({"Pclass": "int64"}) if not res.empty else res
    assert list(res.columns) == list(expected.columns)
    assert res.empty

# Тест на случай, когда в данных есть пропуски в возрасте
def test_nan_ages_excluded():
    df = pd.DataFrame([
        {"Pclass": 2, "Sex": "male", "Age": pd.NA, "Survived": 1},
        {"Pclass": 2, "Sex": "male", "Age": 25,    "Survived": 1},
    ])
    res = compute_survivors(df, age_min=0, age_max=100)
    expected = pd.DataFrame({
        "Pclass": [2],
        "Количество выживших мужчин": [1],
    })
    assert_frame_equal(res.reset_index(drop=True), expected)
