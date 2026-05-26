import pandas as pd
import requests
import io


def run(output_path: str = "data/processed/auto_clean.csv") -> pd.DataFrame:
    res = requests.get("http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data").content

    auto = pd.read_csv(io.StringIO(res.decode("utf-8")), header=None, na_values="?")

    auto.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
                    'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
                    'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore',
                    'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

    auto.dropna(subset=['price'], inplace=True)

    print(auto.isnull().sum().sort_values(ascending=False))

    auto.drop(columns=['normalized-losses'], inplace=True)

    numeric_cols = auto.select_dtypes(include='number').columns
    auto[numeric_cols] = auto[numeric_cols].fillna(auto[numeric_cols].median())

    categorical_cols = auto.select_dtypes(include='object').columns
    auto[categorical_cols] = auto[categorical_cols].fillna(auto[categorical_cols].mode().iloc[0])

    auto.drop_duplicates(inplace=True)

    categorical_cols = ['make', 'fuel-type', 'aspiration', 'num-of-doors',
                        'body-style', 'drive-wheels', 'engine-location',
                        'engine-type', 'num-of-cylinders', 'fuel-system']

    auto = pd.get_dummies(auto, columns=categorical_cols, drop_first=True)

    auto.to_csv(output_path, index=False)
    return auto


if __name__ == "__main__":
    run()
