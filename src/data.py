import pandas as pd
import requests
import io

"""

Importing Data

For this project, I'll use a dataset from UC Irvine Machine Learning Repository,
specifically, http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data .

"""

res = requests.get("http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data").content

auto = pd.read_csv(io.StringIO(res.decode("utf-8")), header=None, na_values="?")

auto.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
                            'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
                            'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore',
                            'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']


# Drop rows where price is missing
auto.dropna(subset=['price'], inplace=True)

# Check for almost empty rows
print(auto.isnull().sum().sort_values(ascending=False))

# Drop them
auto.drop(columns=['normalized-losses'], inplace=True)

# fill numeric NaNs with median (robust to outliers)
numeric_cols = auto.select_dtypes(include='number').columns
auto[numeric_cols] = auto[numeric_cols].fillna(auto[numeric_cols].median())

# fill categorical NaNs with mode
categorical_cols = auto.select_dtypes(include='str').columns
auto[categorical_cols] = auto[categorical_cols].fillna(auto[categorical_cols].mode().iloc[0])

# Remove duplicates
auto.drop_duplicates(inplace=True)


categorical_cols = ['make', 'fuel-type', 'aspiration', 'num-of-doors',
                    'body-style', 'drive-wheels', 'engine-location',
                    'engine-type', 'num-of-cylinders', 'fuel-system']

auto = pd.get_dummies(auto, columns=categorical_cols, drop_first=True)


## Save to CSV
auto.to_csv('data/processed/auto_clean.csv', index=False)