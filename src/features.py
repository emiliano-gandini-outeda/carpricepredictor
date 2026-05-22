import pandas as pd
import numpy as np

auto = pd.read_csv("data/processed/auto_clean.csv")

# convert booleans to 0/1
bool_cols = auto.select_dtypes(include='bool').columns
auto[bool_cols] = auto[bool_cols].astype(int)

# Features

auto['horsepower_per_kg']   = auto['horsepower'] / auto['curb-weight']
auto['engine_per_kg']       = auto['engine-size'] / auto['curb-weight']
auto['mpg_avg']             = (auto['city-mpg'] + auto['highway-mpg']) / 2
auto['footprint']           = auto['length'] * auto['width']


# check correlations of new features
pd.set_option('display.max_rows', None)
print(auto.corr(numeric_only=True)['price'].sort_values())

# check if any two features are highly correlated with EACH OTHER (multicollinearity)
corr_matrix = auto.corr(numeric_only=True)

# find pairs above 0.9
high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix != 1.0)
cols = high_corr.any()
print(corr_matrix[cols.index[cols]])

# Drop unnecesary vals
auto.drop(columns=[
    # near-zero correlation to price
    'fuel-system_mfi',
    'engine-type_rotor',
    'num-of-cylinders_two',
    'fuel-system_4bbl',
    'fuel-system_spfi',
    'body-style_wagon',
    'num-of-doors_two',
    'engine-type_ohcf',
    'make_mercury',
    'make_renault',
    'make_isuzu',
    'drive-wheels_fwd',
    'engine-type_l',

    # multicollinearity
    'city-mpg',
    'highway-mpg',       # replaced by mpg_avg
    'length',            # replaced by footprint
    'fuel-type_gas',     # inverse of fuel-system_idi
    'compression-ratio', # ~same as fuel-system_idi
    'width',             # captured by footprint
    'wheel-base',        # captured by footprint
], inplace=True)

# Log transform skewed columns
skewed_cols = auto.select_dtypes(include='number').skew()
skewed_cols = skewed_cols[skewed_cols > 1.0].index.tolist()
binary_cols = [col for col in skewed_cols if auto[col].nunique() == 2]
skewed_cols = [col for col in skewed_cols if col not in binary_cols]
print("Log transforming:", skewed_cols)
auto[skewed_cols] = np.log1p(auto[skewed_cols])

# save
auto.to_csv('data/processed/auto_features.csv', index=False)