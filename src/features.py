import pandas as pd
import numpy as np


def run(input_path: str = "data/processed/auto_clean.csv",
        output_path: str = "data/processed/auto_features.csv") -> pd.DataFrame:
    auto = pd.read_csv(input_path)

    bool_cols = auto.select_dtypes(include='bool').columns
    auto[bool_cols] = auto[bool_cols].astype(int)

    auto['horsepower_per_kg'] = auto['horsepower'] / auto['curb-weight']
    auto['engine_per_kg'] = auto['engine-size'] / auto['curb-weight']
    auto['mpg_avg'] = (auto['city-mpg'] + auto['highway-mpg']) / 2
    auto['footprint'] = auto['length'] * auto['width']

    pd.set_option('display.max_rows', None)
    print(auto.corr(numeric_only=True)['price'].sort_values())

    corr_matrix = auto.corr(numeric_only=True)
    high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix != 1.0)
    cols = high_corr.any()
    print(corr_matrix[cols.index[cols]])

    auto.drop(columns=[
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
        'city-mpg',
        'highway-mpg',
        'length',
        'fuel-type_gas',
        'compression-ratio',
        'width',
        'wheel-base',
    ], inplace=True)

    skewed_cols = auto.select_dtypes(include='number').skew()
    skewed_cols = skewed_cols[skewed_cols > 1.0].index.tolist()
    binary_cols = [col for col in skewed_cols if auto[col].nunique() == 2]
    skewed_cols = [col for col in skewed_cols if col not in binary_cols]
    print("Log transforming:", skewed_cols)
    auto[skewed_cols] = np.log1p(auto[skewed_cols])

    auto.to_csv(output_path, index=False)
    return auto


if __name__ == "__main__":
    run()
