import numpy as np
import pandas as pd


def preprocess_data():
    cgm_df = pd.read_csv("CGMData.csv", low_memory=False)
    cgm_df["Datetime"] = pd.to_datetime(cgm_df["Date"] + " " + cgm_df["Time"])
    cgm_df = cgm_df[["Datetime", "Sensor Glucose (mg/dL)"]]

    insulin_df = pd.read_csv("InsulinData.csv", low_memory=False)
    insulin_df["Datetime"] = pd.to_datetime(
        insulin_df["Date"] + " " + insulin_df["Time"]
    )
    insulin_df = insulin_df[["Datetime", "BWZ Carb Input (grams)"]]

    return cgm_df, insulin_df


def extract_bins(cgm_df, insulin_df):
    valid_meals = []
    meal_cgm_stretch = []
    valid_carb_input_df = insulin_df[
        (insulin_df["BWZ Carb Input (grams)"].notna())
        & (insulin_df["BWZ Carb Input (grams)"] > 0)
    ]

    time_diff = valid_carb_input_df["Datetime"].diff(periods=-1)

    for i in range(len(time_diff) - 1):
        if (
            i > 0
            and time_diff.iloc[i] < pd.Timedelta(hours=2)
            and time_diff.iloc[i - 1] < pd.Timedelta(hours=2)
        ):
            continue
        else:
            valid_meals.append(valid_carb_input_df.iloc[i])

    valid_meals = pd.DataFrame(valid_meals)

    for i in range(len(valid_meals)):
        cgm_meal_start = cgm_df[cgm_df["Datetime"] > valid_meals["Datetime"].iloc[i]][
            "Datetime"
        ].min()
        meal_stretch_start = cgm_meal_start - pd.Timedelta(minutes=30)
        meal_stretch_end = cgm_meal_start + pd.Timedelta(minutes=120)
        meal_cgm_stretch.append(
            cgm_df[
                (cgm_df["Datetime"] >= meal_stretch_start)
                & (cgm_df["Datetime"] < meal_stretch_end)
            ]
        )

    num_bins = np.ceil(
        valid_meals["BWZ Carb Input (grams)"].max()
        - valid_meals["BWZ Carb Input (grams)"].min() / 20
    )
    length = [len(i) for i in meal_cgm_stretch]
    print(length)
    return num_bins


def main():
    cgm_df, insulin_df = preprocess_data()
    bins = extract_bins(cgm_df, insulin_df)


if __name__ == "__main__":
    main()
