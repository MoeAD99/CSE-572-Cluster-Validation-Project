import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, rfft
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def extract_meal_data(cgm_df, insulin_df):
    valid_meals = []
    meal_cgm_stretch = []
    bins = []
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

    valid_meals_insulin_df = pd.DataFrame(valid_meals)
    min_carb = valid_meals_insulin_df["BWZ Carb Input (grams)"].min()
    max_carb = valid_meals_insulin_df["BWZ Carb Input (grams)"].max()
    num_bins = np.ceil((max_carb - min_carb) / 20)

    for i in range(len(valid_meals_insulin_df)):
        meal_start_cgm = cgm_df[
            cgm_df["Datetime"] > valid_meals_insulin_df["Datetime"].iloc[i]
        ]["Datetime"].min()
        meal_stretch_start = meal_start_cgm - pd.Timedelta(minutes=30)
        meal_stretch_end = meal_start_cgm + pd.Timedelta(minutes=120)
        meal_cgm_stretch.append(
            cgm_df[
                (cgm_df["Datetime"] >= meal_stretch_start)
                & (cgm_df["Datetime"] < meal_stretch_end)
            ]
        )
        cur_meal_bin = int(
            np.floor(
                (valid_meals_insulin_df["BWZ Carb Input (grams)"].iloc[i] - min_carb)
                / 20
            )
        )
        bins.append(cur_meal_bin)

    valid_meals_insulin_df["Bin"] = bins
    # print(len(meal_cgm_stretch))

    meals_with_complete_data = [
        len(i) == 30 and not i["Sensor Glucose (mg/dL)"].hasnans
        for i in meal_cgm_stretch
    ]

    valid_meals_insulin_df["Meal with complete data"] = meals_with_complete_data

    cleaned_meals_df = valid_meals_insulin_df[
        valid_meals_insulin_df["Meal with complete data"] == True
    ]
    # print(cleaned_meals_df)
    meal_cgm_stretch = [
        i
        for i in meal_cgm_stretch
        if len(i) == 30 and not i["Sensor Glucose (mg/dL)"].hasnans
    ]

    meal_matrix = np.array(
        [i["Sensor Glucose (mg/dL)"].to_numpy() for i in meal_cgm_stretch]
    )
    # print(valid_meals_df[:10])
    return num_bins, meal_matrix, cleaned_meals_df


def extract_meal_features(meal_matrix):
    time_to_max_cgm_from_meal_start = []
    diff_cgm_max_meal = []
    features = []
    two_hour_stretch = pd.period_range(
        start="2024-01-01 00:00:00", end="2024-01-01 02:29:00", freq="5min"
    )
    two_hour_stretch = [i.to_timestamp() for i in two_hour_stretch][::-1]
    # two_hours = list(two_hours[::-1])
    for i, meal in enumerate(meal_matrix):
        meal_start_time = two_hour_stretch[23]
        max_cgm_after_meal_index = meal[:23].argmax()
        max_cgm_time = two_hour_stretch[max_cgm_after_meal_index]
        time_diff = pd.Timedelta(max_cgm_time - meal_start_time).seconds / 60 / 60
        time_to_max_cgm_from_meal_start.append(time_diff)

        max_cgm_after_meal = meal[max_cgm_after_meal_index]
        cgm_meal = meal[23]
        diff_cgm_max_cgm_meal = (max_cgm_after_meal - cgm_meal) / cgm_meal
        diff_cgm_max_meal.append((max_cgm_after_meal - cgm_meal) / cgm_meal)

        rfft_meal = list(abs(rfft(meal)))
        rfft_meal_sorted = list(np.sort(rfft_meal))
        second_peak = rfft_meal_sorted[-2]
        second_peak_index = rfft_meal.index(second_peak)
        third_peak = rfft_meal_sorted[-3]
        third_peak_index = rfft_meal.index(third_peak)

        differential = np.mean(np.diff(list(meal[:23])))
        second_differential = np.mean(np.diff(np.diff(list(meal[:23]))))
        std = np.std(meal)
        mean = np.mean(meal)

        features.append(
            [
                time_diff,
                diff_cgm_max_cgm_meal,
                second_peak,
                second_peak_index,
                third_peak,
                third_peak_index,
                differential,
                second_differential,
            ]
        )
    # print(features[0])
    return features


def main():
    cgm_df, insulin_df = preprocess_data()
    num_bins, meal_matrix, cleaned_meals_df = extract_meal_data(cgm_df, insulin_df)
    meal_features = extract_meal_features(meal_matrix)
    ground_truth_bins = cleaned_meals_df["Bin"].to_numpy()
    kmeans = KMeans(n_init=10,n_clusters=int(num_bins), random_state=0).fit(meal_features)
    kmean_sse = kmeans.inertia_
    
    kmeans_bins = kmeans.labels_
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)
    # print(kmeans.inertia_)
    pca_features = KernelPCA(n_components=2).fit_transform(meal_features)
    pca_centers = KernelPCA(n_components=2).fit_transform(kmeans.cluster_centers_)
    # print(pca_centers)
    # plt.figure(figsize=(10, 8))
    # plt.scatter(pca_features[:, 0], pca_features[:, 1], c=kmeans.labels_)
    # plt.scatter(pca_centers[:, 0], pca_centers[:, 1], color="red", marker="x")
    # plt.show()
    scaled_features = StandardScaler().fit_transform(meal_features)
    # nn = NearestNeighbors(n_neighbors=2)
    # nbrs = nn.fit(scaled_features)
    # distances, indices = nbrs.kneighbors(scaled_features)

    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.show()

    # print(len(valid_meals_df))
    # print(len(meal_matrix))

    dbscan = DBSCAN(eps=43, min_samples=7).fit(meal_features)
    # print(dbscan.labels_)
    # print(dbscan.components_)


if __name__ == "__main__":
    main()
