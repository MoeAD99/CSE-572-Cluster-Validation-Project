import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, rfft
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import variation


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
    print(max_carb)
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
    # print(len([i for i in bins if i == 6]))
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
        gradient = np.mean(
            np.gradient(meal[:23])
        )
        variance = np.var(meal[:23])
        var = variation(meal[:23])
        min_cgm = np.min(meal)
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
    return np.array(features)


def get_truth_matrix(num_bins, ground_truth_bins, labels):
    matrix = []
    # print(np.argwhere(ground_truth_bins == 0).flatten())
    # print(np.unique(labels))
    for i in np.unique(labels):
        if i != -1:
            label_indices = np.argwhere(labels == i).flatten()
            corresponding_bins = ground_truth_bins[label_indices]
            row = [len(np.argwhere(corresponding_bins == i)) for i in range(num_bins)]
            matrix.append(row)
    return np.array(matrix)


def calc_dbscan_sse(meal_features, labels):
    sse = 0
    for i in np.unique(labels):
        if i != -1:
            label_indices = np.argwhere(labels == i).flatten()
            cluster = meal_features[label_indices]
            cluster_center = np.mean(cluster, axis=0)
            distances = np.sum((cluster - cluster_center) ** 2, axis=1)
            sse += np.sum(distances)
    return sse


def calc_entropy(truth_matrix):
    num_clusters = truth_matrix.shape[0]
    total_entropy = 0
    for i in range(num_clusters):
        # cluster_entropy = np.sum(-truth_matrix[:, i] * np.log2(truth_matrix[:, i]))
        num_datapoints = np.sum(truth_matrix[i])
        cluster_entropy = -1 * np.sum(
            [
                (truth_matrix[i, j] / num_datapoints)
                * (np.log2(truth_matrix[i, j] / num_datapoints))
                for j in range(truth_matrix.shape[0])
                if truth_matrix[i, j] > 0
            ]
        )
        total_entropy += cluster_entropy * num_datapoints
    weighted_entropy = total_entropy / np.sum(truth_matrix)
    return weighted_entropy


def calc_purity(truth_matrix):
    total_purity = 0
    num_clusters = truth_matrix.shape[0]
    for i in range(num_clusters):
        num_datapoints = np.sum(truth_matrix[i])
        cluster_purity = np.max(truth_matrix[i] / num_datapoints)
        total_purity += cluster_purity * num_datapoints
    weighted_purity = total_purity / np.sum(truth_matrix)

    return weighted_purity


def main():
    cgm_df, insulin_df = preprocess_data()
    num_bins, meal_matrix, cleaned_meals_df = extract_meal_data(cgm_df, insulin_df)
    meal_features = extract_meal_features(meal_matrix)
    ground_truth_bins = cleaned_meals_df["Bin"].to_numpy()
    kmeans = KMeans(n_init=10, n_clusters=int(num_bins), random_state=0).fit(
        meal_features
    )
    km_truth_matrix = get_truth_matrix(int(num_bins), ground_truth_bins, kmeans.labels_)
    kmean_sse = kmeans.inertia_
    kmean_entropy = calc_entropy(km_truth_matrix)
    kmean_purity = calc_purity(km_truth_matrix)

    # print(ground_truth_bins)
    # print(kmeans.labels_)
    print(kmean_entropy)
    print(kmean_purity)
    # print(np.sum(km_truth_matrix))
    # print(len(meal_features))

    # print(ground_truth_bins == kmeans.labels_)
    # print(np.sum(kmeans.labels_ == 0))
    # print(kmeans.cluster_centers_)
    # print(kmeans.inertia_)
    pca_features = KernelPCA(n_components=2).fit_transform(meal_features)
    pca_centers = KernelPCA(n_components=2).fit_transform(kmeans.cluster_centers_)
    # print(pca_centers)
    # plt.figure(figsize=(10, 8))
    # plt.scatter(pca_features[:, 0], pca_features[:, 1], c=kmeans.labels_)
    # plt.scatter(pca_centers[:, 0], pca_centers[:, 1], color="red", marker="x")
    # plt.show()

    # scaled_features = StandardScaler().fit_transform(meal_features)
    # nn = NearestNeighbors(n_neighbors=2)
    # nbrs = nn.fit(meal_features)
    # distances, indices = nbrs.kneighbors(meal_features)

    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.show()

    # eps_values = np.arange(20, 120, 5)
    # min_samples = range(1,40)  

    # for eps in eps_values:
    #     for samples in min_samples:
    #         dbscan = DBSCAN(eps=eps, min_samples=samples).fit(meal_features)
    #         n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    #         n_noise = list(dbscan.labels_).count(-1)
    #         if n_clusters == 7:
    #             print(f"Eps: {eps}, min_samples: {samples}, Clusters: {n_clusters}, Noise Points: {n_noise}")

    dbscan = DBSCAN(eps=40, min_samples=5).fit(meal_features)
    labels = dbscan.labels_
    dbs_truth_matrix = get_truth_matrix(int(num_bins), ground_truth_bins, labels)
    dbs_sse = calc_dbscan_sse(meal_features, labels)
    # dbs_entropy = calc_entropy(dbs_truth_matrix)
    # dbs_purity = calc_purity(dbs_truth_matrix)
    # print(dbs_entropy)
    # print(dbs_purity)

    # print("-1: ", np.sum(labels == -1))
    # print("0:  ", np.sum(labels == 0))
    # print("1:  ", np.sum(labels == 1))
    # print("2:  ", np.sum(labels == 2))
    # print("3:  ", np.sum(labels == 3))
    # print("4:  ", np.sum(labels == 4))
    # print("5:  ", np.sum(labels == 5))
    # print("6:  ", np.sum(labels == 6))
    # print(">6:  ", np.sum(labels > 8))

    # result = pd.DataFrame(
    #     [[kmean_sse, dbs_sse, kmean_entropy, dbs_entropy, kmean_purity, dbs_purity]]
    # )
    # result.to_csv("Result.csv", header=False, index=False)


if __name__ == "__main__":
    main()
