import numpy as np
import pandas as pd

import surprise
from surprise import SVD, NMF
from surprise import Dataset, accuracy
from surprise.reader import Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore
from joblib import dump, load

# Must add save and load feature

"""
Returns the predicted recipes dataframe
"""


def get_similar_items_knn(user_id: int, item_id: int, output_values: int) -> pd.DataFrame:

    reader = Reader(rating_scale=(0, 5))

    df = pd.read_csv("data/interactions_train.csv")
    df = df.drop(columns=["u", "i", "date"])
    df = df.drop(df[df.rating == 0].index)

    data = Dataset.load_from_df(df, reader=reader)

    train_set, test_set = train_test_split(data, test_size=0.99)

    df_real = pd.read_csv("data/recipes.csv")

    # using msd similarity measure
    sim_options = {"name": "pearson_baseline", "min_support": 5, "user_based": False}

    # Knn baseline with 21 neighbors
    base13 = KNNBaseline(k=21, sim_options=sim_options)

    base13.fit(train_set)

    # testing and predicting with test dataset
    base13_predictions = base13.test(test_set)
    accuracy.rmse(base13_predictions)

    knn_user_id = user_id
    item_id = int(item_id)

    rec_cat_val = df_real[df_real["RecipeId"] == item_id]["RecipeCategory"]
    print(rec_cat_val.values[0])

    # dataset to get recipe details
    # find category of the item
    suggest = set(df_real[df_real["RecipeCategory"] == rec_cat_val.values[0]].index) - set(
        df_real[df_real["RecipeId"] == item_id].index
    )
    suggest_list = list(suggest)
    suggest_list_df = df_real.loc[suggest_list]
    prediction_of_recipe_cat = []
    for index, rows in suggest_list_df.iterrows():
        prediction_of_recipe_cat.append(base13.predict(uid=knn_user_id, iid=suggest_list_df.loc[index]["RecipeId"]))

    predicted_recipe_id = []
    for i in range(len(prediction_of_recipe_cat)):
        if prediction_of_recipe_cat[i].est > 4.6:
            predicted_recipe_id.append(prediction_of_recipe_cat[i].iid)

    predicted_result_list = df_real.loc[predicted_recipe_id[:output_values]]

    return predicted_result_list


def knn_inference(model_load_path: str, knn_user_id: int, item_id: int, output_values: int) -> pd.DataFrame:
    df_real = pd.read_csv("data/recipes.csv")
    item_id = int(item_id)

    rec_cat_val = df_real[df_real["RecipeId"] == item_id]["RecipeCategory"]

    knn_loaded = load(model_load_path)

    suggest = set(df_real[df_real["RecipeCategory"] == rec_cat_val.values[0]].index) - set(
        df_real[df_real["RecipeId"] == item_id].index
    )
    suggest_list = list(suggest)
    suggest_list_df = df_real.loc[suggest_list]
    prediction_of_recipe_cat = []
    for index, rows in suggest_list_df.iterrows():
        prediction_of_recipe_cat.append(knn_loaded.predict(uid=knn_user_id, iid=suggest_list_df.loc[index]["RecipeId"]))

    predicted_recipe_id = []
    for i in range(len(prediction_of_recipe_cat)):
        if prediction_of_recipe_cat[i].est > 4.6:
            predicted_recipe_id.append(prediction_of_recipe_cat[i].iid)

    predicted_result_list = df_real.loc[predicted_recipe_id[:output_values]]

    return predicted_result_list


if __name__ == "__main__":
    final = get_similar_items_knn(2046, 10744, 10)
    print(final)
