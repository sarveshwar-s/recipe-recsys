import pandas as pd
from surprise import SVD,NMF
from surprise import Dataset, accuracy
from surprise.reader import Reader
from surprise.model_selection import train_test_split
from surprise import NormalPredictor
from surprise import BaselineOnly


# Saves the model
def store_model(model, name) -> list:
    model_dump_name = name + ".joblib"
    dump(model, model_dump_name)
    print(f"model {name} is stored")

def classic_rec_sys(dataset_path: str):
    reader = Reader(rating_scale=(0, 5))

    df= pd.read_csv(dataset_path)
    df= df.drop(columns=['u','i','date'])
    df = df.drop(df[df.rating == 0].index)

    data = Dataset.load_from_df(df, reader=reader)


    train_set, test_set = train_test_split(data, test_size=0.2)

    #  Use the famous SVD train_setsetithm.
    model_svd = SVD()
    final_svd_model = model_svd.fit(train_set)
    store_model(final_svd_model, "svd_model")
    pred_svd=model_svd.test(test_set)


    # NMF
    model_nmf = NMF()
    final_nmf_model = model_nmf.fit(train_set)
    store_model(final_nmf_model, "nmf_model")
    pred_nmf=model_nmf.test(test_set)



    # Normal predictor
    model_normal = NormalPredictor()
    final_normal_model = model_normal.fit(train_set)
    store_model(final_normal_model, "normal_model")
    pred_normal = model_normal.test(test_set)


    # KNN baseline
    model_knnbase = BaselineOnly()
    final_baseline_model = model_knnbase.fit(train_set)
    store_model(final_baseline_model, "baseline_model")
    pred_baseline = final_baseline_model.test(test_set)

    return [accuracy.rmse(pred_svd), accuracy.rmse(pred_nmf), accuracy.rmse(pred_normal), accuracy.rmse(pred_baseline)]