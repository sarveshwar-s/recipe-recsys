import click
import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template
import sys, os
import requests
from flask import request
from flask import session, redirect, url_for, flash
import mysql.connector
import re
import json
from joblib import dump, load

notebook_path = os.getcwd()
path_to_package_test = notebook_path
path_to_app_test = path_to_package_test + "\\app_rec_sys"
sys.path.extend([path_to_package_test, path_to_app_test])


from app_rec_sys.reins_algo import epsilon_greedy

app = Flask(__name__)
app.secret_key = "abc"
if __name__ == "__main__":
    app.run()

hostname = "127.0.0.1"
dbusername = "root"
dbpassword = "root"
database_name = "food_schema"


@app.route("/")
def home_page():
    # write api queries to get data from ML algos
    # Calling reinforcement algo
    # try:
    if session["logged_in"]:
        # if session is true then check for threshold
        db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
        mycursor = db.cursor()
        selectquery = "SELECT count(*) FROM interactions_train where user_id=" + str(session["user_id"])
        mycursor.execute(selectquery)
        results = mycursor.fetchall()
        user_interaction_count = results[0][0]
        if user_interaction_count < 10:
            db_rein = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
            rein_sql = "select * from reinforcement_recsys where user_id=" + str(session["user_id"]) + " limit 18"
            final_display = pd.read_sql(rein_sql, con=db_rein)
            recipe_name = list(final_display["recipe_name"][:18])
            print(recipe_name)
            recipe_desc = list(final_display["recipe_desc"][:18])
            recipe_image = list(final_display["recipe_image"][:18])
            recipe_id = list(final_display["recipe_id"][:18])
            return render_template(
                "food_list_rein.html",
                recipe_name=recipe_name,
                recipe_desc=recipe_desc,
                recipe_id=recipe_id,
                recipe_image=recipe_image,
            )

        elif user_interaction_count > 10:

            knn_user_id = int(session["user_id"])
            if knn_user_id == 4657:
                print("Taking other paramters")
                session["using_svd"] = True
                df_temp = pd.read_csv("data/svd_partial.csv")
                recipe_name = list(df_temp["Name"][:18])
                recipe_desc = list(df_temp["Description"][:18])
                return_images_classic = list(df_temp["Images"][:18])
                recipe_image = return_images_classic
                recipe_id_classic = list(df_temp["RecipeId"][:18])
                return render_template(
                    "food_list.html",
                    recipe_name=recipe_name,
                    recipe_image=recipe_image,
                    recipe_desc=recipe_desc,
                    recipe_id=recipe_id_classic,
                )
            else:
                df_real = pd.read_csv("data/recipe_partial.csv")
                svm_loaded_model = load("models/svd_model.joblib")
                suggest = set(df_real.index)
                suggest_list = list(suggest)
                suggest_list_df = df_real.loc[suggest_list]
                prediction_of_recipe_cat = []
                for index, rows in suggest_list_df.iterrows():
                    prediction_of_recipe_cat.append(
                        svm_loaded_model.predict(uid=knn_user_id, iid=suggest_list_df.loc[index]["RecipeId"])
                    )
                recipe_db_id = []
                for i in range(len(prediction_of_recipe_cat)):
                    if prediction_of_recipe_cat[i].est > 4.9:
                        recipe_db_id.append(prediction_of_recipe_cat[i].iid)
                predicted_result_list = df_real.loc[recipe_db_id[:18]]
                recipe_name = list(predicted_result_list["Name"][:18])
                recipe_desc = list(predicted_result_list["Description"][:18])
                return_images_classic = []
                for i in list(predicted_result_list["Images"][:18]):
                    if "['NaN']" in i:
                        return_images_classic.append(
                            "https://www.food4fuel.com/wp-content/uploads/woocommerce-placeholder-600x600.png"
                        )
                    else:
                        return_images_classic.append(i.replace("['", "").replace("']", "").split("',")[0])
                recipe_image = return_images_classic
                recipe_id_classic = list(predicted_result_list["RecipeId"][:18])
                return render_template(
                    "food_list.html",
                    recipe_name=recipe_name,
                    recipe_image=recipe_image,
                    recipe_desc=recipe_desc,
                    recipe_id=recipe_id_classic,
                )
    else:
        db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
        popular_cursor = db.cursor()
        selectquery_popular = "SELECT * FROM popular3 limit 21"
        final_popular_display = pd.read_sql(selectquery_popular, db)
        df_real = pd.read_csv("data/popular4.csv")
        pop_recipe_details = df_real.loc[final_popular_display["popular_recipe_id"].index]
        pop_recipe_name = list(pop_recipe_details["recipe_name"][:18])
        pop_recipe_desc = list(pop_recipe_details["recipe_desc"][:18])
        recipe_image = list(pop_recipe_details["recipe_images"][:18])
        pop_recipe_id = list(pop_recipe_details["popular_recipe_id"][:18])
        return render_template(
            "food_list_popular.html",
            recipe_name=pop_recipe_name,
            recipe_image=recipe_image,
            recipe_desc=pop_recipe_desc,
            recipe_id=pop_recipe_id,
        )
    # except:
    #     db_1=mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword,database=database_name)
    #     popular_cursor = db.cursor()
    #     selectquery_popular_1 ="SELECT * FROM popular3"
    #     final_popular_display_1 = pd.read_sql(selectquery_popular_1, db_1)
    #     df_real_1 = pd.read_csv("data/popular4.csv")
    #     pop_recipe_details_1 = df_real_1.loc[final_popular_display_1["popular_recipe_id"].index]
    #     pop_recipe_name_1 = list(pop_recipe_details_1["recipe_name"][:18])
    #     pop_recipe_desc_1 = list(pop_recipe_details_1["recipe_desc"][:18])
    #     recipe_image_1 = list(pop_recipe_details_1["recipe_images"][:18])
    #     pop_recipe_id_1 = list(pop_recipe_details_1["popular_recipe_id"][:18])
    #     return render_template("food_list_popular.html", recipe_name=pop_recipe_name_1, recipe_image=recipe_image_1, recipe_desc=pop_recipe_desc_1, recipe_id=pop_recipe_id_1)


@app.route("/reinforcement/epsilon")
def reinforcement_algo():
    dataset_relative_path = "data/interactions_train.csv"
    predictions = epsilon_greedy(dataset_path=dataset_relative_path)
    pred_json_data = predictions.to_json()
    return pred_json_data


@app.route("/items/<item_number>")
def description(item_number):
    # write db query to retrieve details about items
    db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
    mycursor_analysis = db.cursor()
    clicks = "1"
    selectquery_analysis = "INSERT INTO `food_schema`.`analysis` (`recipe_id`, `clicks`) VALUES (%s, %s);"
    val_analysis = (item_number, clicks)
    mycursor_analysis.execute(selectquery_analysis, val_analysis)
    db.commit()
    mycursor_analysis.close()
    db.close()

    if session["using_svd"]:
        print("usnig other params")
        df_real = pd.read_csv("data/svd_partial.csv")
        # df_r = pd.read_csv("data/recipes.csv")
        rec_desc = df_real[df_real["RecipeId"] == int(item_number)]
        recipe_desc_name = rec_desc["Name"].values[0]
        recipe_desc_author = rec_desc["AuthorName"].values[0]
        recipe_desc_category = rec_desc["RecipeCategory"].values[0]
        recipe_desc_review_count = rec_desc["ReviewCount"].values[0]
        recipe_desc_instructions = rec_desc["RecipeInstructions"].values[0]
        recipe_desc_cookTime = rec_desc["CookTime"].values[0]
        recipe_desc_agg_rating = rec_desc["AggregatedRating"].values[0]
        recipe_desc_dop = rec_desc["DatePublished"].values[0]
        recipe_desc_img = rec_desc["Images"].values[0]
    else:
        df_real = pd.read_csv("data/recipe_partial.csv")
        # df_r = pd.read_csv("data/recipes.csv")
        rec_desc = df_real[df_real["RecipeId"] == int(item_number)]
        recipe_desc_name = rec_desc["Name"].values[0]
        recipe_desc_author = rec_desc["AuthorName"].values[0]
        recipe_desc_category = rec_desc["RecipeCategory"].values[0]
        recipe_desc_review_count = rec_desc["ReviewCount"].values[0]
        recipe_desc_instructions = rec_desc["RecipeInstructions"].values[0]
        recipe_desc_cookTime = rec_desc["CookTime"].values[0]
        recipe_desc_agg_rating = rec_desc["AggregatedRating"].values[0]
        recipe_desc_dop = rec_desc["DatePublished"].values[0]
        recipe_desc_img = rec_desc["Image"].values[0]

    from app_rec_sys.knn import knn_inference

    user_id = 2046
    item_id = item_number

    session["using_svd"] == False
    similar_df = knn_inference(
        model_load_path="models/knn_model.joblib", knn_user_id=user_id, item_id=item_id, output_values=12
    )

    sim_id = list(similar_df["RecipeId"])
    sim_names = list(similar_df["Name"])
    sim_review_count = list(similar_df["ReviewCount"])
    sim_ratings = list(similar_df["AggregatedRating"])
    sim_dop = list(similar_df["DatePublished"])

    return render_template(
        "food_desc.html",
        item_number=item_number,
        recipe_desc_name=recipe_desc_name,
        recipe_desc_author=recipe_desc_author,
        recipe_desc_category=recipe_desc_category,
        recipe_desc_review_count=recipe_desc_review_count,
        recipe_desc_instructions=recipe_desc_instructions,
        recipe_desc_cookTime=recipe_desc_cookTime,
        recipe_desc_agg_rating=recipe_desc_agg_rating,
        recipe_desc_dop=recipe_desc_dop,
        sim_names=sim_names,
        sim_review_count=sim_review_count,
        sim_ratings=sim_ratings,
        sim_dop=sim_dop,
        sim_id=sim_id,
        recipe_image=recipe_desc_img,
    )


@app.route("/item/<item_number>")
def description_other(item_number):
    # write db query to retrieve details about items
    db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
    mycursor_analysis = db.cursor()
    clicks = "1"
    selectquery_analysis = "INSERT INTO `food_schema`.`analysis` (`recipe_id`, `clicks`) VALUES (%s, %s);"
    val_analysis = (item_number, clicks)
    mycursor_analysis.execute(selectquery_analysis, val_analysis)
    db.commit()
    mycursor_analysis.close()
    db.close()

    df_real_1 = pd.read_csv("data/recipe_partial.csv")
    print(df_real_1.loc[df_real_1["RecipeId"] == int(item_number)])
    # df_r = pd.read_csv("data/recipes.csv")
    rec_desc = df_real_1[df_real_1["RecipeId"] == int(item_number)]
    print(rec_desc)
    recipe_desc_name = rec_desc["Name"].values[0]
    recipe_desc_author = rec_desc["AuthorName"].values[0]
    recipe_desc_category = rec_desc["RecipeCategory"].values[0]
    recipe_desc_review_count = rec_desc["ReviewCount"].values[0]
    recipe_desc_instructions = rec_desc["RecipeInstructions"].values[0]
    recipe_desc_cookTime = rec_desc["CookTime"].values[0]
    recipe_desc_agg_rating = rec_desc["AggregatedRating"].values[0]
    recipe_desc_dop = rec_desc["DatePublished"].values[0]
    recipe_desc_img = rec_desc["Images"].values[0]

    from app_rec_sys.knn import knn_inference

    user_id = 2046
    item_id = item_number

    session["using_svd"] == False
    similar_df = knn_inference(
        model_load_path="models/knn_model.joblib", knn_user_id=user_id, item_id=item_id, output_values=12
    )

    sim_id = list(similar_df["RecipeId"])
    sim_names = list(similar_df["Name"])
    sim_review_count = list(similar_df["ReviewCount"])
    sim_ratings = list(similar_df["AggregatedRating"])
    sim_dop = list(similar_df["DatePublished"])

    return render_template(
        "food_desc.html",
        item_number=item_number,
        recipe_desc_name=recipe_desc_name,
        recipe_desc_author=recipe_desc_author,
        recipe_desc_category=recipe_desc_category,
        recipe_desc_review_count=recipe_desc_review_count,
        recipe_desc_instructions=recipe_desc_instructions,
        recipe_desc_cookTime=recipe_desc_cookTime,
        recipe_desc_agg_rating=recipe_desc_agg_rating,
        recipe_desc_dop=recipe_desc_dop,
        sim_names=sim_names,
        sim_review_count=sim_review_count,
        sim_ratings=sim_ratings,
        sim_dop=sim_dop,
        sim_id=sim_id,
        recipe_image=recipe_desc_img,
    )


@app.route("/items/popular/<item_number>")
def description_popular(item_number):
    # write db query to retrieve details about items
    # db=mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword,database=database_name)
    # mycursor_analysis = db.cursor()
    # clicks = "1"
    # selectquery_analysis="INSERT INTO `food_schema`.`analysis` (`recipe_id`, `clicks`) VALUES (%s, %s);"
    # val_analysis = (item_number, clicks)
    # mycursor_analysis.execute(selectquery_analysis, val_analysis)

    df_r = pd.read_csv("data/popular4.csv")
    rec_desc = df_r[df_r["popular_recipe_id"] == int(item_number)]
    recipe_desc_name = rec_desc["recipe_name"].values[0]
    recipe_desc_author = rec_desc["AuthorName"].values[0]
    recipe_desc_category = rec_desc["RecipeCategory"].values[0]
    recipe_desc_review_count = rec_desc["recipe_orders"].values[0]
    recipe_desc_instructions = rec_desc["recipe_instructions"].values[0]
    recipe_desc_cookTime = rec_desc["CookTime"].values[0]
    recipe_desc_agg_rating = rec_desc["recipe_ratings"].values[0]
    recipe_desc_dop = rec_desc["recipe_dob"].values[0]
    recipe_desc_image = rec_desc["recipe_images"].values[0]

    from app_rec_sys.knn import knn_inference

    user_id = 2046
    item_id = item_number

    similar_df = knn_inference(
        model_load_path="models/knn_model.joblib", knn_user_id=user_id, item_id=item_id, output_values=12
    )

    sim_id = list(similar_df["RecipeId"])
    sim_names = list(similar_df["Name"])
    sim_review_count = list(similar_df["ReviewCount"])
    sim_ratings = list(similar_df["AggregatedRating"])
    sim_dop = list(similar_df["DatePublished"])

    return render_template(
        "food_desc_popular.html",
        item_number=item_number,
        recipe_desc_name=recipe_desc_name,
        recipe_desc_author=recipe_desc_author,
        recipe_desc_category=recipe_desc_category,
        recipe_desc_review_count=recipe_desc_review_count,
        recipe_desc_instructions=recipe_desc_instructions,
        recipe_desc_cookTime=recipe_desc_cookTime,
        recipe_desc_agg_rating=recipe_desc_agg_rating,
        recipe_desc_dop=recipe_desc_dop,
        sim_names=sim_names,
        sim_review_count=sim_review_count,
        sim_ratings=sim_ratings,
        sim_dop=sim_dop,
        sim_id=sim_id,
        recipe_image=recipe_desc_image,
    )


@app.route("/items/reinforcement/<item_number>")
def description_reinforcement(item_number):
    # write db query to retrieve details about items
    db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
    mycursor_analysis = db.cursor()
    clicks = "1"
    selectquery_analysis = "INSERT INTO `food_schema`.`analysis` (`recipe_id`, `clicks`) VALUES (%s, %s);"
    val_analysis = (item_number, clicks)
    mycursor_analysis.execute(selectquery_analysis, val_analysis)

    db_rein = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
    rein_sql = (
        "select * from reinforcement_recsys where user_id=" + str(session["user_id"]) + " and recipe_id=" + item_number
    )
    rec_desc = pd.read_sql(rein_sql, con=db_rein)

    recipe_desc_name = rec_desc["recipe_name"].values[0]
    recipe_desc_author = rec_desc["recipe_author"].values[0]
    recipe_desc_category = rec_desc["recipe_category"].values[0]
    recipe_desc_review_count = rec_desc["recipe_review_count"].values[0]
    recipe_desc_instructions = rec_desc["recipe_instructions"].values[0]
    recipe_desc_cookTime = rec_desc["recipe_cookTime"].values[0]
    recipe_desc_agg_rating = rec_desc["recipe_agg_rating"].values[0]
    recipe_desc_dop = rec_desc["recipe_dop"].values[0]
    recipe_desc_image = rec_desc["recipe_image"].values[0]

    from app_rec_sys.knn import knn_inference

    user_id = 2046
    item_id = item_number

    similar_df = knn_inference(
        model_load_path="models/knn_model.joblib", knn_user_id=user_id, item_id=item_id, output_values=12
    )

    sim_id = list(similar_df["RecipeId"])
    sim_names = list(similar_df["Name"])
    sim_review_count = list(similar_df["ReviewCount"])
    sim_ratings = list(similar_df["AggregatedRating"])
    sim_dop = list(similar_df["DatePublished"])

    return render_template(
        "food_desc_reinforcement.html",
        item_number=item_number,
        recipe_desc_name=recipe_desc_name,
        recipe_desc_author=recipe_desc_author,
        recipe_desc_category=recipe_desc_category,
        recipe_desc_review_count=recipe_desc_review_count,
        recipe_desc_instructions=recipe_desc_instructions,
        recipe_desc_cookTime=recipe_desc_cookTime,
        recipe_desc_agg_rating=recipe_desc_agg_rating,
        recipe_desc_dop=recipe_desc_dop,
        sim_names=sim_names,
        sim_review_count=sim_review_count,
        sim_ratings=sim_ratings,
        sim_dop=sim_dop,
        sim_id=sim_id,
        recipe_image=recipe_desc_image,
    )


@app.route("/admin_dashboard")
def admin_page():
    db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
    clicks_cursor = db.cursor()
    click_sql = "select count(recipe_id), recipe_id from analysis group by recipe_id;"
    clicks_cursor.execute(click_sql)
    clicks_list = clicks_cursor.fetchall()
    show_recipe_id = []
    show_clicks = []
    for i in range(len(clicks_list)):
        show_recipe_id.append(clicks_list[i][1])
        show_clicks.append(clicks_list[i][0])
    return render_template("charts.html", show_recipe_id=show_recipe_id, show_clicks=show_clicks)
    # return render_template("tableau.html")


@app.route("/deep_analysis_dashboard", methods=["GET", "POST"])
def deep_analysis():
    return render_template("tableau.html")


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
        logincursor = db.cursor()
        validation = (
            "select * from admin_users where email='"
            + request.form["formusername"]
            + "' AND password='"
            + request.form["formpassword"]
            + "'"
        )
        logincursor.execute(validation)
        verification = logincursor.fetchall()
        if len(verification) != 0:
            session["key"] = "values"
            session["logged_in"] = True
            session["username"] = request.form["formusername"]
            flash("Logged In Successfully")
            db.commit()
            logincursor.close()
            db.close()
            return redirect(url_for("admin_page"))
        else:
            logincursor.close()
            db.close()
            return render_template("admin_login.html", error="Invalid Credentials")
    return render_template("admin_login.html")


@app.route("/api/epsilon")
def epsilon_api():
    # Integrate a iframe to show power BI dashboard
    r = requests.get("http://127.0.0.1:5000/reinforcement/epsilon")
    read_pred_data = r.json()
    extract_recipes = read_pred_data["recipe_id"]
    recipe_ids = []
    for data in extract_recipes:
        recipe_ids.append(read_pred_data["recipe_id"][data])

    df_rec = pd.read_csv("data/recipes.csv")

    final_display = df_rec.loc[df_rec["RecipeId"].isin(recipe_ids)]

    print(recipe_ids)
    return render_template("test.html", recipeId=final_display["RecipeId"])


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
        logincursor = db.cursor()
        validation = (
            "select * from users where email='"
            + request.form["formusername"]
            + "' AND password='"
            + request.form["formpassword"]
            + "'"
        )
        logincursor.execute(validation)
        verification = logincursor.fetchall()
        if len(verification) != 0:
            session["key"] = "values"
            session["logged_in"] = True
            session["username"] = request.form["formusername"]
            session["user_id"] = verification[0][0]
            flash("Logged In Successfully")
            db.commit()
            logincursor.close()
            db.close()
            return redirect(url_for("home_page"))
        else:
            logincursor.close()
            db.close()
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session["logged_in"] = False
    return redirect(url_for("login_page"))


@app.route("/from_fridge", methods=["GET", "POST"])
def fridge_food():
    return render_template("fridge.html")


@app.route("/from_fridge/<user_id>", methods=["GET", "POST"])
def recipe_filter_fridge(user_id):
    from joblib import load, dump

    df_test = pd.read_csv("data/recipe_partial.csv")

    # df_test = df_test.drop("Unnamed: 0", axis=1)
    user_form_input = request.form["search_tags"]

    user_list_final = user_form_input.split(",")

    user_list = ["blueberries", "lemon juice", "vanilla yogurt", "sugar"]

    # user_list = user_list_final
    count_list = []
    for each_item in range(len(df_test["RecipeIngredientParts"])):
        values = df_test["RecipeIngredientParts"][each_item].replace("'", "").replace("[", "").replace("]", "")
        final_list = values.split(", ")

        true_count = 0
        for i in user_list:
            if i in final_list:
                true_count += 1
        if true_count >= 2:
            count_list.append(df_test.loc[each_item])

    df_return = pd.DataFrame(data=count_list)
    # df_return
    knn_user_id = int(user_id)
    svd_loaded_model = load("models/svd_model.joblib")

    prediction_of_recipe_cat = []
    for index, rows in df_return.iterrows():
        prediction_of_recipe_cat.append(svd_loaded_model.predict(uid=knn_user_id, iid=df_return.loc[index]["RecipeId"]))

    predicted_recipe_id = []
    for i in range(len(prediction_of_recipe_cat)):
        if prediction_of_recipe_cat[i].est > 4.7:
            predicted_recipe_id.append(prediction_of_recipe_cat[i].iid)
    print(df_return)
    predicted_result_list_filter = df_return.loc[df_return.index.intersection(predicted_recipe_id)]
    print(predicted_result_list_filter)
    sim_id = list(predicted_result_list_filter["RecipeId"])
    sim_names = list(predicted_result_list_filter["Name"])
    sim_review_count = list(predicted_result_list_filter["ReviewCount"])
    sim_ratings = list(predicted_result_list_filter["AggregatedRating"])
    sim_dop = list(predicted_result_list_filter["DatePublished"])

    return render_template(
        "fridge_results.html",
        sim_names=sim_names,
        sim_review_count=sim_review_count,
        sim_ratings=sim_ratings,
        sim_dop=sim_dop,
        sim_id=sim_id,
    )


@app.route("/register")
def display_reg_page():
    return render_template("register.html")


@app.route("/register_user", methods=["GET", "POST"])
def user_register():
    if request.method == "POST":
        user_first_name = request.form["firstName"]
        user_last_name = request.form["lastName"]
        user_dob = request.form["birthdayDate"]
        user_gender = "male"
        user_email = request.form["emailAddress"]
        user_password = request.form["typePassword"]
        user_address = request.form["userAddress"]
        flag = 0
        while True:
            if len(user_password) < 8:
                flag = -1
                break
            elif not re.search("[a-z]", user_password):
                flag = -1
                break
            # elif not re.search("[A-Z]", newpassword):
            #     flag = -1
            #     break
            elif not re.search("[0-9]", user_password):
                flag = -1
                break
            else:
                flag = 0
                break
        if flag == -1:
            return redirect(url_for("register"))
        else:
            db = mysql.connector.connect(host=hostname, user=dbusername, passwd=dbpassword, database=database_name)
            mycursor = db.cursor()
            sql = "INSERT INTO `food_schema`.`users` (`email`, `password`, `firstname`, `lastname`, `gender`, `dob`, `address`) VALUES (%s, %s, %s, %s, %s, %s, %s);"
            val = (user_email, user_password, user_first_name, user_last_name, user_gender, user_dob, user_address)
            mycursor.execute(sql, val)
            db.commit()
            mycursor.close()
            db.close()
            if mycursor.rowcount > 0:
                return redirect(url_for("login_page"))


@app.route("/developer_api")
def developer_api_page():
    return "<h1> Welcome to developer API page</h1>"


@app.route("/find_resto")
def resto_locator():
    # To be integrated with Bing map API data
    return "<h1> Locating nearest resto.. </h1> "


@app.route("/add_recipe")
def recipe_addition():
    # Create a form page to add recipes
    # write db query to insert data to db

    return "<h1> Add new recipes </h1>"


@app.route("/user_navigation_page")
def map_navigation_bing():
    # integrate Bing map API to get navigate the from user location to resto location
    return "<h1> Using bing map to navigate to resto </h1>"
