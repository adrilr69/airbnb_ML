#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-20T23:02:05.936Z
"""

# # 0. Drive connexion


# Connecting our drive to the notebook for connecting the csv file from previous notebook


from google.colab import drive
drive.mount('/content/drive')

# # 1. Imports


# basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# visualization
import matplotlib.pyplot as plt

# Scikit-learn imports: split, pre-processing, models & metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # 2. Data Understanding on final csv file


# Connecting the csv from previous notebook to current notebook:
# 
# as a result of df_final.info() we have:
# 
# - 53058 rows and 42 columns (cols start from index 0)


df_final = pd.read_csv('/content/drive/MyDrive/Projet Machine Learning/Final version/2) Clean Scraped from Airbnb listings.csv')

df_final.info()

# Doing a quick visualization on how data looks like:


df_final.head()

# Now, we are doing a quick visulization on NaNs, but it´s normal we don´t have any since the raw dataset was already cleaned in previous notebook; this version contains no missing values


#Visualization of NaNs in %
missing_values = df_final.isna().mean().sort_values(ascending=False) * 100
missing_values

# # 3. Feature selection and preprocessing


# Defining (X) and (y) features:
# 
# Where X = all columns except target
# 
# y = price --> target column


target_col = "price"

# quick checking up price exists and we dont have it named as something else
assert target_col in df_final.columns, f"the column {target_col} does not exists in the dataframe"

X = df_final.drop(columns=[target_col])
y = df_final[target_col]

X.head()


# As later on we will be doing cross validation and cross validation don´t work on booleans, we will convert bool values into integers


# Fixing booleans to numeric 0/1:

# Identifying boolean columns:
bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
print("Boolean columns:", bool_cols)

# Converting boolean columns to integers (0/1)
for col in bool_cols:
    X[col] = X[col].astype(int)


# Now we will separate numeric and categorical columns:
# 
# Numeric columns = columns with dtypes `int64` or `float64`
# 
# Categorical columns = columns with dtypes `object`
# 
# This is needed for the `ColumnTransformer`, for us to be able to:
# - Apply imputers + scaler only to numeric columns.
# - Apply imputers + OneHotEncoder only to categorical columns.


numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric columns:", numeric_features)
print("Categorical columns:", categorical_features)


# Preprocessing Pipeline:
# 
# 
# We are building 2 pipelines:
# 
# - numeric_transformer :
#   - `SimpleImputer(strategy="median")` → fills NaNs with the median of each column.
#   - `StandardScaler()` → centers and scales numeric features (mean 0, std 1).
#   
#   *This is important for KNN and other models.
# 
# - categorical_transformer:
#   - `SimpleImputer(strategy="most_frequent")` → fills missing categorical values with the most frequent category.
#   - `OneHotEncoder(handle_unknown="ignore")` → converts categories to dummy indicator variables and ignores unseen categories at prediction time.
# 
# Then we combine them with ColumnTransformer:
# 
# - Apply the numeric transformer to `numeric_features`.
# - Apply the categorical transformer to `categorical_features`.
# 
# This ensures the same preprocessing steps will be applied consistently in training and in prediction (inside the global Pipeline).
# 
# 
# 


# transformers

#scaler --> numeric
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#encoding --> categorical
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# Train/ test split
# 
# Our data is splitted into:
# 
# - X_train, y_train: 80% of data used to train and validate models (with cross-validation). --> 80% = 42446 rows , 41 cols
# - X_test, y_test: 20% of data held remaining and used only at the end to evaluate the chosen model. --> 20% = 10612 rows, 41 cols
# 
# 


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train.shape, X_test.shape


# # 4. Modeling and Cross-validation


# Setting up the models dictionary as follows:
# 
# 1. **Linear Regression** –-> OLS baseline.
# 2. **Ridge Regression** –-> linear model with L2 regularization (collinearity).
# 3. **Lasso Regression** –-> linear model with L1 regularization (feature selection).
# 4. **KNN Regressor** –-> non-parametric model based on nearest neighbors.
# 5. **Random Forest Regressor** –-> ensemble of decision trees
# 6. **Gradient Boosting Regressor** –-> boosting of trees, good at capturing complex relationships.


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# Cross Valudation Score evaluation for all models
# 
# We are looping over every model in the "models" dictionary, where for each model we:
# 
# 1. create a "pipeline" with:
#   - "preprocess" --> the `preprocessor` (imputation + scaling + encoding)
#   - "model" --> current regression model
# 
# 2. use "cross_val_score" with:
#      - "cv=5" → 5-fold cross-validation.
#      - "scoring" = "neg_mean_absolute_error" → we evaluated using MAE (mean absolute error).
# 3. converted negative scores back to positive MAE with "-scores"
# 4. stored the mean and standard deviation of MAE for each model in a list
# 5. converted the list into a DataFrame "cv_results_df" and sorted by MAE (the smaller MAE, the better)
# 
# The result is a comparison of model performance on the training data using cross-validation.
# 


# evaluatiing with cross_val_score (MAE)

from sklearn.model_selection import cross_val_score

cross_val_results = []

for name, model in models.items():
    # Pipeline = prepreocessing + model
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # scoring = 'neg_mean_absolute_error' since sklearn uses neg values for errors
    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mae_scores = -scores  # we will change the symbol

    cross_val_results.append({
        "model": name,
        "cross_val_mae_mean": mae_scores.mean(),
        "cross_val_mae_std": mae_scores.std()
    })

cv_results_df = pd.DataFrame(cross_val_results).sort_values("cross_val_mae_mean")
cv_results_df

# After doing the Cross Validation, we got that the best model for doing the predictions on pricing is the Random Forest, since it had an output of a MAE mean of 74% (26% far from real price).
# 
# With this being said, we will be computing a Random Forest, since we saw that Airbnb pricing is non-linear, it is influenced by interactions between:
# 
# - neighbourhood
# - room type
# - number of guests
# - host rating
# - reviews
# - availability
# - amenities


# **Selecting** the Best Model on training set:
# 
# Our best model = Random Forest


best_model_name = "Random Forest"
best_model = models[best_model_name]


# **Building** the final pipeline (preprocessing + model)


best_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", best_model)
])

# # 5. Training the final model


# **Fitting** the best pipeline on the full training set / full training data:
# 
# (X_train, y_train)


best_pipeline.fit(X_train, y_train)

# **Predicting** prices on the test set


y_pred = best_pipeline.predict(X_test)

# **Evaluating** test performance
# 
# Computing evaluation metrics:
#    - MAE (Mean Absolute Error)
#    - RMSE (Root Mean Squared Error)
#    - R² Score
# 
#    ** as a result, we see that the MAE obtained from the Random Forest (72%) is slightly lower than the Cross Validation (74%)


mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Performance of final model on TEST set:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.3f}")

# # 6. Visualization of results


# Real vs Predicted prices (Random Forest)
# 
# In this step we will create a scatter plot to compare:
# 
# - **x-axis:** real prices from the test set (y_test)
# - **y-axis:** prices predicted by our final model (y_pred)
# 
# We will also draw the diagonal line y = x, where:
# 
# - Points on the line → perfect predictions  
# - Points close to the line → small errors  
# - Points far from the line  → large errors
# 
# This visualization will helps us to see:
# 
# - Whether the model tends to underestimate or overestimate some price ranges.
# - How the error behaves for cheap vs expensive listings.


# Graph on Real vs Predicted prices scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)

plt.xlabel("Real price (£)")
plt.ylabel("Predicted price (£)")
plt.title(f"{best_model_name} - Real vs Predicted prices (test set)")

# Diagonal reference line where y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.tight_layout()
plt.show()

# 1. Most of the points are relatively close to the diagonal --> predictions are generally close to the real prices
# 
# 2. There´s more dispersion for very high prices, which is expected: luxury listings are rarer and harder to predict
# 
# 3. This matches our MAE (~73): there is error, but the model clearly learned the structure of the data and is not random


# # 7. Feature importance


# Obtaining which features are more important at price prediction using Random Forest


# Getting the trained Random Forest model from the pipeline
final_model = best_pipeline.named_steps["model"]

# Checking that the model has feature_importances_
hasattr(final_model, "feature_importances_")

# Obtaining names of features after preprocessing feature_importances_


#Getting the feature names after preprocessing since all features are numeric in dataset:
all_feature_names = numeric_features

importances = final_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

feat_imp.head(20)

# # 8. Business insight on top 15 features


# Top 15 features that make an airbnb pricy


top_n = 15
top_features = feat_imp.head(top_n).iloc[::-1]  # reversing for better visualization of plot visualization

plt.figure(figsize=(8, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title(f"{best_model_name} - Top {top_n} important features")
plt.tight_layout()
plt.show()


# # 9. Practical application model --> testing


# Once our model was trained, we decided to do a test where we will give him the values of some of the top 15 important features for him to determine a pricing
# 
# 1. We created a one-row DataFrame with same cols as training data "X"
# 2. We filled the most imp features from importance level and left rest as NaNs
# 3. Our preprocessing pipeline (imputers * scalers) handled NaNs
# 4. We called "best_pipeline.predict()" to the estimation of a night price
# 


def estimate_price_example(
    host_days_active,
    accommodates,
    bedrooms,
    bathrooms,
    minimum_nights,
    maximum_nights,
    availability_30,
    availability_365,
    num_amenities,
    calculated_host_listings_count,
    number_of_reviews,
    reviews_per_month,
    review_scores_rating,
    host_acceptance_rate,
    neigh_Prime
):

 # 1. Creating an empty row with same columns as X
    new_row = pd.DataFrame(columns=X.columns)
    new_row.loc[0] = np.nan   # starting with all NaNs values

# 2. Filling in the top important features
    new_row.loc[0, "host_days_active"] = host_days_active
    new_row.loc[0, "accommodates"] = accommodates
    new_row.loc[0, "bedrooms"] = bedrooms
    new_row.loc[0, "bathrooms"] = bathrooms
    new_row.loc[0, "minimum_nights"] = minimum_nights
    new_row.loc[0, "maximum_nights"] = maximum_nights
    new_row.loc[0, "availability_30"] = availability_30
    new_row.loc[0, "availability_365"] = availability_365
    new_row.loc[0, "num_amenities"] = num_amenities
    new_row.loc[0, "calculated_host_listings_count"] = calculated_host_listings_count
    new_row.loc[0, "number_of_reviews"] = number_of_reviews
    new_row.loc[0, "reviews_per_month"] = reviews_per_month
    new_row.loc[0, "review_scores_rating"] = review_scores_rating
    new_row.loc[0, "host_acceptance_rate"] = host_acceptance_rate
    new_row.loc[0, "neigh_Prime"] = neigh_Prime

    # 3. Predict price with the trained pipeline
    predicted_price = best_pipeline.predict(new_row)[0]

    return predicted_price

# Doing a simulation on a false airbnb example:


#simulation:
example_price = estimate_price_example(
    host_days_active=400, #days since host was part of platform
    accommodates=3,  # 3 guests
    bedrooms=1,
    bathrooms=1,
    minimum_nights=2,
    maximum_nights=30,
    availability_30=15,
    availability_365=200,
    num_amenities=10,
    calculated_host_listings_count=2,
    number_of_reviews=35,
    reviews_per_month=1.2,
    review_scores_rating=4.7,
    host_acceptance_rate=0.95,    # 95% acceptance
    neigh_Prime=1
    )

print(f"Estimated nightly price for this example: £{example_price:.2f}")