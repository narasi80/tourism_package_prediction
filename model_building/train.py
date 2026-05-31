
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
# for model training, tuning, and evaluation
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("VisitWithUs_WellnessPackage_Classifier")

api = HfApi()

Xtrain_path = "hf://datasets/narasi80/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/narasi80/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/narasi80/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/narasi80/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define numeric and categorical features
numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'DurationOfPitch',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'CityTier'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Preprocessor with imputation
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features),
    remainder='drop'
)

# XGBoost Classifier
xgb_clf = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 300],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.6, 0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0, 0.1, 1, 10],
    'xgbclassifier__scale_pos_weight': [1, 5, 10]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_clf)

with mlflow.start_run():
    # Randomized Search
    randomized_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring='roc_auc',
        random_state=42,
        verbose=1
    )
    randomized_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = randomized_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Best model
    mlflow.log_params(randomized_search.best_params_)
    best_model = randomized_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Probabilities for ROC-AUC
    y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    # Classification Metrics
    train_roc_auc = roc_auc_score(ytrain, y_proba_train)
    test_roc_auc = roc_auc_score(ytest, y_proba_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)

    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)

    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

    # Save the model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "narasi80/tourism_package_model"
    repo_type = "model"

    # Check if repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(f"Best parameters: {randomized_search.best_params_}")
print(f"Best CV ROC-AUC: {randomized_search.best_score_:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test F1: {test_f1:.4f}")
