# for data manipulation
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
    accuracy_score
)
# for model serialization
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("VisitWithUs_WellnessPackage_Classifier")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/Narasi/tourism_package_prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Narasi/tourism_package_prediction/Xtest.csv"
ytrain_path = "hf://datasets/Narasi/tourism_package_prediction/ytrain.csv"
ytest_path = "hf://datasets/Narasi/tourism_package_prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'DurationOfPitch',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome', 'CityTier'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

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

xgb_clf = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 300],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.6, 0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0, 0.1, 1, 10],
    'xgbclassifier__scale_pos_weight': [1, 5, 10]
}

model_pipeline = make_pipeline(preprocessor, xgb_clf)

with mlflow.start_run():
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

    results = randomized_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_roc_auc", results['mean_test_score'][i])

    mlflow.log_params(randomized_search.best_params_)
    best_model = randomized_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    mlflow.log_metrics({
        "train_roc_auc": roc_auc_score(ytrain, y_proba_train),
        "test_roc_auc": roc_auc_score(ytest, y_proba_test),
        "train_f1": f1_score(ytrain, y_pred_train),
        "test_f1": f1_score(ytest, y_pred_test),
        "train_precision": precision_score(ytrain, y_pred_train, zero_division=0),
        "test_precision": precision_score(ytest, y_pred_test, zero_division=0),
        "train_recall": recall_score(ytrain, y_pred_train),
        "test_recall": recall_score(ytest, y_pred_test),
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test)
    })

    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved: {model_path}")

    repo_id = "Narasi/tourism_package_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(f"Best parameters: {randomized_search.best_params_}")
print(f"Best CV ROC-AUC: {randomized_search.best_score_:.4f}")
