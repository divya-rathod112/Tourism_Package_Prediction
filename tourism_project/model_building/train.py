import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Configure local tracking instance addresses
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction-marketing")

# Initialize Hugging Face API client using environment secrets
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

HF_USERNAME = "divyarathod112" # <-- Replace with your HF username
DATASET_REPO = f"{HF_USERNAME}/Tourism-Package-Prediction"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-prediction-model"

print("Fetching processing frames from Hugging Face Hub Layer...")
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").values.ravel()

# Define structural data features lists explicitly mapping schema properties
numeric_features = [
    'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

# Compute minority class balance scalar weight dynamically to fix conversion skewing
negative_cases = len(ytrain) - sum(ytrain)
positive_cases = sum(ytrain)
class_balancing_factor = negative_cases / positive_cases

# Build custom Preprocessing pipelines
preprocessor = make_column_transformer(
    (StandardScaler(), [col for col in numeric_features if col in Xtrain.columns]),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), [col for col in categorical_features if col in Xtrain.columns])
)

# Instantiate basic pipeline architectures incorporating XGBoost
base_xgb = xgb.XGBClassifier(
    scale_pos_weight=class_balancing_factor,
    random_state=42,
    eval_metric='logloss'
)
model_pipeline = make_pipeline(preprocessor, base_xgb)

# Set hyperparameter grid parameters
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
}

print("Initiating Grid Search Experiment execution loop...")
with mlflow.start_run(run_name="XGBoost_Optimization_Master"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Track cross-validation iteration scores inside nested tracking layers
    cv_results = grid_search.cv_results_
    for idx in range(len(cv_results['params'])):
        with mlflow.start_run(run_name=f"Iteration_{idx}", nested=True):
            mlflow.log_params(cv_results['params'][idx])
            mlflow.log_metric("mean_cv_f1", cv_results['mean_test_score'][idx])

    # Log global parameters and extract best model estimator configuration
    mlflow.log_params(grid_search.best_params_)
    best_pipeline = grid_search.best_estimator_

    # Generate and capture inference validation prediction summaries
    y_predictions = best_pipeline.predict(Xtest)
    metrics_report = classification_report(ytest, y_predictions, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy": metrics_report['accuracy'],
        "test_precision": metrics_report['1']['precision'],
        "test_recall": metrics_report['1']['recall'],
        "test_f1_score": metrics_report['1']['f1-score']
    })

    # Serialize model assets directly onto the disk execution area
    model_output_file = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_pipeline, model_output_file)
    mlflow.log_artifact(model_output_file, artifact_path="final_model")
    print(f"Successfully tracked model runs. Best F1 Score achieved: {metrics_report['1']['f1-score']:.4f}")

# Verify or setup Model Registry repository on Hugging Face Hub
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    print(f"Creating new model repo registry target: {MODEL_REPO}")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)

# Push finalized pipeline components to Hugging Face Model Space Hub
api.upload_file(
    path_or_fileobj=model_output_file,
    path_in_repo=model_output_file,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print("Production ML pipeline artifact cataloged and synchronized successfully!")
