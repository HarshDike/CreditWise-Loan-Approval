import nbformat
from nbclient import NotebookClient


notebook = nbformat.read("credit_wise.ipynb", as_version=4)
notebook.cells.append(
    nbformat.v4.new_code_cell(
        """import pickle

artifact = {
    "model": log_model,
    "model_name": "feature_engineered_logistic_regression",
    "scaler": scaler,
    "one_hot_encoder": ohe,
    "numeric_imputer": num_imp,
    "categorical_imputer": cat_imp,
    "target_label_encoder": le,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": cols,
    "engineered_features": {
        "DTI_Ratio": "DTI_Ratio_sq",
        "Credit_Score": "Credit_Score_sq",
    },
    "test_accuracy": 0.875,
    "test_f1": 0.7967479674796748,
}

with open("credit_wise_model.pkl", "wb") as file:
    pickle.dump(artifact, file)

print("Saved credit_wise_model.pkl")
"""
    )
)

NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": "."}},
).execute()
