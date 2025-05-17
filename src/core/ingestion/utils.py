import pandas as pd

# from prefect import task
from dotenv import load_dotenv
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

load_dotenv()


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load data from CSV file into a DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataframe from {filepath}: {str(e)}")
        raise


# @task(name="Process raw data")
def process_dataframe(
    dataframe: pd.DataFrame, target_col: str, drop_cols: list = None
) -> pd.DataFrame:
    """Clean and preprocess the dataframe for analysis."""
    try:
        df = dataframe.copy()

        # Standardize column names
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        target_col = target_col.lower()

        # Normalize categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].str.replace(" ", "_").str.lower()

        # Drop specified columns
        if drop_cols:
            drop_cols = [col.lower() for col in drop_cols]
            df = df.drop(drop_cols, axis=1)

        # Convert totalcharges to float
        if "totalcharges" in df.columns:
            df = df[df["totalcharges"] != "_"]
            df["totalcharges"] = df["totalcharges"].astype("float64")

        # Convert target column to binary
        if target_col in df.columns:
            df["churn"] = (df[target_col] == "yes").astype(int)
            if target_col != "churn":
                df = df.drop(columns=[target_col])

        return df

    except Exception as e:
        print(f"Error processing dataframe: {str(e)}")
        raise


def process_streamlit_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    handling_missing: str = "drop",
    encoding: str = "none",
    scaling: str = "none",
    job_id: str = None,
    drop_cols: list = None,
) -> pd.DataFrame:
    """Cleans and preprocesses dataframe based on user configurations."""
    df = dataframe.copy()

    # Standardize columns
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    target_column = target_column.lower()

    if job_id:
        df = df[df["job_id"].astype(str).str.lower() == job_id.lower()]

    if df.empty:
        raise ValueError("No data found for the given job_id.")

    if drop_cols:
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Normalize categorical values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")

    # Handle missing values
    if handling_missing == "drop":
        df.dropna(inplace=True)
    elif handling_missing in ["mean", "median", "mode", "constant"]:
        for col in df.columns:
            if df[col].isnull().any():
                if handling_missing == "mean" and df[col].dtype != object:
                    df[col].fillna(df[col].mean(), inplace=True)
                elif handling_missing == "median" and df[col].dtype != object:
                    df[col].fillna(df[col].median(), inplace=True)
                elif handling_missing == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif handling_missing == "constant":
                    df[col].fillna(
                        "missing" if df[col].dtype == object else 0, inplace=True
                    )
    else:
        raise ValueError(f"Invalid missing value strategy: {handling_missing}")

    # Infer target type and apply appropriate transformation
    if target_column in df.columns:
        target_series = df[target_column].dropna()
        unique_vals = target_series.unique()

        if pd.api.types.is_numeric_dtype(target_series):
            task_type = "regression"
            # No transformation needed for regression
        elif len(unique_vals) == 2:
            task_type = "binary"
            yes_values = {"yes", "true", "1"}
            df[target_column] = (
                target_series.astype(str)
                .str.lower()
                .map(lambda x: 1 if x in yes_values else 0)
            )
        elif 2 < len(unique_vals) <= 10:
            task_type = "multiclass"
            # Convert to category codes (label encoding)
            df[target_column] = target_series.astype("category").cat.codes
        else:
            task_type = (
                "regression"  # Assume continuous if many unique non-numeric values
            )
            # Optional: raise warning if string with many values

        print(f"Task type inferred: {task_type}")
    else:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")

    # Encode categorical features
    cat_cols = df.select_dtypes(include="object").columns.difference(["job_id"])
    if encoding == "label":
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
    elif encoding == "onehot":
        df = pd.get_dummies(df, columns=cat_cols)
    elif encoding == "target":
        for col in cat_cols:
            means = df.groupby(col)["churn"].mean()
            df[col] = df[col].map(means)
    elif encoding != "none":
        raise ValueError(f"Unknown encoding method: {encoding}")

    # Scale numeric features
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.difference(
        ["churn", "job_id"]
    )
    if scaling == "standard":
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    elif scaling == "minmax":
        df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    elif scaling == "robust":
        df[num_cols] = RobustScaler().fit_transform(df[num_cols])
    elif scaling != "none":
        raise ValueError(f"Unknown scaling method: {scaling}")

    return df
