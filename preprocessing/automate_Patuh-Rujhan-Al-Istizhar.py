from pathlib import Path

import pandas as pd
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropCorrelatedFeatures
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def run_data_preprocessing(data_input_path: str, output_directory: str):
    """
    Loads the diabetes dataset, performs preprocessing (scaling, outlier handling,
    correlation reduction, SMOTE), and saves the processed training and testing data.

    Args:
        data_input_path (str): Path to the raw diabetes CSV file.
        output_directory (str): Directory where processed CSV files will be saved.
    """
    print("--- Starting Data Preprocessing ---")

    # --- 1. Load Dataset ---
    try:
        df = pd.read_csv(data_input_path)
        print(
            f"Dataset loaded from '{data_input_path}' with {df.shape[0]} rows and {df.shape[1]} columns."
        )
    except FileNotFoundError:
        print(
            f"Error: File '{data_input_path}' not found. Please ensure the path is correct."
        )
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 2. Define Feature Categories for Preprocessing ---
    binary_features = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "HeartDiseaseorAttack",
        "Stroke",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
        "Sex",
    ]
    ordinal_features = ["GenHlth", "Age", "Education", "Income"]
    continuous_features = ["BMI", "MentHlth", "PhysHlth"]

    # --- 3. Feature Validation ---
    all_features_except_target = (
        binary_features + ordinal_features + continuous_features
    )
    if set(df.drop("Diabetes_012", axis=1).columns) != set(all_features_except_target):
        print(
            "Warning: Feature categories do not perfectly match DataFrame columns. Please review."
        )
    else:
        print("Feature validation: All non-target features are categorized.")

    # --- 4. Split Training and Testing Data ---
    X = df.drop("Diabetes_012", axis=1)
    y = df["Diabetes_012"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Data split: Training {X_train.shape}, Test {X_test.shape}")
    print(
        f"Initial Training target distribution:\n{y_train.value_counts(normalize=True).round(3)}"
    )

    # --- 5. Define Preprocessing Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous_pipeline",
                Pipeline(
                    [
                        (
                            "winsorize",
                            Winsorizer(
                                capping_method="iqr",
                                fold=1.5,
                                variables=continuous_features,
                            ),
                        ),
                        ("scaler", RobustScaler()),
                    ]
                ),
                continuous_features,
            ),
            ("ordinal_scaler", RobustScaler(), ordinal_features),
            ("binary_passthrough", "passthrough", binary_features),
        ],
        remainder="drop",
    )

    pipeline = ImbalancedPipeline(
        [
            ("preprocess", preprocessor),
            (
                "feature_selection",
                DropCorrelatedFeatures(threshold=0.8, method="pearson"),
            ),
            ("smote", SMOTE(random_state=42, sampling_strategy="auto")),
        ]
    )

    # --- 6. Apply Pipeline to Data ---
    print("\nApplying preprocessing and SMOTE to training data...")
    X_train_processed, y_train_processed = pipeline.fit_resample(X_train, y_train)

    X_test_processed = pipeline[:-1].transform(X_test)

    final_feature_names = pipeline.named_steps[
        "feature_selection"
    ].get_feature_names_out()

    print("Preprocessing complete.")
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed testing data shape: {X_test_processed.shape}")
    print(
        f"Training target distribution after SMOTE:\n{pd.Series(y_train_processed).value_counts(normalize=True).round(3)}"
    )
    print(f"Number of final features: {len(final_feature_names)}")

    # --- 7. Save Processed Data to CSV Files ---
    output_path_obj = Path(output_directory)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    train_final_df = pd.DataFrame(X_train_processed, columns=final_feature_names)
    train_final_df["Diabetes_012"] = y_train_processed

    test_final_df = pd.DataFrame(X_test_processed, columns=final_feature_names)
    test_final_df["Diabetes_012"] = y_test.reset_index(drop=True)

    train_final_df.to_csv(output_path_obj / "train_processed.csv", index=False)
    test_final_df.to_csv(output_path_obj / "test_processed.csv", index=False)

    print(f"\nProcessed data saved to '{output_path_obj}/'.")
    print(
        f"Train data saved: '{output_path_obj / 'train_processed.csv'}' ({train_final_df.shape})"
    )
    print(
        f"Test data saved: '{output_path_obj / 'test_processed.csv'}' ({test_final_df.shape})"
    )

    print("\n--- Data Preprocessing Finished ---")


if __name__ == "__main__":
    script_directory = Path(__file__).resolve().parent

    DATA_FILE = script_directory.parent / "diabetes_health_indicators_raw.csv"
    OUTPUT_DIR = script_directory / "diabetes_health_indicators_preprocessing"

    run_data_preprocessing(str(DATA_FILE), str(OUTPUT_DIR))
