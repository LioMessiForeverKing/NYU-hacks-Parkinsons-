import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


class ParkinsonDetectionModel:
    """
    Parkinson's Disease Detection using Voice Biomarkers
    Key Feature: Subject-Independent Cross-Validation (no data leakage)
    """

    def __init__(self, data_path="../parkinsons.data"):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.subjects = None
        self.model = None

    def extract_subject_id(self, name):
        """Convert 'phon_R01_S01_1' to 'S01' to group recordings by person"""
        try:
            parts = name.split("_")
            for part in parts:
                if part.startswith("S") and len(part) >= 3:
                    return part[:3]  # Return S01, S02, etc.
            return name
        except:
            return name

    def load_and_prepare_data(self):
        """
        STEP 1: Load data and extract subjects
        Critical: Group recordings by person to prevent data leakage
        """
        print("=" * 60)
        print("PARKINSON'S DISEASE DETECTION - ML ANALYSIS")
        print("=" * 60)

        # Load the CSV file
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded: {self.df.shape[0]} recordings, {self.df.shape[1]-2} features")

        # Extract subject IDs from names (CRITICAL for proper CV)
        self.df["subject_id"] = self.df["name"].apply(self.extract_subject_id)

        # Check data structure
        subject_counts = self.df["subject_id"].value_counts()
        target_counts = self.df["status"].value_counts()

        print(
            f"Subjects: {len(subject_counts)} people ({subject_counts.mean():.1f} recordings each)"
        )
        print(f"Labels: {target_counts[0]} healthy, {target_counts[1]} Parkinson's")

        # Prepare feature matrix and target vector
        feature_cols = [
            col
            for col in self.df.columns
            if col not in ["name", "status", "subject_id"]
        ]
        self.X = self.df[feature_cols].values
        self.y = self.df["status"].values
        self.subjects = self.df["subject_id"].values

        print(f"Ready: X={self.X.shape}, y={self.y.shape}")
        return feature_cols

    def setup_subject_independent_cv(self):
        """
        STEP 2: Setup Cross-Validation Strategy
        Key: Never put same person in both training and testing
        """
        print(f"\n" + "=" * 40)
        print("CROSS-VALIDATION SETUP")
        print("=" * 40)

        # Use 3-fold stratified group CV (maintains class balance + subject independence)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        # Validate no subject leakage
        print("Validating CV splits...")
        for i, (train_idx, test_idx) in enumerate(
            cv.split(self.X, self.y, groups=self.subjects)
        ):
            train_subjects = set(self.subjects[train_idx])
            test_subjects = set(self.subjects[test_idx])

            # Critical check: no overlap
            overlap = train_subjects.intersection(test_subjects)
            if overlap:
                raise ValueError(f"Subject leakage detected in fold {i}: {overlap}")

            print(
                f"  Fold {i+1}: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects"
            )

        print("✓ No data leakage - subject-independent CV validated")
        return cv

    def train_and_evaluate(self):
        """
        STEP 3: Train Model with Proper Cross-Validation
        """
        print(f"\n" + "=" * 40)
        print("MODEL TRAINING & EVALUATION")
        print("=" * 40)

        # Setup CV strategy
        cv = self.setup_subject_independent_cv()

        # Create ML pipeline: Scale features -> Random Forest
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),  # Normalize voice features
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,  # 50 decision trees
                        max_depth=10,  # Prevent overfitting
                        random_state=42,
                        class_weight="balanced",  # Handle 3:1 class imbalance
                    ),
                ),
            ]
        )

        # Cross-validation evaluation
        print("Performing subject-independent cross-validation...")
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        results = {}

        for metric in metrics:
            scores = cross_val_score(
                self.model,
                self.X,
                self.y,
                groups=self.subjects,
                cv=cv,
                scoring=metric,
                n_jobs=-1,
            )
            results[metric] = scores
            print(f"  {metric.upper()}: {scores.mean():.3f} ± {scores.std():.3f}")

        return results

    def get_feature_importance(self, feature_cols):
        """
        STEP 4: Feature Analysis - Which voice measurements matter most?
        """
        print(f"\n" + "=" * 40)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)

        # Train on full dataset to get feature importance
        self.model.fit(self.X, self.y)

        # Get importance scores
        rf = self.model.named_steps["classifier"]
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("Top 10 most important voice features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        return importance_df

    def clinical_evaluation(self):
        """
        STEP 5: Detailed Medical Evaluation
        Calculate sensitivity, specificity, etc.
        """
        print(f"\n" + "=" * 40)
        print("CLINICAL EVALUATION")
        print("=" * 40)

        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        # Collect all predictions across CV folds
        all_y_true, all_y_pred, all_y_prob = [], [], []

        for train_idx, test_idx in cv.split(self.X, self.y, groups=self.subjects):
            # Train model on this fold
            model_fold = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=10,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model_fold.fit(X_train, y_train)

            # Get predictions
            y_pred = model_fold.predict(X_test)
            y_prob = model_fold.predict_proba(X_test)[:, 1]

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

        # Convert to arrays
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob)

        # Confusion matrix and clinical metrics
        cm = confusion_matrix(all_y_true, all_y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Clinical metrics (what doctors care about)
        sensitivity = tp / (tp + fn)  # % of PD patients correctly identified
        specificity = tn / (tn + fp)  # % of healthy people correctly identified
        auc = roc_auc_score(all_y_true, all_y_prob)

        print("Confusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    Healthy  PD")
        print(f"Healthy      {cm[0,0]:3d}    {cm[0,1]:3d}")
        print(f"PD           {cm[1,0]:3d}    {cm[1,1]:3d}")

        print(f"\nClinical Metrics:")
        print(f"  Sensitivity (PD Detection Rate): {sensitivity:.3f}")
        print(f"  Specificity (Healthy Detection Rate): {specificity:.3f}")
        print(f"  AUC-ROC: {auc:.3f}")

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc,
            "confusion_matrix": cm,
        }


def run_analysis(data_path="../parkinsons.data"):
    """
    Main function: Run complete Parkinson's detection analysis
    """
    try:
        # Initialize model
        model = ParkinsonDetectionModel(data_path)

        # Run the pipeline
        feature_cols = model.load_and_prepare_data()  # Load data + extract subjects
        cv_results = model.train_and_evaluate()  # Train with subject-independent CV
        importance = model.get_feature_importance(
            feature_cols
        )  # Which features matter?
        clinical = model.clinical_evaluation()  # Medical evaluation

        # Final summary
        print(f"\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"AUC-ROC Score: {clinical['auc']:.3f}")
        print(f"Sensitivity (PD Detection): {clinical['sensitivity']:.3f}")
        print(f"Specificity (Healthy Detection): {clinical['specificity']:.3f}")
        print(f"\nNote: Results use subject-independent validation")
        print(f"(No person appears in both training and testing)")

        return model, cv_results, clinical

    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        print("Make sure parkinsons.data is outside folder")
    except Exception as e:
        print(f"Error: {str(e)}")


# Run the analysis when script is executed
if __name__ == "__main__":
    print("Starting Parkinson's Disease Detection Analysis...")
    print("Using subject-independent cross-validation for clinical validity\n")

    model, results, clinical_metrics = run_analysis("../parkinsons.data")
