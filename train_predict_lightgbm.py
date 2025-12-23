"""
Script 1: Training & Prediction using LightGBM for Multi-Label Classification

This script trains a LightGBM model using One-Vs-Rest strategy to predict
3 labels per row along with their associated probabilities.

Author: ML Engineering Team
Date: December 2025
"""

import os
import ast
import logging
from collections import Counter
import warnings
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGBMMultiLabelClassifier:
    """
    Multi-label classifier using LightGBM with One-Vs-Rest strategy.
    Predicts exactly 3 labels per row with associated probabilities.
    """
    
    def __init__(self, random_state=42, test_size=0.2):
        """
        Initialize the classifier.
        
        Args:
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of dataset for test split
        """
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, data_path):
        """
        Load the dataset and prepare features and labels.
        
        Args:
            data_path (str): Path to the CSV file containing df_ml2
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        return df
    
    def parse_labels(self, df, label_column='LIBEL_ARTICLE'):
        """
        Parse the label column which contains string representations of lists.
        
        Args:
            df (pd.DataFrame): Input dataframe
            label_column (str): Name of the column containing labels
            
        Returns:
            list: List of label lists (multi-label format)
        """
        logger.info(f"Parsing labels from column: {label_column}")
        
        # Parse string representation of lists
        labels = df[label_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Keep at most 3 labels per row; no padding with UNKNOWN
        labels = labels.apply(lambda x: x[:3] if len(x) > 3 else x)
        
        logger.info(f"Labels parsed. Sample: {labels.iloc[0]}")
        return labels.tolist()

    def get_top_labels(self, labels_list, top_k=4001):
        """
        Compute the most frequent labels.

        Args:
            labels_list (list[list[str]]): Raw labels per row
            top_k (int): Number of most frequent labels to keep

        Returns:
            set: Set of top_k labels
        """
        flat = [label for labels in labels_list for label in labels]
        counter = Counter(flat)
        top_labels = [label for label, _ in counter.most_common(top_k)]
        logger.info(f"Keeping top {top_k} labels (of {len(counter)} unique)")
        return set(top_labels)

    def filter_labels_to_top(self, labels_list, top_labels, k=3):
        """
        Filter labels to the top set and pad/truncate to k items.

        Args:
            labels_list (list[list[str]]): Raw labels per row
            top_labels (set): Labels to keep
            k (int): Number of labels to keep per row

        Returns:
            tuple: (filtered_labels, kept_indices)
        """
        filtered = []
        kept_indices = []

        for idx, labels in enumerate(labels_list):
            kept = [l for l in labels if l in top_labels][:k]
            if len(kept) == 0:
                continue  # drop rows with no top labels
            filtered.append(kept)
            kept_indices.append(idx)

        logger.info(f"Filtered to {len(filtered)} rows containing top labels")
        return filtered, kept_indices
    
    def prepare_features(self, df):
        """
        Prepare feature columns for training.
        Handles both numeric and categorical features.
        Downcasts numerics to float32/int32 to reduce memory (helpful on M2).
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        logger.info("Preparing features...")
        
        # Exclude non-feature columns
        exclude_cols = ['Dossier', 'LIBEL_ARTICLE', 'LIBEL_ARTICLE_Length']
        
        # Get all columns except excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        df_features = df[feature_cols].copy()
        
        # Handle categorical columns with label encoding
        categorical_cols = df_features.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
        
        # Downcast numerics to save memory on Apple Silicon
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].apply(
            pd.to_numeric, errors='coerce', downcast='float'
        )

        # Fill any missing values after conversions
        df_features = df_features.fillna(-1)
        
        self.feature_columns = df_features.columns.tolist()
        logger.info(f"Features prepared. Number of features: {len(self.feature_columns)}")
        
        return df_features
    
    def split_data(self, X, y):
        """
        Split data into training and test sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (np.ndarray): Label matrix
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test_size={self.test_size}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        logger.info(f"Train set size: {self.X_train.shape[0]}")
        logger.info(f"Test set size: {self.X_test.shape[0]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        """
        Build LightGBM multi-output classifier using One-Vs-Rest strategy.
        
        Returns:
            MultiOutputClassifier: Configured model
        """
        logger.info("Building LightGBM multi-output model...")
        
        # Accuracy-focused LightGBM base classifier.
        # Target runtime: c2d-standard-32 (32 vCPU / 128 GB RAM).
        #
        # Note on parallelism:
        # - We parallelize across labels via MultiOutputClassifier(n_jobs=32)
        # - Each LightGBM fit uses a single thread (n_jobs=1) to avoid oversubscription
        lgb_classifier = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            n_estimators=2000,          # more boosting rounds for better accuracy
            learning_rate=0.03,         # smaller LR + more trees typically generalizes better
            num_leaves=256,             # higher capacity (paired with regularization below)
            max_depth=-1,               # let leaves drive complexity; tune if overfitting
            min_child_samples=10,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.1,
            is_unbalance=True,          # helps for highly imbalanced one-vs-rest labels
            force_col_wise=True,        # generally safer on wide/tabular data
            device_type="cpu",
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1
        )
        
        # Wrap in MultiOutputClassifier for multi-label prediction (parallel across outputs)
        self.model = MultiOutputClassifier(lgb_classifier, n_jobs=32)
        
        logger.info("Model built successfully")
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the multi-label classifier.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels (one-hot encoded)
        """
        logger.info("Starting model training...")
        
        # Ensure dense y for LightGBM (it does not accept sparse targets)
        if sparse.issparse(y_train):
            y_train_dense = y_train.toarray()
        else:
            y_train_dense = y_train

        self.model.fit(X_train, y_train_dense)
        
        logger.info("Model training completed")
    
    def predict_top_k_with_probabilities(self, X, k=3):
        """
        Predict top k labels with their probabilities for each sample.
        
        Args:
            X (pd.DataFrame): Feature matrix
            k (int): Number of top labels to predict (default: 3)
            
        Returns:
            tuple: (predicted_labels, predicted_probabilities)
                - predicted_labels: List of lists containing k label names
                - predicted_probabilities: List of lists containing k probabilities
        """
        logger.info(f"Predicting top {k} labels with probabilities...")
        
        # Get probability predictions for all labels
        probabilities = []
        for estimator in self.model.estimators_:
            proba = estimator.predict_proba(X)
            # Get probability for class 1 (positive class)
            probabilities.append(proba[:, 1])
        
        # Stack probabilities (samples x labels)
        all_probabilities = np.column_stack(probabilities).astype(np.float32)
        
        # Get top k indices for each sample
        top_k_indices = np.argsort(all_probabilities, axis=1)[:, -k:][:, ::-1]
        
        # Get corresponding probabilities
        top_k_probas = np.take_along_axis(all_probabilities, top_k_indices, axis=1)
        
        # Convert indices to label names
        predicted_labels = []
        predicted_probas = []
        
        for i in range(len(X)):
            # Get label names for top k predictions
            labels = [self.label_encoder.classes_[idx] for idx in top_k_indices[i]]
            probas = top_k_probas[i].tolist()
            
            predicted_labels.append(labels)
            predicted_probas.append(probas)
        
        logger.info(f"Prediction completed for {len(X)} samples")
        
        return predicted_labels, predicted_probas
    
    def fit_label_encoder(self, labels_list):
        """
        Fit a label encoder and convert multi-labels to one-hot encoding.
        
        Args:
            labels_list (list): List of label lists
            
        Returns:
            np.ndarray: One-hot encoded labels
        """
        logger.info("Fitting label encoder...")
        
        # Get all unique labels
        all_labels = set()
        for labels in labels_list:
            all_labels.update(labels)
        
        # Create and fit MultiLabelBinarizer with sparse output to save memory
        self.label_encoder = MultiLabelBinarizer(sparse_output=True)
        y_encoded = self.label_encoder.fit_transform(labels_list)
        
        logger.info(f"Number of unique labels: {len(self.label_encoder.classes_)}")
        
        return y_encoded

    def decode_true_labels(self, y_encoded):
        """
        Decode one-hot labels (sparse or dense) back to lists of labels.
        """
        if sparse.issparse(y_encoded):
            y_dense = y_encoded.toarray()
        else:
            y_dense = np.asarray(y_encoded)

        decoded = []
        for row in y_dense:
            idxs = np.where(row == 1)[0]
            decoded.append([self.label_encoder.classes_[i] for i in idxs])
        return decoded

    def evaluate_by_cardinality(self, true_labels, pred_labels, output_path):
        """
        Evaluate containment accuracy per true-label cardinality bucket (1,2,3+).
        Containment: all true labels are present in the 3 predicted labels (order-agnostic).
        Wrong rate = 1 - containment_accuracy.
        Saves results to CSV at output_path.
        """
        buckets = ["1", "2", "3+"]
        totals = {b: 0 for b in buckets}
        contain_hits = {b: 0 for b in buckets}

        def bucket_key(n):
            if n <= 1:
                return "1"
            if n == 2:
                return "2"
            return "3+"

        for t_labels, p_labels in zip(true_labels, pred_labels):
            true_set = set(t_labels)
            pred_set = set(p_labels)
            key = bucket_key(len(true_set))
            totals[key] += 1
            if true_set.issubset(pred_set):
                contain_hits[key] += 1

        rows = []
        for b in buckets:
            total = totals[b]
            contain = contain_hits[b] / total if total else 0.0
            rows.append({
                "true_label_cardinality": b,
                "total_samples": total,
                "containment_accuracy": round(contain, 6),
                "wrong_rate": round(1 - contain, 6),
            })

        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(output_path, index=False)
        logger.info(f"Cardinality accuracy saved to {output_path}")
        return df_metrics
    
    def save_model(self, output_dir='./lightgbmV2'):
        """
        Save the trained model and encoders.
        
        Args:
            output_dir (str): Directory to save model artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'lightgbm_multilabel_model_V2_K2.joblib')
        encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, output_dir='./lightgbmV2'):
        """
        Load a previously trained model and encoders.
        
        Args:
            output_dir (str): Directory containing model artifacts
        """
        model_path = os.path.join(output_dir, 'lightgbm_multilabel_model_V2.joblib')
        encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Label encoder loaded from {encoder_path}")


def main():
    """
    Main execution function for training and prediction.
    """
    # Configuration
    DATA_PATH = '../data/df_ml2.csv'
    OUTPUT_DIR = './lightgbmV2'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TOP_K_LABELS = 4001
    
    # Initialize classifier
    classifier = LightGBMMultiLabelClassifier(
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE
    )
    
    # Load and prepare data
    df = classifier.load_and_prepare_data(DATA_PATH)
    
    # Parse labels
    labels_list = classifier.parse_labels(df)

    # Keep only the top K frequent labels
    top_labels = classifier.get_top_labels(labels_list, top_k=TOP_K_LABELS)
    filtered_labels, kept_indices = classifier.filter_labels_to_top(
        labels_list, top_labels, k=3
    )

    # Filter dataframe to kept rows
    df_filtered = df.iloc[kept_indices].reset_index(drop=True)
    logger.info(f"Data filtered to {df_filtered.shape[0]} rows after top-label selection")
    
    # Prepare features
    X = classifier.prepare_features(df_filtered)
    
    # Encode labels to one-hot format
    y_encoded = classifier.fit_label_encoder(filtered_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = classifier.split_data(X, y_encoded)
    
    # Build and train model
    classifier.build_model()
    classifier.train(X_train, y_train)
    
    # Make predictions on test set
    y_pred_labels, y_pred_probas = classifier.predict_top_k_with_probabilities(X_test, k=3)
    
    # Display sample predictions
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PREDICTIONS (First 5 test samples)")
    logger.info("="*80)
    for i in range(min(5, len(y_pred_labels))):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Predicted Labels: {y_pred_labels[i]}")
        logger.info(f"  Probabilities: {[f'{p:.4f}' for p in y_pred_probas[i]]}")
    
    # Save model
    classifier.save_model(OUTPUT_DIR)
    
    # Save train/test split data for Script 2
    split_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_labels': y_pred_labels,
        'y_pred_probas': y_pred_probas
    }
    split_path = os.path.join(OUTPUT_DIR, 'test_predictions_V2.joblib')
    joblib.dump(split_data, split_path)
    logger.info(f"\nTest predictions saved to {split_path}")

    # Evaluate by cardinality (true label count) and save to CSV
    true_labels = classifier.decode_true_labels(y_test)
    card_metrics_path = os.path.join(OUTPUT_DIR, 'cardinality_accuracy_V2.csv')
    classifier.evaluate_by_cardinality(true_labels, y_pred_labels, card_metrics_path)

    # Export compact comparison CSV (true vs pred + QteConso + Clot_1er_Pa + Type_Prediag)
    # Attempt multiple possible spellings for Clot column
    clot_candidates = ['Clot_1er_Pa', 'Clot1erPas', 'Clot1erPa', 'Clot_1er_Pas']
    clot_col = next((c for c in clot_candidates if c in X_test.columns), None)
    if not clot_col:
        raise KeyError(f"None of the Clot columns found in X_test. Candidates: {clot_candidates}")

    # Expect Type_Prediag column
    type_col = 'Type_Prediag'
    if type_col not in X_test.columns:
        raise KeyError("Column 'Type_Prediag' not found in X_test.")

    comp_df = pd.DataFrame({
        'Sample_ID': range(len(X_test)),
        'True_Labels_List': [str(lst) for lst in true_labels],
        'Predicted_Labels_List': [str(lst) for lst in y_pred_labels],
        'QteConso': X_test['QteConso'].values,
        'Clot_1er_Pa': X_test[clot_col].values,
        'Type_Prediag': X_test[type_col].values,
    })
    compare_path = os.path.join(OUTPUT_DIR, 'pred_vs_true_V2.csv')
    comp_df.to_csv(compare_path, index=False)
    logger.info(f"Comparison CSV saved to {compare_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Training and prediction completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
