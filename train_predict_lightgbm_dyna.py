"""
Script: Training & Prediction using LightGBM for Multi-Label Classification (Dynamic)

This script trains a LightGBM model using One-Vs-Rest strategy to predict
labels dynamically based on true label cardinality:
- 1 true label → keep top 1 prediction
- 2 true labels → keep top 2 predictions  
- 3+ true labels → keep top 3 predictions

Evaluation uses exact match accuracy per cardinality bucket.

Optimized for: c2d-standard-32 (32 vCPU, 128 GB RAM)
Designed to run with: nohup python train_predict_lightgbm_dyna.py > output.log 2>&1 &

Author: ML Engineering Team
Date: December 2025
"""

import os
import sys
import ast
import logging
import signal
import traceback
from collections import Counter
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb
import joblib
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ============================================================================
# NOHUP COMPATIBILITY: Force unbuffered output
# ============================================================================
# This ensures output is immediately written to nohup.out
class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after each emit for nohup compatibility."""
    def emit(self, record):
        super().emit(record)
        self.flush()

# Configure logging with immediate flushing for nohup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[FlushingStreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force unbuffered stdout/stderr for nohup
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# MACHINE CONFIGURATION: c2d-standard-32 (32 vCPU, 128 GB RAM)
# ============================================================================
N_CPUS = 32
MEMORY_GB = 128
# Strategy: Parallelize across labels with MultiOutputClassifier
# Each LightGBM estimator uses 1 thread to avoid oversubscription
N_JOBS_LABELS = N_CPUS  # Number of labels to train in parallel
N_JOBS_LGB = 1          # Threads per LightGBM estimator
# For prediction: use all cores for parallel probability computation
N_JOBS_PREDICT = N_CPUS

# ============================================================================
# SIGNAL HANDLING FOR GRACEFUL SHUTDOWN (nohup compatibility)
# ============================================================================
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    logger.warning(f"Received signal {sig_name} ({signum}). Requesting graceful shutdown...")
    SHUTDOWN_REQUESTED = True

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, signal_handler)  # kill command
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
try:
    signal.signal(signal.SIGHUP, signal_handler)   # terminal hangup (nohup)
except (AttributeError, ValueError):
    pass  # SIGHUP not available on Windows


def check_shutdown():
    """Check if shutdown was requested and exit gracefully if so."""
    if SHUTDOWN_REQUESTED:
        logger.warning("Shutdown requested. Saving checkpoint and exiting...")
        raise SystemExit("Graceful shutdown requested")


class LightGBMDynamicMultiLabelClassifier:
    """
    Multi-label classifier using LightGBM with One-Vs-Rest strategy.
    Dynamically predicts labels based on true label cardinality:
    - 1 label → top 1 prediction
    - 2 labels → top 2 predictions
    - 3+ labels → top 3 predictions
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
        Optimized for high-memory machine (128 GB): uses float32 for balance of
        speed and memory, keeps data contiguous for cache efficiency.
        
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
        
        # Handle categorical columns with label encoding (parallelized)
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
        
        # Convert to float32 for optimal LightGBM performance
        # float32 is faster than float64 and sufficient precision for tree-based models
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_features[col] = df_features[col].astype(np.float32)

        # Fill any missing values after conversions
        df_features = df_features.fillna(-1)
        
        # Ensure contiguous memory layout for cache efficiency
        df_features = pd.DataFrame(
            np.ascontiguousarray(df_features.values, dtype=np.float32),
            columns=df_features.columns,
            index=df_features.index
        )
        
        self.feature_columns = df_features.columns.tolist()
        logger.info(f"Features prepared. Number of features: {len(self.feature_columns)}")
        logger.info(f"Memory usage: {df_features.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
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
        Optimized for c2d-standard-32 (32 vCPU, 128 GB RAM).
        
        Returns:
            MultiOutputClassifier: Configured model
        """
        logger.info("Building LightGBM multi-output model...")
        logger.info(f"Parallelization strategy: {N_JOBS_LABELS} labels in parallel, "
                   f"{N_JOBS_LGB} thread(s) per LightGBM estimator")
        
        # Accuracy-focused LightGBM base classifier optimized for c2d-standard-32
        # 
        # Parallelism strategy:
        # - Train multiple label classifiers in parallel (N_JOBS_LABELS=32)
        # - Each LightGBM uses 1 thread (N_JOBS_LGB=1) to avoid oversubscription
        # - This maximizes throughput on 32 vCPU machine
        #
        # Memory optimizations for 128 GB RAM:
        # - max_bin=255 for good precision with reasonable memory
        # - histogram_pool_size optimized for available memory
        lgb_classifier = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            n_estimators=2000,           # more boosting rounds for better accuracy
            learning_rate=0.03,          # smaller LR + more trees generalizes better
            num_leaves=256,              # higher capacity (paired with regularization)
            max_depth=-1,                # let leaves drive complexity
            min_child_samples=10,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.1,
            is_unbalance=True,           # helps for highly imbalanced one-vs-rest labels
            # Performance optimizations for c2d-standard-32
            max_bin=255,                 # default, good balance of speed/accuracy
            force_col_wise=True,         # better for wide data, avoids race conditions
            device_type="cpu",
            random_state=self.random_state,
            n_jobs=N_JOBS_LGB,           # 1 thread per estimator (parallelism at label level)
            verbose=-1
        )
        
        # Wrap in MultiOutputClassifier for multi-label prediction
        # Parallelize across labels (n_jobs=32 for 32 vCPU)
        self.model = MultiOutputClassifier(lgb_classifier, n_jobs=N_JOBS_LABELS)
        
        logger.info("Model built successfully")
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the multi-label classifier.
        Optimized for c2d-standard-32 (32 vCPU, 128 GB RAM).
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels (one-hot encoded)
        """
        import time
        
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Labels: {y_train.shape[1]}")
        logger.info(f"Parallelization: {N_JOBS_LABELS} label classifiers in parallel")
        logger.info("="*60)
        
        # Ensure dense y for LightGBM (it does not accept sparse targets)
        if sparse.issparse(y_train):
            logger.info("Converting sparse labels to dense array...")
            y_train_dense = y_train.toarray().astype(np.int8)  # int8 for memory efficiency
        else:
            y_train_dense = np.asarray(y_train, dtype=np.int8)
        
        # Ensure X is contiguous float32 for optimal LightGBM performance
        if hasattr(X_train, 'values'):
            X_train_arr = np.ascontiguousarray(X_train.values, dtype=np.float32)
        else:
            X_train_arr = np.ascontiguousarray(X_train, dtype=np.float32)
        
        logger.info(f"Training data memory: {X_train_arr.nbytes / 1e9:.2f} GB (features) + "
                   f"{y_train_dense.nbytes / 1e9:.2f} GB (labels)")
        
        start_time = time.time()
        self.model.fit(X_train_arr, y_train_dense)
        elapsed = time.time() - start_time
        
        logger.info("="*60)
        logger.info(f"MODEL TRAINING COMPLETED in {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
        logger.info("="*60)
    
    def predict_all_probabilities(self, X):
        """
        Get probability predictions for all labels.
        Optimized with parallel prediction across estimators.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Probability matrix (samples x labels)
        """
        n_labels = len(self.model.estimators_)
        n_samples = len(X)
        logger.info(f"Computing probabilities for {n_labels} labels on {n_samples} samples...")
        
        # Ensure X is contiguous for cache efficiency
        X_arr = np.ascontiguousarray(X.values, dtype=np.float32) if hasattr(X, 'values') else X
        
        def _get_proba(estimator):
            """Get positive class probability for a single estimator."""
            proba = estimator.predict_proba(X_arr)
            return proba[:, 1].astype(np.float32)
        
        # Parallel probability computation across estimators
        logger.info(f"Running parallel prediction with {N_JOBS_PREDICT} workers...")
        probabilities = Parallel(n_jobs=N_JOBS_PREDICT, prefer="threads")(
            delayed(_get_proba)(est) for est in self.model.estimators_
        )
        
        # Stack probabilities (samples x labels) - pre-allocate for efficiency
        all_probabilities = np.column_stack(probabilities)
        
        logger.info(f"Probability matrix shape: {all_probabilities.shape}")
        return all_probabilities
    
    def predict_top_k_with_probabilities(self, X, k=3):
        """
        Predict top k labels with their probabilities for each sample.
        Optimized with vectorized operations for c2d-standard-32.
        
        Args:
            X (pd.DataFrame): Feature matrix
            k (int): Number of top labels to predict (default: 3)
            
        Returns:
            tuple: (predicted_labels, predicted_probabilities)
                - predicted_labels: List of lists containing k label names
                - predicted_probabilities: List of lists containing k probabilities
        """
        logger.info(f"Predicting top {k} labels with probabilities...")
        
        all_probabilities = self.predict_all_probabilities(X)
        n_samples = len(X)
        
        # Vectorized top-k computation using argpartition (faster than full argsort)
        logger.info(f"Computing top-{k} predictions (vectorized)...")
        top_k_indices_unordered = np.argpartition(all_probabilities, -k, axis=1)[:, -k:]
        top_k_probas_unordered = np.take_along_axis(all_probabilities, top_k_indices_unordered, axis=1)
        # Sort within top-k for proper ordering (descending)
        sort_order = np.argsort(-top_k_probas_unordered, axis=1)
        top_k_indices = np.take_along_axis(top_k_indices_unordered, sort_order, axis=1)
        top_k_probas = np.take_along_axis(top_k_probas_unordered, sort_order, axis=1)
        
        # Convert indices to label names
        classes = self.label_encoder.classes_
        predicted_labels = []
        predicted_probas = []
        
        for i in range(n_samples):
            labels = [classes[idx] for idx in top_k_indices[i]]
            probas = top_k_probas[i].tolist()
            predicted_labels.append(labels)
            predicted_probas.append(probas)
        
        logger.info(f"Prediction completed for {n_samples} samples")
        
        return predicted_labels, predicted_probas
    
    def predict_dynamic_with_probabilities(self, X, true_labels):
        """
        Predict labels dynamically based on true label cardinality:
        - 1 true label → keep top 1 prediction
        - 2 true labels → keep top 2 predictions
        - 3+ true labels → keep top 3 predictions
        
        Optimized with vectorized operations for c2d-standard-32.
        
        Args:
            X (pd.DataFrame): Feature matrix
            true_labels (list[list[str]]): True labels for each sample
            
        Returns:
            tuple: (predicted_labels, predicted_probabilities)
                - predicted_labels: List of lists with dynamic number of labels
                - predicted_probabilities: List of lists with corresponding probabilities
        """
        logger.info("Predicting with dynamic top-k based on true label cardinality...")
        
        all_probabilities = self.predict_all_probabilities(X)
        n_samples = len(X)
        
        # Vectorized top-3 computation using argpartition (faster than full argsort)
        # argpartition is O(n) vs O(n log n) for argsort
        logger.info("Computing top-3 predictions (vectorized)...")
        k = 3
        # Get indices of top-3 (unordered)
        top_k_indices_unordered = np.argpartition(all_probabilities, -k, axis=1)[:, -k:]
        # Get the probabilities for these indices
        top_k_probas_unordered = np.take_along_axis(all_probabilities, top_k_indices_unordered, axis=1)
        # Sort within the top-k to get proper ordering (descending)
        sort_order = np.argsort(-top_k_probas_unordered, axis=1)
        top_3_indices = np.take_along_axis(top_k_indices_unordered, sort_order, axis=1)
        top_3_probas = np.take_along_axis(top_k_probas_unordered, sort_order, axis=1)
        
        # Pre-compute cardinalities as numpy array for vectorized operations
        cardinalities = np.array([min(len(t), 3) for t in true_labels], dtype=np.int32)
        
        # Convert label indices to names (vectorized where possible)
        classes = self.label_encoder.classes_
        
        logger.info("Building dynamic prediction lists...")
        predicted_labels = []
        predicted_probas = []
        
        # Process in batches for better cache utilization
        batch_size = 10000
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            
            for i in range(batch_start, batch_end):
                k_i = cardinalities[i]
                labels = [classes[idx] for idx in top_3_indices[i, :k_i]]
                probas = top_3_probas[i, :k_i].tolist()
                predicted_labels.append(labels)
                predicted_probas.append(probas)
        
        logger.info(f"Dynamic prediction completed for {n_samples} samples")
        
        return predicted_labels, predicted_probas
    
    def fit_label_encoder(self, labels_list):
        """
        Fit a label encoder and convert multi-labels to one-hot encoding.
        Optimized for 128 GB RAM: uses sparse output during fit,
        converts to dense during training for LightGBM compatibility.
        
        Args:
            labels_list (list): List of label lists
            
        Returns:
            sparse matrix: One-hot encoded labels (sparse for memory during split)
        """
        logger.info("Fitting label encoder...")
        
        # Get all unique labels
        all_labels = set()
        for labels in labels_list:
            all_labels.update(labels)
        
        # Create and fit MultiLabelBinarizer
        # Use sparse output for memory efficiency during train/test split
        # Will convert to dense during training (LightGBM requirement)
        self.label_encoder = MultiLabelBinarizer(sparse_output=True)
        y_encoded = self.label_encoder.fit_transform(labels_list)
        
        n_labels = len(self.label_encoder.classes_)
        n_samples = len(labels_list)
        dense_size_gb = (n_samples * n_labels * 1) / 1e9  # int8 size estimate
        
        logger.info(f"Number of unique labels: {n_labels}")
        logger.info(f"Label matrix shape: {y_encoded.shape}")
        logger.info(f"Estimated dense size: {dense_size_gb:.2f} GB (within {MEMORY_GB} GB budget)")
        
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

    def evaluate_exact_match_by_cardinality(self, true_labels, pred_labels, output_path):
        """
        Evaluate exact match accuracy per true-label cardinality bucket (1, 2, 3+).
        Exact match: predicted set must exactly equal true set (order-agnostic).
        
        Args:
            true_labels (list[list[str]]): True labels for each sample
            pred_labels (list[list[str]]): Predicted labels (dynamic k) for each sample
            output_path (str): Path to save CSV results
            
        Returns:
            pd.DataFrame: Metrics dataframe
        """
        buckets = ["1", "2", "3+"]
        totals = {b: 0 for b in buckets}
        exact_hits = {b: 0 for b in buckets}

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
            
            # Exact match: sets must be identical
            if true_set == pred_set:
                exact_hits[key] += 1

        rows = []
        total_samples = sum(totals.values())
        total_exact = sum(exact_hits.values())
        
        for b in buckets:
            total = totals[b]
            exact_acc = exact_hits[b] / total if total else 0.0
            rows.append({
                "true_label_cardinality": b,
                "total_samples": total,
                "exact_match_hits": exact_hits[b],
                "exact_match_accuracy": round(exact_acc, 6),
                "error_rate": round(1 - exact_acc, 6),
            })
        
        # Add overall row
        overall_acc = total_exact / total_samples if total_samples else 0.0
        rows.append({
            "true_label_cardinality": "OVERALL",
            "total_samples": total_samples,
            "exact_match_hits": total_exact,
            "exact_match_accuracy": round(overall_acc, 6),
            "error_rate": round(1 - overall_acc, 6),
        })

        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(output_path, index=False)
        logger.info(f"Exact match accuracy by cardinality saved to {output_path}")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("EXACT MATCH ACCURACY BY CARDINALITY")
        logger.info("="*60)
        for _, row in df_metrics.iterrows():
            logger.info(f"  {row['true_label_cardinality']:8s}: {row['exact_match_accuracy']:.4f} "
                       f"({row['exact_match_hits']}/{row['total_samples']})")
        logger.info("="*60)
        
        return df_metrics
    
    def save_model(self, output_dir='./lightgbm_v2_dyna'):
        """
        Save the trained model and encoders.
        
        Args:
            output_dir (str): Directory to save model artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'lightgbm_multilabel_model_dyna.joblib')
        encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
        features_path = os.path.join(output_dir, 'feature_columns.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        joblib.dump(self.feature_columns, features_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")
        logger.info(f"Feature columns saved to {features_path}")
    
    def load_model(self, output_dir='./lightgbm_v2_dyna'):
        """
        Load a previously trained model and encoders.
        
        Args:
            output_dir (str): Directory containing model artifacts
        """
        model_path = os.path.join(output_dir, 'lightgbm_multilabel_model_dyna.joblib')
        encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
        features_path = os.path.join(output_dir, 'feature_columns.joblib')
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_columns = joblib.load(features_path)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Label encoder loaded from {encoder_path}")
        logger.info(f"Feature columns loaded from {features_path}")


def write_status(output_dir, status, message=""):
    """Write current status to a file for monitoring progress."""
    status_path = os.path.join(output_dir, 'training_status.txt')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(status_path, 'w') as f:
        f.write(f"Status: {status}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Message: {message}\n")
    sys.stdout.flush()


def main():
    """
    Main execution function for training and prediction.
    Optimized for c2d-standard-32 (32 vCPU, 128 GB RAM).
    Designed to run with nohup for long-running background execution.
    
    Usage:
        nohup python train_predict_lightgbm_dyna.py > training.log 2>&1 &
        
        # Monitor progress:
        tail -f training.log
        cat lightgbm_v2_dyna/training_status.txt
    """
    import time
    total_start = time.time()
    
    # Configuration
    DATA_PATH = '../data/df_ml2.csv'
    OUTPUT_DIR = './lightgbm_v2_dyna'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TOP_K_LABELS = 4001
    
    # Create output directory early for status file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        logger.info("="*80)
        logger.info("LIGHTGBM DYNAMIC MULTI-LABEL CLASSIFIER")
        logger.info("Optimized for c2d-standard-32 (32 vCPU, 128 GB RAM)")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"PID: {os.getpid()}")
        logger.info("="*80)
        logger.info(f"Configuration:")
        logger.info(f"  - Data path: {DATA_PATH}")
        logger.info(f"  - Output directory: {OUTPUT_DIR}")
        logger.info(f"  - Top K labels: {TOP_K_LABELS}")
        logger.info(f"  - Test size: {TEST_SIZE}")
        logger.info(f"  - Parallel workers (labels): {N_JOBS_LABELS}")
        logger.info(f"  - Parallel workers (predict): {N_JOBS_PREDICT}")
        logger.info("="*80)
        sys.stdout.flush()
        
        write_status(OUTPUT_DIR, "STARTING", "Initializing classifier")
        
        # Initialize classifier
        classifier = LightGBMDynamicMultiLabelClassifier(
            random_state=RANDOM_STATE,
            test_size=TEST_SIZE
        )
        
        # Check for shutdown request
        check_shutdown()
        
        # Load and prepare data
        write_status(OUTPUT_DIR, "LOADING_DATA", f"Loading from {DATA_PATH}")
        df = classifier.load_and_prepare_data(DATA_PATH)
    
        # Parse labels
        write_status(OUTPUT_DIR, "PARSING_LABELS", "Parsing label column")
        labels_list = classifier.parse_labels(df)
        check_shutdown()

        # Keep only the top K frequent labels
        write_status(OUTPUT_DIR, "FILTERING_LABELS", f"Filtering to top {TOP_K_LABELS} labels")
        top_labels = classifier.get_top_labels(labels_list, top_k=TOP_K_LABELS)
        filtered_labels, kept_indices = classifier.filter_labels_to_top(
            labels_list, top_labels, k=3
        )
        check_shutdown()

        # Filter dataframe to kept rows
        df_filtered = df.iloc[kept_indices].reset_index(drop=True)
        logger.info(f"Data filtered to {df_filtered.shape[0]} rows after top-label selection")
        sys.stdout.flush()
        
        # Prepare features
        write_status(OUTPUT_DIR, "PREPARING_FEATURES", f"Processing {df_filtered.shape[0]} rows")
        X = classifier.prepare_features(df_filtered)
        check_shutdown()
        
        # Encode labels to one-hot format
        write_status(OUTPUT_DIR, "ENCODING_LABELS", "Creating one-hot encoding")
        y_encoded = classifier.fit_label_encoder(filtered_labels)
        check_shutdown()
        
        # Split data
        write_status(OUTPUT_DIR, "SPLITTING_DATA", f"Test size: {TEST_SIZE}")
        X_train, X_test, y_train, y_test = classifier.split_data(X, y_encoded)
        check_shutdown()
        
        # Build and train model
        write_status(OUTPUT_DIR, "TRAINING", f"Training on {X_train.shape[0]} samples with {N_JOBS_LABELS} workers")
        classifier.build_model()
        classifier.train(X_train, y_train)
        check_shutdown()
        
        # Save model immediately after training (checkpoint)
        write_status(OUTPUT_DIR, "SAVING_MODEL", "Saving trained model checkpoint")
        classifier.save_model(OUTPUT_DIR)
        logger.info("Model checkpoint saved successfully")
        sys.stdout.flush()
    
        # Decode true labels for test set
        write_status(OUTPUT_DIR, "PREDICTING", f"Running predictions on {X_test.shape[0]} test samples")
        true_labels = classifier.decode_true_labels(y_test)
        check_shutdown()
        
        # Make dynamic predictions based on true label cardinality
        y_pred_labels_dyna, y_pred_probas_dyna = classifier.predict_dynamic_with_probabilities(
            X_test, true_labels
        )
        check_shutdown()
        
        # Also get standard top-3 predictions for comparison
        y_pred_labels_top3, y_pred_probas_top3 = classifier.predict_top_k_with_probabilities(X_test, k=3)
        check_shutdown()
        
        # Display sample predictions
        logger.info("\n" + "="*80)
        logger.info("SAMPLE PREDICTIONS (First 10 test samples)")
        logger.info("="*80)
        for i in range(min(10, len(y_pred_labels_dyna))):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  True Labels ({len(true_labels[i])}): {true_labels[i]}")
            logger.info(f"  Dynamic Pred ({len(y_pred_labels_dyna[i])}): {y_pred_labels_dyna[i]}")
            logger.info(f"  Probabilities: {[f'{p:.4f}' for p in y_pred_probas_dyna[i]]}")
            match = "EXACT MATCH" if set(true_labels[i]) == set(y_pred_labels_dyna[i]) else "WRONG"
            logger.info(f"  Match: {match}")
        sys.stdout.flush()
        
        # Save train/test split data
        write_status(OUTPUT_DIR, "SAVING_PREDICTIONS", "Saving test predictions")
        split_data = {
            'X_test': X_test,
            'y_test': y_test,
            'true_labels': true_labels,
            'y_pred_labels_dyna': y_pred_labels_dyna,
            'y_pred_probas_dyna': y_pred_probas_dyna,
            'y_pred_labels_top3': y_pred_labels_top3,
            'y_pred_probas_top3': y_pred_probas_top3,
        }
        split_path = os.path.join(OUTPUT_DIR, 'test_predictions_dyna.joblib')
        joblib.dump(split_data, split_path)
        logger.info(f"\nTest predictions saved to {split_path}")
        sys.stdout.flush()

        # Evaluate exact match by cardinality (dynamic predictions)
        write_status(OUTPUT_DIR, "EVALUATING", "Computing accuracy metrics")
        metrics_path = os.path.join(OUTPUT_DIR, 'exact_match_accuracy_dyna.csv')
        classifier.evaluate_exact_match_by_cardinality(true_labels, y_pred_labels_dyna, metrics_path)

        # Export compact comparison CSV
        write_status(OUTPUT_DIR, "EXPORTING", "Creating comparison CSV")
        clot_candidates = ['Clot_1er_Pa', 'Clot1erPas', 'Clot1erPa', 'Clot_1er_Pas']
        clot_col = next((c for c in clot_candidates if c in X_test.columns), None)
        if not clot_col:
            raise KeyError(f"None of the Clot columns found in X_test. Candidates: {clot_candidates}")

        type_col = 'Type_Prediag'
        if type_col not in X_test.columns:
            raise KeyError("Column 'Type_Prediag' not found in X_test.")

        comp_df = pd.DataFrame({
            'Sample_ID': range(len(X_test)),
            'True_Labels_List': [str(lst) for lst in true_labels],
            'True_Cardinality': [len(lst) for lst in true_labels],
            'Predicted_Labels_Dyna': [str(lst) for lst in y_pred_labels_dyna],
            'Pred_Cardinality': [len(lst) for lst in y_pred_labels_dyna],
            'Predicted_Probas_Dyna': [str([round(p, 4) for p in lst]) for lst in y_pred_probas_dyna],
            'Exact_Match': [set(t) == set(p) for t, p in zip(true_labels, y_pred_labels_dyna)],
            'QteConso': X_test['QteConso'].values,
            'Clot_1er_Pa': X_test[clot_col].values,
            'Type_Prediag': X_test[type_col].values,
        })
        compare_path = os.path.join(OUTPUT_DIR, 'pred_vs_true_dyna.csv')
        comp_df.to_csv(compare_path, index=False)
        logger.info(f"Comparison CSV saved to {compare_path}")
        
        # Summary statistics
        total_elapsed = time.time() - total_start
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING AND PREDICTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total test samples: {len(X_test)}")
        logger.info(f"Cardinality distribution:")
        for card in [1, 2, 3]:
            count = sum(1 for t in true_labels if len(t) == card)
            logger.info(f"  {card} label(s): {count} samples ({100*count/len(true_labels):.1f}%)")
        
        overall_exact = sum(1 for t, p in zip(true_labels, y_pred_labels_dyna) if set(t) == set(p))
        logger.info(f"\nOverall Exact Match Accuracy: {overall_exact}/{len(true_labels)} "
                   f"({100*overall_exact/len(true_labels):.2f}%)")
        
        logger.info("\n" + "="*80)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total execution time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} min)")
        logger.info(f"All artifacts saved to: {OUTPUT_DIR}")
        logger.info("Saved files:")
        logger.info(f"  - lightgbm_multilabel_model_dyna.joblib (trained model)")
        logger.info(f"  - label_encoder.joblib (label encoder)")
        logger.info(f"  - feature_columns.joblib (feature names)")
        logger.info(f"  - test_predictions_dyna.joblib (test predictions)")
        logger.info(f"  - exact_match_accuracy_dyna.csv (accuracy metrics)")
        logger.info(f"  - pred_vs_true_dyna.csv (detailed comparison)")
        logger.info(f"  - training_status.txt (status file)")
        logger.info("="*80)
        sys.stdout.flush()
        
        # Write final success status
        write_status(OUTPUT_DIR, "COMPLETED", 
                    f"Successfully completed in {total_elapsed:.1f}s. "
                    f"Accuracy: {100*overall_exact/len(true_labels):.2f}%")
        
    except KeyboardInterrupt:
        logger.warning("\n" + "="*80)
        logger.warning("INTERRUPTED BY USER (Ctrl+C or kill signal)")
        logger.warning("="*80)
        write_status(OUTPUT_DIR, "INTERRUPTED", "User interrupted execution")
        sys.exit(130)
        
    except SystemExit as e:
        logger.warning(f"System exit: {e}")
        write_status(OUTPUT_DIR, "INTERRUPTED", str(e))
        sys.exit(1)
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("EXECUTION FAILED WITH ERROR")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*80)
        sys.stdout.flush()
        sys.stderr.flush()
        
        write_status(OUTPUT_DIR, "FAILED", f"{type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

