import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def main():
    print("=" * 70)
    print("MNIST Experiment 07: Orthogonal Subspace Projection & RBF Support Vector Machine")
    print("Architecture: PCA(55 components) + RBF-Kernel SVC(C=5, gamma='scale')")
    print("=" * 70)

    # 1. Load Datasets
    print("\n[Step 1/5] Loading datasets...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    X = train.drop(columns=['label']).values / 255.0
    y = train['label'].values
    X_test = test.values / 255.0

    print(f"  Training set : {X.shape[0]:,} images, {X.shape[1]} features (pixels)")
    print(f"  Test set     : {X_test.shape[0]:,} images, {X_test.shape[1]} features (pixels)")

    # 2. Holdout Validation Benchmark (5,000 stratified samples)
    print("\n[Step 2/5] Running Holdout Validation Benchmark (5,000 stratified samples)...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=5000, random_state=42, stratify=y)

    t0 = time.time()
    pca_val = PCA(n_components=55, random_state=42)
    X_tr_p = pca_val.fit_transform(X_tr)
    X_val_p = pca_val.transform(X_val)
    pca_time = time.time() - t0
    var_ratio = pca_val.explained_variance_ratio_.sum() * 100
    print(f"  PCA(55) fit & transform completed in {pca_time:.2f}s (retained variance: {var_ratio:.2f}%).")

    t0 = time.time()
    svm_val = SVC(C=5, kernel='rbf', gamma='scale', random_state=42)
    svm_val.fit(X_tr_p, y_tr)
    fit_time = time.time() - t0
    print(f"  RBF-SVM(C=5) fit on 37,000 samples completed in {fit_time:.2f}s.")

    t0 = time.time()
    val_preds = svm_val.predict(X_val_p)
    pred_time = time.time() - t0
    acc = (val_preds == y_val).mean() * 100
    print(f"  Validation Accuracy: {acc:.2f}% (Inference: {pred_time:.2f}s on 5,000 images)")

    # 3. Fit on Full 42,000 Training Samples
    print("\n[Step 3/5] Fitting final model on all 42,000 training images...")
    t0 = time.time()
    pca_full = PCA(n_components=55, random_state=42)
    X_full_p = pca_full.fit_transform(X)
    X_test_p = pca_full.transform(X_test)
    pca_full_time = time.time() - t0
    print(f"  Full PCA(55) fit & transform completed in {pca_full_time:.2f}s.")

    t0 = time.time()
    svm_full = SVC(C=5, kernel='rbf', gamma='scale', random_state=42)
    svm_full.fit(X_full_p, y)
    svm_full_time = time.time() - t0
    print(f"  Full RBF-SVM fit on 42,000 images completed in {svm_full_time:.2f}s.")

    # 4. Generate Predictions on Test Set (28,000 images)
    print("\n[Step 4/5] Generating test set predictions (28,000 images)...")
    t0 = time.time()
    test_preds = svm_full.predict(X_test_p)
    inf_time = time.time() - t0
    throughput = len(test) / inf_time
    print(f"  Test inference completed in {inf_time:.2f}s (~{throughput:.0f} images/second).")

    unique, counts = np.unique(test_preds, return_counts=True)
    print("\n  Predicted test digit distribution:")
    for u, c in zip(unique, counts):
        print(f"    Digit {u}: {c:,} ({c/len(test)*100:.2f}%)")

    # 5. Format and Save Submission
    print("\n[Step 5/5] Formatting and verifying submission file...")
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': test_preds
    })

    assert len(sub) == len(sample_sub), f"Length mismatch: {len(sub)} vs {len(sample_sub)}"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values in predictions!"
    assert set(sub['Label'].unique()).issubset(set(range(10))), "Invalid label values detected!"

    out_path = '/root/digit-recognizer/submission_rbf_svm.csv'
    sub.to_csv(out_path, index=False)
    print(f"  Successfully saved {len(sub):,} predictions to {out_path}.")
    print("=" * 70)

if __name__ == '__main__':
    main()
