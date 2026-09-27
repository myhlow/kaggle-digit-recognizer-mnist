import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.ndimage import shift

def augment_data(X_data, y_data):
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    X_aug = []
    y_aug = []
    n = len(X_data)
    for i in range(n):
        dx, dy = shifts[i % 4]
        shifted = shift(X_data[i].reshape(28, 28), (dy, dx), mode='constant', cval=0.0).flatten()
        X_aug.append(shifted)
        y_aug.append(y_data[i])
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int64)
    X_combined = np.vstack([X_data, X_aug])
    y_combined = np.concatenate([y_data, y_aug])
    return X_combined, y_combined

def main():
    print("=" * 70)
    print("MNIST Experiment 09: Translation-Augmented Orthogonal Subspace SVM")
    print("Pipeline: 2x Spatial Augmentation + PCA(55) + RBF-Kernel SVC(C=5)")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/5] Loading datasets...")
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    X = (train.drop(columns=['label']).values / 255.0).astype(np.float32)
    y = train['label'].values
    X_test = (test.values / 255.0).astype(np.float32)

    print(f"  Training set : {X.shape[0]:,} images, {X.shape[1]} features (pixels)")
    print(f"  Test set     : {X_test.shape[0]:,} images, {X_test.shape[1]} features (pixels)")

    # 2. Holdout Validation Benchmark (5,000 stratified samples)
    print("\n[Step 2/5] Running Holdout Validation Benchmark (5,000 stratified samples)...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=5000, random_state=42, stratify=y)

    t0 = time.time()
    X_tr_aug, y_tr_aug = augment_data(X_tr, y_tr)
    t_aug = time.time() - t0
    print(f"  Spatial augmentation expanded training partition to {len(X_tr_aug):,} images in {t_aug:.2f}s.")

    t0 = time.time()
    pca_val = PCA(n_components=55, random_state=42)
    X_tr_p = pca_val.fit_transform(X_tr_aug)
    X_val_p = pca_val.transform(X_val)
    t_pca_val = time.time() - t0
    print(f"  PCA(55) fit on 74,000 samples completed in {t_pca_val:.2f}s.")

    t0 = time.time()
    svm_val = SVC(C=5, kernel='rbf', gamma='scale', random_state=42)
    svm_val.fit(X_tr_p, y_tr_aug)
    t_svm_val = time.time() - t0
    print(f"  RBF-SVM(C=5) fit on 74,000 samples completed in {t_svm_val:.2f}s.")

    t0 = time.time()
    val_preds = svm_val.predict(X_val_p)
    acc = (val_preds == y_val).mean() * 100
    print(f"  --> Validation Accuracy on untouched 5,000 images: {acc:.2f}% (Predict: {time.time()-t0:.2f}s)")

    # 3. Fit on Full 84,000 Augmented Training Samples
    print("\n[Step 3/5] Fitting final pipeline on 84,000 augmented training images...")
    t0 = time.time()
    X_full_aug, y_full_aug = augment_data(X, y)
    print(f"  Full training set expanded to {len(X_full_aug):,} images ({time.time()-t0:.2f}s).")

    t0 = time.time()
    pca_full = PCA(n_components=55, random_state=42)
    X_full_p = pca_full.fit_transform(X_full_aug)
    X_test_p = pca_full.transform(X_test)
    print(f"  Full PCA(55) fit on 84,000 images completed in {time.time()-t0:.2f}s.")

    t0 = time.time()
    svm_full = SVC(C=5, kernel='rbf', gamma='scale', random_state=42)
    svm_full.fit(X_full_p, y_full_aug)
    print(f"  Full RBF-SVM fit on 84,000 images completed in {time.time()-t0:.2f}s.")

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

    out_path = '/root/digit-recognizer/submission_augmented_rbf_svm.csv'
    sub.to_csv(out_path, index=False)
    print(f"  Successfully saved {len(sub):,} predictions to {out_path}.")
    print("=" * 70)

if __name__ == '__main__':
    main()
