import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.ndimage import shift

def augment_dual_axis(X_data, y_data):
    """
    Expands training set 3x (126,000 samples from 42,000 raw):
    - Original raw images (42,000)
    - Horizontal translation jitter (+1px / -1px alternating, 42,000)
    - Vertical translation jitter (+1px / -1px alternating, 42,000)
    """
    n = len(X_data)
    aug_x = np.zeros_like(X_data)
    aug_y = np.zeros_like(X_data)
    
    for i in range(n):
        dx = 1 if (i % 2 == 0) else -1
        dy = 1 if (i % 2 == 0) else -1
        aug_x[i] = shift(X_data[i].reshape(28, 28), (0, dx), mode='constant', cval=0.0).flatten()
        aug_y[i] = shift(X_data[i].reshape(28, 28), (dy, 0), mode='constant', cval=0.0).flatten()
        
    X_combined = np.vstack([X_data, aug_x, aug_y])
    y_combined = np.concatenate([y_data, y_data, y_data])
    return X_combined, y_combined

def main():
    print("=" * 70, flush=True)
    print("MNIST Experiment 10: 3x Dual-Axis Spatial Augmentation + Subspace SVM", flush=True)
    print("Pipeline: 126,000 Augmented Images + PCA(55) + RBF-SVM(C=5)", flush=True)
    print("Target: Break past Rank #431 into Kaggle Top 50%", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    t_start = time.time()
    print("\n[Step 1/5] Loading datasets...", flush=True)
    train = pd.read_csv('/root/digit-recognizer/train.csv')
    test = pd.read_csv('/root/digit-recognizer/test.csv')
    sample_sub = pd.read_csv('/root/digit-recognizer/sample_submission.csv')

    X = (train.drop(columns=['label']).values / 255.0).astype(np.float32)
    y = train['label'].values
    X_test = (test.values / 255.0).astype(np.float32)

    print(f"  Training set : {X.shape[0]:,} images, {X.shape[1]} features (pixels)", flush=True)
    print(f"  Test set     : {X_test.shape[0]:,} images, {X_test.shape[1]} features (pixels)", flush=True)

    # 2. Dual-Axis Data Augmentation
    print("\n[Step 2/5] Generating 3x dual-axis spatial translations...", flush=True)
    t0 = time.time()
    X_aug, y_aug = augment_dual_axis(X, y)
    t_aug = time.time() - t0
    print(f"  Training data expanded to {len(X_aug):,} images in {t_aug:.2f}s.", flush=True)

    # 3. PCA Dimensionality Reduction
    print("\n[Step 3/5] Fitting PCA(55) manifold projection on 126,000 images...", flush=True)
    t0 = time.time()
    pca = PCA(n_components=55, random_state=42)
    X_aug_p = pca.fit_transform(X_aug)
    X_test_p = pca.transform(X_test)
    t_pca = time.time() - t0
    print(f"  PCA(55) transformation completed in {t_pca:.2f}s.", flush=True)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum()*100:.2f}%", flush=True)

    # 4. Fit RBF-SVM on 126,000 Augmented Samples
    print("\n[Step 4/5] Training RBF-Kernel SVM (C=5, gamma='scale') on 126,000 samples...", flush=True)
    t0 = time.time()
    svm = SVC(C=5, kernel='rbf', gamma='scale', random_state=42)
    svm.fit(X_aug_p, y_aug)
    t_svm = time.time() - t0
    print(f"  RBF-SVM training completed in {t_svm:.2f}s (~{t_svm/60:.2f} min).", flush=True)
    print(f"  Total support vectors: {svm.n_support_.sum():,} ({svm.n_support_.tolist()})", flush=True)

    # 5. Generate Test Set Predictions
    print("\n[Step 5/5] Generating predictions on test set (28,000 images)...", flush=True)
    t0 = time.time()
    test_preds = svm.predict(X_test_p)
    t_inf = time.time() - t0
    throughput = len(test) / t_inf
    print(f"  Inference completed in {t_inf:.2f}s (~{throughput:.0f} img/sec).", flush=True)

    unique, counts = np.unique(test_preds, return_counts=True)
    print("\n  Predicted test digit distribution:")
    for u, c in zip(unique, counts):
        print(f"    Digit {u}: {c:,} ({c/len(test)*100:.2f}%)", flush=True)

    # Format and Verify Submission
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': test_preds
    })

    assert len(sub) == len(sample_sub), f"Length mismatch: {len(sub)} vs {len(sample_sub)}"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values in predictions!"
    assert set(sub['Label'].unique()).issubset(set(range(10))), "Invalid label values detected!"

    out_path = '/root/digit-recognizer/submission_exp10_top50.csv'
    sub.to_csv(out_path, index=False)
    total_pipeline_time = time.time() - t_start
    print(f"\n  Submission successfully written to {out_path}", flush=True)
    print(f"  Total end-to-end execution time: {total_pipeline_time:.2f}s ({total_pipeline_time/60:.2f} min)", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
