import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def main():
    print("=" * 70)
    print("MNIST Experiment 08: Dual-Paradigm Meta-Ensemble")
    print("Architectures: PCA(55)+RBF-SVM(C=5) [Structural Margin]")
    print("             + Deep MLP(256, 128)   [Non-Linear Manifold]")
    print("=" * 70)

    # 1. Load Data
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

    # Component 1: PCA(55) + RBF-SVM
    t0 = time.time()
    pca_val = PCA(n_components=55, random_state=42)
    X_tr_p = pca_val.fit_transform(X_tr)
    X_val_p = pca_val.transform(X_val)
    svm_val = SVC(C=5, kernel='rbf', gamma='scale', probability=True, random_state=42)
    svm_val.fit(X_tr_p, y_tr)
    t_svm_val = time.time() - t0
    p_svm_val = svm_val.predict_proba(X_val_p)
    acc_svm = (p_svm_val.argmax(axis=1) == y_val).mean() * 100
    print(f"  1. PCA(55) + RBF-SVM(C=5)   : {acc_svm:.2f}% accuracy ({t_svm_val:.1f}s)")

    # Component 2: Deep MLP(256, 128)
    t0 = time.time()
    mlp_val = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=25, random_state=42)
    mlp_val.fit(X_tr, y_tr)
    t_mlp_val = time.time() - t0
    p_mlp_val = mlp_val.predict_proba(X_val)
    acc_mlp = (p_mlp_val.argmax(axis=1) == y_val).mean() * 100
    print(f"  2. Deep MLP(256, 128)       : {acc_mlp:.2f}% accuracy ({t_mlp_val:.1f}s)")

    # Weight Tuning Sweep
    print("\n  Evaluating Ensemble Blend Weights:")
    best_w = 0.80
    best_acc = 0.0
    for w in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        p_blend = w * p_svm_val + (1.0 - w) * p_mlp_val
        acc = (p_blend.argmax(axis=1) == y_val).mean() * 100
        print(f"    w_SVM={w:.2f} / w_MLP={1.0-w:.2f} : {acc:.2f}% accuracy")
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print(f"\n  --> Selected Optimal Blend: {int(best_w*100)}% RBF-SVM + {int((1-best_w)*100)}% Deep MLP ({best_acc:.2f}%)")

    # 3. Fit on Full 42,000 Training Samples
    print("\n[Step 3/5] Fitting final models on all 42,000 training images...")
    
    # 3a. Full PCA + RBF-SVM
    t0 = time.time()
    print("  Fitting PCA(55) + RBF-SVM(C=5, probability=True)...")
    pca_full = PCA(n_components=55, random_state=42)
    X_full_p = pca_full.fit_transform(X)
    X_test_p = pca_full.transform(X_test)
    svm_full = SVC(C=5, kernel='rbf', gamma='scale', probability=True, random_state=42)
    svm_full.fit(X_full_p, y)
    t_svm_full = time.time() - t0
    print(f"  Full RBF-SVM pipeline fit completed in {t_svm_full:.1f}s.")

    # 3b. Full Deep MLP
    t0 = time.time()
    print("  Fitting Deep MLP(256, 128)...")
    mlp_full = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=25, random_state=42)
    mlp_full.fit(X, y)
    t_mlp_full = time.time() - t0
    print(f"  Full Deep MLP fit completed in {t_mlp_full:.1f}s.")

    # 4. Generate Predictions on Test Set (28,000 images)
    print("\n[Step 4/5] Generating test set predictions (28,000 images)...")
    t0 = time.time()
    p_test_svm = svm_full.predict_proba(X_test_p)
    print(f"  RBF-SVM inference : {time.time()-t0:.1f}s.")

    t0 = time.time()
    p_test_mlp = mlp_full.predict_proba(X_test)
    print(f"  Deep MLP inference: {time.time()-t0:.1f}s.")

    # Meta-Ensemble Probability Weighting
    p_test_blend = best_w * p_test_svm + (1.0 - best_w) * p_test_mlp
    final_preds = p_test_blend.argmax(axis=1)

    unique, counts = np.unique(final_preds, return_counts=True)
    print("\n  Predicted test digit distribution:")
    for u, c in zip(unique, counts):
        print(f"    Digit {u}: {c:,} ({c/len(test)*100:.2f}%)")

    # 5. Format and Save Submission
    print("\n[Step 5/5] Formatting and verifying submission file...")
    sub = pd.DataFrame({
        'ImageId': np.arange(1, len(test) + 1),
        'Label': final_preds
    })

    assert len(sub) == len(sample_sub), f"Length mismatch: {len(sub)} vs {len(sample_sub)}"
    assert (sub['ImageId'].values == sample_sub['ImageId'].values).all(), "ImageId mismatch!"
    assert not sub['Label'].isnull().any(), "Found null values in predictions!"
    assert set(sub['Label'].unique()).issubset(set(range(10))), "Invalid label values detected!"

    out_path = '/root/digit-recognizer/submission_dual_meta_ensemble.csv'
    sub.to_csv(out_path, index=False)
    print(f"  Successfully saved {len(sub):,} predictions to {out_path}.")
    print("=" * 70)

if __name__ == '__main__':
    main()
