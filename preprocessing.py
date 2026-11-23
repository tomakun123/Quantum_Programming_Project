import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

'''
 Purpose of this file:
    1. Load the breast cancer dataset (569 patients, 30 features)
    2. Make one fixed, stratified 80/20 train/test split
    3. Draw small training subsets (10, 20, 50, 100) from the 80% pool
    4. Reduce 30 features --> n (2 or 4) and scale them to [0, pi] for the feature map

Every preprocessing step is fitted on the training subset only, so the test set never
influences the model (no leakage), and the pipeline only ever "sees" n_train patients.
'''

seed = 42        # fixes the split and the training subset
test_size = 0.2  # fixed 20% test set, the same for every experiment


def load_split(seed=seed):
    '''
    Fixed 80/20 split of the full dataset.
    stratify=y keeps the malignant/benign ratio (~37/63) the same in both halves.

    Returns X_pool, X_test, y_pool, y_test
        - pool: the 80% (455 patients) that training subsets are drawn from
        - test: the 20% (114 patients) held out for every experiment
    '''
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def print_split_summary():
    X_pool, X_test, y_pool, y_test = load_split()
    print(f"Dataset: {len(X_pool) + len(X_test)} patients, {X_pool.shape[1]} features "
          f"(label 0 = malignant, 1 = benign)")
    for name, y in (("Training pool (80%)", y_pool), ("Test set (20%)", y_test)):
        print(f"  {name:<20}: {len(y):>3} patients, {np.mean(y == 0):.1%} malignant")


def sample_training_subset(X_pool, y_pool, n_train, seed=seed):
    '''
    Stratified random subset of n_train patients from the training pool,
    so even 10 samples contain both classes.
    '''
    X_sub, _, y_sub, _ = train_test_split(
        X_pool, y_pool, train_size=n_train, stratify=y_pool, random_state=seed
    )
    return X_sub, y_sub


def build_preprocessor(n_features):
    '''
    30 raw features --> n_features angles in [0, pi]
        1. StandardScaler: zero mean / unit variance, so PCA isn't dominated by large-unit features
        2. PCA: keep the n_features directions with the most variance (one per qubit)
        3. MinMaxScaler: map each component to [0, pi] for use as a rotation angle
            --> clip=True keeps test points that fall outside the training range inside [0, pi]
    '''
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_features),
        MinMaxScaler(feature_range=(0, np.pi), clip=True),
    )


def preprocess(X_train, X_test, n_features):
    '''
    Fit on the training points only, then apply the same transform to the test points.
    Returns the transformed X_train, X_test and the fitted pipeline.
    '''
    prep = build_preprocessor(n_features).fit(X_train)
    return prep.transform(X_train), prep.transform(X_test), prep
