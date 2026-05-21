import numpy as np


def help_test_tiny_dataset(model):
    """
    This is a helper function that is used in both `test_c_naive_bayes` and
    `test_d_naive_bayes_em`. If you're stuck on the `test_tiny_dataset` cases,
    we encourage you to work through the NB calculations by hand using the
    numbers below. There aren't that many, and it will help you understand
    what your code is supposed to do!
    """
    from scipy.sparse import csr_matrix

    # update features to be binary
    X = csr_matrix(np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0],
    ]), shape=(3, 5), dtype=int)
    y = np.array([0, 1, 0])
    nb = model(smoothing=0)
    nb.fit(X, y)

    # alpha should be of shape [n_labels]
    assert nb.alpha.shape == (2, )
    # beta should be of shape [n_vocab, n_labels]
    assert nb.beta.shape == (5, 2)

    # Check that alpha is calculated correctly
    exp = np.array([2/3, 1/3])
    assert np.allclose(nb.alpha, exp), f"{nb.alpha} != {exp}"

    # Check that beta is calculated correctly
    beta_target = np.transpose(np.array([
        [1/2, 0, 1/2, 1/2, 0],
        [0, 1, 0, 1, 1]]))

    msg = f"{nb.beta} != {beta_target}"
    assert np.allclose(nb.beta, beta_target), msg

    # nb = BernoulliNB(alpha=1)
    nb = model(smoothing=1)
    nb.fit(X, y)

    # Predicted probabilities should match reference
    probs = np.array([[0.97156819, 0.02843181],
                      [0.10606722, 0.89393278],
                      [0.81030011, 0.18969989]])
    est = nb.predict_proba(X)
    assert np.all(np.isclose(est, probs)), f"{est} != {probs}"

    # Log likelihood should match reference output
    assert np.isclose(nb.likelihood(X, y), -9.246479418592056)


def build_small_dataset():
    from free_response.data import build_dataset
    from free_response.data import train_test_unlabeled_split

    # Build the train/test split dictionary
    # Train = 0, Test = 1
    splits_dict = {
        "2009_Obama": 1, "2017_Trump": 1,
        "2016_Obama": 0, "2020_Trump": 0,
        "1993_Clinton": 0, "2000_Clinton": 0,
        "1994_Clinton": 1, "2010_Obama": 1,
        "2001_Bush": 0, "2008_Bush": 1,
        "1989_Bush": 1, "1992_Bush": 1,
        "1981_Reagan": 0, "1988_Reagan": 1,
    }
    for year in range(2011, 2016):
        splits_dict[f"{year}_Obama"] = 2
    for year in range(2018, 2020):
        splits_dict[f"{year}_Trump"] = 2
    for year in range(1995, 2000):
        splits_dict[f"{year}_Clinton"] = 2
    for year in range(2002, 2008):
        splits_dict[f"{year}_Bush"] = 2
    for year in range(1990, 1992):
        splits_dict[f"{year}_Bush"] = 2

    # Build and split the dataset
    data, labels, speeches, vocab = build_dataset(
          "data/", num_docs=40, max_words=20, vocab_size=20)
    return train_test_unlabeled_split(data, labels, speeches, splits_dict)


# Build this dataset so it can be imported across tests.
train_data, train_labels, test_data, test_labels = build_small_dataset()
