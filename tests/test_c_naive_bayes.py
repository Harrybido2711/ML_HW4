import pytest
import numpy as np

from tests.utils import train_data, train_labels
from tests.utils import help_test_tiny_dataset


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_tiny_dataset_a():
    from src.naive_bayes import NaiveBayes
    help_test_tiny_dataset(NaiveBayes)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_smoothing():
    from src.naive_bayes import NaiveBayes
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0],
    ]), shape=(3, 5), dtype=int)
    train_y = np.array([0, 1, 0])
    nb = NaiveBayes(smoothing=0)
    nb.fit(X, train_y)
    test_y = np.array([1, 0, 1])

    # The log likelihood should be log(0) = -np.inf or 0 * log(0) = nan
    #   this happens because smoothing = 0, train_y != test_y,
    #   and some words only show up in one of the two documents.
    msg = "likelihood with smoothing=0 should be -inf or nan"
    assert not np.isfinite(nb.likelihood(X, test_y)), msg

    prev_prob = -np.inf
    # smoothing_vals = [1, 2, 4, 1e100]
    # smoothed_beta_values = [-np.log(9)]
    # for i in range(len(smoothing_vals)):
    for smoothing in [1, 2, 4, 1e100]:
        nb = NaiveBayes(smoothing=smoothing)
        nb.fit(X, train_y)
        prob = np.mean(nb.predict_proba(X)[(0, 1), (1, 0)])

        # The probability of seeing the opposite class should keep
        #     increasing as we increase the smoothing parameter
        msg = f"With smoothing={smoothing}, expect {prob} > {prev_prob}"
        assert prob > prev_prob, msg
        prev_prob = prob

        msg = "likelihood with smoothing should be finite"
        assert np.isfinite(nb.likelihood(X, test_y)), msg

        # Zero beta values for y=0 should be smoothed
        target_val = smoothing / (2 + 2 * smoothing)
        assert np.isclose(nb.beta[4, 0], target_val), (nb.beta[4, 0], target_val)

    # When smoothing is near-infinite, probabilities should all be 0.5
    assert np.isclose(prob, 0.5)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_without_em():
    from src.naive_bayes import NaiveBayes

    # Train and evaluate NB without EM
    nb = NaiveBayes()
    nb.fit(train_data, train_labels)
    nb.likelihood(train_data, train_labels)

    is_labeled = np.isfinite(train_labels)
    nb_preds = nb.predict(train_data[is_labeled, :])
    train_accuracy = np.mean(nb_preds == train_labels[is_labeled])

    # NB should get 83% accuracy on the labeled examples
    assert train_accuracy >= 0.83

    nb_probs = nb.predict_proba(train_data)
    # Predict_proba should output a [n_documents, n_labels] array
    assert nb_probs.shape == (train_labels.shape[0], 2)
    # Probabilities should sum to 1
    assert np.all(np.isclose(np.sum(nb_probs, axis=1), np.ones_like(train_labels)))
