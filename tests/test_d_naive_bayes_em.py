import pytest
import numpy as np

from tests.utils import train_data, train_labels, test_data, test_labels
from tests.utils import help_test_tiny_dataset


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_tiny_dataset_b():
    from src.naive_bayes_em import NaiveBayesEM
    help_test_tiny_dataset(NaiveBayesEM)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_alpha_beta_normalized():
    from src.naive_bayes import NaiveBayes
    from src.naive_bayes_em import NaiveBayesEM
    from scipy.sparse import csr_matrix

    vocab_size = 2
    for n in range(2, 10):
        shape = (n, vocab_size)
        X = csr_matrix(np.ones(shape), shape=shape, dtype=int)
        y = np.ones([n])
        y[0] = 0

        for smoothing in (0, 1):
            nb = NaiveBayes(smoothing=smoothing)
            nb.fit(X, y)
            # Alpha should sum to 1
            est = nb.alpha.sum()
            assert np.isclose(est, 1), f"{est} != 1"

            # Beta should sum to target
            est = nb.beta.sum(axis=0)
            n_y = np.array([1, n - 1])
            target = (n_y + smoothing) / (n_y + vocab_size * smoothing) * vocab_size
            assert np.allclose(est, target), f"{est} != {target}"

            for max_iter in range(1, 4):
                nbem = NaiveBayesEM(smoothing=smoothing, max_iter=max_iter)
                nbem.fit(X, y)
                # Alpha should sum to 1
                est = nbem.alpha.sum()
                assert np.isclose(est, 1), f"{est} != 1"

                # Beta should sum to target
                est = nbem.beta.sum(axis=0)
                target = (n_y + smoothing) / (n_y + vocab_size * smoothing) * vocab_size
                assert np.allclose(est, target), f"{est} != {target}"


def test_em_initialization():
    from src.naive_bayes_em import NaiveBayesEM

    nbem = NaiveBayesEM(max_iter=0)
    nbem.initialize_params(train_data.shape[1], 2)
    assert np.isclose(nbem.alpha.sum(), 1)
    assert np.allclose(nbem.beta, 0.5)

    nbem.fit(train_data, train_labels)
    # If you do zero EM steps, your initialized probabilities should be uniform
    assert np.all(nbem.alpha[0] == nbem.alpha)
    assert np.all(nbem.beta[0, :] == nbem.beta)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_em_basics():
    help_test_em(False)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_em_likelihood():
    help_test_em(True)


def help_test_em(include_likelihood=False):
    from src.naive_bayes_em import NaiveBayesEM
    prev_likelihood = -np.inf

    train_labels2 = train_labels.copy()

    alphas = []
    betas = []
    nbem_likelihoods = []
    max_iters = [1, 2, 3, 4, 5]
    for max_iter in max_iters:
        nbem = NaiveBayesEM(max_iter=max_iter)
        nbem.fit(train_data, train_labels)
        likelihood = nbem.likelihood(train_data, train_labels)

        # EM should only ever increase the likelihood
        assert likelihood >= prev_likelihood, "EM should increase likelihood"
        prev_likelihood = likelihood

        msg = "Don't overwrite y!"
        assert np.array_equal(train_labels, train_labels2, equal_nan=True), msg

        alphas.append(nbem.alpha.copy())
        betas.append(nbem.beta.copy())
        nbem_likelihoods.append(likelihood)

    msg = "Each iteration should update alpha/beta"
    assert np.unique(alphas, axis=0).shape[0] == len(max_iters), msg
    assert np.unique(betas, axis=0).shape[0] == len(max_iters), msg

    if include_likelihood:
        nbem_improvements = [nbem_likelihoods[i + 1] - nbem_likelihoods[i]
                             for i in range(len(nbem_likelihoods) - 1)]
        msg = "NBEM likelihood should improve"
        assert np.mean(np.array(nbem_improvements) > 1) >= 0.75, msg


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_comparison_naive_bayes():
    from src.naive_bayes import NaiveBayes
    from src.naive_bayes_em import NaiveBayesEM

    # Train and evaluate NB without EM
    nb1 = NaiveBayes()
    nb1.fit(train_data, train_labels)
    nb1_likelihood = nb1.likelihood(train_data, train_labels)
    nb1_preds = nb1.predict(test_data)
    nb1_accuracy = np.mean(nb1_preds == test_labels)

    # Train and evaluate NB with EM
    nb2 = NaiveBayesEM()
    nb2.fit(train_data, train_labels)
    nb2_likelihood = nb2.likelihood(train_data, train_labels)
    nb2_preds = nb2.predict(test_data)
    nb2_accuracy = np.mean(nb2_preds == test_labels)

    # NB using EM should outperform NB without it
    msg = f"Expect NBEM Acc ({100 * nb2_accuracy:.1f}%)"
    msg += f" > NB Acc ({100 * nb1_accuracy:.1f}%)"
    assert nb2_accuracy > nb1_accuracy, msg

    # NB using EM should outperform NB without it
    msg = f"Expect NBEM Likelihood ({nb2_likelihood:.3e})"
    msg += f" < NB Acc ({nb1_likelihood:.3e})"
    # NB with EM should have a lower likelihood. Why? We'll see in the FRQ
    assert nb2_likelihood < nb1_likelihood, msg
