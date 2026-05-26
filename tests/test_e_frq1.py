import pytest
import numpy as np

from free_response.data import build_dataset
from src.naive_bayes import NaiveBayes
from src.naive_bayes_em import NaiveBayesEM


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply:RuntimeWarning")

def test_frq1():
    """
    A test case that a large-vocab dataset like that used in FRQ1 and checks to
    see whether the likelihood is finite and decreases after a couple iters.

    Note: passing this test case is recommended **but not required** before
        completing FRQ1.
    """

    # Load the dataset
    years_to_include = list(range(1987, 1997))
    data, labels, speeches, vocab = build_dataset(
        "data", max_words=200, vocab_size=1000, years=years_to_include)

    # treat five speeches as unlabeled
    labels[range(0, len(labels), 2)] = np.nan

    naive_bayes = NaiveBayes()
    naive_bayes.fit(data, labels)
    nb_likelihood = naive_bayes.likelihood(data, labels)
    nb_preds = naive_bayes.predict(data)

    isfinite = np.isfinite(labels)  # get rid of unlabelled speeches for accuracy
    assert np.isfinite(nb_likelihood), "NB should have finite likelihood"
    msg = "NB should have 50+% accuracy"
    assert np.mean(nb_preds[isfinite] == labels[isfinite]) > 0.5, msg

    prev_likelihood = -np.inf
    nbem_likelihoods = []
    for i in range(0, 5):
        # Fit and evaluate the NB+EM model
        naive_bayes_em = NaiveBayesEM(max_iter=i)
        naive_bayes_em.fit(data, labels)
        nbem_likelihood = naive_bayes_em.likelihood(data, labels)
        preds = naive_bayes_em.predict(data)

        assert np.isfinite(nbem_likelihood), "NBEM should have finite likelihood"
        assert nbem_likelihood >= prev_likelihood, "NBEM likelihood should improve"
        if i > 0:
            msg = "NBEM should have 50+% accuracy"
            assert np.mean(preds[isfinite] == labels[isfinite]) > 0.5, msg
        prev_likelihood = nbem_likelihood
        nbem_likelihoods.append(nbem_likelihood)

    nbem_improvements = [nbem_likelihoods[i + 1] - nbem_likelihoods[i]
                         for i in range(len(nbem_likelihoods) - 1)]
    # In our solution code, the likelihood improvements are approximately:
    # [3000, 200, 10, 0]
    msg1 = ", ".join(list(map("{:.0f}".format, nbem_improvements)))
    msg = f"NBEM likelihood should improve over as max_iter increases: {msg1}"
    assert np.sum(np.array(nbem_improvements) > 0) > 2, msg

    msg = "NB should differ from NBEM on likelihood"
    assert nb_likelihood != nbem_likelihood, msg
    msg = "NB should differ from NBEM on accuracy"
    assert nb_preds.shape == preds.shape
    assert not np.all(nb_preds == preds), msg


if __name__ == "__main__":
    test_frq1()
