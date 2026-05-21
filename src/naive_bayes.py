import numpy as np
import warnings

from src.utils import softmax
from src.sparse_practice import flip_bits_sparse_matrix


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when calculating beta
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  The returned array should be
            probabilities, not log probabilities. If you use log probabilities
            in any calculations, you can use src.utils.softmax to convert those
            into probabilities that sum to 1 for each row.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        raise NotImplementedError

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta is an array of shape [n_vocab, n_labels]. self.beta[j, k]
            is the probability of seeing the word j in a document with label k.
            Remember to use self.smoothing. If there are M documents with label
            k, and the `j`th word shows up in L of them, then `self.beta[j, k]`
            is `(L + smoothing) / (M + 2 * smoothing)`.

        Note: all tests will provide X to you as a *sparse array* which will
            make calculations with large datasets much more efficient.  We
            encourage you to use sparse arrays whenever possible, but it can be
            easier to debug with dense arrays (e.g., it is easier to print out
            the contents of an array by first converting it to a dense array).

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        raise NotImplementedError

    def likelihood(self, X, y):
        r"""
        Using the self.alpha and self.beta that were already computed in
            `self.fit`, compute the LOG likelihood of the data.  You should use
            logs to avoid underflow.  This function should not use unlabeled
            data. Wherever y is NaN, that label and the corresponding row of X
            should be ignored.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of binary word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        raise NotImplementedError
