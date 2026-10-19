"""The most important tests in the project: they prove the test person is NEVER used for training."""

import numpy as np

from ppghr.evaluate import loso_predict, nested_loso, per_subject_mae


class SpyModel:
    """A fake model that remembers which people it trained on, and fails loudly if it is
    ever asked to predict for one of them."""

    def __init__(self, n_trees=3):
        self.n_trees = n_trees

    def fit(self, X, y):
        self.trained_on = set(X[:, 0].astype(int))
        return self

    def _check(self, X):
        overlap = self.trained_on & set(X[:, 0].astype(int))
        assert not overlap, f"model predicted people it trained on: {overlap}"

    def predict(self, X):
        self._check(X)
        return np.zeros(len(X))

    def staged_predict(self, X):
        self._check(X)
        for _ in range(self.n_trees):
            yield np.zeros(len(X))


def _data():
    subjects = np.repeat([1, 2, 3, 4, 5], 20)
    X = subjects[:, None].astype(float)          # column 0 = person id, so the spy can see it
    return X, np.zeros(len(X)), subjects


def test_loso_never_trains_on_the_test_person():
    X, y, subjects = _data()
    loso_predict(SpyModel, X, y, subjects)


def test_nested_loso_never_uses_the_test_person_anywhere():
    X, y, subjects = _data()
    predictions, choices = nested_loso(lambda n_trees: SpyModel(n_trees), X, y, subjects,
                                       feature_sets={"only": np.array([0])},
                                       peak_bpm=np.zeros((len(y), 5)),
                                       tree_options=(2, 3), inner_folds=3)
    assert len(choices) == 5


def test_per_subject_mae_counts_each_person_equally():
    predictions = np.array([0, 0, 0, 10.0])
    truth = np.zeros(4)
    subjects = np.array([1, 1, 1, 2])
    per_subject, average = per_subject_mae(predictions, truth, subjects)
    assert per_subject == {1: 0.0, 2: 10.0}
    assert average == 5.0            # pooling all windows would give 2.5
