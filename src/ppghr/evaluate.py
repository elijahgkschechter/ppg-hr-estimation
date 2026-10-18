"""Fair testing: always score a model on people it has never seen."""

import numpy as np


def loso_predict(make_model, X, y, subjects):
    """Leave-one-subject-out: for each person, train on everyone else, then predict that person."""
    predictions = np.zeros(len(y))
    for subject in np.unique(subjects):
        test = subjects == subject
        model = make_model()
        model.fit(X[~test], y[~test])
        predictions[test] = model.predict(X[test])
    return predictions


def per_subject_mae(predictions, y, subjects):
    """MAE for each person, plus the average across people (the number papers report)."""
    errors = np.abs(np.asarray(predictions) - np.asarray(y))
    per_subject = {int(s): float(errors[subjects == s].mean()) for s in np.unique(subjects)}
    return per_subject, float(np.mean(list(per_subject.values())))