"""Fair testing: always score a model on people it has never seen."""

from pathlib import Path

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


def nested_loso(make_model, X, y, subjects, feature_sets, peak_bpm,
                tree_options=(300, 1000), snap_options=(None, 10), inner_folds=5,
                cache_dir=None):
    """Leave-one-subject-out where every design choice is made WITHOUT the test person.

    For each test person: try every (feature set, number of trees, snap setting) using
    only the other people (split into `inner_folds` groups of people), keep the best
    combination, retrain on all the other people, and only then predict the test person.

    If cache_dir is given, finished pieces are saved there, so a long run can be
    stopped and restarted without redoing work.
    """
    from sklearn.model_selection import GroupKFold
    from ppghr.models import snap_to_peaks

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.zeros(len(y))
    choices = {}
    for subject in np.unique(subjects):
        train = subjects != subject
        X_tr, y_tr, s_tr, pk_tr = X[train], y[train], subjects[train], peak_bpm[train]
        done_file = cache_dir / f"S{subject}_final.npz" if cache_dir else None
        if done_file is not None and done_file.exists():
            saved = np.load(done_file, allow_pickle=True)
            predictions[~train] = saved["predictions"]
            choices[int(subject)] = tuple(saved["choice"])
            print(f"S{subject}: loaded from cache, picked {choices[int(subject)]}", flush=True)
            continue

        # ---- inner competition: only the other people are used here ----
        inner_pred = {}
        for set_number, (name, columns) in enumerate(feature_sets.items()):
            inner_file = cache_dir / f"S{subject}_set{set_number}.npz" if cache_dir else None
            if inner_file is not None and inner_file.exists():
                saved = np.load(inner_file)
                for n_trees in tree_options:
                    inner_pred[(name, n_trees)] = saved[f"trees_{n_trees}"]
                continue
            for n_trees in tree_options:
                inner_pred[(name, n_trees)] = np.zeros(len(y_tr))
            for fit, check in GroupKFold(n_splits=inner_folds).split(X_tr, y_tr, s_tr):
                model = make_model(n_trees=max(tree_options))
                model.fit(X_tr[np.ix_(fit, columns)], y_tr[fit])
                # staged_predict gives the guesses after 1 tree, 2 trees, 3 trees...
                # so one fit lets us score every tree count we care about.
                for n, p in enumerate(model.staged_predict(X_tr[np.ix_(check, columns)]), 1):
                    if n in tree_options:
                        inner_pred[(name, n)][check] = p
            if inner_file is not None:
                np.savez(inner_file, **{f"trees_{n}": inner_pred[(name, n)] for n in tree_options})

        scores = {}
        for (name, n_trees), p_raw in inner_pred.items():
            for snap in snap_options:
                p = p_raw if snap is None else snap_to_peaks(p_raw, pk_tr, snap)
                scores[(name, n_trees, snap)] = per_subject_mae(p, y_tr, s_tr)[1]

        # ---- winner is retrained on all other people, then tested ONCE ----
        best_name, best_trees, best_snap = min(scores, key=scores.get)
        columns = feature_sets[best_name]
        model = make_model(n_trees=best_trees)
        model.fit(X_tr[:, columns], y_tr)
        p = model.predict(X[~train][:, columns])
        if best_snap is not None:
            p = snap_to_peaks(p, peak_bpm[~train], best_snap)
        predictions[~train] = p
        choices[int(subject)] = (best_name, best_trees, best_snap)
        if done_file is not None:
            np.savez(done_file, predictions=p,
                     choice=np.array([best_name, best_trees, best_snap], dtype=object))
        print(f"S{subject}: picked '{best_name}', {best_trees} trees, snap={best_snap}  "
              f"(inner score {scores[(best_name, best_trees, best_snap)]:.2f} bpm)", flush=True)
    return predictions, choices