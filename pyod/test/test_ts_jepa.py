# -*- coding: utf-8 -*-
"""Tests for TSJEPA."""

import os
import pickle
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_ts_data


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestTSJEPA(unittest.TestCase):
    def setUp(self):
        from pyod.models.ts_jepa import TSJEPA
        self.TSJEPA = TSJEPA
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=1000, n_test=400, contamination=0.05,
                             random_state=42)
        self.clf = TSJEPA(window_size=40, patch_len=5, epochs=3,
                          contamination=0.05)
        self.clf.fit(self.X_train)

    def test_fit(self):
        assert len(self.clf.decision_scores_) == 1000
        assert len(self.clf.labels_) == 1000
        assert np.isfinite(self.clf.decision_scores_).all()
        assert set(np.unique(self.clf.labels_)) <= {0, 1}
        for attr in ('model_', 'threshold_', 'n_channels_', 'X_mean_',
                     'X_std_', 'err_mean_', 'err_std_', 'emb_mean_',
                     'emb_prec_'):
            assert hasattr(self.clf, attr), attr

    def test_decision_function(self):
        scores = self.clf.decision_function(self.X_test)
        assert scores.shape == (400,)
        assert np.isfinite(scores).all()
        assert roc_auc_score(self.y_test, scores) >= 0.8

    def test_decision_function_deterministic(self):
        # the scoring masks are seeded, so repeated calls must agree
        assert_allclose(self.clf.decision_function(self.X_test),
                        self.clf.decision_function(self.X_test))

    def test_predict(self):
        labels = self.clf.predict(self.X_test)
        assert labels.shape == self.y_test.shape
        assert set(np.unique(labels)) <= {0, 1}

    def test_predict_proba(self):
        proba = self.clf.predict_proba(self.X_test)
        assert proba.min() >= 0
        assert proba.max() <= 1

    def test_multivariate(self):
        X_train, X_test, _, _ = generate_ts_data(
            n_train=600, n_test=300, n_channels=3, contamination=0.05,
            random_state=42)
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1)
        clf.fit(X_train)
        assert clf.n_channels_ == 3
        assert clf.X_mean_.shape == (3,)
        assert len(clf.decision_scores_) == 600
        scores = clf.decision_function(X_test)
        assert scores.shape == (300,)
        assert np.isfinite(scores).all()

    def test_mask_modes_and_scores(self):
        X_train, X_test, _, _ = generate_ts_data(
            n_train=400, n_test=300, n_channels=2, contamination=0.05,
            anomaly_type='both', random_state=1)
        for mask_mode in ('block', 'forecast', 'random', 'mixed'):
            for score in ('pred', 'embed', 'combined'):
                clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                                  mask_mode=mask_mode, score=score)
                clf.fit(X_train)
                scores = clf.decision_function(X_test)
                assert scores.shape == (300,), (mask_mode, score)
                assert np.isfinite(scores).all(), (mask_mode, score)

    def test_causal_boundary(self):
        # 'forecast' hides the last 2 of 8 patches of every window, so
        # the first 30 timestamps are never hidden and get threshold_
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                          mask_mode='forecast')
        clf.fit(self.X_train)
        assert np.allclose(clf.decision_scores_[:30], clf.threshold_)
        assert not np.allclose(clf.decision_scores_[30:60], clf.threshold_)
        scores = clf.decision_function(self.X_test)
        assert np.allclose(scores[:30], clf.threshold_)

    def test_step_leaves_no_gaps_unfilled(self):
        # with step > 1 the tail may not be covered by any window; those
        # timestamps must be filled, not left as NaN
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1, step=7)
        clf.fit(self.X_train)
        assert np.isfinite(clf.decision_scores_).all()
        assert np.isfinite(clf.decision_function(self.X_test)).all()

    def test_ema_target_and_max_aggregation(self):
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                          target_mode='ema', recon_weight=0.1,
                          score_aggregation='max', n_mask_samples=3)
        clf.fit(self.X_train)
        assert clf.model_.target_encoder is not None
        assert np.isfinite(clf.decision_function(self.X_test)).all()

    def test_max_train_windows(self):
        clf = self.TSJEPA(window_size=40, patch_len=5, epochs=1,
                          max_train_windows=100)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 1000

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        assert clone_clf.get_params() == self.clf.get_params()

    def test_refit_reproduces_scores(self):
        refit = clone(self.clf).fit(self.X_train)
        assert_allclose(refit.decision_scores_, self.clf.decision_scores_,
                        rtol=1e-4, atol=1e-5)

    def test_pickle_roundtrip(self):
        restored = pickle.loads(pickle.dumps(self.clf))
        assert_allclose(restored.decision_function(self.X_test),
                        self.clf.decision_function(self.X_test))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            # window_size is not a multiple of patch_len
            self.TSJEPA(window_size=45, patch_len=10).fit(self.X_train)
        with self.assertRaises(ValueError):
            # a window must hold at least two patches
            self.TSJEPA(window_size=10, patch_len=10).fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        mask_mode='something').fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        score='something').fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        score_aggregation='something').fit(self.X_train)
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5,
                        mask_ratio=0.0).fit(self.X_train)

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            self.TSJEPA(window_size=40, patch_len=5).fit(self.X_train[:30])
        with self.assertRaises(ValueError):
            self.clf.decision_function(self.X_test[:30])

    def test_channel_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.clf.decision_function(np.random.randn(500, 2))


if __name__ == '__main__':
    unittest.main()
