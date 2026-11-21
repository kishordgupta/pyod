# -*- coding: utf-8 -*-


import os
import pickle
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
sys.path.append(os.path.abspath(os.path.dirname("__file__")))

from pyod.models.jepa import JEPA
from pyod.utils.data import generate_data


class TestJEPA(unittest.TestCase):
    def assertHasAttr(self, obj, intended_attr):
        self.assertTrue(hasattr(obj, intended_attr))

    def assertInRange(self, data, lower, upper):
        self.assertGreaterEqual(data.min(), lower)
        self.assertLessEqual(data.max(), upper)

    def setUp(self):
        self.n_train = 600
        self.n_test = 200
        self.n_features = 12
        self.contamination = 0.1
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = JEPA(epoch_num=3, contamination=self.contamination,
                        verbose=0)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        self.assertHasAttr(self.clf, 'decision_scores_')
        self.assertIsNotNone(self.clf.decision_scores_)
        self.assertHasAttr(self.clf, 'labels_')
        self.assertIsNotNone(self.clf.labels_)
        self.assertHasAttr(self.clf, 'threshold_')
        self.assertIsNotNone(self.clf.threshold_)
        self.assertHasAttr(self.clf, '_mu')
        self.assertIsNotNone(self.clf._mu)
        self.assertHasAttr(self.clf, '_sigma')
        self.assertIsNotNone(self.clf._sigma)
        self.assertHasAttr(self.clf, 'model')
        self.assertIsNotNone(self.clf.model)

    def test_fitted_statistics(self):
        self.assertEqual(self.clf.n_tokens_, self.n_features)
        self.assertEqual(self.clf.token_dim_, 1)
        self.assertEqual(self.clf.err_mean_.shape, (self.n_features,))
        self.assertEqual(self.clf.err_std_.shape, (self.n_features,))
        self.assertEqual(self.clf.emb_mean_.shape,
                         (self.n_features, self.clf.d_model))
        self.assertEqual(self.clf.emb_prec_.shape,
                         (self.clf.d_model, self.clf.d_model))
        self.assertTrue(np.all(self.clf.err_std_ > 0))

    def test_train_scores(self):
        self.assertEqual(len(self.clf.decision_scores_),
                         self.X_train.shape[0])
        self.assertTrue(np.isfinite(self.clf.decision_scores_).all())

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        self.assertEqual(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        self.assertGreaterEqual(roc_auc_score(self.y_test, pred_scores),
                                self.roc_floor)

    def test_prediction_scores_deterministic(self):
        # the scoring masks are seeded, so repeated calls must agree
        assert_allclose(self.clf.decision_function(self.X_test),
                        self.clf.decision_function(self.X_test))

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        self.assertEqual(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_parameter(self):
        with self.assertRaises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        self.assertEqual(pred_labels.shape, self.y_test.shape)
        self.assertEqual(confidence.shape, self.y_test.shape)
        self.assertInRange(confidence, 0, 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(
            self.X_test, method='linear', return_confidence=True)
        self.assertInRange(pred_proba, 0, 1)
        self.assertEqual(confidence.shape, self.y_test.shape)
        self.assertInRange(confidence, 0, 1)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        self.assertEqual(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with self.assertRaises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_predict_rank(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=3)
        self.assertInRange(pred_ranks, 0, self.X_train.shape[0] + 1)

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        self.assertEqual(clone_clf.get_params(), self.clf.get_params())

    def test_refit_reproduces_scores(self):
        refit = clone(self.clf).fit(self.X_train)
        assert_allclose(refit.decision_scores_, self.clf.decision_scores_,
                        rtol=1e-4, atol=1e-5)

    def test_pickle_roundtrip(self):
        restored = pickle.loads(pickle.dumps(self.clf))
        assert_allclose(restored.decision_function(self.X_test),
                        self.clf.decision_function(self.X_test))

    def test_score_pred(self):
        clf = JEPA(epoch_num=3, score='pred', verbose=0).fit(self.X_train)
        pred_scores = clf.decision_function(self.X_test)
        self.assertGreaterEqual(roc_auc_score(self.y_test, pred_scores),
                                self.roc_floor)

    def test_score_embed(self):
        clf = JEPA(epoch_num=3, score='embed', verbose=0).fit(self.X_train)
        pred_scores = clf.decision_function(self.X_test)
        self.assertGreaterEqual(roc_auc_score(self.y_test, pred_scores),
                                self.roc_floor)

    def test_ema_target_and_reconstruction(self):
        clf = JEPA(epoch_num=3, target_mode='ema', recon_weight=0.1,
                   verbose=0).fit(self.X_train)
        self.assertIsNotNone(clf.model.target_encoder)
        self.assertIsNotNone(clf.model.recon_head)
        pred_scores = clf.decision_function(self.X_test)
        self.assertGreaterEqual(roc_auc_score(self.y_test, pred_scores),
                                self.roc_floor)

    def test_mask_modes(self):
        for mask_mode in ('block', 'forecast', 'mixed'):
            clf = JEPA(epoch_num=1, mask_mode=mask_mode,
                       verbose=0).fit(self.X_train)
            pred_scores = clf.decision_function(self.X_test)
            self.assertEqual(pred_scores.shape[0], self.X_test.shape[0])
            self.assertTrue(np.isfinite(pred_scores).all())

    def test_two_features(self):
        # The smallest input: one context token and one target token.
        # Outliers lie far outside the training range here. With
        # encoder_norm=True they saturate and the ROC AUC drops below
        # 0.5, so this guards the default (about 0.9 without the norm).
        X_train, X_test, _, y_test = generate_data(
            n_train=600, n_test=200, n_features=2, random_state=0)
        clf = JEPA(epoch_num=5, verbose=0).fit(X_train)
        self.assertEqual(clf.n_tokens_, 2)
        self.assertGreaterEqual(
            roc_auc_score(y_test, clf.decision_function(X_test)), 0.75)

    def test_feature_grouping(self):
        # 150 features and at most 32 tokens -> 30 tokens of 5 features
        X_train, X_test, _, _ = generate_data(
            n_train=300, n_test=100, n_features=150, random_state=0)
        clf = JEPA(epoch_num=1, max_tokens=32, verbose=0).fit(X_train)
        self.assertEqual(clf.n_tokens_, 30)
        self.assertEqual(clf.token_dim_, 5)
        self.assertEqual(clf.decision_function(X_test).shape[0], 100)

    def test_feature_grouping_with_padding(self):
        # 23 features and at most 8 tokens -> 8 tokens of 3 features,
        # the last token is zero-padded
        X_train, X_test, _, _ = generate_data(
            n_train=300, n_test=100, n_features=23, random_state=0)
        clf = JEPA(epoch_num=1, max_tokens=8, verbose=0).fit(X_train)
        self.assertEqual(clf.n_tokens_, 8)
        self.assertEqual(clf.token_dim_, 3)
        self.assertTrue(np.isfinite(clf.decision_function(X_test)).all())

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            JEPA(epoch_num=1, score='something',
                 verbose=0).fit(self.X_train)
        with self.assertRaises(ValueError):
            JEPA(epoch_num=1, mask_mode='something',
                 verbose=0).fit(self.X_train)
        with self.assertRaises(ValueError):
            JEPA(epoch_num=1, target_mode='something',
                 verbose=0).fit(self.X_train)
        with self.assertRaises(ValueError):
            JEPA(epoch_num=1, mask_ratio=1.0, verbose=0).fit(self.X_train)
        with self.assertRaises(ValueError):
            JEPA(epoch_num=1, max_tokens=1, verbose=0).fit(self.X_train)

    def test_single_feature_raises(self):
        with self.assertRaises(ValueError):
            JEPA(epoch_num=1, verbose=0).fit(self.X_train[:, :1])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
