# -*- coding: utf-8 -*-
"""Example of using TSJEPA for time series anomaly detection
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from pyod.models.ts_jepa import TSJEPA
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_ts_data

if __name__ == "__main__":
    contamination = 0.05

    # Generate synthetic time series with anomalies
    X_train, X_test, y_train, y_test = generate_ts_data(
        n_train=2000, n_test=800, contamination=contamination,
        random_state=42)

    # A window of 40 timestamps is cut into 8 patches of 5 timestamps.
    # mask_mode='block' (default) uses context on both sides of the
    # hidden patches; mask_mode='forecast' only uses the past.
    # variant='lejepa' (default) trains the network with SIGReg; the other
    # JEPA versions are variant='ijepa', 'tsjepa' and 'tjepa'.
    clf_name = 'TSJEPA'
    clf = TSJEPA(window_size=40, patch_len=5, contamination=contamination)
    clf.fit(X_train)

    # per-timestamp labels and scores of the training series
    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])

    # score a new series
    y_test_scores = clf.decision_function(X_test)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
