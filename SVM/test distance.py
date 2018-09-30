"""
This script is used to generate some simple 2d or 3d cases
and train an SVM on them. The main purpose is to see whether
the distance calculation is correct
"""

import os, json
import numpy as np
import mmdps_locale
from mmdps_old import brain_net, brain_template
from mmdps_old.utils import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

X = np.array([[0, 1], [1, 2], [0, 0], [1, 1]])
y = np.array([1, 1, -1, -1])

# classifier
classifier = svm.SVC(C = 100, kernel = 'linear')

classifier.fit(X, y)

# distance is calculated as y/||w|| for linear SVM
df = classifier.decision_function(X) 
w_norm = np.linalg.norm(classifier.coef_)
distance = df/w_norm
print(distance)
