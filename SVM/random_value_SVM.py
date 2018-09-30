"""
This script is used to train an SVM classifier on two datasets.
The two groups are randomly generated noises with high (~400-700)
feature dimension.
Group size: 22, 38, 16
"""

import os, json, random
import numpy as np
import mmdps_locale
from mmdps_old import brain_net, brain_template
from mmdps_old.utils import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

# try for several times and average the LOOCV averaged accuracy
allAccuracy = 0
for testCase in range(100):
	# prepare all training
	X = np.zeros((22+38, 400)) # training data
	y1 = np.ones((22, 1))
	y2 = -1*np.ones((38, 1))
	y = np.concatenate((y1, y2)).ravel()
	for subIdx in range(60):
		for featureIdx in range(400):
			X[subIdx, featureIdx] = random.uniform(-1, 1)

	print('Sample data shape: %s, label shape: %s' % (str(X.shape), str(y.shape)))

	# classifier
	classifier = svm.SVC(kernel = 'linear')

	# leave one out cross validation
	accuracy = []
	loo = model_selection.LeaveOneOut()
	for trainIdx, testIdx in loo.split(X, y):
		classifier.fit(X[trainIdx, :], y[trainIdx])
		accuracy.append(classifier.score(X[testIdx, :], y[testIdx]))
	# print('LOOCV averaged accuracy: %1.4f' % np.mean(accuracy))
	allAccuracy += np.mean(accuracy)
allAccuracy /= 100.0
print('averaged accuracy: %f' % allAccuracy)
