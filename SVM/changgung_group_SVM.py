"""
This script is used to train an SVM classifier on two datasets.
The two groups are Changgung patients or healthy subjects.
"""

import os, sys
import numpy as np
import mmdps_locale
from mmdps.proc import atlas
from mmdps_util import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

if __name__ == '__main__':
	atlasobj = atlas.get('aal')

	# load specific nets as a list
	ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'SCI_incomplete_subjects.txt'))
	ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))

	# prepare all training 
	sig_connections = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, ChanggungHealthyNets)

	X1 = np.zeros((len(ChanggungHealthyNets), 1)) # healthy
	y1 = -1 * np.ones((len(ChanggungHealthyNets), 1))
	X2 = np.zeros((len(ChanggungPatientNets), 1)) # patient
	y2 = np.ones((len(ChanggungPatientNets), 1))
	for c in sig_connections:
		normalCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungHealthyNets)
		X1 = np.insert(X1, 0, normalCList, axis = 1)
		patientCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungPatientNets)
		X2 = np.insert(X2, 0, patientCList, axis = 1)
	X = np.concatenate([X1[:, :-1], X2[:, :-1]])
	y = np.concatenate((y1, y2)).ravel()
	print('Sample data shape: %s, label shape: %s' % (str(X.shape), str(y.shape)))

	# classifier
	classifier = svm.SVC(kernel = 'linear')

	# leave one out cross validation
	accuracy = []
	truePositive = 0
	falsePositive = 0
	trueNegative = 0
	falseNegative = 0
	loo = model_selection.LeaveOneOut()
	for trainIdx, testIdx in loo.split(X, y):
		classifier.fit(X[trainIdx, :], y[trainIdx])
		accuracy.append(classifier.score(X[testIdx, :], y[testIdx]))
		p = classifier.predict(X[testIdx, :])[0]
		if p == 1 and y[testIdx] == 1:
			truePositive += 1
		elif p == -1 and y[testIdx] == -1:
			trueNegative += 1
		elif p == 1 and y[testIdx] == -1:
			falsePositive += 1
		else:
			falseNegative += 1
	print('LOOCV averaged accuracy: %1.4f' % np.mean(accuracy))
	precision = float(truePositive)/(truePositive + falsePositive)
	recall = float(truePositive)/(truePositive + falseNegative)
	specificity = float(trueNegative)/(trueNegative + falsePositive)
	print('Precision: %1.4f, recall/sensitivity: %1.4f, specificity: %1.4f' % (precision, recall, specificity))

	classifier = svm.SVC(kernel = 'linear')
	score, permutation_scores, pvalue = model_selection.permutation_test_score(classifier, X, y, scoring="accuracy", cv=loo, n_permutations=1000, n_jobs=6)

	print("Classification score %s (pvalue : %1.10f)" % (score, pvalue))