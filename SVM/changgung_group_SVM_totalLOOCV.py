"""
This script is used to train an SVM classifier on two datasets.
The two groups are Changgung patients or healthy subjects.
Total LOOCV means that even during feature selection, the 
left out sample is not considered. (totally left out)
"""

import os, json, copy
import numpy as np
import mmdps_locale
from mmdps.proc import atlas
from mmdps_util import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

atlasobj = atlas.get('aicha')

# load specific nets as a list
ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'))
ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))

# classifier
classifier = svm.SVC(kernel = 'linear')

# leave one out cross validation
accuracy = []
discovery_rates = []
truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0
for pidx in range(len(ChanggungPatientNets)):
	# pidx stands for patient index
	trainingNets = copy.deepcopy(ChanggungPatientNets)
	leftOutNet = trainingNets[pidx]
	trainingNets.remove(leftOutNet)

	sig_connections = stats_utils.filter_sigdiff_connections(trainingNets, ChanggungHealthyNets)
	discovery_rates.append(float(len(sig_connections))/(atlasobj.count * (atlasobj.count - 1)/2))
	X1 = np.zeros((len(ChanggungHealthyNets), 1)) # healthy
	y1 = -1 * np.ones((len(ChanggungHealthyNets), 1))
	X2 = np.zeros((len(trainingNets), 1)) # patient
	y2 = np.ones((len(trainingNets), 1))
	z = []
	for c in sig_connections:
		normalCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungHealthyNets)
		X1 = np.insert(X1, 0, normalCList, axis = 1)
		patientCList = result_utils.getAllFCAtIdx(c[0], c[1], trainingNets)
		X2 = np.insert(X2, 0, patientCList, axis = 1)
		z.append(result_utils.getAllFCAtIdx(c[0], c[1], [leftOutNet])[0])
	X = np.concatenate([X1[:, :-1], X2[:, :-1]])
	y = np.concatenate((y1, y2)).ravel()
	print('Sample data shape: %s, label shape: %s' % (str(X.shape), str(y.shape)))
	# train ans score
	classifier.fit(X, y)
	p = classifier.score(np.array(z, ndmin = 2), np.array([1]))
	accuracy.append(p)
	if p == 1.0:
		truePositive += 1
	else:
		falseNegative += 1

for nidx in range(len(ChanggungHealthyNets)):
	# nidx stands for normal index
	trainingNets = copy.deepcopy(ChanggungHealthyNets)
	leftOutNet = trainingNets[nidx]
	trainingNets.remove(leftOutNet)

	sig_connections = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, trainingNets)
	discovery_rates.append(float(len(sig_connections))/(atlasobj.count * (atlasobj.count - 1)/2))
	X1 = np.zeros((len(trainingNets), 1)) # healthy
	y1 = -1 * np.ones((len(trainingNets), 1))
	X2 = np.zeros((len(ChanggungPatientNets), 1)) # patient
	y2 = np.ones((len(ChanggungPatientNets), 1))
	z = []
	for c in sig_connections:
		normalCList = result_utils.getAllFCAtIdx(c[0], c[1], trainingNets)
		X1 = np.insert(X1, 0, normalCList, axis = 1)
		patientCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungPatientNets)
		X2 = np.insert(X2, 0, patientCList, axis = 1)
		z.append(result_utils.getAllFCAtIdx(c[0], c[1], [leftOutNet])[0])
	X = np.concatenate([X1[:, :-1], X2[:, :-1]])
	y = np.concatenate((y1, y2)).ravel()
	print('Sample data shape: %s, label shape: %s' % (str(X.shape), str(y.shape)))
	# train ans score
	classifier.fit(X, y)
	p = classifier.score(np.array(z, ndmin = 2), np.array([-1]))
	accuracy.append(p)
	if p == 1.0:
		trueNegative += 1
	else:
		falsePositive += 1
print('tp: %d, fp: %d, tn: %d, fn: %d' % (truePositive, falsePositive, trueNegative, falseNegative))
print('LOOCV averaged accuracy: %1.4f, averaged discovery_rates: %1.4f' % (np.mean(accuracy), np.mean(discovery_rates)))
precision = float(truePositive)/(truePositive + falsePositive)
recall = float(truePositive)/(truePositive + falseNegative)
specificity = float(trueNegative)/(trueNegative + falsePositive)
print('Precision: %1.4f, recall/sensitivity: %1.4f, specificity: %1.4f' % (precision, recall, specificity))
