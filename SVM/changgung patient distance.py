"""
This script is used to train an SVM classifier on Changgung datasets.
Also, the distance to the separating plane is calculated.
"""

import os, json
import numpy as np
import mmdps_locale
from mmdps.proc import atlas
from mmdps_util import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection
import scipy

atlasobj = atlas.get('aal')

ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'))

ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))

testNets = io_utils.loadAllTemporalNets(mmdps_locale.ChanggungAllFullPath, 4, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'))

print('len of ChanggungPatientNets: %d' % len(ChanggungPatientNets))
print('len of testNets: %d' % len(testNets))

# prepare all training 
sig_connections = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, ChanggungHealthyNets)
X1 = np.zeros((len(ChanggungHealthyNets), 1)) # healthy
y1 = -1 * np.ones((len(ChanggungHealthyNets), 1)) # label = -1 for healthy
X2 = np.zeros((len(ChanggungPatientNets), 1)) # patient
y2 = np.ones((len(ChanggungPatientNets), 1)) # label = 1 for patients

# Z would be a dict of np.arrays
# The key is the subject name
# The value is a 2d time case X feature dim array
Z = dict()
y3 = dict()
for key in testNets:
	Z[key] = np.zeros((len(testNets[key]), 1))
	y3[key] = -1 * np.ones((len(testNets[key]), 1))
for c in sig_connections:
	normalCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungHealthyNets)
	X1 = np.insert(X1, 0, normalCList, axis = 1)
	patientCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungPatientNets)
	X2 = np.insert(X2, 0, patientCList, axis = 1)
	for key in testNets:
		# for each subject
		testList = result_utils.getAllFCAtIdx(c[0], c[1], testNets[key])
		Z[key] = np.insert(Z[key], 0, testList, axis = 1)
X = np.concatenate([X1[:, :-1], X2[:, :-1]])
y = np.concatenate((y1, y2)).ravel()
for key in Z:
	Z[key] = Z[key][:, :-1]

# classifier
classifier = svm.SVC(kernel = 'linear')

# train it on Changgung
classifier.fit(X, y)

# calculate distance
w_norm = np.linalg.norm(classifier.coef_)
for key in Z:
	print('processing subject ' + key)
	df = classifier.decision_function(Z[key])
	distance = df / w_norm
	tao, taop = scipy.stats.kendalltau(distance, range(len(distance)))
	r, rp = scipy.stats.spearmanr(distance, range(len(distance)))
	print('%s tao: %f, %f, rho: %f, %f' % (str(distance), tao, taop, r, rp))
