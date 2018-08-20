"""
This script is used to train an SVM classifier on Changgung datasets, 
and test on Beijing/Cambridge or other datasets.
One of the two training groups must be Changgung healthy subjects.
All of the training data are the first scan of each patient.
The healthy dynamic networks were used to balance positive and negative ratio
"""

import os, json
import numpy as np
import mmdps_locale
from mmdps.proc import atlas, netattr
from mmdps_util import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

atlasobj = atlas.get('brodmann_lr')

# load specific nets as a list
ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'))
ChanggungHealthyNets = io_utils.loadRandomDynamicNets(mmdps_locale.ChanggungAllFullPath, atlasobj, totalNum = len(ChanggungPatientNets), scanList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_scans.txt'))

testNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'), timeCase = 2)

print('len of ChanggungPatientNets: %d' % len(ChanggungPatientNets))
print('len of testNets: %d' % len(testNets))

# prepare all training 
sig_connections = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, ChanggungHealthyNets)
X1 = np.zeros((len(ChanggungHealthyNets), 1)) # healthy
y1 = -1 * np.ones((len(ChanggungHealthyNets), 1)) # label = -1 for healthy
X2 = np.zeros((len(ChanggungPatientNets), 1)) # patient
y2 = np.ones((len(ChanggungPatientNets), 1)) # label = 1 for patients
Z = np.zeros((len(testNets), 1)) # 3rd party test set
y3 = np.ones((len(testNets), 1))
for c in sig_connections:
	normalCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungHealthyNets)
	X1 = np.insert(X1, 0, normalCList, axis = 1)
	patientCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungPatientNets)
	X2 = np.insert(X2, 0, patientCList, axis = 1)
	testList = result_utils.getAllFCAtIdx(c[0], c[1], testNets)
	Z = np.insert(Z, 0, testList, axis = 1)
X = np.concatenate([X1[:, :-1], X2[:, :-1]])
y = np.concatenate((y1, y2)).ravel()
Z = Z[:, :-1]

# classifier
classifier = svm.SVC(kernel = 'linear')

# train it on the first scans
classifier.fit(X, y)

# test it on other dataset
accuracy = classifier.score(Z, y3)
print('Test accuracy: %1.4f' % accuracy)
