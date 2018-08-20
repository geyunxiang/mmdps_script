"""
This script is used to train an SVM classifier on two datasets.
The two groups are Changgung patients or healthy subjects.
After training, 
"""

import os, json
import numpy as np
import mmdps_locale
from mmdps_old import brain_net, brain_template
from mmdps_old.utils import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

# load specific nets as a list

ChanggungSCINets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, template_name = 'brodmann_lr_3', subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'SCI_incomplete_subjects.txt'))

# ChanggungCSNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungCSFullPath)
# ChanggungCSNets += io_utils.loadSpecificNets(mmdps_locale.ChanggungCS_other_FullPath)
# ChanggungCSNets += io_utils.loadSpecificNets(mmdps_locale.ChanggungCS_other_2_FullPath)

ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, template_name = 'brodmann_lr_3', subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))

ChanggungPatientNets = ChanggungSCINets
# prepare all training 
sig_connections = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, ChanggungHealthyNets)
X1 = np.zeros((len(ChanggungHealthyNets), 1)) # healthy
y1 = np.ones((len(ChanggungHealthyNets), 1))
X2 = np.zeros((len(ChanggungPatientNets), 1)) # patient
y2 = -1 * np.ones((len(ChanggungPatientNets), 1))
for c in sig_connections:
	c = c.split('-')
	normalCList = result_utils.getAllFCAtTick(c[0], c[1], ChanggungHealthyNets)
	X1 = np.insert(X1, 0, normalCList, axis = 1)
	patientCList = result_utils.getAllFCAtTick(c[0], c[1], ChanggungPatientNets)
	X2 = np.insert(X2, 0, patientCList, axis = 1)
X = np.concatenate([X1[:, :-1], X2[:, :-1]])
y = np.concatenate((y1, y2)).ravel()
print('Sample data shape: %s, label shape: %s' % (str(X.shape), str(y.shape)))

# classifier
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X, y)
print('training accuracy: %1.4f' % classifier.score(X, y))

# leave one out cross validation
accuracy = []
loo = model_selection.LeaveOneOut()
for trainIdx, testIdx in loo.split(X, y):
	classifier.fit(X[trainIdx, :], y[trainIdx])
	accuracy.append(classifier.score(X[testIdx, :], y[testIdx]))
print('LOOCV averaged accuracy: %1.4f' % np.mean(accuracy))
