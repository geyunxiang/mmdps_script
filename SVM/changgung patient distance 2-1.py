"""
This script is used to train an SVM classifier on Changgung datasets.
The training is performed on the first scanning session.
Also, the distance to the separating plane is calculated for the first and 
second session.
Then, the distance of the first and second session is correlated with scores.
The change of distances between the first and second session is correlated with
the change of scores. 
"""

import os, json
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import mmdps_locale
from mmdps.proc import atlas
from mmdps_util import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection
import scipy
from scipy.stats.stats import pearsonr

scoreList = ['lower limb movement', 'sensory', 'SCIM']
scoreMatrix = dict() # 3 scores X 4 sessions
scoreMatrix['kangcuiping'] = np.array([[28, 34], [166, 168], [60, 67]])
scoreMatrix['qiutongyu'] = np.array([[28, 37], [128, 128], [55, 55]])
scoreMatrix['zhangyuzhen'] = np.array([[18, 27], [161, 161], [33, 38]])
scoreMatrix['chenyifan'] = np.array([[28, 34], [176, 176], [55, 57]])
scoreMatrix['wangxiaoxia'] = np.array([[33, 38], [174, 186], [35, 64]])
scoreMatrix['fuchenhao'] = np.array([[19, 19], [188, 188], [70, 74]])
scoreMatrix['guoqian'] = np.array([[39, 42], [148, 148], [68, 70]])
scoreMatrix['limeihong'] = np.array([[41, 50], [136, 136], [62, 72]])
scoreMatrix['maoli'] = np.array([[50, 50], [177, 177], [77, 90]])

scoreMatrix['xiangyan'] = np.array([[23, 27], [150, 158], [44, 47]])
scoreMatrix['mahuibo'] = np.array([[26, 31], [200, 200], [34, 43]])
scoreMatrix['caochangsheng'] = np.array([[40, 43], [190, 190], [64, 76]])
scoreMatrix['renjie'] = np.array([[45, 46], [72, 72], [60, 64]])
scoreMatrix['hanjunling'] = np.array([[33, 37], [184, 184], [68, 72]])
scoreMatrix['hushufang'] = np.array([[30, 39], [168, 168], [49, 50]])
scoreMatrix['liujinghui'] = np.array([[36, 40], [180, 180], [62, 67]])
scoreMatrix['hongqingfeng'] = np.array([[32, 35], [65, 110], [19, 23]])
scoreMatrix['panqingbin'] = np.array([[37, 38], [200, 204], [33, 33]])

atlasobj = atlas.get('bnatlas')

ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'SCI_pan_selected_all.txt'))

ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))

# Union of the first and second session
ChanggungPatientNets2 = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'SCI_pan_selected_all.txt', timeCase = 2))
ChanggungHealthyNets2 = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt', timeCase = 2))

testNets = io_utils.loadAllTemporalNets(mmdps_locale.ChanggungAllFullPath, 2, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'SCI_pan_selected_all.txt'))
testNameList = []
for name in testNets:
	testNameList.append(name)

print('len of ChanggungPatientNets: %d' % len(ChanggungPatientNets))
print('len of testNets: %d' % len(testNets))

# prepare all training 
sig_connections = []
sig_connections1 = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, ChanggungHealthyNets)
# only first session
sig_connections = sig_connections1

# Union of the first and second session
sig_connections2 = stats_utils.filter_sigdiff_connections(ChanggungPatientNets2, ChanggungHealthyNets2)
sig_connections = list(set(sig_connections1 + sig_connections2))

X1 = np.zeros((len(ChanggungHealthyNets), 1)) # healthy
y1 = -1 * np.ones((len(ChanggungHealthyNets), 1)) # label = -1 for healthy
X2 = np.zeros((len(ChanggungPatientNets), 1)) # patient
y2 = np.ones((len(ChanggungPatientNets), 1)) # label = 1 for patients

# Z would be a dict of np.arrays
# The key is the subject name
# The value is a 2d time case X feature dim array
Z = dict()
y3 = dict()
for key in testNameList:
	Z[key] = np.zeros((len(testNets[key]), 1))
	y3[key] = -1 * np.ones((len(testNets[key]), 1))
for c in sig_connections:
	normalCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungHealthyNets)
	X1 = np.insert(X1, 0, normalCList, axis = 1)
	patientCList = result_utils.getAllFCAtIdx(c[0], c[1], ChanggungPatientNets)
	X2 = np.insert(X2, 0, patientCList, axis = 1)
	for key in testNameList:
		# for each subject
		testList = result_utils.getAllFCAtIdx(c[0], c[1], testNets[key])
		Z[key] = np.insert(Z[key], 0, testList, axis = 1)
X = np.concatenate([X1[:, :-1], X2[:, :-1]])
y = np.concatenate((y1, y2)).ravel()
for key in Z:
	Z[key] = Z[key][:, :-1]

# classifier
classifier = svm.SVC(C = 10, kernel = 'linear')

# train it on Changgung
classifier.fit(X, y)

# calculate distance
distanceMatrix = np.zeros((len(testNets), 2))
w_norm = np.linalg.norm(classifier.coef_)
idx = 0
for key in testNameList:
	print('processing subject ' + key)
	df = classifier.decision_function(Z[key])
	distanceMatrix[idx, :] = df / w_norm
	# distance = df / w_norm
	idx += 1

# paired t-test to see if there is group difference between the first and second session
t, tp = scipy.stats.ttest_rel(distanceMatrix[:, 0], distanceMatrix[:, 1])
print('Paired t-test for the distances of first and second session: t = %1.3f, p = %1.8f' % (t, tp))

# tao, taop = scipy.stats.kendalltau(distance, range(len(distance)))
# r, rp = scipy.stats.spearmanr(distance, range(len(distance)))
# print('%s tao: %f, %f, rho: %f, %f' % (str(distance), tao, taop, r, rp))

# Plot the distance and scores to see linear trend
# Note only the second session is plotted, since the first session contains support vectors
# x: distance
# y: score
x = distanceMatrix[:, 1]
y = np.array([scoreMatrix[name][0, 1] for name in testNameList])
b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b+m*x, '-')
# plt.scatter(x, y)
plt.title('2nd session correlation lower limb movement r = %1.3f, p = %1.3f' % pearsonr(x, y))
plt.xlabel('distance')
plt.ylabel('score')
plt.savefig('C:/Users/geyx/Desktop/2018 paper/distance score/SCI 2-1/2nd session lower limb movement bnatlas.png')
plt.clf()

# Plot the difference of distance and scores
x = distanceMatrix[:, 1] - distanceMatrix[:, 0]
y = np.array([scoreMatrix[name][0, 1] - scoreMatrix[name][0, 0] for name in testNameList])
b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b+m*x, '-')
# plt.scatter(x, y)
plt.title('Difference correlation lower limb movement r = %1.3f, p = %1.3f' % pearsonr(x, y))
plt.xlabel('distance diff')
plt.ylabel('score diff')
plt.savefig('C:/Users/geyx/Desktop/2018 paper/distance score/SCI 2-1/Diff 2-1 lower limb movement bnatlas.png')
