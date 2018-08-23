"""
This script is used to plot distances together with clinical scores
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, model_selection
import scipy

import mmdps_locale
from mmdps.proc import atlas
from mmdps_util import stats_utils, io_utils, result_utils

atlasList = ['brodmann_lr', 'brodmann_lrce', 'aal', 'aicha', 'bnatlas']
distanceMatrix = dict() # 5 atlases X 4 sessions

# SCI
# scoreList = ['lower limb movement', 'sensory', 'SCIM']
# scoreMatrix = dict() # 3 scores X 4 sessions
# scoreMatrix['kangcuiping'] = np.array([[28, 34, 38, 44], [166, 168, 168, 168], [60, 67, 71, 75]])
# scoreMatrix['qiutongyu'] = np.array([[28, 37, 40, 45], [128, 128, 164, 164], [55, 65, 80, 90]])
# scoreMatrix['mahuibo'] = np.array([[26, 31, 35, 36], [199, 200, 200, 200], [34, 43, 50, 54]])
# scoreMatrix['guoqian'] = np.array([[38, 41, 44, 40], [138, 140, 144, 144], [60, 67, 73, 80]])
# scoreMatrix['hongqingfeng'] = np.array([[32, 35, 42, 45], [65, 110, 110, 136], [19, 23, 37, 71]])

# CS
scoreList = ['FMA', 'ARAT', 'WOLF']
scoreMatrix = dict() # 3 scores X 4 sessions
scoreMatrix['daihuachen'] = np.array([[14, 24, 24, 37], [3, 10, 11, 13], [9, 26, 25, 34]])
scoreMatrix['guojiye'] = np.array([[18, 37, 42, 59], [17, 44, 47, 54], [32, 52, 60, 70]])
scoreMatrix['tanenci'] = np.array([[4, 13, 19, 29], [0, 2, 3, 13], [16, 19, 19, 21]])
scoreMatrix['wangjingying'] = np.array([[4, 9, 9, 15], [0, 1, 1, 1], [16, 17, 17, 17]])

for atlasName in atlasList:
	atlasobj = atlas.get(atlasName)

	ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'))
	ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))
	# testNets = io_utils.loadAllTemporalNets(mmdps_locale.ChanggungAllFullPath, 4, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'SCI_incomplete_subjects.txt'))
	testNets = io_utils.loadAllTemporalNets(mmdps_locale.ChanggungAllFullPath, 4, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'), specificTime = {'guojiye':['20170324', '20170426', '20170629', '20170728'], 'tanenci':['20170601', '20170706', '20170922', '20171117'], 'wangjingying': ['20170413', '20170606', '20170704', '20170921']})

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
		if key not in distanceMatrix:
			distanceMatrix[key] = np.zeros((5, 4))
		distanceMatrix[key][atlasList.index(atlasName), :] = distance

# plotting
for subject in distanceMatrix:
	if subject not in scoreMatrix:
		continue
	# reverse distance
	distanceMatrix[subject] = distanceMatrix[subject] * -1
	for scoreIdx in range(len(scoreList)):
		plt.plot(range(1, 5), (distanceMatrix[subject][0, :] - np.mean(distanceMatrix[subject][0, :]))/np.std(distanceMatrix[subject][0, :]))
		plt.plot(range(1, 5), (distanceMatrix[subject][1, :] - np.mean(distanceMatrix[subject][1, :]))/np.std(distanceMatrix[subject][1, :]))
		plt.plot(range(1, 5), (distanceMatrix[subject][2, :] - np.mean(distanceMatrix[subject][2, :]))/np.std(distanceMatrix[subject][2, :]))
		plt.plot(range(1, 5), (distanceMatrix[subject][3, :] - np.mean(distanceMatrix[subject][3, :]))/np.std(distanceMatrix[subject][3, :]))
		plt.plot(range(1, 5), (distanceMatrix[subject][4, :] - np.mean(distanceMatrix[subject][4, :]))/np.std(distanceMatrix[subject][4, :]))
		
		plt.plot(range(1, 5), (scoreMatrix[subject][scoreIdx, :] - np.mean(scoreMatrix[subject][scoreIdx, :]))/np.std(scoreMatrix[subject][scoreIdx, :]), linewidth = 4, color = 'brown')

		plt.legend(('Brodmann', 'Brodmann+ce', 'AAL', 'AICHA', 'Brainnetome', scoreList[scoreIdx]))
		plt.title('Distance changes with %s scores' % scoreList[scoreIdx])
		plt.xlabel('Scanning session')
		plt.xticks(range(1, 5))
		plt.ylabel('Centered values')
		plt.grid(True)
		plt.savefig('C:/Users/geyx/Desktop/2018 paper/distance score/CS/%s %s.png' % (subject, scoreList[scoreIdx]))
		plt.clf()
