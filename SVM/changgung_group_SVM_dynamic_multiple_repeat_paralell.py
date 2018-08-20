"""
This script is used to train an SVM classifier on two datasets.
The two groups are Changgung patients or healthy subjects.
Dynamic networks are used to balance the difference in positive
and negative sample sizes. 

This script will run the LOOCV multiple times to calculate the averaged 
accuracy etc. 
"""

import os, sys, multiprocessing, queue, datetime
import numpy as np
from mmdps.proc import atlas, parabase
from mmdps_util import stats_utils, io_utils, result_utils
from sklearn import svm, model_selection

# atlasList = ['brodmann_lr', 'brodmann_lrce', 'aicha', 'bnatlas']
atlasList = ['aal']
ChanggungAllFullPath = 'I:/geyunxiang/Changgung processed/BOLD/'
ChanggungRootPath = 'I:/geyunxiang/Changgung processed/'

def func(args):
	# input parameters
	atlasobj = args[0]
	ChanggungPatientNets = args[1]

	# return results
	ret = [None, None, None, None, None] # discover rate, accuracy, precision, recall, specificity

	ChanggungHealthyNets = io_utils.loadRandomDynamicNets(ChanggungAllFullPath, atlasobj, totalNum = len(ChanggungPatientNets), scanList = os.path.join(ChanggungRootPath, 'normal_scans.txt'))
	sig_connections = stats_utils.filter_sigdiff_connections(ChanggungPatientNets, ChanggungHealthyNets)
	ret[0] = float(len(sig_connections))/(atlasobj.count*(atlasobj.count-1)/2.0)
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
	ret[1] = np.mean(accuracy)
	
	precision = float(truePositive)/(truePositive + falsePositive)
	recall = float(truePositive)/(truePositive + falseNegative)
	specificity = float(trueNegative)/(trueNegative + falsePositive)
	ret[2] = precision
	ret[3] = recall
	ret[4] = specificity
	return ret

if __name__ == '__main__':
	outfile = open('result.txt', 'w', buffering = 1)
	for atlasName in atlasList:
		atlasobj = atlas.get(atlasName)
		outfile.write('processing atlas: %s\n' % atlasobj.name)
		
		# load specific nets as a list
		ChanggungPatientNets = io_utils.loadSpecificNets(ChanggungAllFullPath, atlasobj, subjectList = os.path.join(ChanggungRootPath, 'SCI_incomplete_subjects.txt'))
		totalIteration = 1000
		numCPU = 6

		with multiprocessing.Pool(processes = numCPU) as pool:
			manager = multiprocessing.Manager()
			managerQueue = manager.Queue()
			fwrap = parabase.FWrap(func, managerQueue)
			result = pool.map_async(fwrap.run, [(atlasobj, ChanggungPatientNets)] * totalIteration)
			ntotal = totalIteration
			print('Begin proc, {} cpus, {} left, start at {}'.format(numCPU, ntotal, datetime.datetime.now()))
			nfinished = 0
			while True:
				if result.ready():
					break
				else:
					try:
						res = managerQueue.get(timeout=1)
					except queue.Empty:
						continue
					else:
						nfinished += 1
						print('just finished one. {} left, at {}'.format(ntotal - nfinished, datetime.datetime.now()))
			print('End proc, end at {}'.format(datetime.datetime.now()))
			outputs = result.get()
		totalAccuracy = []
		totalPrecision = []
		totalSensitivity = []
		totalSpecificity = []
		totalDiscoveryRate = []
		for res in outputs:
			totalDiscoveryRate.append(res[0])
			totalAccuracy.append(res[1])
			totalPrecision.append(res[2])
			totalSensitivity.append(res[3])
			totalSpecificity.append(res[4])
		
		# permutation test
		ChanggungHealthyNets = io_utils.loadRandomDynamicNets(ChanggungAllFullPath, atlasobj, totalNum = len(ChanggungPatientNets), scanList = os.path.join(ChanggungRootPath, 'normal_scans.txt'))
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
		classifier = svm.SVC(kernel = 'linear')
		loo = model_selection.LeaveOneOut()
		score, permutation_scores, pvalue = model_selection.permutation_test_score(classifier, X, y, scoring="accuracy", cv=loo, n_permutations=1000, n_jobs=numCPU)

		print("Classification score %s (pvalue : %1.10f)" % (score, pvalue))
		outfile.write('Averaged discover rate: %1.4f, max: %1.4f, min: %1.4f, std: %1.4f\n' % (np.mean(totalDiscoveryRate), np.max(totalDiscoveryRate), np.min(totalDiscoveryRate), np.std(totalDiscoveryRate)))
		outfile.write('Averaged accuracy: %1.4f, max: %1.4f, min: %1.4f, std: %1.4f\n' % (np.mean(totalAccuracy), np.max(totalAccuracy), np.min(totalAccuracy), np.std(totalAccuracy)))
		outfile.write('Averaged precision: %1.4f, max: %1.4f, min: %1.4f, std: %1.4f\n' % (np.mean(totalPrecision), np.max(totalPrecision), np.min(totalPrecision), np.std(totalPrecision)))
		outfile.write('Averaged recall: %1.4f, max: %1.4f, min: %1.4f, std: %1.4f\n' % (np.mean(totalSensitivity), np.max(totalSensitivity), np.min(totalSensitivity), np.std(totalSensitivity)))
		outfile.write('Averaged specificity: %1.4f, max: %1.4f, min: %1.4f, std: %1.4f\n' % (np.mean(totalSpecificity), np.max(totalSpecificity), np.min(totalSpecificity), np.std(totalSpecificity)))
		outfile.write('Classification score %s (pvalue : %1.10f)\n\n' % (score, pvalue))
	outfile.close()
