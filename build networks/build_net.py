"""
This script is used to build brain networks, both
static network and dynamic network. 
"""

import os, json, gzip, shutil
from mmdps_old import brain_net
import mmdps_locale

net = brain_net.BrainNet(net_config = {'template': 'brodmann_lr_3'}, raw_data_path = os.path.join(mmdps_locale.ChanggungAllFullPath, 'caochangsheng_20161027', 'Filtered_4DVolume.nii'))
net.saveNet('E:/test_net/old/')
exit()
# calculate static and dynamic whole brain networks
counter = 0
total = len(list(os.listdir(mmdps_locale.HCPProcessedFullPath)))
for subject in sorted(os.listdir(mmdps_locale.HCPProcessedFullPath)):
	counter += 1
	print('%s, %d/%d, %f\r' % (subject, counter, total, float(counter)/total))
	subjectPath = os.path.join(mmdps_locale.HCPProcessedFullPath, subject)
	# statInfo = os.stat(os.path.join(subjectPath, 'rfMRI_REST1_LR.nii.gz'))
	# if statInfo.st_size < 10240:
	# 	continue
	# if not os.path.isfile(os.path.join(subjectPath, 'rfMRI_REST1_LR.nii')):
	# 	with gzip.open(os.path.join(subjectPath, 'rfMRI_REST1_LR.nii.gz'), 'rb') as f_in:
	# 		with open(os.path.join(subjectPath, 'rfMRI_REST1_LR.nii'), 'wb') as f_out:
	# 			shutil.copyfileobj(f_in, f_out)
	net = brain_net.BrainNet(net_config = {'template': 'brodmann_lr_2'}, raw_data_path = os.path.join(subjectPath, 'Filtered_4DVolume.nii'))
	net.saveNet(os.path.join(subjectPath, 'bold_net'))
	# dnet = brain_net.DynamicNet(net)
	# dnet.generate_dynamic_nets()
	# dnet.save_dynamic_nets(os.path.join(subjectPath, 'bold_net'))
exit()
