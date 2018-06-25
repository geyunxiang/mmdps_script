"""
This script is used to calculate the dynamic networks for healthy subjects
and store them somewhere proper.
"""

import os
from mmdps_old import brain_net
import mmdps_locale

normalList = []
with open(os.path.join(mmdps_locale.ChanggungRootPath, 'normal_scans.txt')) as f:
	normalList = f.readlines()

# calculate static and dynamic whole brain networks
for subject in normalList:
	subjectPath = os.path.join(mmdps_locale.ChanggungAllFullPath, subject)
	net = brain_net.BrainNet(net_config = {'template': 'brodmann_lr_3'}, raw_data_path = os.path.join(subjectPath, 'Filtered_4DVolume.nii'))
	dnet = brain_net.DynamicNet(net)
	dnet.generate_dynamic_nets()
	dnet.save_dynamic_nets(os.path.join(subjectPath, 'bold_net'))
exit()