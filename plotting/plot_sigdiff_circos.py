"""
This script is used to plot circos graphs of significant different
connections for one person.
"""

import os
import numpy as np

import mmdps_locale
from mmdps_util import stats_utils, io_utils
from mmdps.proc import atlas, netattr
from mmdps.util.loadsave import load_csvmat, save_csvmat
from mmdps.vis import braincircos

atlasobj = atlas.get('aicha')

ChanggungPatientNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'CS_subjects.txt'))

ChanggungHealthyNets = io_utils.loadSpecificNets(mmdps_locale.ChanggungAllFullPath, atlasobj, subjectList = os.path.join(mmdps_locale.ChanggungRootPath, 'normal_subjects.txt'))

sig_connections = stats_utils.filter_sigdiff_connections_Bonferroni(ChanggungPatientNets, ChanggungHealthyNets)

sigDiffNet = netattr.Net(np.zeros((atlasobj.count, atlasobj.count)), atlasobj)
for conn in sig_connections:
	sigDiffNet.data[conn[0], conn[1]] = 1

title = 'CS_signet'
outfilepath = 'E:/Results/CS_signet/test.png'

builder = braincircos.CircosPlotBuilder(atlasobj, title, outfilepath)
builder.add_circoslink(braincircos.CircosLink(sigDiffNet))
builder.add_circosvalue(braincircos.CircosValue(netattr.Attr(np.random.uniform(size = atlasobj.count), atlasobj)))
builder.customizeSize('0.80', '10p')
builder.plot()
