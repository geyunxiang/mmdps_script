import os
import numpy as np

from mmdps.proc import atlas
from mmdps.util.loadsave import load_nii, save_csvmat
from mmdps.util import path

import mmdps_locale

class Calc:
	def __init__(self, atlasobj, volumename, img, outfolder):
		self.img = img
		self.atlasobj = atlasobj
		self.atlasimg = load_nii(atlasobj.get_volume(volumename)['niifile'])
		self.outfolder = outfolder

	def outpath(self, *p):
		return os.path.join(self.outfolder, *p)

	def gen_timeseries(self):
		data = self.img.get_data()
		atdata = self.atlasimg.get_data()
		timepoints = data.shape[3]
		timeseries = np.empty((atlasobj.count, timepoints))
		for i, region in enumerate(atlasobj.regions):
			regiondots = data[atdata==region, :]
			regionts = np.mean(regiondots, axis=0)
			timeseries[i, :] = regionts
		return timeseries

	def gen_net(self):
		ts = self.gen_timeseries()
		save_csvmat(self.outpath('timeseries.csv'), ts)
		tscorr = np.corrcoef(ts)
		save_csvmat(self.outpath('corrcoef.csv'), tscorr)
		
	def run(self):
		self.gen_net()

if __name__ == '__main__':
	atlasobj = atlas.get('aicha')
	volumename = '3mm'
	for subject in sorted(os.listdir(mmdps_locale.ChanggungAllFullPath)):
		print('building subject %s' % subject)
		outfolder = os.path.join(mmdps_locale.ChanggungAllFullPath, subject, 'bold_net', 'aicha_3')
		os.makedirs(outfolder, exist_ok = True)
		img = load_nii(os.path.join(mmdps_locale.ChanggungAllFullPath, subject, 'Filtered_4DVolume.nii'))
		c = Calc(atlasobj, volumename, img, outfolder)
		c.run()