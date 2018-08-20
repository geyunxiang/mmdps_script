import os
import numpy as np
import matplotlib.pyplot as plt

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
		self.timeseries = np.empty((atlasobj.count, timepoints))
		for i, region in enumerate(atlasobj.regions):
			regiondots = data[atdata==region, :]
			regionts = np.mean(regiondots, axis=0)
			self.timeseries[i, :] = regionts

	def plot(self, tick1, tick2):
		idx1 = self.atlasobj.ticks.index(tick1)
		idx2 = self.atlasobj.ticks.index(tick2)
		plt.plot(range(1, 1+self.timeseries.shape[1]), self.timeseries[idx1, :] - np.mean(self.timeseries[idx1, :]))
		plt.plot(range(1, 1+self.timeseries.shape[1]), self.timeseries[idx2, :] - np.mean(self.timeseries[idx2, :]))
		corr = np.corrcoef(self.timeseries[(idx1, idx2), :])[0, 1]
		plt.title('%s %s-%s connection %1.3f' % (atlasobj.name, tick1, tick2, corr))
		plt.xlabel('Time points')
		plt.ylabel('Centered time series')
		plt.grid(True)
		plt.show()

	def gen_net(self):
		ts = self.gen_timeseries()
		save_csvmat(self.outpath('timeseries.csv'), ts)
		tscorr = np.corrcoef(ts)
		save_csvmat(self.outpath('corrcoef.csv'), tscorr)
		
	def run(self):
		self.gen_net()

if __name__ == '__main__':
	atlasobj = atlas.get('brodmann_lr')
	volumename = '3mm'
	for subject in sorted(os.listdir(mmdps_locale.ChanggungAllFullPath)):
		print('building subject %s' % subject)
		img = load_nii(os.path.join(mmdps_locale.ChanggungAllFullPath, subject, 'pBOLD.nii'))
		c = Calc(atlasobj, volumename, img, 'E:/Results/timeseries.png')
		c.gen_timeseries()
		c.plot('L4', 'R4')
		exit()