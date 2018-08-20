"""
This script is used to calculate the dynamic networks
and store them somewhere proper.
"""

import os
import numpy as np

import mmdps_locale
from mmdps.proc import netattr, atlas
from mmdps.util.loadsave import load_nii, save_csvmat

class CalcDynamic:
	def __init__(self, atlasobj, volumename, img, outfolder, stepsize = 3, windowLength = 100):
		"""
		volumename = '3mm' is the name of the atlas volume
		img is the nii file loaded using load_nii function
		"""
		self.img = img
		self.atlasobj = atlasobj
		self.atlasimg = load_nii(atlasobj.get_volume(volumename)['niifile'])
		self.outfolder = outfolder
		self.stepsize = stepsize
		self.windowLength = windowLength

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
		timepoints = ts.shape[1] # number of total timepoints
		start = 0
		while start + self.windowLength < timepoints:
			tscorr = np.corrcoef(ts[:, start:start + self.windowLength])
			save_csvmat(self.outpath('corrcoef-%d.%d.csv' % (start, start + self.windowLength)), tscorr)
			start += self.stepsize

	def run(self):
		self.gen_net()

if __name__ == '__main__':
	atlasList = ['brodmann_lr', 'brodmann_lrce', 'aicha', 'bnatlas']
	volumename = '3mm'
	for subject in sorted(os.listdir(mmdps_locale.ChanggungAllFullPath)):
		print('building subject %s' % subject)
		img = load_nii(os.path.join(mmdps_locale.ChanggungAllFullPath, subject, 'pBOLD.nii'))
		for atlasname in atlasList:
			atlasobj = atlas.get(atlasname)
			outfolder = os.path.join(mmdps_locale.ChanggungAllFullPath, subject, atlasobj.name, 'bold_net')
			os.makedirs(outfolder, exist_ok = True)
			c = CalcDynamic(atlasobj, volumename, img, outfolder)
			c.run()