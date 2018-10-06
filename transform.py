#coding: utf-8

'''
transform to produce pseudo parents
'''

from __future__ import division, print_function, unicode_literals
import numpy as np 

def get_pseudos(X, gen_type=0):
	'''
	return pseudo image
	'''
	if gen_type == 0:
		return X
	elif gen_type == 1:
		'''
		rotate 90, channel_last
		'''
		return np.rot90(X)
	elif gen_type == 2:
		'''
		rotate 180
		'''
		return np.rot90(np.rot90(X))
	elif gen_type == 3:
		'''
		rotate 270
		'''
		return np.rot90(np.rot90(np.rot90(X)))
	elif gen_type == 4:
		'''
		flip horizontally, channel_last
		'''
		return np.fliplr(X)
	elif gen_type == 5:
		'''
		flip horizontally + rotate90
		'''
		return np.rot90(np.fliplr(X))
	elif gen_type == 6:
		'''
		flip horizontally + rotate180
		'''
		return np.rot90(np.rot90(np.fliplr(X)))
	elif gen_type == 7:
		'''
		flip horizontally + rotate270
		'''
		return np.rot90(np.rot90(np.rot90(np.fliplr(X))))