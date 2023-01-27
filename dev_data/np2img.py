import numpy as np
import os
import csv
from PIL import Image


def load_csv(path):
	reader = csv.reader(open(path, 'r'))
	next(reader)
	content_list = list(reader)
	return content_list

def np2img(input_matrix, filename):
	img = Image.fromarray(input_matrix.transpose(1, 2, 0), 'RGB')
	img = img.resize((299,299))
	img.save(os.path.join('result', filename))



if __name__ == '__main__':
	input_npy = np.load('SIM1.npy')
	content_list = load_csv('val_rs.csv')
	for idx in range(input_npy.shape[0]):
		np2img(input_npy[idx], content_list[idx][0])
