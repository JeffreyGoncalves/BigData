import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def copy_pic(picture) :
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	new_pic = np.zeros((lines, columns, 3))
	for x in range(lines) :
		for y in range(lines) : 
			for p in range(3) :
				new_pic[x][y][p] = int(picture[x][y][p])
				print(new_pic[x][y][p],picture[x][y][p])
	return new_pic

def increase_contrast(picture, contrast) :
	new_pic = copy_pic(picture)
	plt.imshow(new_pic)
	plt.show()
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	f =(259 * (contrast + 255)) / (255 * (259 - contrast))
	for x in range(lines) :
		for y in range(columns) :
			r = f*(picture[x][y][0]-128) + 128
			g = f*(picture[x][y][1]-128) + 128
			b = f*(picture[x][y][2]-128) + 128
			if r < 0 or r > 255 :
				if r < 0 : r = 0
				else : 	   r = 255
			if g < 0 or g > 255 :
				if g < 0 : g = 0
				else : 	   g = 255
			if b < 0 or b > 255 :
				if b < 0 : b = 0
				else : 	   b = 255
			new_pic[x][y][0] = r
			new_pic[x][y][1] = g
			new_pic[x][y][2] = b
	return new_pic

train_data = loadmat("../Data/train_32x32.mat")
test_data = loadmat("../Data/test_32x32.mat")

for image_idx in range(len(train_data["y"])) :
	new_pic = increase_contrast(train_data["X"][:, :, :, image_idx], 50)
	plt.imshow(train_data["X"][:, :, :, image_idx])
	plt.show()
	plt.imshow(new_pic)
	plt.show()
