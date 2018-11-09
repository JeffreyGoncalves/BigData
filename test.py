import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def toGrayScale(picture) :
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	for i in range(lines) :
		for j in range(columns) :
			mean = int(picture[i][j][0] * 0.299 + picture[i][j][1] * 0.587 + picture[i][j][2] * 0.114)
			picture[i][j][0] = int(mean)
			picture[i][j][1] = int(mean)
			picture[i][j][2] = int(mean)
	return picture

def createHist(picture) :
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	histogram = []
	for p in range(256) :
		histogram.append(0)
	for i in range(lines) :
		for j in range(columns) :
			histogram[np.amax(picture[i][j])] += 1
	return histogram
	
def cumulativeHistogram(picture) :
	histogram = createHist(picture)
	cumulHist = []
	for p in range(256) :
		if p == 0 :
			cumulHist.append(histogram[0])
		else :
			cumulHist.append(cumulHist[p-1] + histogram[p])
	return cumulHist
	
def equalize(picture) :
	new_pic = picture.copy()
	cumulHist = cumulativeHistogram(new_pic)
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	for i in range(lines) :
		for j in range(columns) :
			max_value = np.amax(new_pic[i][j])
			trans_value = 255/(lines*columns) * cumulHist[max_value]
			factor = trans_value/max_value
			for p in range(3) :
				new_pic[i][j][p] = np.round(new_pic[i][j][p]*factor)
	return new_pic
	
def calc_mean(picture) :
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	value = 0
	for i in range(lines) :
		for j in range(columns) :
			value += int(np.mean(picture[i][j]))
	return int(value/(lines*columns))

def binarise(picture) :
	new_pic = picture.copy()
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	mean = calc_mean(picture)
	for i in range(lines) :
		for j in range(columns) :
			for p in range(3) :
				if new_pic[i][j][p] > mean :
					new_pic[i][j][p] = 255
				else :
					new_pic[i][j][p] = 0
	return new_pic

def increase_contrast(picture, contrast) :
	new_pic = picture.copy()
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
	toGrayScale(train_data["X"][:, :, :, image_idx])
	pic_test = equalize(train_data["X"][:, :, :, image_idx])
	pic_test_2 = binarise(train_data["X"][:, :, :, image_idx])
	#new_pic = increase_contrast(train_data["X"][:, :, :, image_idx], 100)
	plt.imshow(train_data["X"][:, :, :, image_idx])
	plt.show()
	plt.imshow(pic_test)
	plt.show()
	plt.imshow(pic_test_2)
	plt.show()
	#plt.imshow(new_pic)
