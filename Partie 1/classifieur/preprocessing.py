import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat

def toGrayScale(picture) :
	new_picture = picture.copy()
	lines = np.shape(new_picture)[0]
	columns = np.shape(new_picture)[1]
	for i in range(lines) :
		for j in range(columns) :
			mean = int(new_picture[i][j][0] * 0.299 + new_picture[i][j][1] * 0.587 + new_picture[i][j][2] * 0.114)
			new_picture[i][j][0] = int(mean)
			new_picture[i][j][1] = int(mean)
			new_picture[i][j][2] = int(mean)
	return new_picture

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
	

def dynamicExpansion(picture) :
	bwPicture = toGrayScale(picture)
	lines = np.shape(bwPicture)[0]
	columns = np.shape(bwPicture)[1]
	total_pixels = lines*columns
	new_values = []
	cumulHist = cumulativeHistogram(bwPicture)
	borneMin = -1
	borneMax = -1
	# Detection des bornes min et max
	for i in range(256) :
		if borneMin == -1 :
			if cumulHist[i] >= total_pixels*0.01 :
				borneMin = i
		if borneMax == -1 :
			if total_pixels - cumulHist[255-i] >= total_pixels*0.01 :
				borneMax = 255-i
	# Creation du tableau des valeurs de pixels
	for i in range(256) :
		new_values.append(np.floor(255*(i-borneMin)/(borneMax-borneMin)))
	# Remplacement des valeurs dans la nouvelle image
	for x in range(lines) :
		for y in range(columns) :
			max_value = np.amax(bwPicture[x][y])
			trans_value = 255/(lines*columns) * cumulHist[max_value]
			if max_value != 0 :
				factor = trans_value/max_value
				for p in range(3) :
					bwPicture[x][y][p] = np.round(bwPicture[x][y][p]*factor)
			else : 
				for p in range(3) :
					bwPicture[x][y][p] = 0
	return bwPicture;
	

def equalize(picture) :
	new_pic = picture.copy()
	cumulHist = cumulativeHistogram(new_pic)
	lines = np.shape(picture)[0]
	columns = np.shape(picture)[1]
	for i in range(lines) :
		for j in range(columns) :
			max_value = np.amax(new_pic[i][j])
			trans_value = 255/(lines*columns) * cumulHist[max_value]
			if max_value != 0 :
				factor = trans_value/max_value
				for p in range(3) :
					new_pic[i][j][p] = np.round(new_pic[i][j][p]*factor)
			else : 
				for p in range(3) :
					new_pic[i][j][p] = 0
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

def preprocessing(filepath, newFilepath) :
	data = loadmat(filepath).copy()
	for i in range(len(data['y'])) :
		pic = binarise(data['X'][:,:,:,i])
		data['X'][:,:,:,i] = pic
		if i==0 :
			plt.imshow(data['X'][:,:,:,i])
			plt.show()
		if i%1000 == 0 :
			print("Avancement : ", i)
	savemat(newFilepath, data)

if name == "__main__" :
	preprocessing("", "")
