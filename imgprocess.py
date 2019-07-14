from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import pickle
import math
import imutils
from imutils import perspective
import scipy.ndimage as ndimage
import numpy as np
import cv2
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import feature
import os

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

class ImageProcess:
	def _init_(self,img):
		self.img = img

	def process(self):
		#img = cv2.imread("images\\Leaves\\chinese tulip tree\\3535.jpg")
		data = []
		label = []
		Features = []
		scale = 50
		
		img = cv2.resize(self.img, (512, 512))
		img = np.array(img, dtype=np.uint8)
		img = ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)
		# Grayimage
		grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# thresolding
		ret, binary = cv2.threshold(grayimg, 199, 255, cv2.THRESH_BINARY)
		str = "Binary"

		# edge detection
		# 0.33 is choosen by data science

		v = np.median(binary)
		low = int(max(0, (1 - 0.5) * v))
		upper = int(min(255, (1 + 0.5) * v))
		edged = cv2.Canny(binary, low, upper)
		edged = cv2.dilate(edged, None, iterations=1)
		edged = cv2.erode(edged, None, iterations=1)

		# find contours in the edge map
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# sort the contours from left-to-right and initialize the
		# 'pixels per metric' calibration variable
		# cnts,_ = contours.sort_contours(cnts)
		pixelsPerMetric = None
		# loop over the contours individually
		for c in cnts:
		    # if the contour is not sufficiently large, ignore it
		    if cv2.contourArea(c) < 200:
		        continue

		    # compute the rotated bounding box of the contour
		    orig = img.copy()
		    box = cv2.minAreaRect(c)
		    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		    box = np.array(box, dtype=int)

		    # order the points in the contour such that they appear
		    # in top-left, top-right, bottom-right, and bottom-left
		    # order, then draw the outline of the rotated bounding
		    # box
		    box = perspective.order_points(box)
		    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		    rc = cv2.minAreaRect(cnts[0])
		    box = cv2.boxPoints(rc)
		    # loop over the original points and draw them
		    for (x, y) in box:
		        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
		    # unpack the ordered bounding box, then compute the midpoint
		    # between the top-left and top-right coordinates, followed by
		    # the midpoint between bottom-left and bottom-right coordinates
		    (tl, tr, br, bl) = box
		    (tltrX, tltrY) = midpoint(tl, tr)
		    (blbrX, blbrY) = midpoint(bl, br)
		    # compute the midpoint between the top-left and top-right points,
		    # followed by the midpoint between the top-righ and bottom-right
		    (tlblX, tlblY) = midpoint(tl, bl)
		    (trbrX, trbrY) = midpoint(tr, br)

		    # length and width
		    a = (tl + tr) / 2
		    b = (bl + br) / 2
		    c = (tl + bl) / 2
		    d = (tr + br) / 2
		    #print(a, b, c, d)
		    width = math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
		    length = math.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2))
		    #print(length)

		    # draw the midpoints on the image
		    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
		# contours
		contours, hierarchy = cv2.findContours(binary, 1, 2)

		cnts = contours[0]
		M = cv2.moments(cnts)
		# print(M)
		if M["m00"] != 0:
		    cX = int(M["m10"] / M["m00"])
		    cY = int(M["m01"] / M["m00"])
		else:
		    # set values as what you need in the situation
		    cX, cY = 0, 0
		# finding centroid
		# cx = int(M['m10']/M['m00'])
		# cy = int(M['m01']/M['m00'])


		# area, perimeter, epsilon to remove spaces
		area = cv2.contourArea(cnts)
		perimeter = cv2.arcLength(cnts, True)

		# epsilon is maximum distance from contour to approximated contour. It is an accuracy parameter.
		epsilon = 0.1*cv2.arcLength(cnts, True)
		approx = cv2.approxPolyDP(cnts, epsilon, True)

		# convex hull
		hull = cv2.convexHull(cnts)

		# convexity
		k = cv2.isContourConvex(cnts)

		# rotetad rectangle
		rect = cv2.minAreaRect(cnts)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		im = cv2.drawContours(binary, [box], 0, (0, 0, 255), 2)

		# aspect ratio
		x, y, w, h = cv2.boundingRect(cnts)
		width = w
		length = h
		aspect_ratio = float(length)/width

		# extent
		area = cv2.contourArea(cnts)
		rect_area = width*length
		extent = float(area)/rect_area

		# solidity
		area = cv2.contourArea(cnts)
		hull = cv2.convexHull(cnts)
		hull_area = cv2.contourArea(hull)
		#print(hull_area)
		if(hull_area == 0):
		    hull_area = 1
		    #print(hull_area)
		solidity = float(area)/hull_area

		# mask and pixel points
		mask = np.zeros(binary.shape, np.uint8)
		cv2.drawContours(mask, [cnts], 0, 255, -1)
		pixelpoints = np.transpose(np.nonzero(mask))
		# pixelpoints = cv2.findNonZero(mask)'''
		# extreme points
		leftmost = tuple(cnts[cnts[:, :, 0].argmin()][0])
		rightmost = tuple(cnts[cnts[:, :, 0].argmax()][0])
		topmost = tuple(cnts[cnts[:, :, 1].argmin()][0])
		bottommost = tuple(cnts[cnts[:, :, 1].argmax()][0])

		# form factor
		form_factor = (3.14 * 4 * cv2.contourArea(cnts)) / (perimeter * perimeter)

		# rectengularity
		rectengulaerity = (length * width) / area

		'''
		print(area, perimeter, aspect_ratio, form_factor, rectengulaerity, epsilon)
		#show points
		font = cv2.FONT_HERSHEY_SIMPLEX
		plt.subplot(2, 3, 1), plt.imshow(orig, 'gray')
		plt.title(str)
		plt.xticks([])
		plt.yticks([])
		plt.show()'''

		# smooth factor
		k = np.ones((5, 5), np.float32)/25
		dst = cv2.blur(grayimg, (5, 5))
		k1 = np.ones((2, 2), np.float32)/4
		dst1 = cv2.blur(grayimg, (2, 2))

		ret, binary1 = cv2.threshold(dst, 199, 255, cv2.THRESH_BINARY, 0)
		ret, binary2 = cv2.threshold(dst1, 199, 255, cv2.THRESH_BINARY, 0)
		contours1, hierarchy = cv2.findContours(binary1, 1, 2)
		contours2, hierarchy = cv2.findContours(binary2, 1, 2)
		cnt1 = contours1[0]
		a1 = cv2.contourArea(cnt1)
		# print(a1)
		cnt2 = contours2[0]
		a2 = cv2.contourArea(cnt2)
		# print(a2)
		smooth_factor = a1 / a2

		# narrow factor
		D = width
		Narrow_Factor = D / length

		# perimeter ratio of diameter
		PROFD = perimeter / D

		# Perimeter ratio of length and width
		PLW = perimeter / (length + width)

		# circularity
		circularity = ((perimeter) ** 2) / area

		# Color features
		red_channel = img[:, :, 0]
		green_channel = img[:, :, 1]
		blue_channel = img[:, :, 2]
		blue_channel[blue_channel == 255] = 0
		green_channel[green_channel == 255] = 0
		red_channel[red_channel == 255] = 0

		red_mean = np.mean(red_channel)
		green_mean = np.mean(green_channel)
		blue_mean = np.mean(blue_channel)

		red_std = np.std(red_channel)
		green_std = np.std(green_channel)
		blue_std = np.std(blue_channel)

		Features = [D, length, area, hull_area, aspect_ratio, form_factor, smooth_factor, rectengulaerity, perimeter, Narrow_Factor, PROFD, PLW, solidity, circularity, \
		                  red_mean, green_mean, blue_mean, red_std, green_std, blue_std]

		# Local Binary Pattern
		kneighbours = 3
		noofpoints = 8 * kneighbours
		hist = desc.describe(grayimg)
		hist = cv2.normalize(hist, hist)
		hist = hist.flatten()

		# data.append(hist)
		data.append(np.concatenate((hist, Features), axis=None))


		f = open("traindata.pkl", 'rb')
		g = open("trainlabel.pkl", 'rb')
		trainData = pickle.load(f)
		trainLabels = pickle.load(g)
		f.close()
		g.close()
		# scale the input image pixels to the range [0, 1], then transform
		# the labels into vectors in the range [0, num_classes] -- this
		# generates a vector for each label where the index of the label
		# is set to `1` and all other entries to `0`
		data = np.array(data) / 255.0
		#label2 = imagePaths.split("\\")[-2]
		#label.append(label2)
		# partition the data into training and testing splits, using 75%
		# of the data for training and the remaining 25% for testing
		# print("Training started:")
		#(trainData, testData, trainLabels, testLabels) = train_test_split(data, label, test_size=0.25, random_state=42)
		# print(testLabels)
		scores = []
		knn = KNeighborsClassifier(n_neighbors=3)
		knn.fit(trainData, trainLabels)
		scores.append(knn.score(trainData, trainLabels))
		# pred = knn.predict(data)
		pred = knn.predict(data.reshape(1, -1))[0]
		#if pred != "":
		cv2.putText(img, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
		#else:
		#    cv2.putText(img, "Not Found", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
		cv2.imshow('Result', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		cv2.waitKey(-1)
		print(pred)
