from array import array
from email.mime import image
from keras_preprocessing.image import load_img, img_to_array
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import cv2 as cv
import numpy as np
import math

def load_image_pixels(filename, shape):
  # load image to get its shape
  image = load_img(filename)
  width, height = image.size

  # load image with required size
  image = load_img(filename, target_size=shape)
  image = img_to_array(image)

  # grayscale image normalization
  image = image.astype('float32')
  image /= 255.0

  # add a dimension so that we have one sample
  image = expand_dims(image, 0)
  return image, width, height

def draw_lines_from_houghP(image, lines):
	if lines is not None:
		for line in lines:
			pt1 = (line[0], line[1])
			pt2 = (line[2], line[3])
			cv.line(image, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

def plot_image(image):
	img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
	pyplot.imshow(img_rgb, cmap = 'gray')
	pyplot.show()

def split_image_vertical(image: cv.Mat, split_number: int):

	heigth, width = image.shape[:2]
	
	image_slices = []
	for i in range(split_number):
		image_slices.append(image[((heigth//split_number)*i):((heigth//split_number)*(i+1)), 0:width])

	return image_slices

def merge_image_vertical(image_slices: array):
	image = np.concatenate((image_slices[0],image_slices[1]), axis=0)
	for i in range(2, len(image_slices)):
		image = np.concatenate((image, image_slices[i]), axis=0)

	return image

def create_image_for_test(image_source, lines):

    image_slices = split_image_vertical(image_source, len(lines))

    for i in range(len(image_slices)):
        draw_lines_from_houghP(image_slices[i], lines[i])
    
    image = merge_image_vertical(image_slices)
    plot_image(image)


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
  
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth = '2')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='red')
	# show the plot
	pyplot.show()
