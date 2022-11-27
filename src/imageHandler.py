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

def mark_points_for_test(image_source, points):
	for point in points:
		mark = (int(point[0]), int(point[1]))
		image_source = cv.circle(image_source, mark, radius=0, color=(0,0,255), thickness=10)	
		
	plot_image(image_source)


def mark_lane_contours(image_source, points, k, vanish_row):
	for point in points:
		coord_x = int(point[0])
		coord_y = int(point[1])

		d = (k*(coord_y - vanish_row))/2

		c_right = int(coord_x + d)
		c_left = int(coord_y - d)


		mark_right = (int(c_right), int(coord_y))
		mark_left = (int(c_left), int(coord_y))
		image_source = cv.circle(image_source, mark_right, radius=0, color=(0,0,255), thickness=10)
		image_source = cv.circle(image_source, mark_left, radius=0, color=(0,0,255), thickness=10)
		
	plot_image(image_source)

def draw_images_from_points(image, x,y):

	print(x)
	print(y)
	pyplot.imshow(image)
	pyplot.scatter(x[:],y[:], color='red')

	pyplot.show()


def plot_gvf_images(img, edge, fx, fy, gx, gy):
	def plot_vector_field(ax, vx, vy):
		scale = np.sqrt(np.max(vx**2+vy**2))*20.0
		ax.imshow(img, cmap='gray')
        # vy shold be inversed (top=+Y -> top=-Y)
		ax.quiver(X, Y, vx[Y, X], -vy[Y, X], scale=scale, color='blue', headwidth=5)
    
	def vmin(values): return -max(values.max(), -values.min())
    
	def vmax(values): return max(values.max(), -values.min())

	H, W = img.shape	
	Y, X = np.meshgrid(range(0, H, 5), range(0, W, 5))
	fig, axs = pyplot.subplots(2, 4, figsize=(16, 8))
	fig.suptitle('Gradient Vector Flow (2D) demo')
	ax = axs[0][0]; ax.imshow(img, cmap='gray'); ax.set_title('org');ax.set_axis_off()
	ax = axs[0][1]; ax.imshow(edge[:, :], cmap='gray'); ax.set_title('edge');ax.set_axis_off()
	ax = axs[0][2]; ax.imshow(fx, vmin=vmin(fx), vmax=vmax(fx), cmap='seismic'); ax.set_title('fx');ax.set_axis_off()
	ax = axs[0][3]; ax.imshow(fy, vmin=vmin(fx), vmax=vmax(fx), cmap='seismic'); ax.set_title('fy');ax.set_axis_off()
	ax = axs[1][0]; ax.imshow(gx, vmin=vmin(gx), vmax=vmax(gx), cmap='seismic'); ax.set_title('GVFx');ax.set_axis_off()
	ax = axs[1][1]; ax.imshow(gy, vmin=vmin(gy), vmax=vmax(gy), cmap='seismic'); ax.set_title('GVFy');ax.set_axis_off()
	ax = axs[1][2]; plot_vector_field(ax, fx, fy); ax.set_title('f');ax.set_axis_off()
	ax = axs[1][3]; plot_vector_field(ax, gx, gy); ax.set_title('GVF');ax.set_axis_off()
	fig.tight_layout()
	fig, axs = pyplot.subplots(1, 2, figsize=(16, 9))
	fig.suptitle('Gradient Vector Flow (2D) demo')
	ax = axs[0]; plot_vector_field(ax, fx, fy); ax.set_title('f');ax.set_axis_off()
	ax = axs[1]; plot_vector_field(ax, gx, gy); ax.set_title('GVF');ax.set_axis_off()
	fig.tight_layout()
	pyplot.show()

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
