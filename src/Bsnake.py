import cv2 as cv
import imageHandler
import numpy as np

class streetModel:
    def __init__(self):
        pass

def run_bSnake(imagePath: str, min_thresh_canny=100, max_thresh_canny=200, process_slices=5):
    image = cv.imread(imagePath)
    
    edges = cv.Canny(image, min_thresh_canny, max_thresh_canny)

    edges_slices = imageHandler.slip_image_vertical(edges, process_slices)
    image_slices = imageHandler.slip_image_vertical(image, process_slices)

    for i in range(process_slices):
        # Using this 
        lines = cv.HoughLines(edges_slices[i], 1, np.pi / 180, 50, None, 0, 0, max_theta=(np.pi/3))
        lines = [lines, cv.HoughLines(edges_slices[i], 1, np.pi / 180, 50, None, 0, 0, min_theta=(2*np.pi/3), max_theta=np.pi)]
        print(lines)
        teste = imageHandler.draw_lines_from_hough(image_slices[i], lines)
        imageHandler.plot_image(teste)
    

if __name__ == '__main__':
    run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data_teste/images/b1c81faa-c80764c5.jpg')