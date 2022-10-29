from array import array
from mailbox import linesep
import cv2 as cv
import imageHandler
import numpy as np

class streetModel:
    def __init__(self):
        pass

def line_intersection(line1, line2):
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(line1[:2], line1[2:]), det(line2[:2], line2[2:]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_center_lanes(lines, image_center):
    right_lanes_centers = []
    left_lanes_centers = []
    for line in lines:
        if(line[0]>image_center and line[2]>image_center):
            right_lanes_centers.append((line[0]+line[2])/2)
        elif(line[0]>image_center and line[2]>image_center):
            left_lanes_centers.append((line[0]+line[2])/2)
    
    right_lanes_centers = sorted(right_lanes_centers)
    left_lanes_centers = sorted(left_lanes_centers, reverse=True)

    print(left_lanes_centers)

    right_lane_center = right_lanes_centers[0]
    left_lane_center = left_lanes_centers[0]

    center_lane_lines = []
    curve_threshold = 50
    right_lanes = 0
    left_lanes = 0
    top_right = 0
    top_left = 0
    for line in lines:
        if(line[0]>image_center and line[2]>image_center):
            if ((line[0]+line[2])/2 < right_lane_center + curve_threshold):
                center_lane_lines.append(line)
                right_lanes += 1
                top_right += (line[2] - line[0])*(-line[1])/(line[3] - line[1])+line[0]
        elif(line[0]>image_center and line[2]>image_center):
            if ((line[0]+line[2])/2 < left_lane_center + curve_threshold):
                center_lane_lines.append(line)
                left_lanes += 1
                top_left += (line[2] - line[0])*(-line[1])/(line[3] - line[1])+line[0]

    top_right = int(top_right/right_lanes)
    top_left = int(top_left/left_lanes)

    lane_center = (top_right/top_left)/2

    print(lane_center)
    
    return lane_center

def create_image_for_test(image_source, lines):

    image_slices = imageHandler.split_image_vertical(image_source, len(lines))

    for i in range(len(image_slices)):
        imageHandler.draw_lines_from_houghP(image_slices[i], lines[i])
    
    image = imageHandler.merge_image_vertical(image_slices)
    imageHandler.plot_image(image)

# Implementation based in seancyw`s implementation: https://github.com/seancyw/bsnake_lane_detection
def run_bSnake(imagePath: str, min_threshold_canny=50, max_threshold_canny=150, process_slices=5, hough_threshold=50):
    
    def find_lines(edges_slices: list, hough_threshold: int):
        lines = []
        for i in range(len(edges_slices)):
            hough_result = cv.HoughLinesP(edges_slices[i], 1, np.pi/180, hough_threshold, minLineLength=5, maxLineGap=50)
            
            line_seg=[]
            if hough_result is not None:
                for j in range(len(hough_result)):
                    if ((hough_result[j][0][2] - hough_result[j][0][0]) == 0):
                        line_seg.append(hough_result[j][0])
                    else:
                        m = (hough_result[j][0][3] - hough_result[j][0][1])/(hough_result[j][0][2] - hough_result[j][0][0]) # Line angle in rad

                        if not (-0.2<m<0.2):
                            line_seg.append(hough_result[j][0])

            lines.append(line_seg)
        return lines

    def find_lanes_lines(lines: list, process_slices: int, slice_size: int, vanish_row: int):
        lane_lines = []
        image_height=0
        for slice in range(process_slices):
            lane_lines_slice = []
            for i in range(len(lines[slice])):
                for j in range(len(lines[slice])):
                    if i==j:
                        continue
                    
                    try:
                        _ , y_inter = line_intersection(lines[slice][i], lines[slice][j])
                        
                        intersection = (y_inter + image_height)
                        
                        if(-20<= intersection - vanish_row <=20):
                            lane_lines_slice.append(lines[slice][i])
                            lane_lines_slice.append(lines[slice][j])

                    except:
                        print("Lines do not intersect!!!!!")
            image_height+=slice_size
            lane_lines.append(lane_lines_slice)
        return lane_lines 

    def find_vanish_row(lines: list, process_slices: int, slice_size):
        vanish_votes = {}
        image_height = 0 
        for slice in range(process_slices):
            for i in range(len(lines[slice])):
                for j in range(len(lines[slice])):
                    if i==j:
                        continue
                    try:
                        _ , y_inter = line_intersection(lines[slice][i], lines[slice][j])
                        
                        vote = int((y_inter + image_height)//10) # This division allows to group votes
                        if str(vote) in vanish_votes.keys():
                            vanish_votes[str(vote)] += 1
                        else:
                            vanish_votes[str(vote)] = 1

                    except:
                        print("Lines do not intersect!!!!!")
            image_height+=slice_size

        max_votes = 0
        max_voted = ""
        for key in vanish_votes:
            if vanish_votes[key] > max_votes:
                max_votes = vanish_votes[key]
                max_voted = key

        vanish_row = int(max_voted)*10
        vanish_row_segment = vanish_row//slice_size

        print("The voting result was: " + str(vanish_row) + " With: " + str(max_votes) + " votes")

        return vanish_row

    def find_control_points():
        def find_center_lines():
            pass
        pass
    
    def calculate_spline():
        pass
    
    image = cv.imread(imagePath)
    blured_img = cv.medianBlur(image, 9)
    edges = cv.Canny(blured_img, min_threshold_canny, max_threshold_canny)
    #imageHandler.plot_image(edges)

    edges_slices = imageHandler.split_image_vertical(edges, process_slices)

    lines = find_lines(edges_slices, hough_threshold)
    #create_image_for_test(image, lines)
    
    slice_size = round(image.shape[0]/process_slices)
    vanish_row = find_vanish_row(lines, process_slices, slice_size)
    
    lane_lines = find_lanes_lines(lines, process_slices, slice_size, vanish_row)
    create_image_for_test(image, lane_lines)
    
    find_control_points()

    calculate_spline()

if __name__ == '__main__':
    run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cabc30fc-eb673c5a.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data_teste/images/b1c81faa-c80764c5.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cb86b1d9-7735472c.jpg')