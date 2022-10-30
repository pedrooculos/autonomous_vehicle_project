from operator import itemgetter
import cv2 as cv
import imageHandler
import numpy as np
import time
from scipy.interpolate import interp1d

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


# Implementation based in seancyw`s implementation: https://github.com/seancyw/bsnake_lane_detection
def run_bSnake(imagePath: str, min_threshold_canny=50, max_threshold_canny=150, process_slices=5, hough_threshold=50):
    
    def find_lines(edges_slices: list, hough_threshold: int, min_line_length=5, max_line_gap=50):
        lines = []
        for i in range(len(edges_slices)):
            hough_result = cv.HoughLinesP(edges_slices[i], 1, np.pi/180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
            
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

    def find_vanish_row(lines: list, slice_size):
        vanish_votes = {}
        image_height = 0 
        for slice in range(len(lines)):
            for i in range(len(lines[slice])):
                for j in range(i+1,len(lines[slice])):
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
        #print("The voting result was: " + str(vanish_row) + " With: " + str(max_votes) + " votes")
        return vanish_row

    def find_lanes_lines(lines: list, slice_size: int, vanish_row: int):
        lane_lines = []
        image_height=0
        for slice in range(len(lines)):
            lane_lines_slice = []
            for i in range(len(lines[slice])):
                for j in range(i+1, len(lines[slice])):                  
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

    def find_control_points(lane_lines: list, slice_size: int, image_center: int):
        def find_center_lines(lines):
            right_lanes = []; left_lanes = []
            for line in lines:
                try:
                    slope = (line[3] - line[1])/(line[2] - line[0])
                    if (slope > 0):
                        right_lanes.append(line)
                    else:
                        left_lanes.append(line)
                except:
                    print("Error: This is a vertical line!!!")

            right_lanes = sorted(right_lanes, key=itemgetter(3))
            left_lanes = sorted(left_lanes, key=itemgetter(3), reverse=True)         

            center_lanes = [right_lanes[0], left_lanes[0]]

            return center_lanes
            
        control_points = []
        height = 0
        for slice in range(len(lane_lines)):
            if (2 < len(lane_lines[slice])):
                center_lanes = find_center_lines(lane_lines[slice])
                control_point  = [(center_lanes[0][2] + center_lanes[1][2])/2, height]
            else:
                control_point = [image_center, height]

            #print(control_point)
            control_points.append(control_point)
            height += slice_size
        
        control_point = [image.shape[0]/2, height]
        control_points.append(control_point)
    
        return control_points 
    
    def calculate_spline(control_points, spline_kind = 'cubic'):
        x = control_points[:][1]
        y = control_points[:][0]

        spline_function = interp1d(x,y, kind=spline_kind)

        return spline_function
    
    image = cv.imread(imagePath)
    blured_img = cv.medianBlur(image, 9)
    edges = cv.Canny(blured_img, min_threshold_canny, max_threshold_canny)
    #imageHandler.plot_image(edges)

    edges_slices = imageHandler.split_image_vertical(edges, process_slices)

    lines = find_lines(edges_slices, hough_threshold)
    imageHandler.create_image_for_test(image, lines)
    
    slice_size = round(image.shape[0]/process_slices)
    vanish_row = find_vanish_row(lines, slice_size)
    
    lane_lines = find_lanes_lines(lines, slice_size, vanish_row)
    imageHandler.create_image_for_test(image, lane_lines)
    
    find_control_points(lane_lines, slice_size, image_center=int(image.shape[0]/2))
    #imageHandler.create_image_for_test(image, lane_lines)

    calculate_spline()

if __name__ == '__main__':
    start_time = time.time()
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cabc30fc-eb673c5a.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data_teste/images/b1c81faa-c80764c5.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cb86b1d9-7735472c.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/caee33ed-136d6b7c.jpg')
    run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cbd35eab-7eacd236.jpg')
    print("Time in seconds: %s" % (time.time() - start_time))