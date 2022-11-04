import cv2 as cv
import imageHandler
import numpy as np
import time
import utils.houghBundler as houghBundler

# class SolverMMSE:
#     def __init__(self):
        
#         m_i = np.array()
#         pass

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

def create_image_sections(image_height: int, number_of_sections: int):
    value = int(image_height/(number_of_sections**2))
    sections = []
    for i in range(number_of_sections):
        sections.append(value*i)


# Implementation based in seancyw`s implementation: https://github.com/seancyw/bsnake_lane_detection
def run_bSnake(imagePath: str, min_threshold_canny=50, max_threshold_canny=100, process_slices=15, hough_threshold=20):
    
    def find_lines(edges_slices: list, hough_threshold: int, min_line_length=10, max_line_gap=25):
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

    def find_center_lane_marks(lines: list, slice_size: int, vanish_row: int):
        def remove_lines_out_of_road(lines, slice_size, vanish_row):
            for i in range(len(lines)):
                if (vanish_row > slice_size*i):
                    lines[i] = []
            return lines
        
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
        
        def create_bundle_lines(lines):
            bundler = houghBundler.HoughBundler(min_distance=10,min_angle=5)

            lines_slices = []
            for slice in lane_lines:
                if (len(slice) == 0):
                    lines_slices.append([])
                    continue
                lines_slices.append(bundler.process_lines(slice))

            new_lines = []
            for i in range(len(lines_slices)):
                new_lines_2 = []
                for j in range(len(lines_slices[i])):
                    new_lines_2.append(lines_slices[i][j][0])
                new_lines.append(new_lines_2)
            return new_lines


        lane_lines = remove_lines_out_of_road(lines, slice_size, vanish_row)
        lane_lines = find_lanes_lines(lane_lines, slice_size, vanish_row)
        clean_lines = create_bundle_lines(lane_lines)

        # center_lanes = []
        # for slice in lines:
        #     if len(slice) == 0:
        #         center_lanes.append(slice)
        #         continue
        #     right_lanes = []; left_lanes = []
        #     for line in slice:
        #         try:
        #             slope = (line[3] - line[1])/(line[2] - line[0])
        #             if (slope > 0):
        #                 right_lanes.append(line)
        #             else:
        #                 left_lanes.append(line)
        #         except:
        #             print("Error: This is a vertical line!!!")

        #     right_lanes = sorted(right_lanes, key=itemgetter(3))
        #     left_lanes = sorted(left_lanes, key=itemgetter(3), reverse=True)         

        #     center_lane = [right_lanes[0], left_lanes[0]]
        #     center_lanes.append(center_lane)

        return clean_lines

    def find_path_points(image_shape, slice_size, lines, vanish_row):
        points = []
        points.append([image_shape[1]/2,image_shape[0]])
        
        heigth = image_shape[0] - slice_size
        for i in range(len(lines) -1, -1, -1):
            if (len(lines[i]) == 0):
                if heigth < vanish_row:
                    continue
                points.append([points[-1][0], heigth])
                heigth -= slice_size
            else:
                intersect_point = line_intersection(lines[i][0], lines[i][-1])
                if (intersect_point[0] - points[-1][0]) == 0:
                    points.append([points[-1][0], heigth])  
                else:
                    m = (intersect_point[1] - points[-1][1])/(intersect_point[0] - points[-1][0])
                    b = points[-1][1] - m*intersect_point[0]
                    points.append([(heigth - b)/m, heigth])

                heigth -= slice_size

        return points
        
    def find_control_points(points):

        Q0 = points[-1]
        Q2 = points[0]

        if (points[-1][1] - points[-2][1]) == 0:
            p1 = points[-3]
        elif (points[-2][1] - points[-3][1]):
            p1 = points[-2]
        else:
            p1 = [(points[-2][0] + points[-3][0])/2,(points[-2][1] + points[-3][1])/2]
            
        x = 3* p1[0]/2 - (Q0[0] + Q2[0])/4
        y = 3* p1[1]/2 - (Q0[1] + Q2[1])/4

        Q1 = [x, y]
        
        control_points = [Q0, Q1, Q2]
    
        return control_points 
    
    
    image = cv.imread(imagePath)

    blured_img = cv.medianBlur(image, 9)
    edges = cv.Canny(blured_img, min_threshold_canny, max_threshold_canny)
    #imageHandler.plot_image(edges)

    edges_slices = imageHandler.split_image_vertical(edges, process_slices)
    lines = find_lines(edges_slices, hough_threshold)
    #imageHandler.create_image_for_test(image, lines)
    
    slice_size = round(image.shape[0]/process_slices)
    vanish_row = find_vanish_row(lines, slice_size)

    center_lane_lines = find_center_lane_marks(lines, slice_size, vanish_row)
    # imageHandler.create_image_for_test(image, center_lane_lines)

    points = find_path_points(image.shape, slice_size, center_lane_lines, vanish_row)
    imageHandler.mark_points_for_test(image, points)
    
    control_points = find_control_points(points)
    #imageHandler.mark_points_for_test(image, control_points)


if __name__ == '__main__':
    start_time = time.time()
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cabc30fc-eb673c5a.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data_teste/images/b1c81faa-c80764c5.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cb86b1d9-7735472c.jpg')
    #run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/caee33ed-136d6b7c.jpg')
    run_bSnake('/home/pedro/repos/autonomous_vehicle_project/data/images/test/cbd35eab-7eacd236.jpg')
    print("Time in seconds: %s" % (time.time() - start_time))