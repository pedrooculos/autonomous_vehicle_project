import cv2 as cv
import imageHandler
import numpy as np
import time
import utils.houghBundler as houghBundler
import utils.gvf as gvf
import utils.spline as spline
import math
from operator import itemgetter

class Bsnake:
    def __init__(self, min_threshold_canny, max_threshold_canny,
                 process_slices, hough_threshold, min_line_length,
                 max_line_gap, median_blur_ksize) -> None:
        
        self.min_threshold_canny = min_threshold_canny
        self.max_threshold_canny = max_threshold_canny
        self.process_slices = process_slices
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.median_blur_ksize = median_blur_ksize     
        
    @classmethod
    def init_default(cls):
        bsnake = Bsnake(50, 100, 15, 15, 10, 20, 9)
        return bsnake

    def __find_lines(self, edges_slices: list):
        lines = []
        for i in range(len(edges_slices)):
            hough_result = cv.HoughLinesP(edges_slices[i], 1, np.pi/180, self.hough_threshold, 
                                          minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
            
            line_seg=[]
            if hough_result is not None:
                for j in range(len(hough_result)):
                    if ((hough_result[j][0][2] - hough_result[j][0][0]) == 0):
                        line_seg.append(hough_result[j][0])
                    else:
                        m = (hough_result[j][0][3] - hough_result[j][0][1])/(hough_result[j][0][2] - hough_result[j][0][0])

                        if not (-0.2<m<0.2):
                            line_seg.append(hough_result[j][0])

            lines.append(line_seg)
        return lines
    
    def __find_vanish_row(self, lines: list, slice_size: int):
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
        return vanish_row
    
    def __find_center_lane_marks(self, lines: list, slice_size: int, vanish_row: int, image_center_x: int):
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
            for slice in lines:
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

        center_lanes = []
        for slice in clean_lines:
            marks = sorted(slice, key=itemgetter(3))
            
            right_lane = []; left_lane = []; last_middle = 0
            for mark in marks:
                x_middle = (mark[2] + mark[0])/2
                if (x_middle > last_middle) and (x_middle < image_center_x):
                    right_lane = mark
                    last_middle = x_middle

                elif (x_middle > last_middle) and (x_middle > image_center_x):
                    left_lane = mark
                    break                

            if (len(right_lane) == 0) or (len(left_lane)==0):
                center_lanes.append([])
            else:
                center_lanes.append([right_lane, left_lane])

        return center_lanes
    
    def __estimate_k(self, lines: list, vanish_row: int):
        for slice in lines:
            if len(slice) >= 2:
                line_1 = slice[1]
                line_2 = slice[-1]

                a_2 = line_2[3] - line_2[1]
                b_2 = line_2[0] - line_2[2] 
                c_2 = a_2*line_2[0] + b_2*line_2[1]

                slope = a_2/b_2
                const = -c_2/b_2

                x_2 = line_1[1]/slope - const/slope

                k = (x_2 - line_1[0] )/(line_1[1] - vanish_row)

                return k 

        return 0
    
    def __find_path_points(self, image_shape: tuple, slice_size: int, lines: list, vanish_row: int):
        points = []
        points.append([image_shape[1]/2,image_shape[0] -1])
        
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
    
    def __find_control_points(self, points: list):
        Q0 = points[-1]
        Q2 = points[0]

        b0 = (points[-3][1] - points[-4][1]) / (points[-3][0] - points[-4][0]) if (points[-3][0] - points[-4][0]) else np.pi/2
        b1 = (points[-2][1] - points[-3][1]) / (points[-2][0] - points[-3][0]) if (points[-2][0] - points[-3][0]) else np.pi/2
        b2 = (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0]) if (points[-1][0] - points[-2][0]) else np.pi/2

        if (b0 == b1) and (b2 != b1):
            p1 = points[-2]
        elif (b2 == b1) and (b0 != b1):
            p1 = points[-3]
        else:
            p1 = [(points[-2][0] + points[-3][0])/2,(points[-2][1] + points[-3][1])/2]
            
        x = 3* p1[0]/2 - (Q0[0] + Q2[0])/4
        y = 3* p1[1]/2 - (Q0[1] + Q2[1])/4

        Q1 = [x, y]
        
        control_points = [Q0, Q1, Q2]
    
        return control_points 
    
    def __apply_gvf_minimization(self, image: cv.Mat, bsnake: np.ndarray, k: float, vanish_row: int):

        _, edge = gvf.prepare_img_for_gvf(image, sigma=2)

        fx, fy = gvf.gradient_field(edge)
        gx, _ = gvf.gradient_vector_flow(image, fx, fy, mu=1.0)

        # imageHandler.plot_gvf_images(image2, edge, fx,fy,gx,gy)

        lamb = 1
        alpha = 100
        interactions = 0
        while True:
            interactions += 1
            error = 0
            error_k = 0
            # let the first point aways in the middle of the vehicle
            for i in range(0,len(bsnake)):
                coord_x = int(bsnake[i][0])
                coord_y = int(bsnake[i][1])

                d = (k*(coord_y - vanish_row))/2

                c_right = int(coord_x + d)
                c_left = int(coord_y - d)

                force = gx[coord_y, c_right] + gx[coord_y, c_left]

                new_x = coord_x + alpha*force

                k_new = k + lamb*(gx[coord_y, c_right] - gx[coord_y, c_left])

                error_k += 0 * math.sqrt((k_new - k)**2)
                error += math.sqrt((new_x - coord_x)**2)
                bsnake[i][0] = new_x
                

            error /= len(bsnake)
            error_k /= len(bsnake)
            if(error <= 10**(-5)) and (error_k <= 10**(-5)):
                break

            if interactions > 50000:
                print("Too many interactions")
                break
            
        return bsnake

    def create_spline(self, control_points: list):
        control_points = np.array(control_points)
        xp = control_points[:, 0]
        yp = control_points[:, 1]

        x, y = spline.Bezier(list(zip(xp, yp))).T

        bezier = [ [x[i], y[i]]  for i in range(len(x))]
        
        return bezier

    def run(self, image: cv.Mat):

        image_shape = image.shape
        slice_size = round(image_shape[0]/self.process_slices)
        image_center_x = image_shape[1]/2

        blured_img = cv.medianBlur(image, self.median_blur_ksize)
        edges = cv.Canny(blured_img, self.min_threshold_canny, self.max_threshold_canny)
        edges_slices = imageHandler.split_image_vertical(edges, self.process_slices)

        lines = self.__find_lines(edges_slices)
        vanish_row = self.__find_vanish_row(lines, slice_size)
        center_lane_lines = self.__find_center_lane_marks(lines, slice_size, vanish_row, image_center_x)
        k = self.__estimate_k(center_lane_lines, vanish_row)
        points = self.__find_path_points(image_shape, slice_size, center_lane_lines, vanish_row)
        control_points = self.__find_control_points(points)

        bsnake = self.create_spline(control_points)

        bsnake = self.__apply_gvf_minimization(image, bsnake, k, vanish_row)

        return bsnake

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


def test_bSnake():
    image_path = '/home/pedro/repos/autonomous_vehicle_project/data/teste.jpg'
    
    image = cv.imread(image_path)

    bsnake = Bsnake.init_default()
    bsnake_result = bsnake.run(image)  

    imageHandler.mark_points_for_test(image, bsnake_result)

if __name__ == '__main__':
    start_time = time.time()

    test_bSnake()

    print("Time in seconds: %s" % (time.time() - start_time))