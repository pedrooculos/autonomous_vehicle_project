import cv2 as cv
import imageHandler
import numpy as np
import time
import utils.houghBundler as houghBundler
import utils.gvf as gvf
from scipy import interpolate
from scipy.special import binom
import math
from operator import itemgetter

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

def create_bezier(control_points):
            # Code based in https://gist.github.com/astrojuanlu/7284462
    def Bernstein(n, k):
        """Bernstein polynomial.
        """
        coeff = binom(n, k)

        def _bpoly(x):
            return coeff * x ** k * (1 - x) ** (n - k)

        return _bpoly

    def Bezier(points, num=20):
        """Build Bézier curve from points.
        """
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for ii in range(N):
            curve += np.outer(Bernstein(N - 1, ii)(t), points[ii])
        return curve

    control_points = np.array(control_points)
    xp = control_points[:, 0]
    yp = control_points[:, 1]

    x, y = Bezier(list(zip(xp, yp))).T

    bezier = [ [x[i], y[i]]  for i in range(len(x))]
    
    return bezier        

def create_BSpline(control_points, num=100):
    points = np.array(control_points)

    x = points[:,0]
    y = points[:,1]

    degree = min(len(points), 3)
    
    t, c, k = interpolate.splrep(y, x, s=10, k=degree)

    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    y_new = np.linspace(y.min(), y.max(), num)
    x_new = spline(y_new)
    
    bsnake = np.dstack((x_new,y_new))

    return bsnake[0]


# Implementation based in seancyw`s implementation: https://github.com/seancyw/bsnake_lane_detection
def run_bSnake(image, min_threshold_canny=50, max_threshold_canny=100, process_slices=15, hough_threshold=15):
    
    def find_lines(edges_slices: list, hough_threshold: int, min_line_length=10, max_line_gap=20):
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

    def find_center_lane_marks(lines: list, slice_size: int, vanish_row: int, image_center_x):
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

    def estimate_k(lines, vanish_row):

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

    def find_path_points(image_shape, slice_size, lines, vanish_row):
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

    def find_control_points(points):
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

    def apply_gvf_minimization(image, bsnake, k, vanish_row):

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


    blured_img = cv.medianBlur(image, 9)
    edges = cv.Canny(blured_img, min_threshold_canny, max_threshold_canny)

    edges_slices = imageHandler.split_image_vertical(edges, process_slices)
    lines = find_lines(edges_slices, hough_threshold)
    
    slice_size = round(image.shape[0]/process_slices)
    vanish_row = find_vanish_row(lines, slice_size)

    center_lane_lines = find_center_lane_marks(lines, slice_size, vanish_row, image.shape[1]/2)

    k = estimate_k(center_lane_lines, vanish_row)

    points = find_path_points(image.shape, slice_size, center_lane_lines, vanish_row)
    
    control_points = find_control_points(points)

    bsnake = create_bezier(control_points)

    bsnake = apply_gvf_minimization(image, bsnake, k, vanish_row)

    return bsnake

def test_bSnake():
    image_number = 5
    match image_number:
        case 1:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/cabc30fc-eb673c5a.jpg'
        case 2:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data_teste/images/b1c81faa-c80764c5.jpg'
        case 3:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/cb86b1d9-7735472c.jpg'
        case 4:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/caee33ed-136d6b7c.jpg'
        case 5:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/cbd35eab-7eacd236.jpg'
        case 6:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data_teste/images/teste.jpg'
        case 7:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/cc599c2f-daa3587e.jpg'
        case 8:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/ccb961b6-66bd00c0.jpg'
        case 9:
            image_path = '/home/pedro/repos/autonomous_vehicle_project/data/images/test/cd75d282-a85a83e1.jpg'
        case _:
            "Passar uma imagem por favor!!!"
    
    image = cv.imread(image_path)
    bsnake_result = run_bSnake(image)  

    imageHandler.mark_points_for_test(image, bsnake_result)

if __name__ == '__main__':
    start_time = time.time()

    test_bSnake()

    print("Time in seconds: %s" % (time.time() - start_time))