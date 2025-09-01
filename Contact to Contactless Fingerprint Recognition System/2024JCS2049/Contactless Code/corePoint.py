import math
import os
import json
import cv2 
import cv2 as cv

def match_corepoints(self, image_path, json_database_file_path):
        return self.CHECK_FINGERPRINT(image_path, json_database_file_path)

def distance(self, p1, p2):
    return math.sqrt(abs(p1[0] - p2[0]) ** 2 + abs(p1[0] - p2[0]) ** 2)

def save_munitia_points_in_database(self):
    src_folder = self.directories[9]
    dest_folder = self.directories[11]

    for file in os.listdir(src_folder):

        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(dest_folder, file.split(".")[0] + ".json")

        image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            self.SAVE_DATA(image, dest_path)
            # cv2.imwrite(dest_path, ret_image)
        else:
            print(f"Warning: Could not read {src_path} as an image.")

def save_data(self, distance_vector, core_point, munitia_points, output_file):
    # Prepare the data dictionary
    data = {
        "distance_vector_core_as_origin": distance_vector,
        "core_point": list(core_point),
        "munitia_points": list(munitia_points)
    }

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

def load_data(self, input_file):
    with open(input_file, "r") as json_file:
        data = json.load(json_file)
    return data

def SAVE_DATA(self, thined_image_file, output_file="output.json"):
        im = thined_image_file

        ending_points, bifurcation_points = [], []

        result = self.get_core_point(im, 480)
        # print(result)
        # Adjust kernel_size as needed
        kernel_size = 3

        points = self.get_minutia_points(im, kernel_size)
        for pt in points:
            if pt[2] == "ending":
                ending_points.append((pt[0], pt[1]))
            if pt[2] == "bifurcation":
                bifurcation_points.append((pt[0], pt[1]))

        if len(result) == 0:
            result = [[(0, 0), (0, 0)], (0, 0)]
        # print(result)

        side_threshold = 30
        distance_threshold = 15
        ending_points_filtered = []
        for pt in ending_points:
            flag = True
            if side_threshold > pt[0] or pt[0] > im.shape[0] - side_threshold:
                continue
            if side_threshold > pt[1] or pt[1] > im.shape[1] - side_threshold:
                continue

            for pt2 in ending_points:
                if math.sqrt(abs(pt[0] - pt2[0]) ** 2 + abs(pt[1] - pt2[1]) ** 2) < distance_threshold and pt2 != pt:
                    flag = False
                    break
            if flag:
                ending_points_filtered.append(pt)

        bifurcation_points_filtered = []
        for pt in bifurcation_points:
            flag = True
            if side_threshold > pt[0] or pt[0] > im.shape[0] - side_threshold:
                continue
            if side_threshold > pt[1] or pt[1] > im.shape[1] - side_threshold:
                continue
            for pt2 in bifurcation_points:
                if math.sqrt(abs(pt[0] - pt2[0]) ** 2 + abs(pt[1] - pt2[1]) ** 2) < distance_threshold and pt2 != pt:
                    # and side_threshold < pt[0] < im.shape[0] - side_threshold\
                    # and side_threshold < pt[1] < im.shape[1] - side_threshold:
                    flag = False
                    break
            if flag:
                bifurcation_points_filtered.append(pt)

        im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        for (x, y) in ending_points_filtered:
            cv2.circle(im_color, (x, y), radius=3, color=(255, 64, 64), thickness=2)

        for (x, y) in bifurcation_points_filtered:
            cv2.circle(im_color, (x, y), radius=3, color=(0, 64, 0), thickness=2)

        cv2.circle(im_color, result[0][0], radius=3, color=(0, 0, 255), thickness=2)
        cv2.circle(im_color, result[0][1], radius=3, color=(0, 0, 255), thickness=2)
        cv2.circle(im_color, result[1], radius=3, color=(0, 255, 255), thickness=2)

        final_list = []
        final_list.extend(ending_points_filtered)
        final_list.extend(bifurcation_points_filtered)
        core_point = result[1]
        distance_vector = [self.distance(core_point, a) for a in final_list]


        self.save_data(distance_vector=distance_vector, core_point=core_point, munitia_points=final_list,
                       output_file=output_file)

def CHECK_FINGERPRINT(self, thined_image_file, database_file="output.json"):
    im = cv.imread(thined_image_file, cv.IMREAD_GRAYSCALE)

    ending_points, bifurcation_points = [], []

    result = self.get_core_point(im, 480)
    kernel_size = 3

    points = self.get_minutia_points(im, kernel_size)
    for pt in points:
        if pt[2] == "ending":
            ending_points.append((pt[0], pt[1]))
        if pt[2] == "bifurcation":
            bifurcation_points.append((pt[0], pt[1]))

    side_threshold = 30
    distance_threshold = 15
    ending_points_filtered = []
    for pt in ending_points:
        flag = True
        if side_threshold > pt[0] or pt[0] > im.shape[0] - side_threshold:
            continue
        if side_threshold > pt[1] or pt[1] > im.shape[1] - side_threshold:
            continue

        for pt2 in ending_points:
            if math.sqrt(abs(pt[0] - pt2[0]) ** 2 + abs(pt[1] - pt2[1]) ** 2) < distance_threshold and pt2 != pt:
                flag = False
                break
        if flag:
            ending_points_filtered.append(pt)

    bifurcation_points_filtered = []
    for pt in bifurcation_points:
        flag = True
        if side_threshold > pt[0] or pt[0] > im.shape[0] - side_threshold:
            continue
        if side_threshold > pt[1] or pt[1] > im.shape[1] - side_threshold:
            continue
        for pt2 in bifurcation_points:
            if math.sqrt(abs(pt[0] - pt2[0]) ** 2 + abs(pt[1] - pt2[1]) ** 2) < distance_threshold and pt2 != pt:
                # and side_threshold < pt[0] < im.shape[0] - side_threshold\
                # and side_threshold < pt[1] < im.shape[1] - side_threshold:
                flag = False
                break
        if flag:
            bifurcation_points_filtered.append(pt)

    im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for (x, y) in ending_points_filtered:
        cv2.circle(im_color, (x, y), radius=3, color=(255, 64, 64), thickness=2)
        pass
    for (x, y) in bifurcation_points_filtered:
        cv2.circle(im_color, (x, y), radius=3, color=(0, 64, 0), thickness=2)
        pass

    cv2.circle(im_color, result[0][0], radius=3, color=(0, 0, 255), thickness=2)
    cv2.circle(im_color, result[0][1], radius=3, color=(0, 0, 255), thickness=2)
    cv2.circle(im_color, result[1], radius=3, color=(0, 255, 255), thickness=2)

    cv2.imshow("Thinned (Skeletonized) Image", im_color)
    cv2.waitKey(0)
    final_list = []
    final_list.extend(ending_points_filtered)
    final_list.extend(bifurcation_points_filtered)
    core_point = result[1]

    distance_vector = [self.distance(core_point, a) for a in final_list]


    loaded_data = self.load_data(database_file)

    if (loaded_data["distance_vector_core_as_origin"]) == distance_vector:
        print("SUCCESSFULL AUTHENTICATION")
        return True
    else:
        print("AUTHENTICATION FAILED")
        return False