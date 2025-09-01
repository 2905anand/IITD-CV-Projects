import numpy as np
import csv
import os

def read_minutiae_csv(file_path):
    
    points = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            orientation = float(row["orientation"])
            
            points.append([x, y, orientation])
    return np.array(points, dtype=float)

def hough_align(template_minutiae, query_minutiae, angle_bin_size=5, translation_bin_size=10):
    
    candidates = []
    for q in query_minutiae:
        for t in template_minutiae:
            
            candidate_angle = t[2] - q[2]
            
            R = np.array([[np.cos(candidate_angle), -np.sin(candidate_angle)],
                          [np.sin(candidate_angle),  np.cos(candidate_angle)]])
            
            candidate_translation = t[:2] - np.dot(R, q[:2])
            candidates.append([candidate_angle, candidate_translation[0], candidate_translation[1]])
    candidates = np.array(candidates, dtype=float)

    
    angle_bin_rad = np.deg2rad(angle_bin_size)
    angle_bins = np.arange(-np.pi, np.pi + angle_bin_rad, angle_bin_rad)
    translation_bins = np.arange(-500, 500 + translation_bin_size, translation_bin_size)
    
    
    H, edges = np.histogramdd(candidates, bins=(angle_bins, translation_bins, translation_bins))
    idx = np.unravel_index(np.argmax(H), H.shape)
    best_angle = (edges[0][idx[0]] + edges[0][idx[0] + 1]) / 2
    best_tx = (edges[1][idx[1]] + edges[1][idx[1] + 1]) / 2
    best_ty = (edges[2][idx[2]] + edges[2][idx[2] + 1]) / 2

    return best_angle, best_tx, best_ty

def apply_transformation(query_minutiae, angle, tx, ty):
    
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    transformed = query_minutiae.copy()
    for i in range(query_minutiae.shape[0]):
        point = query_minutiae[i, :2]
        new_point = np.dot(R, point) + np.array([tx, ty])
        transformed[i, :2] = new_point
        
        transformed[i, 2] = query_minutiae[i, 2] + angle
    return transformed

def match_minutiae(template, aligned_query, distance_threshold=25, angle_threshold=np.deg2rad(20)):
    
    matched = 0
    used_template = np.zeros(template.shape[0], dtype=bool)
    for q in aligned_query:
        for j, t in enumerate(template):
            if used_template[j]:
                continue
            dist = np.linalg.norm(q[:2] - t[:2])
            diff = abs(q[2] - t[2])
            diff = min(diff, 2 * np.pi - diff)
            if dist < distance_threshold and diff < angle_threshold:
                matched += 1
                used_template[j] = True
                break
    return matched

def generalized_hough_match(template_minutiae, query_minutiae):
    
    best_angle, best_tx, best_ty = hough_align(template_minutiae, query_minutiae)
    aligned_query = apply_transformation(query_minutiae, best_angle, best_tx, best_ty)
    matched = match_minutiae(template_minutiae, aligned_query)
    score = matched / ((template_minutiae.shape[0] + query_minutiae.shape[0]) / 2)
    return best_angle, best_tx, best_ty, score

if __name__ == '__main__':
    
    template_csv = "./contact_minutiae_points/240_minutiae.csv"    
    query_csv = "./contactless_minutiae_points/140_minutiae.csv"    

    
    if not os.path.exists(template_csv):
        print(f"Error: {template_csv} does not exist.")
        exit(1)
    if not os.path.exists(query_csv):
        print(f"Error: {query_csv} does not exist.")
        exit(1)

    
    template_minutiae = read_minutiae_csv(template_csv)
    query_minutiae = read_minutiae_csv(query_csv)

    
    est_angle, est_tx, est_ty, similarity_score = generalized_hough_match(template_minutiae, query_minutiae)

    print("Estimated Transformation:")
    print("Rotation (degrees): {:.2f}".format(np.rad2deg(est_angle)))
    print("Translation: ({:.2f}, {:.2f})".format(est_tx, est_ty))
    print("Matching Score: {:.2f}".format(similarity_score))
