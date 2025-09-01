import csv
import cv2 as cv
import numpy as np
import os

def angle_diff(a, b):
    diff = abs(a - b) % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff

def compute_orientation(binary_image, x, y, window_size=3):
    half = window_size // 2
    
    if (y - half < 0 or y + half >= binary_image.shape[0] or 
        x - half < 0 or x + half >= binary_image.shape[1]):
        return 0.0
    window = binary_image[y - half:y + half + 1, x - half:x + half + 1].astype(np.float32)
    grad_x = cv.Sobel(window, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(window, cv.CV_32F, 0, 1, ksize=3)
    avg_dx = np.mean(grad_x)
    avg_dy = np.mean(grad_y)
    orientation = np.arctan2(avg_dy, avg_dx)
    if orientation < 0:
        orientation += 2 * np.pi
    return orientation

def minutiae_at(pixels, i, j, kernel_size):
    
    if pixels[i][j] == 1:
        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),
                     (0, 1),  (1, 1),  (1, 0),
                     (1, -1), (0, -1), (-1, -1)]
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                     (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),
                     (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]
        
        values = [pixels[i + l][j + k] for k, l in cells]
        crossings = 0
        for k in range(len(values) - 1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"

def calculate_minutiae_points(im, kernel_size=3):

    binary_image = np.zeros_like(im)
    binary_image[im < 10] = 1
    binary_image = binary_image.astype(np.int8)

    (height, width) = im.shape
    minutiae_list = []
    for i in range(1, width - kernel_size // 2):
        for j in range(1, height - kernel_size // 2):
            mtype = minutiae_at(binary_image, j, i, kernel_size)
            if mtype != "none":
                orient = compute_orientation(binary_image, i, j, window_size=3)
                minutiae_list.append((i, j, mtype, orient))
    return minutiae_list

def filter_false_minutiae_advanced(points, image_shape, 
                                    cluster_radius=15, border_margin=25, 
                                    D1=20, D2=15, D3=15, angle_threshold=0.35):
    height, width = image_shape

    points_border = [pt for pt in points if (pt[0] >= border_margin and pt[0] <= width - border_margin and 
                                               pt[1] >= border_margin and pt[1] <= height - border_margin)]
    
    points_remaining = points_border.copy()
    cluster_points = []
    while points_remaining:
        p = points_remaining.pop(0)
        cluster = [p]
        to_remove = []
        for q in points_remaining:
            if np.hypot(q[0] - p[0], q[1] - p[1]) < cluster_radius:
                cluster.append(q)
                to_remove.append(q)
        points_remaining = [pt for pt in points_remaining if pt not in to_remove]
        
        xs = [pt[0] for pt in cluster]
        ys = [pt[1] for pt in cluster]
        center = (np.mean(xs), np.mean(ys))
        
        best_pt = min(cluster, key=lambda pt: np.hypot(pt[0]-center[0], pt[1]-center[1]))
        cluster_points.append(best_pt)
    
    filtered = cluster_points  

    
    to_remove = set()
    n = len(filtered)
    for i in range(n):
        for j in range(i+1, n):
            p1 = filtered[i]
            p2 = filtered[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            
            if p1[2] == "ending" and p2[2] == "ending" and dist < D1:
                
                if angle_diff(p1[3], (p2[3] + np.pi) % (2*np.pi)) < angle_threshold:
                    to_remove.add(i)
                    to_remove.add(j)
            
            if ((p1[2] == "ending" and p2[2] == "bifurcation") or (p1[2] == "bifurcation" and p2[2] == "ending")) and dist < D2:
                to_remove.add(i)
                to_remove.add(j)
            
            if p1[2] == "bifurcation" and p2[2] == "bifurcation" and dist < D3:
                to_remove.add(i)
                to_remove.add(j)
                
    final_points = [pt for idx, pt in enumerate(filtered) if idx not in to_remove]
    return final_points

def draw_minutiae(im, points):
    result = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}
    for pt in points:
        x, y, mtype, orient = pt
        cv.circle(result, (x, y), radius=2, color=colors[mtype], thickness=2)
    return result

def store_minutiae_points(points, image_filename, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base, _ = os.path.splitext(image_filename)
    csv_filename = os.path.join(output_folder, base + "_minutiae.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "type", "orientation"])
        for pt in points:
            writer.writerow(pt)
    print(f"Stored minutiae points to {csv_filename}")

def process_image(image_path, output_image_folder, minutiae_folder, kernel_size=3):
    
    im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if im is None:
        print(f"Error reading image: {image_path}")
        return

    
    minutiae_points = calculate_minutiae_points(im, kernel_size)
    
    
    filtered_points = filter_false_minutiae_advanced(minutiae_points, im.shape, 
                                                     cluster_radius=15, border_margin=25,
                                                     D1=20, D2=15, D3=15, angle_threshold=0.35)
    

    result_image = draw_minutiae(im, filtered_points)
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    
    filename = os.path.basename(image_path)
    out_path = os.path.join(output_image_folder, filename)
    cv.imwrite(out_path, result_image)
    print(f"Processed and saved true minutiae image to {out_path}")

    
    store_minutiae_points(filtered_points, filename, minutiae_folder)

if __name__ == '__main__':
    
    input_folder = "./thinnedContactImages"
    
    output_image_folder = "true_minutiae_point_Contact_images"  
    minutiae_folder = "contact_minutiae_points"
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    for file in image_files:
        image_path = os.path.join(input_folder, file)
        process_image(image_path, output_image_folder, minutiae_folder, kernel_size=3)


