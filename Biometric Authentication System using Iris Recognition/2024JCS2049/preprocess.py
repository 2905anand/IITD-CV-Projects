import os
import cv2
import numpy as np
from skimage.transform import radon
from PIL import Image
from functools import wraps
import time

def mask_black_borders(bin_img, win=(5, 5)):
    """
    Purpose:
        For a given binary image, look at small windows (of size 'win') and set
        the entire window to 0 if its border pixels are all black (0).
    
    Inputs:
        bin_img (numpy.ndarray): 2D binary image.
        win (tuple): (window_height, window_width).
    
    Outputs:
        out_img (numpy.ndarray): Modified binary image.
    """
    H, W = bin_img.shape
    out_img = bin_img.copy()
    for i in range(H - win[0] + 1):
        for j in range(W - win[1] + 1):
            patch = bin_img[i:i + win[0], j:j + win[1]]
            borders = np.concatenate([
                patch[0, :],
                patch[-1, :],
                patch[1:-1, 0],
                patch[1:-1, -1]
            ])
            if np.all(borders == 0):
                out_img[i:i + win[0], j:j + win[1]] = 0
    return out_img


def interp_bilinear(img, X, Y):
    """
    Purpose:
        Perform bilinear interpolation on a single coordinate (X, Y) in a 2D image.
    
    Inputs:
        img (numpy.ndarray): 2D image (grayscale).
        X (float): X-coordinate (column index) at which we want the interpolated value.
        Y (float): Y-coordinate (row index) at which we want the interpolated value.
    
    Outputs:
        float: The interpolated pixel value at (X, Y).
    """
    eps = 1e-6
    X = np.clip(X, 0, img.shape[1] - 1 - eps)
    Y = np.clip(Y, 0, img.shape[0] - 1 - eps)

    x_low = int(np.floor(X))
    x_high = min(x_low + 1, img.shape[1] - 1)
    y_low = int(np.floor(Y))
    y_high = min(y_low + 1, img.shape[0] - 1)

    Ia = img[y_low, x_low]
    Ib = img[y_high, x_low]
    Ic = img[y_low, x_high]
    Id = img[y_high, x_high]

    wa = (x_high - X) * (y_high - Y)
    wb = (x_high - X) * (Y - y_low)
    wc = (X - x_low) * (y_high - Y)
    wd = (X - x_low) * (Y - y_low)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def rubber_sheet_norm(img, pup_center, pup_rad, ir_center, ir_rad, n_rad=64, n_ang=512):
    """
    Purpose:
        Apply rubber-sheet (Daugman's) normalization to unwrap the circular iris region
        into a rectangular block of dimensions (n_rad x n_ang).
    
    Inputs:
        img (numpy.ndarray): Grayscale eye image.
        pup_center (tuple): (x, y) coordinates of the pupil center.
        pup_rad (float): Pupil radius.
        ir_center (tuple): (x, y) coordinates of the iris center.
        ir_rad (float): Iris radius.
        n_rad (int): Number of radial samples (vertical dimension of output).
        n_ang (int): Number of angular samples (horizontal dimension of output).
    
    Outputs:
        norm_img (numpy.ndarray): Normalized (unwrapped) iris image of shape (n_rad, n_ang).
    """
    norm_img = np.zeros((n_rad, n_ang), dtype=img.dtype)
    for j in range(n_ang):
        angle = 2 * np.pi * j / n_ang
        x_p = pup_center[0] + pup_rad * np.cos(angle)
        y_p = pup_center[1] + pup_rad * np.sin(angle)
        x_i = ir_center[0] + ir_rad * np.cos(angle)
        y_i = ir_center[1] + ir_rad * np.sin(angle)
        for i in range(n_rad):
            r = i / (n_rad - 1)
            X = (1 - r) * x_p + r * x_i
            Y = (1 - r) * y_p + r * y_i
            norm_img[i, j] = interp_bilinear(img, X, Y)
    return norm_img

def remove_specular(img):
    """
    Purpose:
        Remove bright specular reflections in the eye image via inpainting.
    
    Inputs:
        img (numpy.ndarray): BGR image of the eye.
    
    Outputs:
        inpainted (numpy.ndarray): Image with specular highlights removed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def find_pupil(image, dbg=False):
    """
    Purpose:
        Detect the pupil center and radius in a BGR eye image using thresholding
        and Hough Circles.
    
    Inputs:
        image (numpy.ndarray): BGR eye image.
        dbg (bool): Debug flag (optional, not used).
    
    Outputs:
        (center, radius) (tuple, int): Where center=(x, y). Returns (None, None)
                                       if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    clean = mask_black_borders(thr, (5, 5))
    clean = mask_black_borders(clean, (10, 10))

    circles = cv2.HoughCircles(clean, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=10, param2=10,
                               minRadius=1, maxRadius=100)
    if circles is not None:
        pup = np.uint16(np.around(circles[0][0]))
        return (pup[0], pup[1]), pup[2]
    return None, None

def find_iris(image, pup_center, pup_rad, dbg=False):
    """
    Purpose:
        Detect the iris boundary (center and radius) in a BGR eye image.
        Uses Canny edges and Hough Circles. A region around the pupil is masked out.
    
    Inputs:
        image (numpy.ndarray): BGR eye image.
        pup_center (tuple): (x, y) for pupil center.
        pup_rad (int): pupil radius.
        dbg (bool): Debug flag (optional, not used).
    
    Outputs:
        (center, radius) (tuple, int): Where center=(x, y). Returns (None, None)
                                       if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enh = clahe.apply(blur)
    edges = cv2.Canny(enh, 50, 150)
    
    mask = np.zeros_like(edges)
    cv2.circle(mask, pup_center, pup_rad + 40, 255, -1)
    masked = cv2.bitwise_and(edges, edges, mask=~mask)
    circles = cv2.HoughCircles(masked, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=100, param2=30,
                               minRadius=int(pup_rad * 2),
                               maxRadius=int(pup_rad * 3))
    if circles is not None:
        ir = np.uint16(np.around(circles[0][0]))
        return (ir[0], ir[1]), ir[2]
    return None, None

def diffuse_aniso(img, iterations=10, kappa=50, gamma=0.1):
    """
    Purpose:
        Apply anisotropic diffusion on a grayscale image to smooth out noise
        while preserving edges.
    
    Inputs:
        img (numpy.ndarray): Grayscale input image.
        iterations (int): Number of diffusion iterations.
        kappa (float): Conductance coefficient (sensitivity to edges).
        gamma (float): Time step size.
    
    Outputs:
        (numpy.ndarray): The diffused (smoothed) image.
    """
    img = img.astype(np.float32)
    I = np.pad(img, 1, mode='edge')
    for _ in range(iterations):
        dN = I[0:-2, 1:-1] - I[1:-1, 1:-1]
        dS = I[2:, 1:-1] - I[1:-1, 1:-1]
        dW = I[1:-1, 0:-2] - I[1:-1, 1:-1]
        dE = I[1:-1, 2:] - I[1:-1, 1:-1]

        cN = np.exp(-(dN / kappa) ** 2)
        cS = np.exp(-(dS / kappa) ** 2)
        cW = np.exp(-(dW / kappa) ** 2)
        cE = np.exp(-(dE / kappa) ** 2)

        diff = gamma * (cN * dN + cS * dS + cW * dW + cE * dE)
        I[1:-1, 1:-1] += diff
    return np.clip(I[1:-1, 1:-1], 0, 255).astype(np.uint8)

def eyelid_boundaries(img, ir_center, ir_rad, iterations=10, kappa=50, gamma=0.1):
    """
    Purpose:
        Estimate the positions of upper and lower eyelids relative to the iris
        by applying anisotropic diffusion and searching via radon transform.
    
    Inputs:
        img (numpy.ndarray): BGR eye image.
        ir_center (tuple): Iris center (x, y).
        ir_rad (int): Iris radius.
        iterations (int): Number of diffusion iterations.
        kappa (float): Conductance coefficient (for diffusion).
        gamma (float): Time step size (for diffusion).
    
    Outputs:
        (upper_eyelid_y, lower_eyelid_y, diffused) where:
            - upper_eyelid_y is the approximate row index for the upper eyelid
            - lower_eyelid_y is the approximate row index for the lower eyelid
            - diffused is the final diffused (smoothed) image used for analysis
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diffused = diffuse_aniso(gray, iterations, kappa, gamma)

    H, _ = diffused.shape
    centerY = int(ir_center[1])
    ir_rad = int(ir_rad)

    # Upper region
    upper_start = max(0, int(centerY - 1.2 * ir_rad))
    upper_roi = diffused[upper_start:centerY, :]

    # Lower region
    lower_start = centerY
    lower_end = min(H, int(centerY + 1.2 * ir_rad))
    lower_roi = diffused[lower_start:lower_end, :]

    # Apply Radon transform over a small range of angles
    angles = np.linspace(80, 100, 5)
    sino_up = radon(upper_roi, theta=angles, circle=False)
    sino_low = radon(lower_roi, theta=angles, circle=False)

    up_sum = np.sum(sino_up, axis=1)
    low_sum = np.sum(sino_low, axis=1)

    up_idx = np.argmax(up_sum)
    low_idx = np.argmax(low_sum)

    return upper_start + up_idx, lower_start + low_idx, diffused

def iris_localize(img):
    """
    Purpose:
        A pipeline to localize the iris in a BGR eye image:
         1. Remove specular reflections,
         2. Detect pupil,
         3. Detect iris,
         4. Estimate eyelid boundaries,
         5. Return an annotated image and detected features.
    
    Inputs:
        img (numpy.ndarray): BGR eye image.
    
    Outputs:
        (annotated_image, features_dict) or (None, None) if detection fails.
        features_dict contains:
            - pup_center, pup_rad
            - ir_center, ir_rad
            - up_y, low_y
    """
    ref_removed = remove_specular(img)
    pup_center, pup_rad = find_pupil(ref_removed)
    if pup_center is None:
        return None, None

    ir_center, ir_rad = find_iris(ref_removed, pup_center, pup_rad)
    if ir_center is None:
        return None, None

    up_y, low_y, _ = eyelid_boundaries(ref_removed, ir_center, ir_rad)

    annotated = ref_removed.copy()
    cv2.circle(annotated, pup_center, pup_rad, (0, 255, 0), 2)
    cv2.circle(annotated, ir_center, ir_rad, (0, 0, 255), 2)
    cv2.line(annotated, (0, up_y), (annotated.shape[1], up_y), (255, 0, 0), 2)
    cv2.line(annotated, (0, low_y), (annotated.shape[1], low_y), (255, 255, 0), 2)

    feats = {
        "pup_center": pup_center,
        "pup_rad": pup_rad,
        "ir_center": ir_center,
        "ir_rad": ir_rad,
        "up_y": up_y,
        "low_y": low_y
    }
    return annotated, feats

def resize_bilinear(im):
    """
    Purpose:
        Resize a given 2D image to 64x64 using OpenCV's bilinear interpolation.
    
    Inputs:
        im (numpy.ndarray): A 2D (grayscale) image.
    
    Outputs:
        (numpy.ndarray): The resized 64x64 image.
    """
    return cv2.resize(im, (64, 64), interpolation=cv2.INTER_LINEAR)

def process_preproc(src, dst):
    """
    Purpose:
        Load an image from src, localize its iris (with annotations),
        and then save the annotated result to dst if successful.
    
    Inputs:
        src (str): Source image path.
        dst (str): Destination path for the annotated image.
    
    Outputs:
        None (writes the annotated image or prints the source path on failure).
    """
    im = cv2.imread(src)
    if im is None:
        print(src)
        return

    ann, feats = iris_localize(im)
    if ann is not None:
        cv2.imwrite(dst, ann)
        print(dst)
    else:
        print(src)

def do_normalization(src, dst, n_rad=64, n_ang=512):
    """
    Purpose:
        Load an image, localize the iris, apply rubber-sheet normalization,
        resize the result to 64x64, and save to dst.
    
    Inputs:
        src (str): Source image path.
        dst (str): Destination path for the normalized image.
        n_rad (int): Number of radial samples.
        n_ang (int): Number of angular samples.
    
    Outputs:
        None (writes the normalized iris image or prints the source path on failure).
    """
    im = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(src)
        return

    # We need BGR image for detection, but we only read grayscale. Convert to BGR for consistency:
    _, feats = iris_localize(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
    if feats is None:
        print(src)
        return

    pc = feats["pup_center"]
    pr = feats["pup_rad"]
    ic = feats["ir_center"]
    ir = feats["ir_rad"]

    norm_im = rubber_sheet_norm(im, pc, pr, ic, ir, n_rad, n_ang)
    norm_im = resize_bilinear(norm_im)
    cv2.imwrite(dst, norm_im)
    print(dst)

def main():
    """
    Purpose:
        Example main function that:
          1. Iterates through an 'IITD Database' folder structure,
          2. Saves grayscale versions to 'Local/original_dataset',
          3. Demonstrates a call to process_preproc on 'Local/original_dataset',
          4. Demonstrates a call to do_normalization on 'Local/preprocessed_dataset'.
    
    Implementation Details:
        - The code tries to read images from 'IITD Database/<001-224>/*'.
        - Converts them to grayscale and writes them to 'Local/original_dataset'.
        - Then calls process_preproc() on "Local/original_dataset",
          saving results to "Local/preprocessed_dataset".
        - Then calls do_normalization() on "Local/preprocessed_dataset",
          saving results to "Local/normalized_dataset".
    """
    srcFolder = "IITD Database"
    dstFolder = "Local/original_dataset"

    # Copy and convert original images to grayscale
    for cnt in range(1, 225):
        sp = str(cnt).zfill(3)
        folder_path = os.path.join(srcFolder, sp)
        if not os.path.exists(folder_path):
            continue
        for f in os.listdir(folder_path):
            sPath = os.path.join(folder_path, f)
            # Ensure output .png extension
            dPath = os.path.join(dstFolder, f"{sp}_{f.replace('.bmp','.png')}")
            if not os.path.isfile(dPath):
                im = cv2.imread(sPath, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    cv2.imwrite(dPath, im)

    # Example usage of preprocessing on an entire directory:
    # Here, the function is called with the same src & dst, but you'd typically
    # have different source/destination directories in practice.
    process_preproc("Local/original_dataset", "Local/preprocessed_dataset")

    # Example usage of normalization on the preprocessed dataset:
    do_normalization("Local/preprocessed_dataset", "Local/normalized_dataset")

if __name__ == "__main__":
    main()
