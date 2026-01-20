import cv2
import numpy as np

# MediaPipe Indices
# Left Eye
LEFT_EYE_INNER = 33
LEFT_EYE_OUTER = 133
LEFT_IRIS_CENTER = 468
# Dense contour (ordered loop)
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

# Right Eye
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_IRIS_CENTER = 473
# Dense contour (ordered loop)
RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

def get_roi_rect(image, landmarks, indices, padding=10):
    xs = [landmarks[i][0] for i in indices]
    ys = [landmarks[i][1] for i in indices]
    
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(image.shape[1], max(xs) + padding)
    y2 = min(image.shape[0], max(ys) + padding)
    
    return x1, y1, x2, y2

def calculate_ear(landmarks, is_left_eye):
    # Eye Aspect Ratio to detect if eye is open
    if is_left_eye:
        v1 = np.linalg.norm(np.array(landmarks[159]) - np.array(landmarks[145]))
        v2 = np.linalg.norm(np.array(landmarks[158]) - np.array(landmarks[153]))
        h = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[133]))
    else:
        v1 = np.linalg.norm(np.array(landmarks[386]) - np.array(landmarks[374]))
        v2 = np.linalg.norm(np.array(landmarks[387]) - np.array(landmarks[373]))
        h = np.linalg.norm(np.array(landmarks[362]) - np.array(landmarks[263]))
        
    ear = (v1 + v2) / (2.0 * h)
    return ear

def sharpen_image(image):
    # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    # Apply the sharpening kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their new adjusted values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def warp_eye(image, landmarks, is_left_eye, intensity=2.8):
    if is_left_eye:
        idx_inner = LEFT_EYE_INNER
        idx_outer = LEFT_EYE_OUTER
        idx_iris = LEFT_IRIS_CENTER
        contour_indices = LEFT_EYE_CONTOUR
        # Iris perimeter indices for Left Eye (469-472)
        idx_iris_edges = [469, 470, 471, 472]
    else:
        idx_inner = RIGHT_EYE_INNER
        idx_outer = RIGHT_EYE_OUTER
        idx_iris = RIGHT_IRIS_CENTER
        contour_indices = RIGHT_EYE_CONTOUR
        # Iris perimeter indices for Right Eye (474-477)
        idx_iris_edges = [474, 475, 476, 477]
        
    try:
        # 1. Check EAR
        ear = calculate_ear(landmarks, is_left_eye)
        if ear < 0.2:
            return image

        if len(landmarks) <= max(idx_iris_edges):
            return image
            
        inner_pt = np.array(landmarks[idx_inner])
        outer_pt = np.array(landmarks[idx_outer])
        iris_pt = np.array(landmarks[idx_iris])
        
        # Determine Iris Radius accurately
        r_sum = 0
        for edge_idx in idx_iris_edges:
             edge_pt = np.array(landmarks[edge_idx])
             r_sum += np.linalg.norm(edge_pt - iris_pt)
        iris_radius = int(r_sum / len(idx_iris_edges))
        
        # Make radius just big enough (don't over-expand, or we grab skin)
        iris_radius = int(iris_radius * 1.05) 
        
        # ROI based on contour
        # Pad enough to cover the iris move
        x1, y1, x2, y2 = get_roi_rect(image, landmarks, contour_indices, padding=int(iris_radius * 2.5))
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return image
            
        h, w = roi.shape[:2]
        
        # 2. Define Masks
        # Sclera/Eye Mask (The Window)
        mask_sclera = np.zeros((h, w), dtype=np.uint8)
        contour_pts = []
        for idx in contour_indices:
            px, py = landmarks[idx]
            contour_pts.append((px - x1, py - y1))
        contour_pts = np.array(contour_pts, dtype=np.int32)
        cv2.fillPoly(mask_sclera, [contour_pts], 255)
        
        # Iris Mask (The Object)
        ix, iy = int(iris_pt[0] - x1), int(iris_pt[1] - y1)
        mask_iris = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_iris, (ix, iy), iris_radius, 255, -1)
        
        # 3. Inpaint the original Iris to check a blank eyeball
        # We dilate the iris mask significantly to ensure we don't leave dark edges
        mask_iris_dilated = cv2.dilate(mask_iris, np.ones((5,5), np.uint8), iterations=3)
        
        # CRITICAL FIX: Clip the inpainting mask to the eye contour (sclera)
        # This prevents inpainting from eating into the eyelids/skin
        mask_inpaint = cv2.bitwise_and(mask_iris_dilated, mask_sclera)
        
        # Inpaint: Replaces the iris with surrounding skin/sclera texture
        clean_eye = cv2.inpaint(roi, mask_inpaint, 3, cv2.INPAINT_TELEA) # Reduced radius for sharper fill
        
        # 4. Extract the Iris
        iris_texture = roi.copy()
        
        # 5. Calculate New Position
        # Target is the geometric center of the eye corners (visual center)
        target_pt = (inner_pt + outer_pt) / 2
        
        # Shift vector from Iris to Target
        # Intensity = 1.0 means move exactly to the center.
        # Intensity > 1.0 means overshoot (cartoonish or extreme correction).
        shift = (target_pt - iris_pt) * intensity
        
        new_ix = int(ix + shift[0])
        new_iy = int(iy + shift[1])
        
        # 6. Paste Iris at New Position
        # We need to construct the result image.
        result = clean_eye.copy()
        
        # Source (Iris) Rect
        src_x1 = ix - iris_radius
        src_y1 = iy - iris_radius
        src_x2 = ix + iris_radius
        src_y2 = iy + iris_radius
        
        # Dest (New Iris) Rect
        dst_x1 = new_ix - iris_radius
        dst_y1 = new_iy - iris_radius
        dst_x2 = new_ix + iris_radius
        dst_y2 = new_iy + iris_radius
        
        # Extract iris chip
        # Boundary checks
        if src_x1 < 0 or src_y1 < 0 or src_x2 > w or src_y2 > h:
             return image # Safety
        
        iris_chip = roi[src_y1:src_y2, src_x1:src_x2]
        
        # Place it
        # Handle overlaps
        h_chip, w_chip = iris_chip.shape[:2]
        
        # Calculate valid placement area
        start_y = max(0, dst_y1)
        end_y = min(h, dst_y2)
        start_x = max(0, dst_x1)
        end_x = min(w, dst_x2)
        
        if start_x >= end_x or start_y >= end_y:
            return image
            
        # Offsets into chip
        chip_start_y = start_y - dst_y1
        chip_end_y = chip_start_y + (end_y - start_y)
        chip_start_x = start_x - dst_x1
        chip_end_x = chip_start_x + (end_x - start_x)
        
        # Prepare valid chip part
        valid_chip = iris_chip[chip_start_y:chip_end_y, chip_start_x:chip_end_x]
        
        # ENHANCE IRIS STRUCTURE & FIX PUPIL
        # 1. Darken Shadows (Gamma) to make pupil black again
        # Gamma < 1.0 makes image darker? No, 255^(1/gamma). 
        # If gamma=1.5, invGamma=0.66. 255^0.66 is small? 
        # Wait. x^0.5 is sqrt (increases values).
        # x^2 (decreases values for 0..1).
        # Standard implementation use 'gamma' as value to divide by in exponent??
        # OpenCV logic: V_out = (V_in)^gamma. 
        # Let's trust logic: adjust_gamma(img, 0.6) -> Darker. adjust_gamma(img, 1.5) -> Lighter?
        # NO. V_new = 255 * (V_old / 255) ^ (1/gamma).
        # If gamma=2.0 -> inv=0.5. x^0.5 -> Boosts darks (lighter).
        # If gamma=0.5 -> inv=2.0. x^2 -> Crushes darks (darker).
        
        # We want to CRUSH darks (make gray pupil black). So we need exponent > 1.
        # So we need invGamma > 1. So gamma < 1.
        # Let's use gamma=0.6.
        valid_chip = adjust_gamma(valid_chip, gamma=0.6)
        
        # 2. Sharpen (subtle)
        valid_chip = sharpen_image(valid_chip)
        
        # 3. Add Catchlight (Shiny reflection for life)
        cl_offset = int(iris_radius * 0.25)
        cl_x = iris_radius - cl_offset
        cl_y = iris_radius - cl_offset
        cl_radius = max(1, int(iris_radius * 0.12))
        
        vis_cl_x = cl_x - chip_start_x
        vis_cl_y = cl_y - chip_start_y
        
        h_valid, w_valid = valid_chip.shape[:2]
        if 0 <= vis_cl_x < w_valid and 0 <= vis_cl_y < h_valid:
             cv2.circle(valid_chip, (vis_cl_x, vis_cl_y), cl_radius, (255, 255, 255), -1, cv2.LINE_AA)

        # Prepare destination part (from clean_eye)
        dest_slice = result[start_y:end_y, start_x:end_x]
        
        # Prepare Sclera Mask slice at destination
        mask_slice = mask_sclera[start_y:end_y, start_x:end_x]
        
        # Create a mini circular mask for the chip (SHARPER EDGES)
        mini_mask = np.zeros((h_chip, w_chip), dtype=np.float32)
        # Use anti-aliased circle but NOT blurry
        cv2.circle(mini_mask, (w_chip//2, h_chip//2), iris_radius - 1, 1.0, -1, lineType=cv2.LINE_AA)
        
        # Minimal softening (1.5 sigma instead of 5)
        mini_mask = cv2.GaussianBlur(mini_mask, (3, 3), 1.0)
        
        mini_mask_valid = mini_mask[chip_start_y:chip_end_y, chip_start_x:chip_end_x]
        
        # Normalize Sclera Mask
        mask_slice_float = mask_slice.astype(float) / 255.0
        
        # Combine masks: Multiply
        final_mask_float = mask_slice_float * mini_mask_valid
        
        # Composite
        final_mask_3ch = cv2.merge([final_mask_float, final_mask_float, final_mask_float])
        
        blended = valid_chip.astype(float) * final_mask_3ch + dest_slice.astype(float) * (1.0 - final_mask_3ch)
        
        # Update result
        result[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        # Paste result back to image
        image[y1:y2, x1:x2] = result
        
        return image

    except Exception as e:
        print(f"Error in warp_eye: {e}")
        return image

def correct_gaze(image, landmarks, intensity=1.0):
    img = image.copy()
    img = warp_eye(img, landmarks, is_left_eye=True, intensity=intensity)
    img = warp_eye(img, landmarks, is_left_eye=False, intensity=intensity)
    return img

def draw_debug_gaze(image, landmarks, is_left_eye):
    pass # Disabled for final output
