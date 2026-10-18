import cv2
import numpy as np
import math
from scipy.ndimage import distance_transform_edt

class Stroke:
    def __init__(self, points, width=2.0, color=(0, 0, 0), stage=1, is_red=False):
        self.points = np.array(points, dtype=np.float32) # Nx2 array of (x, y)
        self.width = max(1.0, float(width))
        self.color = color
        self.stage = stage
        self.is_red = is_red

def skeletonize(binary_mask):
    """Skeletonizes a binary mask (255 for foreground) using OpenCV morphology."""
    img = (binary_mask > 128).astype(np.uint8)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    temp = img.copy()
    while True:
        eroded = cv2.erode(temp, element)
        temp_opened = cv2.dilate(eroded, element)
        sub = cv2.subtract(temp, temp_opened)
        skel = cv2.bitwise_or(skel, sub)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel

def extract_paths_from_skeleton(skel, dist_transform):
    """
    Traces continuous curves from skeleton graph.
    Disconnects at junction points to produce simple smooth open curves.
    """
    h, w = skel.shape
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel)
    skel_pts = (skel > 0)
    
    junctions = skel_pts & (neighbor_count > 2)
    non_junction_skel = skel_pts & (~junctions)
    
    contours, _ = cv2.findContours(non_junction_skel.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    raw_strokes = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2)
        
        ordered = order_points_along_path(pts)
        if len(ordered) < 2:
            continue
            
        widths = [dist_transform[int(clamp(y, 0, h-1)), int(clamp(x, 0, w-1))] * 2.0 for x, y in ordered]
        avg_w = float(np.median(widths)) if len(widths) > 0 else 2.0
        
        simplified = simplify_path(ordered, epsilon=1.0)
        if len(simplified) >= 2:
            raw_strokes.append((simplified, avg_w))
            
    contour_strokes = extract_contour_boundaries(skel, dist_transform)
    raw_strokes.extend(contour_strokes)
    
    return raw_strokes

def clamp(val, min_v, max_v):
    return max(min_v, min(max_v, val))

def order_points_along_path(pts):
    """Orders an unstructured set of 2D points into a continuous path using greedy nearest-neighbor."""
    if len(pts) <= 2:
        return pts
        
    pts = list(pts)
    centroid = np.mean(pts, axis=0)
    start_idx = max(range(len(pts)), key=lambda i: (pts[i][0]-centroid[0])**2 + (pts[i][1]-centroid[1])**2)
    
    path = [pts.pop(start_idx)]
    while pts:
        curr = path[-1]
        nearest_idx = min(range(len(pts)), key=lambda i: (pts[i][0]-curr[0])**2 + (pts[i][1]-curr[1])**2)
        dist_sq = (pts[nearest_idx][0]-curr[0])**2 + (pts[nearest_idx][1]-curr[1])**2
        if dist_sq > 400:
            break
        path.append(pts.pop(nearest_idx))
    return np.array(path, dtype=np.float32)

def simplify_path(pts, epsilon=1.0):
    """Simplifies a 2D path using Ramer-Douglas-Peucker algorithm via OpenCV."""
    pts_int = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts_int, epsilon, closed=False)
    return approx.reshape(-1, 2).astype(np.float32)

def extract_contour_boundaries(binary_mask, dist_transform):
    """Extracts outer contours for large features or filled shapes."""
    contours, _ = cv2.findContours((binary_mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    boundary_strokes = []
    h, w = binary_mask.shape
    for c in contours:
        if len(c) < 5:
            continue
        pts = c.reshape(-1, 2)
        simplified = simplify_path(pts, epsilon=1.2)
        if len(simplified) >= 3:
            widths = [dist_transform[int(clamp(y, 0, h-1)), int(clamp(x, 0, w-1))] * 2.0 for x, y in simplified]
            avg_w = float(np.median(widths)) if len(widths) > 0 else 2.0
            boundary_strokes.append((simplified, avg_w))
    return boundary_strokes

def assign_stroke_stage(pts, is_red=False):
    """
    Assigns a stroke to one of 10 drawing stages based on feature position and type.
    
    Stages:
    Stage 2: Major structural outlines
    Stage 3: Durga facial contours, eyes, eyebrows, nose, lips, forehead bindi
    Stage 4: Crown (Mukut), gemstones, top ornament
    Stage 5: Flowing hair strands
    Stage 6: Earrings, necklaces, jewelry, blouse, sari folds, red sari border
    Stage 7: Arms, fingers, bangles, hand grip around trident
    Stage 8: Trident (3 prongs, central spear, shaft, red cloth)
    Stage 9: Lion facial features, ears, nose, mouth, whiskers, forehead mark, mane
    Stage 10: Fine details & lotus ornament at bottom center
    """
    mean_x = float(np.mean(pts[:, 0]))
    mean_y = float(np.mean(pts[:, 1]))
    
    if is_red:
        if mean_y < 280 and 320 < mean_x < 440:
            return 3 # Bindi mark on Durga's forehead
        elif mean_y > 550 and 300 < mean_x < 420:
            return 9 # Lion forehead bindi mark
        elif 500 < mean_x < 680 and 200 < mean_y < 750:
            return 8 # Trident red ribbon
        else:
            return 6 # Red sari border
            
    # Black linework stage classification
    # 1. Crown: Top head area y < 185, x between 250 and 530
    if mean_y < 185 and 250 < mean_x < 530:
        return 4
        
    # 2. Face: y between 160 and 330, x between 340 and 500
    if 160 <= mean_y <= 330 and 340 <= mean_x <= 500:
        return 3
        
    # 3. Trident: top spear (x > 500, y < 250) or vertical shaft (x > 560, 240 < y < 950)
    if (mean_x > 510 and mean_y < 240) or (mean_x > 565 and mean_y > 240):
        return 8
        
    # 4. Hand & Arm gripping trident: x between 500 and 650, y between 330 and 480
    if 510 <= mean_x <= 660 and 330 <= mean_y <= 480:
        return 7
        
    # 5. Lion: bottom left region y > 520, x < 460
    if mean_y > 520 and mean_x < 460:
        return 9
        
    # 6. Lotus ornament: bottom center y > 910, 280 < x < 440
    if mean_y > 910 and 280 < mean_x < 440:
        return 10
        
    # 7. Jewelry & Earrings (x: 340..580, y: 280..480)
    if 280 <= mean_y <= 480 and 340 <= mean_x <= 580:
        return 6
        
    # 8. Hair: flowing locks (x < 360 and 150 < y < 600)
    if mean_x < 360 and 150 < mean_y < 600:
        return 5
        
    # 9. Sari folds & body: y > 350
    if mean_y > 350:
        return 6
        
    return 2 if mean_y < 400 else 10

def is_watermark_stroke(pts):
    """Filters out any remaining stroke inside the watermark text box (@sanatandigiart)."""
    mean_x = float(np.mean(pts[:, 0]))
    mean_y = float(np.mean(pts[:, 1]))
    if 385 <= mean_x <= 575 and 880 <= mean_y <= 918:
        return True
    return False

def extract_all_drawing_strokes(dark_mask, red_mask):
    """
    Extracts, classifies, and orders all drawing strokes for the animation.
    NO circular halo strokes and NO watermark strokes are extracted.
    """
    all_strokes = []
    
    # 1. Black linework strokes
    dist_dark = distance_transform_edt(dark_mask > 0)
    skel_dark = skeletonize(dark_mask)
    dark_paths = extract_paths_from_skeleton(skel_dark, dist_dark)
    
    for pts, width in dark_paths:
        if is_watermark_stroke(pts):
            continue
        stage = assign_stroke_stage(pts, is_red=False)
        all_strokes.append(Stroke(pts, width=width, color=(20, 20, 20), stage=stage, is_red=False))
        
    # 2. Red accent strokes
    dist_red = distance_transform_edt(red_mask > 0)
    skel_red = skeletonize(red_mask)
    red_paths = extract_paths_from_skeleton(skel_red, dist_red)
    
    for pts, width in red_paths:
        if is_watermark_stroke(pts):
            continue
        stage = assign_stroke_stage(pts, is_red=True)
        all_strokes.append(Stroke(pts, width=max(2.5, width), color=(180, 25, 25), stage=stage, is_red=True))
        
    # Sort strokes primarily by stage, secondarily by spatial position (top-to-bottom, left-to-right)
    all_strokes.sort(key=lambda s: (s.stage, np.min(s.points[:, 1]), np.min(s.points[:, 0])))
    
    print(f"Total extracted strokes (Clean pure white canvas): {len(all_strokes)}")
    for stage_num in range(1, 11):
        count = sum(1 for s in all_strokes if s.stage == stage_num)
        if count > 0:
            print(f"  Stage {stage_num}: {count} strokes")
            
    return all_strokes

if __name__ == "__main__":
    dark = cv2.imread("assets/dark_mask.png", cv2.IMREAD_GRAYSCALE)
    red = cv2.imread("assets/red_mask.png", cv2.IMREAD_GRAYSCALE)
    strokes = extract_all_drawing_strokes(dark, red)
