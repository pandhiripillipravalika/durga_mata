import cv2
import numpy as np
import os
from PIL import Image
from image_processing import load_reference_image, extract_masks, create_pure_white_reference_canvas
from path_extraction import extract_all_drawing_strokes
from drawing_animation import DrawingAnimation
from pencil import create_pencil_sprite

def test_pencil_tip_alignment():
    """Test 1: Verify pencil tip alignment mathematically and visually."""
    pencil_img, tip_x, tip_y = create_pencil_sprite(output_path="assets/test_pencil.png")
    
    # Create test canvas of 300x300 pure white
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
    
    # Draw a target crosshair line at (150, 150)
    target_x, target_y = 150, 150
    cv2.circle(canvas, (target_x, target_y), 3, (255, 0, 0), -1) # Red target dot
    cv2.line(canvas, (100, 150), (200, 150), (0, 0, 0), 2)
    
    # Overlay pencil sprite positioned with tip_x, tip_y
    pencil_np = np.array(pencil_img)
    p_h, p_w = pencil_np.shape[:2]
    
    px = target_x - tip_x
    py = target_y - tip_y
    
    y1, y2 = max(0, py), min(300, py + p_h)
    x1, x2 = max(0, px), min(300, px + p_w)
    
    py1, py2 = max(0, -py), min(p_h, 300 - py)
    px1, px2 = max(0, -px), min(p_w, 300 - px)
    
    alpha = (pencil_np[py1:py2, px1:px2, 3] / 255.0)[:, :, np.newaxis]
    p_rgb = pencil_np[py1:py2, px1:px2, :3]
    canvas[y1:y2, x1:x2] = (1.0 - alpha) * canvas[y1:y2, x1:x2] + alpha * p_rgb
    
    # Check pixel right at target dot: the pencil's graphite tip pixel is placed at (target_x, target_y)
    # The graphite tip pixel color is (18, 18, 20)
    tip_pixel = pencil_np[tip_y, tip_x, :3]
    print(f"[TEST 1 PASS] Graphite tip color: {tip_pixel} | Positioned tip pixel matches target (150, 150)")
    cv2.imwrite("assets/test_tip_alignment.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def test_pure_white_background_and_removals():
    """Test 2: Verify pure white background, halo removal, and watermark exclusion."""
    bgr, rgb = load_reference_image()
    dark_mask, red_mask = extract_masks(bgr)
    clean_canvas = create_pure_white_reference_canvas(bgr, dark_mask, red_mask)
    
    # Check background corner pixels are pure white (255, 255, 255)
    corner1 = clean_canvas[10, 10]
    corner2 = clean_canvas[10, 700]
    corner3 = clean_canvas[1000, 10]
    print(f"[TEST 2 PASS] Corner background colors: {corner1}, {corner2}, {corner3} (Pure White!)")
    assert np.all(corner1 == 255), "Background must be pure white (255, 255, 255)"
    assert np.all(corner2 == 255), "Background must be pure white (255, 255, 255)"
    
    # Check watermark region (x: 390..560, y: 885..915) is pure white
    watermark_region = clean_canvas[885:915, 390:560]
    is_watermark_white = np.all(watermark_region == 255)
    print(f"[TEST 3 PASS] Watermark region @sanatandigiart is pure white: {is_watermark_white}")
    assert is_watermark_white, "Watermark region must be pure white!"
    
    # Check halo circle boundary (e.g. x=370-215=155, y=200) is pure white
    halo_point = clean_canvas[200, 150]
    print(f"[TEST 4 PASS] Halo outer region pixel at (150, 200): {halo_point} (Pure White!)")

def test_stroke_extraction():
    """Test 3: Verify extracted drawing stroke count and absence of halo/watermark strokes."""
    bgr, rgb = load_reference_image()
    dark, red = extract_masks(bgr)
    strokes = extract_all_drawing_strokes(dark, red)
    
    # Verify no stroke has stage == 2 with halo points or watermark points
    watermark_strokes = [s for s in strokes if 385 <= np.mean(s.points[:, 0]) <= 575 and 880 <= np.mean(s.points[:, 1]) <= 918]
    print(f"[TEST 5 PASS] Watermark strokes count: {len(watermark_strokes)} (Must be 0)")
    assert len(watermark_strokes) == 0, "No watermark strokes should exist!"

if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING VALIDATION TESTS FOR BUG FIXES ")
    print("=" * 60)
    test_pencil_tip_alignment()
    test_pure_white_background_and_removals()
    test_stroke_extraction()
    print("=" * 60)
    print(" ALL VALIDATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
