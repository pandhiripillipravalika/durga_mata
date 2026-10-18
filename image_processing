import cv2
import numpy as np

def load_reference_image(image_path="assets/reference.jpg"):
    """Loads reference image in BGR and RGB."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load reference image from {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

def extract_masks(img_bgr):
    """
    Extracts clean linework and red accent masks while excluding
    the circular halo and artist watermark (@sanatandigiart).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Red mask (Durga bindi, lion bindi, sari red border, trident tied ribbon)
    lower_red1 = np.array([0, 65, 65])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 65, 65])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 2. Dark linework mask (black ink lines)
    # Threshold dark graphite/ink lines (intensity < 170)
    dark_mask = (gray < 170).astype(np.uint8) * 255
    dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(red_mask))
    
    # 3. EXCLUDE WATERMARK (@sanatandigiart near bottom right of goddess sari: x: 385..575, y: 880..918)
    watermark_mask = np.zeros_like(gray, dtype=np.uint8)
    watermark_mask[880:918, 385:575] = 255
    
    dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(watermark_mask))
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(watermark_mask))
    
    # 4. EXCLUDE CIRCULAR HALO OUTER BOUNDARY
    # The halo circle is centered around cx=370, cy=200, r=215
    # We remove outer thin circular ring contour that belongs only to the background halo
    halo_ring_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(halo_ring_mask, (370, 200), 215, 255, thickness=12)
    # Only remove halo ring where it does NOT overlap Durga's crown/hair/head
    # Durga's crown and hair extend into y < 190. Halo ring is in background.
    # Crown is at x: 270..530, y: 10..185
    # Hair is at x: 100..360, y: 150..400
    # Let's filter out halo ring pixels that are faint background circle lines (gray > 140)
    faint_halo_ring = (gray > 140) & (halo_ring_mask > 0)
    dark_mask[faint_halo_ring] = 0
    
    # Clean up isolated single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    return dark_mask, red_mask

def create_pure_white_reference_canvas(img_bgr, dark_mask, red_mask):
    """
    Creates a clean reference artwork image on a PURE WHITE background (RGB 255, 255, 255).
    Replaces cream/tan paper background with crisp white while retaining dark linework and red accents.
    """
    h, w = img_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8) # Pure white background
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Dark linework rendering: map darkness onto pure white
    dark_indices = dark_mask > 0
    darkness = (255.0 - gray[dark_indices].astype(np.float32)) / 255.0
    darkness = np.clip(darkness * 1.25, 0.0, 1.0)[:, np.newaxis]
    
    # Graphite black ink color: RGB (20, 20, 20)
    ink_color = np.array([20, 20, 20], dtype=np.float32)
    white_bg = np.array([255, 255, 255], dtype=np.float32)
    
    rendered_dark = (1.0 - darkness) * white_bg + darkness * ink_color
    canvas[dark_indices] = rendered_dark.astype(np.uint8)
    
    # Red accents rendering: copy vibrant red colors from reference image
    red_indices = red_mask > 0
    canvas[red_indices] = img_rgb[red_indices]
    
    return canvas

if __name__ == "__main__":
    bgr, rgb = load_reference_image()
    dark, red = extract_masks(bgr)
    clean_canvas = create_pure_white_reference_canvas(bgr, dark, red)
    cv2.imwrite("assets/clean_reference_white.png", cv2.cvtColor(clean_canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite("assets/dark_mask.png", dark)
    cv2.imwrite("assets/red_mask.png", red)
    print("Clean pure white reference canvas and masks saved!")
