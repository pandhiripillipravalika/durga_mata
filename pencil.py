import os
import math
import numpy as np
from PIL import Image, ImageDraw

def create_pencil_sprite(output_path="assets/pencil.png", length=90, width=14, angle_deg=45):
    """
    Creates a high-resolution, realistic graphite pencil sprite.
    The pencil body slants gracefully from top-left to bottom-right,
    with its sharp graphite tip pointing directly at the stroke coordinate.
    
    Returns (pencil_image_pil, tip_x, tip_y) where (tip_x, tip_y) is the exact
    pixel offset of the sharp graphite tip relative to the top-left of the sprite PNG.
    """
    v_len = length
    v_w = width
    pad = 25
    canvas_w = v_w + pad * 2
    canvas_h = v_len + pad * 2
    
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    cx = canvas_w // 2
    top_y = pad
    tip_y = pad + v_len
    
    body_len = int(v_len * 0.68)
    cone_len = int(v_len * 0.22)
    
    y_body_bottom = top_y + body_len
    y_cone_bottom = y_body_bottom + cone_len
    y_tip = tip_y
    
    # 1. Pencil Body (Hexagonal dark graphite/blue body with metallic cap at top)
    x_left = cx - v_w // 2
    x_right = cx + v_w // 2
    x1 = cx - v_w // 6
    x2 = cx + v_w // 6
    
    # Left facet (darker shadow)
    draw.polygon([(x_left, top_y), (x1, top_y), (x1, y_body_bottom), (x_left, y_body_bottom)],
                 fill=(35, 42, 50, 255))
    # Center facet (main highlight)
    draw.polygon([(x1, top_y), (x2, top_y), (x2, y_body_bottom), (x1, y_body_bottom)],
                 fill=(70, 80, 92, 255))
    # Right facet (midtone)
    draw.polygon([(x2, top_y), (x_right, top_y), (x_right, y_body_bottom), (x2, y_body_bottom)],
                 fill=(48, 55, 65, 255))
    
    # Metallic top cap at flat back end
    draw.ellipse([x_left, top_y - 3, x_right, top_y + 3], fill=(210, 170, 80, 255))
    
    # 2. Sharpened Wooden Cone (Cedar wood color)
    x_graphite_top_left = cx - 2
    x_graphite_top_right = cx + 2
    
    draw.polygon([(x_left, y_body_bottom), (x_right, y_body_bottom), 
                  (x_graphite_top_right, y_cone_bottom), (x_graphite_top_left, y_cone_bottom)],
                 fill=(225, 190, 140, 255))
    draw.polygon([(x_left, y_body_bottom), (cx, y_body_bottom), 
                  (cx, y_cone_bottom), (x_graphite_top_left, y_cone_bottom)],
                 fill=(205, 168, 120, 255))
    draw.line([(x_left, y_body_bottom), (x_right, y_body_bottom)], fill=(20, 20, 20, 255), width=1)
    
    # 3. Sharp Graphite Tip (Dark graphite black)
    draw.polygon([(x_graphite_top_left, y_cone_bottom), (x_graphite_top_right, y_cone_bottom),
                  (cx, y_tip)], fill=(18, 18, 20, 255))
    
    # Rotate pencil so body extends up-left and sharp tip points down-right toward paper
    # 315 deg (or -45 deg) places tip at bottom-right of rotated sprite
    rotated = img.rotate(315, resample=Image.BICUBIC, expand=True)
    
    # Compute precise graphite tip coordinates by scanning the rotated alpha & color channels
    arr = np.array(rotated)
    alpha = arr[:, :, 3]
    y_indices, x_indices = np.where(alpha > 100)
    
    rgb = arr[:, :, :3]
    dist_graphite = np.linalg.norm(rgb - [18, 18, 20], axis=2)
    dist_graphite[alpha < 100] = 999999
    
    max_x, max_y = x_indices.max(), y_indices.max()
    
    best_tip_x, best_tip_y = max_x, max_y
    min_score = 999999
    for y in range(max(0, max_y - 12), max_y + 1):
        for x in range(max(0, max_x - 12), max_x + 1):
            if alpha[y, x] > 100:
                # Score balances graphite color proximity and extreme bottom-right tip location
                score = dist_graphite[y, x] + (max_x - x) * 3 + (max_y - y) * 3
                if score < min_score:
                    min_score = score
                    best_tip_x, best_tip_y = x, y
                    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rotated.save(output_path)
    print(f"Pencil sprite created: {output_path} | Size: {rotated.size} | Graphite Tip Anchor: ({best_tip_x}, {best_tip_y})")
    return rotated, best_tip_x, best_tip_y

if __name__ == "__main__":
    create_pencil_sprite()
