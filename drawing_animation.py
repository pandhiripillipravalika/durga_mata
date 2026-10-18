import cv2
import numpy as np
import pygame
import sys
import time
import os
from PIL import Image

from image_processing import load_reference_image, extract_masks, create_pure_white_reference_canvas
from path_extraction import extract_all_drawing_strokes, Stroke
from pencil import create_pencil_sprite

class DrawingAnimation:
    def __init__(self, reference_path="assets/reference.jpg", width=723, height=1024, fps=60):
        self.reference_path = reference_path
        self.width = width
        self.height = height
        self.fps = fps
        
        # Load reference image
        self.img_bgr, self.img_rgb = load_reference_image(reference_path)
        if self.img_rgb.shape[:2] != (height, width):
            self.img_rgb = cv2.resize(self.img_rgb, (width, height), interpolation=cv2.INTER_AREA)
            self.img_bgr = cv2.resize(self.img_bgr, (width, height), interpolation=cv2.INTER_AREA)
            
        # Extract masks and build clean pure white reference canvas
        dark_mask, red_mask = extract_masks(self.img_bgr)
        self.clean_ref_canvas = create_pure_white_reference_canvas(self.img_bgr, dark_mask, red_mask)
        
        # Extract drawing stroke paths
        self.strokes = extract_all_drawing_strokes(dark_mask, red_mask)
        
        # Initialize working canvas as PURE WHITE (RGB 255, 255, 255)
        self.white_bg = np.full((height, width, 3), 255, dtype=np.uint8)
        self.canvas_rgb = self.white_bg.copy()
        
        # Mask layer to keep track of revealed artwork pixels
        self.revealed_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create pencil sprite and get exact graphite tip anchor offset
        self.pencil_img, self.tip_x, self.tip_y = create_pencil_sprite(
            output_path="assets/pencil.png", length=90, width=14, angle_deg=45
        )
        self.pencil_surface = None # Converted in run_interactive after display init
        
        # Animation state variables
        self.current_stroke_idx = 0
        self.current_point_idx = 0
        self.pencil_pos = (width // 2, height // 2)
        self.pencil_is_down = False
        self.is_paused = False
        self.is_finished = False
        self.speed_multiplier = 3.0 # Points per frame
        
        # Total points calculation for progress bar
        self.total_points = sum(len(s.points) for s in self.strokes)
        self.drawn_points = 0
        
    def render_stroke_segment(self, pt1, pt2, width, is_red=False):
        """
        Renders a line segment from pt1 to pt2 by progressively revealing
        the clean artwork lines onto the pure white canvas.
        """
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        
        r = max(1, int(round(width / 2.0 + 0.8)))
        
        # Draw onto revealed_mask
        cv2.line(self.revealed_mask, (x1, y1), (x2, y2), 255, thickness=r*2, lineType=cv2.LINE_AA)
        
        # Update canvas_rgb in bounding box of line segment
        min_x = max(0, min(x1, x2) - r - 2)
        max_x = min(self.width, max(x1, x2) + r + 3)
        min_y = max(0, min(y1, y2) - r - 2)
        max_y = min(self.height, max(y1, y2) + r + 3)
        
        if max_x > min_x and max_y > min_y:
            sub_mask = (self.revealed_mask[min_y:max_y, min_x:max_x] > 0)[:, :, np.newaxis]
            ref_sub = self.clean_ref_canvas[min_y:max_y, min_x:max_x]
            white_sub = self.white_bg[min_y:max_y, min_x:max_x]
            
            self.canvas_rgb[min_y:max_y, min_x:max_x] = np.where(sub_mask, ref_sub, white_sub)

    def update(self):
        """Advances the drawing animation by speed_multiplier points per frame."""
        if self.is_paused or self.is_finished:
            return
            
        if self.current_stroke_idx >= len(self.strokes):
            # Final pass: set canvas to clean pure white reference image
            self.canvas_rgb = self.clean_ref_canvas.copy()
            self.is_finished = True
            return
            
        stroke = self.strokes[self.current_stroke_idx]
        pts = stroke.points
        
        points_to_draw = int(self.speed_multiplier) + (1 if np.random.rand() < (self.speed_multiplier % 1) else 0)
        points_to_draw = max(1, points_to_draw)
        
        for _ in range(points_to_draw):
            if self.current_point_idx < len(pts) - 1:
                pt1 = pts[self.current_point_idx]
                pt2 = pts[self.current_point_idx + 1]
                
                self.render_stroke_segment(pt1, pt2, stroke.width, stroke.is_red)
                
                # Graphite tip follows exact stroke endpoint coordinate
                self.pencil_pos = (float(pt2[0]), float(pt2[1]))
                self.pencil_is_down = True
                
                self.current_point_idx += 1
                self.drawn_points += 1
            else:
                self.current_stroke_idx += 1
                self.current_point_idx = 0
                if self.current_stroke_idx < len(self.strokes):
                    next_stroke = self.strokes[self.current_stroke_idx]
                    next_pt = next_stroke.points[0]
                    self.pencil_pos = (float(next_pt[0]), float(next_pt[1]))
                    self.pencil_is_down = False
                break

    def draw_progress_bar(self, surface):
        """Draws top progress bar overlay."""
        bar_height = 28
        overlay = pygame.Surface((self.width, bar_height), pygame.SRCALPHA)
        overlay.fill((25, 28, 35, 215))
        surface.blit(overlay, (0, 0))
        
        pct = min(1.0, self.drawn_points / max(1, self.total_points)) if not self.is_finished else 1.0
        
        bar_w = int((self.width - 20) * pct)
        pygame.draw.rect(surface, (230, 165, 40), (10, bar_height - 6, bar_w, 4))
        
        current_stage = 11
        if self.current_stroke_idx < len(self.strokes):
            current_stage = self.strokes[self.current_stroke_idx].stage
            
        stage_names = {
            1: "Stage 1: Pure White Canvas Setup",
            2: "Stage 2: Goddess Structural Outlines",
            3: "Stage 3: Durga Facial Features & Bindi",
            4: "Stage 4: Ornate Crown & Gemstones",
            5: "Stage 5: Flowing Hair Strands",
            6: "Stage 6: Jewelry, Sari & Red Border",
            7: "Stage 7: Hand Grip & Bangles",
            8: "Stage 8: Trident & Tied Red Ribbon",
            9: "Stage 9: Lion Facial Features & Mane",
            10: "Stage 10: Lotus Ornament & Shading",
            11: "Stage 11: Artwork Complete"
        }
        
        font = pygame.font.SysFont("Arial", 13, bold=True)
        txt_str = f"{stage_names.get(current_stage, 'Drawing...')} | {pct*100:.1f}% | [SPACE] Pause | [+ / -] Speed | [R] Reset | [E] Video"
        txt_surf = font.render(txt_str, True, (245, 245, 245))
        surface.blit(txt_surf, (15, 4))

    def run_interactive(self):
        """Runs the Pygame interactive window loop."""
        pygame.init()
        pygame.display.set_caption("Goddess Durga Pencil Drawing Animation (Pure White Canvas)")
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        
        pencil_bytes = self.pencil_img.tobytes()
        self.pencil_surface = pygame.image.fromstring(
            pencil_bytes, self.pencil_img.size, "RGBA"
        ).convert_alpha()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS or event.key == pygame.K_UP:
                        self.speed_multiplier = min(30.0, self.speed_multiplier + 1.5)
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_DOWN:
                        self.speed_multiplier = max(0.5, self.speed_multiplier - 1.5)
                    elif event.key == pygame.K_p:
                        self.speed_multiplier = 25.0
                    elif event.key == pygame.K_e:
                        print("Exporting MP4 video...")
                        self.export_video("output_durga_drawing.mp4")
                        
            self.update()
            
            canvas_surf = pygame.surfarray.make_surface(np.transpose(self.canvas_rgb, (1, 0, 2)))
            screen.blit(canvas_surf, (0, 0))
            
            if not self.is_finished:
                lift_offset_y = 0 if self.pencil_is_down else -5
                # Exact graphite tip placement formula: T(tip_x, tip_y) = (draw_x, draw_y)
                px = int(round(self.pencil_pos[0])) - self.tip_x
                py = int(round(self.pencil_pos[1])) - self.tip_y + lift_offset_y
                screen.blit(self.pencil_surface, (px, py))
                
            self.draw_progress_bar(screen)
            
            pygame.display.flip()
            clock.tick(self.fps)
            
        pygame.quit()

    def reset(self):
        self.canvas_rgb = self.white_bg.copy()
        self.revealed_mask.fill(0)
        self.current_stroke_idx = 0
        self.current_point_idx = 0
        self.drawn_points = 0
        self.is_finished = False
        self.is_paused = False

    def export_video(self, output_path="output_durga_drawing.mp4", target_fps=30):
        """Exports the drawing animation to an MP4 video file using OpenCV VideoWriter."""
        print(f"Exporting video to {output_path}...")
        self.reset()
        self.speed_multiplier = 6.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (self.width, self.height))
        
        pencil_np = np.array(self.pencil_img)
        p_h, p_w = pencil_np.shape[:2]
        
        frame_count = 0
        while not self.is_finished:
            self.update()
            
            frame_bgr = cv2.cvtColor(self.canvas_rgb, cv2.COLOR_RGB2BGR)
            
            if not self.is_finished:
                lift_offset_y = 0 if self.pencil_is_down else -5
                px = int(round(self.pencil_pos[0])) - self.tip_x
                py = int(round(self.pencil_pos[1])) - self.tip_y + lift_offset_y
                
                y1, y2 = max(0, py), min(self.height, py + p_h)
                x1, x2 = max(0, px), min(self.width, px + p_w)
                
                py1, py2 = max(0, -py), min(p_h, self.height - py)
                px1, px2 = max(0, -px), min(p_w, self.width - px)
                
                if y2 > y1 and x2 > x1 and py2 > py1 and px2 > px1:
                    alpha = (pencil_np[py1:py2, px1:px2, 3] / 255.0)[:, :, np.newaxis]
                    p_bgr = pencil_np[py1:py2, px1:px2, :3][:, :, ::-1]
                    frame_bgr[y1:y2, x1:x2] = (1.0 - alpha) * frame_bgr[y1:y2, x1:x2] + alpha * p_bgr
                    
            out.write(frame_bgr.astype(np.uint8))
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Exported {frame_count} frames... ({self.drawn_points}/{self.total_points} points)")
                
        final_bgr = cv2.cvtColor(self.clean_ref_canvas, cv2.COLOR_RGB2BGR)
        for _ in range(target_fps * 3):
            out.write(final_bgr)
            
        out.release()
        print(f"Video export complete! Saved to {output_path}. Total frames: {frame_count + target_fps*3}")

if __name__ == "__main__":
    anim = DrawingAnimation()
    if "--export" in sys.argv:
        anim.export_video("output_durga_drawing.mp4")
    else:
        anim.run_interactive()
