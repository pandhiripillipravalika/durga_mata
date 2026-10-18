import sys
import argparse
import os
from drawing_animation import DrawingAnimation

DEFAULT_REFERENCE_PATH = os.path.join("assets", "reference.jpg")

def main():
    parser = argparse.ArgumentParser(description="Realistic Pencil Drawing Animation of Goddess Durga")
    parser.add_argument("--image", "--reference", type=str, default=DEFAULT_REFERENCE_PATH,
                        help="Path to the reference image (default: assets/reference.jpg)")
    parser.add_argument("--export", action="store_true",
                        help="Export the drawing animation to an MP4 video file")
    parser.add_argument("--output", type=str, default="output_durga_drawing.mp4",
                        help="Output MP4 file path for video export (default: output_durga_drawing.mp4)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames per second for animation window (default: 60)")
    parser.add_argument("--speed", type=float, default=3.0,
                        help="Drawing speed multiplier (default: 3.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Reference image not found at '{args.image}'")
        sys.exit(1)
        
    print("=" * 60)
    print(" Goddess Durga Realistic Pencil Drawing Animation ")
    print("=" * 60)
    print(f" Reference Image : {args.image}")
    print(f" Export Mode     : {'MP4 Video (' + args.output + ')' if args.export else 'Interactive Window'}")
    print(" Controls        : [SPACE] Pause/Resume | [+ / -] Speed | [R] Reset | [E] Export")
    print("=" * 60)
    
    anim = DrawingAnimation(reference_path=args.image, fps=args.fps)
    anim.speed_multiplier = args.speed
    
    if args.export:
        anim.export_video(output_path=args.output, target_fps=30)
    else:
        anim.run_interactive()

if __name__ == "__main__":
    main()
