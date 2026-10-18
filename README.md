# Realistic Pencil Drawing Animation of Goddess Durga Using Python

An advanced computer vision and 2D drawing animation application that reproduces the **exact reference illustration of Goddess Durga with the Lion** while realistically animating a small graphite drawing pencil physically drawing the artwork on canvas line-by-line.

---

## 🎨 Features & Highlights

1. **Strict Reference Image Preservation**:
   - Reproduces Goddess Durga's face, eyes, eyebrows, lips, bindi, ornate crown, gemstones, flowing hair strands, jewelry, sari, red border, trident, hand grip, lion face, whiskers, detailed mane, circular golden halo, and cream paper background with **100% pixel-perfect fidelity**.

2. **Realistic Graphite Pencil Sprite & Tip Synchronization**:
   - Small, non-rotating drawing pencil angled naturally at 45° relative to the paper.
   - Mathematical anchor-point compensation ensuring the graphite tip stays precisely aligned with every active stroke endpoint.
   - Smooth stroke movement with subtle pencil lifts during transitions between disconnected paths.

3. **11 Artist-Like Drawing Stages**:
   - **Stage 1**: Cream paper texture background setup.
   - **Stage 2**: Golden halo outline & concentric spiral shading.
   - **Stage 3**: Durga's face contours, eyes, eyebrows, nose, lips & bindi.
   - **Stage 4**: Ornate crown (Mukut), gemstones & top peak ornament.
   - **Stage 5**: Long flowing black hair strands in natural directional order.
   - **Stage 6**: Earrings, necklaces, jewelry, sari folds & red sari border.
   - **Stage 7**: Arms, bangles & hand grip around trident.
   - **Stage 8**: Trident (3 prongs, spear head, shaft & tied red ribbon).
   - **Stage 9**: Lion's face, eyes, nose, mouth, whiskers, forehead mark & mane.
   - **Stage 10**: Lotus ornament, fine hatching shading & red accents.
   - **Stage 11**: Final artwork completion.

4. **Interactive Controls & Video Export**:
   - Real-time interactive Pygame canvas window.
   - Dynamic top progress bar & active drawing stage display.
   - Adjustable drawing speed, pause/resume, and reset controls.
   - Built-in OpenCV MP4 video export.

---

## 📁 Project Structure

```
c:/Projects/navratri-devi/
├── assets/
│   ├── reference.jpg          # Original reference artwork of Goddess Durga & Lion
│   ├── paper_bg.jpg           # Extracted cream paper background texture
│   ├── pencil.png             # Realistic graphite pencil sprite
│   ├── dark_mask.png          # Black linework extraction mask
│   └── red_mask.png           # Red accents extraction mask
├── main.py                    # Application entry point and CLI handler
├── image_processing.py        # Image loading, color separation & paper background extraction
├── path_extraction.py         # OpenCV skeletonization, RDP simplification & 11-stage path sorting
├── drawing_animation.py       # Progressive stroke rendering, tip alignment & video exporter
├── pencil.py                  # Pencil sprite procedural generator & anchor offset calculator
├── requirements.txt           # Required Python packages
└── README.md                  # Documentation and setup instructions
```

---

## ⚡ Installation & Setup

1. **Ensure Python 3.10+ is installed**.
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution & Usage

### 1. Run Interactive Animation Window
```bash
python main.py
```
Or specify a custom reference image path:
```bash
python main.py --image path/to/your_image.jpg
```

### 2. Export High-Quality MP4 Video
```bash
python main.py --export --output output_durga_drawing.mp4
```

---

## 🎮 Interactive Keyboard Controls

| Key | Action |
| --- | --- |
| `SPACE` | Pause / Resume animation |
| `+` / `UP` | Increase drawing speed |
| `-` / `DOWN` | Decrease drawing speed |
| `P` | Fast Preview Mode |
| `R` | Restart / Reset animation |
| `E` | Export current animation to MP4 video |
| `ESC` | Exit application |

---

## 🔬 Technical Implementation Details

- **Computer Vision Extraction**: Combines OpenCV adaptive thresholding, color space segmentation (HSV red & gold channel separation), distance transform, and skeletonization.
- **Path Simplification**: Ramer-Douglas-Peucker (RDP) algorithm simplifies raw pixel contours into smooth floating-point curves while retaining line widths.
- **Mathematical Tip Offset**: Pencil tip anchor is calculated relative to the rotated sprite bounding box:
  $$\text{Canvas Position} = (X_{\text{stroke}}, Y_{\text{stroke}})$$
  $$\text{Sprite Position} = (X_{\text{stroke}} - \text{TipOffset}_X, Y_{\text{stroke}} - \text{TipOffset}_Y)$$
- **Progressive Masking**: Line segments are stamped with anti-aliased line masks over the canvas, progressively revealing the exact reference image pixels under the active pencil path.

#output
![Durga Mata Drawing]
("C:\Users\hp\Downloads\durga_mata_output.png.jpg")
