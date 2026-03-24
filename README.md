# 🧬 Cell Viability Analysis

A comprehensive Python script for automated cell viability analysis using AI-powered cell segmentation with dual classification methods.

## 🎯 Features

- **🤖 AI-Powered Segmentation**: Uses Cellpose4 with SAM integration for accurate cell detection
- **🔬 Dual Classification Methods**: 
  - Temporal + Circularity (biological timeline-based)
  - Area & Brightness (threshold-based)
- **📹 Video Outputs**: Separate AVI videos for each cell class plus combined overlays
- **📊 Data Export**: CSV files with frame-by-frame quantitative measurements
- **🎨 Publication Ready**: Professional outputs suitable for scientific publications
- **💻 Cross-Platform**: Works on local machines, Google Colab, and HPC systems

## 📋 Classification Systems

### Method 1: Temporal + Circularity
- **🟢 Circular before fixation**: Initial frames, round/healthy cells
- **🔵 Fixed cells**: Mid-point frames, cells after fixation treatment  
- **🔴 Circular dead cells**: Later frames, round cells that died after fixation
- **🟡 Fragments/debris**: Small irregular pieces throughout

### Method 2: Area & Brightness Based
- **🟢 Live cells**: Area 26-350 px², Brightness ≥100
- **🔵 Fixed cells**: Area 29-2770 px², Brightness 100-172
- **🔴 Dead cells**: Area 1-12 px², Brightness ≤250
- **🟡 Debris**: Area <10 px² or doesn't match other criteria

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cell-viability-analysis.git
cd cell-viability-analysis

# Install dependencies
pip install cellpose[gui]>=4.0 opencv-python-headless scikit-image pandas matplotlib torch torchvision tqdm
```

### Usage
```bash
python Cell_Viability_Analysis.py
```

The script will interactively prompt for:
1. **Input directory**: Path to your image files (JPG/PNG/TIF)
2. **Output directory**: Where results will be saved
3. **Analysis method**: Choose between temporal, area-based, or both

## 📁 Input Requirements

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)

### Image Organization
```
your_images/
├── frame_001.jpg
├── frame_002.jpg
├── frame_003.jpg
└── ...
```

Images should be sequentially ordered and represent a time-lapse sequence.

## 📁 Output Files

### Video Files (AVI format)
- `class1_circular_before_fixation.avi` - Circular cells before fixation
- `class2_fixed_cells.avi` - Fixed cells
- `class3_circular_dead_cells.avi` - Circular dead cells
- `class4_fragments_debris.avi` - Fragments and debris
- `combined_all_classes.avi` - All classes overlaid on original images

### Data Files (CSV format)
- `cell_measurement_data.csv` - Temporal method measurements
- `cell_measurement_area_based.csv` - Area-based method measurements

### CSV Data Structure
```csv
frame,class1_count,class1_avg_area,class1_avg_brightness,class2_count,class2_avg_area,class2_avg_brightness,...
1,15,245.3,132.7,8,567.2,98.4,...
2,17,238.1,128.9,12,598.7,102.3,...
...
```

## ⚙️ Configuration

### Analysis Parameters
```python
# Temporal method
MIN_CELL_AREA = 100              # Minimum area for cells (pixels²)
CIRCULARITY_THRESHOLD = 0.7      # Circularity threshold
FIXATION_FRAME_RATIO = 0.5       # Fixation point (50% through sequence)

# Area-based method
LIVE_MIN_AREA = 26               # Live cell minimum area
LIVE_MAX_AREA = 350              # Live cell maximum area
LIVE_BRIGHTNESS_THRESHOLD = 100  # Live cell brightness threshold
# ... (see full parameter list in script)
```

### Video Settings
- **Format**: AVI (XVID codec)
- **Frame Rate**: 10 FPS
- **Resolution**: Matches input images

## 🔧 Advanced Usage

### GPU Acceleration
The script automatically detects and uses CUDA-compatible GPUs for faster processing:

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Batch Processing
For multiple datasets, create a simple batch script:

```bash
#!/bin/bash
datasets=("dataset1" "dataset2" "dataset3")

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python Cell_Viability_Analysis.py << EOF
/path/to/$dataset/images
/path/to/$dataset/output
1
EOF
done
```

### Custom Parameters
Modify the parameters directly in the script for your specific cell types and experimental conditions.

## 🐛 Troubleshooting

### Common Issues

**"No images found!"**
- Check that your image directory path is correct
- Ensure images are in supported formats (JPG/PNG/TIF)
- Verify file permissions

**"GPU not detected"**
- Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Update NVIDIA drivers
- The script will automatically fall back to CPU processing

**"Memory error"**
- Reduce image resolution or process in smaller batches
- Close other memory-intensive applications
- Use a machine with more RAM

**"Cellpose model error"**
- Ensure internet connection for first-time model download
- Check firewall/proxy settings
- Models will be cached locally after first download

### Performance Tips

1. **Use GPU**: 10-50x faster than CPU
2. **SSD Storage**: Faster image loading
3. **Batch Size**: Process 100-500 images at once
4. **Image Size**: 1024x1024 or smaller recommended
