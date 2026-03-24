#!/usr/bin/env python
# coding: utf-8

# # 🧬 Cell Viability Analyzer
# 
# **🎯 Automated Cell Viability Analyzer System:**
# 1. **① Circular cells before fixation** (initial frames)
# 2. **② Fixed cells** (midpoint frames) 
# 3. **③ Circular dead cells** (from midpoint onwards)
# 4. **④ Fragments** (debris throughout)
# 
# **📹 Outputs:**
# - AVI mask videos for each class
# - Combined AVI video (original + masks)
# - CSV data table with counts, areas, brightness per frame
# 
# **🔧 Features:**
# - Interactive path input
# - Two classification methods
# - Automatic file validation
# - Publication-ready outputs

# ## 🔸 **STEP 1: Install Required Packages**

# Install required packages (run once)
import subprocess
import sys
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cellpose[gui]>=4.0", "opencv-python-headless", "scikit-image", "pandas", "matplotlib", "torch", "torchvision", "tqdm"])
    print("✅ All packages installed successfully!")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Package installation failed: {e}")
    print("Please install manually: pip install cellpose[gui]>=4.0 opencv-python-headless scikit-image pandas matplotlib torch torchvision tqdm")

# ## 🔸 **STEP 2: Import Libraries and Configure Paths**

# Import libraries and configure paths
import os
import cv2
import numpy as np
import pandas as pd
from cellpose import models
from skimage.measure import regionprops
import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Configuration
print("📁 Configuring image directories...")

def get_valid_path(prompt_text, default_path=None, must_exist=True):
    """Get and validate path input from user"""
    while True:
        if default_path:
            path_input = input(f"{prompt_text} (default: {default_path}): ").strip()
            if not path_input:
                path_input = default_path
        else:
            path_input = input(f"{prompt_text}: ").strip()
        
        # Add trailing slash if not present
        if not path_input.endswith('/'):
            path_input += '/'
        
        # Validate path exists if required
        if must_exist and not os.path.exists(path_input):
            print(f"❌ Path does not exist: {path_input}")
            print("   Please enter a valid path or press Ctrl+C to exit")
            continue
        
        return path_input

# Get directories
IMAGE_DIR = get_valid_path("Enter path to image directory")
OUTPUT_DIR = get_valid_path("Enter output directory", "./output", must_exist=False)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
CLASS1_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'class1_circular_before_fixation.avi')
CLASS2_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'class2_fixed_cells.avi')
CLASS3_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'class3_circular_dead_cells.avi')
CLASS4_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'class4_fragments_debris.avi')
COMBINED_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'combined_all_classes.avi')
DATA_TABLE_PATH = os.path.join(OUTPUT_DIR, 'cell_measurement_data.csv')

LIVE_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'live_cells_area_based.avi')
FIXED_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'fixed_cells_area_based.avi')
DEAD_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'dead_cells_area_based.avi')
DEBRIS_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'debris_area_based.avi')
COMBINED_AREA_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'combined_area_based.avi')
DATA_AREA_TABLE_PATH = os.path.join(OUTPUT_DIR, 'cell_measurement_area_based.csv')

# Analysis parameters
MIN_CELL_AREA = 100
CIRCULARITY_THRESHOLD = 0.7
FIXATION_FRAME_RATIO = 0.5
VIDEO_FPS = 10

# Class colors (BGR format)
class_colors = {
    'live': (0, 255, 0),
    'fixed': (255, 0, 0),
    'dead': (0, 0, 255),
    'debris': (0, 255, 255)
}

# Area-based classification thresholds
LIVE_MIN_AREA = 26
LIVE_MAX_AREA = 350
LIVE_BRIGHTNESS_THRESHOLD = 100
FIXED_MIN_AREA = 29
FIXED_MAX_AREA = 2770
FIXED_BRIGHTNESS_MIN = 100
FIXED_BRIGHTNESS_MAX = 172
DEAD_MIN_AREA = 1
DEAD_MAX_AREA = 12
DEAD_BRIGHTNESS_MAX = 250
MIN_DEBRIS_AREA = 10

print(f"📁 Input: {IMAGE_DIR}")
print(f"📁 Output: {OUTPUT_DIR}")
print(f"🎛️ Fixation point: {FIXATION_FRAME_RATIO*100}% through sequence")
print(f"⭕ Circularity threshold: {CIRCULARITY_THRESHOLD}")

print(f"\n📊 Area-Based Classification Thresholds:")
print(f"   🟢 Live: Area {LIVE_MIN_AREA}-{LIVE_MAX_AREA} px², Brightness ≥{LIVE_BRIGHTNESS_THRESHOLD}")
print(f"   🔵 Fixed: Area {FIXED_MIN_AREA}-{FIXED_MAX_AREA} px², Brightness {FIXED_BRIGHTNESS_MIN}-{FIXED_BRIGHTNESS_MAX}")
print(f"   🔴 Dead: Area {DEAD_MIN_AREA}-{DEAD_MAX_AREA} px², Brightness ≤{DEAD_BRIGHTNESS_MAX}")
print(f"   🟡 Debris: Area <{MIN_DEBRIS_AREA} px² or doesn't match other criteria")

# GPU detection and model initialization
def detect_gpu():
    """Detect if CUDA/GPU is available for acceleration."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detected: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        print(f"   Number of GPUs: {gpu_count}")
        return True
    else:
        print("⚠️  No CUDA-compatible GPU detected. Using CPU.")
        print("   For faster processing, consider using GPU.")
        return False

use_gpu = detect_gpu()

print(f"\n🤖 Loading Cellpose4 model...")
try:
    model = models.CellposeModel(gpu=use_gpu, model_type='cyto3')
    print("✅ Model ready for analysis!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("🔄 Falling back to cyto model...")
    model = models.CellposeModel(gpu=use_gpu, model_type='cyto')
    print("✅ Fallback model ready!")

# Find and validate image files
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))
if not image_files:
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
if not image_files:
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.tif')))

if image_files:
    print(f"✅ Found {len(image_files)} images")
    print(f"📄 Format: {os.path.splitext(image_files[0])[1].upper()}")
    print(f"📂 First image: {os.path.basename(image_files[0])}")
    print(f"📂 Last image: {os.path.basename(image_files[-1])}")
    
    total_frames = len(image_files)
    fixation_frame = int(total_frames * FIXATION_FRAME_RATIO)

    print(f"\n🧬 Temporal Classification Timeline:")
    print(f"   ① Circular before fixation: Frames 1-{fixation_frame}")
    print(f"   ② Fixed cells: Frames {fixation_frame+1}-{total_frames}")
    print(f"   ③ Circular dead cells: Frames {fixation_frame+1}-{total_frames} (circular only)")
    print(f"   ④ Fragments: All frames (small irregular objects)")

    print(f"\n🔬 Area-Based Classification:")
    print(f"   Each cell classified by area and brightness thresholds")
    print(f"   Priority: Dead → Live → Fixed → Debris (default)")
else:
    print("❌ No images found! Check your IMAGE_DIR path.")
    print("   Please run the script again with correct path.")

# Analysis functions

def analyze_cell_viability_temporal():
    """Temporal + circularity based classification"""

    if not image_files:
        print("❌ No images to process!")
        return None

    print(f"🚀 Starting temporal analysis of {len(image_files)} frames...")

    # Get image dimensions
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]

    # Initialize video writers (AVI format)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    class1_video = cv2.VideoWriter(CLASS1_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    class2_video = cv2.VideoWriter(CLASS2_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    class3_video = cv2.VideoWriter(CLASS3_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    class4_video = cv2.VideoWriter(CLASS4_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    combined_video = cv2.VideoWriter(COMBINED_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))

    print(f"📹 Initialized 5 temporal video writers (AVI format)")

    # Data storage
    all_frame_data = []
    total_frames = len(image_files)
    fixation_frame = int(total_frames * FIXATION_FRAME_RATIO)

    # Class colors (BGR format)
    temporal_colors = {
        1: (0, 255, 0),      # Green - Circular before fixation
        2: (255, 0, 0),      # Blue - Fixed cells
        3: (0, 0, 255),      # Red - Circular dead cells
        4: (0, 255, 255)     # Yellow - Fragments/debris
    }

    # Progress tracking
    progress_bar = tqdm(enumerate(image_files, 1), total=len(image_files), desc="🕐 Temporal")

    for frame_num, image_path in progress_bar:
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Could not load image: {image_path}")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Cellpose segmentation
            masks, flows, styles = model.eval(
                image,
                diameter=None,
                channels=[0,0],
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

            # Initialize frame data
            frame_data = {
                'frame': frame_num,
                'class1_count': 0, 'class1_avg_area': 0, 'class1_avg_brightness': 0,
                'class2_count': 0, 'class2_avg_area': 0, 'class2_avg_brightness': 0,
                'class3_count': 0, 'class3_avg_area': 0, 'class3_avg_brightness': 0,
                'class4_count': 0, 'class4_avg_area': 0, 'class4_avg_brightness': 0,
            }

            # Create separate masks for each class
            class_masks = {1: np.zeros_like(image), 2: np.zeros_like(image),
                          3: np.zeros_like(image), 4: np.zeros_like(image)}
            
            # Initialize data storage for areas and brightness
            class_areas = {1: [], 2: [], 3: [], 4: []}
            class_brightness = {1: [], 2: [], 3: [], 4: []}

            # Analyze detected objects
            props = regionprops(masks, intensity_image=gray_image)

            for prop in props:
                area = prop.area
                mean_intensity = prop.mean_intensity
                perimeter = prop.perimeter

                # Calculate circularity
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0

                # Temporal classification logic
                cell_class = 4  # Default: debris

                if area >= MIN_CELL_AREA:
                    frame_position = frame_num / total_frames

                    if frame_position <= FIXATION_FRAME_RATIO:
                        # Before fixation - all cells are class 1
                        cell_class = 1
                    else:
                        # After fixation - classify by circularity
                        if circularity >= CIRCULARITY_THRESHOLD:
                            cell_class = 3  # Circular dead cells
                        else:
                            cell_class = 2  # Fixed cells

                # Store classification results
                frame_data[f'class{cell_class}_count'] += 1
                class_areas[cell_class].append(area)
                class_brightness[cell_class].append(mean_intensity)

                # Color mask for each class separately
                class_masks[cell_class][masks == prop.label] = temporal_colors[cell_class]

            # Calculate averages for each class
            for cell_class in [1, 2, 3, 4]:
                if frame_data[f'class{cell_class}_count'] > 0:
                    frame_data[f'class{cell_class}_avg_area'] = np.mean(class_areas[cell_class])
                    frame_data[f'class{cell_class}_avg_brightness'] = np.mean(class_brightness[cell_class])

            all_frame_data.append(frame_data)

            # Write separate video frames for each class
            class1_video.write(class_masks[1])
            class2_video.write(class_masks[2])
            class3_video.write(class_masks[3])
            class4_video.write(class_masks[4])

            # Combined frame with transparency
            combined_mask = np.zeros_like(image)
            for class_num in [1, 2, 3, 4]:
                combined_mask = cv2.add(combined_mask, class_masks[class_num])
            combined_frame = cv2.addWeighted(image, 0.6, combined_mask, 0.4, 0)
            combined_video.write(combined_frame)

        except Exception as e:
            print(f"❌ Error processing frame {frame_num}: {e}")
            continue

    # Release video writers
    class1_video.release()
    class2_video.release()
    class3_video.release()
    class4_video.release()
    combined_video.release()

    # Save data to CSV
    if all_frame_data:
        df = pd.DataFrame(all_frame_data)
        df.to_csv(DATA_TABLE_PATH, index=False)

        print(f"\n🎉 Temporal Analysis Complete!")
        print(f"📁 Files saved to: {OUTPUT_DIR}")
        print(f"   📹 Class videos: 4 files")
        print(f"   📹 Combined video: 1 file")
        print(f"   📊 Data table: {os.path.basename(DATA_TABLE_PATH)}")

        return df
    else:
        print("❌ No data was processed successfully!")
        return None

def analyze_cell_viability_area_based():
    """Area and brightness based classification system"""

    if not image_files:
        print("❌ No images to process!")
        return None

    print(f"🚀 Starting area-based analysis of {len(image_files)} frames...")

    # Get image dimensions
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]

    # Initialize video writers (AVI format)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    live_video = cv2.VideoWriter(LIVE_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    fixed_video = cv2.VideoWriter(FIXED_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    dead_video = cv2.VideoWriter(DEAD_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    debris_video = cv2.VideoWriter(DEBRIS_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    combined_video = cv2.VideoWriter(COMBINED_AREA_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))

    print(f"📹 Initialized 5 area-based video writers (AVI format)")

    # Data storage
    all_frame_data = []

    # Class colors (BGR format)
    area_colors = {
        'live': (0, 255, 0),      # Green - Live cells
        'fixed': (255, 0, 0),      # Blue - Fixed cells
        'dead': (0, 0, 255),      # Red - Dead cells
        'debris': (0, 255, 255)    # Yellow - Debris/fragments
    }

    # Progress tracking
    progress_bar = tqdm(enumerate(image_files, 1), total=len(image_files), desc="🔬 Area-based")

    for frame_num, image_path in progress_bar:
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Could not load image: {image_path}")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Cellpose segmentation
            masks, flows, styles = model.eval(
                image,
                diameter=None,
                channels=[0,0],
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

            # Initialize frame data
            frame_data = {
                'frame': frame_num,
                'live_count': 0, 'live_avg_area': 0, 'live_avg_brightness': 0,
                'fixed_count': 0, 'fixed_avg_area': 0, 'fixed_avg_brightness': 0,
                'dead_count': 0, 'dead_avg_area': 0, 'dead_avg_brightness': 0,
                'debris_count': 0, 'debris_avg_area': 0, 'debris_avg_brightness': 0,
            }

            # Create separate masks for each class
            class_masks = {'live': np.zeros_like(image), 'fixed': np.zeros_like(image),
                          'dead': np.zeros_like(image), 'debris': np.zeros_like(image)}
            
            # Initialize data storage for areas and brightness
            class_areas = {'live': [], 'fixed': [], 'dead': [], 'debris': []}
            class_brightness = {'live': [], 'fixed': [], 'dead': [], 'debris': []}

            # Analyze detected objects
            props = regionprops(masks, intensity_image=gray_image)

            for prop in props:
                area = prop.area
                mean_intensity = prop.mean_intensity

                # Area + brightness classification logic
                cell_class = 'debris'  # Default classification

                if DEAD_MIN_AREA <= area <= DEAD_MAX_AREA and mean_intensity <= DEAD_BRIGHTNESS_MAX:
                    cell_class = 'dead'
                elif LIVE_MIN_AREA <= area <= LIVE_MAX_AREA and mean_intensity >= LIVE_BRIGHTNESS_THRESHOLD:
                    cell_class = 'live'
                elif FIXED_MIN_AREA <= area <= FIXED_MAX_AREA and FIXED_BRIGHTNESS_MIN <= mean_intensity <= FIXED_BRIGHTNESS_MAX:
                    cell_class = 'fixed'

                # Store classification results
                frame_data[f'{cell_class}_count'] += 1
                class_areas[cell_class].append(area)
                class_brightness[cell_class].append(mean_intensity)

                # Color mask for each class separately
                class_masks[cell_class][masks == prop.label] = area_colors[cell_class]

            # Calculate averages for each class
            for cell_class in ['live', 'fixed', 'dead', 'debris']:
                if frame_data[f'{cell_class}_count'] > 0:
                    frame_data[f'{cell_class}_avg_area'] = np.mean(class_areas[cell_class])
                    frame_data[f'{cell_class}_avg_brightness'] = np.mean(class_brightness[cell_class])

            all_frame_data.append(frame_data)

            # Write separate video frames for each class
            live_video.write(class_masks['live'])
            fixed_video.write(class_masks['fixed'])
            dead_video.write(class_masks['dead'])
            debris_video.write(class_masks['debris'])

            # Combined frame with transparency
            combined_mask = np.zeros_like(image)
            for cell_class in ['live', 'fixed', 'dead', 'debris']:
                combined_mask = cv2.add(combined_mask, class_masks[cell_class])
            combined_frame = cv2.addWeighted(image, 0.6, combined_mask, 0.4, 0)
            combined_video.write(combined_frame)

        except Exception as e:
            print(f"❌ Error processing frame {frame_num}: {e}")
            continue

    # Release video writers
    live_video.release()
    fixed_video.release()
    dead_video.release()
    debris_video.release()
    combined_video.release()

    # Save data to CSV
    if all_frame_data:
        df = pd.DataFrame(all_frame_data)
        df.to_csv(DATA_AREA_TABLE_PATH, index=False)

        print(f"\n🎉 Area-Based Analysis Complete!")
        print(f"📁 Files saved to: {OUTPUT_DIR}")
        print(f"   📹 Class videos: 4 files")
        print(f"   📹 Combined video: 1 file")
        print(f"   📊 Data table: {os.path.basename(DATA_AREA_TABLE_PATH)}")

        return df
    else:
        print("❌ No data was processed successfully!")
        return None

print("✅ Analysis functions defined and ready!")

# Run analysis

if image_files:
    print("\n🔬 Choose Analysis Method:")
    print("1️⃣ Temporal + Circularity (biological timeline)")
    print("2️⃣ Area + Brightness based (threshold classification)")
    print("3️⃣ Both methods (comparison)")

    analysis_method = input("Enter choice (1/2/3): ")

    if analysis_method == '1':
        print("🔬 Running Temporal + Circularity based analysis...")
        results_df = analyze_cell_viability_temporal()
        classification_stats = None
    elif analysis_method == '2':
        print("🔬 Running Area + Brightness based analysis...")
        results_df = analyze_cell_viability_area_based()
        classification_stats = None
    elif analysis_method == '3':
        print("🔬 Running BOTH analysis methods...")
        
        print("\n📊 Method 1: Temporal + Circularity")
        temporal_df = analyze_cell_viability_temporal()
        
        print("\n📊 Method 2: Area + Brightness")
        area_df = analyze_cell_viability_area_based()
        
        if temporal_df is not None and area_df is not None:
            print("\n✅ Both analyses completed!")
            print("📊 You can compare results from both methods")
            results_df = temporal_df  # Use temporal for display
        else:
            print("❌ One or both analyses failed!")
            results_df = None
    else:
        print("❌ Invalid choice. Please run script again.")
        results_df = None

else:
    print("❌ No images found to process!")

# Display results

if 'results_df' in locals() and results_df is not None:
    print("\n✅ Analysis completed successfully!")
    print("📋 Sample results:")
    print(results_df.head(5))
else:
    print("\n❌ No results generated. Please check for errors above.")

print("\n" + "="*60)
print("🎯 ANALYSIS COMPLETE!")
print("📁 All results saved to:", OUTPUT_DIR)
print("📊 Thank you for using Cell Viability Analyzer!")
print("="*60)
