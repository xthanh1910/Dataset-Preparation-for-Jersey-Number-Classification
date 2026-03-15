# Dataset Preparation for Jersey Number Classification

This repository focuses on the automated extraction and preparation of a jersey number dataset from professional football match footage. It serves as the bridge between raw video data and deep learning models for image classification.

---

## The Extraction Pipeline

This project leverages the same raw dataset structure used in my YOLOv5 detection project but shifts focus to individual player attributes.

### 1. Dynamic Video Frame Extraction
* **Global Index Mapping**: The `FootballDataset` class maps a single global index across multiple video files (each containing ~1,500 frames), allowing for seamless iteration through the entire match series.
* **On-the-fly Decoding**: Utilizes OpenCV's `CAP_PROP_POS_FRAMES` to extract specific frames precisely when requested, saving memory by not storing thousands of extracted images on disk.

### 2. Automated Player Cropping
* **Targeted Filtering**: The pipeline specifically filters for `category_id: 4` (players) and extracts their corresponding bounding boxes from high-fidelity JSON annotations.
* **Dynamic Cropping**: Each player is cropped directly from the video frame in real-time, creating a localized image ready for classification.

### 3. Attribute Parsing
* **Label Extraction**: Automatically parses the `jersey_number` attribute from nested JSON data, pairing each cropped image with its correct numeric label.
* **Data Transformation**: Includes a built-in transformation pipeline (Resize, Normalization, ToTensor) to standardize player crops into a uniform $224 \times 224$ format.

---
