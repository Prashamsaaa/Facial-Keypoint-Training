# Facial Key Point Detection
## Overview
This project features a facial key point detection system using a modified VGG16 architecture. It identifies key facial points, such as eyes, nose, and mouth, from images. Built with PyTorch, it utilizes transfer learning for efficient and accurate inference. A Streamlit app is included for easy interaction and demonstration.

## Features
- **Modified VGG16 Architecture**: Utilizes a tailored version of the VGG16 model for enhanced performance in facial key point detection.
- **Image Preprocessing**: Automatically resizes and normalizes images for optimal model input.
- **Key Point Prediction**: Outputs the coordinates of key facial points for any given input image.
- **Interactive Demo**: A Streamlit app allows users to upload images and visualize detected key points in real time.

## Requirements

- Python 3.x
- PyTorch
- Pillow
- NumPy
- Streamlit

## Usage

1. **Clone the Repository**:

   
```
   git clone https://github.com/Prashamsaaa/Facial-Keypoint-Training
   cd facial-key-point-detection
```

2. **Install required packages**:

```
    pip install requirements.txt
```

3. **Run the Streamlit Application**:

   To launch the demo, run:

   ```
   streamlit run demo/demo.py
   ```

