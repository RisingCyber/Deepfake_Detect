"""
deepfake_detect.py
==================
Deepfake & Image Forgery Detection — Unified Pipeline
Australian Phoenix CyberOps | Chadi Saliby

Runs three forensic detection methods against any JPEG image:
  1. EXIF Metadata Audit
  2. Error Level Analysis (ELA)
  3. Discrete Fourier Transform (DFT) Frequency Analysis

Usage:
    python deepfake_detect.py --image Lab/suspect.jpg

Output:
    - Console: EXIF audit results with flagged fields
    - Image:   Lab/suspect_analysis.png  (3-panel forensic report)
    - Report:  Lab/suspect_report.txt    (plain-text findings summary)

Requirements:
    pip install Pillow numpy matplotlib exifread
    
Dependencies:
pip install Pillow numpy matplotlib exifread

"""
