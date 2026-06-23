

Deepfake Detector
==================
Deepfake & Image Forgery Detection


Australian Phoenix CyberOps | Chadi Saliby


Type: Experimental  

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

<img width="592" height="120" alt="struct" src="https://github.com/user-attachments/assets/e53dd5c3-e706-4b6b-831c-c906fbf8b36a" />

