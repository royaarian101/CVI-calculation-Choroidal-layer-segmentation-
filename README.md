# CVI-calculation-Choroidal-layer-segmentation-
## Automatic choroid vascularity index calculation in optical coherence tomography images with low-contrast sclerochoroidal junction using deep learning

# Introduction
This project aims to calculate Choroid Vascularity Index (CVI) in optical coherenece tomography (OCT) images, using loss modified U-Net. 
The method is detailed in "Automatic Choroid Vascularity Index Calculation in Optical Coherence Tomography Images low contrast sclerochoroidal junction Using Deep Learning".


# Dependencies
  •	Python 3.7+
  
  •	Tensorflow 1.15.2
  
  •	Keras


# Dataset
We used Enhanced-depth imaging optical coherence tomography images from two patient groups. 

   •	First dataset is including Raster OCT B-scans from patients with diabetic retinopathy.
 
   •	Second dataset is including EDI-HD OCT B-scans from patients with pachychoroid spectrum.
https://drive.google.com/drive/folders/1a9cmSmYwpob3sybTn0ivbK5T2uGMkYNt

# Network
You can use or define your network in CVI_net.py. Two baseline network has been provided in CVI_net.py to use for training.
For each network, a test file (CVI_net_just test data.py) and two model (saved model for raster data.h5 and saved model for EDI data.h5) have been provided using saved weights for more simplifications.

# Citing
**Please ensure to include the following citations when utilizing any part of the code:**
Arian, R.; Mahmoudi, T.; Riazi-Esfahani, H.; Faghihi, H.; Mirshahi, A.; Ghassemi, F.; Khodabande, A.; Kafieh, R.; Khalili Pour, E. Automatic Choroid Vascularity Index Calculation in Optical Coherence Tomography Images with Low-Contrast Sclerochoroidal Junction Using Deep Learning. Photonics 2023, 10, 234. https://doi.org/10.3390/photonics10030234
