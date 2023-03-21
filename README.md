# OCT-Multitask
Multi-Task Joint Segmentation of Retinal Layers and Fluid Lesions in Optical Coherence Tomography

## 1. Environment
- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare OCTA datasets 
### 1) Run pre-processing step for OCT B-scans
- The pre-processing step contains OCT flattening and intensity normalization.
- An example of pre-processing on Duke dataset:
```bash
python terminal.py --preprocessing --dataset duke
```
- If you want to do this process on other dataset, you can set the parameter value of dataset to "yifuyuan" or "retouch". Specially, If you want to do this on Retouch dataset, you have to set a specific device type of --oct_device, including "Cirrus", "Spectralis" and "Topcon".
### 2) Split dataset to train and test datasets 
## 3. Train and test the proposed model on differient datasets
### 1) Yifuyuan dataset
### 2) Duke dataset
### 3) Retouch dataset
