# OCT-Multitask
## Multi-Task Joint Segmentation of Retinal Layers and Fluid Lesions in Optical Coherence Tomography ##
## 1. Environment
- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare OCTA datasets 
#### (1) Run pre-processing step for OCT B-scans
- The pre-processing step contains OCT flattening and intensity normalization.
- An example of pre-processing on Duke dataset:
```bash
python terminal.py --preprocessing --dataset duke
```
- If you want to do this process on other dataset, you can set the parameter value of dataset to "yifuyuan" or "retouch". Specially, If you want to do this on Retouch dataset, you have to set a specific device type of --oct_device, including "Cirrus", "Spectralis" and "Topcon".
```bash
python terminal.py --preprocessing --dataset retouch --oct_device Cirrus
```
#### (2) Split dataset into train and test datasets 
- The clinical and Duke dataset are automatically splited into training set and test set based on the index json files in ["./datasets/yifuyuan/"](datasets/yifuyuan/) and ["./datasets/duke/"](datasets/duke/). To do these, you can type the following:
```bash
python terminal.py --split_dataset --dataset yifuyuan
```
```bash
python terminal.py --split_dataset --dataset duke
```
## 3. Train and test the proposed model on Yifuyuan and Duke dataset
- If you have done the above two steps for specific dataset, you can type the training and testing commands on the corresponding dataset given below:
#### (1) Our clinical dataset
```bash
python terminal.py --training --testing --dataset yifuyuan --backbone resnetv2 --seedlist 8830
```
#### (2) Duke dataset
```bash
python terminal.py --training --testing --dataset duke --backbone resnetv2 --seedlist 8830
```
## 4. Semi-supervised learning on Retouch dataset
- To do this, You must first get a pre-training weight file by training on a fully annotated dataset, like our clinical dataset.
