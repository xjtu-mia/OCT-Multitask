## Data Preparing
1. Download the OCT datasets from the [Baidu Netdisk](https://pan.baidu.com/s/1SzuMVWN1AteufH0JIj6lhw?pwd=ra2z) (access code: ra2z) and decompress it to this directory.
2. Description of the OCTA dataset:
* The dataset in folder **yifuyuan** was collected at the First Affiliated Hospital of Xi’an Jiaotong University, consisting of 140 OCT B-scans from 50 AMD subjects. All OCT B-scans were acquired with Cirrus HD 4000 and resized to 1024×1024 pixels (axial resolution of 3.92μm). Among these images, SRF are shown in 124 images, PED in 93 images, and IRF in 28 images. 7 retinal layer-surfaces (i.e., ILM, IPL-INL, INL-OPL, outer surface of neurosensory retina (ONR), inner surface of retinal pigment epithelium (IRPE), outer surface of retinal pigment epithelium (ORPE), and BM) and the contour of IRF were manually annotated at pixel level by three experts. **(This dataset is temporarily not publicly available)**
* The Duke dataset contains 110 OCT B-scans with pixel-level annotation from 10 diabetes macular edema (DME) patients (11 OCT scans per patient). OCT B-scans were acquired with Spectralis SD-OCT device with 512×728 pixels and axial resolutions of 3.87μm. The annotations were done by two experts, including IRF and eight retinal layer-surfaces (i.e., ILM, NFL-GCL, IPL-INL, INL-OPL, OPL-ONL, ISM-ISE, OS-RPE, and BM).
* The RETOUCH dataset consisted of 112 OCT volumes (11334 B-scans) from 112 AMD and RVO patients, 70 of which are training set and the rest are testing set. A portion of B-scans in these volumes are annotated with three types of fluid, including SRF, PED, and IRF. The training set contains 24, 24 and 22 OCT volumes acquired with Cirrus, Spectralis and Topcon scanners, respectively. The size and axial resolution of B-scans in OCT volumes are as follows: (512×1024 pixels, 1.96μm) for Cirrus, (512×496 pixels, 3.87μm) for Spectralis and (512×885/512×650 pixels, 2.60/3.50μm) for Topcon T2000/T1000. In RETOUCH, only the annotations on training set are available to the public, thus we select the annotated B-scans from each volume in the training set for training and validating the model, of which 1568 are from Cirrus, 711 are from Spectralis and 1106 are from Topcon.
3. Folder hierarchy
* Each dataset folder has two subfolder to save original images and original labels. For Yifuyuan and Duke datasets, there are several subfolders in the folder of Original_label to save multiple types of labels, their correspondence is as follows: covered (region label of layers and fluids), covered_wo_irf (region label of layers and fluids (without IRF)), fluid/fluid3 (label of IRF), fluid2 (label of PED), fluid1 (label of SRF), layers (surface label of layers).

```bash
  .
  ├── yifuyuan
  |     ├── Original_image
  |     |     └── *.png
  |     └── Original_label
  |           ├── covered
  |           |     └── *.png
  |           ├── covered_wo_irf
  |           |     └── *.png
  |           ├── fluid1
  |           |     └── *.png
  |           ├── fluid2
  |           |     └── *.png
  |           ├── fluid3
  |           |     └── *.png
  |           └── layers
  |                 └── layer*
  |                       └── *.png
  ├── duke        
  |     ├── Original_image
  |     |      └── *.png
  |     └── Original_label
  |           ├── covered
  |           |     └── *.png
  |           ├── covered_wo_irf
  |           |     └── *.png
  |           ├── fluid
  |           |     └── *.png
  |           └── layers
  |                 └── layer*
  |                       └── *.png    
  └── retouch
        ├── Cirrus
        |     ├── Original_image   
        |     |      └── *.png
        |     └── Original_label
        |            └── *.png
        ├── Spectralis
        |
        └──Topcon
```
