# RWEI_AECRN
The code of 6-DoF Pose Relocalization for Event Cameras With Entropy Frame and Attention Networks in VRCAI 2022

We are glad to share our code to public for further research. 

There are two main content in our code, one for RWEI generation, and another for relocalization.

## For RWEI generation
The file event2RWEI.py is the main RWEI generation method we proposed in paper 6-DoF Pose Relocalization for Event Cameras With Entropy Frame and Attention Networks in VRCAI 2022.

### Usage
1. Download the datasets from [ECD](https://rpg.ifi.uzh.ch/davis_data.html), we used the Text(zip) format data.   
2. Unzip the datasets zip file.  
4. Install the necessary python package: opencv-python and numpy    
```
pip install opencv-python numpy
```
6. Change the root path in event2RWEI.py begin.    
7. Run the event2RWEI.py in terminal.    
```
python event2RWEI.py  
```
8.Then, you will see the RWEI image and its corresponds camera pose in the data root path which you provided in the code.  

## For Attention based relocalization  
Our code mainly based on [DSAC*](https://github.com/vislearn/dsacstar).  

### Usage
1. prepare the enviroments with [DSAC*](https://github.com/vislearn/dsacstar).  
2. copy the RWEI images and poses to the datasets folders with the same rule in [DSAC*](https://github.com/vislearn/dsacstar).  
3. train the network at initial phase:  
```
python train_init_ecd.py <data_path> <output_network> --mode 0  
for example: python train_init_ecd.py ecd_shapes_6dof ecd_shapes_6dof_init.net --mode 0  
```
notice: we have not depth information, so the mode only can be 0.  

5. train the network at end2end phase:  
6. python train_e2e_ecd.py <data_path> <input_network> <output_network> --mode 1  
for example: python train_init_ecd.py ecd_shapes_6dof ecd_shapes_6dof_init.net --mode 1  

## Publications
Please cite the following paper if you use RWEI or parts of this code in your own work.
```
@inproceedings{10.1145/3574131.3574457,
author = {Lin, Hu and Li, Meng and Xia, Qianchen and Fei, Yifeng and Yin, Baocai and Yang, Xin},
title = {6-DoF Pose Relocalization for Event Cameras With Entropy Frame and Attention Networks},
year = {2023},
isbn = {9798400700316},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3574131.3574457},
doi = {10.1145/3574131.3574457},
booktitle = {Proceedings of the 18th ACM SIGGRAPH International Conference on Virtual-Reality Continuum and Its Applications in Industry},
articleno = {29},
numpages = {8},
keywords = {entropy image, event image, camera relocalization, event camera},
location = {Guangzhou, China},
series = {VRCAI '22}
}
```
The relocalization code builds on camera re-localization pipeline, namely DSAC*:

```
@article{brachmann2021dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}
```


