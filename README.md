# MRCNN - greeny
## Introduction
Greeny is an application which uses [Mask R-CNN TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2) the Tensorflow 2 version of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) originally meant for object detection and scene segmentation, to add a green screen to the background of people in any video. It supports multiple people and creates an exact cut-out of the people present in the video.

## Here are the demo images:
|**Original Image 1**|**New Image 1**|
| :--: | :--: |
|![](demo/1.png)|![](demo/1_g.png)|
|**Original Image 2**|**New Image 2**|
| :--: | :--: |
|![](demo/2.png)|![](demo/2_g.png)|

## Here are the demo gifs:

|**Demo 1**|**Demo 1 Greened**|
| :--: | :--: |
|![](demo/1.gif)|![](demo/1_g.gif)|


# Compatibility 
1. This project requires python version to be 3.8 >= (python versions) >=3.5.
2. This project requires a gpu for faster process times.
3. This project will not work with tensorflow 1.


# Installation steps
Greeny doesn't require installation. 
1. You can download dependencies as follows
using package manager [pip](https://pip.pypa.io/en/stable/).
```bash
pip install -r requirements.txt
```


2. Download the pre-trained weights inside the root directory `mrcnn-greeny`. The weights can be downloaded from [this link](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5): https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 
3. Run the script `run.py`

```bash
python3 run.py
```
4. Enter video file name 

```python3
Enter video filename: video.mp4
```

The directory tree of the project is as follows:
```
mrcnn:
   helpers.py
   requirements.txt
   run.py
   Google_Colab_notebook.ipynb
```
# Run online
<a href="https://colab.research.google.com/github/divij-pawar/mrcnn-greeny/blob/main/Google_Colab_notebook.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a> 

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# References
[Mask R-CNN](https://github.com/matterport/Mask_RCNN) <br>
[Mask R-CNN TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2) <br>
[Mask-RCNN Shiny](https://github.com/huuuuusy/Mask-RCNN-Shiny) 
