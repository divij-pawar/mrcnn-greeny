# MRCNN - greeny

Greeny is an application which uses [Mask R-CNN TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2) the Tensorflow2 version of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) to add green screen to the background of people in an mp4 video.It supports multiple people and creates an exact cut-out of the people present in the video.

# Compatibility 
1. This project requires python version to be 3.8 >= (python versions) >=3.5.
2. This project requires a gpu for faster process times.
3. This project will not work with tensorflow 1.
4. It requires the input video to be mp4.


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
# Run without downloading
<a href="https://github.com/divij-pawar/mrcnn-greeny/blob/main/Google_Colab_notebook.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a> 

