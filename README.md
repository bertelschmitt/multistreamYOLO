# TrainYourOwnYOLO XXL: Build a Custom Object Detector from Scratch, and run many in parallel [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repo works with TensorFlow 2.3 and Keras 2.4. This repo builds on the very successful [BuildYourOwnYOLO](https://github.com/AntonMu/TrainYourOwnYOLO) repo maintained by Anton Mu. His  repo lets you train a custom image detector using the state-of-the-art [YOLOv3](https://pjreddie.com/darknet/yolo/) computer vision algorithm. For a short write up check out this [medium post](https://medium.com/@muehle/how-to-train-your-own-yolov3-detector-from-scratch-224d10e55de2).  

This repo brings everything BuildYourOwnYOLO does, and on top, it allows you to detect objects in multiple streams, with multiple GPUs, and with multiple models, all at the same time, all in parallel in multiple independent Python processes. The number of streams depends on the amount of memory available on the GPU and in your computer. A YOLO process demands around a gigabyte of GPU memory, therefore, 11 streams can be squeezed into a Geforce 1080ti with 11 Gbytes. This is achieved with a modified YOLO object. A more [**in-depth description is here.**](https://github.com/bertelschmitt/multistreamYOLO/blob/master/MultiYOLO.md)

This repo comes with a very early version of MultiDetect.py, an application that makes use of the multi-stream, multi-GPU YOLO object. MultiDetect.py allows you to manage multiple streams and GPUs, to display the output on one or many monitors, and to automatically record video and attendant data files. An [**in-depth description of  MultiDetect.py is here**](https://github.com/bertelschmitt/multistreamYOLO/blob/master/MultiDetect.md). 

Both the modified YOLO process and MultiDetect.py are written in pure Python3.7. They integrate with TrainYourOwnYOLO, use the same models, workflows, file and directory structures. This version targets the Linux platform only. I have not yet tested it on Windows. I do not have access to a MacOS machine, please help. Let's work on it a bit, shake the bugs out, and then offer it as a merge. 

![4windows](/Utils/Screenshots/4stream.gif)

**You can create as many independent YOLO video streams as your GPU can stomach**
            

### Pipeline Overview

To build and test your YOLO object detection algorithm follow the below steps:

 1. [Image Annotation](/1_Image_Annotation/)
     - Install Microsoft's Visual Object Tagging Tool (VoTT)
     - Annotate images
 2. [Training](/2_Training/)
    - Download pre-trained weights
    - Train your custom YOLO model on annotated images 
 3. [Inference](/3_Inference/)
    - Detect objects in new images and videos
    - Detect objects in parallel in multiple streams and on multiple GPUs

## Repo structure
+ [`1_Image_Annotation`](/1_Image_Annotation/): Scripts and instructions on annotating images
+ [`2_Training`](/2_Training/): Scripts and instructions on training your YOLOv3 model
+ [`3_Inference`](/3_Inference/): Scripts and instructions on testing your trained YOLO model on new images and videos
+ [`Data`](/Data/): Input Data, Output Data, Model Weights and Results
+ [`Utils`](/Utils/): Utility scripts used by main scripts

## Getting Started

### Requisites
The only hard requirement is a running version of python 3.6 or 3.7. To install python 3.7 go to 
- [python.org/downloads](https://www.python.org/downloads/release/python-376/) 

and follow the installation instructions. Note that this repo has only been tested python 3.7 thus it is recommended to use `python3.7`.

To speed up training, it is recommended to use a **GPU with CUDA** support. For example on [AWS](/2_Training/AWS/) you can use a `p2.xlarge` instance (Tesla K80 GPU with 12GB memory). Inference is very fast even on a CPU with approximately ~2 images per second. If you want to use your own machine, follow the instructions at [tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) to install CUDA drivers. Make sure to install the [correct version of CUDA and cuDNN](https://www.tensorflow.org/install/source#linux). Note: This repo has not been tested with anything else than pure metal GPUs. 

MultiDetect.py offers you audible prompts. For that, it uses the pydub library. Pydub can't function withoout working audio either. If no audio is found, pydub will complain with a 

`
RuntimeWarning: Couldn't find ffplay or avplay - defaulting to ffplay, but may not work
`
You can safely ignore the warning, or you can install ffmpeg:

`
sudo apt install ffmpeg
`


### Installation

You have two choices. 

### Either

You can graft multistreamYOOLO upon an existing TrainYourOwnYOLO installation like so:

- Rename .../TrainYourOwnYOLO/2_Training/src/keras_yolo3/yolo.py to yolo.py.ori, and replace the file with the [new version from thios repo](https://github.com/bertelschmitt/multistreamYOLO/blob/master/2_Training/src/keras_yolo3/yolo.py) This is the modified YOLO object that does all the work. It should be a drop-in, bolt-on replacement, compatible with the current BuildYourOwnYOLO version.
- Add the complete content of [.../TrainYourOwnYOLO/3_Inference](https://github.com/bertelschmitt/multistreamYOLO/tree/master/3_Inference), including the [MDResource](https://github.com/bertelschmitt/multistreamYOLO/tree/master/3_Inference/MDResource) folder to .../TrainYourOwnYOLO/3_Inference. This brings in MultiDetect.py and a feww attendant files. MultiDetect.conf is the config file of MultiDetect.py, and it's where the magic happens. [See in-depth explanantion here]https://github.com/bertelschmitt/multistreamYOLO/blob/master/MultiDetect.md). There are a few conf file versions for multiple scenarios for you to play with. Edit to your use case and liking, and rename to MultiDetect.conf. 
- Replace your current requirements.txt with the [new requirements.txt in this repo](https://github.com/bertelschmitt/multistreamYOLO/blob/master/requirements.txt) 
- Enter your virtualenv if you use one
- Run pip install -r requirements.txt , and you are good to go.


### OR

Clone this complete repo, follow the steps below, read [MultiYOLO.md](https://github.com/bertelschmitt/multistreamYOLO/blob/master/MultiYOLO.md) (docs for the modofied YOLO object) and [MultiDetect.md](https://github.com/bertelschmitt/multistreamYOLO/blob/master/MultiDetect.md), and you are an expert.


#### Setting up Virtual Environment 

Note: This repo so far has been developed and tested on Ubuntu (20.04, and 18.04) only. 

Clone this repo with:
```
git clone https://github.com/AntonMu/TrainYourOwnYOLO
cd TrainYourOwnYOLO/
```
Create Virtual **(Linux)** Environment:
```
python3 -m venv env
source env/bin/activate
```
Make sure that, from now on, you **run all commands from within your virtual environment**.


#### Install Required Packages
Install required packages (from within your virtual environment) via:

```
pip install -r requirements.txt
```
If this fails, you may have to upgrade your pip version first with `pip install pip --upgrade`.

## Quick Start (Inference only)
To test the cat face detector on test images located in [`TrainYourOwnYOLO/Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images) run the `Minimal_Example.py` script in the root folder with:

```
python Minimal_Example.py
```

The outputs are saved in [`TrainYourOwnYOLO/Data/Source_Images/Test_Image_Detection_Results`](/Data/Source_Images/Test_Image_Detection_Results). This includes:
 - Cat pictures with bounding boxes around faces with confidence scores and
 - [`Detection_Results.csv`](/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) file with file names and locations of bounding boxes.

 If you want to detect cat faces in your own pictures, replace the cat images in [`Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images) with your own images.

## Full Start (Training and Inference)

To train your own custom YOLO object detector please follow the instructions detailed in the three numbered subfolders of this repo:
- [`1_Image_Annotation`](/1_Image_Annotation/),
- [`2_Training`](/2_Training/) and
- [`3_Inference`](/3_Inference/).

When your model(s) run, then venture forth to multiple streams, possibly even on multiple GPUs. 
 
**To make everything run smoothly it is highly recommended to keep the original folder structure of this repo!**

Each `*.py` script has various command line options that help tweak performance and change things such as input and output directories. All scripts are initialized with good default values that help accomplish all tasks as long as the original folder structure is preserved. To learn more about available command line options of a python script `<script_name.py>` run:

```
python <script_name.py> -h
```

## License

Unless explicitly stated otherwise at the top of a file, all code is licensed under the MIT license. This repo makes use of [**ilmonteux/logohunter**](https://github.com/ilmonteux/logohunter) which itself is inspired by [**qqwweee/keras-yolo3**](https://github.com/qqwweee/keras-yolo3).

## Troubleshooting

0. If you encounter any error, please make sure you follow the instructions **exactly** (word by word). Once you are familiar with the code, you're welcome to modify it as needed but in order to minimize error, I encourage you to not deviate from the instructions above. If you would like to file an issue, please use the provided template and make sure to fill out all fields. 

1. If you encounter a `FileNotFoundError`, `Module not found` or similar error, make sure that you did not change the folder structure. Your directory structure **must** look exactly like this: 
    ```
    TrainYourOwnYOLO
    └─── 1_Image_Annotation
    └─── 2_Training
    └─── 3_Inference
    └─── Data
    └─── Utils
    ```
    If you use a different name such as e.g. `TrainYourOwnYOLO-master` you will have to specify the correct paths as command line arguments in every function call.

    Don't use spaces in file or folder names, i.e. instead of `my folder` use `my_folder`.

2. If you are a Linux user and having trouble installing `*.snap` package files try:
    ```
    snap install --dangerous vott-2.1.0-linux.snap
    ```
    See [Snap Tutorial](https://tutorials.ubuntu.com/tutorial/advanced-snap-usage#2) for more information.

## Need more help? File an Issue!
If you would like to file an issue, please use the provided issue template and make sure to complete all fields. This makes it easier to reproduce the issue for someone trying to help you. 

![Issue](/Utils/Screenshots/Issue.gif)

Issues without a completed issue template will be closed after 7 days. 

## Stay Up-to-Date

- ⭐ **star** this repo to get notifications on future improvements and
- 🍴 **fork** this repo if you like to use it as part of your own project.

![CatVideo](/Utils/Screenshots/CatVideo.gif)

## Licensing 
This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by]. This means that you are free to:

 * **Share** — copy and redistribute the material in any medium or format
 * **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:

 * **Attribution** 
 
 Cite as:
 
  ```
  @misc{TrainYourOwnYOLO,
    title={TrainYourOwnYOLO: Building a Custom Object Detector from Scratch},
    author={Anton Muehlemann},
    year={2019},
    url={https://github.com/AntonMu/TrainYourOwnYOLO}
  }
  ```
If your work doesn't include a citation list, simply link this [github repo](https://github.com/AntonMu/TrainYourOwnYOLO)!
 
[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


