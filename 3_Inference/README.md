# TrainYourOwnYOLO: Inference

This directory conntains two Python scripts, Detector.py and MultiDetect.py. Detector.py allows you to quickly test a new YOLO model. MultiDetect.py lets you detect objects in many video streams at the same time, using many YOLO models in parallel, while sharing one or secveral GPUs.

## Detector.py

To detect objects, run the detector script from within the [`TrainYourOwnYOLO/3_Inference`](/3_Inference/) directory:.

```
python Detector.py
```

It allows you to test our detector on cat and dog images and videos located in [`.../Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images). If you like to test the detector on your own images or videos, place them in the [`Test_Images`](/Data/Source_Images/Test_Images) folder. The outputs are saved to [`.../Data/Source_Images/Test_Image_Detection_Results`](/Data/Source_Images/Test_Image_Detection_Results). The outputs include the original images with bounding boxes and confidence scores as well as a file called [`Detection_Results.csv`](/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) containing the image file paths and the bounding box coordinates. For videos, the output files are videos with bounding boxes and confidence scores. To list available command line options run `python Detector.py -h`.

## MultiDetect.py

This video-centric Python script allows you to detect objects in many video streams at the same time. The video sources can be files, webams, or video-streams created by IP cameras and such.  Start MultiDetect.py from the .../3_Inference directory with

```
python MultiDetect.py
```

If you use Python3.7, you can start it directly with


```
./MultiDetect.py
```

For that, you need to make MultiDetect.py executable. For Python flavors other than 3.7, change the `#!/usr/bin/env python3.7` shebang on top of the  MultiDetect.py script.

To start ./MultiDetect.py without reams of nagging status messages cluttering your terminal, set the **hush:** option to True in MultiDetect.conf, and start MultiDetect.py with this bash script (after having marked it as executable):

```
./MD
```

MultiDetect.py will start with one stream, TrainYourOwnYOLO's trademark black&white cat video. For more fun, set up multiple sources and multiple models by editing MultiDetect.conf

## For more information ... 

[**README!**](/README.md) A MultiDetect intro<br>
[**YOLO on all cylinders**](/MultiYOLO.md) The YOLO object, tuned for multiple processes<br>
[**CUDA crash course**](/CUDA101.md) Some CUDA installation pointers<br> 

- Please **star** ⭐ this repo to get notifications on future improvements and
- Please **fork** 🍴 this repo if you like to use it as part of your own project.


 
 
