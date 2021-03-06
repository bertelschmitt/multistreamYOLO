# YOLO on all cylinders - The YOLO object, tuned for multiple processes

**So you have successfully mastered TrainYourOwnYOLO, and it happily processes any video stream you can throw at it. But what about two, or more streams at the same time? No matter how big and powerful your GPU is, no matter how many GPUs you stick into your computer, YOLO just won’t let you process two, or more streams at once. 
Until now, that is. With this tweak to the popular TrainYourOwnYOLO, and with a halfway powerful GPU, you will be able to process video streams in the double digits.** 

With this bored-out YOLO object, your only limit will be memory (both GPU and system, more of the latter below). The more memory on your GPU, the more streams it can accommodate. I am waiting for the RTX 3090 with 24 G of memory to become available to mere mortals, and it should be good for ~25 streams all at the same time on one GPU. 

## The problem.
What kept us from processing two or more streams at once is that the YOLO object allocates all available GPU memory on initialization. If there are more CUDA-enabled GPUs in the machine, the most powerful GPU will be grabbed, and all others will be ignored. Try initializing a second YOLO instance, and one will get `Unexpected CUDA error: out of memory`, never mind that there could be a second, completely idle GPU in your machine. 
There is a **gpu_num** flag in init_yolo, but it won’t let us assign another GPU. It purports to use multiple GPUs for inference. My tests showed that in the best case, that flag will get us a couple of fps more, in the worst case, YOLO will slow down. I could not find anything in the code that allowed for separate YOLO instances, whether it’s on the same, or on multiple GPUs in one machine. 

![2windows](/Utils/Screenshots/catwide.jpg)
**18 streams with 2 GPUs with room to spare**

## The solution.
Following init_yolo() down the dark rabbit-hole of keras_yolo, and digging around its scant documentation, I developed an idea of what happens when you initialize the session. Left alone, the session indeed grabs all of the memory of the most powerful GPU in the machine, whether you need it, or not. However, there are other options YOLO currently does not use. 

**config.gpu_options.per_process_gpu_memory_fraction** is the most important option of all. It allows to allocate just a fraction of the GPU memory. Set it to 0.5, and only half of the GPU memory will be used. (Close, but not quite, as we will see below.) The remainder of the memory will be available for subsequent sessions. [More on memory settings is here](/Memory_settings.md).

**config.gpu_options.visible_device_list** allows to (hooray) select the GPU the session will use (assuming that you have more than one GPU.) Set to “0,” the first GPU will be used. Set to “1,” the second will be used, and so forth. “0,1” supposedly allows both GPUs to be used at the same time. When I tried that setting, I received an “Unexpected CUDA error: out of memory” when the second process was fired-up, and I left that alone. A more knowledgeable and courageous soul possible will find out more.

**config.gpu_options.allow_growth** allows you to use the memory on your GPU more sensibly: Set to False, the session will grab all of the allacate GPU’s available memory (or rather the assigned gpu_memory_fraction thereof) whether it is needed, or not. Set to True, the session will claim only the memory it needs at initialization, and it will “grow” more memory if it needs it, as long as there is memory available. To find out the maximum memory amount one YOLO process will be happy with, run a single process with the allow_growth flag set to True. Watch GPU memory for a while. If it doesn’t grow, then that’s all the GPU memory the process will ever need, more would simply go to waste. Under TensorFlow 1.X, a single stream with per_process_gpu_memory_fraction set to 1, and with allow_growth enabled, claimed 8,107 MiB of the 11,178 MiB available on a 1080ti. With TensorFlow 2.3, the footprint grew to 8,575 MiB (YMMV). I usually keep allow_growth set to true.

With these options in place, multiple Python processes can claim their own YOLO instance, and run in parallel. They can use the same model, or different models. Of course, throughput per process will drop as you add more processes, after all, you are sharing GPU power. You can get around this problem by sending only every n-th frame to YOLO while displaying all. Depending on your use case, it probably won’t make much difference whether you try detecting an object twenty times, or four times a second.

## Improvements. 

While we are at it, let’s implement a few improvements.

With the **hush** flag set, most of the annoying notices cluttering your monitor during init_yolo will be silenced.

**Ignore_labels** allows you to specify a list of objects that will be ignored, and not reported by YOLO if detected. Use it to ignore objects you are no longer interested in. If you have problematic parts of your video stream that get mis-identified, you can train them as “NULL” or somesuch. Ignore_labels=[“NULL”] will make it disappear. Ignore_labels takes a list, so you can shun multiple objects.

The strategically important routine is **detect_image()**. We feed it an image, and if a match (or matches) are found, the coordinates of the object and their confidence, along with an index into an object list are returned in out_prediction, the image, adorned with bounding boxes, are returned in image. You can detect objects in stills in parallel, or in video streams. After all, video streams are nothing but series of many stills.

What if we want the actual name of the detected object, along with the time spent on the detection? **detect_image_extended()** will provide that additional info, returning the (possibly annotated) image, time-spent, and out_prediction_ext, which is a list of list, containing, for each object dectected [left, top, right, bottom, predicted_class, score]. **detect_image()** is kept for backward compatibility.

(Tangent: detect_image spends quite some time drawing boxes, which made the original author leave an exasperated comment: “My kingdom for a good redistributable image drawing library.” OpenCV has a snappy box draw routine, and it is generally faster than the Pillow library used in detect_image. To shave off a few cycles, OpenCV could be used. If all you are interested in are the coordinates, and the detected object, the image drawing could be (optionally) avoided altogether. Room for improvement ….)


## Your options.

Here are all options, old and new, that are built into the YOLO object. You don't have to use them all, there are defaults ...

**Familiar settings:**

**model_path** :  Path to the model

**anchors_path** :  Path to the anchors

**classes_path** :  Path to the classes

**score** :     Don't report an object if less confidence

**model_image_size** :  The image size used by the model

**New settings:**

**hush** :    Flag to quiet YOOLO down

**iou** :     Intersection over union, avoid multiple bounding boxes for same object

**run_on_gpu** :  Which GPU to run on

**gpu_memory_fraction** : How much GPU memory to claim

**allow_growth** :  Dynamic memory allocation, or not

**ignore_labels** :   List of objects to ignore


## Other than GPU factors.
Each separate process creates its own YOLO object along with a completely separate model. This can translate into a healthy chunk of systems memory. On my system, TOP reports the resident memory footprint of one YOLO process as 2.8 Gigabytes. 10 streams amount to 28 Gigabytes, and many machines don’t have that much. While the GPU and memory play the leading role, the CPU is not that much of a factor. An aging 4core Intel 6700K, taxed with 9 processes run in parallel on a 1080 ti, would deliver around 4 fps per process, while the load average zoomed to 13. A beefy 32core monster, the Threadripper 3970x, also delivered 4 fps per each of the 9 processes, but with a load average of 3.8, it barely broke a sweat. 

## Needless to say, but said anyway: 
All the settings used in init_yolo are on a per-session basis. From session to session, settings can be completely different, or mostly all the same. 

## About detect_video() and detect_webcam()

I suggest staying away from detect_video() and detect_webcam().  Both functions were brought in unchanged from TrainYourOwnYOLO, and are here for compatibility. In my opinion, they should be application-level functions. At its core, detecting video is nothing else than detecting an image, 24 (or whatever) times a second, and you should use detect_image_extended() for that.  As is, detect_webcam() will break if the cam is not in  USB(0), and often, it is not. Its address might even have changed when re-plugged. As is, detect_video() will give alarming results if vid.get(cv2.CAP_PROP_FPS) produces nothing, or worse, produces FPS in the thousands, as it is the case with many Chinese IP cams. IMHO, the place for detect_webcam() and detect_video() are code snippets that show the best use of detect_image().  The convoluted video_process in [**MultiDetect.py**](/3_Inference/MultiDetect.py) is testament to what happens when you try to handle real-world situations.

## To Do

-	Each YOLO instance is completely separate, resulting in a massive duplication of memory usage, even if the same model (or mostly the same) is used in the processes. At up to 3 gigabyte per process, it can add up. Investigate where shared memory can be used.
-	Send error messages back to the calling process for graceful error handling.
-	Investigate using CV2 instead of PIL for image processing, the drawing of boxes takes too much time. Dispense with the box drawing if only the coordinates are needed

## Infamous last words.
Development was on Ubuntu 18.04, with Python3.7. Except for a successful run on python 3.8, no other systems were tested. 
My programming skills are completely self-taught. I tried my hands on assembler and BASIC half a century ago, and I took up Python to keep me busy after retirement. My code definitely is in need of improvement, and it could be completely flawed. Have at it. 

## For more information ... 

[**README!**](/README.md) A MultiDetect intro<br>
[**CUDA crash course**](/CUDA101.md) Some CUDA installation pointers<br> 
[**How much, how little memory**](/Memory_settings.md). The best memory settings<br>
[**Running inference**](/3_Inference/README.md) Putting it to use<br>

- Please **star** ⭐ this repo to get notifications on future improvements and
- Please **fork** 🍴 this repo if you like to use it as part of your own project.

