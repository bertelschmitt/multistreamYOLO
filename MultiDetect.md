# MultiDetect.py

MultiDetect.py started as a quick and dirty test-bench to validate multi-stream YOLO, and it (d)evolved into a >2,000 lines monster that can show multiple YOLO streams on multiple (or one) monitor, it can record video automatically when a desired object is detected, it can even audibly alert the bored operator. 
MultiDetect.py has been written in pure python 3.7. It also works in 3.8. To run, it requires [the modified YOLO object in this repo](/2_Training/src/keras_yolo3/polo.py) and a working CUDA installation

## The Master process
The main process parses the configuration file, and launches the video processes with their chunk of the configuration settings. That done, master acts as a common switchboard to facilitate communication with and between the video_processes. This is done via queues. The master process also maintains a rudimentary GUI status window. The demise of a video process will be announced on the master panel. If all video processes are dead, the master will exit also.

## The video_process(es)
Each of the YOLO streams is handled by a completely independent python process. You can launch as many processes as you desire, and as your hardware can stomach. The video process is, not surprisingly, called video_process(). It grabs video frames from a webcam, a file, or an on-line stream. It runs the video frames through a YOLO model specified by you, it then goes on to display the video, and optionally store the results in a file.
Using the modded YOLO class ([**more on that in its description**](/MultiYOLO.md)), the video_process can use a dedicated GPU as specified in **FLAGS.run_on_gpu**, and/or it will run on a fraction of a GPU as specified in **gpu_memory_fraction**. There is no bounds checking. The process will crash if the GPU, or the total of claimed GPU memory are out of bounds. Be aware that each video_process will initialize and maintain its own copy of the YOLO class, and can require around 2.5 G of main memory each. With multiple streams, that can quickly add up. When memory gets too tight, a video process can abort. MultiDetect.py may crash ignominiously when memory-starved. The process will run forever unless stopped by the operator, or if an unrecoverable error occurs. 

![You can have as many processes as your GPU can stomach](/Utils/Screenshots/MD-arch2.png)

## Process communication
The master communicates with the processes, and the processes can communicate with each other through queues. Communication takes place via a standardized command block, a dictionary, structured as follows:

**{'To': To, 'From': From , 'Command': Command, 'Args': {‘Arg1’:arg, ‘Arg2’:arg …. }}**

**‘To”** specifies the recipient, specified as a process number. A ‘To’ larger than 0 is sent to the process specified. A ‘To’ equal to 0 is sent to all processes. A ‘To’ equal to -1 is sent to the master process only. Addresses smaller than -1 are available to be used for specialized processes (currently not implemented.)
**‘From’** specifies the sender. It is either a process number, -1 for the master process, or less than -1 for any special processes.
**'Command'** can be any command agreed upon. **‘Args’** is a dict of arguments for that command. 
For instance, a ‘startrecording’ command, sent to 0, i.e. all processes, would cause all processes to start recording their video.
New commands can be implemented in code. 

## Configuration and initialization
MultiDetect.py is configured with an extensive configuration file, **MultiDetect.conf**. If you run MultiDetect.py without the MultiDetect.conf file, it will try starting with limited defaults. **DO NOT CHANGE OPTIONS IN CODE** – it would break the logic. The code is there to set the defaults only. **CHANGE OPTIONS IN THE CONFIG FILE ONLY** 

The MultiDetect.conf file has the following sections:

**The Master: block** has settings for the main program.<br>
The most important setting is **num_processes:** MultiDetect.py will try launching the number of processes specified in this setting.<br>
**master_window_title:** title of the master monitor window.<br>
**redraw:** Set to True to redraw screen to settle windows into their place. When building a grid of multiple small windows, initial positioning of the small windows can be off on certain x-window implementations. The redraw: setting will put the screens into their intended places. Same can be done manually via the “Redraw” menu option.<br> 
**hush:** will, when True, try to suppress the annoying status messages during startup. It also may suppress non-fatal error messages. TensorFlow has the nasty habit to clutter the screen with rather useless status messages. Not only do they look messy, they also drown out real error messages. Hush will try to hush-up the chatter. It is recommended to set hush: to False during setup and testing. It can be turned on when things run smoothly.<br>
**sync_start:** will, when True cause all video streams to start at the same time to mitigate time drift between videos. Some video sources, especially IP cameras, can take a while before sending video. With **sync_start:** set, each video process will wait until all video processes are ready.<br>
**sync_start_wait:** is the time, in seconds, to wait for the start signal. This will be multiplied by num_processes to allow for longer staging due to higher load. When the wait time is exceeded without a start signal, video will start playing regardless. If a video process dies during startup, sync_start is canceled, and all surviving video processes will play.<br>

If no **Master:** block is specified, or if the settings are empty, MultiDetect.py will start with defaults.<br>

**The Common: block** has the settings for the video processes. Settings will propagate to all processes with ID > 0, i.e. all except the master and any special processes.<br>

If you stick a setting into a **Process_** block, that setting will affect this particular process only. For instance, if **gpu_memory_fraction** is set to 0.1 in the Common: block, all YOLO processes will claim 10% of the available GPU memory. However, if in the Process_3: block gpu_memory_fraction is set to 0.5, then process #3 will claim 50% of the available GPU memory, while all other processes will continue allocating 10% each. No sanity check is performed. If there is no corresponding **Process_** block, the process will use the settings in **Common:**. If there are no settings in Common:, defaults will be used.<br> 

All settings are documented in **MultiDetect.conf**. Here are a few that need more explaining.<br>

## The YOLO settings

**model_path:** should point to where you put your model, usually ".../Data/Model_Weights /trained_weights_final.h5".<br>
**classes_path:** should point to where you stored the file, usually ".../Data/Model_Weights/data_classes.txt".<br>
**anchors_path:** should point to wherever you stored the anchors, usually "…/2_Training/src/keras_yolo3/model_data/yolo_anchors.txt".<br>
(Of course, multiple models need to go into their respective directories, and the paths would need to be likewise adjusted on a per-process basis.)<br>
**run_on_gpu:** The GPU number you want the process to run on. The number is the one reported by TensorFlow and shown in the Master Window. It may be different from what nvidia-smi says.<br>
**gpu_memory_fraction:** How much GPU memory to allocate to the process. 1 = 100%, 0.1 = 10% . Process will crash if GPU memory is insufficient. When set to less than 1 (100%), the total for all processes must be less than 100% to allow for overhead. You will be able to fit more processes into a card that is not used for video output. Experiment.<br>
**allow_growth:** GPU memory allocation strategy. -1 let Keras decide, 0 disallow, 1 allow memory to dynamically grow. Best setting to optimize memory usage appears to be 1. If you really, really want to live on the edge, you can set **allow_growth:** to 1 while setting **gpu_memory_fraction:** to 1 (i.e.100%) also. In that setting, Keras/Tensorflow appear to initially grab only the barest amounts of memory necessary, and they won't (well, can't) grow it further. However, you might lose the occasional process due to memory starvation.<br> 
**score:** YOLO will report objects at and above that confidence score, it will keep anything lower to itself.<br>
ignore_labels: A list (i.e. ['aaa','bbb'] ) of object names YOLO will not report when found. Keep empty [ ] to disable this feature.<br>

**video_path:** specifies the incoming video source for that process. It can be anything understood by cv2.videocapture. It can be a video file, an URL of a video stream, or a webcam. For a webcam, set video-path to 0 (1, 2, 3 ....) no quotes. For video file, set to path to file name, in quotes. For live stream, set to the URL of the stream, in quotes.

Like all of the Common settings, you can put the YOLO settings once into the Common: block, and they will be used by all processes. If you put the settings into a Process_ block, they will be used by that process only. This way, you can use different models in different processes, and you can assign a specific GPU to a process. If a setting is the same in all Process_ blocks, there is no need to repeat it. Simply keep it once in Common: 

## Output settings

**window_wide:** and **window_high:** set the dimensions of the video output of the process.
**window_x:** and **window_y:** specify where on the screen the respective window is to be placed.
With these settings, you can put multiple video windows on one monitor. To move video output to separate monitors, first set-up multi-monitors in your display settings. Set window_wide: and window_high: to match the resolution of your separate monitors. Then set window_x: in Process_1: large enough so that the output window gets pushed to the separate monitor. Set window_y: in Process_1: to 0. Repeat for a second separate monitor as needed. 

## Alerts and recordings

**soundalert:** will, when True, alert you to the presence of a member of an object family specified in presence_trigger: If soundalert: is set to True, MultiDetect.py will probe for a valid sound output, and it will turn itself off when none is found. Due to the wild and woolly world of Linux audio, the probing is rather messy, and it can take time. Set **soundalert:** to False if your machine has no sound, or if you don’t want to hear any. 

The object families are created in **labeldict:** If, for instance, labeldict is equal to { 'Lily':'cat', 'Bella':'cat','Chloe':'cat','Tweety':'bird'}, and if presence_trigger: is set to ‘cat,’ an audible sound will be played when Lily, Bella, or Chloe are detected. The bird Tweety will be ignored. 

The same logic is used for **record_autovideo:** When True, video is recorded if a member of an object family specified in **presence_trigger:** is detected. If set as above, the appearance of Lily, Bella, and Chloe will be recorded. The video files will be stored in subdirectories of **output_path:**

**Record_framecode:** causes the creation of special framecode files. The framecode files are supplemental to their respective video files. For each frame where an object is detected, a line in the framecode file is created. Recorded will be the dimensions of the bounding box(es), the name of the detected object, the confidence, and the frame number. If more stats are needed, they can easily be added in code. The framecode file can be helpful for statistics, or for feedback training. 

For more in-depth studies, a result log can be kept by setting **maintain_result_log:** to True. The result logs will be stored in subdirectories off **result_log_basedir:** The result log will document the result of each call to detect_image_extended(), along with the round-trip time of each call. The file is in CSV format and can be opened in Excel for further studies. 

## The On-Screen-Display
An on-screen-display (OSD) can optionally be put on the outgoing video. The display is controlled by the **OSD:** setting in the config file. There are further settings to control the font of the display. 
The OSD shows:
**IFPS:** Incoming FPS. This is a best effort number, dependent on the willingness of the incoming stream to reveal that number. It can range from real-time update when the stream supports CAP_PROP_FPS, to a crude, one-time measurement at startup when the stream does not expose the frame rate, or, worse, when it gives bogus data. **YFPS** is the current max frame rate supported by YOLO. It is a rolling average over **rolling_average_n:** frames. **OFPS** is the outgoing frame rate. **RT** is the rolling average round-trip time of a video frame processed by YOLO.


## The Master Window

The Master Window is a rudimentary status and control window. It is driven by the master process. It allows you to Quit and Restart the app, to start and stop recording of video and to turn on/off audible prompts. Most of all, it gives you a picture of the frame rates achieved by each video_process. You can run MultiDetect.py without the master window by setting **showmaster:** to False in the **Master:** section of ther config file.

The window will list the GPU(s) usable by YOLO via TensorFlow and CUDA. Note: The GPU number is as reported by TensorFlow, it sometimes is different from what nvidia-smi says. 

The Master Window will, in the "Process" tab, give you a per-process status report, report any errors in the "Error" tab, and give you running stats per process in the "Stats" tab. If your machine is low on resources, constant updates of the stats can be a performance hit. Stats are off be default.

Stats can help in adjusting the proper frame rates. Let stats run for a little while, the numbers are running averages and need a little time to settle. Current frames and seconds are displayed as reported by the streams. Keep in mind that IP cams often report bogus data. **IFPS** is the incoming frame rate, again as reported. It depends on what the video source claims it is. 

**The strategy to determine incoming fps is as follows:**

If the stream exposes **PROP_FPS**, and if the PROP_FPS reported look somewhat believable ( 0 < PROP_FPS < 200 – adjust the code for a super high speed camera), we take that number. The incoming FPS are updated continuously to reflect any changes during the run.
If the above fails, we will try **PROP_POS_FRAMES / PROP_POS_MSEC * 1000** , and if the result passes the sanity check as above, we will take it. The incoming FPS are updated continuously, HOWEVER, as the total of frames since start is divided by the total of seconds since start, the fps will be a cumulative average.
If all else fails, we will measure the incoming fps in a **timing loop**. We do this only once after the stream starts.

A --- denotes a missing, or bogus frame rate. **YFPS** is the frame rate the respective YOLO stream currently can handle. **OFPS** is the outgoing frame rate. **N** tells you that every nth frame is being processed. **EFPS** is the effective frame rate of what is sent to YOLO, i.e. IFPS divided by n.

Let’s say you process two streams, and you see a YFPS of 14, a common number for a 1080ti. To avoid video drift, the number of frames (EFPS) sent to YOLO should not exceed its YFPS, actually, it should be a little lower to allow for overhead. Be aware that YFPS is the average round-trip speed of your video sent through YOLO. The calling video_process needs time also, as a matter of fact, a lot of time is spent in waiting for the next video frame. (If you want to investigate this a bit further, set **profile:** to true. With the help of cProfile, a timing profile of the video_process() will be built for 3000 frames, and shown for each process, so you can see where all the time is spent.)

If EFPS > YFPS, either lower the frame rate at the source, or adjust the do_only_every settings. Say you incoming frame rate is 16, and do_only_every_initial is set to 4. This will result in an EFPS of 4, a number that will be easily handled. These settings become important as you add multiple streams to one GPU. You will see the per-process YFPS go down, because the power of the GPU is shared by multiple processes. If OFPS sinks below IFPS, you will experience frame drift in short order. Reduce the incoming frame rate, and/or adjust the do_only_every settings.

When recording video, outgoing OFPS should be the same as incoming IFPS. If the incoming stream is a webstream, or produced by a webcam, the outgoing FPS cannot exceed incoming FPS. If the source is a file, the file will be consumed rapidly, and thus the outgoing rate can be much higher the the rated fps. So if the incoming source is a file, MultiDetect.py will adjust **cv2.waitkey** to approximate the proper outgoing rate. This is an iterative process, made difficult by the fact that YFPS constantly fluctuates. It may take 30 seconds or so until something close to an equilibrium is reached. 
 
As a default, the rolling YFPS average is calculated for the last 32 frames. You can adjust this number with the rolling_average_n setting in MultiDetect.conf

The master window also will show the total of frames and seconds of each video stream and any differences between the streams and stream #1. This is based on what the video streams report via cv2.videocapture. These properties can be very unreliable, especially between IP cameras of different brands. As long as the actual video streams are halfway in sync, do not be alarmed if you see the frame and second differences pile up.



## The Menu 

**Stop** Menu": **“Quit”** will quit. **“Restart”** will restart MultiDetect.py. 
**Record** Menu: **“Record on”** will cause all video_processes to record their video stream. **“Record off”** will stop recording. "AutoRec on" enables automatic recording, "AutoRec off" disables it.
**Misc Menu:** **Ding On** enables chimes that alert you to the presence of objects, **Ding Off** disables. **OSD on** enables on-screen display, **OSD off** disables. **Hush on** enables hush, **Hush off** disables. **Stats on** enables running stats, **Stats off** disables.


## Hands-on MultiDetect.conf

MultiDetect.conf may have a myriad of options, but you will use only a few because most will work with their defaults, and you can use the few sparingly. Remember: If it’s the same setting for all the video_processes, leave the setting in **Common:** <br>
Put into **Process_1, 2,3 etc*** only what is special for the respective process. A setting in **Common:** is a catch-all for each video_process.

Here are a few scenarios, along with their respective skeleton MultiDetect.conf files. To use a .conf file, rename tio to MultiDetect.conf, insert your paths to video sources and models, and run MultiDetect.py

### One video source, one model, one video output: 
Everything goes into, and stays in **Common:** Your **video_path:** that points to your video source file, the paths to your model, the size and coordinates of your output widow, everything goes into **Common:**. Done. [A skeleton conf file is in MultiDetect.conf.111](/3_Inference/MultiDetect.conf.111)

### Four video sources, one model, four video outputs: 
Put the paths to your model into **Common:** Also in **Common:** set the proper **gpu_memory_fraction** for your GPU. You want at least 1Gbyte of video RAM for each process. Define, in **Common:** again, the size of the output windows by dividing your screen into same-sized quadrants. Now in each **Process_** section, define the video source, and the window_x: and window_y: coordinates of each output window. Done. [A skeleton conf file is in MultiDetect.conf.411](/3_Inference/MultiDetect.conf.411)

### Four video sources, four models, four video outputs: 
This is pretty much the same as the previous, except that now the paths to the different models go into their respective **Process_** sections. I’ve painted you a picture:




![4-up](/Utils/Screenshots/1920screen_4up.png)
[A skeleton conf file is in MultiDetect.conf.444](/3_Inference/MultiDetect.conf.444)



### Nine video sources, nine models, nine video outputs: 
Similar to the preceding, except that by now, we need to give serious consideration to the memory size of our GPU. **Careful, at these settings, you are approaching the ragged edge. Run MultiDetect.py without the hush setting, and watch for out of memory errors, and/or processes dying.** In a production setting, it would be advisable to back-off from the ragged edge, run fewer processes on one GPU, or get a GPU with more memory. The nine processes fit into an 11 Gbyte GPU, but they would be too much for a 6 Gbyte GPU. For that, a second GPU would be needed. How to use two GPUs will follow. Again, a picture will explain our 9/9/9 setup much better.



![9-up](/Utils/Screenshots/1920screen_9up.png)
[A skeleton conf file is in MultiDetect.conf.999](/3_Inference/MultiDetect.conf.999)



### 18 video sources, 18 models, 18 video outputs: 
For that, we will definitely need two GPUs, and we will need to put a **run_on_gpu:** setting into each **Process_** section. Half of the processes will **run_on_gpu: 0**, the other half will **run_on_gpu: 1** We also need a 2nd monitor. How do we move the output windows to that 2nd monitor? Simple, we add the width of the monitor, in pixels, to the **window_x:** of the window we want to show on the 2nd monitor. If our monitors are 1920 pixels wide, and if the output window of **process_1** is at **window_x: 0** and **window_y: 28**, then **process_10** would live at **window_x: 1920** and **window_y: 28**, and so forth . Why **window_y: 28 ???** Because your screen may go crazy with **window_y:** set to 0. See below under **Flicker!!** 

Here is, in Super Todd-AO, the picture of 18/18/18. You may have to zoom in ... For anything above 18, use your imagination. 



![18-up](/Utils/Screenshots/1920screen_18up.png)
[A skeleton conf file is in MultiDetect.conf.181818](/3_Inference/MultiDetect.conf.181818)



## Fighting frame drift

Frame drift will be an ever-present problem when processing multiple real-time streams. Most IP cameras lack common timecode, have an unreliable timebase, or report unreliable or plain false fps etc data. MultiDetect.py will attempt every mitigation possible at its end, but it can be a losing battle. Here are a few steps to keep frame drift in check as much as possible. 

**buffer_depth:** sets the size of the frame buffer in frames. If the buffer is set too high, frames will simply pile up in the buffer. Keep the buffer low to combat drift, and high enough for smooth video. A buffer of around 20 should be fine. Note: Some webcams do not allow buffers > 10.

IP cameras don’t always send video immediately, some take considerably longer than others. This is exacerbated when a number of IP cameras is started up at the same time by multiple processes. MultiDetect.py addresses this with **sync_start:** When set to True, the respective video_process will initialize its video source, and it will signal readiness to the master process while holding the video. Once the master process has received a ‘videoready’ signal from all processes, it will send a sync_start signal to all processes, and they will all start processing video at the same time, and in a semblance of sync. If no videoready signal has been received from all processes (probably because one crashed) the surviving processes will start playing after a timeout. The timeout is sync_start_wait:, internally multiplied by num_processes: 

Even the most capable GPU can easily be overwhelmed when subjected to higher frame rates than it can process, even more so when multiple frames are being handled. This will result in video lag, and quickly, the timing of the individual streams will drift apart. To help mitigate video drift, you should send fewer frames to YOLO than a single process can handle. The first step would be to lower the incoming frame rate at the source. For further mitigation, MultiDetect.py has three settings that allow only a subset of the video frames to be run through YOLO, while all frames are displayed.

**do_only_every_initial:** causes only every nth image to be run through YOLO. If set to 4, every 4th frame would be run through YOLO. With a 24fps video, YOLO would only be subjected to 6fps

**do_only_every_autovideo:** runs only every nth image through YOLO during autovideo recording 

**do_only_every_present:** runs every nth image through YOLO after activity was detected and until presence_timer_interval times out

A judicious use of these settings will also result in considerable power savings. 



## Flicker?

If your screen flickers during full screen playback, or if your carefully calculated layout of multiple windows on one screen goes haywire, don’t despair: The new version of CV2 appears to have a, well, feature that makes the window larger than specified. Let’s say you create a window, and size it to 1024x600 to fit a small 1024x600 monitor. You run the app, and the window flickers madly. Why? The created window is actually higher than 600 pixel, and your monitor doesn't like it. To solve the problem, we must size the window smaller, in that case 1024x570. Apparently, CV2 does not include the black bar on top of the window. Likewise, if we place 4 windows on a monitor with the dimensions of 1920 x 1080, one would think that each window should be 960 wide by 540 high, right? Wrong. What works is something like 960x490, or even a little shorter to account for the bar on top of the main monitor screen, and for the bars on top of each ouput window. On my screen, the top menu bar is 28 pixels high, while the bar over each output window is 37 pix high, no matter how high the rest of the window if. YMMV, experiment!


## Bugs!

MultiDetect.py is a very early version, and it is full of bugs. I find new ones every day. This version is on Github, because I’m under pressure to release something. It also is on Github, because we are a community of programmers. If you find a bug, please don’t just report it. Try to find out why it fails. Much, if not most of the code can be coded better. If you know a better, faster, more elegant way, please let us know. Here are some bugs I know of:

- In a multi-monitor situation, and if the Master Window is moved to another monitor, an orphan drop-down menu sometimes is left on the other monitor. Tkinter issue. Cosmetic only.
- If a video process crashes in the YOLO object/Tensorflow/Keras, the calling procedure never gets notified. We are trying to get around a never-to-return call to init_YOLO by setting a timer which triggers and abort when the call won't return. This is currently set to 20 seconds. It can be changed by editing **YOLO_AWOL_wait:** in the config file.
MultiDetect.py will eventually try to kill a crashed/hung process, but it can take a while, and it may not happen. 



## General comments

Throughput doesn’t seem very sensitive to the amount of GPU memory allocated to a video_process. I have reduced the memory allocation of a single process to just 10% of the available memory of a 11G 1080ti, and YOLO still ran at 18fps. When multiple processes share one GPU, the frame rate of each process of course will sink. However, the sum of the frame rates of all processes occasionally is higher than a single process running at full speed. Variations between runs are quite common. Frame rates often are higher after a hard reset. I noticed that occasionally, and very counter-intuitively, a higher **do_only_every_** setting can result in lower YFPS. I have no idea why. Also quite mysteriously, YOLO occasionally delivered a higher frame rate when objects were detected, and it would report lower YFPS when there was nothing to see. This could be caused by dynamic frequency scaling of a bored CPU, but it’s just a guess.

## A word on IP cameras 
The market is flooded with cheap IP cameras. Their picture quality can be quite decent these days, their software quality often is lousy. You will often find them contacting servers in China. If you don’t want to star on insecam.org, the infamous database of live IP cameras, do the following: Avoid WiFi cams, use hardwired. Put the cams behind a firewall, making sure that the cameras can’t be reached from the outside, AND MOST OF ALL make sure that the cameras cannot reach the outside. This also keeps the cams from updating their internal clock via NTP. For that, set up your own local NTP server that acts as a common reference for your cams.

## Infamous last words
Development was on Ubuntu 18.04 and 20.04, with python 3.7, CUDA 10.1, and the Nvidia 450 video driver. CUDA 10.1 appears to get along best with the Tensorflow version used in this repo. I have developed and tested MultiDetect.py on a machine with a 3970x Threadripper and 128G of memory, and on an ancient Intel 6700K with 64G of memory. I have a stack of Geforce 1060/6G and Geforce 1080ti/11G GPUs, and I used them in various combinations. No other systems were tested. MultiDetect.py makes use of certain Unix functions and would have to be adapted to Windows, and possibly Mac.

I am a retired advertising executive, and my programming “skills” are completely self-taught. I tried my hands on assembler and BASIC half a century ago when computers had 8 bits and 4K of memory. I took up Python to keep me busy after retirement. My code definitely is in need of improvement, and it could be completely flawed. The stuff is on Github in hope for improvement – of the code, and of myself. 

Have at it. 

