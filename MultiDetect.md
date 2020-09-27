# MultiDetect.py

MultiDetect.py started as a quick and dirty testbench to validate multi-stream YOLO, and it (d)evolved into a 2,000+ lines monster that can show multiple YOLO streams on multiple (or one) monitor, it can record video automatically when a desired object is detected, it can even audibly alert the bored operator.

## The video_process(es)
Each of the YOLO streams is handled by a completely independent Python process. You can launch as many processes as you desire, and as your hardware can stomach. The video process is, not surprisingly, called video_process(). It grabs video frames from a webcam, file, or on-line stream, it runs the video frames through a YOLO model specified by you, it then goes on to display, and optionally store the results in a video file.
Using the modded YOLO class ([**more on that in its description**](https://github.com/bertelschmitt/multistreamYOLO/blob/master/MultiYOLO.md)), the video_process can use a dedicated GPU as specified in **FLAGS.run_on_gpu**, and/or it will run on a fraction of a GPU as specified in **gpu_memory_fraction**. There is no bounds checking. The process will crash if the GPU, or the total of claimed GPU memory are out of bounds. Be aware that each video_process will initialize and maintain its own copy of the YOLO class, and can require around 2.5 G of main memory each. With multiple streams, it can quickly add up. MultiDetect.py may crash ignominiously when memory-starved. The process will run forever unless stopped by the operator, or if an unrecoverable error occurs. 

![You can have as many processes as your GPU xan stomach](/Utils/Screenshots/MD-arch.png)

## The master process
The master process is a central hub that acts as a common switchboard to facilitate communication with and between the video_processes. The master process also maintains a rudimentary GUI status window. 

##Process communication
The master communicates with the processes, and the processes can communicate with each other through queues. There is a pair of queues (in and out) between the master and each video_process. Communication is via a standardized command block, which is a dictionary, structured as follows:

**{'To': To, 'From': From , 'Command': Command, 'Args': {‘Arg1’:arg, ‘Arg2’:arg …. }}**

**‘To”** specifies the recipient, specified as a process number. A ‘To’ larger than 0 is sent to the process specified. A ‘To’ equal to 0 is sent to all processes. A ‘To’ equal to -1 is sent to the master process only. Addresses smaller than -1 can be used for specialized processes (currently not implemented.)
**‘From’** specifies the sender. It is either a process number, -1 for the master process, or less than -1 for a special process.
**'Command'** can be any command agreed upon. **‘Args’** is a dict of arguments for that command. 
For instance, a ‘startrecording’ command, sent to 0, i.e. all processes, would cause all processes to start recording their video.
New commands can be implemented in code. 

## Configuration and initialization
**DO NOT CHANGE OPTIONS IN CODE** – it would break the logic. MultiDetect.py is configured with an extensive configuration file, MultiDetect.conf. If you run MultiDetect.py without the MultiDetect.conf file, it will try starting with limited defaults 

The MultiDetect.conf file has the following sections:

**The Startup: block** has settings for the main program. It currently has only one setting, namely num_processes: MultiDetect.py will try launching the number of processes specified in this setting.

**The Master: block** has settings for the master process, i.e.
**master_window_title:**  title of the master monitor window
**redraw:**  Set to True to redraw screen to settle windows into their place. When building a grid of multiple small windows, initial positioning of the small windows can be off on certain x-window implementations. The redraw: setting will put the screens into their intended places.
Same can be done manually via the “Redraw” button in the master window.

**The Common: block** has the settings for the video processes. Settings that will propagate to all processes with ID > 0, i.e. all except the master and any special processes.
If you stick a setting into a **Process_** block, then the setting will be for this particular process only. It overrides the same setting in Common: For instance, if **gpu_memory_fraction** is set to 0.1 in the Common: block, all YOLO processes will claim 10% of the available GPU memory. However, if in the Process_3: block gpu_memory_fraction is set to 0.5, then process #3 will claim 50% of the available GPU memory, while all other processes will continue allocating 10% each. No sanity check is performed.  
If there are more **num_processes:** than per process settings, these processes will use the settings in **Common:**. If there are no settings in Common:, the process will use the settings of the last good process. If all fails, the process will attempt falling back to defaults.

All settings are documented in MultiDetect.conf. Here are a few that need more explaining.

**soundalert:** will, when True, alert you to the presence of a member of an object family specified in presence_trigger: If soundalert: is set to True, MultiDetect.py will probe for a valid sound output, and it will turn itself off when none is found. Due to the wild and wooly world of Linux audio, the probing is rather messy, and it can take time. Set soundalert: to False if your machine has no sound, or you don’t want to hear any. 

The object families are created in **labeldict:** If, for instance, labeldict is equal to { 'Lily':'cat', 'Bella':'cat','Chloe':'cat','Tweety':'bird'}, and if presence_trigger: is set to ‘cat,’ an audible sound will be played when Lily, Bella, or Chloe are detected. The bird Tweety will be ignored. 

The same logic is used for **record_autovideo:** When True, video is recorded if a member of an object family specified in **presence_trigger:** is detected. If set as above, the appearance of Lily, Bella, and Chloe will be recorded. The video files will be stored in subdirectories of **output_path:**

**Record_framecode:** causes the creation of special framecode files. The framecode files are supplemental to their respective video files. For each frame where an object is detected, a line in the framecode file is created. Recorded will be the dimensions of the bounding box(es), the name of the detected object, the confidence, and the frame number. If more stats are needed, they can easily be added in code. The framecode file can be helpful for statistics, or for feedback training. 

For more in-depth studies, a result log can be kept by setting **maintain_result_log:** to True. The result logs will be stored in subdirectories off **result_log_basedir:**  The result log will document the result of each call to detect_image_extended(), along with the round-trip time of each call. The file is in CSV format and can be opened in Excel for further studies. 

## Fighting frame drift
Frame drift will be an ever-present problem when processing multiple real-time streams. Most IP cameras lack common timecode, have an unreliable timebase, or report unreliable or plain false fps etc data. MultiDetect.py will attempt every mitigation possible at its end, but it can be a losing battle. Here are a few steps to keep frame drift in check as much as possible. 

**buffer_depth:** sets the size of the frame buffer in frames. If the buffer is set too high, frames will simply pile up in the buffer. Keep the buffer low to combat drift, and high enough for smooth video. A buffer of around 20 should be fine. Note: Some webcams do not allow buffers > 10.

IP cameras don’t always send video immediately, some take considerably longer than others. This is exacerbated when a number of IP cameras is started up at the same time by multiple processes. MultiDetect.py addresses this with **sync_start:** When set to True, the respective video_process will initialize its video source, and it will signal readiness to the master process while holding the video. Once the master process has received a ‘videoready’ signal from all processes, it will send a sync_start signal to all processes, and they will all start processing video at the same time, and in a semblance of sync. If no videoready signal has been received from all processes (probably because one crashed) the surviving processes will start playing after a timeout. The timeout is sync_start_wait:, internally multiplied by num_processes: 

Even the most capable GPU can easily be overwhelmed when subjected to higher frame rates than it can process, even more so when multiple frames are being handled. This will result in video lag, and quickly, the timing of the individual streams will drift apart. To help mitigate video drift, you should send fewer frames to YOLO than a single process can handle. The first step would be to lower the incoming frame rate at the source. For further mitigation, MultiDetect.py has three settings that allow only a subset of the video frames to be run through YOLO, while all frames are displayed.

**do_only_every_initial:**  causes only every nth image to be run through YOLO. If set to 4, every 4th frame would be run through YOLO. With a 24fps video, YOLO would only be subjected to 6fps

**do_only_every_autovideo:** runs only every nth image through yolo during autovideo recording 

**do_only_every_present:**  runs every nth image through yolo after activity was detected and until presence_timer_interval times out

A judicious use of these settings will also result in considerable power savings. 

## The Master Window

The Master Window is a rudimentary status and control window. It is driven by the master process. It allows you to Quit and Restart the app, to start and stop recording of video and to turn on/off audible prompts. Most of all, it gives you a picture of the frame rates achieved by each video_process.

The window will list the GPU(s) usable by YOLO via Tensorflow and CUDA. Note: The GPU number is as reported by Tensorflow, it sometimes is different from what Nvidia-smi says. 

To help you in adjusting the proper frame rates, the master window will give you running stats per process. Let it run for a little while, the numbers are running averages and need a little time to settle. Current frames and seconds are displayed as reported by the streams. Keep in mind that IP cams often report bogus data. **IFPS** is the incoming frame rate, again as reported. It depends on what the videosource claims it is. 

**The strategy to determine incoming fps is as follows:**

If the stream exposes **PROP_FPS**, and if the PROP_FPS reported look somewhat believable ( 0 < PROP_FPS < 200 – adjust the code for a super high speed camera), we take that number. The incoming FPS are updated continuously to reflect any changes during the run.
If the above fails, we will try **PROP_POS_FRAMES / PROP_POS_MSEC * 1000** , and if the result passes the sanity check as above, we will take it.  The incoming FPS are updated continuously, HOWEVER, as the total of frames since start is divided by the total of seconds since start, the fps will be a cumulative average.
If all else fails, we will measure the incoming fps in a **timing loop**. We do this only once after the stream starts.

A --- denotes a missing, or bogus frame rate. **YFPS** is the frame rate the respective YOLO stream currently can handle. **OFPS** is the outgoing frame rate. **N** tells you that every nth frame is being processed. **EFPS** is the effective frame rate of what is sent to YOLO, i.e. IFPS divided by n.

Let’s say you process two streams, and you see a YFPS of 14, a common number for a 1080ti. To avoid video drift, the number of frames (EFPS) sent to YOLO should not exceed its YFPS, actually, it should be a little lower to allow for overhead. Be aware that YFPS is the average round-trip speed of your video sent through YOLO. The calling video_process needs time also, as a matter of fact, a lot of time is spent in waiting for the next video frame. (If you want to investigate this a bit further, set **profile:** to true. With the help of cProfile, a timing profile of the video_process() will be built for 3000 frames, and shown for each process, so you can see where all the time is spent.)

If EFPS > YFPS, either lower the frame rate at the source, or adjust the do_only_every settings. Say you incoming frame rate is 16, and do_only_every_initial is set to 4. This will result in an EFPS of 4, a number that will be easily handled. These settings become important as you add multiple streams to one GPU. You will see the per-process YFPS go down, because the power of the GPU is shared by multiple processes. If OFPS sinks below IFPS, you will experience frame drift in short order. Reduce the incoming frame rate, and/or adjust the do_only_every settings.

When redcording video, outgoing OFPS should be the same as incoming IFPS. If the incoming stream isa webstream, or prodcued by a webcam, the outgoing FPS cannot exceed icoming FPS. If the sopurce is a file, the file can be consumed rapidly, and the outgoing rate can be much higher the the rated fps. If the incoming source is a file, MultiDetect.py will adjust **cv2.waitkey** to approximate the proper outgoing rate. It may take 30 seconds or so until an equilibrium is reached. 
 
As a default, the rolling YFPS average is calculated for the last 32 frames. You can adjust this number with the rolling_average_n setting in MultiDetect.conf

The master window also will show the total of frames and seconds of each video stream and any differences between the streams and stream #1. This is based on what the video streams report via cv2.videocapture. These properties can be very unreliable, especially between IP cameras of different brands. As long as the actual video streams are halfway in sync, do not be alarmed if you see the frame and second differences pile up.


**Buttons:**  **“Quit”** will quit. **“Restart”** will restart  MultiDetect.py. **“Record,”** if green, will cause as video_processes to record their video stream. If red, the button will stop recording. The **“Redraw”** button will cause the video_processes to move their output windows into the coordinates specified in their Process_ block. **Aduio** will turn on/off audible chimes.
 

## The YOLO settings

**model_path:** should point to "…/TrainYourOwnYOLO/Data/Model_Weights /trained_weights_final.h5" or wherever you put your model.
**classes_path:** should point to "…/TrainYourOwnYOLO/Data/Model_Weights/data_classes.txt" or wherever you stored the file.
**anchors_path:** should point to "…/TrainYourOwnYOLO/2_Training/src/keras_yolo3/model_data/yolo_anchors.txt"or wherever you stored the anchors.
**run_on_gpu:** The GPU number you want the process to run on. The number is the one reported by Tensorflow and shown in the Master Window. It may be different from what nvidia-smi says.
**gpu_memory_fraction:** How much GPU memory to allocate to the process. 1 = 100%, 0.1 = 10% . Process will crash if GPU memory insufficient. When set to less than 1 (100%), the total for all processes must be less than 100% to allow for overhead. You will be able to fit more processes into a card that is not used for video output. The number is a recommendation, and will result in slightly different memory footprints. Experiment.
**hush:** will, when True, try to suppress the annoying status messages during startup. It also may suppress non-fatal error messages. It is recommended to set hush: to false during setup and testing. It can be turned on when things run smoothly.
**allow_growth:** GPU memory allocation strategy. -1 let Keras decide, 0 disallow, 1 allow memory to dynamically grow. Best setting to optimize memory usage appears to be 1 
**score:** YOLO will report objects at and above that confidence score, it will keep anything lower to itself.
ignore_labels: A list (i.e. ['Aaa','Bbb'] ) of object names YOLO will not report when found. Keep empty [ ] to disable this feature.

**video_path:** specifies the incoming video source for that process. It can be anything understood by cv2.videocapture. It can be a video file, an URL of a video stream, or a webcam. For a webcam, set video-path to 0 (1, 2, 3 ....) no quotes. For video file, set to path to file name, in quotes. For live stream, set to the url of the stream, in quotes.


Like all of the settings, you can put the YOLO settings once into the Common: block, and they will be used by all processes. If you put the settings into a Process_ block, they will be used by that process only. This way, you can use different models in different processes, and you can assign a specific GPU to a process. If a setting is the same in all Process_ blocks, simply keep in it Common: 

## Output settings

**window_wide:** and **window_high:** set the dimensions of the video output of the process.
**window_x:** and **window_y:**  specify where on the screen the respective window is to be placed.
With these settings, you can put multiple video windows on one monitor. To move video output to separate monitors, first set-up multi-monitors in your display settings. Set window_wide: and window_high: to match the resolution of your separate monitors. Then set window_x: in Process_1: large enough so that the output window gets pushed to the separate monitor. Set window_y: in Process_1: to 0. Repeat for a second separate monitor as needed. 

## Hush!

Tensorflow has the nasty habit to clutter the screen with rather useless status messages. Not only do they look messy, they also drown out real error messages. You can shut-up MultiDetect.py with the hush: setting. When True, most chattiness should cease. “Should,” because Tensorflow 2.3 introduced new chitchat during the initialization phase of YOLO. This can be silenced by starting TensorFlow.py with a small script called MM. MD will set the environment variable **TF_CPP_MIN_LOG_LEVEL=3** and start MultiDetect.py. You can also move MD somewhere on the path, for instance into /usr/local/bin, make it change to the directory where MultiDetect.py resides, and make md executable. Start MultiDetect.py via MD, and all will be quiet.

## General comments

Throughput doesn’t seem very sensitive to the amount of GPU memory. I have reduced the memory allocation of a single process to just 4% of the available memory of a 11G 1080ti, and YOLO still ran at 18fps. With multiple processes, the frame rate of each process of course sinks. However, the total frame rates of all processes occasionally is higher than a single process running at full speed. Variations between runs are quite common. Frame rates often are higher after a hard reset. I noticed that occasionally and very counterintuitively, a higher do_only_every_ setting can result in lower YFPS. I have no idea why. Also quite mysteriously, YOLO occasionally delivered a higher frame rate when objects were detected, and lower FPS when there was nothing to see. The above could be caused by dynamic frequency scaling, but it’s just a guess.

## A word on IP cameras 
The market is flooded with cheap IP cameras. Their picture quality can be quite decent these days, their software quality often is lousy. You will often find them contacting servers in China. If you don’t want to star on insecam.org, the infamous database of live IP cameras, do the following: Avoid WiFi cams, use hardwired. Put the cams behind a firewall, making sure that the cameras can’t be reached from the outside, AND MOST OF ALL make sure that the cameras cannot reach the outside. This also keeps the cams from updating their internal clock via NTP. For that, set up your own local NTP server that acts as a common reference for your cams.

## Infamous last words
Development was on Ubuntu 18.04 and 20.04, with Python3.7, CUDA 10.0, and the Nvidia 450 video driver. I have developed and tested MultiDetect.py on a machine with a 3970x Threadripper and 128G of memory, and on an ancient Intel 6700K with 64G of memory. I have a stack of Geforce 1060/6G and Geforce 1080ti/11G GPUs, and I used them in various combinations. No other systems were tested. 

I am a retired advertising executive, and my programming “skills” are completely self-taught. I tried my hands on assembler and BASIC half a century ago when computers had 8 bits and 4K of memory. I took up Python to keep me busy after retirement. My code definitely is in need of improvement, and it could be completely flawed. The stuff is on Github in hope for improvement – of the code, and of myself. Have at it. 
