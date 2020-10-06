#!/usr/bin/env python3.7
"""
Testbench app for multi-stream, multi-gpu YOLO
File name: MultiDetect.py
Author: Bertel Schmitt, with inspirations from a cast of thousandss
Date created: 4/20/2013
Date last modified: 4/25/2013
Python Version: 3.7
License: This work is licensed under a Creative Commons Attribution 4.0 International License.
"""
from warnings import simplefilter
import os
import sys
from sys import argv
import time
import fcntl
import pickle
import atexit
from threading import Timer
import multiprocessing
import queue
import ast
import datetime
import subprocess
import tkinter
import cProfile
from pstats import SortKey
import pstats
import io
import copy
import functools
from timeit import default_timer as timer
from PIL import Image
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import cv2
#for error suppression
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import logging


def getappname(ending):
    """
    Return the name of the running app while stripping off ending
    """
    mf = os.path.basename(sys.argv[0])
    if mf.endswith(ending):
        return (mf[:-len(ending)])
    return ("")

def get_parent_dir(n=1):
    """
    Returns the n-th parent dicrectory of the current
    working directory
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")
resource_path = os.path.join(get_parent_dir(1), "3_Inference/MDResource")
data_path = os.path.join(get_parent_dir(1), "Data")
sys.path.append(src_path)
sys.path.append(utils_path)
pid_file = f'{resource_path}/{getappname(".py")}.pid'


from keras_yolo3.yolo import YOLO  # pylint: disable=C0413


simplefilter(action='ignore', category=FutureWarning)

def getparent():
    """
    Attempt to locate any possible parent of the running process. We go through all that work, because when we finally manage to shut down all
    those processes, the console may need a nudge with the enter key to accept new input, all because we started MultiDetect.py with another little bash script.
    So we hunt that down as well.
    """
    mypid = os.getpid()
    process = subprocess.Popen([f'pstree -hp | grep {mypid}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    mout, _ = process.communicate()
    mout = mout.decode('utf-8')
    if mout == '':
        return('')
    mout = mout.replace('-+-', '---')
    mouts = mout.split('---')
    matching = [s for s in mouts if str(mypid) in s]
    if matching == []:
        return('')
    ind = mouts.index(matching[0])
    if ind < 1:
        return('')
    parentproc = mouts[ind - 1]
    x = parentproc.split('(')
    parents = x[0]
    try:
        _ = int(x[1].replace(')', ''))
    except ValueError:
        return('')
    if parents in ('bash', 'sh', 'csh', 'zsh'):
        return('')
    print(f"Parents:{parents}")
    return(parents) # return what looks like the parent

def harakiri():
    '''
    Hack to kill any running process for good
    '''
    parent = getparent() # try to get possible parent process
    mf = os.path.basename(__file__)
    with open(f'{resource_path}/harakiri.sh', 'w+') as file:  # Create a small script
        file.write(f'pkill -f {mf}\n') #kill any instance of running script
        if parent != '':
            file.write('pkill -f cdd\n') #kill any instance of running script
        file.write(f'rm -rf {resource_path}/harakiri.sh\n') #delete the script
    _ = subprocess.Popen(['/bin/sh', os.path.expanduser(f'{resource_path}/harakiri.sh')])

def resurrect():
    '''
    Hack to kill running process, restart app when all killed
    '''
    mp = __file__
    mf = os.path.basename(__file__)
    with open(f'{resource_path}/resurrect.sh', 'w+') as file:  # Create a small script
        file.write(f'pkill -f  {mf}\n') #kill any instance of running python program
        file.write(f'{mp}\n') #run the killed python program
        file.write(f'rm -rf {resource_path}/resurrect.sh\n') #delete the script
    _ = subprocess.Popen(['/bin/sh', os.path.expanduser(f'{resource_path}/resurrect.sh')]) #now run the script, kill and restart python program, then delete resurrect.sh



def checkopen(filepath, filearg,auto_open = False, buffering = 0):
    """
    Checks whether file exists, if not, creates file and any necessary directories
    return file handle if autoopen
    """
    if not os.path.isfile(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True) #create directory and any parents if not exist
    if auto_open:
        return(open(filepath, filearg, buffering))
    return()


def get_device_dict():
    """
    Get available GPUs as a dict
    Each GPU returns a dict like {'device': '0', 'name': 'GeForce GTX 1080 Ti', 'pci bus id': '0000:00:00.0', 'compute capability': '6.1'}}
    Returned devdict is a dict of dicts, with the GPU index as the key. The whole thing will look something like:
    {'0': {'device': '0', 'name': 'GeForce GTX 1080 Ti', 'pci bus id': '0000:21:00.0', 'compute capability': '6.1'},
    '1': {'device': '1', 'name': 'GeForce GTX 1080 Ti', 'pci bus id': '0000:4c:00.0', 'compute capability': '6.1'}}
    Below code is a kludge and could most likely be written in a more elegant way, but it works, and it is called only once a program startup.
    """

    devdict={}
    for i in device_lib.list_local_devices():
        workdict={}
        b  = str(i).split('physical_device_desc:')
        if len(b) > 1 and 'XLA_' not in b[1]:
            subsplit=b[1].replace('"','').replace('\n','').split(', ')
            for it in subsplit:
                pair=it.strip().split(': ')
                if len(pair) > 1:
                    workdict[pair[0].strip()] = pair[1].strip()
            try:
                devdict[workdict['device']] = workdict
            except:
                pass
    return(devdict)

class CTL:
    """
    Container for sundry working variables
    """
    def __init__(self):
        self.procnum = 0

class DefStartup:
    """
    DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf
    Defaults for blk_Startup, will be overwritten by any existing settings in config file
    """
    def __init__(self):
        self.num_processes = 1 # Number of processes to launch
    #DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf

class DefMaster:
    """
    DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf
    Defaults for blk_Master, will set mctl in master process.
    Will be overwritten by any existing master settings in config file
    """
    def __init__(self):
        self.initdict = {}  #dictionary for device stats
        self.statsdict = {} # dictionary for stats messages
        self.devdict = {} # device dictionary
        self.coldict = {} # dictionary of column lengths
        self.gridtop = 0  # workvar for status message
        self.master_window_title = "Master" # title for master monitor window
        self.redraw = False #used for optional screen redraw
        self.hush = False
        #DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf

class DefCommon:
    """
    DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf
    Defaults for blk_Common, will set FLAGS for each video process.
    Will be overwritten by any existing per-process setting in config file
    """
    def __init__(self):
        self.showmaster = True #if true, show master window, if flase, turn master window off
        self.hush = False    #Try to suppress most warnings and error messages
        self.testing = False #if True, play video as specified in testvideo. Set true as default in case no video source specified
        self.max_cams_to_probe =  10 #maximun number of usb cameras to probe for. Probing takes time at startup. 10 is a good number
        self.profile = False #if true, profile video_process() for 3000 frames, then show results for each process

        self.testvideo =   resource_path + "/testvid.mp4"  #Dummy video file. Default: .../TrainYourOwnYOLO/3_Inference/MDResource/testvid.mp4
        self.clicksound =  resource_path + "/chime.mp3"    #Audible alert.  Default: .../TrainYourOwnYOLO/3_Inference/MDResource/chime.mp3
        self.silence =     resource_path + "/silence.mp3"  #needed for test of audio capabilities. Default: .../TrainYourOwnYOLO/3_Inference/MDResource/silence.mp3
        self.soundalert =  True  #play sound when a class specified in presend_trigger  is detected

        self.sync_start = False  #synchronize start of all video streams in an attempt to mitigate time drift between videos
        self.sync_start_wait = 10 #time in seconds to wait for sync_start before timeout. This will be multiplied by num_processes to allow for longer staging due to higher load

        self.buffer_depth = 20   #size of frame buffer in frames. Keep low to combat drift, high enough for smooth video. Note: Some webcams do not allow buffers > 10.
        self.avlcam = [] #placeholder for available camera list
        self.workingcam = [] #placeholder for working camera list


        # Families for automatic action. If set as below, "Fluffy" and "Ginger" would be members of the "cat" family, and all "cat" family members would trigger automatic action, such as sound alert (if set and supported) or automatic recording of video (if set)
        #labeldict = {'Fluffy':'cat','Ginger':'cat','Crow':'bird'}
        #presence_trigger = ['cat']

        self.labeldict = {}
        self.presence_trigger = []
        self.ignore_labels =  []  #List of labels for YOLO to ignore and not to report


        self.presence_timer_interval = 20   # seconds until reset
        self.ding_interval = 20   #allow only one ding within that period
        self.record_autovideo = False # If true, automatically record captured video, and store it in output_path
        self.output_path = resource_path+ "/videos/"  #basepath for recorded video files. Default: .../TrainYourOwnYOLO/3_Inference/MDResource/videos/ Specify yours in MultiDetect.conf
        self.record_framecode =  True  #record framecode files corresponding to recorded video files
        self.framecode_path =  resource_path+ "/framecode/"  #basepath for framecode files. Default  .../TrainYourOwnYOLO/3_Inference/MDResource/framecode/ Specify yours in MultiDetect.conf


        self.maintain_result_log =  True  # Whether to keep a running log with result timings etc.
        self.result_log_basedir =  resource_path+ "/resultlogs/" # Where to keep the result logs. Default  .../TrainYourOwnYOLO/3_Inference/MDResource/resultlogs/ Specify yours in MultiDetect.conf

        self.rolling_average_n =  32      # Length of rolling average used to determine average YOLO-fps
        self.show_stats =  False        #print stats to console, or not

        # To make up for less capable GPUs, or for high frame rates, use these settings to run only every nth frame through YOLO detection
        # For instance, a frame rate of 25fps, and a setting of 2 would result in ~12 frames per second to be run through YOLO, which is well within the capabilities of a moderately-priced GPU
        # These settings also make for considerable power savings

        self.do_only_every_autovideo =  4   #run only every nth image through YOLO during autovideo recording
        self.do_only_every_initial =    4   #run only every nth image through YOLO during normal times
        self.do_only_every_present =    4   #run every nth image through YOLO after activity was detected and until presence_timer_interval times out

        #Default settings for on-screen type. Can be overridden in app
        self.osd =              False        #  True show on screen display, False don't
        self.osd_fontFace =      0           #  index of cv2.FONT_HERSHEY_SIMPLEX
        self.osd_fontScale =     1.2         #  font size
        self.osd_fontColor =     (0,255, 0)  #  green
        self.osd_fontWeight =    3           #  will be adjusted for smaller/bigger screens
        self.osd_lineSpacing =   45          #  line-to-line spacing, will be adjusted for smaller/bigger screens

        #global yolo settings. To use different models on a per stream basis, specify these settings in the respective process block
        self.score = 0.45              #report detections with this confidence score, or higher
        self.run_on_gpu = "0"             #specify gpu "0", "1". "0,1" for both is required by spec, has little or no effect. Set to "-1" to let Keras pick the best GPU
        self.gpu_memory_fraction = 1        #how much GPU memory to claim for process. 1 = 100%, 0.1 = 10% . Performance will suffer when memory-starved, process will crash if GPU memory insufficient
        self.allow_growth = 1             #-1 let Keras decide, 0 disallow, 1 allow memory used on the GPU to dynamically grow. DISALLOW when setting gpu_memory_fraction < 1
        #output window settings
        self.window_title = "Master"

        self.video_path = resource_path+ "/CatVideo.mp4" #path to incoming video stream.
        self.model_path =   data_path +  "/Model_Weights/trained_weights_final.h5" # Default location, specify yours in MultiDetect.conf
        self.anchors_path = src_path  + "/keras_yolo3/model_data/yolo_anchors.txt"  # Default location, specify yours in MultiDetect.conf
        self.classes_path = data_path + "/Model_Weights/data_classes.txt" # Default location, specify yours in MultiDetect.conf
        self.iou =   0.9 # intersection over union threshold

        self.window_wide = 1024  # Default width of output window. Specify yours in in the config file
        self.window_high = 600   # Default height of output window. Specify yours in in the config file
        self.window_x = 200      # Default x-position of output window on screen. Set this in config file to move window, also to a separate monitor
        self.window_y = 200      # Default y-position of output window on screen. Set this in config file to move window, also to a separate monitor

        self.presence_counter = 0

        ## DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf


def RetriggerableTimer(*args, **kwargs):
    """
    Global function for Timer
    """
    return _RetriggerableTimer(*args, **kwargs)

class _RetriggerableTimer(object):
    """
    Retriggerable timer
    """
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.timer = Timer(self.interval, self.function)

    def start(self):
        """
        Alias for run
        """
        self.timer.start()

    def reset(self, interval=None):
        """
        Reset the timer (to the old interval if not defined) and restart
        """
        if interval is None:
            interval = self.interval
        self.interval = interval
        self.timer.cancel()
        self.timer = Timer(self.interval, self.function)
        self.timer.start()

    def cancel(self):
        """stop the timer"""
        self.timer.cancel()

    def is_alive(self):
        """signal running, or not"""
        return(self.timer.is_alive())

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    Routine courtesy https://stackoverflow.com/users/772487/jeremiahbuddha

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def listcams(maxcams = 10):
    """
    List available and working cams. Routine courtesy https://stackoverflow.com/users/2132157/g-m
    """
    dev_port = 0
    working_ports = []
    available_ports = []
    for _ in range(0,maxcams):
        camera = cv2.VideoCapture(dev_port) # pylint: disable=E1101
        if camera.isOpened():
            is_reading, _ = camera.read()
            if is_reading:
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
        dev_port +=1
    return(available_ports,working_ports)


def settext(window, mytext, myrow,  mycol,  font = "Courier", bold = False, size = 12, anchor = "SW", columnspan = 26, borderwidth = 0):
    """
    draw text on a tk window
    """
    thefont = f"{font} {size}"
    if bold:
        thefont = thefont + " bold"
    mylabel = tkinter.Label(window, text = mytext , font = thefont, borderwidth = borderwidth)
    mylabel.grid(sticky=anchor, row = myrow, column = mycol, columnspan = columnspan)
    return(mylabel)

def setrows(window, rows):
    """Create empty rows"""
    for n in range(0,rows+1):
        settext(window,"   ",n,0,columnspan=1)
    return()

def clear_tk_window(window):
    """
    Wipe a tkinter window clean
    """
    for widget in window.winfo_children():
        widget.destroy()

def record_all(mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0,Command = "startrecording")

    mctl.button_r.configure( text="Rec Stop" , bg='red',command=functools.partial(stoprecord_all, mctl))
    mctl.window.update()

def stoprecord_all(mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0,Command = "stoprecording")
    mctl.button_r.configure( text="Record" , bg='green', command=functools.partial(record_all, mctl))
    mctl.window.update()

def audio_on(mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0,Command = "audioon")
    mctl.button_a.configure( text="Audio-off" , bg='green',command=functools.partial(audio_off, mctl))
    mctl.window.update()

def audio_off(mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0, Command = "audiooff")
    mctl.button_a.configure( text="Audio-on" , bg='red',command=functools.partial(audio_on, mctl))
    mctl.window.update()

def autovid_on(mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0, Command = "autovidon")
    mctl.button_av.configure( text="Autorec-off" , bg='green',command=functools.partial(autovid_off, mctl))
    mctl.window.update()

def autovid_off(mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0, Command = "autovidoff")
    mctl.button_av.configure( text="Autorec-on" , bg='red',command=functools.partial(autovid_on, mctl))
    mctl.window.update()


def window_redraw(mctl, doqueue = True):
    """Cause video_processes to set their output windows into their proper places"""
    if doqueue:
        put_in_queue(mctl, From = -1 ,To = 0, Command = "windowredraw")
    mctl.window.update()

def button_shutdown(mctl):
    """Process shutdown"""
    mctl.shutdown_action = -1 #shutdown
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "abortvideowriter", 'Args':{}}, mctl.procdict)
    mctl.window.update()
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "shutdown", 'Args':{}}, mctl.procdict)
    mctl.window.update()
    panic_shutdown_timer = RetriggerableTimer(10, harakiri) # timer for emergency shutdown in case not all processes acknowledge shutdown command
    panic_shutdown_timer.start()
    return(mctl)

def button_resurrect(mctl):
    """Process shutdown"""
    mctl.shutdown_action = 1 #resurrect
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "abortvideowriter", 'Args':{}}, mctl.procdict)
    mctl.window.update()
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "shutdown", 'Args':{}}, mctl.procdict)
    mctl.window.update()
    panic_shutdown_timer = RetriggerableTimer(10, resurrect) # timer for emergency shutdown in case not all processes acknowledge shutdown command
    panic_shutdown_timer.start()
    return(mctl)


def do_masterscreen(mctl):
    """
    Monitor screen - set-up master screen
    """
    if not mctl.showmaster:
        return()
    clear_tk_window(mctl.window)
    setrows(mctl.window,mctl.totallines) # set up the grid we'll be using

    setattr(mctl,'button_q', tkinter.Button(mctl.window, text=" Quit ", bg='red', command=functools.partial(button_shutdown, mctl)))
    mctl.button_q.grid(row=0, column=0, ipadx = 5 , sticky='SW')

    setattr(mctl,'button_rs',tkinter.Button(mctl.window, text="Restart", bg='orange',command=functools.partial(button_resurrect, mctl)))
    mctl.button_rs.grid(row=0, column=1 , sticky='SW')

    setattr(mctl,'button_r',tkinter.Button(mctl.window, text="Record", bg='green',command=functools.partial(record_all, mctl)))
    mctl.button_r.grid(row=0, column=2 , sticky='SW')

    if mctl.soundavailable:
        if mctl.soundalert:
            setattr(mctl,'button_a',tkinter.Button(mctl.window, text="Audio-off", bg='green',command=functools.partial(audio_off, mctl)))
        else:
            setattr(mctl,'button_a',tkinter.Button(mctl.window, text="Audio-on", bg='red',command=functools.partial(audio_on, mctl)))
    else: #no sound support
        setattr(mctl,'button_a',tkinter.Button(mctl.window, text="Audio", bg='grey',command=None))
    mctl.button_a.grid(row=0, column=4 , sticky='SW')

    if mctl.record_autovideo:
        setattr(mctl,'button_av', tkinter.Button(mctl.window, text="Autorec-off", bg='green',command=functools.partial(autovid_off, mctl)))
    else:
        setattr(mctl,'button_av', tkinter.Button(mctl.window, text="Autorec-on", bg='red',command=functools.partial(autovid_on, mctl)))
    mctl.button_av.grid(row=0, column=3 , sticky='SW')

    setattr(mctl,'button_rd',tkinter.Button(mctl.window, text="Redraw", bg='yellow',command=functools.partial(window_redraw, mctl)))
    mctl.button_rd.grid(row=0, column=5 , sticky='SW')

    # line_1 = "Initial Stats                                                    ".rjust(mctl.maxline).ljust(mctl.maxline)
    line_1 = "GPU Stats".ljust(mctl.maxline).rjust(mctl.maxline)
    line_2 = ("GPU#".rjust(4) +   "_____________ GPU Name _".rjust(25) + "CCap".rjust(5)  + "            ").ljust(mctl.maxline).rjust(mctl.maxline)
    line_5 = "Current Stats                                                    ".ljust(mctl.maxline).rjust(mctl.maxline)
    line_6 = ("Proc".rjust(4) + "Frames".rjust(9) + "FDiff".rjust(7) + "Seconds".rjust(9) + "SecDiff".rjust(
        8) + "IFPS".rjust(5) + "EFPS".rjust(5) + "YFPS".rjust(5) + "OFPS".rjust(5) + "n".rjust(3) + "Timestamp".rjust(12) + "GPU".rjust(4) + "FRAC".rjust(6)).ljust(mctl.maxline).rjust(mctl.maxline)
    settext(mctl.window,line_1,2,1, bold = True, size = 16, font = "Arial")
    settext(mctl.window,line_2,3,1, bold = True)

    mybase = 3
    for gpunum in mctl.devdict:
        mybase = mybase + 1
        locdict = mctl.devdict[gpunum]
        l = (gpunum.rjust(4) + locdict['name'].rjust(25) + locdict['compute capability'].rjust(5)).ljust(mctl.maxline).rjust(mctl.maxline)
        settext(mctl.window, l, mybase,  1)
    mybase = mybase + 2
    settext(mctl.window,line_5,mybase,1, bold = True, size = 16, font = "Arial")
    mybase = mybase + 1
    settext(mctl.window,line_6,mybase,1, bold = True)
    mctl.topdataline = mybase + 1  # set top location for ongoing data lines
    mctl.window.attributes("-topmost", True)
    mctl.window.update()
    return (mctl)


def get_currentline(mctl, procnum):
    """
    Monitor screen - format current data for each process
    """
    mpos = int(mctl.statsdict[str(procnum)]['POS_FRAMES'])
    msec = mctl.statsdict[str(procnum)]['POS_MSEC']
    ofps = mctl.statsdict[str(procnum)]['StrFPS']
    mstamp = mctl.statsdict[str(procnum)]['TimeStamp']
    yfps = mctl.statsdict[str(procnum)]['MaxYoloFPS']
    n = int(mctl.statsdict[str(procnum)]['DoOnlyEvery'])
    infps = mctl.statsdict[str(procnum)]['InFPS']
    mygpu= str(mctl.initdict[str(procnum)]['run_on_gpu'])
    try:
        memfrac = '{:.2f}'.format(round(mctl.initdict[str(procnum)]['gpu_memory_fraction'], 2))
    except(ZeroDivisionError, TypeError) as e:
        memfrac = '---'

    if infps > 1000 or infps < 1 : #bogus fps
        strefps = strinfps = "---"
    else:
        strinfps= '{:.1f}'.format(round(infps, 1))
        strefps= '{:.1f}'.format(round(infps/n, 1))

    if msec/1000 <= 1: #reports no proper msec
        msecds = "---"
        strmsec = "---"
    else:
        try:
            msecds  = '{:.0f}'.format(round(msec / 1000 - int(mctl.statsdict['1']['POS_MSEC']) / 1000, 0))
        except KeyError:
            msecds = "---"

        strmsec = '{:.0f}'.format(round(msec / 1000, 0))

        # no diff for #1, compare others with #1
    if procnum == '1':
        mposds = "---"
        msecds = "---"
    else:
        try:
            mposds = str(int(mpos - int(mctl.statsdict['1']['POS_FRAMES'])))
        except KeyError:
            mposds = "---"


    if mpos <= 0:  # bogus, or unavailable frame position
        mpos = "---" # # frame position should be > 0
        mposds = "---" # In that case, frame difference would not make sense either
    il = (str(procnum).rjust(4) + str(mpos).rjust(9) + mposds.rjust(7) + strmsec.rjust(9) + msecds.rjust(8) + strinfps.rjust(5) + strefps.rjust(5)
        + '{:.1f}'.format(round(yfps, 1)).rjust(5) + '{:.1f}'.format(round(ofps, 1)).rjust(5) + str(n).rjust(3)
        + milliconv(mstamp).rjust(12) + mygpu.rjust(4) + memfrac.rjust(6)  ).ljust(mctl.maxline).rjust(mctl.maxline)
    return (il)


def do_dataline(mctl, currproc, emerg = ""):
    """
    Monitor screen - draw data line for a process
    """
    if emerg == "":
        mytext = get_currentline(mctl, str(currproc))
    else:
        mytext = str(currproc).rjust(4) +"   "+ emerg  # update with the new text

    if currproc not in mctl.datalinedict:
        curline = settext(mctl.window, mytext, mctl.topdataline + int(currproc) -1 ,1)
        mctl.datalinedict[str(currproc)] = curline  # preserve data line object
    else:
        textobj = mctl.datalinedict.get(str(currproc),None) # get a prior text label
        textobj.configure(text = mytext ) # update with the new text
    mctl.window.update()
    #When we have a videoprocesses reporting, try settling the windows into their proper places
    if len(mctl.datalinedict) >= mctl.procnum and mctl.redraw:
        for _ in range(0,3):
            send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "windowredraw", 'Args':{}}, mctl.procdict)
            time.sleep(0.5)
        mctl.redraw = False
    return (mctl)



def init_yolo(FLAGS):
    """
    create a yolo session
    Removed "gpu_num" to avoid confusion. gpu_num allegedly allows running THE SAME SESSION on multiple GPUs, but in my testing, it didn't do much, if anything.
    If you need the parameter, simply add   "gpu_num": FLAGS.gpu_num,   to the parameter block below, and add "gpu_num:  1" (or 2, or 3 , or ... without the quotes)  in the config file
    This uses a slightly changed Yolo. In the old version, detect_image returns out_prediction, image. The new version of detect image, detect_image_extendedm, returns out_prediction, image, labels, elapsed_time
    Most of the settings can be set in the config file
    """
    if FLAGS.hush:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)
        silence(True)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        tf.logging.set_verbosity(3)
        silence(False)


    mykwargs = {
        "model_path": FLAGS.model_path,
        "anchors_path": FLAGS.anchors_path,
        "classes_path": FLAGS.classes_path,
        "score": FLAGS.score,
        "model_image_size": (416, 416),
        "hush": FLAGS.hush,
    }

    # only set optional parameters that have been specified, allow defaults to be set for all others

    if hasattr(FLAGS, 'gpu_num'):
        mykwargs['gpu_num'] = FLAGS.gpu_num
    if hasattr(FLAGS, 'iou'):
        mykwargs['iou'] = FLAGS.iou
    if hasattr(FLAGS, 'run_on_gpu'):
        mykwargs['run_on_gpu'] = FLAGS.run_on_gpu
    if hasattr(FLAGS, 'gpu_memory_fraction'):
        mykwargs['gpu_memory_fraction'] = FLAGS.gpu_memory_fraction
    if hasattr(FLAGS, 'allow_growth'):
        mykwargs['allow_growth'] = FLAGS.allow_growth
    if hasattr(FLAGS, 'ignore_labels'):
        mykwargs['ignore_labels'] = FLAGS.ignore_labels

    yolo = YOLO( **mykwargs )
    return (yolo)

def writethelog(x):
    """
    Write x to log
    """
    nfile = argv[0]
    nfile = nfile[:-3] + ".log"
    with open(nfile, 'a+') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + " -  " + str(x) + "\n")
    return ()

def presence_timeout():
    """
    Actions when presence_timer times out
    """
    return ()

def ding_timeout():
    """
    Audible prompt window timed out. No action needed
    """
    return ()

def soundtest(FLAGS):
    """
    Test audio with FLAGS.silence. Return True if successful, false if not.
    """
    try:
        sound = AudioSegment.from_mp3(FLAGS.silence)
    except FileNotFoundError:
        return(False)
    else:
        return(dosound(sound,FLAGS.hush))

def playsound(FLAGS):
    """
    Play sound at sound file. Return True if successful, false if not. Needs working sound system. If sound not working.
    app will not play sound now, and won't try playing sound later
    """
    try:
        sound = AudioSegment.from_mp3(FLAGS.clicksound)
    except FileNotFoundError:
        return(False)
    else:
        return(dosound(sound,FLAGS.hush))

def dosound(sound,hush):
    """
    play sound, without screen clutter if hush is set
    """
    if hush:
        with suppress_stdout_stderr():
            play(sound)
    else:
        play(sound)
    return(True)


def vidproc_shutdown(myctl, echo):
    """
    shut down this video_process. If echo == True, signal other video_process(es) to do same
    """


    writethelog(f"{myctl.procnum}- shutdown")
    if echo:
        put_in_queue(myctl, From = myctl.procnum,To = 0,Command = "stoprecording")
        put_in_queue(myctl, From = myctl.procnum,To = 0,Command = "shutdown")
    myctl.cv2.destroyWindow(myctl.window_title)

    if myctl.out is not None and myctl.out.isOpened():
        myctl.out.release()
        myctl.cv2.VideoWriter = None

    try:
        myctl.ding_timer.cancel()
    except AttributeError:
        pass
    try:
        myctl.presence_timer.cancel()
    except AttributeError:
        pass
    try:
        myctl.yolo.close_session()
    except AttributeError:
        pass
    try:
        myctl.vid.release()
    except AttributeError:
        pass
    try:
        myctl.out.release()
    except AttributeError:
        pass
    try:
        myctl.fcf.close()
    except AttributeError:
        pass
    try:
        myctl.rsf.close()
    except AttributeError:
        pass
    raise SystemExit(f"Shutting down video_process #{myctl.procnum}")


def startup_shutdown():
    """ registered exit routine of startup"""
    writethelog("Startup - shutdown")
    sys.exit()


def master_shutdown(mctl):
    """
    Shutdown all video-process(es), followed by master_process
    """
    writethelog("Master shutdown")
    mctl.showmaster = False # prevent further status updates
    clear_tk_window(mctl.window)
    setrows(mctl.window,mctl.totallines) # set up the grid we'll be using
    settext(mctl.window,"Shutting down ...", 1, 1, font = "Arial" , bold = True, size = 24)
    mctl.window.update()
    #shutdown video_processes
    for key in mctl.procdict: #It's a dict
        if key in (-2,-1,0):
            continue
        myrec = mctl.procdict.get(key,[])
        if myrec[5]:  #Record alive True/False
            myrec[3].terminate()
    time.sleep(1)
    harakiri() # do just that
    raise SystemExit("Shutting down Master")

def rollcall(myctl,my_procdict):
    """
    Ask all video_processes and the master process whether they are still alive
    """
    #structure:      0:in_queue, 1:out_queue, 2:procnum, 3:process, 4:procname, 5:running, 6:counter
    #print("In rollcall")

    for key in my_procdict: #It's a dict
        if key in (-2, 0): #Only check on master and video processes
            continue
        myrec = my_procdict.get(key,[])
        if not myrec[5]: #Leave dead processes alone
            continue
        if not myrec[3].is_alive(): #check whether the process is alive
            if key == -1: # master process ...
                print("\n\nMaster process dead - stopping all processes, and shutting down\n")
                harakiri()

            if key != myrec[2]:
                print(f"Procdict corruption @ line {sys._getframe().f_lineno}. Key {key} not equal to key {myrec[2]} in record. Aborting.") # pylint: disable=W0212
            put_in_queue(myctl, From = -2, To = -1, Command="procdied", Args={'name': f'{myrec[3].name}', 'number': f'{myrec[2]}' })
            myrec[6] -= 1 #decrement count ... we are counting down the myrec[6] setting before we pronounce the process dead
            if myrec[6] <= -1: # Officially deceased
                myrec[5] = False
            procdict[key]  = myrec # store myrec back into prodict
    return()


def start_recording(myctl, putqueue=False):
    """
    Sundry stuff to do before recording
    """
    myctl.do_only_every = myctl.do_only_every_autovideo
    basenamevid=f'{datetime.datetime.now().strftime("%y%m%d")}/P{myctl.procnum}/{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}_P{myctl.procnum}.mp4'
    if myctl.output_path.strip().endswith("/"):
        myctl.videofile=myctl.output_path.strip()+basenamevid
    else:
        myctl.videofile=myctl.output_path.strip()+"/"+basenamevid

    if myctl.framecode_path.strip().endswith("/"):
        myctl.fcfile=myctl.framecode_path.strip()+basenamevid+".fc"
    else:
        myctl.fcfile=myctl.framecode_path.strip()+"/"+basenamevid+".fc"
    myctl.is_recording = True
    myctl.framecounter = 0
    if putqueue:
        put_in_queue(myctl, From = myctl.procnum, To=0, Command="startrecording", Args={})
    return (myctl)


def stop_recording(myctl, putqueue=False, delete = False):
    """
    switch off recorder via
    myctl properties, signal all to do same if putqueue True
    """
    myctl.do_only_every = myctl.do_only_every_initial
    myctl.is_recording = False
    myctl.autovideo_running = False
    if delete:
        myctl.presence_counter = -1 # signal deletion to fileops
    if putqueue:
        put_in_queue(myctl, From = myctl.procnum, To=0, Command="stoprecording", Args={'Delete': delete})
    return (myctl)

def preroll(myvid, frames):
    """roll some video frames"""
    for _ in range(0, frames):
        _, _ = myvid.read()
    return ()

def milliconv(ts):
    """convert float timestamp to hr:min:sec.1"""
    _ , mytime = datetime.datetime.fromtimestamp(ts).strftime(
        '%Y-%m-%d %H:%M:%S.%f').split()  # convert to date and time
    t, rm = mytime.split(".")  # split off the remaining millisecs
    rmf = round(float(f"0.{rm}"), 1)  # round remainder to 1 decimal
    ntime = (f"{t}.{str(int(rmf * 10))[0]}")  # stick it to time removing 0, and making sure only 1 char
    return (ntime)

def getoflfps(mycv2, vid, init_frames, init_msec):
    """Try getting official fps, either by direct query, or by calculation
    return fps, mode 0 = CAP_PROP_FPS, 1 = POS_FRAMES/POS/MSEC, 2 = timed loop"""
    oflfps = int(round(vid.get(mycv2.CAP_PROP_FPS), 0))
    if 0 < oflfps < 200:
        return (oflfps, 0)  #looks like OK fps, set mode 1

    #We need a few seconds of frame history for POS_FRAMES/POS_MSEC to work anyway, so we do a preroll, and time it here, in case POS_FRAMES/POS_MSEC also fails
    start = time.time()
    num_frames = 24
    preroll(vid,num_frames)
    seconds = (time.time() - start)
    loopfps = round(num_frames/seconds,1)
    if ('CAP_PROP_POS_MSEC' in supplist and 'CAP_PROP_POS_FRAMES' in supplist):  # try POS_FRAMES/POS_MSEC substitute
        oflfps = round((vid.get(mycv2.CAP_PROP_POS_FRAMES) - init_frames) / (vid.get(mycv2.CAP_PROP_POS_MSEC) - init_msec) * 1000, 1)
        if 0 < oflfps < 200: #sanity check
            return(oflfps, 1)  #looks like OK fps, set mode 2
    #all getprops methods failed, so return brute force timing loop
    return(loopfps,2)



def list_supported_capture_properties(cap, mycv2):
    """
    List the properties supported by the capture device.
    """
    print("")
    print("===============================================================================")
    print("Test of supported capture properties. Ignore errors")
    supported = list()
    for attr in dir(mycv2):
        if attr.startswith('CAP_PROP'):
            try:
                if cap.get(getattr(mycv2, attr)) != -1:
                    supported.append(attr)
            except:
                pass
    print("End of test")
    print("===============================================================================")
    print("")
    return supported

def makectl(FLAGS):
    """
    set up a subset of FLAGS as a multipurpose-object
    """
    myctl = CTL()  # make new controller
    setattr(myctl, 'procnum', FLAGS.procnum)
    setattr(myctl, 'record_autovideo', bool(FLAGS.record_autovideo))
    setattr(myctl, 'presence_timer_interval', FLAGS.presence_timer_interval)
    setattr(myctl, 'presence_counter', 0)
    setattr(myctl, 'autovideo_running', False)
    setattr(myctl, 'do_only_every_initial', FLAGS.do_only_every_initial)
    setattr(myctl, 'do_only_every_autovideo', FLAGS.do_only_every_autovideo)
    setattr(myctl, 'video_path', FLAGS.video_path)
    setattr(myctl, 'testvideo', FLAGS.testvideo)
    setattr(myctl, 'testing', FLAGS.testing)
    setattr(myctl, 'output_path', FLAGS.output_path)
    setattr(myctl, 'is_recording', False)
    setattr(myctl, 'yolo', init_yolo(FLAGS))
    setattr(myctl, 'is_test', False)
    setattr(myctl, 'run_on_gpu', FLAGS.run_on_gpu)
    setattr(myctl, 'gpu_memory_fraction', FLAGS.gpu_memory_fraction)
    setattr(myctl, 'osd_help', False)
    setattr(myctl, 'osd', FLAGS.osd)
    setattr(myctl, 'fontFace', FLAGS.osd_fontFace)
    setattr(myctl, 'fontScale', FLAGS.osd_fontScale)
    setattr(myctl, 'fontColor', FLAGS.osd_fontColor)
    setattr(myctl, 'fontThickness', int(FLAGS.osd_fontWeight))
    setattr(myctl, 'fontSpacing', FLAGS.osd_lineSpacing)
    setattr(myctl, 'window_title', FLAGS.window_title)
    setattr(myctl, 'framecode_path', FLAGS.framecode_path)
    setattr(myctl, 'framecounter', 0)
    setattr(myctl, 'videofile', "")
    setattr(myctl, 'fcfile', "")
    setattr(myctl, 'out', None) #placeholder
    setattr(myctl, 'fcf', None) #placeholder
    return (myctl)


def vidproc_checkqueue(myctl, FLAGS):
    """
    Verify content of queue coming into the video process. Operate on results if needed
    """
    while not myctl.in_queue.empty():
        # work the queue
        try:
            # commandset structure {'To':process (1....),master (-1), or all (0),'From':process (1...),or master (-1),'Command':'','Args':{'Arg1': 'Val1','Arg2': 'Val2'}}
            my_commandset = myctl.in_queue.get_nowait()
        except queue.Empty:
            continue
        try:
            # do not process if not To this process, or To: All  or from this process
            if (my_commandset['To'] != FLAGS.procnum and my_commandset['To'] != 0 and my_commandset['To'] != "master") or \
                    my_commandset['From'] == FLAGS.procnum:
                print(
                    f"{myctl.procnum}-Commandset NG @ line {sys._getframe().f_lineno}, TO:{my_commandset['To']} From: {my_commandset['From']}  Commandset: {my_commandset}") # pylint: disable=W0212
                continue
        except KeyError:
            print(f"{myctl.procnum}-Unaddressed command block @ line {sys._getframe().f_lineno}, ignoring") # pylint: disable=W0212
            continue
        retval, myctl, FLAGS = vidproc_process_queue(my_commandset, myctl, FLAGS)
        if not retval:  # Process_queue said bad coommandblock
            print(f'{myctl.procnum}-Processqueue @ line {sys._getframe().f_lineno}, bad retval for {my_commandset}') # pylint: disable=W0212
            continue
    return (myctl, FLAGS)


def silence(ison):
    """
    Attempt to silence way too chatty tensorflow
    """
    if ison:
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(logging.ERROR)
        tf.autograph.set_verbosity(0)
        tf.logging.set_verbosity(0)
        os.environ['FFREPORT'] = 'level=0:file=/var/log/ffmpeg.log'
    else:
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.logging.set_verbosity(3)
        tf.autograph.set_verbosity(3)
        tf.logging.set_verbosity(3)
        os.environ['FFREPORT'] = 'level=3:file=/var/log/ffmpeg.log'


def getcap(cap, myctl, default):  # pylint: disable=W0613
    """
    Get value of capability cap in stream vid if available. Set to -1 if not
    """
    myval = -1
    global supplist
    if cap in supplist:
        myval = eval(f'myctl.vid.get(myctl.cv2.{cap})')
        return (myval)
    return (default)


def open_VideoWriter(myctl):
    """
    Make sure myctl.videofile exists, if not, create it and any parent dirs, and open it
    """
    checkopen(myctl.videofile, "w")
    return(myctl.cv2.VideoWriter(myctl.videofile, int(myctl.video_FourCC), myctl.video_fps, myctl.video_size))


def fileops(myctl, FLAGS, my_frame,out_pred_ext):
    """
    Video_process file operations
    """
    if myctl.is_recording:
        if myctl.out is None:  #has not been initialized
            myctl.framecounter = 0
            myctl.out = open_VideoWriter(myctl)
        if not myctl.out.isOpened():
            myctl.do_only_every = myctl.do_only_every_autovideo
            myctl.framecounter = 0
            myctl.out = open_VideoWriter(myctl )
        if FLAGS.record_framecode and (myctl.fcf is None or myctl.fcf.closed):
            myctl.fcf = checkopen(myctl.fcfile, "w", auto_open = True, buffering = 32768) #open framecode file with generous buffer
        #if FLAGS.record_framecode and myctl.fcf.closed:
        #    myctl.fcf = checkopen(myctl.fcfile, "w", auto_open = True, buffering = 32768) #open framecode file with generous buffer
        #write to the video file

        myctl.framecounter += 1
        myctl.out.write(my_frame)
        my_frame = None
        if FLAGS.record_framecode and out_pred_ext != []: #record framecode, but only if objects have been found
            myctl.fcf.write(f'{out_pred_ext},{myctl.framecounter}'+"\n")
    else:  # we aren't recording
        if myctl.out is not None:  # no action needed if not initialized
            if myctl.out.isOpened():
                myctl.do_only_every = myctl.do_only_every_initial
                #myctl.output_type = ""
                myctl.framecounter = 0
                myctl.out.release()
                if myctl.presence_counter == -1:
                    try:
                        os.remove(myctl.videofile)
                    except FileNotFoundError:
                        pass
                    myctl.presence_counter = 0


        if FLAGS.record_framecode and myctl.fcf is not None:   # no action needed if not initialized
            if not myctl.fcf.closed:
                myctl.fcf.flush()
                myctl.fcf.close()
            try:
                if os.stat(myctl.fcfile).st_size == 0: # don't leave any empty fcfiles
                    os.remove(myctl.fcfile)
            except FileNotFoundError:
                pass
    return(myctl)

def rollavg(arrlen, value):
    """simple rolling average"""
    global workarr
    if len(workarr) >= arrlen:
        workarr.pop(0) #remove oldest
    workarr.append(value)
    if len(workarr) < arrlen:
        return(-1)
    return(sum(workarr)/len(workarr))

def video_file_type(myvideo_path):
    """
    Determine type of video file, and return
    """
    if isinstance(myvideo_path,int):
        return("webcam")
    if '://' in myvideo_path:
        return('stream')
    if myvideo_path.startswith('/'):
        return('file')
    if myvideo_path.startswith('\\'):
        return('file')
    return('file')


def video_process(my_in_queue,my_out_queue,FLAGS, myevent, procnum):
    """
    Main video process. Captures video from webcam, file, or on-line stream. Runs video frames through specified YOLO model, displays and optionally stores the results.
    This process runs on its own GPU as specified in FLAGS.run_on_gpu, and/or it will run on a fraction of a GPU as specified in gpu_memory_fraction. There is no bounds checking,
    process will crash if GPU, or the total of gpu_memory_fraction per GPU are out of bounds. Each process can require around 2.5 G of resident memory, and it may crash
    ignominiously when memory-starved.
    Process will run forever unless stopped by operator, or a crashed video source.
    """
    global supplist

    myevent.clear()
    silence(FLAGS.hush)

    FLAGS.procnum = procnum  # record the passed process number
    ctl = makectl(FLAGS)  # set up a subset of FLAGS as multipurpose-object
    setattr(ctl, 'ding_timer', None)
    setattr(ctl, 'presence_timer', None)
    setattr(ctl, 'in_queue', my_in_queue)
    setattr(ctl, 'out_queue', my_out_queue)
    put_in_queue(ctl, From = procnum, To=-1, Command='status', Args={'StatusMsg': "Initializing YOLO", 'Column': 0})
    atexit.register(vidproc_shutdown, ctl, False)  # orderly shutdown, do not echo




    # check directories
    if not os.path.isdir(FLAGS.output_path):
        try:
            os.makedirs(FLAGS.output_path)
        except OSError:
            print(f'{ctl.procnum}-Error @ line {sys._getframe().f_lineno}: Cannot create directory {FLAGS.output_path}') # pylint: disable=W0212
            sys.exit()
    put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': "Staging video", 'Column': 1})


    if ctl.testing: # pylint: disable=E1101
        FLAGS.video_path = FLAGS.testvideo

    setattr(ctl, 'vid', cv2.VideoCapture(FLAGS.video_path)) # pylint: disable=E1101
    if not ctl.vid.isOpened(): # pylint: disable=E1101
        print(f'{ctl.procnum}-Error @ line {sys._getframe().f_lineno}: Cannot open video source {FLAGS.video_path}') # pylint: disable=W0212
        sys.exit()
    setattr(ctl, 'cv2', cv2)


    if not ctl.vid.isOpened() and not isinstance(FLAGS.video_path, int):  #can't open stream or vide file # pylint: disable=E1101
        put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot open stream or video file {FLAGS.video_path}",'Action': "Abandoning process"})
        sys.exit() #abandon this process
    if not ctl.vid.isOpened() and isinstance(FLAGS.video_path, int):  #can't open a webcam # pylint: disable=E1101
        if FLAGS.video_path not in FLAGS.workingcam:
            put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot access cam {FLAGS.video_path}",'Action': f"Working: {FLAGS.workingcam}"})
            sys.exit()
        else:
            put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot access cam {FLAGS.video_path}",'Action': ""})
            sys.exit()

    if FLAGS.hush:
        with suppress_stdout_stderr():
            supplist = list_supported_capture_properties(ctl.vid, ctl.cv2)  # retrieve capabilities supported by stream/cam # pylint: disable=E1101
    else:
        supplist = list_supported_capture_properties(ctl.vid, ctl.cv2)  # retrieve capabilities supported by stream/cam  # pylint: disable=E1101

    setattr(ctl, 'Frame_Height', getcap('CAP_PROP_FRAME_HEIGHT', ctl, -1)) #getcap retrieves capability if supported. Default if not
    setattr(ctl, 'Frame_Width', getcap('CAP_PROP_FRAME_WIDTH', ctl, -1))

    # adjustment values for screens other than 1280x720, on which the current OSD layout is based. App will dynamically adjust to other sizes
    if ctl.Frame_Width > -1: # pylint: disable=E1101
        setattr(ctl, 'Type_Adjust', ctl.Frame_Width / 1280)  # create adjustment factor for screens other than 1280 wide # pylint: disable=E1101
        setattr(ctl, 'Line_Adjust', ctl.Frame_Height / 720)  # create adjustment factor for screens other than 720 high # pylint: disable=E1101
        setattr(ctl, 'Type_Thickness', int(round(3 * ctl.Type_Adjust, 0)))  # also adjust thickness if needed # pylint: disable=E1101
    else:
        setattr(ctl, 'Type_Adjust', 1)  # Leave as is if no width exposed
        setattr(ctl, 'Line_Adjust', 1)  # Leave as is if no width exposed
        setattr(ctl, 'Type_Thickness', 2)  # Leave as is if no width exposed

    setattr(ctl, 'video_FourCC', ctl.cv2.VideoWriter_fourcc("m","p","4","v"))  # pylint: disable=E1101


    if isinstance(FLAGS.video_path, int) and FLAGS.buffer_depth > 10 : #assume webcam, can't have buffer > 10
        FLAGS.buffer_depth = 10

    # long list of capability tests made necessary to support webcam that won't support most of these capabilities .....
    if 'CAP_PROP_BUFFERSIZE' in supplist:
        try:
            ctl.vid.set(ctl.cv2.CAP_PROP_BUFFERSIZE, FLAGS.buffer_depth)  # set very small buffer to combat video drift # pylint: disable=E1101
        except:
            pass

    myfps, myfpsmode = getoflfps(ctl.cv2, ctl.vid, 0, 0) # pylint: disable=E1101
    setattr(ctl, 'init_fps', myfps)
    setattr(ctl, 'fps_mode', myfpsmode)
    if ctl.fps_mode == 0: # pylint: disable=E1101
        preroll(ctl.vid, 24) # pylint: disable=E1101


    setattr(ctl, 'video_size', (
        int(ctl.vid.get(ctl.cv2.CAP_PROP_FRAME_WIDTH)), # pylint: disable=E1101
        int(ctl.vid.get(ctl.cv2.CAP_PROP_FRAME_HEIGHT)),)) # pylint: disable=E1101

    #keytype = -1
    osdstr = "FPS: ??"
    #time.sleep(2)
    prev_time = timer()

    #do_only_every = FLAGS.do_only_every_initial  # set to one for every frame, 2 for every 2nd etc
    curtype = ""
    setattr(ctl, 'init_frames', getcap('CAP_PROP_POS_FRAMES', ctl, -1))
    setattr(ctl, 'init_msec', getcap('CAP_PROP_POS_MSEC', ctl, -1))
    #setattr(ctl, 'on_start_fps', getcap('CAP_PROP_POS_FPS', ctl, -1))
    setattr(ctl, 'do_only_every', FLAGS.do_only_every_initial)

    ctl.ding_timer = RetriggerableTimer(0, ding_timeout)  # initialize, but don't start
    ctl.presence_timer = RetriggerableTimer(0, presence_timeout) # initialize, but don't start
    my_start_time = time.time()
    ctl.framecounter = itercounter = accum_time = curr_fps = sec_counter = 0
    x = 1

    ctl.out = ctl.fcf = None
    #Result log file
    if FLAGS.maintain_result_log:  # open the log file
        rslfile=f'{FLAGS.result_log_basedir}/{datetime.datetime.now().strftime("%y%m%d")}/P{ctl.procnum}/{datetime.datetime.now().strftime("%y%m%d_%H%M%S").strip()}_P{ctl.procnum}.csv'
        setattr(ctl, 'rsf',checkopen(rslfile, "w", auto_open = True, buffering = 32768)) #open running log file with generous buffer
        ctl.rsf.write("process,time,ytime,rolltime,rollfps,objects,n,gpu,gpufract,allowgrowth,infps,outfps,waitkeytime,adj,out_pred\n\n") # pylint: disable=E1101

    if FLAGS.sync_start:  # stage video and wait for common start signal
        put_in_queue(ctl, From = ctl.procnum, To=-1, Command='videoready', Args={})  # tell master we are ready
        put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': "Ready", 'Column': 2})
        event_set = myevent.wait(FLAGS.sync_start_wait)  # now wait for event
        if event_set:
            pass
        else:
            print(f"{ctl.procnum}-Time out, moving ahead without event...")

    retval, frame = ctl.vid.read() # pylint: disable=E1101
    #Assume start at this point. Build offsets for CAP_PROP_POS_FRAMES and POS_MSEC, relative to this point
    msec_offset = getcap('CAP_PROP_POS_MSEC', ctl, -1)
    pos_offset  = getcap('CAP_PROP_POS_FRAMES', ctl, -1)

    if msec_offset == -1:
        init_msec = -1
    else:
        init_msec = 0

    if pos_offset == -1:
        init_pos = -1
    else:
        init_pos = 0

    put_in_queue(ctl, From = ctl.procnum, To=-1, Command='initstats', Args={'InitPOS_FRAMES': init_pos,'InitPOS_MSEC': init_msec,'InitStrFPS': ctl.init_fps,'InitTimeStamp': datetime.datetime.timestamp(datetime.datetime.now()), 'run_on_gpu': ctl.run_on_gpu,'gpu_memory_fraction': ctl.gpu_memory_fraction}) # pylint: disable=E1101
    if FLAGS.profile:  # run profiler
        print(f"{ctl.procnum} - Profiler activated.")
        pr = cProfile.Profile()
        pr.enable()

    global workarr #used for rolling average
    workarr =[]
    infps = roll = rollfps = 0

    file_type = video_file_type(FLAGS.video_path) # determine type


    if file_type == 'file'  and ctl.init_fps > 0: # pylint: disable=E1101
        waitkeytime = int((1/ctl.init_fps)*1000)  # pylint: disable=E1101
        do_adjust = True
    else:
        waitkeytime = 1
        do_adjust = False

    #if ctl.procnum == 3:
    #    sys.exit()

    while ctl.vid.isOpened():  # pylint: disable=E1101
        mystr_1 = mystr_2 = ""
        ctl,FLAGS = vidproc_checkqueue(ctl, FLAGS)  # check the queue, and act on it if necessary


        # work the video
        retval, frame = ctl.vid.read()
        if not retval:  # bogus return, try again
            continue
        frame = frame[:, :, :: 1]

        if itercounter % ctl.do_only_every == 0 or ctl.do_only_every == -1:  # do only every n frame, -1 is each
            curtype = curlabel = ""
            image = Image.fromarray(frame)
            _ , image, elapsed_time, out_pred_ext = ctl.yolo.detect_image_extended(image, show_stats=FLAGS.show_stats)
            #build rolling average
            roll=rollavg(FLAGS.rolling_average_n,elapsed_time)
            if roll > 0:
                rollfps = round(1/roll,1)
            else:
                rollfps = 0
            result = np.asarray(image)
        else:
            result = frame

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1


        if accum_time > 1:  #happens each second
            ctl.video_fps = curr_fps  # set output stream to the real fps this is running at
            accum_time = accum_time - 1
            if rollfps > 0:
                myyfps = str(rollfps)
            else:
                myyfps = "---"

            osdstr = "IFPS: " + str(round(infps,1))+ " YFPS: " + myyfps + " OFPS: " + str(round(ctl.video_fps,1)) +  " RT: " + str(round(roll,3))
            curr_fps = 0
            # send to master
            posframes = getcap('CAP_PROP_POS_FRAMES', ctl, -1)
            if posframes != -1:
                posframes = posframes - pos_offset
            posmsec = getcap('CAP_PROP_POS_MSEC', ctl, -1)
            if posmsec != -1:
                posmsec = posmsec - msec_offset


            if ctl.fps_mode == 0: #if PROP_FPS works, take it continuously
                cur_infps = getcap('CAP_PROP_FPS', ctl, -1)
            if ctl.fps_mode == 1: #rely on POS_FRAMES/POS_MSEC, provides average only
                cur_infps =  round((ctl.vid.get(cv2.CAP_PROP_POS_FRAMES) - ctl.init_frames) / (ctl.vid.get(cv2.CAP_PROP_POS_MSEC) - ctl.init_msec) * 1000, 1) # pylint: disable=E1101
            if ctl.fps_mode == 2: #Can't do timing loop each time, so rely on what was measured at start of stream
                cur_infps = ctl.init_fps

            put_in_queue(ctl, From = ctl.procnum, To=-1, Command='stats',
                         Args={'POS_FRAMES': getcap('CAP_PROP_POS_FRAMES', ctl, -1) -pos_offset,
                               'POS_MSEC': getcap('CAP_PROP_POS_MSEC', ctl, -1) - msec_offset,
                               'MaxYoloFPS': rollfps,
                               'InFPS': cur_infps,
                               'StrFPS': ctl.video_fps,
                               'DoOnlyEvery' : ctl.do_only_every,
                               'TimeStamp': datetime.datetime.timestamp(datetime.datetime.now())}
                               )
            sec_counter += 1
            if FLAGS.maintain_result_log and  sec_counter > 60 :
                ctl.rsf.flush() # flush every 60 secs
                sec_counter = 0

        #update incoming FPS. Mode was set when setting up stream
        if ctl.fps_mode == 0: #if PROP_FPS works, take it continuously
            infps = getcap('CAP_PROP_FPS', ctl, -1)
        if ctl.fps_mode == 1: #rely on POS_FRAMES/POS_MSEC, provides average only
            infps = round((ctl.vid.get(cv2.CAP_PROP_POS_FRAMES) - ctl.init_frames) / (ctl.vid.get(cv2.CAP_PROP_POS_MSEC) - ctl.init_msec) * 1000, 1) # pylint: disable=E1101
        if ctl.fps_mode == 2: #Can't do timing loop each time, so rely on what was measured at start of stream
            infps = ctl.init_fps
        outfps = ctl.video_fps

        #governor - adjust waitkey time to match incoming fps
        if do_adjust:
            diff = abs(int(outfps)-int(infps))
            adj = 1
            if diff > 5:
                adj = 20
            if diff > 3:
                adj = 10
            if int(outfps) < int(infps) and waitkeytime > adj:
                #waitkeytime = waitkeytime - adj
                adj = adj * -1
            if int(outfps) == int(infps):
                adj = 0
            if waitkeytime + adj > 0:
                waitkeytime = waitkeytime + adj
        else:
            adj = 0


        if FLAGS.maintain_result_log and roll > 0: #write to result log, but only if we have a valid rolling average
            ctl.rsf.write(f'{ctl.procnum},\'{datetime.datetime.now().strftime("%H-%M-%S_%f")}\',{round(elapsed_time,5)},{round(roll,5)},{round(1/roll,1)},{len(out_pred_ext)},{ctl.do_only_every},{FLAGS.run_on_gpu},{FLAGS.gpu_memory_fraction},{FLAGS.allow_growth},{infps},{ctl.video_fps}, {waitkeytime},{adj},{out_pred_ext}\n')

        mystr_1 =  osdstr  + " n=%s " % str(ctl.do_only_every)
        if ctl.record_autovideo:

            mystr_1 = mystr_1 + "Auto"
        if ctl.is_recording:
            mystr_1 = mystr_1 + " *REC*"
        else:
            mystr_1 = mystr_1 + ""


        if out_pred_ext != []:

            #BS Special __ find out if y < 80 ##############
            trigger_OK = True
            for pred in out_pred_ext:
                if pred[1] < 72:
                    trigger_OK = False
            ################################################


            labstring = ""
            curtype = curlabel = ""

            lablist = []
            #extract labels
            for pred in out_pred_ext:
                lablist.append(pred[4])

            #now we have all possible labels in lablist
            for curlabel in lablist:
                curtype = FLAGS.labeldict.get(curlabel,'notype')

                if curtype in FLAGS.presence_trigger:  # check the access list

                    if FLAGS.soundalert:
                        if not ctl.ding_timer.is_alive():
                            playsound(FLAGS)
                            ctl.ding_timer = RetriggerableTimer(FLAGS.ding_interval, ding_timeout)  # set-up the timer
                            ctl.ding_timer.start()
                        else:
                            ctl.ding_timer.reset()
                    ctl.presence_counter += 1
                    if not ctl.presence_timer.is_alive():
                        if ctl.record_autovideo and trigger_OK:
                            ctl = start_recording(ctl, putqueue=True) #if putqueue true, cause all processes to record
                            ctl.autovideo_running = True
                        ctl.presence_timer = RetriggerableTimer(FLAGS.presence_timer_interval,
                                                            presence_timeout)  # set-up the timer
                        ctl.presence_timer.start()
                    else:
                        ctl.presence_timer.reset(FLAGS.presence_timer_interval)

                labelstring = labstring + curlabel + " "
                mystr_2 = mystr_2 + curtype.capitalize() + " " + labelstring

        osdkwarg = {"fontFace": 0, "fontScale": 1.2 * ctl.Type_Adjust, "color": (0, 255, 0),
                    "thickness": int(ctl.Type_Thickness)}


        ctl = fileops(ctl,FLAGS, result,out_pred_ext) #save frame to file, but without OSD

        if ctl.osd: #only show OSD if enabled
            result = ctl.cv2.putText(result, text=mystr_1, org=(20, int(ctl.video_size[1] - 20 * ctl.Line_Adjust)   ),**osdkwarg   )
            result = ctl.cv2.putText(result, text=mystr_2, org=(20, int(ctl.video_size[1] - 65 * ctl.Line_Adjust)),**osdkwarg)
        ctl.cv2.namedWindow(FLAGS.window_title, ctl.cv2.WINDOW_NORMAL)
        ctl.cv2.moveWindow(FLAGS.window_title, FLAGS.window_x, FLAGS.window_y)
        ctl.cv2.resizeWindow(FLAGS.window_title, FLAGS.window_wide, FLAGS.window_high)
        ctl.cv2.imshow(FLAGS.window_title, result) #show the frame

        _ = ctl.cv2.waitKey(waitkeytime)

        #check for autovideo no longer running
        if not ctl.presence_timer.is_alive() and ctl.autovideo_running:  #we need to switch it off
            ctl = stop_recording(ctl, putqueue=True, delete= ctl.presence_counter < 5)
            ctl.presence_counter = 0

        itercounter += 1
        if (time.time() - my_start_time) > x:
            my_start_time = time.time()

        if itercounter > 1000000:
            itercounter = 0

        if itercounter == 3000 and FLAGS.profile:
            break

    if FLAGS.profile:
        #actions if profiler enabled
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        proffile = getbasefile()+f"_{ctl.procnum}.prof"
        ps.print_stats()
        logout=f'Proc {ctl.procnum} {s.getvalue()}'
        with open(proffile, "a+") as f:
            f.write(logout)
        print(logout)
        time.sleep(20) #wait to give all processes a chance to save
        print(f"{ctl.procnum} - Profiler deactivated.")

    #should never get here, just in case
    vidproc_shutdown(ctl, True)  # shutdown, and echo to others
    try:
        ctl.rsf.close()
    except:
        pass


def put_in_queue(myctl,From=0, To=0, Command='', Args=None):
    """
    Put command and args in queue, send to To
    """
    if Args is None:
        Args = {}
    my_commandset = {'To': To, 'From': From , 'Command': Command, 'Args': Args}

    #if sent not from master, send to master for distribution
    if myctl.procnum != -1:
        which_queue = myctl.out_queue
        which_queue.put(my_commandset)
    else:
        #if sent from master, distribute immediately
        myctl = process_master_commandset(myctl, my_commandset)
    return(myctl)

def vidproc_process_queue(my_commandset, myctl, FLAGS):
    """
    This processes a command block coming from master into video process
    returns retval True/False (False = error) and myctl
    """
    if not isinstance(my_commandset, dict):
        print(f"{myctl.procnum}-Queue sent bad command block @ line {sys._getframe().f_lineno}, ignoring") # pylint: disable=W0212
        return (False, myctl, FLAGS)
    try:
        mycommand = my_commandset['Command'].strip()
        myarglist = my_commandset['Args']
    except KeyError:
        print(f"{myctl.procnum}-Queue sent bad command block @ line {sys._getframe().f_lineno}, ignoring") # pylint: disable=W0212
        return (False, myctl, FLAGS)
    # work the commands, set putqueue=False to prevent cascading queues
    if mycommand == "shutdown":
        myctl = stop_recording(myctl, putqueue=False, delete=False)
        put_in_queue(myctl,  From = myctl.procnum, To=-1, Command='ackshutdown', Args={})
        time.sleep(1)
        vidproc_shutdown(myctl, False)
        sys.exit()
    if mycommand == "startrecording":
        myctl = start_recording(myctl, putqueue=False)
        return (True, myctl, FLAGS)
    if mycommand == "stoprecording":
        myctl = stop_recording(myctl, putqueue=False, delete=myarglist.get("Delete",False))
        return (True, myctl, FLAGS)

    if mycommand == "autovidoff":
        myctl = stop_recording(myctl, putqueue=False, delete=False)
        myctl.record_autovideo = False
        FLAGS.record_autovideo = False
        return (True, myctl, FLAGS)

    if mycommand == "autovidon":
        myctl.record_autovideo = True
        FLAGS.record_autovideo = True
        return (True, myctl, FLAGS)

    if mycommand == "abortvideowriter":
        myctl.is_recording = False
        if myctl.out is not None and myctl.out.isOpened():
            myctl.out.release()
        return (True, myctl, FLAGS)
    if mycommand == "audioon":
        FLAGS.soundalert = True
        playsound(FLAGS)
        return (True, myctl, FLAGS)
    if mycommand == "audiooff":
        FLAGS.soundalert = False
        return (True, myctl, FLAGS)
    if mycommand == "queuetest":
        print(f"{myctl.procnum}-Would execute queuetest")
    if mycommand == "windowredraw":
        myctl.cv2.moveWindow(FLAGS.window_title, FLAGS.window_x, FLAGS.window_y - 50)
    return (True, myctl, FLAGS)

def process_master_commands(my_commandset, mctl):
    """
    Act on commands sent to master by video_processes
    """
    mycommand = my_commandset.get('Command','').strip()
    myarglist = my_commandset.get('Args',{})
    myproc = my_commandset.get('From', None)
    iam = str(myproc)
    if mycommand == "procdied": # We've lost a process
        procname = myarglist.get('name','')
        procnumber = int(myarglist.get('number',-1))
        if 0 >= procnumber > mctl.procnum: #Bogus
            print(f"Error @ line {sys._getframe().f_lineno} in dead process report, reported procnum: {procnumber}. Resuming, but taking no action") # pylint: disable=W0212
            return(mctl) # Do nothing
        myrec = mctl.procdict.get(procnumber,[])
        if myrec == []:
            print(f'Procdict err @ line {sys._getframe().f_lineno}. Key {procnumber} not found in procdict {mctl.procdict}. Resuming, but taking no action') # pylint: disable=W0212
            return(mctl) # Do nothing
        if not myrec[5]: #Already set
            return(mctl) # Do nothing
        myrec[5] = False
        mctl.procdict[procnumber] = myrec  #Update myrec in procdict
        #decrement counters to reflect the lost process
        mctl.shutdown_counter -= 1
        mctl.videoready_counter -= 1
        mctl.initready_counter -= 1
        #Announce the death of the process
        mctl = do_dataline(mctl, procnumber, emerg = "Process defunct")
        #Ack deathj
        put_in_queue(mctl,From = -1, To=-2, Command='ackprocdied', Args={'name':procname,'number':procnumber})
        return(mctl)


    if mycommand == "error"  and mctl.showmaster:
        message = f"Err in #{iam}: {myarglist['ErrMsg']}"
        action = myarglist['Action']
        message = message+" "+ action
        settext(mctl.window,message,mctl.next_statusline,1)
        mctl.window.update()
        mctl.next_statusline = mctl.next_statusline +1
        time.sleep(10)
        return(mctl)

    if mycommand == "status" and mctl.showmaster:
        mymsg = myarglist.get('StatusMsg', '')
        mycol = myarglist.get('Column', -1)

        if mycol > -1: #We are doing a grid
            if mctl.gridtop == 0: # we have first grid msg, save the topmost line
                mctl.gridtop = mctl.next_statusline

            if mycol == 0: #Tag on a proc number
                mymsg = f"Proc#{str(iam).rjust(3)}: {mymsg}"
                showcol = mycol + 1
            else:
                showcol = 0
                for i in range(0,mycol):
                    showcol = showcol + mctl.coldict.get(i,0)
                if mycol >= 2:
                    showcol = showcol + 15
                mymsg = "   " + mymsg
            settext(mctl.window,mymsg,mctl.gridtop+myproc,showcol)
            if mctl.coldict.get(mycol,-1) < len(mymsg):
                mctl.coldict[mycol] = len(mymsg)
        else:
            message = f"Proc#{str(iam).rjust(3)}: {mymsg}"
            settext(mctl.window,message,mctl.next_statusline,1)
            mctl.next_statusline = mctl.next_statusline +1

        mctl.window.update()
        time.sleep(0.1)
        return(mctl)

    if mycommand == "initstats"  and mctl.showmaster:
        mctl.initdict[iam] = myarglist
        mctl.initready_counter -= 1
        if mctl.initready_counter == 0:  # we have all initial stats
            mctl = do_masterscreen(mctl)
        return (mctl)

    if mycommand == "stats"  and mctl.showmaster:
        mctl.statsdict[iam] = myarglist
        mctl = do_dataline(mctl, iam)  # from now on, write single lines
        return(mctl)


    if mycommand == "pickle":
        with open("/usr/local/bin/catdetectorp/TrainYourOwnYOLO/3_Inference/mctl.pkl", "wb") as f:
            pickle.dump(mctl, f, protocol=pickle.HIGHEST_PROTOCOL)
        return (mctl)
    if mycommand == "videoready":
        if not mctl.sync_start:
            mctl.event.set()
        else:
            mctl.videoready_counter -= 1  # decrement until we reach 0, meaning all processes ready
            if mctl.videoready_counter == 0:
                mctl.event.set()
        return (mctl)

    if mycommand == "ackshutdown":  #Wait for all video_processes signaling shutdown, then shutdown master
        mctl.shutdown_counter -= 1  # decrement until we reach 0, meaning all processes ready
        if mctl.shutdown_counter == 0:
            if mctl.shutdown_action == -1: # master shutdown
                master_shutdown(mctl)
            if mctl.shutdown_action == 1: # resurrect
                resurrect()
        return (mctl)
    return (mctl)


def converttype(myval):
    """
    Massage conf file settings
    """
    # check for string
    myval = myval.strip()  # cleanup
    if myval.startswith('"') and myval.endswith('"'):
        return (myval.strip('"'))  # assume string
    if myval.startswith("'") and myval.endswith("'"):
        return (myval.strip("'"))  # assume string
    if (myval.startswith("{") and myval.endswith("}")) or (myval.startswith("[") and myval.endswith("]")) or (
            myval.startswith("(") and myval.endswith(")")):
        return (ast.literal_eval(myval))  # assume dict, list, tuple
    if (myval.startswith("{") and not myval.endswith("}")) or (not myval.startswith("{") and myval.endswith("}")) or (
            myval.startswith("[") and not myval.endswith("]")) or (
            not myval.startswith("[") and myval.endswith("]")) or (
            myval.startswith("(") and not myval.endswith(")")) or (not myval.startswith("(") and myval.endswith(")")):
        print(f"Error in config file @ line {sys._getframe().f_lineno}, unbalaced paren in {myval}") # pylint: disable=W0212
        sys.exit()

    if myval.isdecimal():
        return (int(myval))  # assume integer

    return (ast.literal_eval(myval))

def getbasefile():
    """
    returns path to current app with ".py" stripped off
    """
    basefile = __file__  # get path to app
    if basefile.endswith('.py'):
        basefile = basefile[:-3]
    return(basefile)

def makeconfig():
    """
    This provides the settings for the startup module, the master module, and the individual processes
    For this, we use a config file with the same app name and the ".conf" extension, situated in the app directory
    If there is no config file, the app will default to one process using the test video file and the common settings/directories of TrainYourOwnYolo
    """
    # convert config file into blk_Startup, blk_Master, blk_Common, and blk_Procdict objects

    confpath = getbasefile() + '.conf'
    #assert os.path.exists(confpath), f'Cannot find conf file {confpath}'
    if os.path.exists(confpath):

        #if there is a config file, update defaults from config
        with open(confpath, 'r') as f:
            conf = f.read()
        confl = conf.splitlines()  # read the file into memory, turned into a list
        has_config = 1 #Config file good, so far
    else:
        has_config = -1 #Tell Master we are working without config file
        confl = []


    myblk_Startup, myblk_Startup_OK = getblock("Startup:", confl,
                       DefStartup())  # add all entries in Startup block of the confl list to an (empty) CTL object, and store as blk_Startup
    myblk_Master, myblk_Master_OK = getblock("Master:", confl,
                      DefMaster())  # add all entries in Master block of the confl list to an (empty) CTL object, and store as blk_Master

    myblk_Common, myblk_Common_OK = getblock("Common:", confl,
                      DefCommon())  # add all entries in Common block of the confl list to an (empty) CTL object, and store as blk_Common

    #print(f'myblk_Startup_OK: {myblk_Startup_OK} myblk_Master_OK {myblk_Master_OK} myblk_Common_OK {myblk_Common_OK}')

    if has_config > -1:
        if not myblk_Startup_OK and not myblk_Master_OK  and not myblk_Common_OK:
            setattr(myblk_Master,'has_config', -2) #Config file empty
        else:
            setattr(myblk_Master,'has_config', -3) #Config file partial
        if myblk_Startup_OK and myblk_Master_OK  and myblk_Common_OK:
            setattr(myblk_Master,'has_config', 1) #Good config file
    else:
        setattr(myblk_Master,'has_config', -1) #No config file

    #print(f'has_config: {myblk_Master.has_config}')

    #exit()

    #(myblk_Common,'hush',getattr(myblk_Common,'hush',False)) #If hush is not set, make it False
    if myblk_Common.hush: #try to silence chatty tensorflow
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['FFREPORT'] = 'level=0:file=/var/log/ffmpeg.log'
    silence(myblk_Common.hush)
    setattr(myblk_Master, "hush", myblk_Common.hush)
    if myblk_Common.soundalert:  # test audio output if sound requested
        if not soundtest(myblk_Common):  # if output fails, disable sound
            print("No audio output found, disabling chime")
            myblk_Common.soundalert = False #This disables soundalert in all processes
            setattr(myblk_Master, "soundavailable", False)
        else:
            setattr(myblk_Master, "soundavailable", True)
    if hasattr(myblk_Common, 'sync_start'):
        if myblk_Startup.num_processes == 1 and myblk_Common.sync_start:  # Ignore sync start if only 1 process
            myblk_Common.sync_start = False
        setattr(myblk_Master, "sync_start", myblk_Common.sync_start)
        setattr(myblk_Common, "sync_start_wait", myblk_Common.sync_start_wait * myblk_Startup.num_processes)  #wait sync_start_wait multiplied by num_processes for sync start
    else:
        setattr(myblk_Master, "sync_start", False)
        setattr(myblk_Common, "sync_start", False)
    if myblk_Common.hush: #suppress messages when hushed
        with suppress_stdout_stderr():
            avlcam, workingcam = listcams(myblk_Common.max_cams_to_probe)
    else:
        avlcam, workingcam = listcams(myblk_Common.max_cams_to_probe)
    setattr(myblk_Common,'avlcam',avlcam) # get available usb/internal camera indices
    setattr(myblk_Common,'workingcam',workingcam) # get working usb/internal camera indices
    #grab a few Common settings to be used by Master
    setattr(myblk_Master, 'record_autovideo', myblk_Common.record_autovideo)
    setattr(myblk_Master, 'showmaster', myblk_Common.showmaster )
    setattr(myblk_Master, 'soundalert', myblk_Common.soundalert)
    if not hasattr(myblk_Common, 'testvideo') or myblk_Common.testvideo == "":
        setattr(myblk_Common,'testvideo', resource_path + "/testvid.mp4")
    if not hasattr(myblk_Common, 'clicksound') or myblk_Common.clicksound== "":
        setattr(myblk_Common,'clicksound', resource_path + "/chime.mp3")
    if not hasattr(myblk_Common, 'silence') or myblk_Common.silence == "":
        setattr(myblk_Common,'silence', resource_path + "/silence.mp3")
    myblk_Procdict = getprocdict(confl, myblk_Common,
                               myblk_Startup.num_processes)  # get dict of Process  blocks, each added to properties in blk_common dict is {"procnum": block object}

    return (myblk_Startup, myblk_Master, myblk_Procdict)

def find_between(mystart,myend,mystring):
    """Return string between mystart and myend. Empty if mystring  won't start with mystart or end with myend"""
    if not mystring.startswith(mystart):
        return('')
    if not mystring.endswith(myend):
        return('')
    return(mystring[mystring.find(mystart)+len(mystart):mystring.rfind(myend)])

def getblock(blockname, confl, addobj):
    """
    Get config block specified in blockname from config list confl, add to blockobj, and return blockobj
    """
    try: #confl is the whole config file, split into a list
        i = confl.index(blockname) # get the position of the block we are looking for
    except ValueError:
        #print(f"'{blockname}' not specified in config file @ line {sys._getframe().f_lineno}") # pylint: disable=W0212
        return(addobj,False) # return the default block, no success
        #sys.exit()
    except NameError:
        print(f"Config file failed @ line {sys._getframe().f_lineno}") # pylint: disable=W0212
        return(addobj,False) # return the default block, no success

    blockobj = copy.deepcopy(addobj)
    while True:
        theprop = ""
        i += 1
        if i >= len(confl):  # end of file!
            return (blockobj,True)
        if confl[i].strip().startswith('#') or ':' not in confl[i]:  # skip comment, or line without :, therefore also empty line
            continue
        sl = confl[i].strip().split('#', 1)[0]  # chop off trailing comment
        li = sl.strip().split(':', 1)  # split at colon, one split
        if len(li) < 2:  ##must be a NEW block identifier no arg, we are done
            return (blockobj,True)
        theprop = li[0].strip()
        theval = li[1].strip()
        if theval.strip() == "":  ##must be a NEW block identifier no arg, we are done
            return (blockobj,True)
        cval = converttype(theval)  # convert the type
        setattr(blockobj, theprop, cval)  # .. and add to the blockobject, blockobject.theprop = cval
    return (blockobj,True)  # just to make sure, should never see this


def getprocdict(confl, blk_Common, num_proc):
    """
    get dict of numproc process blocks, each added to properties in blk_common dict is {"procnum": block object}.
    get config block specified in blockname from config list confl, add to blockobj, and return blockobj
    """
    my_Procdict = {}

    procsfound = [i for i in confl if i.startswith('Process_')] # list all Process_ blocks

    for procblk in procsfound: #go through all Process_ blocks
        #procblk is "Process_1"  etc. Now find the 1
        #print("\n\n"+procblk)
        try:
            j = int(find_between('Process_',':',procblk))
            #if j == 4:
            #    continue
        except ValueError: # procblk failed
            #print(f"Procblk {procblk} failed")
            continue
        #print(procblk)
        try:
            del myblock
        except:
            pass
        myblock, okblock = getblock(procblk, confl, blk_Common)  # .... add result to the dict
        if not okblock:
            continue
        my_Procdict[str(j)] = copy.deepcopy(myblock)


    for k in range(1, int(num_proc) + 1):  #go through all pending processes
        if str(k) not in my_Procdict.keys():
            my_Procdict[str(k)] = copy.deepcopy(blk_Common)

    return (my_Procdict)




def master(num_processes, myprocdict, myevent, mctl):
    """
    Central hub used to distribute messages between processes, collect stats etc.
    """
    silence(mctl.hush)
    #setattr(mctl, "main_in_queue",main_in_queue)
    #setattr(mctl, "main_out_queue",main_out_queue)
    #setattr(mctl, 'in_queue', my_in_queue)
    #setattr(mctl, 'out_queue', my_out_queue
    setattr(mctl, 'procnum',-1)  #set procnum -1, because master
    setattr(mctl, 'devdict',get_device_dict())  #get GPU devices
    setattr(mctl, 'num_gpu', len(mctl.devdict)) #number of GPUs is number of device settings in devdict
    totallines = num_processes + 5 + mctl.num_gpu + 5
    setattr(mctl, "event", myevent)  # event to use for video ready
    setattr(mctl, "videoready_counter", num_processes)  # used to count down
    setattr(mctl, "initready_counter", num_processes)  # used to count down until init ready
    setattr(mctl, "shutdown_counter", num_processes)  # used to count down to monitor process shutdown
    setattr(mctl, "shutdown_action", None)  # placeholder, shutdown action. -1 = shutdown, +1 = resurrect

    setattr(mctl, "totallines", totallines)  # total lines master display
    setattr(mctl, "topdataline", 0)  # position of topmost data line master display
    setattr(mctl, "datalinedict", {})  # holds the dataline objects in use
    setattr(mctl, "masterwin_x", 980)  # width master display
    setattr(mctl, "masterwin_y", (mctl.totallines + 2) * 20)  # height master display
    setattr(mctl, 'maxline', 82)  # maximum number of characters in master display line
    setattr(mctl, 'procdict', myprocdict) # store procdict
    setattr(mctl, 'centertext', mctl.masterwin_x / 2)

    window = tkinter.Tk() # setup our master window
    #get monitor dimensions
    tk_height = window.winfo_screenheight()
    #print(f'Screen = {window.winfo_screenwidth()}x{window.winfo_screenheight()}')
    windypos = tk_height - mctl.masterwin_y
    window.title(mctl.master_window_title)
    window.geometry(f"{mctl.masterwin_x}x{mctl.masterwin_y}")  # size the window
    window.geometry(f"+40+{windypos}")  # size the window
    setrows(window,totallines) # set up the grid we'll be using
    setattr(mctl, 'window',window)
    setattr(mctl, 'initdict', {})
    setattr(mctl, 'statsdict', {})
    setattr(mctl, 'num_processes', num_processes)
    if mctl.showmaster:
        settext(window,"Initializing", 1, 1, font = "Arial" , bold = True, size = 24)
        window.update()
        if mctl.has_config == 1:
            settext(window,"Please wait, spooling up ...", 2, 1, font = "Arial" , bold = False, size = 14)
        if mctl.has_config == -1:
            settext(window,"No config file, working with defaults", 2, 1, font = "Arial" , bold = False, size = 14)
        if mctl.has_config == -2:
            settext(window,"Config file empty, working with defaults", 2, 1, font = "Arial" , bold = False, size = 14)
        if mctl.has_config == -3:
            settext(window,"Partial config file, working with defaults", 2, 1, font = "Arial" , bold = False, size = 14)
        setattr(mctl, 'next_statusline', 3) # set line for next status message
        window.update()
        if mctl.has_config < 1:
            time.sleep(10)

    while True:
        """
        This monitors the Master in-queue for commandsets sent by processes
        Commansets are validated on input, and discarded if invalid
        A commandset addressed to a specific video-process (To = x) is forwarded to that process
        A commandset addressed to all (To = 0) is forwarded to all video-processes except the sending
        A commandset addressed to Master (To = -1) is acted upon by the master process
        A commandset addressed to Master (To = -2) is forwarded to Main
        Slots < -2 are reserved for future expansion """  # pylint: disable=W0105

        for key in mctl.procdict: #It's a dict
            myset = mctl.procdict.get(key,[])
            if myset[5]:  #Process dead or alive. Don't interrogate dead process
                try:
                    my_commandset = myset[1].get_nowait()  #myset[1] is the in_queue
                except queue.Empty:
                    continue
                if not validate_commandset(my_commandset,
                                           num_processes):  # perform tests, returns True for good, False for bad
                    continue
                # now we have a good commandset, operate on To (all parameters validated, no further checks necessary)
                # check whether one process is signaling the others to stop recording. Set the Record button back to record,
                # in case it was set to stop recording

                if my_commandset != {}:

                    if my_commandset['Command'].strip() == "stoprecording":
                        stoprecord_all(mctl, doqueue = False)
                    #same for startrecording
                    if my_commandset['Command'].strip() == "startrecording":
                        record_all(mctl, doqueue = False)

                    mctl = process_master_commandset(mctl, my_commandset)

        if mctl.showmaster:
            mctl.window.update()
        time.sleep(0.005)  # waste a little time





def process_master_commandset(mctlx, my_commandset):
    """
    This distributes commands queued to master to their final destination(s)
    Command sets sent top master will stay in maaster, all others are passed on to their intended processes
    """
    if my_commandset['To'] == 0:  # send the received commandset out to all processes, except the one that sent it
        send_to_all(my_commandset, mctlx.procdict)
        return(mctlx)
    if my_commandset['To'] > 0 or my_commandset['To'] < -1:  # Single recipient, send to
        mctlx.procdict[my_commandset['To']][1].put(my_commandset)  # put the commandset into the queue indexed by To
        return(mctlx)
    # at this point, it must be addressed to master (< 0) but check to make sure
    assert my_commandset['To'] < 0, "Failure in Master recipient logic"
    mctlx = process_master_commands(my_commandset, mctlx)
    return(mctlx)


def validate_commandset(my_commandset, num_processes):
    """
    Perform sanity checks on commandset, return True if good, False if bad
    """
    if not isinstance(my_commandset,dict):
        print(f'Commandset {my_commandset} must be dict, is {type(my_commandset)} @ line {sys._getframe().f_lineno}') # pylint: disable=W0212
        return (False)
    try:
        if not isinstance(my_commandset['To'], int) or not isinstance(my_commandset['From'], int):
            print(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno} check To and From') # pylint: disable=W0212
            return (False)
    except KeyError:
        print(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno} check To and From, key failed') # pylint: disable=W0212
        return (False)
    if my_commandset['To'] > num_processes or my_commandset['From'] > num_processes:
        print(f"Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, To {my_commandset['To']} or From {my_commandset['From']} out of range") # pylint: disable=W0212
        return (False)
    try:
        if not isinstance(my_commandset['Command'], str)  or my_commandset['Command'] == "":
            print(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, check Command') # pylint: disable=W0212
            return (False)
    except KeyError:
        print(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, check Command, key failed') # pylint: disable=W0212
        return (False)
    if not my_commandset['Command'].isalpha():
        print(f"Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, Command {my_commandset['Command']} must be letters only") # pylint: disable=W0212
        return (False)
    if 'Args' in my_commandset:
        try:
            if not isinstance(my_commandset['Args'],dict):
                print(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, check Args, must be dict') # pylint: disable=W0212
                return (False)
        except KeyError:
            print(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, Args failed test') # pylint: disable=W0212
            return (False)
    return (True)


def send_to_all(my_commandset, myprocdict):
    """
    Send same message to all processes in procdict
    """
    for key in myprocdict: #
        myset = myprocdict.get(key,[])
        if myset[2] != my_commandset['From'] and myset[5]:  # Myset[2] holds the process identifier. Do not send back to sender. Don't send to dead process myset[5]
            myset[0].put(my_commandset)  # myset[0] is the in_queue of the process
    return ()


if __name__ == "__main__":
    """
    Main routine.
    Sets up Master and Video processes.
    Then gets out of the way, except for monitoring that processes have started alright.
    If a video process won't start, or if it dies, there will be a notification on the master screen, and the
    rest of the processes will continue.
    If the master process won't start, or if it dies, execution will be halted.
    """  # pylint: disable=W0105

    fp = open(pid_file, 'w')
    ### Check for running instance
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another instance running.")
        sys.exit(0)
    atexit.register(startup_shutdown)  # orderly shutdown
    writethelog("Startup")
    print("MultiDetect start")

    blk_Startup, blk_Master, blk_Procdict = makeconfig()
    # blk_Startup has the settings for this section
    # blk_Master has the settings for the Master process
    # blk_Procdict has one set of settings for each process, like {"1": FLAGS} and so forth
    assert blk_Startup.num_processes > 0, "No suitable num_processes in Startup block of config file. Aborting"

    m_in_queue = multiprocessing.Queue()
    m_out_queue = multiprocessing.Queue()

    mainctl = CTL()  # Create short ctl object
    setattr(mainctl, 'procnum', -2)  #Main process is -2
    setattr(mainctl, 'in_queue', m_in_queue)
    setattr(mainctl, 'out_queue', m_out_queue)

    processes = blk_Startup.num_processes

    #print("\n"+f"num_processes: {blk_Startup.num_processes}")
    #print(f"blk_Procdict: {blk_Procdict}"+"\n")

    procdict = {}
    mpe = multiprocessing.Event()

    for pc in range(1, processes + 1):
        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()
        myprocess = multiprocessing.Process(target=video_process, name=f'VP{pc}', args=(in_queue, out_queue, blk_Procdict[str(pc)],mpe, pc ))

        #structure:      0:in_queue, 1:out_queue, 2:procnum, 3:process, 4:procname, 5:running, 6:counter
        procdict[pc] = [in_queue,    out_queue,   pc,        myprocess, f'VP{pc}',  True,      4         ] #set couter to 4 for 5 trials
        myprocess.start()

    procdict[-2]=[m_in_queue,  m_out_queue,  -2,  None, 'VPMain',  True,  4 ] #add a record for communication between master and main
    masterprocess = multiprocessing.Process(target=master, args=(pc, procdict, mpe, blk_Master))
    masterprocess.start()
    procdict[-1]=[None,  None,  -1,  masterprocess, 'VPMaster',  True,  4 ] #add a record for communication between master and main

    while True:
        if not m_out_queue.empty():
            # work the queue
            try:
                # commandset structure {'To':process (1....),master (-1), or all (0),'From':process (1...),or master (-1),'Command':'','Args':{'Arg1': 'Val1','Arg2': 'Val2'}}
                commandset = m_out_queue.get_nowait()
            except queue.Empty:
                pass
            co = commandset.get('Command','')
            if co == "ackprocdied": #turn off the "running" flag when ack received. Should alreday been disabled by automatic countdown in rollcall(), extra precaution ...
                nu=commandset['Args']['number']
                if nu > 0:
                    rec = procdict[nu]
                    rec[5] = False
                    procdict[nu] = rec

        rollcall(mainctl,procdict) #Check for all processes present
        time.sleep(0.1)  # waste a little time


    sys.exit()
