#!/usr/bin/env python3.7
"""
Stub for Testbench app for multi-stream, multi-gpu YOLO
File name: MultiDetect.py
Launches MultiDetect.wk
Author: Bertel Schmitt, with inspirations from a cast of thousandss
Date last modified: 10/29/2020
Version: 0.02
Python Version: 3.7
License: This work is licensed under a Creative Commons Attribution 4.0 International License.
"""
"""
This contortion has become necessary to protect the user from launching MultiDetect.py with python 2.7
Python 2.7 occasionally is launched when sudo is invoked. By the time the vesrion has been checked, Python
already has performed a syntax check, and it errors-out befor our app gets to version-checking. So we need
to do this 2-stage hack ...

"""
import os
import sys

def getappname():
    an = __file__
    if "/" in an:
     _ , an = an.strip().split('/',1)
    appname, _ = an.strip().split('.',1)
    return(appname)

#############################################
#assert python >= 3.7
if not sys.version_info >= (3, 7):
    thispython = "python " +str(sys.version_info[0])+"."+str(sys.version_info[1])
    print(thispython)
    print("\n\n################################################################")
    print("MultiDetect.py needs python version 3.7 and later. This seems ")
    print("to be %s. If you use 'sudo python ...', make sure it " % thispython)
    print("hits python 3.7 or later. Sudo sometimes still invokes python 2.7")
    print("################################################################\n\n")
    sys.exit()
else:
    exec(compile(open('MultiDetect.wk', "rb").read(), 'MultiDetect.wk', 'exec'))


