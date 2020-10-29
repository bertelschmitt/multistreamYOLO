#!/usr/bin/env python3.7

import os
import subprocess
import time
import sys
import argparse
import requests
import progressbar
from os import system, name, path

FLAGS = None

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ownload_folder = os.path.join(
#    root_folder, "2_Training", "src", "keras_yolo3", "testing")
data_folder = os.path.join(root_folder, "Data")
model_folder = os.path.join(data_folder, "Model_Weights")
#download_script = os.path.join(model_folder, "Download_Weights.py")
#gdrive_id = "1IHbJlsHmLB8IXnAG4S-mMM8J34CJ2hRg"

print(f"Model Folder: {model_folder}")

_ = system('clear')

print("\n\n"+f"This will download a ready-made model, trained with the cat pictures in\n{data_folder}/Source_Images/Training_Images"+"\n"+"Of course, you can - even should - train your own model.\nIf you don't have the time, this pre-trained model will help you get going.\n\n")

resp="z"
while resp.upper() not in 'YN':
    resp = input("Do you want to download? [Y/N]?")

if resp.upper() == "N":
    sys.exit()

file_exists = False

if path.exists(f"{model_folder}/data_classes.txt"):
    print(f"\n-  File data_classes.txt already exists in {model_folder}.")
    file_exists = True
if path.exists(f"{model_folder}/trained_weights_final.h5"):
    print(f"\n-  File trained_weights_final.h5 already exists in {model_folder}.")
    file_exists = True

if file_exists:
    print("\nIf you want to keep the file(s), please A)bort, back-up or re-name, and run agin.\n"+"Otherwise, choose O)verwrite.\n")


    resp="z"
    while resp.upper() not in 'AO':
        resp = input("Do you want to A)bort or O)verwrite [A/O]")

    if resp.upper() == "A":
        sys.exit()
    else:
        resp="z"
        while resp.upper() not in 'AO':
            resp = input("\nLast chance, A)bort or O)verwrite [A/O]")

    if resp.upper() == "A":
        sys.exit()

basecmd=f' --show-progress --no-cookies --user-agent="Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" -P {model_folder} '

cmdstring = f'wget http://bertelschmitt.tokyo/download/multidetect/data_classes.txt {basecmd} ; wget http://bertelschmitt.tokyo/download/multidetect/trained_weights_final.h5 {basecmd}'
subprocess.call(cmdstring, shell=True)

if path.exists(f"{model_folder}/data_classes.txt") and path.exists(f"{model_folder}/trained_weights_final.h5"):
    print("\n\nDownload successful")
else:
    print("\n\nSomething went wrong. Please try again")

