## A quickie CUDA primer

**Before you delve into YOLO, you need a running CUDA installation. If you already have one, then you are a decorated CUDA veteran. Stop reading, you'll find nothing new.
If you are new at CUDA, here are some pointers.**

This is an Ubuntu-centric howto. I used to be a CENTOS afficionado, but early CUDA on Linux was pretty much all Ubuntu, and to this day, most of the CUDA ecosystem speak Ubuntu-ese. CUDA is the glue that allows computer applications make use of the powerhouse GPUs in our machines. CUDA once stood for "Compute Unified Device Architecture," but Nvidia, maker of GPUs and CUDA, dropped the long form, probaby because there is little "unified" to CUDA - things seem to break royally from version to version. 

Please note that the Tensorflow version used in the repo seems to be happiest with CUDA 10.1 so that's what you should install. Don't install anything else, it could send you to version hell. Nvidia is pushing the latest 11.X CUDA version, and is making 10.1 a little hard to find. [Follow this link.](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
Be wary of the [offered .deb files](https://developer.nvidia.com/cuda-10.1-download-archive-update2target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal). They will work on a system untouched by prior CUDA installations, but if your Ubuntu wears scars of prior battles with the installer, the .deb files might install a different version than 10.1, or none at all. The easiest and safest installation is with this runfile:

```
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run
```

If your Ubuntu is newer, the .run file may complain about the wrong gcc compiler. Don't install an older compiler. Simply restart the run file with --override. If you have a newer Nvidia video driver, don't let the .run file install an older one. Simply deselect the driver option. Newer Nvidia video drivers are compatible with older CUDA versions.
Now add the following lines to your .bashrc file:

```
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
  export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

This will put your cuda-10.1 on the path, but only if /usr/local/cuda-10.1/bin/ exists.

## Test your CUDA

Type this into your command line monitor:

```
nvidia-smi  
```

This should list your GPUs, along with the driver and CUDA version. Don't be shocked if the shown CUDA version is higher than 10.1. Nvidia-smi lies.
Now type

```
nvcc -V
```

If CUDA is installed correctly, you will now see the proper 10.1 version. If you get a no found error, you hopefully only forgot the lines in the .bashrc file. Not so hopefully, you CUDA installation failed.

All O.K.? You are nearly out of the woods! 

## Now for the final test

For that, make sure you have your virtualenv enabled. If you haven't run the requirements.txt file yet, navigate to the base directory of the repo, and type:

```
pip install -r requirements.txt
```

This installs all the packages required by Python to run your YOLO.

For the final test, fire up Python, and type:

```
import tensorflow as tf
```

You should get something like 

```
"I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1 ... "
```

You definitely don’t want to see something like

```
"ModuleNotFoundError: No module named 'tensorflow' ... " 
```

In that case, either the installation of the modules in requirements.txt went awry, or you forgot to enable your virtualenv. 
If you see something like "Could not load dynamic library 'libcudart.so.10.1' ..." your CUDA installation is broken. 

If all is well, enter this into Python:

```
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

That line should cause screen output that ends in "Num GPUs Available: ..." and the number of GPUs in your machine. If it says 0 are available, and if you have Nvidia GPUs in your machine, your CUDA installation needs attention.

The "Could not load dynamic library 'libcudart.so.10.1" reminds us that the installed TensorFlow version is happiest with CUDA 10.1. Please install that. 
As long as you see errors, do not proceed. It won't work. You need to resolve the problem first. CUDA installation can be a pain. We've all been there, don't give up.

## For more information ... 

[**README!**](/README.md) A MultiDetect intro<br>
[**YOLO on all cylinders**](/MultiYOLO.md) The YOLO object, tuned for multiple processes<br>
[**How much, how little memory](/Memory_settings.md). The best memory settings<br>
[**Running inference**](/3_Inference/README.md) Putting it to use<br>

- Please **star** ⭐ this repo to get notifications on future improvements and
- Please **fork** 🍴 this repo if you like to use it as part of your own project. 

