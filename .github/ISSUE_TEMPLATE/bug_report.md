---
name: Bug Report or Feature Request
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---
Before filing a report consider the following questions:

### Have you read and followed all [general readme instructions](/README.md), the [documentation of the modified YOLO object](/MultiYOLO.md), the [documentation of MultiDetect.py](/MultiDetect.md), and the [CUDA primer](CUDA101.md) exactly? 

### Are you running either python3.7 or python3.8?

### Have you installed all required libraries using [requirements.txt?](/requirements.txt)

### Have you checked the [troubleshooting](https://github.com/AntonMu/TrainYourOwnYOLO#troubleshooting) section? 

Once you are familiar with the code, you're welcome to modify it. Please be aware that MultiDetect.py can be quite complex, especially due to multiple processes and a GUI. Please only continue to file a bug report if you encounter an issue with the provided code and after having followed the instructions.

If you have followed the instructions exactly, couldn't solve your problem with the provided troubleshooting tips and would still like to file a bug or make a feature requests please follow the steps below.

1. It must be a bug, a feature request, or a significant problem with the documentation (for small docs fixes please send a PR instead).
2. **Every section** of the form below must be filled out.

------------------------

### Readme 

- I have followed all Readme instructions carefully: **Answer**
- I am aware that this is an experimental project: **Answer**

### Troubleshooting Section

- I have looked at the  [troubleshooting](https://github.com/AntonMu/TrainYourOwnYOLO#troubleshooting) section: **Answer**

### System information
- **What is the top-level directory of the model you are using?**:
- **Have you written custom code (as opposed to using a stock example script provided in the repo)?**:
- **Is your [CUDA installed correctly?](/CUDA101.md)**:<br>
`
nvidia-smi
`

- **Does your [CUDA recognize and report your GPU(s) correctly?](/CUDA101.md)**:
`
python -c 'import tensorflow as tf; len(tf.config.experimental.list_physical_devices("GPU"))'
`

- **Do you have enough main memory (12 Gigabytes for 2 concurrent processes, 3+ Gigabytes for each additional?**:
- **Did you experiment with the CUDA settings in MultiDetect.conf, especially with gpu_memory_fraction and allow_growth?**:   
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:
- **TensorFlow version (use command below)**:
`
python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"
`
- **CUDA/cuDNN version**:<br>
'
nvcc -V
`
- **GPU model and memory**:
- **Exact command to reproduce**:


### Describe the problem
Describe the problem clearly here. Be sure to convey here why it's a bug or a feature request.

### Source code / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem.

