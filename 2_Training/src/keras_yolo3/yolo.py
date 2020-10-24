# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
BS- adapted for multi-stream, muiti-GPU by Bertel Schmitt 2020
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import colorsys
from timeit import default_timer as timer
import numpy as np
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from .yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from .yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import tensorflow.compat.v1 as tf
import tensorflow.python.keras.backend as K
tf.disable_eager_execution()


#-BS -

import threading
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# BS- import for silence
from keras.constraints import maxnorm
from tensorflow.compat.v1 import logging


def silence(on=True):
    """
    BS -
    attempt to silence way too chatty tensorflow
    Is triggered by setting hush flag tro True
    """
    if on:
        #print("YOLO - silence on")
        tf.logging.set_verbosity(tf.logging.ERROR)
        # tf.autograph.set_verbosity(0)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['AUTOGRAPH_VERBOSITY'] = '0'
        warnings.filterwarnings("ignore")
    else:
        #print("YOLO - silence off")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
        os.environ['AUTOGRAPH_VERBOSITY'] = '5'


class YOLO(object):
    """
    BS -1
    Adapted for multi-stream, multi GPU. This class allow the model to be run on a GPU chosen by the caller,
    and (optionally) using a set fraction of the GPU memory.
    This opens the door to simultaneously detect objects in multiple streams on one, or more GPUs at the same time.
    Each YOLO object will be given its own model.
    This way, different models can (but don't have to) be used for different video streams.
    The hush flag will try dialing down the noisy warnings and messages emitted by Keras/Tensorflow
    """
    _defaults = {
        "model_path": "model_data/yolo.h5",
        "anchors_path": "model_data/yolo_anchors.txt",
        "classes_path": "model_data/coco_classes.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        # BS-1 Changes and additions:
        "gpu_num": 1,       # legacy setting. Did not note any significant changes when setting higher. Recommend leaving alone
        # Default -1: let Keras decide. 0 run on GPU 0, 1 rund on GPU 1 etc. Keras also allows for "0,1" (etc.) but saw no effect
        "run_on_gpu": -1,
        "gpu_memory_fraction": 1,
        "allow_growth": -1,  # default: -1 let Keras decide. 1 allow growth, 0 do not allow
        "hush": True,  # Set to true to suppress noisy status output
        "ignore_labels": [],  # list of labels/objects not to report when detected
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        if self.hush:
            silence(on=True)
        else:
            silence(on=False)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        # make Keras/TF use GPUs and memory parts as specified
        config = tf.ConfigProto()
        # may not work with allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        # if -1: let Keras decide, else ...
        if self.allow_growth > -1 and self.allow_growth < 2:
            config.gpu_options.allow_growth = bool(
                self.allow_growth)  # allow_growth 0/False  1/True
        if str(self.run_on_gpu) != "-1":  # if -1: let Keras decide, else ...
            config.gpu_options.visible_device_list = str(
                self.run_on_gpu)  # set required GPU

        session = tf.Session(config=config)
        K.set_session(session)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            ".h5"), "Keras model or weights must be a .h5 file."

        # Load model, or construct model and load weights.
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = (
                tiny_yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 2, num_classes
                )
                if is_tiny_version
                else yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 3, num_classes
                )
            )
            self.yolo_model.load_weights(
                self.model_path
            )  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(
                self.yolo_model.output
            ) * (
                num_classes + 5
            ), "Mismatch between model and given anchor and class sizes"

        end = timer()
        # turn off the noise
        if not self.hush:
            print(
                "{} model, anchors, and classes loaded in {:.2f}sec.".format(
                    model_path, end - start
                )
            )

        # Generate colors for drawing bounding boxes.
        if len(self.class_names) == 1:
            self.colors = ["GreenYellow"]
        else:
            hsv_tuples = [
                (x / len(self.class_names), 1.0, 1.0)
                for x in range(len(self.class_names))
            ]
            self.colors = list(
                map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(
                    lambda x: (int(x[0] * 255),
                               int(x[1] * 255), int(x[2] * 255)),
                    self.colors,
                )
            )
            # Fixed seed for consistent colors across runs.
            np.random.seed(10101)
            np.random.shuffle(
                self.colors
            )  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(
                self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        return boxes, scores, classes

    def detect_image(self, image, show_stats=False):
        """
        To maintain backward compatibility, detect_image calls detect_image_extended,
        but returns out_prediction and image, just like original detect_image did
        """
        return(self.detect_image_extended(image, show_stats, old_style=True))

    def detect_image_extended(self, image, show_stats=False, old_style=False):
        """
        BS-
        This is detect_image, rewritten to also return labels (including confidence) and time spent in routine
        Returns (annotated) image, time-spent, and out_prediction_ext, which is a list of list, containing, for each object dectected [left, top, right, bottom, predicted_class, score]
        """
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")
        if show_stats:
            print(f"image_data.shape: {image_data.shape}")
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )
        # BS- No stats if there is nothing to show
        if show_stats and len(out_boxes) > 0:
            print("Found {} boxes for {}".format(len(out_boxes), "img"))
        out_prediction = []
        out_prediction_ext = []  # BS- also return label in the same set
        labels = []  # BS- keep track of labels
        font_path = os.path.join(os.path.dirname(
            __file__), "font/FiraMono-Medium.otf")
        font = ImageFont.truetype(
            font=font_path, size=np.floor(
                3e-2 * image.size[1] + 0.5).astype("int32")
        )
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class in self.ignore_labels:  # BS- optional ignore
                continue
            box = out_boxes[i]
            score = out_scores[i]

            label = "{} {:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype("int32"))
            left = max(0, np.floor(left + 0.5).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
            right = min(image.size[0], np.floor(right + 0.5).astype("int32"))

            # image was expanded to model_image_size: make sure it did not pick
            # up any box outside of original image (run into this bug when
            # lowering confidence threshold to 0.01)
            if top > image.size[1] or right > image.size[0]:
                continue
            if show_stats:
                print(label, (left, top), (right, bottom))
                print(f'Predicted_class: {predicted_class}')
                print(
                    f'Out_prediction: left: {left}, top: {top}, right: {right}, bottom: {bottom}, c: {c}, Score: {score} Predicted_class: {predicted_class}')
            # output as xmin, ymin, xmax, ymax, class_index, confidence
            out_prediction.append([left, top, right, bottom, c, score])
            out_prediction_ext.append(
                [left, top, right, bottom, predicted_class, score])
            # labels.append(label)  # BS - keep track of labels
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, bottom])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline=self.colors[c]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c],
            )

            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        if show_stats:
            print("Time spent: {:.3f}sec".format(end - start))
        if old_style:
            return(out_prediction, image)
        else:
            return(image, end - start, out_prediction_ext)

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    # int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    isOutput = True if output_path != "" else False
    if isOutput:
        print(
            "Processing {} with frame size {} at {:.1f} FPS".format(
                os.path.basename(video_path), video_size, video_fps
            )
        )
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        # opencv images are BGR, translate to RGB
        frame = frame[:, :, ::-1]
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image, show_stats=False)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(
            result,
            text=fps,
            org=(3, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50,
            color=(255, 0, 0),
            thickness=2,
        )
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        if isOutput:
            out.write(result[:, :, ::-1])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    vid.release()
    out.release()
    # yolo.close_session()
