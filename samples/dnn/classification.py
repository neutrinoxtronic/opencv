import os
import glob
import argparse
import cv2 as cv
import numpy as np
from common import *

def help():
    print(
        '''
        Firstly, download required models using `download_models.py` (if not already done). \n

        To run:
            python classification.py model_name --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)

        Sample command:
            python classification.py googlenet --input=path/to/image
        Model path can also be specified using --model argument
        '''
    )

def get_args_parser(func_args):
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
               cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input',
                        help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--crop', type=bool, default=False,
                        help='Center crop the image.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA fp16 (half-float preprocess)'% targets)

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'classification')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='Use this script to run classification deep learning networks using OpenCV.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args(func_args)

def load_images(directory):
    # List all common image file extensions, feel free to add more if needed
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    files = []
    for extension in extensions:
        files.extend(glob.glob(os.path.join(directory, f'*.{extension}')))
    return files

def main(func_args=None):
    help()
    args = get_args_parser(func_args)
    args.model = findFile(args.model)
    args.classes = findFile(args.classes)

    # Load names of classes
    classes = None
    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

    # Load a network
    net = cv.dnn.readNet(args.model)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)

    winName = 'Deep learning image classification in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    isdir = False

    if args.input:
        input_path = args.input

        if os.path.isdir(input_path):
            isdir = True
            image_files = load_images(input_path)
            if not image_files:
                print("No images found in the directory.")
                exit(-1)
            current_image_index = 0
        else:
            cap = cv.VideoCapture(input_path)
            if not cap.isOpened():
                print("Failed to open the input video")
                exit(-1)
    else:
        cap = cv.VideoCapture(0)

    while cv.waitKey(1) < 0:
        if isdir:
            if current_image_index >= len(image_files):
                break
            frame = cv.imread(image_files[current_image_index])
            current_image_index += 1
        else:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break

        # Create a 4D blob from a frame.
        inpWidth = args.width if args.width else frame.shape[1]
        inpHeight = args.height if args.height else frame.shape[0]

        blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=args.crop)
        if args.std:
            blob[0] /= np.asarray(args.std, dtype=np.float32).reshape(3, 1, 1)

        # Run a model
        net.setInput(blob)
        out = net.forward()

        # Get a class with a highest score.
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        # Put efficiency information.
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Print predicted class.
        label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
        cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow(winName, frame)

        if cv.waitKey(1000) & 0xFF == ord('q'):  # Wait for 1 second on each image, press 'q' to exit
            break



if __name__ == "__main__":
    main()