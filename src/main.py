import argparse
from src.cnnModels import *
from src.imageHandler import *

def parseCommandLinesArgs():
    parser = argparse.ArgumentParser(description="Arguments for autonomous vehicle computer vision processing")

    parser.add_argument("-i","--imageInput", type=str, help="Input image that will be processed by YOLO")
    parser.add_argument("-m", "--model", type=str, help="Path to the saved YOLOV3 model. This must be a keras model!")

    return parser.parse_args()

# This function coordinates all the procedures to the navigation system
#   Considering lane tracking for navigation and CNN object detection
def main(model: str, image: str):

    yolo, anchors, class_threshold, labels = model_setup_h5(model)
    return

if __name__ == '__main__':
    args = parseCommandLinesArgs()
    main(args.model)