import numpy as np
import cv2
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.core.postprocess import morphological_process, connect_components_analysis, fitlane, connect_lane


def read_data(path):
    file_list = os.listdir(path)
    file_names = [path + file_name for file_name in file_list]
    return file_names

if __name__ == '__main__':
    path = 'DWA/lane/'
    output = 'DWA/lanepost/'
    img_list = read_data(path)
    for img_name in img_list:
        # print(img_name)
        img = cv2.imread(img_name, 0)
        print(img_name)
        img = morphological_process(img, kernel_size=3, func_type=cv2.MORPH_OPEN)
        img = connect_lane(img)
        name = os.path.basename(img_name)
        # # print(output)
        # # print(type(img))
        cv2.imwrite(output+name, img*255)

