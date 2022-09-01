from email.mime import base
import torch
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import pytorch_lightning as pl
from PIL import Image
from torchvision.transforms import Compose
import cv2

#include the path of the dataset(s) and the model(s)
CUR_DIR_PATH = Path(__file__).resolve()
ROOT = CUR_DIR_PATH.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))# add ROOT to PATH
CUR_DIR_PATH=os.path.dirname(CUR_DIR_PATH)
print("CUR_DIR_PATH : "+ CUR_DIR_PATH)



#include customed class
from TransformHgr import PilToTensor, ResizeImage
from Cnn import Cnn 

import time
import lynred_py as lynred
import calibrate

import numpy as np
import cv2
import os
import datetime


def update_calibration(pipe, camera):
    update_offset = []
    for i in range(50):
        image = lynred.base_t.image_t()
        camera.get_image(image)
        update_offset.append(image)

    lynred.algo.pipe_shutterless_2ref_t.update_calibration(pipe, update_offset)
    return pipe

if __name__ == '__main__':

    # ----Model----
    #Fix random
    pl.utilities.seed.seed_everything(42,True)

    #Transforms
    TrToTensor = PilToTensor()
    TrResize = ResizeImage()
    transform = Compose([TrResize,TrToTensor])

    #Load Model
    path_weight_model = os.path.join(CUR_DIR_PATH,"hand_recognition/Project/Log/Cnn/version_6/epoch=39-step=3120.ckpt")
    pretrained_model = Cnn.load_from_checkpoint(path_weight_model)
    pretrained_model.eval() # No need drop out in inference
    pretrained_model.freeze() # No gradient needed
    # labels = ["palm","l","fist","fist_moved","thumb","index","ok","palm_moved","c","down" ]
    labels = ["fist","palm","index","L" ]
    # labels = ["palm","two","fist"]
    #Select device -> use gpu if is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model=pretrained_model.to(device)

    
    # ----Camera----

    CUR_DIR_PATH=os.path.dirname(__file__)
    os.chdir(CUR_DIR_PATH)

    calibration_dir = 'images/calibration'
    config_path = 'configs/shutterless.json'
    cold_img_dir = 'images/calibrations/Tamb=20/TCN=10'
    hot_img_dir = 'images/calibrations/Tamb=40/TCN=40'
    ref_img_dir = 'images/calibrations/Tamb=40/TCN=10'

    # Read processing configuration
    config = calibrate.read_config(config_path)

    # Read calibration images
    cold_images = calibrate.read_image_dir(calibrate.get_calibration_dir(
        cold_img_dir, config, 'cold', calibration_dir))
    hot_images = calibrate.read_image_dir(calibrate.get_calibration_dir(
        hot_img_dir, config, 'hot', calibration_dir))
    ref_images = calibrate.read_image_dir(calibrate.get_calibration_dir(
        ref_img_dir, config, 'ref', calibration_dir))

    # Create processing pipe
    pipe = calibrate.create_pipe(
        config['mode'],
        cold_images=cold_images,
        hot_images=hot_images,
        ref_images=ref_images,
        steps=config['steps'])

    # Initialize camera
    dalab_cameras = lynred.acq.discover_device_alab_cameras()
    for camera_name in dalab_cameras:
        print("Connecting to the Device Alab camera " + camera_name.c_str())
        camera = lynred.acq.create_camera_device_alab(camera_name.c_str())


    image = lynred.base_t.image_t()
    corrected_image = lynred.base_t.image_t()
    correction_times = []
    camera.start()

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1024, 1024)

    while True:
        try:
            camera.get_image(image)
            if (image.empty()):
                continue

            start = time.time()

            pipe.execute(image, corrected_image)
            correction_times.append(time.time() - start)
            img_arr = np.array(corrected_image)/255
            img_arr_pl = Image.fromarray(img_arr)
            sample = {'x' : img_arr_pl }
            sample = transform(sample)
            x = torch.unsqueeze(sample['x'],dim=0) #Image as tensor and add batch dim
            pred = torch.argmax(pretrained_model.forward(x.to(device))).cpu().numpy()
            x =  torch.squeeze(x).cpu().numpy()
            cv2.putText(img_arr, labels[pred], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
            cv2.imshow('image', img_arr)
            # print(labels[pred])

            # Press Q on keyboard to  exit, S to save an image
            keypress = cv2.waitKey(25)
            if keypress & 0xFF == ord('q'):
                break
            elif keypress & 0xFF == ord('r'):
                print('Calibrating')
                pipe = update_calibration(pipe, camera)
            elif keypress & 0xFF == ord('s'):
                now = datetime.datetime.now()
                print("Saving image")
                cv2.imwrite('saved_images/result'+str(now.hour)+str(now.minute)+str(now.second) + '.jpg', img_arr)

        except KeyboardInterrupt:
            break

    camera.stop()
    image.release()
