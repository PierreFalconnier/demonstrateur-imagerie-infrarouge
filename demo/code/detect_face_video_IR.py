# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import os
import time

import camera as cam
import lynred_py as lynred
import calibrate
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_one(model, orgimg, device):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    if orgimg.ndim == 2 : 
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_GRAY2BGR) # convertion en rbg
        # orgimg = np.stack((orgimg,)*3, axis=-1)  

    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xyxy, conf, landmarks, class_num)

    return orgimg






if __name__ == '__main__':
    CUR_DIR_PATH=os.path.dirname(__file__)
    os.chdir(CUR_DIR_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n_face.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)

    # Camera infrarouge

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
            img_arr = np.array(corrected_image)
            img_arr = detect_one(model, img_arr  ,device)

            cv2.imshow('image', img_arr)

            # Press Q on keyboard to  exit
            keypress = cv2.waitKey(25)
            if keypress & 0xFF == ord('q'):
                break
            elif keypress & 0xFF == ord('r'):
                print('Calibrating')
                pipe = cam.update_calibration(pipe, camera)

        except KeyboardInterrupt:
            break

    camera.stop()
    image.release()



    
