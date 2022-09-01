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
import math

def update_calibration(pipe, camera):
    update_offset = []
    for i in range(50):
        image = lynred.base_t.image_t()
        camera.get_image(image)
        update_offset.append(image)

    lynred.algo.pipe_shutterless_2ref_t.update_calibration(pipe, update_offset)
    return pipe

def nothing(x):
    pass

if __name__ == '__main__':

    # ----Camera----

   
    CURRENT_PATH = os.path.dirname(__file__)
    os.chdir(CURRENT_PATH)

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

    # other

    kernel = np.ones((3,3),np.uint8)
    thresh = 127
    roi_x = 0   # coin supérieur gauche de la roi
    roi_y = 0
    roi_d = 300           # côté du carré


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1024, 1024)


    cv2.namedWindow("Trackbar")
    cv2.createTrackbar("Seuil", "Trackbar", 105,255, nothing)

    while True:
        try:
            camera.get_image(image)
            if (image.empty()):
                continue

            start = time.time()

            pipe.execute(image, corrected_image)
            correction_times.append(time.time() - start)
            img_arr = np.array(corrected_image)     # image codé en uint8
            img_arr_color = cv2.cvtColor(img_arr,cv2.COLOR_GRAY2RGB)


            roi = img_arr[roi_x:roi_x+roi_d,roi_y:roi_y+roi_d]
            roi_color = img_arr_color[roi_x:roi_x+roi_d,roi_y:roi_y+roi_d]

            # roi = img_arr
            # roi_color = img_arr_color

            cv2.rectangle(img_arr_color,(roi_x,roi_y),(roi_x+roi_d,roi_y+roi_d),(0,255,0),0)  


            # Processing

            roi = cv2.GaussianBlur(roi,(3,3),100) 
            # thresh = cv2.getTrackbarPos("Seuil","Trackbar")
            # _, roi_thresh = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
            _, roi_thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            roi_thresh = cv2.erode(roi_thresh, kernel)
            roi_thresh = cv2.dilate(roi_thresh, kernel)
            contours, _ = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
           
            # if len(contours)>3: thresh+=1
            # # if len(contours)==1 and thresh>0: thresh-=1
            # if len(contours)==0: thresh=50

            
            

            if contours : 
                cnt = max(contours, key = lambda x: cv2.contourArea(x))     # contour avec la plus grande aire

                epsilon = 0.0005*cv2.arcLength(cnt,True)
                approx= cv2.approxPolyDP(cnt,epsilon,True) # légère approx pour lisser / réduir le nombre de pts
                hull = cv2.convexHull(cnt)
                areahull = cv2.contourArea(hull)  # calcul de l'aire
                areacnt = cv2.contourArea(cnt)


                # pourcentage de l'aire couverte par la main dans la convex hull
                arearatio=((areahull-areacnt)/areacnt)*100  


                # calcul des defects
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)

                # nombre de defects
                l=0
                
                # détection des defects dus aux doigts
                if defects is not None :
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])
                        # longueurs des côtés
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        s = (a+b+c)/2
                        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))  # aire triangle, formule de héron
                        # calcul de la hauteur du triangle
                        d=(2*ar)/a
                        # pythagore généralisé
                        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                        # ignorer angles > 90 (côtés de la main) et pts trop proche de la convex hull
                        if angle <= 90 and d>30:
                            l += 1
                            cv2.circle(roi_color, far, 3, [255,0,0], -1)
                        
                        # dessiner lignes
                        cv2.line(roi_color,start, end, [0,255,0], 2)
                        
                        
                    l+=1  # nombre de doigts


                    # classification
                    if l==1:
                        if areacnt<2000:
                            text ='Placer main'
                        else:
                            if arearatio<12:
                                text = '0'
                            # elif arearatio<17.5:
                            #     text = 'Best of luck'
                            else:
                                text="1"
                    elif l==2:text="2"
                    elif l==3:
                        text = '3'
                        # if arearatio<27:
                        #         text = '3'
                        # else:
                        #         text = 'ok'
                    elif l==4:text = '4'
                    elif l==5:text = '5'
                    elif l==6:text = 'reposition'
                    else     : text = 'reposition'
                    
                    cv2.putText(img_arr_color,text,(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
                    

            cv2.imshow('image', img_arr_color)

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
