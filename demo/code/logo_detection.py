import sys, os
sys.path.append(os.path.dirname(__file__))
import time
import lynred_py as lynred
import calibrate
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np


def update_calibration(pipe, camera):
    update_offset = []
    for i in range(50):
        image = lynred.base_t.image_t()
        camera.get_image(image)
        update_offset.append(image)

    lynred.algo.pipe_shutterless_2ref_t.update_calibration(pipe, update_offset)
    return pipe

if __name__ == '__main__':
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

    # image processing

    # cv2.namedWindow("Trackbar")
    # cv2.createTrackbar("thresh", "Trackbar", 147,254, nothing)
    # cv2.createTrackbar("kernel_size", "Trackbar", 20,20, nothing)

    thresh = 120
    kernel_size = 20

    img_ref = cv2.imread("ref_binary_logo.jpg", cv2.IMREAD_GRAYSCALE)
    cnt_ref,_ = cv2.findContours(img_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_ref = sorted(cnt_ref, key=lambda x: cv2.contourArea(x))
    cnt_ref = cnt_ref[-1]

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1024, 1024)

    while True:
        try:
            camera.get_image(image)
            if (image.empty()):
                continue

            # image
            start = time.time()
            pipe.execute(image, corrected_image)
            correction_times.append(time.time() - start)
            img_infra = np.array(corrected_image)
            
            # img_infra = cv2.applyColorMap(img_arr, cv2.COLORMAP_JET)
            
            _, img_infra_thresh = cv2.threshold(img_infra, thresh, 255, cv2.THRESH_BINARY)
            # ouverture morphologique pour enlever le bruit
            # kernel_size = cv2.getTrackbarPos("kernel_size","Trackbar")
            kernel = np.ones((kernel_size,kernel_size), np.uint8)
            img_infra_thresh = cv2.erode(img_infra_thresh, kernel)
            img_infra_thresh = cv2.dilate(img_infra_thresh, kernel)
            
            contours, _ = cv2.findContours(img_infra_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours)>3: thresh+=1
            # if len(contours)==1 and thresh>0: thresh-=1
            if len(contours)==0: thresh=50


            if contours :
                contours = sorted(contours, key=lambda x: cv2.contourArea(x))
                cnt = contours[-1]

                x,y,w,h = cv2.boundingRect(cnt)
                img_infra = cv2.cvtColor(img_infra, cv2.COLOR_GRAY2BGR)
                # cv2.drawContours(img_infra, [cnt], 0, (0,0, 255), 3)
                cv2.rectangle(img_infra,(x,y),(x+w,y+h),(255,0,0),2)
                
                m = cv2.matchShapes(cnt,cnt_ref,3,0.0)
                if m < 0.1 : 
                    cv2.putText(img_infra, "LYNRED", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                else:
                    cv2.putText(img_infra, "Pont thermique", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

            # cv2.imshow("frame_thresh",img_infra_thresh)
            cv2.imshow("frame",img_infra)

            # Press Q on keyboard to exit, S to save an image

            keypress = cv2.waitKey(25)
            if keypress & 0xFF == ord('q'):
                break
            elif keypress & 0xFF == ord('r'):
                print('Calibrating')
                pipe = update_calibration(pipe, camera)
            elif keypress & 0xFF == ord('t'):
                print('Reset threshold = 50')
                thresh=50
            elif keypress & 0xFF == ord('s'):
                now = datetime.datetime.now()
                cv2.imwrite('saved_images/result'+str(now.hour)+str(now.minute)+str(now.second) + '.jpg', img_infra)
                cv2.imwrite('saved_images/result_binary'+str(now.hour)+str(now.minute)+str(now.second) + '.jpg', img_infra_thresh)
                print("Image saved")

        except KeyboardInterrupt:
            break

    camera.stop()
    image.release()
