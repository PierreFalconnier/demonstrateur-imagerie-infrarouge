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
    k=False

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

            # image
            start = time.time()
            pipe.execute(image, corrected_image)
            correction_times.append(time.time() - start)
            img_arr = np.array(corrected_image)
            if k : img_arr = cv2.applyColorMap(img_arr, cv2.COLORMAP_JET)
            cv2.imshow('image', img_arr)

            # histogramme (grosse perte de fps)

            # plt.hist(img_arr.ravel(), bins = 256, density=True)
            # plt.plot()
            # plt.pause(0.001)
            # plt.clf()


            # Press Q on keyboard to exit, S to save an image


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
            elif keypress & 0xFF == ord('c'):
                if k : 
                    k=False
                else :
                    k=True
                


        except KeyboardInterrupt:
            break

    camera.stop()
    image.release()
