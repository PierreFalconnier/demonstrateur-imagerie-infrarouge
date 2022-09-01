import os
import time
import json
from argparse import ArgumentParser

import lynred_py as lynred

try:
    lynred.algo.pipe_shutter_t()
except Exception as err:
    print('Invalid License.')
    raise err


IMG_EXTS = (
    'ulis', 'bmp', 'tiff', 'tif', 'tiff',
    'png', 'hdf5', 'jpg', 'jpeg', 'avi')


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(PACKAGE_DIR, 'configs/settings.json')


def is_image_file(filepath):
    """Check if file is an image."""
    if (os.path.isfile(filepath)):
        if (filepath.endswith(IMG_EXTS)):
            return True
        else:
            return False
    return False


def read_image_dir(image_dir, read_images=True, walk=False):
    """Read images as a list of lynred::base_t::imvector_t
    from directory path."""

    if (image_dir is None):
        return []

    # Check if directory exists
    cond = (not os.path.exists(image_dir))
    cond = (cond or (not os.path.isdir(image_dir)))
    if (cond):
        raise RuntimeError(
            f'{image_dir} is not a directory or does not exist.')

    image_paths = []
    # Read image paths from directory (all levels)
    if (walk):
        for root, dirs, files in os.walk(image_dir):
            for name in files:
                filepath = os.path.join(root, name)
                if (is_image_file(filepath)):
                    image_paths.append(filepath)
    else:
        # Read image paths from directory (one level)
        for file in os.listdir(image_dir):
            filepath = os.path.join(image_dir, file)
            if (is_image_file(filepath)):
                image_paths.append(filepath)

    # Read images and return list of images
    if (read_images):
        images = []
        for filepath in image_paths:
            images.append(read_image(filepath))
        return images
    # Return image paths
    else:
        return image_paths


def read_image(filepath):
    """Read image as lynred::base_t::imvector_t  from filepath."""
    if (is_image_file(filepath)):
        # Initialize image as lynred::base_t::imvector_t
        img = lynred.base_t.image_t()
        try:
            img.load(filepath)
        except lynred.error_t as err:
            raise RuntimeError(
                f'Error loading image from file {filepath}') from err
    else:
        raise OSError(
            f'{filepath} is not a supported image file or does not exist.')

    return img


def toggle_pipe_step(pipe, step_name, enable):
    """Toggle pipe execution steps on or off."""
    pipe.enable_exec_step(
        getattr(pipe.exec_step_t, step_name), int(enable))


def create_pipe(
    mode,
    cold_images=None,
    hot_images=None,
    ref_images=None,
    steps=None):

    """Create processing pipe, calibrate, and set steps."""

    # Initialize and calibrate pipe
    if (mode == 'shutter'):
        # Initialize pipe
        pipe = lynred.algo.pipe_shutter_t()
        # Calibrate pipe
        pipe.calibrate(cold_images, hot_images)
    elif (mode == 'shutterless'):
        # Initialize pipe
        pipe = lynred.algo.pipe_shutterless_2ref_t()
        pipe.exec_params_t.sync_shutterless_compute = False
        # Calibrate pipe
        pipe.calibrate(cold_images, hot_images, ref_images)
    else:
        raise ValueError(
            'Only "shutter" and "shutterless" modes are currently supported.')

    # Load settings
    settings = read_config(SETTINGS_PATH)

    if (steps is not None):
        if (isinstance(steps, dict)):
            for step in steps:

                # Toggle steps
                enable = steps[step].get('enable', True)
                toggle_pipe_step(pipe, step, enable)

                # Get step parameters
                exec_params = pipe.get_exec_params()

                # Set new step top-level parameters
                for param, value in steps[step].get('params', {}).items():
                    set_step_param(
                        settings, mode, exec_params, step, param, value)

                # Set new step top-level methods
                methods = steps[step].get('method', {})
                if (methods is not None):
                    method = methods.get('selected')
                    if (method is not None):
                        set_step_method(
                            settings, mode, exec_params, step, method)
                    method_params = methods.get('params', {})
                    for method, params in method_params.items():
                        for param, value in params.items():
                            set_step_method_param(
                                settings, mode, exec_params,
                                step, method, param, value)

        else:
            raise TypeError('`steps` must be either None or dict')

    return pipe



def set_step_method_param(
    settings, mode, exec_params, step, method, param, value):
    """Set pipe step method parameters."""

    mapper = settings[mode][step]

    # Get step method parameters
    step_params = getattr(exec_params,  mapper['param_name'] + '_params')
    method_params = getattr(step_params, method + '_params')

    # Get parameter setter
    if (isinstance(value, bool)):
        set_param = getattr(method_params, param)
    else:
        set_param = getattr(method_params, 'set_' + param)

    # Set parameter value
    set_param(value)


def set_step_param(settings, mode, exec_params, step, param, value):
    """Set pipe step parameter."""

    mapper = settings[mode][step]

    # Get step parameters
    step_params = getattr(exec_params, mapper['param_name'] + '_params')
    if (mapper.get('param_attr') is not None):
        step_params = getattr(step_params, mapper['param_attr'])

    # Get parameter setter
    if (isinstance(value, bool)):
        set_param = getattr(step_params, param)
    else:
        set_param = getattr(step_params, 'set_' + param)

    # Set parameter value
    set_param(value)


def set_step_method(settings, mode, pipe_params, step, method_name):
    """Set pipe step method."""
    mapper = settings[mode][step]
    algorithm = getattr(
        lynred.algo, mapper['method_getter'])
    method = getattr(algorithm.method_t, method_name)

    step_params = getattr(pipe_params, mapper['param_name'] + '_params')
    if (mapper.get('param_attr') is not None):
        step_params = getattr(step_params, mapper['param_attr'])
    method_attr = getattr(step_params, mapper['method_attr'])
    setter = mapper['method_setter']
    if (setter == 'call'):
        method_attr(method)
    elif (setter == 'set'):
        method_attr = method
    else:
        raise ValueError(f'Unsupported method setting `{setter}`')


def read_config(filepath):
    """Read JSON configuration for filepath."""
    with open(filepath, 'r') as fobj:
        config = json.load(fobj)
    return config


def find_calibration_images(calibration_dir, tamb, tobj):
    img_dir = os.path.join(
        calibration_dir, f'Tamb={tamb}', f'TCN={tobj}')
    if (os.path.isdir(img_dir)):
        return img_dir
    else:
        return None


def get_calibration_dir(img_dir, config, calib, calibration_dir):
    """Get calibration image directory from argument input or config."""

    if (img_dir is None) and ('calibration' in config):
        if (calib in config['calibration']):
            img_dir = find_calibration_images(
                calibration_dir,
                config['calibration'][calib].get('Tamb'),
                config['calibration'][calib].get('Tobj'))

    return img_dir


def main(
    config_path,
    raw_img_dir,
    output_dir,
    calibration_dir=None,
    cold_img_dir=None,
    hot_img_dir=None,
    ref_img_dir=None):

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read images from directory
    raw_image_paths = read_image_dir(
        raw_img_dir, read_images=False, walk=True)

    # Read processing configuration
    config = read_config(config_path)

    # Read calibration images
    cold_images = read_image_dir(get_calibration_dir(
        cold_img_dir, config, 'cold', calibration_dir))
    hot_images = read_image_dir(get_calibration_dir(
        hot_img_dir, config, 'hot', calibration_dir))
    ref_images = read_image_dir(get_calibration_dir(
        ref_img_dir, config, 'ref', calibration_dir))

    # Create processing pipe
    pipe = create_pipe(
        config['mode'],
        cold_images=cold_images,
        hot_images=hot_images,
        ref_images=ref_images,
        steps=config['steps'])

    # Correct input images with calibrated pipe
    corrected_image = lynred.base_t.image_t()
    corrected_paths = []
    times = []
    for idx, img_path in enumerate(raw_image_paths):
        image = read_image(img_path)
        start = time.time()
        pipe.execute(image, corrected_image)
        times.append(time.time() - start)
        output_path = os.path.join(
            output_dir, img_path.split('/')[-1])
        corrected_image.save(output_path)
        corrected_paths.append(output_path)

    metrics = {
        'correction': {
            'mean': sum(times)/len(times),
            'total': sum(times),
            'N': len(times),
            'times': times
        }
    }

    # Save calibration times
    metrics_path = os.path.join(os.path.dirname(output_dir), 'correction.json')
    with open(metrics_path, 'w') as fobj:
        json.dump(metrics, fobj)


def parse_args():
    """Parse commandline arguments for training."""

    argparser = ArgumentParser(
        description='Correct and re-align raw IR images.')
    argparser.add_argument(
        '-c', '--config',
        type=str, required=True,
        help='Path to model config file.')
    argparser.add_argument(
        '-r', '--raw',
        type=str, required=True,
        help='Path to raw images directory.')
    argparser.add_argument(
        '-o', '--output',
        type=str, required=True,
        help='Path to output images directory.')
    argparser.add_argument(
        '-cd', '--calibration-dir',
        type=str, required=False,
        default='datasets/VRU/raw/calibrations',
        help='Path to output images directory.')
    argparser.add_argument(
        '-cold', '--cold',
        type=str, required=False,
        default=None,
        help='Path to cold images directory.')
    argparser.add_argument(
        '-hot', '--hot',
        type=str, required=False,
        default=None,
        help='Path to hot images directory.')
    argparser.add_argument(
        '-ref', '--reference',
        type=str, required=False,
        default=None,
        help='Path to cold images directory.')

    args = argparser.parse_args()

    return args


if __name__ == '__main__':

    """
    python calibrate.py \
    -c configs/shutter.json \
    -r ../datasets/VRU/raw/images \
    -o test_lynred \
    -cold ../datasets/VRU/raw/calibrations/Tamb=12/TCN=10 \
    -hot ../datasets/VRU/raw/calibrations/Tamb=40/TCN=40
    """

    args = parse_args()
    main(
        args.config, args.raw, args.output,
        calibration_dir=args.calibration_dir,
        cold_img_dir=args.cold,
        hot_img_dir=args.hot,
        ref_img_dir=args.reference
    )
