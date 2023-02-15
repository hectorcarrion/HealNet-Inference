from tensorflow import keras
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from tensorflow.keras.utils import img_to_array
import numpy as np
import os
from pathlib import Path
from skimage import color as skcolor
from skimage import filters as skfilters
import cv2

# Constants
avg_dv = np.array([108.16076384,  61.49104917,  55.44175686])
color_correct = True
center = True
max_noise_level = 10000
stage_idx = {"Maturation":0,
             "Hemostasis":1,
             "Inflammatory":2,
             "Proliferation":3}
crop_size = 1024

if os.name == 'nt':
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    root_images = Path(f"{desktop}\Porcine_Exp_Davis")
    # fixing windows path bug as per 
    # https://stackoverflow.com/questions/5629242/getting-every-file-in-a-windows-directory
    image_paths = list(root_images.glob("**/*.jpg"))
else:
    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    root_images = Path(f"{desktop}/Porcine_Exp_Davis")
    # fixing windows path bug as per 
    # https://stackoverflow.com/questions/5629242/getting-every-file-in-a-windows-directory
    image_paths = list(root_images.glob("**/*.jpg"))

dir_ =  os.path.join(desktop, "HealNet-Inference")
model_path = os.path.join(dir_, "HealNet_cls.h5")
prob_table_path = f"{desktop}/HealNet-Inference/prob_table.csv"

class GaussianBlur(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Applies a Gaussian Blur with random sigma to an image.
    Args:
        kernel_size: int, 2 element tuple or 2 element list. x and y dimensions for
            the kernel used. If tuple or list, first element is used for the x dimension
            and second element is used for y dimension. If int, kernel will be squared.
        sigma: float, 2 element tuple or 2 element list. Interval in which sigma should
            be sampled from. If float, interval is going to be [0, float), else the
            first element represents the lower bound and the second element the upper
            bound of the sampling interval.
    """

    def __init__(self, kernel_size, sigma, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

        if isinstance(kernel_size, (tuple, list)):
            self.x = kernel_size[0]
            self.y = kernel_size[1]
        else:
            if isinstance(kernel_size, int):
                self.x = self.y = kernel_size
            else:
                raise ValueError(
                    "`kernel_size` must be list, tuple or integer "
                    ", got {} ".format(type(self.kernel_size))
                )

        if isinstance(sigma, (tuple, list)):
            self.sigma_min = sigma[0]
            self.sigma_max = sigma[1]
        else:
            self.sigma_min = type(sigma)(0)
            self.sigma_max = sigma

        if not isinstance(self.sigma_min, type(self.sigma_max)):
            raise ValueError(
                "`sigma` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.sigma_min), type(self.sigma_max)
                )
            )

        if self.sigma_max < self.sigma_min:
            raise ValueError(
                "`sigma` cannot have upper bound less than "
                "lower bound, got {}".format(sigma)
            )

        self._sigma_is_float = isinstance(self.sigma, float)
        if self._sigma_is_float:
            if not self.sigma_min >= 0.0:
                raise ValueError(
                    "`sigma` must be higher than 0"
                    "when is float, got {}".format(sigma)
                )

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        sigma = self.get_sigma()
        blur_v = GaussianBlur.get_kernel(sigma, self.y)
        blur_h = GaussianBlur.get_kernel(sigma, self.x)
        blur_v = tf.reshape(blur_v, [self.y, 1, 1, 1])
        blur_h = tf.reshape(blur_h, [1, self.x, 1, 1])
        return (blur_v, blur_h)

    def get_sigma(self):
        sigma = self._random_generator.random_uniform(
            shape=(), minval=self.sigma_min, maxval=self.sigma_max
        )
        return sigma

    def augment_image(self, image, transformation=None):

        image = tf.expand_dims(image, axis=0)

        num_channels = tf.shape(image)[-1]
        blur_v, blur_h = transformation
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
        blurred = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding="SAME"
        )
        blurred = tf.nn.depthwise_conv2d(
            blurred, blur_v, strides=[1, 1, 1, 1], padding="SAME"
        )

        return tf.squeeze(blurred, axis=0)

    @staticmethod
    def get_kernel(sigma, filter_size):
        x = tf.cast(
            tf.range(-filter_size // 2 + 1, filter_size // 2 + 1), dtype=tf.float32
        )
        blur_filter = tf.exp(
            -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0))
        )
        blur_filter /= tf.reduce_sum(blur_filter)
        return blur_filter

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma, "kernel_size": self.kernel_size})
        return config

def get_blur(image_path_obj):

    image_path = str(image_path_obj)
    
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image and take the absolute value
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)

    # Compute the standard deviation of the Laplacian values
    std_dev = lap.std()

    # Return the blur level as a value between 0 and 1
    return std_dev / 255

def aggregate(probs, stage_idx):

    hemo_total = 0
    inf_total = 0
    prolif_total = 0
    matu_total = 0

    for prob in probs:
        hemo = prob[stage_idx["Hemostasis"]]
        inf = prob[stage_idx["Inflammatory"]]
        prolif = prob[stage_idx["Proliferation"]]
        matu = prob[stage_idx["Maturation"]]

        hemo_total += hemo
        inf_total += inf
        prolif_total += prolif
        matu_total += matu

    hemo_avg = hemo_total / len(probs)
    inf_avg = inf_total / len(probs)
    prolif_avg = prolif_total / len(probs)
    matu_avg = matu_total / len(probs)

    return hemo_avg, inf_avg, prolif_avg, matu_avg

try:
    prob_table = pd.read_csv(prob_table_path)

except:
    headers = {"Image":[], "Time Processed":[], "Blur":[], "Patches": [], "Hemostasis":[],
               "Inflammation":[], "Proliferation":[], "Maturation":[]}

    table = pd.DataFrame.from_dict(headers)
    table.to_csv(prob_table_path, index=False)
    prob_table = pd.read_csv(prob_table_path)

model = keras.models.load_model(model_path, custom_objects={"GaussianBlur": GaussianBlur})
processed_ctr = 0
for image in tqdm(image_paths):
    if str(image) not in list(prob_table["Image"]):
        try:
            blur = get_blur(image)
            device_image = img_to_array(Image.open(image))

            if color_correct:
                img_avg = device_image.mean(axis=(0,1))
                device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)

            if center:
                device_image = device_image[1000:4000, 1500:5500]

            gray = skcolor.rgb2gray(device_image/255)
            blurred_image = skfilters.gaussian(gray, sigma=1.0)
            thresh = blurred_image > 0.5

            max_y, max_x, _ = device_image.shape
            ys = np.random.randint(0, max_y-crop_size, 5)
            xs = np.random.randint(0, max_x-crop_size, 5)
            # Max tries is 5x5 or 25
            tries = np.array(np.meshgrid(ys, xs)).T.reshape(-1, 2)

            preds = []
            for y, x in tries:
                # good crop
                if np.count_nonzero(thresh[y:y+crop_size, x:x+crop_size]) < max_noise_level:
                    patch = Image.fromarray(device_image[y:y+crop_size, x:x+crop_size].astype(np.uint8))
                    patch = patch.resize((128,128))
                    image_data = img_to_array(patch)

                    #image_data = densenet_preprocess(image_data) # densenet hardcoded!
                    image_data = np.expand_dims(image_data, axis=0) # adds batch dim
                    pred = model.predict(image_data, verbose=0)
                    pred = pred.flatten()
                    preds.append(pred)

            hemo, infl, prol, matu = aggregate(preds, stage_idx)

            time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            prob_table.loc[len(prob_table)] = [str(image), time, blur, len(preds),
                                               hemo, infl, prol, matu]
            processed_ctr += 1
        except Exception as e:
            print(f"Unable to open {image} (check if corrupted). Skipping...")
            print(f"Exception: {e}")

prob_table.to_csv(prob_table_path, index=False)

if processed_ctr:
    print()
    print(f"Added {processed_ctr} new predictions to {prob_table_path}")
else:
    print()
    print(f"No new images found in {root_images}")
    
