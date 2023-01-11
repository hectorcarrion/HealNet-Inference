from tensorflow import keras
from PIL import Image
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from tensorflow.keras.utils import array_to_img, img_to_array
import numpy as np
import os
from pathlib import Path
import subprocess
import urllib.request
from skimage import color as skcolor
from skimage import filters as skfilters
import cv2

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
dir_ =  os.path.join(desktop, "HealNet-Inference")
model_path = dir_ + "\\" + "HealNet_cls.h5"

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


try: # Try Git Pull to current directory
    urllib.request.urlopen("https://github.com/hectorcarrion/HealNet-Inference")
    subprocess.call("git pull", shell=True, cwd=dir_)
except:
    print("No internet connectionm, cannot pull.")

# Read from file in future
avg_dv = np.array([108.16076384,  61.49104917,  55.44175686])
color_correct = True
center = True
max_noise_level = 10000
stage_idx = {"Maturation":0,
             "Hemostasis":1,
             "Inflammatory":2,
             "Proliferation":3}

# Handles windows specific paths well
root_images = Path(f"{desktop}\Porcine_Exp_Davis")
prob_table_path = f"{desktop}/HealNet-Inference/prob_table.csv"

model = keras.models.load_model(model_path)

# fixing windows path bug as per 
# https://stackoverflow.com/questions/5629242/getting-every-file-in-a-windows-directory
image_paths = list(root_images.glob("**/*.jpg"))

try:
    prob_table = pd.read_csv(prob_table_path)

except:
    headers = {"Image":[], "Time Processed":[], "Blur":[], "Patches": [], "Hemostasis":[],
               "Inflammation":[], "Proliferation":[], "Maturation":[]}

    table = pd.DataFrame.from_dict(headers)
    table.to_csv(prob_table_path, index=False)
    prob_table = pd.read_csv(prob_table_path)

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
        except:
            print(f"Unable to open {image} (check if corrupted). Skipping...")

prob_table.to_csv(prob_table_path, index=False)

if processed_ctr:
    print(f"Added {processed_ctr} new predictions to {prob_table_path}")
else:
    print(f"No new images found in {root_images}")
print("Running again in 1 hour.")

try:
    # Try Git Push to repo
    urllib.request.urlopen("https://github.com/hectorcarrion/HealNet-Inference")
    subprocess.call("git status", shell=True, cwd=dir_)
    subprocess.call("git add .", shell=True, cwd=dir_)
    subprocess.call("git commit -am \"autocommit\"",shell=True, cwd = dir_  )
    subprocess.call("git push", shell=True, cwd=dir_)
except:
    print("No internet connection, cannot push.")
