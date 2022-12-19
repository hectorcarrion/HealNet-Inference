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

try: # Try Git Pull to current directory
    urllib.request.urlopen("https://github.com/hectorcarrion/HealNet-Inference")
    subprocess.call("git pull", shell=True, cwd=dir_)
except:
    print("No internet connectionm, cannot pull.")

# Read from file in future
avg_dv = np.array([108.16076384,  61.49104917,  55.44175686])
color_correct = True
stage_cls = {"Proliferation/Maturation":0,
             "Hemostasis":1,
             "Inflammatory":2}

# Handles windows specific paths well
root_images = Path(f"{desktop}\Porcine_Exp_Davis")
prob_table_path = f"{desktop}/HealNet-Inference/prob_table.csv"

model = keras.models.load_model(model_path)

# fixing windows path bug as per 
# https://stackoverflow.com/questions/5629242/getting-every-file-in-a-windows-directory
image_paths = list(root_images.glob("**/*.jpg"))

try:
    prob_table = pd.read_csv(prob_table_path)
    if len(prob_table.columns) >= 6: # change if needed
        save_blur = True
        import cv2
    else:
        save_blur = False
except:
    headers = {"Image":[], "Time Processed":[], "Blur":[], "Hemostasis":[],
               "Inflammation":[], "Proliferation/Maturation":[]}
    save_blur = True
    import cv2
    table = pd.DataFrame.from_dict(headers)
    table.to_csv(prob_table_path, index=False)
    prob_table = pd.read_csv(prob_table_path)

processed_ctr = 0
for image in tqdm(image_paths):
    if str(image) not in list(prob_table["Image"]):
        try:
            resized_im = Image.open(image).resize((128,128))
            image_data = img_to_array(resized_im)
            if color_correct:
                img_avg = image_data.mean(axis=(0,1))
                image_data = np.clip(image_data + 
                                     np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)
            #image_data = densenet_preprocess(image_data) # densenet hardcoded!
            image_data = np.expand_dims(image_data, axis=0) # adds batch dim
            pred = model.predict(image_data, verbose=0)
            pred = pred.flatten()
            time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if save_blur:
                blur = get_blur(image)
                prob_table.loc[len(prob_table)] = [str(image), time, blur,
                                                   pred[stage_cls["Hemostasis"]],
                                                   pred[stage_cls["Inflammatory"]],
                                                   pred[stage_cls["Proliferation/Maturation"]]]
            else:
                prob_table.loc[len(prob_table)] = [str(image), time,
                                                   pred[stage_cls["Hemostasis"]],
                                                   pred[stage_cls["Inflammatory"]],
                                                   pred[stage_cls["Proliferation/Maturation"]]]
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
