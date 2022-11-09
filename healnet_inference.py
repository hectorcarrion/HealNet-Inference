from tensorflow import keras
from PIL import Image
import pandas as pd
import glob
from datetime import datetime
from tqdm.auto import tqdm
from tensorflow.keras.utils import array_to_img, img_to_array
import numpy as np
import os
import subprocess
import urllib.request

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
dir_ =  os.path.join(desktop, "HealNet-Inference")
model_path = dir_ + "\\" + "HealNet_cls.h5"

try: # Try Git Pull to current directory
    urllib.request.urlopen("https://github.com/hectorcarrion/HealNet-Inference")
    subprocess.call("git pull",shell=True, cwd = dir_ )
except:
    print("No internet connectionm, cannot pull.")





densenet_cluster_stage = {0:"Proliferation/Maturation",
                          1:"Hemostasis",
                          2:"Inflammatory"}
root_images = f"{desktop}/Porcine_Exp_Davis" # change this to real folder
prob_table_path = f"{desktop}/HealNet-Inference/prob_table.csv"

model = keras.models.load_model(model_path)
image_paths = glob.glob(f"{root_images}/**/*.jpg", recursive=True)

try:
    prob_table = pd.read_csv(prob_table_path)
except:
    headers = {"Image":[], "Time Processed":[], "Hemostasis":[],
                  "Inflammation":[], "Proliferation/Maturation":[]}
    table = pd.DataFrame.from_dict(headers)
    table.to_csv(prob_table_path, index=False)
    prob_table = pd.read_csv(prob_table_path)

processed_ctr = 0
for image in tqdm(image_paths):
    if image not in list(prob_table["Image"]):
        resized_im = Image.open(image).resize((128,128))
        image_data = img_to_array(resized_im)
        #image_data = densenet_preprocess(image_data) # densenet hardcoded!
        image_data = np.expand_dims(image_data, axis=0) # adds batch dim
        pred = model.predict(image_data, verbose=0)
        pred = pred.flatten()
        time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        prob_table.loc[len(prob_table)] = [image, time, pred[1], pred[2], pred[0]]
        processed_ctr += 1

prob_table.to_csv(prob_table_path, index=False)
if processed_ctr:
    print(f"Added {processed_ctr} new predictions to {prob_table_path}")
else:
    print(f"No new images found in {root_images}")
print("Running again in 1 hour.")

try:
    # Try Git Push to repo
    urllib.request.urlopen("https://github.com/hectorcarrion/HealNet-Inference")
    subprocess.call("git status",shell=True, cwd = dir_  )
    subprocess.call("git add .",shell=True, cwd = dir_  )
    subprocess.call("git commit -am \"autocommit\"",shell=True, cwd = dir_  )
    subprocess.call("git push",shell=True, cwd = dir_  )
except:
    print("No internet connection, cannot push.")
