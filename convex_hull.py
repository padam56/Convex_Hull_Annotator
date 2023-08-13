import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
import tensorflow_addons as tfa
import random as rng
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rng.seed(12345)

image_dir = "/home/pthapa2/snap/padam/SandBoilNet/datasets/videoframe"
output_dir = "/home/pthapa2/snap/padam/SandBoilNet/output_images"
annotated_image_dir = "/home/pthapa2/snap/padam/SandBoilNet/annotated_images"
json_file = '/home/pthapa2/snap/padam/SandBoilNet/convex_hull_coordinates.json'

os.makedirs(annotated_image_dir, exist_ok=True)

def load_custom_model(model_name, custom_layer=False):
    model_path = os.path.join("/home/pthapa2/snap/padam/SandBoilNet", "models", str(model_name), "best_model.h5")
    tf.keras.backend.clear_session()

    if custom_layer:
        best_model = keras_load_model(model_path, custom_objects={'GroupNormalization': tfa.layers.GroupNormalization}, compile=False)
    else:
        best_model = keras_load_model(model_path, compile=False)

    print(f'=========Loaded {model_name}===========')
    return best_model

model = load_custom_model('SandBoilNet_Dropout_Without_PCA_bce_dice_loss_new_withgan', custom_layer=True)

all_hull_coords = {}
start_time = time.time()

for image_name in os.listdir(image_dir):
    print(f"Processing image: {image_name}")

    img = cv2.imread(os.path.join(image_dir, image_name))
    img = cv2.resize(img, (512, 512))
    img = img / 255.0

    pred_mask = model.predict(img[np.newaxis, ...])[0]
    pred_mask = np.max(pred_mask, axis=-1)
    thresh_mask = (pred_mask > np.mean(pred_mask)).astype(np.uint8)
    labeled_mask = label(thresh_mask)

    hull_list = []
    for region in regionprops(labeled_mask):
        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            region_image = labeled_mask[minr:maxr, minc:maxc] == region.label
            hull = convex_hull_image(region_image)
            contours, _ = cv2.findContours(hull.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt + np.array([[minc, minr]]) for cnt in contours]
            hull_list.extend(contours)

    if len(hull_list) == 0:
        print(f"No convex hulls found for {image_name}.")
        continue

    annotated_image = (img * 255).astype(np.uint8).copy()
    for i in range(len(hull_list)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(annotated_image, hull_list, i, color, lineType=cv2.LINE_AA) # Added lineType for smooth boundaries

    cv2.imwrite(os.path.join(annotated_image_dir, f"{image_name}_annotated.png"), annotated_image)

    regions = []
    for hull in hull_list:
        hull_points = [[int(point[0][0]), int(point[0][1])] for point in hull]
        regions.append({"shape_attributes": {"name": "polygon", "all_points_x": [point[0] for point in hull_points], "all_points_y": [point[1] for point in hull_points]}})
    all_hull_coords[image_name] = {"filename": image_name, "size": img.shape[:2], "regions": regions, "file_attributes": {}}

    print(f"Convex hull coordinates for {image_name} saved.")

with open(json_file, 'w') as f:
    json.dump(all_hull_coords, f)

print("All images processed and results saved to JSON.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds.")

# Create a video from the annotated images
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (512, 512))

for image_name in sorted(os.listdir(annotated_image_dir)):
    frame = cv2.imread(os.path.join(annotated_image_dir, image_name))
    out.write(frame)

out.release()
print("Video created from annotated images.")