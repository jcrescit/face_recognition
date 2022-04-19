# Selective Search using Salvador Dali's image

import selectivesearch
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

img = cv2.imread('D:\Computer_Vision\computer__vision\Computer Vision\selective search\image\dali.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape : ', img.shape)

_, regions = selectivesearch.selective_search(img_rgb, scale = 100, min_size = 5000)

# color
green = (125, 255, 51)
red = (255, 0, 0)

cand_rects = [cand['rect'] for cand in regions if cand['size'] > 3000]
gt_box = [290, 10, 430, 230]
img_rgb = cv2.rectangle(img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color = red, thickness = 2)


# IOU setting
def compute_iou(cand_box, gt_box):

    # calculate intersection areas
    # cand_box : 예측한 bounding box (selective search에서 추천해준 box), gt_box : ground truth bounding box
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.maximum(cand_box[2], gt_box[2])
    y2 = np.maximum(cand_box[3], gt_box[3])

    intersection = np.maximum(x2 -x1, 0) * np.maximum(y2 -y1, 0)

    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection

    iou = intersection / union
    return iou

for index, cand_box in enumerate(cand_rects):

    cand_box = list(cand_box)
    cand_box[2] += cand_box[0]
    cand_box[1] += cand_box[3]
    
    iou = compute_iou(cand_box, gt_box)

    if iou >0.5:
        print('index:', index, "iou:", iou, 'rectangle"', (cand_box[0], cand_box[1], cand_box[2], cand_box[3]))
        cv2.rectangle(img_rgb, (cand_box[0], cand_box[1]), (cand_box[2],  cand_box[3]), color = green, thickness = 1)
        text = "{}: {:.2f}".format(index, iou)
        cv2.putText(img_rgb, text, (cand_box[0] + 100, cand_box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color = green, thickness = 1)

plt.figure(figsize = (12, 12))
plt.imshow(img_rgb)
plt.show()