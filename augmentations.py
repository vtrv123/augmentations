#%%
import random
import albumentations as A
import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy.random as random
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

#%%
def yolo2bbox(bboxes):
    """
    Function to convert bounding boxes in YOLO format to 
    xmin, ymin, xmax, ymax.
    
    Parmaeters:
    :param bboxes: Normalized [x_center, y_center, width, height] list
    return: Normalized xmin, ymin, xmax, ymax
    """
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def draw_boxes(image, bboxes):
    """
    Function accepts an image and bboxes list and returns
    the image with bounding boxes drawn on it.
    Parameters
    :param image: Image, type NumPy array.
    :param bboxes: Bounding box in Python list format.
    :param format: One of 'coco', 'voc', 'yolo' depending on which final
        bounding noxes are formated.
    Return
    image: Image with bounding boxes drawn on it.
    box_areas: list containing the areas of bounding boxes.
    """
    box_areas = []

    # need the image height and width to denormalize...
    # ... the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        width = xmax - xmin
        height = ymax - ymin
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=2
        ) 
        box_areas.append(width*height) 
    return image, box_areas


def transform(image,bbox):
  

    transform = A.Compose([
        A.VerticalFlip(p=0.3),
        A.GaussianBlur(p=0.2,blur_limit=(3,9)),
        A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0,
        always_apply=False, p=0.5) 
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'],
        min_area = 0
    ))

    transformed_instance = transform(
        image=image, bboxes=bbox, class_labels=['granito']
    )

    transformed_image = transformed_instance['image']
    transformed_bboxes = transformed_instance['bboxes']

    return(transformed_image,transformed_bboxes)

#%%
images_list = glob.glob('/home/vitor/Desktop/PG/augmentation/dataset/*.jpg')
labels_list = glob.glob('/home/vitor/Desktop/PG/augmentation/dataset/*.txt')

#%%
random_images = random.randint(0,338, size = 900)

for image_num in random_images:
    name = images_list[image_num][:-4]
    image = cv2.imread(name + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    text_file = open((name + '.txt'), 'r')
    label = text_file.readline()
    text_file.close()

    bbox = [[float(coord) for coord in label[2:].split(' ')]]

    transformed_image, transformed_bboxes = transform(image, bbox)

    random_name = str(random.randint(500,500000))

    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('/home/vitor/Desktop/PG/augmentation/augmented/'+ random_name+ '.jpg', transformed_image)
    
    bbox_new = [str(coord) for coord in transformed_bboxes[0]]
    bbox_new = ' '.join(bbox_new)
    bbox_new = '0 ' + bbox_new
    with open('/home/vitor/Desktop/PG/augmentation/augmented/'+ random_name + '.txt', 'w+') as f:
        f.write(bbox_new)
    f.close()



#%%


# image = cv2.imread(name + '.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# text_file = open((name + '.txt'), 'r')
# label = text_file.readline()

# bbox = [[float(coord) for coord in label[2:].split(' ')]]
# #%%

# transformed_image, transformed_bboxes = transform(image, bbox)
# annot_image, box_areas = draw_boxes(transformed_image, transformed_bboxes)

# #%%
# plt.imshow(transformed_image)


#%%
# image = cv2.imread('dataset/1_0_left-2-_png.rf.f5088362f3500c1b62986c9b8352a879.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# bbox = [[0.50390625, 0.541015625, 0.658984375, 0.58984375]]

