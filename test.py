from yolo_with_trt_plugins import TrtYOLO
import cv2
import pycuda.autoinit
import os

class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

model = TrtYOLO('./model/yolov4-tiny.trt')

img_dir = './image'
save_path = './images_detect/'
label = './labels/'

for image in os.listdir(img_dir):

     img_path = os.path.join(img_dir, image)
     img = cv2.imread(img_path)

     boxes, score, classes = model.detect(img, 0.5)

     i = 0
     for box in boxes:
         x1, y1, x2, y2 = box
         cls = int(classes[i])
         name = class_names[cls]
         cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
         cv2.putText(img, f'{name}(detected)', (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3, cv2.LINE_AA)

     save = os.path.join(save_path, image)
     cv2.imwrite(save, img)
     i+=1

print('successfully  detect!!')

compare = './compare/'
for im in os.listdir(save_path):

    image = cv2.imread(save_path + im)
    hi, wi = image.shape[0], image.shape[1]
    name = im.split('.')[0]
    
    try:
        with open('./labels/'+name+'.txt', 'r') as file:

            for line in  file:
                line = line.strip()

                box = line.split(' ')
                #print(type(box))

                cls = int(box[0])
                x1, y1, x2, y2 = box[1:]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                x1 = x1-x2/2.0
                y1 = y1-y2/2.0
                x2 = x1+x2
                y2 = y1+y2
                x1, y1, x2, y2 = int(x1*wi), int(y1*hi), int(x2*wi), int(y2*hi)

                name = class_names[cls]
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(image, f'{name}(by_text)', (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3, cv2.LINE_AA)
            
        save_comp = os.path.join(compare, im)
        cv2.imwrite(save_comp, image)

    except :
        pass

print('successfully compared')
