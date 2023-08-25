import numpy as np
import os
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import matplotlib.pyplot as plt

n_classes = 80
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

logger = trt.Logger(trt.Logger.WARNING)
logger.min_severity = trt.Logger.Severity.ERROR
runtime = trt.Runtime(logger)
trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
with open('yolov8s_trt.engine', "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
context = engine.create_execution_context()
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append({'host': host_mem, 'device': device_mem})
    else:
        outputs.append({'host': host_mem, 'device': device_mem})

img = cv2.imread('image2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im1 = cv2.resize(img, (640, 640)).astype('float32')
input_image = im1.transpose((2,0,1))
input_image = input_image/255.0
input_image = np.expand_dims(im1, axis=0)

inputs[0]['host'] = np.ravel(input_image)
# transfer data to the gpu
for inp in inputs:
    cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # run inference
context.execute_async_v2(
    bindings=bindings,
    stream_handle=stream.handle)
    # fetch outputs from gpu
for out in outputs:
    cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    # synchronize stream
stream.synchronize()

data = [out['host'] for out in outputs]

outs = np.array(data)
outs = outs.reshape(1, 84, 8400)

pred = outs[0].transpose()
print(pred.shape)

boxes = pred[:,:4]
scores = np.amax(pred[:,4:], axis=1)
# nms = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.4, 0.5)
# print(len(nms))

# #print(type(scores), type(boxes), sep="======\n")
nms = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.5)
print(nms)
idx = nms[:,0]
res = []
for i in idx:
    if scores[i]>0.5:
        x1, y1, x2, y2 = pred[i,:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        scr = np.amax(pred[i,4:])
        cls = np.argmax(pred[i,4:])

        res.append([x1, y1, x2, y2, scr, cls])

image = cv2.imread('image2.jpg')
image = cv2.resize(image, (640,640))

for b in res:
    x1, y1, x2, y2 = b[:4]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = x1 - x2//2
    y1 = y1 - y2//2
    x2 = x1 + x2
    y2 = y1 + y2


    cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
cv2.imwrite('image111.jpg', image)
# plt.imshow('window', image)
# plt.show()
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()

# # box = res[:,:4]
# # scr = res[:,4]
# # nms_cv = cv2.dnn.NMSBoxes(box.tolist(), scr.tolist(), 0.5, 0.5)
# # print(nms_cv)