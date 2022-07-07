import yolov5
import torch

#pre-trained model
model = yolov5.load('yolov5s.pt')


# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image


#image can be file,path,PIL,opencv,numpy,list
image = 'https://ultralytics.com/images/zidane.jpg'


model = torch.hub.load('ultralytics/yolov5','yolov5s')
results = model(image)
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

#save results
results.save()





