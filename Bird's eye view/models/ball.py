import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

classes_to_filter = ["ball"]
opt  = {
            
            "weights": r".\weights\player.pt", # Path to weights file default weights are for nano model
            "yaml"   : "data.yaml",
            "img-size": 640, # default image size
            "conf-thres": 0.15, # confidence threshold for inference.
            "iou-thres" : 0.45, # NMS IoU threshold for inference.
            # "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
            "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
            "classes" : classes_to_filter  # list of classes to filter or None
}

# Initializing model and setting it for inference
with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    half = False
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    classes = None
    if opt['classes']:
        classes = []
        for class_name in opt['classes']:
            classes.append(names.index(class_name))

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect_ball(img0, f_track, ball_xyxy ,count):
    with torch.no_grad():
        have_ball = False
        y_max, x_max = img0.shape[:2]
        y_max -= 1
        x_max -= 1
        if f_track == False:
            # opt['conf-thres'] = 0.2
            opt["weights"] = r".\weights\player.pt"
            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment= False)[0]



            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
            t2 = time_synchronized()
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det): # det(conf:high->low)
                        f_track = True
                        ball_xyxy = xyxy
                        count = 0
                else:
                    f_track = False
        if f_track == True:
            # h = int(ball_xyxy[3])-int(ball_xyxy[1])
            # w = int(ball_xyxy[2])-int(ball_xyxy[0])
            # hw = (h+w)/2
            # hw = 10
            # x0 = int(int(xyxy[0]) - 10*hw)
            # y0 = int(int(xyxy[1]) - 7*hw)
            # x1 = int(int(xyxy[2]) + 10*hw)
            # y1 = int(int(xyxy[3]) + 7*hw)
            x0 = int((int(ball_xyxy[0]) + int(ball_xyxy[2]))/2 - x_max/6)
            y0 = int((int(ball_xyxy[1]) + int(ball_xyxy[3]))/2 - y_max/6)
            x1 = int((int(ball_xyxy[0]) + int(ball_xyxy[2]))/2 + x_max/6)
            y1 = int((int(ball_xyxy[1]) + int(ball_xyxy[3]))/2 + y_max/6)
            if y0 < 0:
                y1 -= y0
                y0 = 0
            if y1 > y_max:
                y0 = y0 - (y1 - y_max)
                y1 = y_max
            if x0 < 0:
                x1 -= x0
                x0 = 0
            if x1 > x_max:
                x0 = x0 - (x1 - x_max)
                x1 = x_max  
            if y0 < 0:
                y0 = 0
            if x0 < 0:
                x0 = 0
            # opt['conf-thres'] = 0.2
            opt["weights"] = r".\weights\ball.pt"
            # opt["weights"] = r"C:\yolov7\yolov7\player.pt"
            img0 = img0[y0:y1, x0:x1] 
            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment= False)[0]


            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
            t2 = time_synchronized()
            
            for i, det in enumerate(pred):
                if len(det):
                    count = 0
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det): # det(conf:high->low)
                        ball_xyxy = xyxy
                        have_ball = True
            if have_ball:
                ball_xyxy[0] += x0
                ball_xyxy[1] += y0
                ball_xyxy[2] += x0
                ball_xyxy[3] += y0
            else:
                count +=1
        if count >= 10:
            f_track = False
        return f_track, ball_xyxy ,count, have_ball