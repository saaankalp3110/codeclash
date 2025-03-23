#Needed imports
import cv2
import image_dehazer
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def calculate_IoU(rect1, rect2):

    #Gets box coord inputs, disregards class.
    x1_r1, y1_r1, x2_r1, y2_r1, discard1 = rect1
    x1_r2, y1_r2, x2_r2, y2_r2, discard2 = rect2

    area_r1 = (x2_r1 - x1_r1) * (y2_r1 - y1_r1)
    area_r2 = (x2_r2 - x1_r2) * (y2_r2 - y1_r2)

    #Finds intersection and intersetc area
    x_intersect_start = max(x1_r1, x1_r2)
    y_intersect_start = max(y1_r1, y1_r2)
    x_intersect_end = min(x2_r1, x2_r2)
    y_intersect_end = min(y2_r1, y2_r2)

    intersection_area = max(0, x_intersect_end - x_intersect_start) * max(0, y_intersect_end - y_intersect_start)

    #Finds union area, return IoU
    union_area = area_r1 + area_r2 - intersection_area
    return intersection_area/union_area

#Run command for YOLO neural network
def runYOLO(image, YOLO_model='yolov8m.pt', dehaze=False, derain=False):

    #Dummy list for coordinate storage and passing
    rect=[]

    # Load YOLO model (in case of changes)
    model = YOLO(YOLO_model)
    
    # Run any deraining/dehazing. Brighten final product. Pass said product into YOLO for analysis
    if derain==True:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    if dehaze==True:
        image, _ = image_dehazer.remove_haze(image)
    image=cv2.convertScaleAbs(image, alpha=1, beta=50)
    results = model(image)

    # Class names for YOLO, ChatGPT generated
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

    # Process results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{classNames[cls]} {conf:.2f}"

            # Draw bounding box and label, save rectangular coords
            if classNames[cls] in ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]:
                rect.append((x1,y1,x2,y2,classNames[cls]))
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    #pass on coords and newly scanned image
    rect = sorted(rect,key=lambda x: x[0])
    return image,rect



#______ Run code start _____:

#Image is loaded from path, then put into YOLO twice; once normal, again with dehazing on.
img_dir=r'C:\Users\whate\OneDrive\Documents\Python Scripts\CodeClash24\Foggy_Driving\leftImg8bit\test_extra\web\web_2269771660_d43b11e65e_leftImg8bit.png'
img_base=cv2.imread(img_dir)
base_scan,rect_base=runYOLO(img_base)
dehaze_scan,rect_dehaze=runYOLO(img_base, dehaze=True)

#Output box coordinates and images, destroy windows if 0 is pressed and tabs are clicked on
#print(f"Base Image box coords: {rect_base}")
#print(f"Dehazed Image box coords: {rect_dehaze}")
cv2.imshow("Scanned Image",base_scan)
cv2.imshow("Dehazed Image",dehaze_scan)
cv2.waitKey(0)
cv2.destroyAllWindows()

#True Positives, False Positives
TP={"person":0, "bicycle":0, "car":0, "motorbike":0, "aeroplane":0, "bus":0, "train":0, "truck":0, "boat":0, "traffic light":0, "fire hydrant":0, "stop sign":0, "parking meter":0, "bench":0}
FP={"person":0, "bicycle":0, "car":0, "motorbike":0, "aeroplane":0, "bus":0, "train":0, "truck":0, "boat":0, "traffic light":0, "fire hydrant":0, "stop sign":0, "parking meter":0, "bench":0}


#Look at the picture input, find the ground truths (Fase Negatives in code) prior to running!!
FN={"person":0, "bicycle":0, "car":3, "motorbike":0, "aeroplane":0, "bus":0, "train":0, "truck":0, "boat":0, "traffic light":1, "fire hydrant":0, "stop sign":0, "parking meter":0, "bench":0}

#Run through each coordinate and its pair to find IoU
for keys in range(len(rect_base)):
    if rect_base[keys][4]==rect_dehaze[keys][4]:
        IoU=calculate_IoU(rect_base[keys],rect_dehaze[keys])
    else:
        IoU=0

    
    if IoU>=0.5:
        TP[rect_base[keys][4]]+=1
    else:
        FP[rect_base[keys][4]]+=1

#Sort classes by their value. 
TP = dict(sorted(TP.items(), key=lambda item: item[1], reverse=True))
FP = dict(sorted(FP.items(), key=lambda item: item[1], reverse=True))
FN = dict(sorted(FN.items(), key=lambda item: item[1], reverse=True))

#Loop Count is used to see total classes used, PR is a list of tuples for Prescision/Recall
LoopCount=0
PR=[]

#Calculates prescision and recall for each class. If it doesnt exist in ground truths, it ends.
for keys in FN.keys():
    if FN[keys]==0:
        break
    if TP[keys]==0:
        PR.append((0,0))
    else:
        PR.append((((TP[keys])/(TP[keys]+FP[keys])),((TP[keys])/(TP[keys]+FN[keys]))))
    LoopCount+=1

#Sums up precision across classes
P=0
for x in PR:
    P+=x[0]

#Final output:Total precision divided by # of seen classes : closest to mean average precision as with the 
# small pictures our dataset provides, mAP normally is 0 which isnt representative of our model.
print(f"Average Precision for Pictures: {P/LoopCount}")
