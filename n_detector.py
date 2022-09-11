import tensorflow.keras.models
import imutils.contours 
import numpy as np
import imutils
import cv2

print("[INFO] loading handwriting OCR model...")
model = tensorflow.keras.models.load_model("Data\\Trained Models\\Model.h5") # laods the trained model
def detector_op(imagepath):
    #----------VARIABLES---------


    sent = ""
    labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames += "0123456789"
    labelNames = [l for l in labelNames]
    words = []
    key = True
    letters = ""
    bb =[]
    chars = []
    #----------VARIABLES---------

    # The Image to be read 
    image = cv2.imread(imagepath)

    # Remove border
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    temp1 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    temp3 = cv2.add(temp1, temp2)
    result = cv2.add(temp3, image)

    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = imutils.contours.sort_contours(cnts,method = "top-to-bottom")[0] 
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #Extracting the area and resizing to 32*32 pixel inorder to predict the image
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if(w>=20 and w<=150) and (h>=20 and h<=120):

            roi = gray[y:y+h, x:x+w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width = 32)
            else : 
                thresh = imutils.resize(thresh, height =32)

            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right = dX, borderType = cv2.BORDER_CONSTANT, value = (0,0,0))
            padded = cv2.resize(padded, (32,32))
            padded = padded.astype("float32")/ 255.0
            padded = np.expand_dims(padded, axis = -1)

            chars.append((padded, (x, y, w, h)))

    boxes = [[b[1][0],b[1][1],b[1][2],b[1][3]] for b in chars]
    temp = boxes.copy()
    line_bottom = boxes[0][1]+boxes[0][3]-1
    line_begin_idx = 0
    #To sort the images in an order from top left to bottom right
    for i in range(len(boxes)):
        if boxes[i][1] > line_bottom:
            boxes[line_begin_idx:i] = sorted(boxes[line_begin_idx:i], key=lambda b: b[0])
            line_begin_idx = i
        line_bottom = max(boxes[i][1]+boxes[i][3]-1, line_bottom)
    boxes[line_begin_idx:] = sorted(boxes[line_begin_idx:], key=lambda b: b[0])

    aa = [c[0] for c in chars]

    for asd in boxes:
        for sdf in temp:
            if(asd == sdf):
                bb.append(aa[temp.index(sdf)])

    chars = np.array(bb, dtype = "float32")


    preds = model.predict(chars)

    qwsx = boxes.copy()


    #Predicting the letter in the image using the model we have trained
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        if(key):
            key = False
            letters+=label
        else:
            if(x - (qwsx[0][0] + qwsx[0][2]) > 17 or x - qwsx[0][0] < 0):
                words.append(letters)
                qwsx.pop(0)
                letters = label
            else:
                qwsx.pop(0)
                letters+=label
        print("[INFO] {} - {:.2f}% ".format(label, prob * 100),end = '')
        # print(f"Area = {w*h}")
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)

    if __name__ == '__main__':
        cv2.imshow("image", image)
        cv2.waitKey(0)

    if(letters != ""):
        words.append(letters)

    for we in words:
        sent = sent + we + " "
    # print(sent)
    return sent

if __name__ == '__main__':
    detector_op("Images\image2.png")