import cv2
import pytesseract
from numpy import intc

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def part1():
        # Load the image
        filename = 'es.png'

        # Load the image
        img = cv2.imread(filename)

        # convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # smooth the image to avoid noises
        gray = cv2.medianBlur(gray,5)
        gray = cv2.medianBlur(gray, 5)
       # gray = cv2.medianBlur(gray, 5)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

        # apply some dilation and erosion to join the gaps
        thresh = cv2.dilate(thresh,None,iterations = 4)
        cv2.imshow("cropped", thresh)
        cv2.waitKey(0)
        thresh = cv2.erode(thresh,None,iterations = 3)

        # Find the contours
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
        print(hierarchy)
        # For each contour, find the bounding rectangle and draw it
        cont=0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
           # cv2.rectangle(img,(x-12,y-12),(x+w+12,y+h+12),(0,255,0),2)
            cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
            crop_img = img[y-10:y + h+10, x-10:x + w+10]
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)
            cv2.imwrite(str(cont)+'.png',crop_img)
            cont+=1

        # Finally show the image
        cv2.imshow('img',img)
        cv2.imshow('res',thresh_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def part2():
    filename = 'es.png'

    # read the image and get the dimensions
    img = cv2.imread(filename)
    h, w, _ = img.shape  # assumes color image

    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img)  # also include any config options you use

    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # show annotated image and wait for keypress
    cv2.imshow(filename, img)
    cv2.waitKey(0)
from PIL import Image

filename = 'ww.jpg'

# Load the image
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray, 5)
gray = cv2.medianBlur(gray, 5)
 # run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_string(gray)
print(boxes)
part1()