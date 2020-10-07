#Recognizing Handwritten Numbers
import cv2
import numpy as np
                                                                                # generate the  Number set
txtSize, baseline = cv2.getTextSize('0123456789', cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
                                                                                # creating an image of size txt_Size
digits_img = np.zeros((txtSize[1] + 7, txtSize[0]), np.uint8)
cv2.putText(digits_img, '0123456789', (0, txtSize[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

cnts, hierarchy = cv2.findContours(digits_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts.sort(key=lambda c: cv2.boundingRect(c)[0])

digits = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    digits.append(digits_img[y:y + h, x:x + w])


def detect(img):                                                        #the draw
    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6), (3, 3))
    dilat = cv2.dilate(img, elem, iterations=3)
                                                                        # find the drawed digit
    cnts, hierarchy = cv2.findContours(dilat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(cnts[0])
    roi = dilat[y:y + h, x:x + w]
                                                                        # find the best match
    percent_white_pix = 0
    digit = -1
    for i, d in enumerate(digits):
        scaled_roi = cv2.resize(roi, d.shape[:2][::-1])
        bitwise = cv2.bitwise_and(d, cv2.bitwise_xor(scaled_roi, d))
                                                                        # match is given by the highest loss of white pixel
        before = np.sum(d == 255)
        matching = 100 - (np.sum(bitwise == 255) / before * 100)
        if percent_white_pix < matching:
            percent_white_pix = matching
            digit = i

    return digit
''' Drawing on screen using pointer '''
drawing = False                                                         #mouse is pressed
pt1_x, pt1_y = None, None
# mouse callback function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
        cv2.rectangle(img, (0, 0, 512, 512), (0, 0, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)
        digit = detect(img)

        cv2.putText(img, 'It is a %d' % digit, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

img = np.zeros((360, 512, 1), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw', line_drawing)

while (1):
    cv2.imshow('test draw', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

''' End drawing '''
cv2.destroyAllWindows()
