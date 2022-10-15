import cv2
import imutils as im
import pytesseract as pyt
import numpy as np
import re

pyt.pytesseract.tesseract_cmd = 'Tesseract/tesseract.exe'


image = cv2.imread('test1.jpg')
image = im.resize(image, width=300)

cv2.imshow("Original Image", image)
cv2.waitKey(0)

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Greyed Image", grey_image)
cv2.waitKey(0)

grey_image = cv2.bilateralFilter(grey_image, 11, 17, 17)
cv2.imshow("Smoothened Image", grey_image)
cv2.waitKey(0)

edged = cv2.Canny(grey_image, 30, 200)
cv2.imshow("Edged Image", edged)
cv2.waitKey(0)

cnts, new = cv2.findContours(
    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Contours", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = 0
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", image2)
cv2.waitKey(0)

i = 7
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    screenCnt = approx

    x, y, w, h = cv2.boundingRect(c)
    new_img = image[y:y+h, x:x+w]
    cv2.imwrite('./'+str(i)+'.png', new_img)
    i += 1
    break


cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Detected Registration Plate", image)
cv2.threshold(np.array(image), 125, 255, cv2.THRESH_BINARY)
cv2.waitKey(0)

Cropped_loc = './7.png'
cv2.imshow("Cropped", cv2.imread(Cropped_loc))

plate = pyt.image_to_string(Cropped_loc, lang='eng',
                            config='--psm 8 -c page_separator=""')

plate = re.sub(r'[^\w\s]', '', plate)


print(f"Registration Plate is: {plate}")

cv2.waitKey(0)
cv2.destroyAllWindows()
