import easyocr
import cv2
from matplotlib import pyplot as plt

# Read in image, setting path of image and reading through easy-orc module
Image_path ='test1.jpg'

reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(Image_path)



img = cv2.imread(Image_path)
for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    font = cv2.FONT_HERSHEY_PLAIN
# Draws a green rectangle over text and also prints the text.
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
    print(text)
    
# Opening the test file in both the reading and appending mode.

    with open('test.txt', 'a+') as file:

         # the file will save and close after you exit this block scope
        
         file.write(text)
         file.write("\n")
# cv. LINE_AA gives anti-aliased line which looks great for curves.
    img = cv2.putText(img, text, top_left, font, 3, (0,0,0), 2, cv2.LINE_AA)

plt.figure(figsize=(10, 10)) # increasing the size of image, makes us easier to see text
plt.imshow(img)
plt.show()
#img1=cv2.imwrite("NEWf.txt",text)