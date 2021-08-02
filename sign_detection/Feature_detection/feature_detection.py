import cv2
import numpy as np
import os


orb = cv2.ORB_create(nfeatures=500)
images = []
className = []
train_image_path = 'img'
myList = os.listdir(train_image_path)
for cl in myList:
    imgcur = cv2.imread(f'{train_image_path}/{cl}', 0)
    images.append(imgcur)
    className.append(os.path.splitext(cl)[0])
def findDes(images):
        desList = []
        for imgg in images:
            kp, des = orb.detectAndCompute(imgg, None)
            desList.append(des)
        return desList

def findID(img, desList, thres=30):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:

        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.85 * n.distance:
                    good.append([m])

            matchList.append(len(good))

    except:
        pass

    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))

    return finalVal
img_sexy=cv2.imread("turn_right.png")
img_gray=cv2.cvtColor(img_sexy,cv2.COLOR_BGR2GRAY)
deslist = findDes(images)
id =findID(img_gray, deslist)
out = 0
if id != -1:
    out = className[id]
print(out)
cv2.imshow("out",img_sexy)