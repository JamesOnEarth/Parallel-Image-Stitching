import cv2
import time
import numpy as np
from PIL import Image

# SIFT feature extraction
def findKeyPoints(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    (kps, features) = sift.detectAndCompute(image, None)

    return (kps, features)


# Bruteforce matching
def matchKeyPoints(feature1, feature2, ratio = 0.75):
    match = cv2.BFMatcher()
    matches = match.knnMatch(feature1, feature2, k=2)

    good = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    return good

def getHomography(kp1, kp2, matches):
    if len(matches) <= 4:
        return None

    kpA = np.float32([kp.pt for kp in kp1])
    kpB = np.float32([kp.pt for kp in kp2])

    ptsA = np.float32([kpA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpB[m.trainIdx] for m in matches])
    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4.0)

    return (H, status)

def crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    (x, y, w, h) = cv2.boundingRect(cnt)

    return image[y:y + h, x:x + w]

image_paths=['yosemite1.jpg', 'yosemite2.jpg', 'yosemite3.jpg', 'yosemite4.jpg']
# image_paths=['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg']
# image_paths=['test1.jpg', 'yosemite3.jpg']

imgs = []

for i in range(len(image_paths)):
    img = cv2. imread(image_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)
    output = Image.fromarray(img)
    output.save(str(i) + "test.jpg")

start = time.time()

image = imgs[0]
for i in range(1,len(imgs)):
    (kp1, feature1) = findKeyPoints(image)
    (kp2, feature2) = findKeyPoints(imgs[i])
    matches = matchKeyPoints(feature1, feature2)
    homography = getHomography(kp1, kp2, matches)

    width = image.shape[1] + imgs[i].shape[1]
    height = image.shape[0] + imgs[i].shape[0]

    result = np.zeros([height,width,3], dtype=np.uint8)
    result[0:image.shape[0], 0:image.shape[1]] = image
    warp = cv2.warpPerspective(imgs[i], homography[0], (width, height))
    mask = warp != 0
    result[mask] = 0
    result += warp

    inter = Image.fromarray(result)
    inter.save(str(i) + "_inter.jpg")
    image = crop(result)
    
    output = Image.fromarray(image)
    output.save(str(i) + "_output.jpg")