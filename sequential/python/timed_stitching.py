import cv2
import time
import numpy as np
#import matplotlib.pyplot as plt

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

def getHomography(kp1, kp2, img1, img2, matches):
    if len(matches) <= 4:
        return None

    kpA = np.float32([kp.pt for kp in kp1])
    kpB = np.float32([kp.pt for kp in kp2])

    ptsA = np.float32([kpA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpB[m.trainIdx] for m in matches])
    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4.0)

    return (img1, img2, H, status)


#image_paths=['yosemite1.jpg', 'yosemite2.jpg', 'yosemite3.jpg', 'yosemite4.jpg']
image_paths=['1.jpg', '2.jpg']

imgs = []

for i in range(len(image_paths)):
    img = cv2.imread(image_paths[i])
    imgs.append(img)

start = time.time()

kps = []
features = []
for i in range(len(imgs)):
    (kp, feature) = findKeyPoints(imgs[i])
    kps.append(kp)
    features.append(feature)
    cv2.imwrite('keypoints' + str(i) + ".jpg",cv2.drawKeypoints(imgs[i], kp, None))

end = time.time()

print("Find Keypoints Time: " + str(end - start))
    
draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = 2)

start = time.time()

matches = []
for i in range(len(imgs)):
    kp1 = kps[i]
    feature1 = features[i]
    for j in range(i+1, len(imgs)):
        kp2 = kps[j]
        feature2 = features[j]
        matches.append(matchKeyPoints(feature1, feature2))
        res = cv2.drawMatches(imgs[i], kp1, imgs[j], kp2, matchKeyPoints(feature1, feature2), None, **draw_params)
        cv2.imwrite("match" + str(i) + str(j) + ".jpg", res)

end = time.time()

print("Match Keypoints Time: " + str(end - start))

start = time.time()

homographies = []
counter = 0
for i in range(len(imgs)):
    kp1 = kps[i]
    feature1 = features[i]
    for j in range(i+1, len(imgs)):
        kp2 = kps[j]
        feature2 = features[j]
        match = matches[counter]

        homography = getHomography(kp1, kp2, imgs[i], imgs[j], match)
        if homography != None:
            homographies.append(homography)
        counter += 1

end = time.time()

print("Homography Time: " + str(end - start))

result = None

for i in range(len(homographies)):
    width = homographies[i][0].shape[1] + homographies[i][1].shape[1]
    height = homographies[i][0].shape[0] + homographies[i][1].shape[0]

    result = cv2.warpPerspective(homographies[i][1], homographies[i][2], (width, height))
    result[0:homographies[i][0].shape[0], 0:homographies[i][0].shape[1]] = homographies[i][0]

#plt.figure(figsize=(20,10))
#plt.imshow(result)
#plt.savefig('output.jpg')
