import cv2
import time

image_paths=['yosemite1.jpg', 'yosemite2.jpg', 'yosemite3.jpg', 'yosemite4.jpg']

imgs = []
for i in range(len(image_paths)):
    imgs.append(cv2.imread(image_paths[i]))

stitcher = cv2.Stitcher.create()

start = time.time()

(res, output) = stitcher.stitch(imgs)

end = time.time()

print("Computational Time: " + str(end - start))

if res == cv2.STITCHER_OK:
    cv2.imwrite("yosemite.jpg", output)