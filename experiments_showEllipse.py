
#encoding=utf-8

import cv2
import numpy as np

def bgr2cmyk(cv2_bgr_image):
    bgrdash = cv2_bgr_image.astype(float) / 255.0

    # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    K = 1 - np.max(bgrdash, axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate C
        C = (1 - bgrdash[..., 2] - K) / (1 - K)
        C = 255 * C
        C = C.astype(np.uint8)

        # Calculate M
        M = (1 - bgrdash[..., 1] - K) / (1 - K)
        M = 255 * M
        M = M.astype(np.uint8)

        # Calculate Y
        Y = (1 - bgrdash[..., 0] - K) / (1 - K)
        Y = 255 * Y
        Y = Y.astype(np.uint8)

    return np.dstack((C, M, Y, K))


img1 = np.zeros((512,512,3), np.uint8) + np.random.randint(0,255,(512,512,3), dtype=np.uint8)
img2 = np.zeros((512,512,3), np.uint8) + np.random.randint(0,255,(512,512,3), dtype=np.uint8)

img = cv2.imread("screensshots/new1.bmp")

# img = bgr2cmyk(img)

img1 = cv2.absdiff(img[:,:, 0], img[:,:, 1])
img2 = cv2.absdiff(img[:,:, 0], img[:,:, 2])
img3 = cv2.absdiff(img[:,:, 1], img[:,:, 2])



cv2.imshow("0", img)
cv2.imshow("1", img1)
cv2.imshow("2", img2)
cv2.imshow("3", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()