import cv2
import numpy as np

def showimg(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
# encoding=utf-8

def detect_scaler_from_image(img):

    return 0.0



if __name__ == '__main__':

    file_name = 'screensshots/1.bmp'

    img = cv2.imread(file_name)

    kedu_start_col = 1270
    kedu_len = 6
    kedu_end = kedu_start_col + kedu_len

    img_ultrasound = img[80:-20, kedu_start_col:kedu_end]

    # showimg(img_ultrasound)

    kedu_line = img_ultrasound[:, kedu_len//2, 1]

    kedu_line = np.convolve(kedu_line, np.array([1, 10, 1]), mode='valid')

    kedu_line = np.diff(kedu_line)
    #
    kedu_line = np.convolve(np.abs(kedu_line), np.ones(3), mode='same')
    # kedu_line = np.convolve(kedu_line, np.ones(3), mode='same')
    # diff_kedu = np.convolve(diff_kedu, np.ones(3), mode='same')
    # diff_kedu = np.convolve(diff_kedu, np.ones(3), mode='same')
    # diff_kedu = np.convolve(diff_kedu, np.ones(3), mode='same')
    # diff_kedu = np.convolve(diff_kedu, np.ones(3), mode='same')
    # diff_kedu = np.convolve(diff_kedu, np.ones(3), mode='same')


    from matplotlib import pyplot as plt
    len = 23 * 4
    plt.plot(kedu_line[431+len-1:455+len+1])
    plt.title(file_name)
    plt.show()

    print(kedu_line.shape)
    # showimg(img_ultrasound)
