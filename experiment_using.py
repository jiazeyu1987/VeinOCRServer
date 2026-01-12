import numpy as np
import cv2
import pyautogui
from datetime import datetime
import os

def screen_shot():
    img = pyautogui.screenshot(allScreens=False, region=(0, 0, 600, 1080))
    img = np.array(img)

    compare_before = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    timestamp = datetime.now()
    formatted_time = timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

    img_dir = "D:/software_data/imgs"

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    before_path = os.path.join(img_dir, formatted_time + "_before.png")

    print(img_dir, before_path)
    cv2.imwrite(before_path, compare_before)

# cv2.imshow("before_path", compare_before)
# cv2.waitKey(0)

if __name__ == '__main__':
    while 1:
        screen_shot()