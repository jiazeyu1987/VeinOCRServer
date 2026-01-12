# encoding=utf-8
import sqlite3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyautogui
from datetime import datetime
import os

# class ComparePoints:
#     def __init__(self, id, duration, mode):
#         self.duration = duration
#         self.before_screen = None
#         self.after_screen = None
#
#     def compute_grayscale(self):
#         self.before_screen = self.after_screen
#         self.after_screen = screen_img

def compute_grayscale_v1(screen_img): # 传原始的整个屏幕的截图过来，如果指定了某个区域，那么会造成copy的赋值操作
    ultra_col_start = 1269 # 最新版本的超声投屏图像的相对位置
    ultra_col_end = 1920 # 截屏的最大位置
    row_start = 256
    row_end = 808
    mask = np.ones((row_end - row_start, ultra_col_end - ultra_col_start), dtype=np.uint8)
    gray_value = cv2.mean(screen_img[row_start:row_end, ultra_col_start:ultra_col_end], mask)[0]
    return gray_value

def compute_grayscale_v2(ultra_img): # 传原始的整个屏幕的截图过来，如果指定了某个区域，那么会造成copy的赋值操作
    row_start = 256
    row_end = 808

    mask = np.ones((row_end - row_start, ultra_img.shape[1]), dtype=np.uint8)
    gray_value = cv2.mean(ultra_img[row_start:row_end, :], mask)[0]
    return gray_value


def inser_info_database(db_dir, id, before_path, after_path):
    dbpath = os.path.join(db_dir, "ccwssm")
    backup_dbpath = os.path.join(db_dir, "zccwssm")

    db = sqlite3.connect(dbpath)
    db_backup = sqlite3.connect(backup_dbpath)

    sql_sentence = """
        UPDATE SegmentImagesInfo
        SET ImagePath =?
        WHERE ID =? 
        """

    db.cursor().execute(sql_sentence, (before_path+";"+after_path, id))
    db_backup.cursor().execute(sql_sentence, (before_path+";"+after_path, id))


def convert_timestamp2str(timestamp):

    # 格式化为日期字符串（年-月-日 时:分:秒）
    formatted_time = timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

    return formatted_time + '.png'


def detect_compare_points(duration=3, point_id=None):
    # duration 是整个治疗周期；如果是连续治疗，则需要计算最后一次治疗的理论时间点
    # 本来想着有一个进程中一直在截图了，只需要调用这个函数即可，所以此处不用截图；但是发现，里面处理的时间有延迟，所以，不能从那边调用，这里也得截屏

    ultra_col_start = 1269  # 最新版本的超声投屏图像的相对位置
    ultra_col_end = 1920  # 截屏的最大位置
    row_start = 256
    row_end = 808

    compare_before = None
    compare_after = None

    before = None
    after = None

    gray_before = None
    gray_after = None

    frame_counter = 0

    base_gray_value_arr = [] # 记录首次的2s钟，作为base

    after_time = None
    before_time = None

    after_name = ''
    before_name = ''

    time_after = None
    time_start = datetime.now()

    try:
        while True:
            before = after
            before_time = after_time

            gray_before = gray_after

            after_time = datetime.now()
            after = pyautogui.screenshot(allScreens=False, region=(ultra_col_start, 0, ultra_col_end - ultra_col_start, 1080))

            frame_counter += 1

            after = np.array(after)

            # cv2.imshow("f1", after)
            # cv2.waitKey(0)

            gray_after = compute_grayscale_v2(after)

            if gray_before is None:# 刚开始的第一帧
                continue

            difference = gray_after - gray_before

            if difference > 0.4:  # 1或者0.4; 0.3才能全部抓住治疗前，完全没有照射，但是不需要，只要是刚开始照射，还没有到靶点
                # if gray_after - gray_before < -0.4: # 1或者0.4;
                if compare_before is None: # 即首次出现diff>0.4的before
                    compare_before = before
                    before_name = convert_timestamp2str(before_time)

            elif difference <= -0.4:
                if gray_after - np.median(base_gray_value_arr) < 0:  # 避免过时之后，出现了一个弹框，把截图区域盖住，造成灰度下降; 但是如果原本的超声就很黑，盖住之后
                    continue

                compare_after = after # 最后一个间隔很大的地方，为compare_after；因为每次都赋值
                after_name = convert_timestamp2str(after_time)

            time_diff = datetime.now() - time_start

            if time_diff.total_seconds() < 2: #记录刚开始的2s灰度，作为base;2s是根据设定的休息时间进行调整的
                base_gray_value_arr.append(gray_after)

            if time_diff.total_seconds() > duration + 0.5:
                break

    except Exception as e:
        print("in detect_compare_points, some error occurred:\t", e)


    # write img
    img_dir = "D:/software_data/imgs"

    before_path = os.path.join(img_dir, before_name)
    after_path = os.path.join(img_dir, after_name)

    print(img_dir, before_path, after_path)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if compare_before is not None:
        print(before_path)
        compare_before = cv2.cvtColor(compare_before, cv2.COLOR_RGB2BGR)
        cv2.imwrite(before_path, compare_before)
    if compare_after is not None:
        print(after_path)
        compare_after = cv2.cvtColor(compare_after, cv2.COLOR_RGB2BGR)
        cv2.imwrite(after_path, compare_after)

    db_dir = "D:/software_data"
    inser_info_database(db_dir, point_id, before_path, after_path)



if __name__ == '__main__':
    def read_single_png():
        img_path = 'screensshots/new1.bmp'
        img = cv2.imread(img_path)

        print(img.shape)
        cv2.imshow("img1", img)
        cv2.waitKey(0)

        gray_value = compute_grayscale_v1(img)
        print(gray_value)
    def read_video_experiment():
        video_path = 'videos/2025-06-11 10-27-50.mkv' # 连续治疗的视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print('fps: ', fps)
        gray_means = []

        # start_frames = (2* 60 + 4) * 60 + 20# 11日
        # start_frames = (7 * 60 + 31) * 60 + 217 - 5 # 10日 after
        # start_frames = (7 * 60 + 31) * 60 + 147 - 30  # 10日 before
        # end_frames = (2* 60 + 8) * 60 + 50 + 60# 11日
        # end_frames = (7 * 60 + 36) * 60 + 240 # 10日 after
        # end_frames = (7 * 60 + 36) * 60 - 148 # 10日 before

        start_frames = (2 * 60 + 28) * 60 + 20  # 11日 连续治疗
        end_frames = (3 * 60 + 4) * 60 + 20  # 11日 连续治疗
        frame_no = end_frames - start_frames

        frame_counter = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames)


        before = None
        after = None

        gray_before = 0
        gray_after = 0

        index_before = []
        index_after = []

        while cap.isOpened():

            if frame_counter == frame_no:
                break

            ret, frame = cap.read()
            if not ret:
                break


            # print(frame.shape)
            gray_value = compute_grayscale_v1(frame)
            gray_means.append(gray_value)


            before = after
            after = frame
            frame_counter += 1
            if before is None:
                continue

            gray_before = compute_grayscale_v1(before)
            gray_after = compute_grayscale_v1(after)

            if abs(gray_after - gray_before) > 0.4: # 1或者0.4; 0.3才能全部抓住治疗前，完全没有照射，但是不需要，只要是刚开始照射，还没有到靶点
            # if gray_after - gray_before < -0.4: # 1或者0.4;
                print(gray_after, gray_before)
                cv2.imshow("before", before[:, 1269:])
                cv2.imshow("after", after[:, 1269:])
                # cv2.waitKey(0)

                index_before.append(frame_counter-2)
                index_after.append(frame_counter-1)

            # if frame_counter-2 == 195:
            #     cv2.imshow("frame", before)
            #     cv2.waitKey(0)
            #     index_before.append(195)
            #     print("195:", gray_after, gray_before)



        index_after = np.array(index_after)
        index_before = np.array(index_before)
        gray_means = np.array(gray_means)

        print(gray_means, np.std(gray_means))

        plt.plot(gray_means)
        if len(index_after) != 0:
            plt.plot(index_after, gray_means[index_after], '*')
            plt.plot(index_before, gray_means[index_before], '+')
        plt.show()


    read_video_experiment()

    # detect_compare_points(6, point_id=123) # 冷却时间1s + 延迟时间2s + 治疗时间1s