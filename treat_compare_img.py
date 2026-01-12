# encoding=utf-8
import sqlite3
import time
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyautogui
from datetime import datetime
import os
import image_difference
import threading

class ComparePoints:
    def __init__(self, setting={}, logger=None):

        if setting is None:
            setting = {}

        self.logger = logger
        self.setting = {"width_x": setting['width_x'] if "width_x" in setting else 2,
                        "height_y": setting['height_y'] if "height_y" in setting else 4,
                        "binary_threshold": setting['binary_threshold'] if "binary_threshold" in setting else 10,
                        "drawcontour": setting['drawcontour'] if "drawcontour" in setting else False,
                        "if_align": setting['if_align'] if "if_align" in setting else False}

        self.point_id = None
        self.save_point_id = None

        self.is_save = True # 0 不存到数据库；1存入数据库
        self.db_dir = "D:/software_data"

        self.response = {'success': True}


        self.active_task =  None


    def compute_grayscale_v1(self, screen_img): # 传原始的整个屏幕的截图过来，如果指定了某个区域，那么会造成copy的赋值操作
        ultra_col_start = 1269 # 最新版本的超声投屏图像的相对位置
        ultra_col_end = 1920 # 截屏的最大位置
        row_start = 256
        row_end = 808
        mask = np.ones((row_end - row_start, ultra_col_end - ultra_col_start), dtype=np.uint8)
        gray_value = cv2.mean(screen_img[row_start:row_end, ultra_col_start:ultra_col_end], mask)[0]
        return gray_value

    def compute_grayscale_v2(self, ultra_img): # 传原始的整个屏幕的截图过来，如果指定了某个区域，那么会造成copy的赋值操作
        row_start = 256
        row_end = 808

        mask = np.ones((row_end - row_start, ultra_img.shape[1]), dtype=np.uint8)
        gray_value = cv2.mean(ultra_img[row_start:row_end, :], mask)[0]
        return gray_value


    def  inser_info_database(self, db_dir, id, before_path, after_path):
        # dbpath = os.path.join(db_dir, "ccwssm")
        # backup_dbpath = os.path.join(db_dir, "zccwssm")

        dbpath = db_dir + "/ccwssm"
        backup_dbpath = db_dir + "/zccwssm"


        db = sqlite3.connect(dbpath, check_same_thread=False, timeout=30)
        db_backup = sqlite3.connect(backup_dbpath, check_same_thread=False, timeout=30)

        modifytime = datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')[:-3]

        sql_sentence = '''
            UPDATE SegmentImagesInfo
            SET ImagePath = ?, ModifyTime = ?
            WHERE ID = ? 
            '''

        self.logger.info(f"{before_path};{after_path};{modifytime};{id}")

        image_path = before_path + ";" + after_path+";" + after_path.replace('_after', '_diff')

        db.cursor().execute(sql_sentence, (image_path, modifytime, id))

        db_backup.cursor().execute(sql_sentence, (image_path, modifytime, id))

        db.commit()
        db_backup.commit()

        db.cursor().close()
        db_backup.cursor().close()


    def convert_timestamp2str(self, timestamp):

        # 格式化为日期字符串（年-月-日 时:分:秒）
        formatted_time = timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

        return formatted_time

    def write_img(self):
        # write img
        self.logger.info("write_img...")
        img_dir = "D:/software_data/imgs"

        before_path = f"{img_dir}/{self.before_name}_before.png"
        after_path = f"{img_dir}/{self.after_name}_after.png"

        if not self.is_save: # 能量预测的时候，图片不存数据库，需要覆盖，否则越来越多
            before_path = img_dir + "/energy_before.png"
            after_path = img_dir + "/energy_after.png"

            if os.path.exists(before_path):
                os.remove(before_path)
            if os.path.exists(after_path):
                os.remove(after_path)

        # self.logger.info(f'{img_dir}; {before_path}; {after_path}' )

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if self.compare_before is not None:
            # print(before_path)
            compare_before = cv2.cvtColor(self.compare_before, cv2.COLOR_RGB2BGR)
            cv2.imwrite(before_path, compare_before)
        else:
            self.logger.info("no compare before")

        if self.compare_after is not None:
            try:
                compare_after = image_difference.process_two_images(cv2.cvtColor(self.compare_before, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.compare_after,cv2.COLOR_RGB2BGR), if_align=self.setting["if_align"], binary_threshold=self.setting["binary_threshold"], width_x=self.setting["width_x"], height_y=self.setting["height_y"], drawcontour=self.setting["drawcontour"])
                # compare_after = image_difference.process_two_images(self.compare_before, self.compare_after, if_align=self.setting["if_align"], binary_threshold=self.setting["binary_threshold"], width_x=self.setting["width_x"], height_y=self.setting["height_y"], drawcontour=self.setting["drawcontour"])
                # cv2.imshow("test show diff image", compare_after)
                # cv2.waitKey(0)
            except Exception as e:
                self.logger.error(f"after的处理出错：{e}， 使用不画contour的after")
                compare_after = None

            if compare_after is None: # process_two_images说明返回的，或者出错了，或者没有识别到center等
                compare_after = cv2.cvtColor(self.compare_after, cv2.COLOR_RGB2BGR)

            cv2.imwrite(after_path, compare_after)

        else:
            self.logger.info("no compare after")

        # 存储diff的image
        if self.compare_before is not None and self.compare_after is not None:
            direct_diff = np.array(self.compare_after).astype(np.float32) - np.array(self.compare_before).astype(np.float32)
            direct_diff[np.where(direct_diff < 0)] = 0
            direct_diff = direct_diff.astype(np.uint8)

            cv2.imwrite(after_path.replace('_after', '_diff'), direct_diff)
        else:
            self.logger.info("no compare diff")

        if self.is_save:
            try:
                self.logger.info(f"插入数据库---")
                self.inser_info_database(self.db_dir, self.save_point_id, before_path, after_path)
                self.logger.info(f"插入数据库---成功。")

            except OSError as e:
                # print("插入数据库： ", e)
                self.logger.error(f"路径错误：\t{e}, {self.db_dir}, {self.save_point_id}, {before_path}, {after_path}")
            except sqlite3.OperationalError as e:
                self.logger.error(f"数据库操作错误：\t{e}")
            except Exception as e:
                self.logger.error(f"其他错误：\t{e}")


    # def stop_detect(self):
    #     self._stop_event.set()

    # def detect_compare_points(self, point_id=None, duration=3, is_save=True):
    #
    #     # with self.task_lock:
    #
    #         # print(self.point_id)
    #
    #     img_dir = "D:/software_data/imgs"
    #     before_path = os.path.join(img_dir, "energy_before.png")
    #     after_path = os.path.join(img_dir, "energy_after.png")
    #     try:
    #         if os.path.exists(before_path):
    #             os.remove(before_path)
    #         if os.path.exists(after_path):
    #             os.remove(after_path)
    #     except Exception as e:
    #         self.logger.error(f"删除能量预测文件:\t{e}")
    #
    #     self.is_save = is_save
    #
    #     if point_id == self.point_id:
    #
    #         self.point_id = None
    #         # 为了回复消息
    #         img_dir = "D:/software_data/imgs"
    #         if not self.is_save:  # 能量预测的时候，图片不存数据库，需要覆盖，否则越来越多
    #             before_path = os.path.join(img_dir, "energy_before.png")
    #             after_path = os.path.join(img_dir, "energy_after.png")
    #             self.response = before_path + ";" + after_path
    #         else:
    #             self.response = "识别结束..."
    #         self._stop_event.set()
    #
    #     else:
    #         if self.active_task is not None:
    #             del self.active_task
    #
    #         self.point_id = None
    #
    #         # 创建新任务
    #         task = threading.Thread(target=self.detect, args=(point_id, duration, ), daemon=True)
    #         self.active_task = task
    #         self.response = "正在识别..."
    #         task.start()
    def get_screen_shot(self):
        ultra_col_start = 1269  # 最新版本的超声投屏图像的相对位置
        ultra_col_end = 1920  # 截屏的最大位置
        row_start = 256
        row_end = 808
        img_time = datetime.now()
        img = pyautogui.screenshot(allScreens=False, region=(ultra_col_start, 0, ultra_col_end - ultra_col_start, 1080))
        return img, img_time
    def detect(self, point_id=None, duration=3, is_save=None, stop_event=None):
        self.logger.info("detect: " + str(point_id) + str(self.point_id) + " " + str(duration), )

        self.point_id = point_id
        self.save_point_id = point_id
        self.is_save = is_save

        self._stop_event = stop_event

        # mode=0: stop
        # mode=1: start
        # duration 是整个治疗周期；如果是连续治疗，则需要计算最后一次治疗的理论时间点
        # 本来想着有一个进程中一直在截图了，只需要调用这个函数即可，所以此处不用截图；但是发现，里面处理的时间有延迟，所以，不能从那边调用，这里也得截屏


        self.compare_before = None
        self.compare_after = None

        before_default = None  # 如果没有识别到，默认的图片
        after_default = None  # 如果没有识别到，默认的图片
        after_min_gray = 10000
        start_to_find_after = False
        find_after_counter = 0  # 连续5个点在2std以上，开始回落以后，才开始找after


        before_name_default = None
        after_name_default = None


        before = None
        after = None

        gray_before = None
        gray_after = None

        frame_counter_stop_delay = 5
        frame_counter_after = 0
        frame_counter_after_found = 0

        base_gray_value_arr = [] # 记录首次的2s钟，作为base

        after_time = None
        before_time = None

        self.after_name = ''
        self.before_name = ''

        time_after = None
        time_start = datetime.now()

        std_value = 0
        mean_value = 0


        try:
            while not self._stop_event.is_set():
                before = after
                before_time = after_time

                if before_default is None and before is not None:
                    before_default = before
                    before_name_default = before_time

                gray_before = gray_after

                after, after_time = self.get_screen_shot()
                frame_counter_after += 1

                if frame_counter_after % 60 == 0:
                    self.logger.info('fps:' + str(frame_counter_after))

                after = np.array(after)

                # cv2.imshow("f1", after)
                # cv2.waitKey(0)

                gray_after = self.compute_grayscale_v2(after)

                if gray_before is None:# 刚开始的第一帧
                    continue


                time_diff = datetime.now() - time_start


                if time_diff.total_seconds() < 0.2: #记录刚开始的2s灰度，作为base;2s是根据设定的休息时间进行调整的
                    base_gray_value_arr.append(gray_after)
                    # continue # 去掉continue，适应减少治疗前的等待时间的更改2025年9.16

                std_value = np.std(base_gray_value_arr)
                mean_value = np.mean(base_gray_value_arr)


                difference = gray_after - gray_before

                if gray_after > mean_value + 3 * std_value or gray_after > mean_value * 1.2:
                    find_after_counter += 1
                    if find_after_counter > 3: # 至少连续5个超出范围后，才开始找after；但是如果治疗时间点，截图还需要时间，所以，设置的5可以更改小或者大
                        # self.logger.error(f"start to find after img")
                        start_to_find_after = True

                if difference > 0.1:  # 1或者0.4; 0.3才能全部抓住治疗前，完全没有照射，但是不需要，只要是刚开始照射，还没有到靶点
                    # if gray_after - gray_before < -0.4: # 1或者0.4;
                    if self.compare_before is None:  # 即首次出现diff>0.4的before
                        self.compare_before = before
                        self.before_name = self.convert_timestamp2str(before_time)
                        self.logger.info(self.before_name + ", before img founded")

                elif difference <= -0.1:
                    if self.compare_before is None: # 找到before以后，再考虑after
                        continue
                    # if gray_after > mean_value + 2 * std_value: # 虽然满足diff<-std, 但是after太大
                    #     continue
                    # if gray_after > mean_value * 1.2: # 虽然满足diff<-std, 但是after太大
                    #     continue

                    if not start_to_find_after:
                        continue
                    # if gray_after - np.min(base_gray_value_arr) < 0:  # 避免过时之后，出现了一个弹框，把截图区域盖住，造成灰度下降; 但是如果原本的超声就很黑，盖住之后
                    #     continue
                    # if gray_after < after_min_gray:
                    #     after_min_gray = gray_after
                    #     after_default = after
                    #     after_name_default = after_time


                    frame_counter_after_found = frame_counter_after  # 只是记录，当前满足要求的点
                    # frame_counter_stop_delay = 5

                    # after, after_time = self.get_screen_shot() # 最后一个间隔很大的地方，为compare_after；因为每次都赋值; 本来应该直接幅值after，但是1%的概率还是会截图到发射中
                    # self.compare_after = np.array(after)
                    # self.after_name = self.convert_timestamp2str(after_time)
                    # self.logger.info(self.after_name + 'after image acc')


                # if start_to_find_after: # 开始找after之后，记录最小的点，作为没有识别到的默认截图
                #     if gray_after < after_min_gray:
                #         after_min_gray = gray_after
                #         after_default = after
                #         after_name_default = after_time

                # if frame_counter_after < frame_counter_after_found + frame_counter_stop_delay:
                #     if gray_after < after_min_gray:
                #         after_min_gray = gray_after
                #         after_default = after
                #         after_name_default = after_time

                if (frame_counter_after_found + 0) == frame_counter_after:  # 对记录的counter，后面的第3帧进行记录
                    self.compare_after = after
                    self.after_name = self.convert_timestamp2str(after_time)
                    self.logger.info(self.after_name + ', after image found delay 0')

                # if frame_counter_stop_delay > 0: # found 之后，设置为10
                #     frame_counter_stop_delay -= 1
                #     if frame_counter_stop_delay == 0:
                #         break


                if self._stop_event.is_set(): # stop event 触发之后，需要持续0.5s
                    self.logger.info("stop detect")
                    # frame_counter_stop_delay -= 1
                    #
                    # if frame_counter_stop_delay >= 0:
                    #     continue
                    break


        except Exception as e:
            print("in detect_compare_points, some error occurred:\t", e)
            self.logger.error(f"in detect_compare_points, some error occurred:\t{e}")
            self.response = {'success': False, 'info': 'error_in_detect', 'detail': e}

            return


        if self.compare_before is None:
            self.logger.warning("can not find before, use default")
            self.compare_before = before_default
            self.before_name = self.convert_timestamp2str(before_name_default)[:-1]  # 没识别到，名字的毫秒少一位

        if self.compare_after is None:
            # self.compare_after = after_default
            # self.after_name = self.convert_timestamp2str(after_name_default)[:-1]  # 没识别到，名字的毫秒少一位
            self.logger.warning("can not find after, use last time before as default")
            self.compare_after = before
            self.after_name = self.convert_timestamp2str(before_time)[:-1]  # 没识别到，名字的毫秒少一位

        try:
            self.write_img()
        except Exception as e:
            # print("对比图片，写失败： ", e)
            self.logger.error(e)
            self.response = {'success': False, 'info': 'error_in_write img', 'detail': e}
            return



if __name__ == '__main__':

    compare = ComparePoints()

    def show(img):
        cv2.imshow('img', img)
        cv2.waitKey(0)


    def read_single_png():
        img_path = 'screensshots/new1.bmp'
        img = cv2.imread(img_path)

        print(img.shape)
        cv2.imshow("img1", img)
        cv2.waitKey(0)

        gray_value = compare.compute_grayscale_v1(img)
        print(gray_value)
    def mean_k(data, days):
        mean_20_x = []
        mean_20_y = []
        for id in range(days, len(data) - days):
            mean_20_x.append(id)
            mean_20_y.append(np.mean(data[id - days+1: id+1]))
        return mean_20_x, mean_20_y

    def mean_middle(data, days):
        x = []
        y = []
        for id in range(days, len(data) - days):
            x.append(id)
            y.append(np.mean(data[id - days//2: id + days//2 + 1]))
        return x, y

    def mean_cover(data, days):
        x = []
        y = []
        for id in range(days, len(data) - days):
            x.append(id)
            data[id] = (np.mean(data[id - days // 2: id + days // 2 + 1]))
            y.append(data[id])
        return x, y

    def read_video_experiment():
        video_path = 'videos/2025-08-23 11-18-13.mkv' # 连续治疗的视频
        # video_path = 'videos/2025-06-25 00-19-03.mkv' # 连续治疗的视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print('fps: ', fps)
        gray_means = []

        # fps = 60

        # start_frames = (2* 60 + 4) * 60 + 20# 11日
        # start_frames = (7 * 60 + 31) * 60 + 217 - 5 # 10日 after
        # start_frames = (7 * 60 + 31) * 60 + 147 - 30  # 10日 before
        # end_frames = (2* 60 + 8) * 60 + 50 + 60# 11日
        # end_frames = (7 * 60 + 36) * 60 + 240 # 10日 after
        # end_frames = (7 * 60 + 36) * 60 - 148 # 10日 before

        start_frames = (5 * 60 + 57) * fps + 0 +  0   + 0# 8月22日 连续治疗
        end_frames = (6 * 60 + 2) * fps -  200 + 300 -   0# 8月22日 连续治疗

        # start_frames = (1 * 60 + 37) * fps + 0  # 6月25日 连续治疗
        # end_frames = (1 * 60 + 44) * fps    # 6月25日 连续治疗


        frame_no = end_frames - start_frames

        frame_counter = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames)


        before = None
        after = None

        gray_before = 0
        gray_after = 0

        index_before = None
        index_after = None

        gray_base = []

        mean_frame = 0
        ultra_col_start = 1269 # 最新版本的超声投屏图像的相对位置
        ultra_col_end = 1920 # 截屏的最大位置
        row_start = 256
        row_end = 808
        while cap.isOpened():

            if frame_counter == frame_no:
                break

            ret, frame = cap.read()
            if not ret:
                break
            # show(frame)

            # print(frame.shape)
            gray_value = compare.compute_grayscale_v1(frame)
            # if frame_counter < 60:
            #     mean_frame = mean_frame + np.array(frame[row_start:row_end, ultra_col_start:ultra_col_end]).astype(np.float32)
            # if frame_counter == 60:
            #     cv2.imshow("mean frame", (mean_frame / 60).astype(np.uint8))
            #     cv2.waitKey(0)

            gray_means.append(gray_value)


            before = after
            after = frame
            frame_counter += 1
            if before is None:
                continue

            gray_before = compare.compute_grayscale_v1(before)
            gray_after = compare.compute_grayscale_v1(after)

            if gray_after - gray_before >= 0.1: # 1或者0.4; 0.3才能全部抓住治疗前，完全没有照射，但是不需要，只要是刚开始照射，还没有到靶点
            # if gray_after - gray_before < -0.4: # 1或者0.4;

                # cv2.imshow("before", before[:, 1269:])
                # cv2.imshow("after", after[:, 1269:])
                # cv2.waitKey(0)
                if index_before is None:
                    index_before = frame_counter - 2

            elif gray_after - gray_before <= -0.15:

                # index_before.append(frame_counter-2)
                index_after = frame_counter-1

            # if frame_counter-2 == 195:
            #     cv2.imshow("frame", before)
            #     cv2.waitKey(0)
            #     index_before.append(195)
            #     print("195:", gray_after, gray_before)

            if frame_counter < 2 * fps:
                gray_base.append(gray_before)


        gray_means = np.array(gray_means)
        mean_value = np.mean(gray_means[:60])
        std_value = np.std(gray_means[:60])

        print("mean:", mean_value, "std:", std_value)
        # plt.plot([mean_value] * len(gray_means), 'o')
        # plt.plot([mean_value - 2 * std_value] * len(gray_means))
        # plt.plot([mean_value + 2 * std_value] * len(gray_means))
        # plt.plot([mean_value - 3 * std_value] * len(gray_means))
        # plt.plot([mean_value + 3 * std_value] * len(gray_means))

        # plt.plot(gray_means + 2 * std_value, "o")
        # plt.plot(gray_means + 2 * std_value, 'x')

        print(np.argwhere(np.diff(gray_means) < -3 * std_value) )



        # plt.plot(gray_means)

        # kernel = np.ones(3) / 3
        # conv = np.convolve(gray_means, kernel, 'same')
        # plt.plot(conv)
        #
        # kernel = np.ones(5) / 5
        # conv = np.convolve(gray_means, kernel, 'same')

        print(index_before, index_after)

        if index_after != None:
            # plt.plot(index_after, gray_means[index_after], '*')
            # plt.plot(index_before, gray_means[index_before], '+')
            pass
        else:
            index_after = np.argwhere(np.diff(gray_means) < -1 * std_value) + 1
            index_before = np.argwhere(np.diff(gray_means) > 1 * std_value)
            plt.plot(index_after, gray_means[index_after], '*')
            plt.plot(index_before, gray_means[index_before], '+')
        # index_after = np.argwhere(np.diff(gray_means) < -1 * std_value) + 1
        # index_before = np.argwhere(np.diff(gray_means) > 1 * std_value)

        print(index_before, index_after)
        fig1 = plt.figure(1)

        plt.plot(gray_means, 'y')

        plt.plot(index_after, gray_means[index_after], '*')
        plt.plot(index_before, gray_means[index_before], '+')


        mean_20_x, mean_20_y = mean_k(gray_means, 5)
        mean_5_x, mean_5_y = mean_k(gray_means, 3)

        plt.plot(mean_20_x, mean_20_y, 'r')
        # plt.plot(mean_5_x, mean_5_y, 'b')

        x,y = mean_middle(gray_means, 15)
        print(np.argwhere(abs(np.diff(y)) > 2))
        plt.plot(x, y, 'b')
        x,y = mean_cover(gray_means, 3)
        plt.plot(x, y, 'g')

        # plt.ylim([24, 24.4])
        plt.xlim([0, 80])
        plt.ylim([24.2, 24.5])
        plt.xlim([210, 230])

        fig2 = plt.figure(2)
        start = 300
        leng = 150
        plt.plot(gray_means[start:start+leng])
        plt.plot([mean_value]*leng)
        plt.plot([mean_value+3*std_value]*leng)
        plt.plot([mean_value+2*std_value]*leng)
        plt.plot([mean_value+1*std_value]*leng)
        plt.ylim([24, 25])
        # plt.plot(index_after-200, gray_means[index_after-200], '*')
        # plt.plot(index_before-200, gray_means[index_before-200], '+')



        plt.show()


    read_video_experiment()

    # compare.detect_compare_points(point_id=123, duration=100) # 冷却时间1s + 延迟时间2s + 治疗时间1s

