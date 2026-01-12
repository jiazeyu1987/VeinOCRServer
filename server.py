import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import socket
import threading
# import multiprocessing
import json
import time

from ocr_detect import OCRDetect
import logging
from datetime import datetime
from treat_compare_img import ComparePoints

class ImageProcessServer:
    def __init__(self):

        # logging.basicConfig(
        #     level=logging.INFO,  # 设置日志级别为 INFO，这意味着 INFO 及以上级别的日志会被记录
        #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        #     filename='ocrserver.log',  # 指定日志输出文件，如果不指定则默认输出到控制台
        #     filemode="a"
        # )


        self.init_logger()


        # 定义支持的请求类型
        self.REQUEST_TYPES = {

            'Offline': '不需要实时获取的，界面显示的内容; 以json格式返回',
            # {'SCALER':'进行超声可以放大和缩小，每个小刻度代表1mm，此时对应的图片像素点的个数'},


            'Online' : '需要实时获取的,界面显示的任何东西; 以json的字符串返回',
            # {'SKINDEPTH': '思多科测量显示的皮肤到焦点的距离',
            # 'DEPTH': '思多科超声显示的深度',
            # 'A': '思多科显示的长度A',
            # 'B': '思多科显示的长度B',
            # 'ANGLE': '思多科显示的A与B的角度',}
            'OCR': "獲取OCR的狀態",
            'CLOSEOCR': "關掉OCR的一直識別",
            'OPENOCR': "打開OCR，一直識別",
        }

        self.setting = self.load_setting()

        self.ocrserver = OCRDetect(self.setting, self.logger)

        self.start_ocr_server()



        # for 治疗前后对比截图
        self.logger.info("成功导入对比截图工具")

        self.point_id = None
        self.client_thread = None
        self.compare_client = None
        self.stop_event = threading.Event()

        self.compareTool = ComparePoints(self.setting, self.logger)
        # 为了首次调用服务器时不延迟，此处默认调用一次
        # default_offline = {"point_id": 3141592653, "is_save": False, "time_out": 100}
        # self.get_offline(json.dumps(default_offline))
        # time.sleep(1)
        # self.get_offline(json.dumps(default_offline))


    def init_logger(self, dst='ocrlog'):
        # 如果没有指定日志文件，则使用当前日期作为文件名
        if not os.path.exists(dst):
            os.makedirs(dst)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(dst, f"ocrapp_{today}.log")

        # 配置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 设置日志级别为DEBUG

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 创建格式化器并添加到处理器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # 添加处理器到日志器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)


    def load_setting(self):
        cur_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        setting_path = os.path.join(cur_dir, 'settings')

        if os.path.exists(setting_path):
            with open(setting_path, 'r') as f:
                try:
                    return json.load(f)
                except:
                    return None
        else:
            return None

    def start_ocr_server(self):
        # 啓動實時識別
        # self.ocrthread = multiprocessing.Process(
        self.ocrthread = threading.Thread(
            target=self.ocrserver.start_ocr_server,
            args=(),
            daemon=True
        )

        self.ocrthread.start()


    def close_ocr_server(self):
        pass


    def get_online(self):
        """截图识别"""

        results = {
            'SkinDepth': self.ocrserver.MEASSURE['skin_distance'],
            'A': self.ocrserver.MEASSURE['A'],
            'B': self.ocrserver.MEASSURE['B'],
            'Alpha': self.ocrserver.MEASSURE['Alpha'],

            'Depth': self.ocrserver.MEASSURE['深度'],
            'IsFreeze': self.ocrserver.MEASSURE['Is_Freeze'],
            'Points_Per_MM': self.ocrserver.MEASSURE['Points_Per_MM'],
        }

        # # 用于不截图的测试
        # results = {
        #     'SkinDepth': 5,
        #     'A': 4,
        #     'B': 3,
        #     'Alpha': 0,
        #
        #     'Depth': 6,
        #     'IsFreeze': False,
        #     'Points_Per_MM': 13,
        # }

        return results

    def get_offline(self, arg):

        results = ""

        if arg is None:
            results = {'success':False, 'info': "输入参数有误！"}
            return results

        arg = json.loads(arg)

        point_id = arg['point_id']
        time_out = arg['time_out']
        is_save = arg['is_save']

        """计算非实时的结果"""
        # if self.compareTool is None:
        #     from treat_compare_img import ComparePoints
        #     self.compareTool = ComparePoints(self.logger)
        #     self.logger.info("成功导入对比截图工具")

        if self.point_id != point_id:

            self.point_id = point_id

            # 1检查线程状态，确保上次运行已停止
            if self.compare_client is not None and self.compare_client.is_alive():
                self.stop_event.set()
                self.compare_client.join(timeout=2)
                del self.compare_client

            # self.stop_event = multiprocessing.Event()

            # 2重置停止信号（关键：必须清除之前的set状态）
            self.stop_event.clear()

            # 3创建并启动新线程
            # self.compare_client = multiprocessing.Process(
            self.compare_client = threading.Thread(
                target=self.compareTool.detect,
                args=(point_id, float(time_out), is_save, self.stop_event),
            )

            self.compare_client.start()

        else:
            self.point_id = None
            self.stop_event.set()
            self.logger.info("stop set成功。")
            self.compare_client.join(timeout=3)



        # self.compareTool.detect_compare_points(point_id, time_out, is_save)

        results = self.compareTool.response

        return results

    def handle_client(self, client_socket, client_address):
        """处理客户端请求的函数"""

        self.logger.info(f"接收到来自 {client_address} 的连接")

        try:
            while True:
                # 接收客户端请求
                request = client_socket.recv(1024).decode('utf-8').strip()

                # 如果客户端关闭连接，请求将为空
                if not request:
                    break

                # 解析请求格式: TYPE[:PARAM]
                parts = request.split(';')

                req_type = parts[0].upper()
                param = parts[1] if len(parts) > 1 else None # 密码
                arg = parts[2] if len(parts) > 2 else None

                code = '31415'
                # print(param)
                # 根据请求类型处理
                if param != code:
                    self.logger.info("密码错误")
                    response = '密码错误'
                else:
                    # print("request :", req_type)
                    if req_type == 'OFFLINE':
                        try:
                            self.logger.info(req_type + arg)
                            response = self.get_offline(arg)
                            response = None # 如果点击太快，造成online和offline同时返回，会造成包的组合发送，导致解析端出错，所以把offline的返回置空
                        except Exception as e:
                            self.logger.error("offline返回错误:如下")
                            self.logger.error(e)
                    elif req_type == 'ONLINE':
                        try:
                            self.logger.info(req_type)
                            response = self.get_online()
                        except Exception as e:
                            self.logger.error("online返回错误:如下")
                            self.logger.error(e)

                    elif req_type == 'CLOSE':
                        self.close_ocr_server()
                        response = {'success':True,  'info': "close successfully"}
                    else:
                        # 未知请求类型
                        response = {'success':False, 'info': f"错误: 未知请求类型 '{req_type}'。支持的类型: {', '.join(self.REQUEST_TYPES.keys())}"}

                if response:
                    response = json.dumps(response)
                    # 发送响应给客户端
                    client_socket.send(response.encode('utf-8'))


        except Exception as e:
            self.logger.error(f"处理客户端 {client_address} 时发生错误: {e}")
        finally:
            # 关闭客户端连接
            client_socket.close()
            print(f"与 {client_address} 的连接已关闭")

    def start_server(self, host='localhost', port=12345):
        """启动TCP服务器"""
        # 创建TCP套接字

        # self.logger.info(f"start server: host={host}, port={port}")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # 设置套接字选项，允许地址重用
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 绑定地址和端口
            server_socket.bind((host, port))

            # 开始监听，最大连接数为5
            server_socket.listen(5)
            print(f"服务器已启动，监听地址: {host}:{port}")
            print(f"支持的请求类型: {', '.join([f'{k}({v})' for k, v in self.REQUEST_TYPES.items()])}")
            self.logger.info(f"服务器已启动，监听地址: {host}:{port}")

            try:
                while True:
                    # 接受客户端连接
                    client_socket, client_address = server_socket.accept()

                    if self.client_thread is not None:
                        del self.client_thread

                    # 为每个客户端创建一个新线程
                    # client_thread = multiprocessing.Process(
                    self.client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    self.client_thread.start()
                    # self.client_thread.join()

            except KeyboardInterrupt:
                print("\n服务器已停止")
                self.logger.info("\n服务器已停止")


def run(host="localhost", port=30415):
    imgProcess = ImageProcessServer()
    imgProcess.start_server(host=host, port=port)

if __name__ == "__main__":
    run("127.0.0.1")

