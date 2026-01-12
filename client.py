import socket
import json
import time

# 定义支持的请求类型
REQUEST_TYPES = {

    'ONLINE': '实时处理的结果',
    'OFFLINE': '非实时的结果',
    'CLOSEOCR': "關掉OCR的一直識別",
    'OPENOCR': "打開OCR，一直識別",
}

# 创建TCP套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.4.107', 30415))


def send_request(host='localhost', port=12345, req_type='RANDOM', param=None, arg=None):
    """向服务器发送请求并接收响应"""

    try:
        # 连接服务器


        # 构建请求
        request = f"{req_type}"
        if param is not None:
            request += f";{param}"
        if arg is not None:
            request = request + ";" + json.dumps(arg)
        # 发送请求
        client_socket.send(request.encode('utf-8'))

        # 接收响应
        # response = client_socket.recv(1024).decode('utf-8')
        # print(response)

        return None

    except ConnectionRefusedError:
        print("错误: 无法连接到服务器，请确保服务器正在运行")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def start_client(host='localhost', port=12345):
    """客户端主函数"""
    print("数值请求客户端")
    print(f"支持的请求类型: {', '.join([f'{k}({v})' for k, v in REQUEST_TYPES.items()])}")

    
    while True:
        print("\n请选择请求类型:")
        for i, (code, desc) in enumerate(REQUEST_TYPES.items(), 1):
            print(f"{i}. {code} - {desc}")
        
        choice = input("请输入选项编号 (q退出): ")
        
        if choice.lower() == 'q':
            break
        
        try:
            choice = int(choice)
            if 1 <= choice <= len(REQUEST_TYPES):
                req_type = list(REQUEST_TYPES.keys())[choice - 1]
                
                # 根据请求类型获取参数

                response = send_request(host, port, req_type,)

                if response:
                    print(f"服务器响应: {response}")
                    print(type(json.loads(response)))
            else:
                print("错误: 无效的选项编号")
        except ValueError:
            print("错误: 请输入有效的数字")

if __name__ == "__main__":

    # start_client(port=30145)

    #response = send_request('localhost', 30415, 'ONLINE', '31415')  # 'CLOSE', 'OPEN'
    point_id = 123

    counter = 1
    while True:
        counter += 1
        response = send_request('localhost', 30415, 'OFFLINE', '31415', {'point_id': point_id, 'time_out': 100, 'is_save': True})  # 'CLOSE', 'OPEN'
        time.sleep(2)
        response = send_request('localhost', 30415, 'ONLINE', '31415', {'point_id': point_id, 'time_out': 100, 'is_save': True})  # 'CLOSE', 'OPEN'

        print(response)
        response = send_request('localhost', 30415, 'ONLINE', '31415',
                                {'point_id': point_id, 'time_out': 100, 'is_save': True})  # 'CLOSE', 'OPEN'
        print(response)
        response = send_request('localhost', 30415, 'OFFLINE', '31415',
                                {'point_id': point_id, 'time_out': 100, 'is_save': True})  # 'CLOSE', 'OPEN'
        print(response)
        time.sleep(1)
        point_id += 1

        print(counter)


    # response = send_request('localhost', 30415, 'CLOSE', '31415')  # 'CLOSE', 'OPEN'



