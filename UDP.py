import socket
import time


def udp_broadcast(message, port=12345, interval=1):
    """
    通过UDP协议在局域网内广播消息

    参数:
        message (str): 要广播的消息内容
        port (int): 广播使用的端口号，默认为12345
        interval (float): 广播间隔时间(秒)，默认为1秒
    """
    # 创建UDP套接字
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # 设置套接字选项，允许广播
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # 设置超时时间，避免阻塞
        sock.settimeout(0.5)

        # 广播地址
        broadcast_addr = ('<broadcast>', port)

        try:
            print(f"开始在端口 {port} 广播消息: '{message}'")
            while True:
                # 发送广播消息
                sock.sendto(message.encode('utf-8'), broadcast_addr)
                print(f"已广播: '{message}'")
                # 等待指定间隔时间
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n广播已停止")


def udp_receive(port=12345, timeout=10):
    """
    接收局域网内的UDP广播消息

    参数:
        port (int): 监听的端口号，默认为12345
        timeout (int): 超时时间(秒)，默认为10秒，0表示无限等待

    返回:
        str: 接收到的消息内容，如果超时则返回None
    """
    # 创建UDP套接字
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # 设置套接字选项，允许重用地址
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定到所有可用地址和指定端口
        sock.bind(('', port))

        # 设置超时时间
        if timeout > 0:
            sock.settimeout(timeout)
            print(f"开始监听端口 {port}，超时时间 {timeout} 秒")
        else:
            print(f"开始监听端口 {port}，无限等待")

        try:
            # 接收数据
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')
            print(f"从 {addr} 接收到消息: '{message}'")
            return message
        except socket.timeout:
            print(f"等待超时，{timeout} 秒内未收到广播消息")
            return None
        except KeyboardInterrupt:
            print("\n接收已停止")
            return None


# 示例使用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python udp_broadcast.py [broadcast|receive] [port]")
        sys.exit(1)

    command = sys.argv[1].lower()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 12345

    if command == "broadcast":
        message = input("请输入要广播的消息: ")
        udp_broadcast(message, port, 1)
    elif command == "receive":
        timeout = int(input("请输入超时时间(秒)，0表示无限等待: ") or "10")
        udp_receive(port, timeout)
    else:
        print("未知命令，可用命令: broadcast, receive")