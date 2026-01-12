import logging
from datetime import datetime


class MyClassWithLogging:
    def __init__(self, log_file=None):
        # 如果没有指定日志文件，则使用当前日期作为文件名
        if log_file is None:
            today = datetime.now().strftime("%Y-%m-%d %H")
            log_file = f"{today}.log"

        # 配置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 设置日志级别为DEBUG

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # 创建格式化器并添加到处理器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # 添加处理器到日志器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        self.logger.info("MyClassWithLogging initialized")

    def do_something(self):
        """执行某些操作并记录日志"""
        self.logger.debug("Starting do_something method")

        try:
            # 模拟一些操作
            result = 100 / 20
            self.logger.info(f"Successfully executed do_something: result={result}")
            return result
        except Exception as e:
            self.logger.error(f"Error in do_something: {str(e)}", exc_info=True)
            return None

    def cleanup(self):
        """清理资源并记录日志"""
        self.logger.info("Cleaning up resources")
        # 这里可以添加实际的清理代码


# 使用示例
if __name__ == "__main__":
    obj = MyClassWithLogging()
    obj.do_something()
    obj.cleanup()