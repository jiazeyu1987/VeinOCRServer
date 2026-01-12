# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
        name='tools',
        ext_modules=cythonize(
        module_list=["treat_compare_img.py", "server.py", "ocr_detect.py", 'pynvml.py', 'image_difference.py'],  # 要编译的Python文件
        language_level=3,  # 使用Python 3语法
    ),
)