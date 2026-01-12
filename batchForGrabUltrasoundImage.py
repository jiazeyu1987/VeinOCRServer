#encoding = utf-8
import os
import cv2
import argparse

def readPngs(path=None, out_dir=None):
    png_list = os.listdir(path)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for png in png_list:
        print(png)
        if png[-3:] != 'png' and png[-3:] != 'jpg' and png[-3:] != 'bmp':
            continue

        png_image = cv2.imread(os.path.join(path, png), cv2.IMREAD_COLOR)
        png_image = png_image[:,1269:]

        out_path = os.path.join(out_dir, png)
        cv2.imwrite(out_path, png_image)


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='png截取工具')
    parser.add_argument('--dir', type=str, default='./', help='输入文件夹 (默认: ./RecordsDir)')
    parser.add_argument('--outdir', type=str, default="outs", help='结果保存地址')

    # 解析命令行参数
    args = parser.parse_args()

    readPngs(path=args.dir, out_dir=args.outdir)
