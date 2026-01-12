
from paddleocr import PaddleOCR
import cv2


def ocr_deepth(img_path):

    # en ch
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, rec_image_shape='3, 24, 160', rec_batch_num=8, precision='fp32', show_log=False)  # need to run only once to download and load model into memory

    img = cv2.imread(img_path)

    img = img[170:260, 860:] # 這個是右上側框的截圖

    result = ocr.ocr(img, cls=True)

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = line[1][0]
            print(text)
            if 'mm' in text:
                startid = text.index('：') + 1
                endid = text.index('mm')
                return float(text[startid:endid])

    return 0.0


if __name__ == '__main__':
    deepth = ocr_deepth(img_path = '46b6c66d1b6a0616132677855da0af5.jpg')
    print(deepth)