
import cv2
import numpy as np
import os
import pywt


def wavelet_denoise(image, wavelet='db4', level=1, threshold_type='soft', threshold_value=None):
    """
    使用小波变换对图像进行去噪滤波（修复细节系数结构错误）

    参数:
        image: 输入图像(灰度图)
        wavelet: 小波类型
        level: 分解层数
        threshold_type: 阈值类型 ('soft' 或 'hard')
        threshold_value: 阈值，若为None则自动计算
    """
    # 小波分解 - 返回结构: [cA, (cH1, cV1, cD1), (cH2, cV2, cD2), ...]
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # 提取系数
    cA = coeffs[0]  # 近似系数
    cD_levels = coeffs[1:]  # 各层的细节系数（每层都是3元组）

    # 如果未指定阈值，则使用通用阈值
    if threshold_value is None:
        # 从所有细节系数中计算阈值（展平为一维数组）
        all_details = np.concatenate([np.ravel(detail) for level_details in cD_levels for detail in level_details])
        sigma = np.median(np.abs(all_details)) / 0.6745
        threshold_value = sigma * np.sqrt(2 * np.log(image.size))

    # 对每一层的细节系数应用阈值（保持3元组结构）
    denoised_coeffs = [cA]  # 保留近似系数
    for level_details in cD_levels:
        denoised_level = []
        # 对当前层的3个细节分量分别应用阈值
        for detail in level_details:
            if threshold_type == 'soft':
                denoised_detail = pywt.threshold(detail, threshold_value, mode='soft')
            else:
                denoised_detail = pywt.threshold(detail, threshold_value, mode='hard')
            denoised_level.append(denoised_detail)
        # 将处理后的3个分量重新组合为元组，保持结构
        denoised_coeffs.append(tuple(denoised_level))

    # 小波重构（要求细节系数保持3元组结构）
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)

    # 确保像素值在有效范围内
    denoised_image = np.clip(denoised_image, 0, 255)
    return denoised_image.astype(np.uint8)
def compare_images(image1, image2=None, filter_type='median'):

    image1 = np.array(cv2.imread(image1))
    image2 = np.array(cv2.imread(image2))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if filter_type == 'origin':
        pass

    # median
    # 1. 中值滤波（适合椒盐噪声）
    elif filter_type == 'median':
        gray_image1 = cv2.medianBlur(gray_image1, 5)
        gray_image2 = cv2.medianBlur(gray_image2, 5)

    # gaussian
    # 2. 高斯滤波（适合高斯噪声）
    elif filter_type == 'gaussian':
        gray_image1 = cv2.GaussianBlur(gray_image1, (7, 7), 1)
        gray_image2 = cv2.GaussianBlur(gray_image2, (7, 7), 1)
        # gray_image1 = cv2.GaussianBlur(gray_image1, (5, 5), 0)
        # gray_image2 = cv2.GaussianBlur(gray_image2, (5, 5), 0)

    # bilateral
    # 3. 双边滤波（保持边缘的同时降噪）
    elif filter_type == 'bilateral':
        gray_image1 = cv2.bilateralFilter(gray_image1, 9, 75, 75)
        gray_image2 = cv2.bilateralFilter(gray_image2, 9, 75, 75)
    # box
    elif filter_type == 'box':
        gray_image1 = cv2.boxFilter(gray_image1, -1, (5, 5), normalize=True)
        gray_image2 = cv2.boxFilter(gray_image2, -1, (5, 5), normalize=True)

    # 5. 非局部均值去噪（效果好但计算量大）
    # 对于灰度图，h值控制去噪强度（值越大去噪越强但可能模糊细节）
    elif filter_type == 'fastNlMeansDenoising':
        gray_image1 = cv2.fastNlMeansDenoising(gray_image1, h=10, templateWindowSize=7, searchWindowSize=21)
        gray_image2 = cv2.fastNlMeansDenoising(gray_image2, h=10, templateWindowSize=7, searchWindowSize=21)

    else:
        gray_image1 = wavelet_denoise(gray_image1, wavelet=filter_type, threshold_type='hard')
        gray_image2 = wavelet_denoise(gray_image2, wavelet=filter_type, threshold_type='hard')


    direct_diff = gray_image2.astype(np.float32) - gray_image1.astype(np.float32)
    # cv2.imshow('image1', direct_diff.astype(np.uint8))
    # cv2.waitKey(0)
    # direct_diff = (direct_diff - np.min(direct_diff)) / (np.max(direct_diff) - np.min(direct_diff)) * 512
    direct_diff_10 = direct_diff.copy()
    direct_diff_10[np.where(direct_diff < 10)] = 0


    direct_diff_15 = direct_diff.copy()
    direct_diff_15[np.where(direct_diff < 15)] = 0

    direct_diff_20 = direct_diff.copy()
    direct_diff_20[np.where(direct_diff < 20)] = 0

    auto_diff = direct_diff.copy()
    auto_diff[auto_diff<0] = 0
    auto_mean = np.mean(auto_diff[280:700, 20:-20])
    auto_std = np.std(auto_diff[280:700, 20:-20])
    print(auto_std, auto_mean)
    cutoff = np.mean(auto_diff[auto_diff > 0])
    print(cutoff, np.mean(auto_diff[auto_diff > auto_mean]))
    auto_diff[auto_diff<cutoff] = 0



    #
    # direct_diff[np.where(direct_diff > 255)] = 255
    #
    # cv2.imshow('image1', direct_diff.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.imshow('direct min', direct_diff.astype(np.uint8))

    # diff = cv2.absdiff(image1, image2).astype(np.uint8)
    # cv2.imshow('abs min', )
    img = np.hstack([gray_image1.astype(np.uint8), gray_image2.astype(np.uint8), direct_diff_10.astype(np.uint8), direct_diff_15.astype(np.uint8), direct_diff_20.astype(np.uint8), auto_diff.astype(np.uint8)])

    return img

    # cv2.waitKey(0)

if __name__ == '__main__':
    dirname = 'research_imgs'
    # dirname = '0729'

    save_dir = os.path.join(dirname, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = os.listdir(dirname)
    # files = os.listdir('compareImages/0729')

    files = [file for file in files if file.endswith('.png')]

    filter_types = ['origin', 'median', 'gaussian', 'bilateral', 'box',
                   'db4', 'haar', 'sym5',
                   'fastNlMeansDenoising'];

    # filter_types = ['gaussian']

    for filter in filter_types:
        for index in range(0, len(files) - 1, 2):
            print(files[index])
            img = compare_images(os.path.join(dirname, files[index]), os.path.join(dirname, files[index+1]), filter_type=filter)

            cv2.imwrite(os.path.join(save_dir, files[index].replace('.png', f'_{filter}.jpg')), img)