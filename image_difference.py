import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mode
# from PIL import ImageFont, ImageDraw, Image

def extract_points_mm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # show(gray)

    ruler = gray[:, 12]
    # ruler = gray[:, 5] # 左侧没有黑色的区域的竖条

    # 满足局部最大
    local_max_index = []
    winow_radius = 2
    for i in range(winow_radius + 10, ruler.shape[0] - winow_radius - 10):
        local_max_index.append(np.argmax( ruler[i-winow_radius: i+winow_radius +1]))
        # print(ruler[i], np.max(ruler[i-winow_radius: i+winow_radius +1]))
        # if ruler[i] == np.max(ruler[i-winow_radius: i+winow_radius +1]):
        #     local_max_index.append(i)

    candidate = []
    cand_diff = np.diff(local_max_index)
    # print(cand_diff)
    for i in range(len(local_max_index) - 2*winow_radius - 1):
        uniquearray = np.unique(cand_diff[i : i + 2 * winow_radius])
        if len(uniquearray) == 1 and uniquearray[0] == -1: # 5,4,3,2,1,0
        # if np.unique(np.diff(local_max_index[i : i + 2 * winow_radius + 1])) == np.array([-1]):
            candidate.append(i + 2 * winow_radius)

    distance = np.diff(candidate)

    # 计数每个元素的出现次数, 大于5次的才用于后续计算
    count = {}
    for item in distance:
        count[item] = count.get(item, 0) + 1

    # 保留计数大于5的distance； 小于5个刻度，说明，总数可能>10个刻度，有时候不一定显示了这么多刻度，这个数值可以调整。
    new_distance = []
    for item in distance:
        if count[item] > 5:
            new_distance.append(item)

    distance = np.array(new_distance)

    # 首先去掉1倍标准差以外的数据, 已经不需要，因为计数的步骤中，已经排除了很多内容
    # std_value = np.std(distance)
    # mean_value = np.mean(distance)
    #
    # print(mean_value, std_value)
    #
    # distance = distance[np.where(np.logical_and(distance >= mean_value-2*std_value, distance <= mean_value+2*std_value))]

    # 然后去掉众数间隔2及以上的数据
    mode_value = mode(distance)
    # 基本假设，众数是真实值，真实值最多出现两个，且相差一
    distance = distance[np.where(np.logical_and(distance >mode_value-2, distance < mode_value+2))]

    # plt.plot(distance, "o")
    # plt.show()

    # std_value = np.std(distance)
    # mean_value = np.mean(distance)
    # median_value = np.median(distance)
    # mode_value = mode(distance)

    return np.mean(distance)
def line_intersection(p1, p2, p3, p4):
    # 计算直线的斜率和截距
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None  # 平行或重合

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    return (x, y)

def remove_outliers(cross_points, level=3):
    std_values = np.std(cross_points, axis=0)
    mean_values = np.mean(cross_points, axis=0)
    points_result = []
    for point in cross_points:
        if point[0] <= mean_values[0] + level * std_values[0] and point[0] >= mean_values[0] - level * std_values[0]: # 加上等号，为了避免，std为0，即可能只有1个点时
            if point[1] <= mean_values[1] + level * std_values[1] and point[1] >= mean_values[1] - level * std_values[1]:
                points_result.append(point)

    print(mean_values, std_values)
    print(len(points_result))
    return np.array(points_result)

def extract_lines(img=None, k_expect=0.93):
    # cv2.imshow('img1', np.hstack([cv2.absdiff(img[:, :, 0], img[:, :, 1]), cv2.absdiff(img[:, :, 0], img[:, :, 2]),
    #                               cv2.absdiff(img[:, :, 2], img[:, :, 1])]))
    # cv2.waitKey(0)

    if img.ndim == 3:
        # 因为圆圈是红色的，所以1,vs,2; 详细分析见印象笔记-young
        img = cv2.absdiff(img[:, :, 0], img[:, :, 1])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测器
    # edges = cv2.Canny(gray, 200, 250, apertureSize=3) #250 280 # 200 250
    # edge1 = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=1)
    # edge1 = gray
    edges = cv2.Canny(img, 50, 150, apertureSize=3)  # 250 280 # 200 250对黄色的可以


    # show_img(edges)
    # 使用霍夫线变换检测线条
    lines = None
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=130, maxLineGap=15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=15)

    #numpy_img = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 9)  # 自动阈值二值化

    #circles = None
    #circles = cv2.HoughCircles(numpy_img, cv2.HOUGH_GRADIENT, 1, 50, param1=20, param2=60, minRadius=5, maxRadius=190)
    pos_lines = []
    neg_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                k = 1024 # 很大的数，几乎垂直
            else:
                k = (y2 - y1) / (x2 - x1)
            print(f"k: {k}")
            if np.abs(np.abs(k) - k_expect) < 0.05: # 此处的逻辑是，两条红色的线，斜率接近于0.93
                if k <= 0:
                    neg_lines.append(line[0])
                else:
                    pos_lines.append(line[0])
                # candidates.append(line[0])




    # cv2.imshow('Difference', img1 - img2)
    # cv2.imshow("Org img1", numpy_img)


    # cv2.imshow("Abs Diff23", img1)
    # cv2.imshow('Abs_Diff23.png', edge1)
    # cv2.imshow('edges.png', img1)


    # diff_img12 = cv2.absdiff(img2, img1)
    # diff_img13 = cv2.absdiff(img3, img1)
    # diff_img23 = cv2.absdiff(img2, img3)

    # cv2.imwrite("absDiff12.png", diff_img12)
    # cv2.imwrite("absDiff13.png", diff_img13)
    # cv2.imwrite("absDiff23.png", diff_img23)
    #
    # img = np.hstack([diff_img12, diff_img13, diff_img23])
    # cv2.imshow('img', img)
    # cv2.imwrite('12-13-23.png', img)
    # print(img.shape)
    # cv2.waitKey(0)

    return pos_lines, neg_lines

def find_cross_points(lines):
    pos_lines = lines[0]
    neg_lines = lines[1]

    cross_points = []
    for pos_l in pos_lines:
        for neg_l in neg_lines:
            crosspoint = line_intersection([pos_l[0],pos_l[1]], [pos_l[2],pos_l[3]], [neg_l[0],neg_l[1]], [neg_l[2],neg_l[3]])
            cross_points.append(crosspoint)

    # 可能会找到很多个交点
    cross_points = np.array(cross_points)
    # print(cross_points)
    cross_points = remove_outliers(cross_points, level=3)
    if cross_points.size > 4:
        cross_points = remove_outliers(cross_points, level=2)


    return cross_points
def scale_contour(contour, scale):
    """
    等比例放大轮廓
    :param contour: 输入轮廓
    :param scale: 缩放比例（>1表示放大，<1表示缩小）
    :return: 放大后的轮廓
    """
    # 计算轮廓的质心（中心点）
    M = cv2.moments(contour)
    if M["m00"] == 0:  # 避免除以零
        return contour

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # 将轮廓点相对于质心进行缩放
    scaled_contour = []
    for point in contour:
        x, y = point[0]
        # 计算点到质心的向量
        dx = x - cX
        dy = y - cY
        # 按比例放大向量
        new_x = cX + dx * scale
        new_y = cY + dy * scale
        scaled_contour.append([[new_x, new_y]])

    # 转换为numpy数组并确保是整数类型
    return np.array(scaled_contour, dtype=np.int32)


def get_intersection(rect, contour, img_shape):
    """
    计算矩形与轮廓的相交部分
    :param rect: 矩形区域 (x, y, width, height)
    :param contour: 轮廓
    :param img_shape: 图像形状 (height, width)
    :return: 相交部分的轮廓
    """
    # 创建两个空白图像
    contour_mask = np.zeros(img_shape[:2], np.uint8)
    rect_mask = np.zeros(img_shape[:2], np.uint8)

    # 绘制轮廓到掩码
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    # 绘制矩形到掩码
    x, y, w, h = rect
    cv2.rectangle(rect_mask, (x, y), (x + w, y + h), 255, -1)

    # 计算交集（逻辑与操作）
    intersection_mask = cv2.bitwise_and(contour_mask, rect_mask)

    # if np.mean(intersection_mask.ravel()) > 0:
    #     cv2.imshow("inter", intersection_mask)
    #     cv2.waitKey(0)

    # 从交集中提取轮廓
    # intersections, _ = cv2.findContours(intersection_mask,
    #                                     cv2.RETR_EXTERNAL,
    #                                     cv2.CHAIN_APPROX_SIMPLE)

    return intersection_mask

def align_images(image1, image2, max_features=500, good_match_percent=0.10):
    """
    对齐两张图像，处理位置移动问题
    返回对齐后的image2和变换矩阵
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用ORB检测器检测特征点和描述符
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # 匹配特征点
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)


    matches = matcher.match(descriptors1, descriptors2, None)

    if isinstance(matches, tuple):
        matches = list(matches)
    # 按匹配距离排序，保留最佳匹配
    matches.sort(key=lambda x: x.distance, reverse=False)
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]


    # 提取匹配点的坐标
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # print(points1)
    # 计算变换矩阵（使用RANSAC剔除异常值）
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # 应用变换矩阵对齐image2到image1
    height, width, channels = image1.shape
    aligned_image2 = cv2.warpPerspective(image2, h, (width, height))

    return aligned_image2, h


def detect_differences(image1, aligned_image2, threshold=30, kernel_size=3, rect=None):
    """
    检测两张对齐图像的差异
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(aligned_image2, cv2.COLOR_BGR2GRAY)

    # 计算差异
    # diff = cv2.absdiff(gray1, gray2)
    diff = gray2.astype(np.float32) - gray1.astype(np.float32)
    diff[diff<10] = 0
    diff = diff.astype(np.uint8)


    # 应用阈值突出明显差异
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY )
    # thresh = cv2.adaptiveThreshold(
    #     diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 125, 5)

    # 形态学操作去除噪声和小区域
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 找到差异区域的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 放大轮廓
    scaled_contours = [scale_contour(c, 1.3) for c in contours]  # 放大1.2倍

    # 在原图上标记差异区域
    result = aligned_image2.copy()

    # 绘制所有轮廓（-1 表示绘制所有轮廓）
    # cv2.drawContours(result, contours, -1, (0,0,255), thickness=1, lineType=None, hierarchy=None, maxLevel=None,
    #                  offset=None)

    contour_type = "contour" # rect
    contour_area_sum = 0 # 计算与特定区域有交集的所有contour的area的和
    for idx, contour in enumerate(contours):
        # 过滤小区域
        # cv2.contourArea()计算面积
        # cv2.arcLength() 计算周长
        contour_area = cv2.contourArea(contour)

        inter_section = get_intersection(rect, contour, image1.shape)

        inter_area = np.sum(inter_section)

        if contour_area > 100 and inter_area/contour_area > 0.1:  # 可调整阈值

            contour_area_sum += contour_area

            x, y, w, h = cv2.boundingRect(contour)
            if contour_type == 'rect':
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 120, 0), 1)
            else:
                # 计算两个矩形的面积
                cv2.drawContours(result, scaled_contours, idx, (0, 0, 120), thickness=1, lineType=None, hierarchy=None,
                                 maxLevel=None,
                                 offset=None)
        # rect[0] = 0
    # cv2.rectangle(result, rect, (100, 120, 250), 1)# 显示了一个矩形框
    # cv2.ellipse(result, (int(rect[0] + rect[2]/2), int(rect[1]+rect[3]/2)), (rect[2]//2, rect[3]//2), 0, 0, 360, (0, 120, 0), 1)

    cv2.putText(
        img=result,
        text="pct.={:.0f}%".format(100 * contour_area_sum/(0.25 * np.pi * rect[2] * rect[3])),
        # text="pct.={:.0f}".format(contour_area_sum),
        org=(50, 170),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2
    )

    return {
        "original1": image1,
        'image2': None,
        "original2": aligned_image2,
        "difference": diff,
        "thresholded": thresh,
        "marked2": result
    }


def visualize_results(results, scale_percent=80):
    """可视化处理结果"""
    # 缩放图像以便显示
    scaled = {}
    for name, img in results.items():
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        scaled[name] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # 转换BGR为RGB用于matplotlib显示
    for name in scaled:
        if len(scaled[name].shape) == 3:
            scaled[name] = cv2.cvtColor(scaled[name], cv2.COLOR_BGR2RGB)

    # 创建显示窗口
    fig = plt.figure(figsize=(18, 12))

    # 显示结果
    titles = {
        "original1": "Image 1",
        "original2": "Aligned Image 2",
        "difference": "Difference Map",
        "thresholded": "Thresholded Difference",
        "marked": "Differences Marked",
        "image2": 'Image 2'
    }

    titles = {
        "original1": "Image 1",
        "10": "10",
        "15": "15",
        "20": "20",
        "original2": "Image 2",
        "25": "25",
        "30": "30",
        "35": '35'
    }

    for i, (name, img) in enumerate(scaled.items(), 1):
        plt.subplot(2, 4, i)
        plt.imshow(img, cmap='gray' if name in ["difference", "thresholded"] else None)
        plt.title(titles[name])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return fig


def process_two_images(image1, image2, if_align=False, binary_threshold=30, width_x = 2, height_y = 4, drawcontour=True):  # x横向的焦域的大小mm  # y纵向的焦域的大小mm
    """处理两张图片的完整流程"""

    if image1 is None or image2 is None:
        raise FileNotFoundError("无法读取图像，请检查路径")

    if not drawcontour: # 如果不画框框，圈出不同的地方，则直接返回
        return image2

    # 对齐图像
    if if_align:
        image2, _ = align_images(image1, image2, 500, 0.5)


    points_mm = None

    # 利用左侧的刻度，识别每mm有多少像素
    try:
        points_mm = extract_points_mm(image1)
    except:
        pass

    if points_mm is None: # 默认一个区域的大小
        region_w = 30
        region_h = 52
    else:
        region_w = int(width_x * points_mm)
        region_h = int(height_y * points_mm)

    # 通过直线的交点，确定焦点，及确定焦点附近的区域大小
    try:
        pos_lines, neg_lines = extract_lines(img=image1, k_expect=0.75)
        cross_points = find_cross_points((pos_lines, neg_lines))

        center = cross_points.mean(axis=0) + 0.5  # 四舍五入更准确
        center = center.astype(int) + (0, 0)
    except:
        center = None

    if center is None:
        return image2

    rect = [center[0] - region_w//2, center[1] - region_h//2, region_w, region_h]
    # rect = [0, center[1] - region_h//2, region_w, region_h]

    # 检测差异
    results_th = detect_differences(image1, image2, threshold=binary_threshold, rect=rect)
    if drawcontour:
        return results_th["marked2"]

    return results_th['original2']




if __name__ == "__main__":
    # 替换为你的两张图片路径

    dirname = 'compareImages/0823/imgs'
    # dirname = '0729'

    save_dir = os.path.join(dirname, 'results-diff')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = os.listdir(dirname)
    # files = os.listdir('compareImages/0729')

    files = [file for file in files if file.endswith('.png')]

    filter_types = ['median', 'gaussian', 'bilateral', 'box',
                    'db4', 'haar', 'sym5',
                    'fastNlMeansDenoising'];

    # for filter in filter_types:
    for index in range(0, len(files) - 1, 2):
        print(files[index])
        # 读取图像
        path1 = os.path.join(dirname, files[index])
        path2 = os.path.join(dirname, files[index + 1])

        # path1 = 'New_Ultrasound_test_lines/5.png'
        # path2 = 'New_Ultrasound_test_lines/2.jpg'

        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)

        image2_marked = process_two_images(image1, image2, if_align=False, binary_threshold=30)
        cv2.imshow('marked2', image2_marked)
        cv2.waitKey(0)


        results = {}
        for th in range(10, 36, 5):
            results_th = process_two_images(image1, image2, if_align=False, binary_threshold=th)
            results[f'{th}'] = results_th
        results['original1'] = image1
        results['original2'] = image2

        # print(np.sum(results['image2']))
        # 可视化结果
        fig = visualize_results(results)

        # 保存结果
        # output_dir = "compareImages/difference_results"
        # os.makedirs(output_dir, exist_ok=True)
        #
        # for name, img in results.items():
        #     cv2.imwrite(f"{output_dir}/{name}.jpg", img)
        # print(f"结果已保存至 {output_dir} 目录")

        fig.savefig(os.path.join(save_dir, files[index]), dpi=800)

        # cv2.imwrite(os.path.join(save_dir, files[index].replace('.png', f'.jpg')), img)
        # break



