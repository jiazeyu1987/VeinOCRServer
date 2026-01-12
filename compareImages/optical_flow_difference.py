import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def resize_images(img1, img2, max_size=800):
    """确保两张图像尺寸完全一致"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 计算统一缩放比例
    scale = min(max_size / max(h1, w1), max_size / max(h2, w2))

    # 计算新尺寸（确保整数）
    new_w = int(min(w1, w2) * scale)
    new_h = int(min(h1, h2) * scale)

    # 强制缩放到完全相同的尺寸
    img1_resized = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img1_resized, img2_resized


def compute_optical_flow(img1, img2):
    """计算稠密光流并验证尺寸匹配"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if gray1.shape != gray2.shape:
        raise ValueError(f"图像尺寸不匹配: {gray1.shape} vs {gray2.shape}")

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    if flow.shape[:2] != gray1.shape:
        raise ValueError(f"光流场尺寸与图像不匹配: {flow.shape[:2]} vs {gray1.shape}")

    return flow, gray1, gray2


def warp_image_using_flow(img, flow):
    """根据光流场对齐图像"""
    if img.shape[:2] != flow.shape[:2]:
        raise ValueError(f"图像与光流场尺寸不匹配: {img.shape[:2]} vs {flow.shape[:2]}")

    h, w = flow.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    x_grid, y_grid = np.meshgrid(x, y)

    src_x = x_grid - flow[..., 0]
    src_y = y_grid - flow[..., 1]

    aligned_img = cv2.remap(
        img, src_x.astype(np.float32), src_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return aligned_img


def visualize_flow(flow, step=16):
    """修复光流可视化坐标生成逻辑，确保尺寸匹配"""
    h, w = flow.shape[:2]
    flow_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白色背景

    # 关键修复：确保生成的坐标网格在图像范围内
    # 计算有效的y坐标范围（避免超出图像高度）
    y_max = h - 1
    y_start = int(step / 2)
    y_stop = y_max if (y_max - y_start) % step == 0 else y_max - ((y_max - y_start) % step)
    y_coords = np.arange(y_start, y_stop + 1, step, dtype=int)

    # 计算有效的x坐标范围（避免超出图像宽度）
    x_max = w - 1
    x_start = int(step / 2)
    x_stop = x_max if (x_max - x_start) % step == 0 else x_max - ((x_max - x_start) % step)
    x_coords = np.arange(x_start, x_stop + 1, step, dtype=int)

    # 创建网格并验证尺寸
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    y_grid = y_grid.flatten()
    x_grid = x_grid.flatten()

    # 验证坐标网格尺寸匹配
    if len(x_grid) != len(y_grid):
        raise ValueError(f"X和Y坐标数量不匹配: {len(x_grid)} vs {len(y_grid)}")

    # 提取光流向量并验证尺寸
    fx = flow[y_grid, x_grid, 0].flatten()
    fy = flow[y_grid, x_grid, 1].flatten()

    if len(fx) != len(x_grid) or len(fy) != len(y_grid):
        raise ValueError(f"光流向量与坐标数量不匹配: {len(fx)} vs {len(x_grid)}")

    # 绘制光流箭头
    for x, y, dx, dy in zip(x_grid, y_grid, fx, fy):
        end_x = int(x + dx)
        end_y = int(y + dy)
        # 确保箭头终点在图像范围内
        end_x = np.clip(end_x, 0, w - 1)
        end_y = np.clip(end_y, 0, h - 1)
        # 绘制箭头
        cv2.arrowedLine(
            flow_img, (x, y), (end_x, end_y),
            (0, 255, 0), 1, tipLength=0.3
        )

    return flow_img


def detect_differences_with_flow(img1, img2):
    """完整处理流程"""
    try:
        img1_resized, img2_resized = resize_images(img1, img2)
        if img1_resized.shape != img2_resized.shape:
            raise ValueError(f"图像缩放后尺寸仍不匹配: {img1_resized.shape} vs {img2_resized.shape}")

        flow, gray1, gray2 = compute_optical_flow(img1_resized, img2_resized)
        aligned_img2 = warp_image_using_flow(img2_resized, flow)

        if aligned_img2.shape != img1_resized.shape:
            raise ValueError(f"对齐后图像尺寸不匹配: {aligned_img2.shape} vs {img1_resized.shape}")

        gray_aligned = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
        if gray1.shape != gray_aligned.shape:
            raise ValueError(f"灰度图尺寸不匹配: {gray1.shape} vs {gray_aligned.shape}")

        # gray1 = cv2.GaussianBlur(gray1, (9, 9), 0)
        # gray_aligned = cv2.GaussianBlur(gray_aligned, (9, 9), 0)

        diff = gray_aligned.astype(np.float32) - gray1.astype(np.float32)
        mean_value = np.mean(diff[50:-200, 50:-50])
        sigma = np.std(diff[50:-200, 50:-50])

        print(mean_value - sigma, mean_value, sigma)

        diff[diff < mean_value*1.1] = 0
        diff = diff.astype(np.uint8)


        print(np.sum(diff[50:-200, 50:-50]))
        diff_orig = cv2.absdiff(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY))
        print(np.sum(diff_orig[50:-200, 50:-50]))


        diff_concate = np.hstack((diff_orig, diff))

        cv2.imshow('diff compare ', diff_concate[50:-200, 50:-50])
        cv2.waitKey(0)


        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        result = img1_resized.copy()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 生成光流可视化（修复后）
        flow_viz = visualize_flow(flow)

        return {
            "original1": img1_resized,
            "original2": img2_resized,
            "flow_visualization": flow_viz,
            "aligned_img2": aligned_img2,
            "difference": diff,
            "marked_differences": result
        }

    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None


def show_results(results):
    """展示处理结果"""
    if not results:
        return

    plt.figure(figsize=(20, 15))
    titles = [
        "Original Image 1",
        "Original Image 2",
        "Optical Flow Visualization",
        "Aligned Image 2",
        "Difference Map",
        "Marked Differences"
    ]
    images = [
        cv2.cvtColor(results["original1"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["original2"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["flow_visualization"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["aligned_img2"], cv2.COLOR_BGR2RGB),
        results["difference"],
        cv2.cvtColor(results["marked_differences"], cv2.COLOR_BGR2RGB)
    ]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray' if i == 4 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    output_dir = "optical_flow_results"
    os.makedirs(output_dir, exist_ok=True)
    for name, img in results.items():
        cv2.imwrite(f"{output_dir}/{name}.jpg", img)
    print(f"结果已保存至 {output_dir} 目录")


if __name__ == "__main__":
    # img1_path = "twosame/2025-08-13_17-16-12.960_before.png"
    # img2_path = "twosame/2025-08-13_17-16-14.071_after.png"

    img2_path

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.GaussianBlur(img1, (7, 7), 0)
    img2 = cv2.GaussianBlur(img2, (7, 7), 0)

    # img1 = cv2.bilateralFilter(img1, 9, 75, 75)
    # img2 = cv2.bilateralFilter(img2, 9, 75, 75)

    # img1 = cv2.medianBlur(img1, 15)
    # img2 = cv2.medianBlur(img2, 15)


    if img1 is None or img2 is None:
        print("无法读取图像，请检查路径")
    else:
        results = detect_differences_with_flow(img1, img2)
        show_results(results)
