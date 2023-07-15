# -*- coding: UTF-8 -*-
import cv2 as cv
import math
import numpy as np


def is_contour_circle(contour):
    """
    判断边界是否像圆形
    :param contour:
    :return:
    """
    # 进行边界近似，epsilon值越小，近似程度越高
    epsilon = 0.03 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    is_circle = False  # 判断边界是否像圆形
    if len(approx) >= 5:
        (x, y), radius = cv.minEnclosingCircle(contour)  # 计算边界的最小外接圆
        area = cv.contourArea(contour)  # 计算边界的面积和周长
        perimeter = cv.arcLength(contour, True)
        # 计算圆形的面积和周长
        circle_area = np.pi * radius ** 2
        circle_perimeter = 2 * np.pi * radius
        area_ratio = area / circle_area  # 计算边界的面积与圆形面积的比值
        perimeter_ratio = perimeter / circle_perimeter  # 计算边界的周长与圆形周长的比值
        # 判断比值是否接近于1，可以根据实际情况来调整abs(area_ratio-1)和abs(perimeter_ratio-1)的临界值
        if abs(area_ratio - 1) < 0.6 and abs(perimeter_ratio - 1) < 0.5:
            is_circle = True
    return is_circle


def distance_point_to_line(point, line_start, line_end):
    """
    计算点到线段的垂直距离
    :param point:
    :param line_start:
    :param line_end:
    :return:
    """
    # 将点和线段的起点、终点转换为numpy数组
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    # 计算线段的向量和点到线段起点的向量
    line_vector = line_end - line_start
    point_vector = point - line_start
    # 计算点在线段上的投影点
    projection = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    # 返回点到线段的垂直距离
    perpendicular = line_start + projection * line_vector
    return np.linalg.norm(point - perpendicular)


def get_min_distance_edge(point, rectangle):
    """
    获取矩形中距离一个点最近的边
    :param point:
    :param rectangle:
    :return:
    """
    l1 = None
    min_distance = float('inf')
    for i in range(4):
        start = rectangle[i]
        end = rectangle[(i + 1) % 4]
        distance = distance_point_to_line(point, start, end)
        if distance < min_distance:
            min_distance = distance
            l1 = [start, end]
    return l1


def digital_segmentation(image):
    """
    数字分割
    :param image:
    :return:
    """
    global min_areas_idx
    image = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 灰度图像
    gray = cv.medianBlur(gray, 5)  # 中值滤波
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)  # 二值化
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    binary1 = binary.copy()
    # area_threshold = 10  # 设置面积阈值为10
    digit_rects = []
    centers = []
    maxarea = 0
    minarea = float('inf')
    dot_contour = None
    angle = 0
    # 获取距离小数点最近的边的旋转角度
    for contour in contours:
        area = cv.contourArea(contour)  # 计算轮廓面积
        # if area < area_threshold:
        #    continue
        if area > 100:
            x, y, w, h = cv.boundingRect(contour)
            center_x = x + w / 2
            center_y = y + h / 2
            centers.append((center_x, center_y))
        rect = cv.minAreaRect(contour)  # 获取最小矩形框
        digit_rects.append(rect)  # 将最小矩形框添加到列表中
        if minarea > area:
            minarea = area
            dot_contour = contour
        if maxarea < area:
            maxarea = area
    # 如果最小面积得5倍小于最大面积，且最小面积得轮廓是圆形，则认为存在小数点
    if 5 * minarea < maxarea and is_contour_circle(dot_contour):
        # 计算整体数字的最小矩形框
        combined_contour = np.concatenate(contours)  # 将所有轮廓点拼接起来
        overall_rect = cv.minAreaRect(combined_contour)  # 获取整体数字的最小矩形框
        # 获取整体数字的最小矩形框的四个顶点
        box = cv.boxPoints(overall_rect)  # 获取最小矩形框的四个顶点
        # 获取 `dot_contour` 中心点
        moments = cv.moments(dot_contour)
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        # 计算整体最小矩形框的边与 `dot_contour` 中心的距离和角度
        min_distance_edge = get_min_distance_edge((center_x, center_y), box)
        angle = math.atan2(min_distance_edge[0][1] - min_distance_edge[1][1],
                           min_distance_edge[0][0] - min_distance_edge[1][0]) * 180 / math.pi
    elif len(centers) > 1:
        # 计算数字的整体方向
        vx, vy, x0, y0 = cv.fitLine(np.array(centers), cv.DIST_L2, 0, 0.01, 0.01)
        angle = math.atan2(vy, vx) * 180 / math.pi
    # 如果角度偏离水平线超过15度，则进行旋转
    if abs(angle) > 15:
        # 获取图像的尺寸
        height, width = image.shape[:2]
        # 计算旋转矩形框的中心坐标
        center_x = width // 2
        center_y = height // 2
        # 创建旋转矩阵
        rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        # 计算旋转后的图像尺寸
        new_width = int(width * abs(math.cos(math.radians(angle))) + height * abs(math.sin(math.radians(angle))))
        new_height = int(height * abs(math.cos(math.radians(angle))) + width * abs(math.sin(math.radians(angle))))
        # 调整旋转矩阵的平移部分，以确保旋转后图像完整显示
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        # 执行图像旋转
        binary1 = cv.warpAffine(binary, rotation_matrix, (new_width, new_height))
    # x轴方向膨胀,将分离的部分与整体连接起来，比如5上面的一横写分开了，就会连接起来
    # kernelX = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    # binary1 = cv.morphologyEx(binary1, cv.MORPH_CLOSE, kernelX, iterations=3)
    contours, hierarchy = cv.findContours(binary1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    items = []
    number_images = []
    areas = []
    binary2 = binary1.copy()
    for contour in contours:
        area = cv.contourArea(contour)
        # if area < area_threshold:
        #     continue
        x, y, w, h = cv.boundingRect(contour)  # 获取最小矩形框
        cv.rectangle(binary2, (x, y), (x + w, y + h), (255, 255, 255), 2)
        items.append([x, y, w, h])
        areas.append([x, area])
        if minarea > area:
            minarea = area  # 获取最小面积
            dot_contour = contour  # 获取最小面积对应的轮廓
    cv.namedWindow("measure_object", cv.WINDOW_NORMAL)
    cv.imshow("measure_object", binary2)  # 显示最小矩形框
    items = sorted(items, key=lambda s: s[0], reverse=False)  # 按照x轴坐标排序
    areas = sorted(areas, key=lambda s: s[0], reverse=False)  # 按照x轴坐标排序
    areas = [i[1] for i in areas]  # 获得二维列表得第一列 这一列是按顺序保存得面积
    min_areas_idx = None
    # 如果最小面积得5倍小于最大面积，且最小面积得轮廓是圆形，则认为存在小数点
    if 5 * min(areas) < max(areas) and is_contour_circle(dot_contour):
        min_areas_idx = areas.index(min(areas))  # 返回最小面积得索引
        print('小数点的位置为：', min_areas_idx)
    else:
        print("不存在小数点")
    for item in items:
        splite_image = binary1[(item[1] - 2):(item[1] - 2) + (item[3] + 2),
                       (item[0] - 2):(item[0] - 2) + (item[2] + 2)]  # 裁剪出每个数字
        # 检查图像是否不是正方形
        height, width = splite_image.shape
        if height != width:
            # 计算填充大小以使图像变为正方形
            padding_size = max(height, width) - min(height, width)
            padding_top = padding_size // 2
            padding_bottom = padding_size - padding_top
            # 添加填充以使图像变为正方形
            if height < width:
                padding = ((padding_top, padding_bottom), (0, 0))
            else:
                padding = ((0, 0), (padding_top, padding_bottom))
            splite_image = np.pad(splite_image, padding, mode='constant', constant_values=0)  # 填充padding成正方形
        splite_image = np.pad(splite_image, pad_width=int(height * 0.18), mode='constant',
                              constant_values=0)  # 填充padding，增加边缘，利于识别
        number_images.append(splite_image)
        cv.imwrite('./get_num_img/' + str(len(number_images) - 1) + '.png', splite_image)  # 保存分割出的元素
    print('已完成数字的定位、分割、保存')
    return number_images, min_areas_idx


if __name__ == '__main__':
    # 测试
    src = cv.imread("./test_img/test13.png")
    number, min_areas_idx = digital_segmentation(src)
    cv.waitKey(0)
    cv.destroyAllWindows()
