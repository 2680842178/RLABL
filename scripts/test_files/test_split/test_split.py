import cv2
import os

folder_path = 'test_split/dataset/0'
saved_images_folder = 'buffer'
if not os.path.exists(saved_images_folder):
    os.makedirs(saved_images_folder)
# 获取文件夹中所有文件的列表
files = os.listdir(folder_path)
images = [f for f in files]

# 按文件名顺序对图片列表进行排序
images.sort()

# 遍历并读取每张图片
for image_name in images:
    # 构建完整的文件路径
    image_path = os.path.join(folder_path, image_name)
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 获取已存储的图片数量
    saved_image_count = len(
        [name for name in os.listdir(saved_images_folder) if os.path.isfile(os.path.join(saved_images_folder, name))])

    # 读取已存储的图片并计算直方图，用于后续比较
    saved_images = []
    for filename in os.listdir(saved_images_folder):
        img = cv2.imread(os.path.join(saved_images_folder, filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            saved_images.append((filename, img, hist))


    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed = []
        # 遍历轮廓，找到最小的包围矩形，并切割保存
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y + h, x:x + w]
        processed.append(roi)

    for processed_image in processed:

        processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()

        # 与已存储的图片比较相似度
        similar = 0
        for saved_image_name, saved_image, saved_hist in saved_images:
            saved_hist = cv2.normalize(saved_hist, saved_hist).flatten()
            # correlation = np.corrcoef(processed_hist, saved_hist)[0, 1]
            correlation = cv2.compareHist(processed_hist, saved_hist, cv2.HISTCMP_CORREL)
            # print(f"直方图交集值: {intersection}")
            similar = max(similar,abs(correlation))
            if similar == 1:
                break

        print(similar)
        # 如果相似度都很低，则存入
        if similar < 1:
            new_filename = f"{saved_image_count}.bmp"  # 生成新的文件名
            save_path = os.path.join(saved_images_folder, new_filename)
            cv2.imwrite(save_path, processed_image)
            saved_images.append((new_filename, processed_image, processed_hist))  # 更新已存储的图片列表
            saved_image_count += 1  # 更新已存储图片的数量
            print(f"Image saved: {new_filename}")
        else:
            print(f"Image skipped due to low similarity: {image_name}")
