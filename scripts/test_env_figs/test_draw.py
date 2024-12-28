from PIL import Image
import os

# 图片文件夹路径
image_folder = '.'
# 获取所有图片文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

# 确保有 15 张图片
if len(image_files) != 15:
    raise ValueError("需要 15 张图片")

# 加载图片
images = [Image.open(os.path.join(image_folder, file)) for file in image_files]

# 设置每张图片的尺寸
image_width, image_height = images[0].size

# 创建一个新的大图，大小为 (5 * image_width, 3 * image_height)
big_image = Image.new('RGB', (5 * image_width, 3 * image_height))

# 将图片放入大图
for index, image in enumerate(images):
    x = (index % 5) * image_width  # 计算 x 坐标
    y = (index // 5) * image_height  # 计算 y 坐标
    big_image.paste(image, (x, y))

# 保存大图
big_image.save('combined_image.jpg')
big_image.show()  # 显示合成后的大图