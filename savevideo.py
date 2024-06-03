import cv2
import os
 
# 图片路径和视频格式
image_folder = './videos'
video_name = 'output_video.avi'
 
# 获取图片列表
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # 如果需要可以对图片进行排序
 
# 从第一张图片获取视频尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
 
# 定义视频编码和创建VideoWriter对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
 
# 将图片逐一写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
 
# 释放VideoWriter对象
video.release()