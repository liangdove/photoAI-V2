# 算法库，已经弃用

import base64

import numpy

from .models import Photo, ProcessedPhoto
from django.core.files.base import ContentFile
import io

import cv2
import face_recognition
from PIL import Image, ImageDraw
import dlib
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def crop(photo):
    processed_image = Image.open(photo.image)
    width, height = processed_image.size
    left = width / 4
    top = height / 4
    right = 3 * width / 4
    bottom = 3 * height / 4
    cropped_image = processed_image.crop((left, top, right, bottom))

    return save_and_encode_image(cropped_image, photo)

def rotate(photo):
    processed_image = Image.open(photo.image)
    rotated_image = processed_image.rotate(90)

    return save_and_encode_image(rotated_image, photo)

def face_detection(photo):
    # processed_image = Image.open(photo.image)
    # processed_image = np.array(Image.open(photo.image))
    processed_image = cv2.imread(photo.image.path)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # 使用预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 绘制矩形到检测到的人脸上
    for (x, y, w, h) in faces:
        cv2.rectangle(processed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # cv2.imread()读取图像时，返回的是一个numpy数组，而不是Pillow的Image对象。需要将numpy数组转换为Pillow的Image对象
    processed_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    return save_and_encode_image(processed_image, photo)

def video_face_detection():
    # 视频人脸检测
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 绘制矩形到检测到的人脸上
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示框出人脸的图像
        cv2.imshow("Face Detection", frame)  # 显示图像

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def capture(video_stream):
    # 从视频流中读取当前帧
    video_stream.seek(0)
    current_frame = Image.open(io.BytesIO(video_stream.read()))

    # 保存当前帧为 JPEG 格式的图像文件
    buffer = io.BytesIO()
    current_frame.save(buffer, format='JPEG')
    image_data = buffer.getvalue()

    return image_data

    # data = json.loads(request.body)
    # image_data = data.get('image_data')
    # if image_data:
    #     # 创建新的 Photo 实例并保存图片
    #     photo = Photo.objects.create(title='Temp Title')
    #     # 解码图片数据并保存到文件
    #     format, imgstr = image_data.split(';base64,')
    #     ext = format.split('/')[-1]
    #     photo.image.save(f'photo_{photo.id}.{ext}', ContentFile(base64.b64decode(imgstr)), save=True)
    #     # 更新标题为图片的ID
    #     photo.title = 'Captured Photo of Photo ' + str(photo.id)
    #     photo.save()
    #     return JsonResponse({'success': True, 'photo_id': photo.id})

def encode_image(image_field):
    # 对上传图像进行数据流编码
    image = Image.open(image_field)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()

def save_and_encode_image(processed_image, photo):
    byte_arr = io.BytesIO()
    processed_image.save(byte_arr, format='JPEG')
    processed_image_file = ContentFile(byte_arr.getvalue(), 'processed.jpg')

    processed_photo = ProcessedPhoto.objects.create(original_photo=photo, processed_image=processed_image_file)
    processed_photo.save()
    # 将处理后的图像转换为Base64
    return base64.b64encode(byte_arr.getvalue()).decode()

def makeup(photo):
    # 加载图片到numpy array
    image = face_recognition.load_image_file(photo.image.path)

    # 标识脸部特征
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # 绘制眉毛
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # 绘制嘴唇
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # 绘制眼睛
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # 绘制眼线
        d.line(
            face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]],
            fill=(0, 0, 0, 110),
            width=6)
        d.line(
            face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]],
            fill=(0, 0, 0, 110),
            width=6)
    return save_and_encode_image(pil_image, photo)

def outline(photo):
    # coding=utf-8
    # 绘制面部轮廓

    # 将图片文件加载到numpy 数组中
    image = face_recognition.load_image_file(photo.image.path)

    # 查找图像中所有面部的所有面部特征
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        facial_features = [
            'chin',  # 下巴
            'left_eyebrow',  # 左眉毛
            'right_eyebrow',  # 右眉毛
            'nose_bridge',  # 鼻樑
            'nose_tip',  # 鼻尖
            'left_eye',  # 左眼
            'right_eye',  # 右眼
            'top_lip',  # 上嘴唇
            'bottom_lip'  # 下嘴唇
        ]
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], fill=(0, 255, 255), width=2)
        return save_and_encode_image(pil_image, photo)

def face_landmark(photo):
    img = cv2.imread(photo.image.path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸分类器
    detector = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    predictor = dlib.shape_predictor("photo_app/static/face_landmark_dilb/shape_predictor_68_face_landmarks.dat")

    dets = detector(gray, 1)
    for face in dets:
        # 在图片中标注人脸，并显示
        # left = face.left()
        # top = face.top()
        # right = face.right()
        # bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        # cv2.imshow("image", img)

        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (0, 255, 0), 2)
        # c
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return save_and_encode_image(img, photo)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def katong_f(photo):

    # 调用model
    img_cartoon = pipeline(Tasks.image_portrait_stylization,
                           model='damo/cv_unet_person-image-cartoon_compound-models')
    img_path = photo.image.path
    result = img_cartoon(img_path)
    cv2.imwrite("media/katong_dir/katong-{}.jpg".format(photo.title.encode('utf-8')), result[OutputKeys.OUTPUT_IMG])

    # 因为result[OutputKeys.OUTPUT_IMG]是BGR，转换通道为RGB
    result[OutputKeys.OUTPUT_IMG] = result[OutputKeys.OUTPUT_IMG][:, :, ::-1]
    # 转成pil，便于存储
    processed_img = Image.fromarray(numpy.uint8(result[OutputKeys.OUTPUT_IMG]), mode='RGB')

    # 将处理后的图像保存到 BytesIO 缓冲区
    processed_img_io = io.BytesIO()
    processed_img.save(processed_img_io, format='PNG')
    # 创建content对象
    processed_image_file = ContentFile(processed_img_io.getvalue(), "processed.jpg")
    processed_photo_db = ProcessedPhoto.objects.create(original_photo=photo, processed_image=processed_image_file)
    return processed_photo_db

def katong2_f(photo):
    # 调用model
    img_cartoon = pipeline(Tasks.image_portrait_stylization,
                           model='damo/cv_unet_person-image-cartoon-handdrawn_compound-models')
    img_path = photo.image.path
    result = img_cartoon(img_path)
    cv2.imwrite("media/katong_dir/katong-{}.jpg".format(photo.title.encode('utf-8')), result[OutputKeys.OUTPUT_IMG])

    # 因为result[OutputKeys.OUTPUT_IMG]是BGR（np类型），转换通道为RGB
    result[OutputKeys.OUTPUT_IMG] = result[OutputKeys.OUTPUT_IMG][:, :, ::-1]
    # 转成pil，便于存储
    processed_img = Image.fromarray(numpy.uint8(result[OutputKeys.OUTPUT_IMG]), mode='RGB')

    # 将处理后的图像保存到 BytesIO 缓冲区
    processed_img_io = io.BytesIO()
    processed_img.save(processed_img_io, format='PNG')
    # 创建content对象
    processed_image_file = ContentFile(processed_img_io.getvalue(), "processed.jpg")
    processed_photo_db = ProcessedPhoto.objects.create(original_photo=photo, processed_image=processed_image_file)
    return processed_photo_db

def shoushi_f(photo):
    hand_static = pipeline(Tasks.hand_static, model='damo/cv_mobileface_hand-static')
    result_status = hand_static(photo.image.path)
    result = result_status[OutputKeys.OUTPUT]

    dir = {
        "bixin": "单手比心",

        "d_bixin": "双手比心",

        "d_first_left": "左手抱拳",

        "d_fist_right": "右手抱拳",

        "d_hand": "双手交叉",

        "fashe": "发射形状",

        "fist": "握拳",

        "five": "手掌张开",

        "ok": "ok",

        "one": "用手比数字1",

        "tuoju": "托举手势",

        "two": "比耶",

        "yaogun": "摇滚手势",

        "zan": "点赞",

        "unrecog": "未识别"
    }
    result_val = dir.get(result, '未识别')
    return result_val








