import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon_compound-models')
# 图像本地路径
# img_path = 'input.png'
# 图像url链接
img_path = 'media/photos/3.jpg'
result = img_cartoon(img_path)
cv2.imwrite('static/result8.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')
