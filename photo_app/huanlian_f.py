#!/usr/bin/python

import cv2 as cv
import dlib
import numpy as np
import sys

PREDICTOR_PATH = 'C:\\E\\AI camera\\photo_app\static\\face_landmark_dilb\\shape_predictor_68_face_landmarks.dat'
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

TRANSFORM_POINT = [17, 26, 57]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def histMatch_core(src, dst, mask=None):
    srcHist = [0] * 256
    dstHist = [0] * 256
    srcProb = [.0] * 256;  # 源图像各个灰度概率
    dstProb = [.0] * 256;  # 目标图像各个灰度概率

    for h in range(src.shape[0]):
        for w in range(src.shape[1]):
            if mask is None:
                srcHist[int(src[h, w])] += 1
                dstHist[int(dst[h, w])] += 1
            else:
                if mask[h, w] > 0:
                    srcHist[int(src[h, w])] += 1
                    dstHist[int(dst[h, w])] += 1

    resloution = src.shape[0] * src.shape[1]

    if mask is not None:
        resloution = 0
        for h in range(mask.shape[0]):
            for w in range(mask.shape[1]):
                if mask[h, w] > 0:
                    resloution += 1

    for i in range(256):
        srcProb[i] = srcHist[i] / resloution
        dstProb[i] = dstHist[i] / resloution

    # 直方图均衡化
    srcMap = [0] * 256
    dstMap = [0] * 256

    # 累积概率
    for i in range(256):
        srcTmp = .0
        dstTmp = .0
        for j in range(i + 1):
            srcTmp += srcProb[j]
            dstTmp += dstProb[j]

        srcMapTmp = srcTmp * 255 + .5
        dstMapTmp = dstTmp * 255 + .5
        srcMap[i] = srcMapTmp if srcMapTmp <= 255.0 else 255.0
        dstMap[i] = dstMapTmp if dstMapTmp <= 255.0 else 255.0

    matchMap = [0] * 256
    for i in range(256):
        pixel = 0
        pixel_2 = 0
        num = 0  # 可能出现一对多
        cur = int(srcMap[i])
        for j in range(256):
            tmp = int(dstMap[j])
            if cur == tmp:
                pixel += j
                num += 1
            elif cur < tmp:  # 概率累计函数 递增
                pixel_2 = j
                break

        matchMap[i] = int(pixel / num) if num > 0 else int(pixel_2)

    newImg = np.zeros(src.shape[:2], dtype=np.uint8)
    for h in range(src.shape[0]):
        for w in range(src.shape[1]):
            if mask is None:
                newImg[h, w] = matchMap[src[h, w]]
            else:
                if mask[h, w] > 0:
                    newImg[h, w] = matchMap[src[h, w]]
                else:
                    newImg[h, w] = src[h, w]

    return newImg


# src1 src2 mask must have the same size
def histMatch(src1, src2, mask=None, dst=None):
    sB, sG, sR = cv.split(src1)
    dB, dG, dR = cv.split(src2)

    if mask.shape[2] > 1:
        rM, gM, bM = cv.split(mask)
        nB = histMatch_core(sB, dB, rM)
        nG = histMatch_core(sG, dG, gM)
        nR = histMatch_core(sR, dR, bM)
    else:
        nB = histMatch_core(sB, dB, mask)
        nG = histMatch_core(sG, dG, mask)
        nR = histMatch_core(sR, dR, mask)

    newImg = cv.merge([nB, nG, nR])

    if dst is not None:
        dst = newImg

    return newImg


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im, winname='debug'):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    draw = im.copy()
    for _, d in enumerate(rects):
        cv.rectangle(draw, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv.putText(im, str(idx), pos,
                   fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
                   fontScale=0.4,
                   color=(0, 0, 255))
        cv.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv.convexHull(points)  # 得到凸包
    cv.fillConvexPoly(im, points, color=color)  # 绘制填充


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))  # -> rgb rgbr rgb

    im = (cv.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


# 读取图片文件并获取特征点
def read_im_and_landmarks(fname):
    im = cv.imread(fname, cv.IMREAD_COLOR)

    im = cv.resize(im, (im.shape[1] * SCALE_FACTOR,
                        im.shape[0] * SCALE_FACTOR))

    # 68个特征点
    s = get_landmarks(im, fname)  # mat

    return im, s


def warp_im(mask, M, dshape):
    output_im = np.zeros(dshape, dtype=mask.dtype)
    cv.warpAffine(mask,
                  M[:2],
                  (dshape[1], dshape[0]),
                  dst=output_im,
                  borderMode=cv.BORDER_TRANSPARENT,
                  flags=cv.WARP_INVERSE_MAP)
    return output_im


def getAffineTransform(_srcPoint, _dstPoint):
    srcPoint = _srcPoint.astype(np.float32)
    dstPoint = _dstPoint.astype(np.float32)
    return cv.getAffineTransform(srcPoint, dstPoint)




