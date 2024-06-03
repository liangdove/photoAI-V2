import base64
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from djangoProject import settings
from . import models
from . import forms # 引入表单
from django.shortcuts import render, redirect
from photo_app.models import Photo
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import os
import cv2
from django.views.decorators import gzip
import dlib
from django.shortcuts import render
from django.http import JsonResponse
from photo_app.models import Photo



