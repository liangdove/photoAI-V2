"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from photo_app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index,name='index'),
    path('login/', views.login,name='login'),
    path('register/', views.register,name='register'),
    path('logout/', views.logout,name='logout'),
    path('captcha/', include('captcha.urls')),
    path('image/',views.image_view,name='image'),
    path('setting/',views.profile_settings,name='setting'),
    path('index/image_recognition/', views.image_recognition, name='image_recognition'),
    path('index/face_recognition/', views.video_face_detection, name='face_recognition'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('index/contour_identification/', views.contour_identification, name='contour_identification'),
    path('index/digital_makeup/', views.digital_makeup, name='digital_makeup'),
    path('index/funtion1/', views.funtion1, name='funtion1'), # 美颜1
    path('index/meiyan2/', views.meiyan2, name='meiyan2'),
    path('index/funtion2/', views.funtion2, name='funtion2'),
    path('index/funtion3/', views.funtion3, name='funtion3'),
    path('index/katong/', views.katong, name='katong'),
    path('index/katong2/', views.katong2, name='katong2'),
    path('index/huanlian/', views.huanlian, name='huanlian'),
    path('index/shoushi/', views.shoushi, name='shoushi'),
    path('search_photos/', views.search_photos, name='search_photos'),
    path('camera_app/', include('camera_app.urls'))
  ]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

