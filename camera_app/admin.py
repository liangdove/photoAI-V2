from django.contrib import admin
from photo_app import models
# Register your models here.
admin.site.register(models.User)
admin.site.register(models.Photo)
admin.site.register(models.ProcessedPhoto)