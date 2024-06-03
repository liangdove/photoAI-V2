from django.db import models

# Create your models here.
class User(models.Model):
    gender = (
        ('male', "男"),
        ('female', "女"),
    )

    name = models.CharField(max_length=128, unique=True)
    password = models.CharField(max_length=128)
    email = models.EmailField(unique=True)
    sex = models.CharField(max_length=32, choices=gender, default='男')
    c_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["-c_time"]
        verbose_name = "用户"
        verbose_name_plural = "用户"

class Photo(models.Model):
    title = models.CharField(max_length=100, default='Default Title')
    image = models.ImageField(upload_to='photos/')
    user = models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)


    def __str__(self):
        return self.title

class ProcessedPhoto(models.Model):
    original_photo = models.ForeignKey(Photo, on_delete=models.CASCADE)
    processed_image = models.ImageField(upload_to='processed_photos/')

    def __str__(self):
        return f"Processed - {self.original_photo.title}"


