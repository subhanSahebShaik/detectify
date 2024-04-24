"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import about, index, predict_page, video_upload_api, result, get_data
from project_settings import settings
from django.conf.urls.static import static

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', index, name='home'),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('video_upload_api/', video_upload_api, name='video_upload_api'),
    path('get_data/', get_data, name="get_data"),
    path("result/", result, name="result"),
    # path('cuda_full/',cuda_full,name='cuda_full'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static("/uploaded_images/", document_root=settings.IMAGE_ROOT)
