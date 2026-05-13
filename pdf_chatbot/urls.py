from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static

from chatbot import views


urlpatterns = [

    path('admin/', admin.site.urls),

    path('', include('chatbot.urls')),

    path("upload-pdf/", views.upload_pdf),

]

if settings.DEBUG:

    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )