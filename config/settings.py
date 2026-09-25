"""Django settings for the X-ray Lens analyzer."""
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or ("django-insecure-local-dev-only-key" if DEBUG else None)
if not SECRET_KEY: raise RuntimeError("Set DJANGO_SECRET_KEY when DJANGO_DEBUG is off.")
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "testserver"]
CSRF_TRUSTED_ORIGINS = []
if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
    ALLOWED_HOSTS.append(os.environ["RENDER_EXTERNAL_HOSTNAME"]); CSRF_TRUSTED_ORIGINS.append(f"https://{os.environ['RENDER_EXTERNAL_HOSTNAME']}")
for host in filter(None, os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")):
    ALLOWED_HOSTS.append(host.strip()); CSRF_TRUSTED_ORIGINS.append(f"https://{host.strip()}")
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https"); SESSION_COOKIE_SECURE = True; CSRF_COOKIE_SECURE = True; SECURE_CONTENT_TYPE_NOSNIFF = True
INSTALLED_APPS = ["django.contrib.admin","django.contrib.auth","django.contrib.contenttypes","django.contrib.sessions","django.contrib.messages","django.contrib.staticfiles","analyzer"]
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware","whitenoise.middleware.WhiteNoiseMiddleware","django.contrib.sessions.middleware.SessionMiddleware","django.middleware.common.CommonMiddleware","django.middleware.csrf.CsrfViewMiddleware","django.contrib.auth.middleware.AuthenticationMiddleware","django.contrib.messages.middleware.MessageMiddleware","django.middleware.clickjacking.XFrameOptionsMiddleware"]
ROOT_URLCONF = "config.urls"
TEMPLATES = [{"BACKEND":"django.template.backends.django.DjangoTemplates","DIRS":[],"APP_DIRS":True,"OPTIONS":{"context_processors":["django.template.context_processors.request","django.contrib.auth.context_processors.auth","django.contrib.messages.context_processors.messages"]}}]
WSGI_APPLICATION = "config.wsgi.application"
DATABASES = {"default":{"ENGINE":"django.db.backends.sqlite3","NAME":BASE_DIR / "db.sqlite3"}}
AUTH_PASSWORD_VALIDATORS = [{"NAME":"django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},{"NAME":"django.contrib.auth.password_validation.MinimumLengthValidator"},{"NAME":"django.contrib.auth.password_validation.CommonPasswordValidator"},{"NAME":"django.contrib.auth.password_validation.NumericPasswordValidator"}]
LANGUAGE_CODE="en-us"; TIME_ZONE="Asia/Kolkata"; USE_I18N=True; USE_TZ=True
STATIC_URL="static/"; STATIC_ROOT=BASE_DIR / "staticfiles"
STORAGES={"default":{"BACKEND":"django.core.files.storage.FileSystemStorage"},"staticfiles":{"BACKEND":"django.contrib.staticfiles.storage.StaticFilesStorage" if DEBUG else "whitenoise.storage.CompressedManifestStaticFilesStorage"}}
DEFAULT_AUTO_FIELD="django.db.models.BigAutoField"
MEDIA_URL="/media/"; MEDIA_ROOT=BASE_DIR / "media"
ML_MODELS_DIR=BASE_DIR / "ml" / "trained_models"
SUPPORTED_BODY_PARTS=["chest","bone"]
MAX_UPLOAD_SIZE_BYTES=10*1024*1024
FILE_UPLOAD_MAX_MEMORY_SIZE=MAX_UPLOAD_SIZE_BYTES+1024*1024
DATA_UPLOAD_MAX_MEMORY_SIZE=MAX_UPLOAD_SIZE_BYTES+1024*1024
LOGGING={"version":1,"disable_existing_loggers":False,"handlers":{"console":{"class":"logging.StreamHandler"}},"root":{"handlers":["console"],"level":"INFO"}}
