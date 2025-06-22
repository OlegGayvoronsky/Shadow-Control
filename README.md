# AniMotion <img src="Source/Images/Am1.ico" width="50"/>
**Генератор анимаций на основе детекции ключевых точек тела**

## 📖 Описание  
Это десктопное приложение на Python с GUI-интерфейсом, разработанным с использованием библиотеки PyQt5, позволяет анимировать риггованные 3D модели на основе видео.<br> 
Распознавание ключевых точек осуществляется с помощью модели компьютерного зрения [**MeTRAbs**](https://github.com/isarandi/metrabs/tree/master?tab=readme-ov-file#metrabs-absolute-3d-human-pose-estimator).

## 🔧 Основные возможности:

- Загрузка видеофайлов или JSON с ключевыми точками  
- Экспорт 3D-модели  
- Управление списком загруженных 3D-моделей  
- Создание анимации  
- Управление списком результатов  
- Выбор между несколькими предобученными MeTRAbs моделями  

## <img width="32" height="32" src="https://img.icons8.com/nolan/64/screenshot.png" alt="screenshot"/> Скриншоты

<h3>✨ Окно создания анимации</h3>
<img src="Source/Images/screenshot_main.png" width="400"/>

<h3>🗂 Окно с результатами</h3>
<img src="Source/Images/screenshot_results.png" width="400"/>

<h3>➕ Окно загрузки модели</h3>
<img src="Source/Images/screenshot_model_upload.png" width="400"/>

## Установка

Для установки программы необходимо скачать [**архив**](https://drive.google.com/file/d/1ebg3AaQ60Y_QX7YL43DFBBFPIhbxjLrh/view?usp=sharing) и распаковать его на устройстве.
Для работы программы на устройстве должны быть установлены: 
- Медиа плеер <img src="Source/Images/vlc.png" width="20" height="20"/>[**VLC**](https://get.videolan.org/vlc/3.0.21/win64/vlc-3.0.21-win64.exe), 64-разрядная версия
- <img src="Source/Images/python.gif" width="20" height="20"/>[**Python**](https://www.python.org/downloads/) 3.10
- <img src="Source/Images/nvidia.png" width="20" height="20"/>[**NVIDIA Driver**](https://www.nvidia.com/en-us/drivers/) >= 460.32

## Развёртывание проекта
**Клонируйте репозиторий:** <br>
<pre><code>git clone https://github.com/IgorGayvoronskiy/Animation-Generator.git <br>
cd Animation-Generator </code> </pre>
**Запустите .bat файл, предварительно поменяв в нём путь к anaconda3/miniconda3 на свой:** <br>
<pre><code>.\setup.bat </code> </pre>

## Дополнительно
Если на устройстве отсутствует GPU, можно запустить в <img src="Source/Images/kaggle.png" width="20" height="20"/>[**Kaggle**](https://www.kaggle.com/)/<img src="Source/Images/gc.png" width="30" height="20"/>[**Google Colab**](https://colab.google/) Jupyter ноутбук cloud_computing.ipynb для вычисления JSON-файлов ключевых точек.<br>
Для добавления meTRAbs моделей нужно скачать [**архив**](https://omnomnom.vision.rwth-aachen.de/data/metrabs/)  и распаковать в AniMotion/_internal/metrabs_models.

## [**Скринкаст**](https://drive.google.com/file/d/1FlGYG4aXq9M4ku4UmfBjENmTkvSR5_q4/view?usp=sharing) ##

*Icons by [**icons8**](https://icons8.ru/)*
