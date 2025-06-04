import os
import platform
import socket
import time
from pathlib import Path

import psutil
import requests
import zipfile
import subprocess
import yaml

from tqdm import tqdm

def find_mediamtx_config(start_dir):
    for root, _, files in os.walk(start_dir):
        for name in files:
            if name in ("mediamtx.yml", "mediamtx.config.yml"):
                return os.path.join(root, name)
    return None

def update_paths_section(yml_path):
    with open(yml_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print("Ошибка разбора YAML:", e)
            return

    if 'paths' not in config or not isinstance(config['paths'], dict):
        config['paths'] = {}

    config['paths']['mystream'] = {
        'source': 'publisher',
        'sourceOnDemand': False,
        'sourceOnDemandStartTimeout': '10s',
        'sourceOnDemandCloseAfter': '10s',
        'record': False
    }

    with open(yml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Файл успешно обновлён: {yml_path}")

def download_mediamtx(dest_folder='mediamtx'):
    system = platform.system().lower()
    arch = platform.machine().lower()

    if system != 'windows' or '64' not in arch:
        raise RuntimeError('Поддерживается только Windows 64-bit')

    version = 'v1.12.3'
    url = f'https://github.com/bluenviron/mediamtx/releases/download/{version}/mediamtx_{version}_windows_amd64.zip'

    if not Path(dest_folder).exists():
        os.makedirs(dest_folder, exist_ok=True)

        archive_path = os.path.join(dest_folder, 'mediamtx.zip')

        print('Скачивание MediaMTX...')
        r = requests.get(url)
        r.raise_for_status()

        with open(archive_path, 'wb') as f:
            f.write(r.content)

        print('Распаковка...')
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)

        os.remove(archive_path)

    exe_path = os.path.join(dest_folder, 'mediamtx.exe')
    if not os.path.exists(exe_path):

        raise FileNotFoundError('Файл mediamtx.exe не найден после распаковки.')
    yml_path = find_mediamtx_config(dest_folder)
    if yml_path:
        update_paths_section(yml_path)
    else:
        print("Файл конфигурации mediamtx не найден.")

    return exe_path

def install_ffmpeg(destination_dir='ffmpeg'):
    ffmpeg_url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
    zip_path = os.path.join(destination_dir, 'ffmpeg.zip')

    if not Path(destination_dir).exists():
        os.makedirs(destination_dir, exist_ok=True)

        # Скачать архив
        print("Скачивание FFmpeg...")
        with requests.get(ffmpeg_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            with open(zip_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc="Загрузка"
            ) as bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    bar.update(len(chunk))

        print("Скачивание завершено.")

        # Распаковать архив
        print("Распаковка...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)

    # Найти bin/ffmpeg.exe
    print("Поиск ffmpeg.exe...")
    for root, dirs, files in os.walk(destination_dir):
        if 'ffmpeg.exe' in files:
            ffmpeg_bin_path = os.path.join(root)
            break
    else:
        raise FileNotFoundError("ffmpeg.exe не найден после распаковки!")

    # Добавить в PATH (временно для текущего скрипта)
    os.environ['PATH'] = ffmpeg_bin_path + os.pathsep + os.environ['PATH']
    print(f"FFmpeg установлен: {ffmpeg_bin_path}")

    # Проверка
    try:
        subprocess.run(['ffmpeg', '-version'], check=True)
        print("FFmpeg готов к использованию.")
    except Exception as e:
        print("Ошибка при запуске FFmpeg:", e)

    return ffmpeg_bin_path

def is_mediamtx_running():
    result = subprocess.run("tasklist", capture_output=True, text=True)
    return "mediamtx.exe" in result.stdout

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0

def read_last_log_lines(log_path, num_lines=20):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-num_lines:]
    except FileNotFoundError:
        return ["Log file not found."]

def check_ffprobe_stream(rtsp_url):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', rtsp_url],
            capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout if result.stdout else result.stderr
    except FileNotFoundError:
        return False, "ffprobe not found. Please install FFmpeg and add it to PATH."

def start_mediamtx(exe_path, cwd):
    log_file_path = os.path.join(cwd, "mediamtx_process_output.log")

    print(cwd)
    with open(log_file_path, "w") as log_file:
        proc = subprocess.Popen(
            [exe_path, "mediamtx.yml"],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

    # Даём немного времени на старт
    time.sleep(3)
    # Проверяем, работает ли процесс MediaMTX
    mediamtx_running = False
    for p in psutil.process_iter(['pid', 'name']):
        if 'mediamtx' in p.info['name'].lower():
            mediamtx_running = True
            print(f"✅ MediaMTX запущен (PID: {p.info['pid']})")
            break

    if not mediamtx_running:
        print("❌ MediaMTX не найден среди процессов.")

    print(f'MediaMTX запущен с PID {proc.pid}')
    print("🔍 Проверка запущен ли MediaMTX...")
    if is_mediamtx_running():
        print("✅ MediaMTX работает.")
    else:
        print("❌ MediaMTX не найден среди процессов.")

    print("\n📡 Проверка, слушается ли порт 8554...")
    if is_port_open(8554):
        print("✅ Порт 8554 открыт.")
    else:
        print("❌ Порт 8554 не слушается.")

    print("\n📄 Последние строки из mediamtx.log:")
    log_lines = read_last_log_lines("mediamtx/mediamtx.log")
    print("".join(log_lines))

    return proc

def stop_mediamtx(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
        print('MediaMTX остановлен.')
    except subprocess.TimeoutExpired:
        proc.kill()
        print('MediaMTX принудительно завершён.')
