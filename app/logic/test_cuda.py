# import torch
#
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
from pathlib import Path

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
from time import time as tme

path = Path(__file__).resolve().parent.parent / "run_model" / "vosk-model-small-ru-0.22"
model = Model(str(path))
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("Слушаю...")
    while True:
        data = q.get()
        start = tme()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            print("Распознано:", text)
        print(tme() - start)
        start = tme()
