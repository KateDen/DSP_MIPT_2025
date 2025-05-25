import os
import torch
import soundfile as sf
import numpy as np
from pesq import pesq
from pystoi import stoi
from deepfilternet.models import DeepFilterNet2
from deepfilternet.util.audio import load_audio, save_audio

# Пути к файлам
clean_path = 'test_clean.wav'
noisy_path = 'test_noisy.wav'
df2_out_path = 'test_df2.wav'

# Загрузка аудио
clean, sr = sf.read(clean_path)
noisy, _ = sf.read(noisy_path)

# Запуск DeepFilterNet2
model = DeepFilterNet2()
model.load_state_dict(torch.load(model.model_path))  # если потребуется, иначе model.load() или аналогичный вызов
# Применить модель, зависит от API, обычно:
from deepfilternet.inference import denoise_file
denoise_file(noisy_path, df2_out_path, model=model)

# Загрузка денойзенного сигнала
df2, _ = sf.read(df2_out_path)

# Вычисление метрик
def SNR(clean, test):
    eps = 1e-8
    noise = clean - test
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + eps))

snr_noisy = SNR(clean, noisy)
snr_df2 = SNR(clean, df2)

pesq_noisy = pesq(sr, clean, noisy, 'wb')
pesq_df2 = pesq(sr, clean, df2, 'wb')

stoi_noisy = stoi(clean, noisy, sr, extended=False)
stoi_df2 = stoi(clean, df2, sr, extended=False)

print(f"SNR: Noisy={snr_noisy:.2f} | DF2={snr_df2:.2f}")
print(f"PESQ: Noisy={pesq_noisy:.2f} | DF2={pesq_df2:.2f}")
print(f"STOI: Noisy={stoi_noisy:.2f} | DF2={stoi_df2:.2f}")

import pandas as pd

results = pd.DataFrame({
    "Signal": ['Noisy', 'DF2'],
    "SNR": [snr_noisy, snr_df2],
    "PESQ": [pesq_noisy, pesq_df2],
    "STOI": [stoi_noisy, stoi_df2],
})
print(results)
