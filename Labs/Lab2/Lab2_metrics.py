import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio, PerceptualEvaluationSpeechQuality

def mixer(original, noise, snr_db):
    L = len(original)
    if len(noise) < L:
        repeats = int(np.ceil(L / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:L]
    rms_signal = np.sqrt((original**2).mean())
    rms_noise = np.sqrt((noise**2).mean())
    snr_linear = 10 ** (snr_db / 20)
    desired_rms_noise = rms_signal / snr_linear
    noise = noise * (desired_rms_noise / (rms_noise + 1e-8))
    mix = original + noise
    return mix.astype('float32')

def load_audio(path, sr=16000):
    audio, fs = torchaudio.load(path)
    if fs != sr:
        audio = torchaudio.functional.resample(audio, fs, sr)
    return audio[0].numpy(), sr

def compute_metrics(clean, mixed, sr=16000):
    clean_torch = torch.tensor(clean).unsqueeze(0)
    mixed_torch = torch.tensor(mixed).unsqueeze(0)
    sdr = SignalDistortionRatio()(mixed_torch, clean_torch).item()
    sisdr = ScaleInvariantSignalDistortionRatio()(mixed_torch, clean_torch).item()
    pesq_metric = PerceptualEvaluationSpeechQuality(sr, mode='wb')
    pesq = pesq_metric(mixed_torch, clean_torch).item()
    return sdr, sisdr, pesq

# Заглушки для NISQA и DNSMOS
def run_nisqa(audio_file):
    return [None] * 6
def get_dnsmos(audio_file):
    return [None] * 3

# === Параметры ===
clean_wav = "../../sounds/voice_audiobook.mp3"
noise_wav = "../../sounds/noise_for_voice.wav"
outdir = 'mixed_audio'
os.makedirs(outdir, exist_ok=True)
SNR_list = [-5, 0, 5, 10]

# === Загрузка аудио ===
assert os.path.exists(clean_wav), f"Файл не найден: {clean_wav}"
assert os.path.exists(noise_wav), f"Файл не найден: {noise_wav}"

clean, sr = load_audio(clean_wav)
print(f'sr1 = {sr}')
assert sr!=0
noise, sr = load_audio(noise_wav, sr)
print(f'sr2 = {sr}')

rows = []
for snr in SNR_list:
    try:
        mix = mixer(clean, noise, snr)
        out_path = os.path.join(outdir, f"mix_snr_{snr}db.wav")
        torchaudio.save(out_path, torch.tensor(mix).unsqueeze(0), sr)
        sdr, sisdr, pesq = compute_metrics(clean, mix, sr)
        nisqa_metrics = [None] * 6
        dnsmos_metrics = [None] * 3
        # MOS вручную/краудсорс — поставить 0/None если не оцениваете
        print(f"Прослушайте файл {out_path} (SNR={snr}dB) и поставьте MOS (от 1 до 5):")
        mos_manual = None

        row = {
            "файл": out_path,
            "SNR": snr,
            "SDR": sdr,
            "SI-SDR": sisdr,
            "PESQ": pesq,
            "NISQA_mos_pred": nisqa_metrics[0],
            "NISQA_mos_pred_full": nisqa_metrics[1],
            "NISQA_mos_pred_d": nisqa_metrics[2],
            "NISQA_mos_pred_mos_sig": nisqa_metrics[3],
            "NISQA_mos_pred_mos_bak": nisqa_metrics[4],
            "NISQA_mos_pred_mos_ovr": nisqa_metrics[5],
            "DNSMOS_OVRL": dnsmos_metrics[0],
            "DNSMOS_SIG": dnsmos_metrics[1],
            "DNSMOS_BAK": dnsmos_metrics[2],
            "MOS": mos_manual
        }
        rows.append(row)
        print(f"Row for SNR={snr} added.")
    except Exception as e:
        print(f"Ошибка на SNR={snr}: {e}")

df = pd.DataFrame(rows)
df.to_csv('results.csv', index=False)
print("\nРезультаты эксперимента:")
print(df.to_string(index=False))










'''
import numpy as np
import torchaudio
import os
import torch
#import torchmetrics
#print(torchmetrics.__version__)
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio#, PerceptualEvaluationSpeechQuality
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


def mixer(original, noise, snr_db):
    """
    Смешивает оригинальный сигнал с шумом согласно заданному SNR (в дБ)
    original: np.ndarray (длина T)
    noise: np.ndarray (длина >= T)
    snr_db: float, SNR в децибелах
    Возвращает смешанный сигнал длины T.
    """
    # Обрезать шум до длины original или повторить noise
    if len(noise) < len(original):
        original = original[:len(noise)]
    # Повторяем original, если он короче
    repeats = int(np.ceil(len(noise) / len(original)))
    original = np.tile(original, repeats)
    
    
    # RMS (Root Mean Square) амплитуда
    rms_signal = np.sqrt(np.mean(original**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    
    # Считаем по формуле, вывод: K = rms_signal / (10**(snr_db/20) * rms_noise)
    desired_rms_noise = rms_signal / (10**(snr_db/20))
    noise = noise * (desired_rms_noise / (rms_noise + 1e-12))
    mix = original + noise
    return mix.astype(np.float32)

voice_path = '../../sounds/voice_audiobook.mp3'  # Путь к чистому голосу
noise_path = '../../sounds/noise_for_voice.wav'  # Путь к шуму (шум из DEMAND, UrbanSound8K и т.д.)

voice, sr = torchaudio.load(voice_path)
noise, sr_noise = torchaudio.load(noise_path)
voice = voice[0].numpy()
noise = noise[0].numpy()
assert sr == sr_noise

snr_list = [-5, 0, 5, 10]
mixed_audios = {}
for snr in snr_list:
    mixed = mixer(voice, noise, snr)
    torchaudio.save(f"mix_{snr}dB.wav", torch.from_numpy(mixed).unsqueeze(0), sr)
    mixed_audios[snr] = mixed

sdr_metric = SignalDistortionRatio()
sisdr_metric = ScaleInvariantSignalDistortionRatio()
# Для PESQ нужна поддержка только Wideband (16кГц) и Narrowband (8кГц)
# Скачайте голос и шум подходящей частоты дискретизации!

pesq_metric = PerceptualEvaluationSpeechQuality(16000, mode='wb')

results = []
for snr in snr_list:
    mixture = mixed_audios[snr]
    # Обрезаем на всякий случай длину оригинала
    mixture = mixture[:len(voice)]
    mixture_tensor = torch.from_numpy(mixture)
    voice_tensor = torch.from_numpy(voice)

    min_len = min(mixture_tensor.shape[-1], voice_tensor.shape[-1])
    mixture_tensor = mixture_tensor[:min_len]
    voice_tensor = voice_tensor[:min_len]

    sdr = sdr_metric(mixture_tensor, voice_tensor).item()
    sisdr = sisdr_metric(mixture_tensor, voice_tensor).item()
    pesq = pesq_metric(mixture_tensor, voice_tensor).item()
    results.append({'snr': snr, 'SDR': sdr, 'SI-SDR': sisdr, 'PESQ': pesq})


# Во-первых, сохраните файлы для оценки: mix_-5dB.wav и т.д.
# Затем используйте NISQA CLI или функцию Python
# Пример используя os.system (CLI):

for snr in snr_list:
    wave = f"mix_{snr}dB.wav"
    cmd = f"python nisqa/run_nisqa.py --mode predict_file --pretrained_model nisqa/NISQA_MODEL --input_file {wave} --output_csv nisqa_out_{snr}dB.csv"
    os.system(cmd)

'''







'''
import torch
print(torch.__version__)  # Например, 2.1.0

import torch
import torchaudio
import torchmetrics
import numpy as np
import pesq
#from pypesq import pesq
from nisqa.NISQA_model import nisqaModel
import sys
#sys.path.append("/path/to/DNS-Challenge/dnsmos_pkg")
#from dnsmos import DNSMOS
#from dnsmos import DNSMOS




import os
import numpy as np
import soundfile as sf
import onnxruntime as ort
from scipy import signal
from urllib.request import urlretrieve

def download_dnsmos_models():
    """Скачивание предобученных моделей DNSMOS с GitHub"""
    base_url = "https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS"
    models = {
        "sig_bak_ovr.onnx": "sig_bak_ovr.onnx",
        "p835_0_1.onnx": "p835_0_1.onnx"
    }
    
    os.makedirs("dnsmos_models", exist_ok=True)
    
    for model_name, save_as in models.items():
        url = base_url + model_name
        save_path = os.path.join("dnsmos_models", save_as)
        if not os.path.exists(save_path):
            print(f"Скачивание {model_name}...")
            urlretrieve(url, save_path)
            print(f"Модель сохранена в {save_path}")

def resample_audio(audio, orig_sr, target_sr=16000):
    """Ресемплирование аудио до целевой частоты дискретизации"""
    if orig_sr == target_sr:
        return audio
    
    duration = len(audio) / orig_sr
    resampled_audio = signal.resample(audio, int(duration * target_sr))
    return resampled_audio

def compute_dnsmos(audio_path):
    """Вычисление метрик DNSMOS для аудиофайла"""
    # Загрузка моделей
    download_dnsmos_models()
    
    # Загрузка аудиофайла
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Используем только первый канал для моно
    
    # Ресемплирование до 16 кГц, если необходимо
    if sr != 16000:
        audio = resample_audio(audio, sr)
    
    # Подготовка аудио для модели (длина 9 секунд)
    required_len = 9 * 16000
    if len(audio) < required_len:
        # Если аудио короче 9 секунд, дополняем нулями
        audio = np.pad(audio, (0, required_len - len(audio)))
    else:
        # Если длиннее, берем первые 9 секунд
        audio = audio[:required_len]
    
    # Загрузка моделей ONNX
    ort_session_sig_bak_ovr = ort.InferenceSession("dnsmos_models/sig_bak_ovr.onnx")
    ort_session_p835 = ort.InferenceSession("dnsmos_models/p835_0_1.onnx")
    
    # Вычисление признаков
    input_features = np.array(audio, dtype=np.float32)[np.newaxis, :]
    
    # Вычисление метрик
    sig_bak_ovr = ort_session_sig_bak_ovr.run(None, {'input': input_features})[0][0]
    p835 = ort_session_p835.run(None, {'input': input_features})[0][0]
    
    # Возвращаем результаты
    return {
        "SIG": sig_bak_ovr[0],       # Signal distortion
        "BAK": sig_bak_ovr[1],       # Background noise
        "OVR": sig_bak_ovr[2],       # Overall quality
        "P808_MOS": p835[0]         # P.808 Mean Opinion Score
    }

if __name__ == "__main__":
    # Пример использования
    audio_file = "corrected_white_noise.wav"  # Замените на путь к вашему аудиофайлу
    if not os.path.exists(audio_file):
        print(f"Файл {audio_file} не найден!")
    else:
        scores = compute_dnsmos(audio_file)
        print("Результаты оценки DNSMOS:")
        print(f"SIG (Signal distortion): {scores['SIG']:.2f}")
        print(f"BAK (Background noise): {scores['BAK']:.2f}")
        print(f"OVR (Overall quality): {scores['OVR']:.2f}")
        print(f"P.808 MOS: {scores['P808_MOS']:.2f}")
        
        



def mixer(original, noise, snr_db):
    """
    Смешивает оригинальный сигнал с шумом по заданному SNR (в dB)
    
    Параметры:
        original (torch.Tensor): оригинальный аудиосигнал
        noise (torch.Tensor): шумовой сигнал
        snr_db (float): целевое SNR в dB
        
    Возвращает:
        torch.Tensor: смешанный сигнал
    """
    # Нормализуем сигналы
    original = original / torch.max(torch.abs(original))
    noise = noise / torch.max(torch.abs(noise))
    
    # Обрезаем более длинный сигнал
    min_len = min(original.shape[-1], noise.shape[-1])
    original = original[..., :min_len]
    noise = noise[..., :min_len]
    
    # Вычисляем мощности сигналов
    power_original = torch.mean(original ** 2)
    power_noise = torch.mean(noise ** 2)
    
    # Вычисляем коэффициент масштабирования шума для достижения нужного SNR
    snr_linear = 10 ** (snr_db / 10)
    scale_factor = torch.sqrt(power_original / (power_noise * snr_linear))
    
    # Смешиваем сигналы
    mixed = original + scale_factor * noise
    mixed = mixed / torch.max(torch.abs(mixed))  # Нормализуем результат
    
    return mixed

def compute_metrics(clean, noisy, sr=16000):
    """
    Вычисляет все метрики качества аудио
    
    Параметры:
        clean (torch.Tensor): чистый сигнал
        noisy (torch.Tensor): зашумленный сигнал
        sr (int): частота дискретизации
        
    Возвращает:
        dict: словарь с метриками
    """
    # Инициализация метрик
    sdr = torchmetrics.SignalDistortionRatio()
    si_sdr = torchmetrics.ScaleInvariantSignalDistortionRatio()
    pesq_metric = torchmetrics.PESQ(fs=sr, mode='wb')
    
    # Вычисление метрик
    metrics = {
        'SDR': sdr(noisy, clean).item(),
        'SI-SDR': si_sdr(noisy, clean).item(),
        'PESQ': pesq_metric(noisy, clean).item()
    }
    
    # Сохраняем файлы для NISQA и DNSMOS
    torchaudio.save('temp_clean.wav', clean.unsqueeze(0), sr)
    torchaudio.save('temp_noisy.wav', noisy.unsqueeze(0), sr)
    
    # Вычисление NISQA
    args = {
        'mode': 'predict_file',
        'pretrained_model': 'nisqa_tts.tar',
        'deg': 'temp_noisy.wav',
        'output_dir': '.',
        'ms_channel': 1
    }
    nisqa = nisqaModel(args)
    nisqa_results = nisqa.predict()
    metrics['NISQA'] = nisqa_results
    
    # Вычисление DNSMOS
    dnsmos = DNSMOS()
    dnsmos_results = dnsmos('temp_noisy.wav')
    metrics['DNSMOS'] = dnsmos_results
    
    return metrics

def main():
    # Загрузка аудиофайлов
    clean, sr_clean = torchaudio.load('clean.wav')
    noise, sr_noise = torchaudio.load('noise.wav')
    
    # Ресемплинг при необходимости
    if sr_clean != sr_noise:
        resample = torchaudio.transforms.Resample(sr_noise, sr_clean)
        noise = resample(noise)
    
    # Список SNR для тестирования
    snr_levels = [-5, 0, 5, 10]
    
    results = []
    
    for snr in snr_levels:
        # Смешивание сигналов
        mixed = mixer(clean, noise, snr)
        
        # Вычисление метрик
        metrics = compute_metrics(clean, mixed, sr_clean)
        
        # Субъективная оценка MOS (примерная, нужно прослушать)
        # Это значение нужно установить после прослушивания
        mos = estimate_mos(mixed)  # Функция, которую нужно реализовать
        
        results.append({
            'SNR': snr,
            'SDR': metrics['SDR'],
            'SI-SDR': metrics['SI-SDR'],
            'PESQ': metrics['PESQ'],
            'NISQA': metrics['NISQA'],
            'DNSMOS': metrics['DNSMOS'],
            'MOS': mos
        })
    
    # Вывод результатов в виде таблицы
    print_table(results)

def estimate_mos(audio):
    """
    Функция для субъективной оценки MOS (1-5)
    Нужно прослушать аудио и поставить оценку
    """
    # В реальном коде нужно прослушать аудио и поставить оценку
    # Здесь возвращаем примерные значения
    if audio_sounds_good(audio):
        return 4.5
    elif audio_sounds_acceptable(audio):
        return 3.0
    else:
        return 1.5

if __name__ == '__main__':
    main()
'''