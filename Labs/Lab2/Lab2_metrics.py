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