import torch
import os

SOURCE_PATH = 'source_voice_3.wav' #монолог Быкова из Интернов
TARGETS_FOLDER = 'targets'         #отрезки спокойной речи Охлобыстина И.И.
OUTPUTS_FOLDER = 'results'
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(TARGETS_FOLDER, exist_ok=True)

seconds_list = [10, 20, 30, 40, 50, 60]
target_files = [f'target_{sec}s.wav' for sec in seconds_list]

print('Загружаю kNN-VC через torch.hub...')
knn_vc = torch.hub.load(
    'bshall/knn-vc',
    'knn_vc',
    device='cpu',
    prematched=True,
    trust_repo=True,
    pretrained=True
)

for sec, tfile in zip(seconds_list, target_files):
    target_path = os.path.join(TARGETS_FOLDER, tfile)
    output_path = os.path.join(OUTPUTS_FOLDER, f'converted_{sec}s.wav')
    print(f'\n====== Длина таргета: {sec} сек. ======')

    #1 Извлечение признаков
    print('Извлекаю признаки source...')
    query_seq = knn_vc.get_features(SOURCE_PATH)
    print('Извлекаю признаки target...')
    matching_set = knn_vc.get_matching_set([target_path])
    print(f'matching_set.shape={matching_set.shape}, query_seq.shape={query_seq.shape}')
    if isinstance(matching_set, (list, tuple)):
        matching_set = torch.cat([torch.as_tensor(x) for x in matching_set], dim=0)
    elif len(matching_set.shape) == 3:
        matching_set = matching_set.reshape(-1, matching_set.shape[-1])

    print(f'[DEBUG] matching_set.shape={matching_set.shape}, query_seq.shape={query_seq.shape}')

    # 2 knn + vocoder
    print('Выполняю конвертацию (k=4)...')
    out_wav = knn_vc.match(query_seq, matching_set, topk=4)

    # 3 Сохранение результата
    print(f'Сохраняю результат: {output_path}')
    import torchaudio
    torchaudio.save(output_path, out_wav.unsqueeze(0), 16000)

print('Эксперимент завершён.')