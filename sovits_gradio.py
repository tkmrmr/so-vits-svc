import io
import logging
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

import gradio as gr
import glob
import os
import json

# モデル
model_names = []
model_folder_path = "logs/44k/"
model_paths = glob.glob(model_folder_path+"G_*.pth")
for model_path in model_paths:
    model_names.append(os.path.basename(model_path))

# Config
config_filename = "config.json"
config_path = "configs/" + config_filename

# 話者名
with open("configs/config.json", "r") as f:
    data = json.load(f)
    spk = data["spk"]
    spk_list = list(spk.keys())

# オプションパラメータ(ここではデフォルト)
cluster_model_path = "logs/44k/kmeans_10000.pt"
cluster_infer_ratio = 0
linear_gradient = 0

# 基本デフォルト
clip = 0
trans = [0]
slice_db = -40
device = None
noice_scale = 0.4
pad_seconds = 0.5
linear_gradient_retain = 0.75

def comparison(audio, model, speker, auto_pred):
    svc_model = Svc(model_folder_path+model, config_path, device, cluster_model_path)
    infer_tool.mkdir(["raw", "results"])
    raw_audio_path = audio
    spk = speker
    auto_predict_f0 = auto_pred
    lg = linear_gradient
    lgr = linear_gradient_retain
    tran = trans[0]

    if "." not in raw_audio_path:
        raw_audio_path += ".wav"
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix('.wav')
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
    per_size = int(clip*audio_sr)
    lg_size = int(lg*audio_sr)
    lg_size_r = int(lg_size*lgr)
    lg_size_c_l = (lg_size-lg_size_r)//2
    lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
    lg = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0
    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        
        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
            audio.extend(list(infer_tool.pad_array(_audio, length)))
            continue
        if per_size != 0:
            datas = infer_tool.split_list_by_n(data, per_size,lg_size)
        else:
            datas = [data]
        for k,dat in enumerate(datas):
            per_length = int(np.ceil(len(dat) / audio_sr * svc_model.target_sample)) if clip!=0 else length
            if clip!=0: print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
            # padd
            pad_len = int(audio_sr * pad_seconds)
            dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, dat, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr = svc_model.infer(spk, tran, raw_path,
                                                cluster_infer_ratio=cluster_infer_ratio,
                                                auto_predict_f0=auto_predict_f0,
                                                noice_scale=noice_scale
                                                )
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * pad_seconds)
            _audio = _audio[pad_len:-pad_len]
            _audio = infer_tool.pad_array(_audio, per_length)
            if lg_size!=0 and k!=0:
                lg1 = audio[-(lg_size_r+lg_size_c_r):-lg_size_c_r] if lgr != 1 else audio[-lg_size:]
                lg2 = _audio[lg_size_c_l:lg_size_c_l+lg_size_r]  if lgr != 1 else _audio[0:lg_size]
                lg_pre = lg1*(1-lg)+lg2*lg
                audio = audio[0:-(lg_size_r+lg_size_c_r)] if lgr != 1 else audio[0:-lg_size]
                audio.extend(lg_pre)
                _audio = _audio[lg_size_c_l+lg_size_r:] if lgr != 1 else _audio[lg_size:]
            audio.extend(list(_audio))
    output = (svc_model.target_sample, np.array(audio))
    return output

app = gr.Interface(
    fn=comparison,
    inputs=[gr.Audio(type="filepath"), gr.Dropdown(choices=model_names), gr.Dropdown(choices=spk_list), "checkbox"],
    outputs=["audio"],
    allow_flagging='never',
    input_conversion=None, 
    output_conversion=None,
)

app.launch(share=True)