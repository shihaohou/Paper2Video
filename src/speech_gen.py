import re
import os
import cv2
import json
import torch
import subprocess

if not getattr(torch.load, "_p2v_patched", False):
    _p2v_orig_torch_load = torch.load
    def _p2v_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False  ## force override; some callers (lightning_fabric) pass True explicitly
        return _p2v_orig_torch_load(*args, **kwargs)
    _p2v_torch_load._p2v_patched = True
    torch.load = _p2v_torch_load

import whisperx
from os import path


def transcribe_with_whisperx(audio_path, lang="en", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")
    model = whisperx.load_model("large-v2", device=device, compute_type="float16" if device == "cuda" else "int8")
    result = model.transcribe(audio_path, language=lang)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    segments = result_aligned["segments"]
    text = " ".join(seg["text"].strip() for seg in segments)
    return text

def inference_f5(text_prompt, save_path, ref_audio, ref_text):
    from f5_tts.api import F5TTS
    f5tts = F5TTS()
    wav, sr, spec = f5tts.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=text_prompt, file_wave=save_path, seed=None,)

def speedup_audio(path, speed):
    """In-place ffmpeg atempo speed change without pitch shift.

    atempo supports 0.5-2.0; we expect speed in [0.8, 1.5] for natural talk.
    """
    if abs(speed - 1.0) < 0.01:
        return
    tmp = path + ".tmp.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-filter:a", "atempo={}".format(speed),
         "-loglevel", "error", tmp],
        check=True,
    )
    os.replace(tmp, path)


def inference_moss(text_prompt, save_path, ref_audio, moss_python, device="cuda"):
    """Voice clone via MOSS-TTS (8B Delay) as a subprocess in its own Python env.

    Invokes `tts_moss_wrapper.py` (sibling of this file) using the supplied
    Python interpreter from the MOSS env. The wrapper handles model loading
    via transformers' AutoModel/AutoProcessor with trust_remote_code=True.
    """
    wrapper = path.join(path.dirname(path.abspath(__file__)), "tts_moss_wrapper.py")
    out_dir = path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cmd = [
        moss_python, wrapper,
        "--ref-audio", ref_audio,
        "--text", text_prompt,
        "--output-audio", save_path,
        "--device", device,
    ]
    subprocess.run(cmd, check=True)

def parse_script(script_text):
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line: 
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        result.append(page_data)
    return result

def tts_per_slide(model_type, script_path, speech_save_dir, ref_audio, ref_text=None,
                  moss_python=None, speed=1.0):
    with open(script_path, 'r') as f: script_with_cursor = ''.join(f.readlines())
    parsed_speech = parse_script(script_with_cursor)

    os.makedirs(speech_save_dir, exist_ok=True)

    for slide_idx in range(len(parsed_speech)):
        speech_with_cursor = parsed_speech[slide_idx]
        subtitle = ""
        for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
            if len(subtitle) == 0: subtitle = prompt
            else: subtitle = subtitle + "\n\n\n" + prompt
        speech_result_path = path.join(speech_save_dir, "{}.wav".format(str(slide_idx)))

        if model_type == "f5":
            if ref_text is None: ref_text = transcribe_with_whisperx(ref_audio)
            inference_f5(subtitle, speech_result_path, ref_audio, ref_text)
        elif model_type == "moss":
            if not moss_python:
                raise ValueError("moss model requires --moss_env")
            inference_moss(subtitle, speech_result_path, ref_audio, moss_python)
        else:
            raise ValueError(f"unknown tts model_type: {model_type}")

        speedup_audio(speech_result_path, speed)