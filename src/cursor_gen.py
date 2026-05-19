import re
import os
import cv2
import pdb
import json
import torch
import string
import subprocess
from os import path
import multiprocessing as mp
from transformers import pipeline
from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code

if not getattr(torch.load, "_p2v_patched", False):
    _p2v_orig_torch_load = torch.load
    def _p2v_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False  ## force override; some callers (lightning_fabric) pass True explicitly
        return _p2v_orig_torch_load(*args, **kwargs)
    _p2v_torch_load._p2v_patched = True
    torch.load = _p2v_torch_load

import whisperx
from whisperx import load_audio
from whisperx.alignment import align


def draw_red_dots_on_image(image_path, point, radius: int = 5) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    red = (0, 0, 255)
    x, y = int(point[0]), int(point[1])
    cv2.circle(image, (x, y), radius, red, thickness=-1)
    cv2.imwrite("output.jpg", image)

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

def infer_cursor(instruction, image_path, device):
    pipe = pipeline("image-text-to-text", model="ByteDance-Seed/UI-TARS-1.5-7B")#, device=device)
    print("running on {}".format(device))
    prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You must to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction {}".format(instruction)
    messages = [{"role": "user", "content": [{"type": "image", "url": image_path}, {"type": "text", "text": prompt}]},]
    result = pipe(text=messages)[0]
    response = result['generated_text'][1]["content"]
    token = prompt + response
    print("kkk", pipe(text=messages))
    
    ori_image = cv2.imread(image_path)
    original_image_width, original_image_height = ori_image.shape[:2]
    parsed_dict = parse_action_to_structure_output(
        response,
        factor=1000,
        origin_resized_height=original_image_height,
        origin_resized_width=original_image_width,
        model_type="qwen25vl"
    )
    parsed_pyautogui_code = parsing_response_to_pyautogui_code(
        responses=parsed_dict,
        image_height=original_image_height,
        image_width=original_image_width)

    match = re.search(r'pyautogui\.click\(([\d.]+),\s*([\d.]+)', parsed_pyautogui_code)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
    else:
        print(instruction)
    return (x, y), token
    
def infer(args):
    slide_idx, sentence_idx, prompt, cursor_prompt, image_path, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch  
    torch.cuda.set_device(0) 
    
    point, token = infer_cursor(cursor_prompt, image_path, device="cuda:{}".format(str(gpu_id)))
    torch.cuda.empty_cache()
    result = {'slide': slide_idx, 'sentence': sentence_idx, 'speech_text': prompt, 'cursor_prompt': cursor_prompt, 'cursor': point, 'token': token}
    return result

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_audio_length(audio_path):
    command = ["ffmpeg", "-i", audio_path]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "Duration" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = map(float, duration_str.split(":"))
            return hours * 3600 + minutes * 60 + seconds
    return 0 

def timesteps(subtitles, aligned_result, audio_path):
    aligned_words_in_order = []
    for idx, segment in enumerate(aligned_result["segments"]):
        aligned_words_in_order.extend(segment["words"])
    aligned_words_num = len(aligned_words_in_order) - 1
    
    result = []
    current_idx = 0
    for idx, sentence in enumerate(subtitles):
        words_num = len(re.findall(r'\b\w+\b', sentence.lower()))
        start = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]
        
        current_idx += words_num
        end = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]

        duration = {"start": start, "end": end, "text": sentence}
        result.append(duration)
    
    result[0]["start"] = 0
    result[-1]["end"] = get_audio_length(audio_path)
    return result

def cursor_gen_per_sentence(script_path, slide_img_dir, slide_audio_dir, cursor_save_path, gpu_list):
    with open(script_path, 'r') as f:script_with_cursor = ''.join(f.readlines())
    parsed_speech = parse_script(script_with_cursor)
    cursor_token = ""
    
    slide_imgs = [name for name in os.listdir(slide_img_dir)]
    slide_imgs = sorted(slide_imgs, key=lambda x: int(re.search(r'\d+', x).group()))
    slide_imgs = [path.join(slide_img_dir, name) for name in slide_imgs]
    
    ## location
    num_gpus = len(gpu_list)
    process_idx = 0
    task_list = []
    for slide_idx in range(len(parsed_speech)):
        speech_with_cursor = parsed_speech[slide_idx]
        image_path = slide_imgs[slide_idx]
        for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
            gpu_id = gpu_list[process_idx % num_gpus]
            task_list.append((slide_idx, sentence_idx, prompt, cursor_prompt, image_path, gpu_id))
            process_idx += 1  
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_gpus) as pool: cursor_result = pool.map(infer, task_list)
    
    slide_w, slide_h = cv2.imread(slide_imgs[0]).shape[:2]
    for index in range(len(cursor_result)):
        if cursor_result[index]["cursor_prompt"] == "no":
            cursor_result[index]["cursor"] == (slide_w//2, slide_h//2)
        cursor_token += cursor_result[index]["token"]
          
    ## timesteps
    slide_sentence_timesteps = []
    slide_audio = os.listdir(slide_audio_dir)
    slide_audio = sorted(slide_audio, key=lambda x: int(re.search(r'\d+', x).group()))
    slide_audio = [path.join(slide_audio_dir, name) for name in slide_audio]
    model = whisperx.load_model("large-v3", device="cuda")
    align_model, metadata = whisperx.load_align_model(language_code="en", device="cuda")
    
    for idx, slide_audio_path in enumerate(slide_audio):
        ## get slide subtitle
        subtitle = []
        cursor = []
        for info in cursor_result: 
            if info["slide"] == idx: 
                subtitle.append(clean_text(info["speech_text"]))
                cursor.append(info["cursor"])
        ## word timesteps  
        audio = load_audio(slide_audio_path)
        result = model.transcribe(slide_audio_path, language="en")
        aligned = align(transcript=result["segments"], align_model_metadata=metadata, model=align_model, audio=audio, device="cuda")
        sentence_timesteps = timesteps(subtitle, aligned, slide_audio_path) # get_sentence_timesteps(subtitle, aligned, slide_audio_path)
        for idx in range(len(sentence_timesteps)): sentence_timesteps[idx]["cursor"] = cursor[idx]
        slide_sentence_timesteps.append(sentence_timesteps)
    # merage
    start_time_now = 0
    new_slide_sentence_timesteps = []
    for sentence_timesteps in slide_sentence_timesteps:
        duration = 0
        for idx in range(len(sentence_timesteps)):
            if sentence_timesteps[idx]["start"] is None: sentence_timesteps[idx]["start"] = sentence_timesteps[idx-1]["end"]
            if sentence_timesteps[idx]["end"] is None: sentence_timesteps[idx]["end"] = sentence_timesteps[idx+1]["start"]

        for idx in range(len(sentence_timesteps)):
            sentence_timesteps[idx]["start"] += start_time_now
            sentence_timesteps[idx]["end"] += start_time_now
            duration += sentence_timesteps[idx]["end"] - sentence_timesteps[idx]["start"]
        start_time_now += duration
        new_slide_sentence_timesteps.extend(sentence_timesteps)
    
    with open(cursor_save_path.replace(".json", "_mid.json"), 'w') as f: json.dump(cursor_result, f, indent=2)
    with open(cursor_save_path, 'w') as f: json.dump(new_slide_sentence_timesteps, f, indent=2)
    return len(cursor_token)/4