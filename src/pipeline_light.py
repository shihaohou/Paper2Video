'''
    1. (LLM) slide generation 
    2. (VLM) subtitle and cursor prompt generation
    3. TTS->audio; GUI&WhisperX Grounding->cursor;
    4. Talking Gen: local-[hallo2, fantasy, ...], api-[HeyGen]
    5. Merage
'''
import cv2
import pdb
import json
import time
import shutil
import asyncio
import os, sys
import argparse
import subprocess
from os import path
import fitz


def pdf_to_pngs(pdf_path, out_dir, dpi=400):
    """Render PDF pages to numbered PNGs via PyMuPDF.

    Replaces pdf2image/poppler, which silently strips complex TikZ
    vector content (gradients, hatching, transparency clouds) embedded
    in figures — symptom: parts of the slide come out blank/white.
    """
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = path.join(out_dir, "{}.png".format(i + 1))
        pix.save(out_path)
        paths.append(out_path)
    doc.close()
    return paths


from speech_gen import tts_per_slide
from subtitle_render import add_subtitles
from cursor_gen import cursor_gen_per_sentence
from slide_code_gen_select_improvement import latex_code_gen_upgrade
from cursor_render import render_video_with_cursor_from_json
from subtitle_cursor_prompt_gen import subtitle_cursor_gen

from wei_utils import get_agent_config


# os.environ["GEMINI_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""

def copy_folder(src_dir, dst_dir):
    if not os.path.exists(src_dir): raise FileNotFoundError(f"no such dir: {src_dir}")
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.copytree(src_dir, dst_dir)

def str2list(s):
    return [int(x) for x in s.split(',')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper2Video Generation Pipeline')
    parser.add_argument('--result_dir', type=str, default='./result/zeyu')
    parser.add_argument('--model_name_t', type=str, default='gpt-4.1') 
    parser.add_argument('--model_name_v', type=str, default='gpt-4.1') 
    parser.add_argument('--paper_latex_root', type=str, default='./assets/demo/latex_proj')
    parser.add_argument('--ref_img', type=str, default=None)
    parser.add_argument('--ref_audio', type=str, default='./assets/demo/zeyu.wav')
    parser.add_argument('--ref_text', type=str, default=None)
    parser.add_argument('--gpu_list', type=str2list, default="")
    parser.add_argument('--if_tree_search', type=bool, default=True)
    parser.add_argument('--beamer_templete_prompt', type=str, default=None)
    parser.add_argument('--stage', type=str, default="[\"0\"]")
    parser.add_argument('--talking_head_env', type=str, default="")
    parser.add_argument('--tts_model_type', type=str, default='f5', choices=['f5', 'moss'])
    parser.add_argument('--moss_env', type=str, default=None,
                        help='Path to MOSS-TTS venv python interpreter (used when --tts_model_type moss)')
    parser.add_argument('--tts_speed', type=float, default=1.0,
                        help='Post-TTS atempo speed factor (0.8-1.5). 1.0 = no change, 1.2 ~ moderate speed-up')
    parser.add_argument('--slide_outline_file', type=str, default=None,
                        help='Path to a plain text file describing the exact slide-by-slide outline. '
                             'When set, LLM is told to ignore the default chapter structure and follow this outline.')
    # slide+subtitle: 1; 
    # tts+cusor: 2; 
    # talking-head: 3: 
    # all: 0
    args = parser.parse_args()
    stage = json.loads(args.stage)
    print("start", "stage:", stage, args.gpu_list)
    
    cursor_img_path = "./cursor_image/red.png"
    os.makedirs(args.result_dir, exist_ok=True) # result dir
    agent_config_t = get_agent_config(args.model_name_t) # LLM
    agent_config_v = get_agent_config(args.model_name_v) # VLM
    copy_latex_proj_path = path.join(args.result_dir, path.basename(args.paper_latex_root))
    if path.exists(copy_latex_proj_path) is False:
        copy_folder(args.paper_latex_root, copy_latex_proj_path)
    args.paper_latex_root = copy_latex_proj_path
    
    if path.exists(path.join(args.result_dir, "sat.json")) is True:
        with open(path.join(args.result_dir, "sat.json"), 'r') as f: 
            time_second = json.load(f)
    else: time_second = {}
        
    if path.exists(path.join(args.result_dir, "token.json")) is True:
        with open(path.join(args.result_dir, "token.json"), 'r') as f: 
            token_usage = json.load(f)
    else: token_usage = {}
    
    ## Step 1: Slide Generation
    slide_latex_path = path.join(args.paper_latex_root, "slides.tex")
    slide_image_dir = path.join(args.result_dir, 'slide_imgs')
    os.makedirs(slide_image_dir, exist_ok=True)
    
    start_time = time.time() # start time
    if "1" in stage or  "0" in stage:
        prompt_path = "./prompts/slide_beamer_prompt.txt"
        outline_override = None
        if args.slide_outline_file:
            with open(args.slide_outline_file, 'r', encoding='utf-8') as f_outline:
                outline_override = f_outline.read()
        if args.if_tree_search is True:
            usage_slide, beamer_path = latex_code_gen_upgrade(prompt_path=prompt_path, tex_dir=args.paper_latex_root, beamer_save_path=slide_latex_path,
                                                            model_config_ll=agent_config_t, model_config_vl=agent_config_v, beamer_temp_name=args.beamer_templete_prompt,
                                                            outline_override=outline_override)
        else:
            paper_latex_path = path.join(args.paper_latex_root, "main.tex") 
            usage_slide = latex_code_gen(prompt_path=prompt_path, tex_dir=args.paper_latex_root, tex_path=paper_latex_path, beamer_save_path=slide_latex_path, model_config=agent_config_t)
            
        pdf_to_pngs(beamer_path, slide_image_dir, dpi=400)  # save slides as images
        if args.model_name_t not in token_usage.keys(): 
            token_usage[args.model_name_t] = [usage_slide]
        else: token_usage[args.model_name_t].append(usage_slide)
        step1_time =  time.time()
        time_second["slide_gen"] = [step1_time-start_time]
        print("Slide Generation", step1_time-start_time)
    
    ## Step 2: Subtitle and Cursor Prompt Generation
    start_time = time.time() # start time
    subtitle_cursor_save_path = path.join(args.result_dir, 'subtitle_w_cursor.txt')
    cursor_save_path = path.join(args.result_dir, 'cursor.json')

    speech_save_dir = path.join(args.result_dir, 'audio')
    if "2" in stage or  "0" in stage:
        prompt_path = "prompts/slide_subtitle_cursor_prompt.txt"
        subtitle, usage_subtitle = subtitle_cursor_gen(slide_image_dir, prompt_path, agent_config_v)
        with open(subtitle_cursor_save_path, 'w') as f: f.write(subtitle)
        if args.model_name_v not in token_usage.keys(): 
            token_usage[args.model_name_v] = [usage_subtitle]
        else: token_usage[args.model_name_v].append(usage_subtitle)
        step2_time =  time.time()
        time_second["subtitle_cursor_prompt_gen"] = [step2_time-start_time]
        print("Subtitle and Cursor Prompt Generation", step2_time-start_time)

    
        ## Step 3-1: Speech Generation
        tts_per_slide(model_type=args.tts_model_type, script_path=subtitle_cursor_save_path,
                    speech_save_dir=speech_save_dir, ref_audio=args.ref_audio, ref_text=args.ref_text,
                    moss_python=args.moss_env, speed=args.tts_speed)
        step3_1_time =  time.time()
        time_second["tts"] = [step3_1_time-step2_time]
        print("Speech Generation", step3_1_time-step2_time)
        
        
        ## Step 3-2: Cursor Generation
        os.environ["PYTHONHASHSEED"] = "random"        
        cursor_token = cursor_gen_per_sentence(script_path=subtitle_cursor_save_path, slide_img_dir=slide_image_dir, 
                                slide_audio_dir=speech_save_dir, cursor_save_path=cursor_save_path, gpu_list=args.gpu_list)
        token_usage["cursor"] = cursor_token
        step3_2_time =  time.time()
        time_second["cursor_gen"] = [step3_2_time-step3_1_time]
        print("Cursor Generation", step3_2_time-step3_1_time)
    
    
    ## Step 4: Video Generation
    start_time = time.time() # start time
    if "3" in stage or  "0" in stage:    
        ## Step5: Merage
        # merage talking and slides
        tmp_merage_dir = path.join(args.result_dir, "merage")
        tmp_merage_1 = path.join(args.result_dir, "1_merage.mp4")
        image_size = cv2.imread(path.join(slide_image_dir, '1.png')).shape
        size = max(image_size[0]//6, image_size[1]//6)
        num_slide = len(os.listdir(slide_image_dir))
        ref_img_name = (path.basename(args.ref_img).rsplit(".", 1)[0]
                        if args.ref_img else "")
        merage_cmd =  ["./1_merage_light.bash", slide_image_dir, speech_save_dir, tmp_merage_dir,
                       str(num_slide), tmp_merage_1, ref_img_name]
        out = subprocess.run(merage_cmd, text=True)
        # render cursor
        cursor_size = size//6
        tmp_merage_2 = path.join(args.result_dir, "2_merage.mp4")
        render_video_with_cursor_from_json(video_path=tmp_merage_1, out_video_path=tmp_merage_2, 
                                        json_path=cursor_save_path, cursor_img_path=cursor_img_path, 
                                        transition_duration=0.1, cursor_size=cursor_size)
        # render subtitle
        front_size = size//10
        tmp_merage_3 = path.join(args.result_dir, "3_merage.mp4")
        add_subtitles(tmp_merage_2, tmp_merage_3, size//10)
        step5_time =  time.time()
        time_second["merage"] = [step5_time-start_time]
        print("Merage", step5_time-start_time)
        
    ## sat. save
    with open(path.join(args.result_dir, "sat.json"), 'w') as f: json.dump(time_second, f, indent=4)
    with open(path.join(args.result_dir, "token.json"), 'w') as f: json.dump(token_usage, f, indent=4)
