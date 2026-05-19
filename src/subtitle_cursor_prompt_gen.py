import re
import pdb
import time
import os, sys
from os import path
from PIL import Image
from pathlib import Path
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType
from wei_utils import get_agent_config


def _retry_step(agent, msg, max_retries=8):
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = agent.step(msg)
            if getattr(response, "msgs", None) and len(response.msgs) > 0:
                return response
            print(f"[Retry {attempt+1}/{max_retries}] empty response, retrying...")
            time.sleep(min(2 ** attempt, 30))
        except Exception as e:
            last_exc = e
            backoff = min(2 ** attempt, 30)
            print(f"[Retry {attempt+1}/{max_retries}] {type(e).__name__}: {e}; "
                  f"sleeping {backoff}s before retry")
            time.sleep(backoff)
    if last_exc is not None:
        raise RuntimeError(
            f"agent failed after {max_retries} retries (last error: "
            f"{type(last_exc).__name__}: {last_exc})"
        ) from last_exc
    raise RuntimeError(f"agent failed after {max_retries} retries (no messages)")


def subtitle_cursor_gen(slide_imgs_dir, prompt_path, model_config):
    model = ModelFactory.create(
        model_platform=model_config["model_platform"],
        model_type=model_config["model_type"],
        model_config_dict=model_config.get("model_config"),
        url=model_config.get("url", None),)
    agent = ChatAgent(model=model, system_message="You are a helpful assistant.",)

    with open(prompt_path, 'r', encoding='utf-8') as f_prompt: task_prompt = f_prompt.read()
    slide_image_list = [path.join(slide_imgs_dir, name) for name in os.listdir(slide_imgs_dir)]
    slide_image_list = sorted(slide_image_list, key=lambda x: int(re.search(r'\d+', x).group()))

    images = []
    for idx, img_path in enumerate(slide_image_list): images.append(Image.open(img_path))
    messages = BaseMessage.make_user_message(role_name="user", content=task_prompt, image_list=images, meta_dict={})
    response = _retry_step(agent, messages)
    subtitle = response.msg.content.strip()
    return subtitle, response.info["usage"]
    
    

