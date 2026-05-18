"""Standalone CLI wrapper for MOSS-TTS (8B Delay) voice cloning.

Runs in the MOSS-TTS Python env (Python 3.12 + CUDA torch + MOSS-TTS
package installed). Invoked as a subprocess by speech_gen.inference_moss.

Does NOT import anything from the Paper2Video src/ package — keep this
file self-contained so it works under a different Python.
"""
import argparse
import os

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="OpenMOSS-Team/MOSS-TTS")
    p.add_argument("--ref-audio", required=True, help="Reference wav for voice cloning")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--output-audio", required=True, help="Output wav path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-implementation", default="sdpa",
                   help="sdpa (default) | flash_attention_2 | eager")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))

    conversations = [[processor.build_user_message(text=args.text, reference=[args.ref_audio])]]
    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            audio_temperature=float(args.temperature),
            audio_top_p=float(args.top_p),
            audio_top_k=int(args.top_k),
            audio_repetition_penalty=float(args.repetition_penalty),
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise RuntimeError("MOSS-TTS did not return a decodable audio result.")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)

    out_dir = os.path.dirname(args.output_audio)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(args.output_audio, audio_np, sample_rate)
    print(f"[moss-tts] wrote {args.output_audio} "
          f"({len(audio_np) / sample_rate:.2f}s @ {sample_rate}Hz)")


if __name__ == "__main__":
    main()
