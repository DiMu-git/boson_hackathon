from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Dict, Any

import openai


def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def synthesize_audio(client: openai.Client,
                     system_text: str,
                     scene_text: str,
                     ref_paths: List[str],
                     speaker_tag: str,
                     content_text: str,
                     decoding: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are an AI assistant designed to convert text into speech.\n"
        "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
        "If no speaker tag is present, select a suitable voice on your own.\n\n"
        "<|scene_desc_start|>\n" + scene_text + "\n<|scene_desc_end|>"
    ) if system_text == "DEFAULT" else system_text

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": "[SPEAKER1] Reference sample for speaker identity."})
    for rp in ref_paths:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": b64(rp), "format": "wav"}
            }],
        })
    final_text = f"{speaker_tag} {content_text}".strip()
    messages.append({"role": "user", "content": final_text})

    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        temperature=decoding["temp"],
        top_p=decoding["top_p"],
        max_completion_tokens=4096,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        stream=False,
        extra_body={"top_k": decoding["top_k"]},
    )
    audio_b64 = resp.choices[0].message.audio.data
    return {"audio_bytes": base64.b64decode(audio_b64)}


if __name__ == "__main__":
    print(json.dumps({"detail": "This module provides synthesize_audio()."}))




