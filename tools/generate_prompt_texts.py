from __future__ import annotations

import random
from pathlib import Path


STYLES = [
    "neutral", "conversational", "formal", "announcement", "calm",
    "excited", "sad", "confident", "fast", "slow",
]

SEED = 123


def build_sentences(n: int = 100) -> list[str]:
    base = [
        "Hey, quick question—are you free later this afternoon?",
        "Good afternoon. Thank you all for joining us today.",
        "Everything is okay. Take a slow breath and relax.",
        "Please proceed to the main hall. The session will begin shortly.",
        "I appreciate your patience; the results will be announced tomorrow.",
        "Could you share a brief update on your progress?",
        "This concludes our meeting. Have a wonderful rest of your day.",
        "Let me double-check the details before I send the email.",
        "Welcome to our demo. We’ll walk through the key features.",
        "No worries—take your time and let me know if you need help.",
    ]
    random.seed(SEED)
    out: list[str] = []
    for i in range(n):
        s = random.choice(base)
        style = random.choice(STYLES)
        if style == "neutral":
            txt = s
        elif style == "conversational":
            txt = s + " Thanks again!"
        elif style == "formal":
            txt = s + " We sincerely appreciate your cooperation."
        elif style == "announcement":
            txt = s + " Please stay tuned for further instructions."
        elif style == "calm":
            txt = s + " Everything is under control."
        elif style == "excited":
            txt = s + " This is fantastic news!"
        elif style == "sad":
            txt = s + " Unfortunately, the outcome wasn’t as expected."
        elif style == "confident":
            txt = s + " I am certain we will deliver on time."
        elif style == "fast":
            txt = s + " Let’s move quickly and keep our momentum."
        elif style == "slow":
            txt = s + " We will proceed carefully, step by step."
        else:
            txt = s
        out.append(txt)
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "src" / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create text prompts directory
    text_dir = out_dir / "auto"
    text_dir.mkdir(parents=True, exist_ok=True)

    sentences = build_sentences(100)
    manifest_lines = []
    for i, s in enumerate(sentences):
        name = f"text_{i:03d}.txt"
        (text_dir / name).write_text(s + "\n", encoding="utf-8")
        manifest_lines.append(f"{name}\t{s}")

    (text_dir / "_manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Generated {len(sentences)} text prompts in {text_dir}")


if __name__ == "__main__":
    main()




