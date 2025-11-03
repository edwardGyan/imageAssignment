#!/usr/bin/env python3
"""
CIFAR-10 classification via ai.sooners.us (OpenAI-compatible Chat Completions).

Spec:
  • Randomly sample 10 images per class (total 100) with a fixed seed.
  • Send each image as base64 Data URL in a user message with an image_url part.
  • Keep the SYSTEM message as the prompt under test.
  • Parse the model’s top prediction as one of the 10 CIFAR-10 class names.
  • Compute overall accuracy and save a 10x10 confusion matrix image.

Artifacts:
  - confusion_matrix.png
  - predictions.csv
  - misclassifications.jsonl

Requires:
  pip install -r requirements.txt
"""

import os
import io
import time
import json
import base64
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
from PIL import Image

import torch
import torchvision
from torchvision.datasets import CIFAR10

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt



CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DEFAULT_SEED = 1337
SAMPLES_PER_CLASS = 10


SYSTEM_PROMPT = (
    "You are an image classifier. Given a CIFAR-10 image, "
    "respond with exactly one of these labels: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
    "Return just the single label, nothing else."
)

# SYSTEM_PROMPT = (
#     "Given these CIFAR-10 images, thoroughly compare the images and classify them into their respective categories. "
#     "Only respond with the appropriate label from the following list, and nothing else: "
#     "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
#     "Be mindful of the details in each image to ensure accurate classification."
# )

# SYSTEM_PROMPT = (
#     "These are CIFAR-10 images. Quickly classify them into the following categories. Give me only the label from this list, and nothing else: "
#     "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
#     "Speed in giving me results is most important."
# )
USER_INSTRUCTION = (
    "Classify this CIFAR-10 image. Respond with exactly one label from this list:\n"
    f"{', '.join(CLASSES)}\n"
    "Your reply must be just the label, nothing else."
)



def load_env():
    # Optional .env by python-dotenv if available, otherwise OS env.
    # We avoid hard-failing if dotenv isn't installed.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
        load_dotenv(".env")
    except Exception:
        pass

    api_key = os.getenv("SOONERAI_API_KEY")
    base_url = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
    model = os.getenv("SOONERAI_MODEL", "gemma3:4b")

    if not api_key:
        raise RuntimeError("Missing SOONERAI_API_KEY (set in ~/.soonerai.env or ./.env)")
    return api_key, base_url, model

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 eval via ai.sooners.us")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    p.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.0, help="sleep between requests (s)")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--save-confusion", type=str, default="confusion_matrix.png")
    p.add_argument("--save-predictions", type=str, default="predictions.csv")
    p.add_argument("--save-miscls", type=str, default="misclassifications.jsonl")
    return p.parse_args()

# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 60,
    max_retries: int = 3,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the text reply.

    Uses content parts with an image_url Data URL for VLM inputs.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    backoff = 1.0
    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                # OpenAI-compatible structure
                return data["choices"][0]["message"]["content"].strip()
            last_err = f"HTTP {resp.status_code}: {resp.text}"
        except requests.RequestException as e:
            last_err = str(e)

        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"API error after {max_retries} attempts: {last_err}")

def normalize_label(text: str) -> str:
    t = text.lower().strip()
    # exact match
    if t in CLASSES:
        return t
    # contains match (e.g., "I think it's a truck")
    for c in CLASSES:
        if c in t:
            return c
    return "__invalid__"



def stratified_sample_cifar10(samples_per_class: int, seed: int, root: str = "./data") -> List[Tuple[Image.Image, int]]:
    ds = CIFAR10(root=root, train=True, download=True)
    per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class[label].append(idx)

    rng = random.Random(seed)
    selected: List[Tuple[Image.Image, int]] = []
    for label in range(10):
        chosen = rng.sample(per_class[label], samples_per_class)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt))
    return selected



def plot_confusion(cm: np.ndarray, classes: List[str], out_path: str, title: str):
    plt.figure(figsize=(7.5, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def main():
    args = parse_args()
    load_env()
    api_key, base_url, model = load_env()
    set_seed(args.seed)

    print(f"Sampling CIFAR-10 (seed={args.seed}, {args.samples_per_class}/class)...")
    samples = stratified_sample_cifar10(args.samples_per_class, args.seed, root=args.data_root)
    assert len(samples) == 10 * args.samples_per_class

    y_true: List[int] = []
    y_pred_valid: List[int] = []   # only valid predictions (0..9), for confusion matrix
    valid_true_for_cm: List[int] = []
    invalid_rows: List[Dict] = []
    rows_for_csv: List[str] = ["index,true_label,pred_label,raw_reply,is_valid"]

    total = len(samples)
    num_correct = 0

    for i, (img, tgt) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img)
        true_label = CLASSES[tgt]

        try:
            reply = post_chat_completion_image(
                image_data_url=data_url,
                system_prompt=SYSTEM_PROMPT,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
        except Exception as e:
            reply = f"__error__: {e}"
            pred_label = "__invalid__"
        else:
            pred_label = normalize_label(reply)

        is_valid = pred_label in CLASSES
        is_correct = is_valid and (pred_label == true_label)
        if is_correct:
            num_correct += 1

        print(f"[{i:03d}/{total}] true={true_label:>10s} | pred={pred_label:>10s} | valid={is_valid} | raw='{reply}'")

        # For confusion matrix: include only valid preds (keeps 10x10 shape honest)
        if is_valid:
            y_pred_valid.append(CLASSES.index(pred_label))
            valid_true_for_cm.append(tgt)
        else:
            invalid_rows.append({"i": i, "true": true_label, "raw_reply": reply})

        y_true.append(tgt)
        rows_for_csv.append(f"{i},{true_label},{pred_label},{json.dumps(reply).strip()},${str(is_valid)}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Accuracy: strict — invalid predictions count as wrong and never receive credit.
    accuracy = num_correct / total
    print(f"\nStrict accuracy over {total} images (invalid = wrong): {accuracy*100:.2f}%")
    print(f"Invalid/Non-CIFAR outputs: {len(invalid_rows)}")

    # Confusion matrix on valid-only predictions
    if len(y_pred_valid) > 0:
        cm = confusion_matrix(valid_true_for_cm, y_pred_valid, labels=list(range(10)))
    else:
        cm = np.zeros((10, 10), dtype=int)

    out_png = args.save_confusion
    plot_confusion(
        cm,
        CLASSES,
        out_path=out_png,
        title=f"CIFAR-10 Confusion Matrix (valid preds only) — {model}",
    )
    print(f"Saved {out_png}")

    # Save misclassifications and predictions
    with open(args.save_miscls, "w", encoding="utf-8") as f:
        for row in invalid_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(invalid_rows)} invalid rows to {args.save_miscls}")

    with open(args.save_predictions, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_for_csv) + "\n")
    print(f"Saved predictions to {args.save_predictions}")

if __name__ == "__main__":
    main()
