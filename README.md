# CIFAR-10 classification via ai.sooners.us (gemma3:4b)

### What this does
- Randomly samples **10 images per class** (total 100) from CIFAR-10 (train split) with a fixed seed.
- Sends each as a **base64 Data URL** in an `image_url` content part to `/api/chat/completions`.
- Keeps the **system message** as the prompt under test (edit `SYSTEM_PROMPT` in `main.py`).
- Parses the model’s top prediction as one of the **10 CIFAR-10 labels**.
- Computes **strict accuracy** (invalid outputs counted wrong) and saves a **10×10 confusion matrix** (valid-only).

### Setup
```bash
python -m venv .venv 
.venv\Scripts\activate
pip install -r requirements.txt
```

### Setup 2
Create .soonerai.env file in current directory. 
Delete plots, csv and jsonl files (you will generate your own, these were just for reference.)

### To run
```bash
python cifar10_classify.py
``` 

## Prompt Comparison:
I tried different prompts to analysze how wording affects accuracy. From theh result it
is clear that precise wording and thorough instrcution yields better results.

Prompt 1 : 57% accuracy (No Invalids)

```bash
    "You are an image classifier. Given a CIFAR-10 image, "
    "respond with exactly one of these labels: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
    "Return just the single label, nothing else."
```

```bash
Prompt 2: 61% accuracy (No Invalids)

    "Given these CIFAR-10 images, thoroughly compare the images and classify them into their respective categories. "
    "Only respond with the appropriate label from the following list, and nothing else: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
    "Be mindful of the details in each image to ensure accurate classification."
```

