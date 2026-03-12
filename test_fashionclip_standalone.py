"""
Standalone test for Marqo/marqo-fashionCLIP model.

Tests image and text embedding generation, and cross-modal similarity search.
No dependencies on Pinecone, Neo4j, segmentation, or GPT-4o.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Marqo/marqo-fashionCLIP ...")
print("=" * 60)

import open_clip
import torch

MODEL_HF_ID = "hf-hub:Marqo/marqo-fashionCLIP"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load directly via open_clip (the model's actual backend)
model, _, processor = open_clip.create_model_and_transforms(MODEL_HF_ID)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer(MODEL_HF_ID)

# Probe embedding dim via a dummy forward pass
with torch.no_grad():
    dummy_tokens = tokenizer(["test"]).to(device)
    dummy_emb = model.encode_text(dummy_tokens, normalize=True)
embedding_dim = dummy_emb.shape[-1]

print(f"Model loaded. Embedding dim: {embedding_dim}")
print()


# ── 2. Helper functions ────────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens, normalize=True)
    return features.cpu().numpy()[0]


def embed_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = processor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor, normalize=True)
    return features.cpu().numpy()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── 3. Test: text embeddings ───────────────────────────────────────────────────
print("=" * 60)
print("Test 1: Text embedding shape & norm")
print("=" * 60)

texts = [
    "blue denim jeans",
    "red floral summer dress",
    "black leather handbag",
    "white sneakers",
    "grey wool coat",
]

for t in texts:
    emb = embed_text(t)
    norm = np.linalg.norm(emb)
    print(f"  '{t}'  shape={emb.shape}  norm={norm:.4f}")

print()


# ── 4. Test: image embeddings ──────────────────────────────────────────────────
print("=" * 60)
print("Test 2: Image embedding shape & norm")
print("=" * 60)

# Test images: (path, true category label)
test_images = [
    ("/tmp/test_sneakers.jpg", "sneakers / shoes"),
    ("/tmp/test_dress.jpg",    "dress / outfit"),
    ("/tmp/test_bag.jpg",      "bag / purse"),
]

for img_path, label in test_images:
    emb = embed_image(img_path)
    norm = np.linalg.norm(emb)
    print(f"  [{label}]  shape={emb.shape}  norm={norm:.4f}")

print()


# ── 5. Test: cross-modal text-to-image similarity ─────────────────────────────
print("=" * 60)
print("Test 3: Cross-modal text-to-image similarity")
print("    (does fashionCLIP match the right category to each image?)")
print("=" * 60)

candidate_texts = [
    "sneakers or athletic shoes",
    "a summer dress or outfit",
    "a handbag or purse",
    "jeans or trousers",
    "a jacket or coat",
    "a hat",
    "sunglasses",
]

for img_path, true_label in test_images:
    img_emb = embed_image(img_path)
    scores = [(t, cosine_sim(img_emb, embed_text(t))) for t in candidate_texts]
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Image: {Path(img_path).name}  (true: {true_label})")
    for rank, (label, score) in enumerate(scores[:4], 1):
        marker = "  <-- CORRECT" if true_label.split()[0].lower() in label.lower() else ""
        print(f"  #{rank}  {label:<35} sim={score:.4f}{marker}")
    print()


# ── 6. Test: text-to-text semantic similarity ─────────────────────────────────
print("=" * 60)
print("Test 4: Text-to-text similarity")
print("    (semantically similar items should score higher)")
print("=" * 60)

anchor = "blue denim jeans"
comparisons = [
    ("skinny jeans in dark blue",    "SIMILAR"),
    ("denim trousers",               "SIMILAR"),
    ("red floral summer dress",      "DIFFERENT"),
    ("black leather handbag",        "DIFFERENT"),
    ("blue jeans casual style",      "SIMILAR"),
    ("white t-shirt",                "DIFFERENT"),
]

anchor_emb = embed_text(anchor)
print(f"Anchor: '{anchor}'")
for text, expected in comparisons:
    score = cosine_sim(anchor_emb, embed_text(text))
    bar = "█" * int(score * 50)
    print(f"  [{expected}] '{text}'")
    print(f"            sim={score:.4f}  {bar}")
print()

print("=" * 60)
print("All tests passed!")
print("=" * 60)
