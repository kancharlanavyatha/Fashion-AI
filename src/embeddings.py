"""
embeddings.py
─────────────
CLIP embedding utilities shared across all three tasks.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import clip


def load_clip(device: str = None):
    """Load CLIP ViT-B/32. Auto-selects GPU if available."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()
    return model, preprocess, device


def get_image_embeddings(
    img_paths: list,
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed a list of image paths with CLIP.

    Returns
    -------
    np.ndarray of shape (N, 512), L2-normalised.
    """
    all_embeds = []
    for start in tqdm(range(0, len(img_paths), batch_size), desc='Embedding images'):
        batch_paths = img_paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert('RGB'))
            except Exception:
                img = preprocess(Image.new('RGB', (224, 224), color=(200, 200, 200)))
            images.append(img)

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(batch).float()

        features = features / features.norm(dim=-1, keepdim=True)
        all_embeds.append(features.cpu().numpy())

    return np.vstack(all_embeds)   # (N, 512)


def encode_text(text: str, model, device: str) -> np.ndarray:
    """
    Encode a text string with CLIP.

    Returns
    -------
    np.ndarray of shape (512,), L2-normalised.
    """
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embed = model.encode_text(tokens).float()
    embed = embed / embed.norm(dim=-1, keepdim=True)
    return embed.cpu().numpy().squeeze()


def save_embeddings(embeddings: np.ndarray, path: str):
    np.save(path, embeddings)
    print(f'Embeddings saved → {path}')


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)
