"""
search.py
─────────
Task 3: Text-to-image product search using CLIP's joint embedding space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip


def text_search(
    query: str,
    model,
    device: str,
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """
    Search products by natural language query.

    Returns
    -------
    List of result dicts (product fields + 'score' + 'rank').
    """
    tokens     = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embed = model.encode_text(tokens).float()
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    text_embed = text_embed.cpu().numpy().squeeze()   # (512,)

    scores     = embeddings @ text_embed              # (N,) cosine sims
    top_idx    = scores.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        r = sample.iloc[idx].to_dict()
        r['score'] = float(scores[idx])
        r['rank']  = rank
        r['idx']   = int(idx)
        results.append(r)
    return results


def visualize_search_results(
    query: str,
    results: list[dict],
    save_path: str = None,
):
    top_k  = len(results)
    fig    = plt.figure(figsize=(3.5 * top_k, 5.5))
    fig.patch.set_facecolor('#f8f4ff')
    fig.text(0.5, 0.97, f'🔍  "{query}"  —  Top {top_k} Matches',
             ha='center', va='top', fontsize=13, fontweight='bold', color='#4A148C')

    axes   = fig.subplots(1, top_k)
    if top_k == 1:
        axes = [axes]
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}
    border_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#90CAF9', '#A5D6A7',
                     '#FFCCBC', '#D1C4E9', '#B2EBF2', '#DCEDC8', '#F8BBD0']

    for ax, res in zip(axes, results):
        try:
            ax.imshow(Image.open(res['img_path']).convert('RGB'))
        except Exception:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)

        rank  = res['rank']
        badge = medals.get(rank, f'#{rank}')
        ax.set_title(
            f"{badge} {str(res.get('productDisplayName',''))[:24]}\n"
            f"{res.get('articleType','')}\n"
            f"Score: {res['score']:.4f}",
            fontsize=7.5, color='#1B5E20'
        )
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors[(rank - 1) % len(border_colors)])
            spine.set_linewidth(3)
        ax.set_frame_on(True)

    plt.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
