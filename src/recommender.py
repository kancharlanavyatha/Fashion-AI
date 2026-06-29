"""
recommender.py
──────────────
Task 1: Smart complementary product recommendation engine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ── Complementary category maps ────────────────────────────────────────────────

MASTER_COMPLEMENT_MAP = {
    'Apparel':        ['Footwear', 'Accessories'],
    'Footwear':       ['Apparel', 'Accessories'],
    'Accessories':    ['Apparel', 'Footwear'],
    'Personal Care':  ['Apparel', 'Accessories'],
    'Sporting Goods': ['Footwear', 'Accessories'],
    'Home':           ['Personal Care'],
}

ARTICLE_COMPLEMENT_MAP = {
    'Tshirts':        ['Jeans', 'Casual Shoes', 'Watches', 'Sunglasses'],
    'Shirts':         ['Trousers', 'Formal Shoes', 'Belts', 'Watches'],
    'Jeans':          ['Tshirts', 'Casual Shoes', 'Belts'],
    'Casual Shoes':   ['Jeans', 'Tshirts', 'Socks'],
    'Formal Shoes':   ['Shirts', 'Trousers', 'Belts'],
    'Sports Shoes':   ['Track Pants', 'Socks'],
    'Watches':        ['Shirts', 'Tshirts', 'Formal Shoes'],
    'Sunglasses':     ['Tshirts', 'Casual Shoes'],
    'Kurtas':         ['Churidar', 'Flats', 'Dupatta'],
    'Sarees':         ['Heels', 'Clutches', 'Earrings'],
    'Dresses':        ['Heels', 'Sandals', 'Clutches', 'Earrings'],
    'Tops':           ['Jeans', 'Skirts', 'Flats', 'Sandals'],
    'Heels':          ['Dresses', 'Tops', 'Skirts'],
    'Flats':          ['Kurtas', 'Tops', 'Jeans'],
    'Bags':           ['Tops', 'Dresses', 'Casual Shoes'],
    'Backpacks':      ['Tshirts', 'Casual Shoes', 'Jeans'],
    'Belts':          ['Jeans', 'Trousers', 'Shirts'],
    'Socks':          ['Sports Shoes', 'Casual Shoes'],
    'Track Pants':    ['Sports Shoes', 'Socks'],
    'Jackets':        ['Tshirts', 'Jeans', 'Casual Shoes'],
    'Sweatshirts':    ['Track Pants', 'Sports Shoes'],
}


def get_recommendations(
    query_idx: int,
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 4,
) -> tuple[dict, list[dict]]:
    """
    Find top-k complementary products for the item at `query_idx`.

    Returns
    -------
    (query_row_dict, list_of_result_dicts)
    Each result dict includes the product row fields + 'score' and 'idx'.
    """
    query_row    = sample.iloc[query_idx]
    query_embed  = embeddings[query_idx]
    article_type = query_row.get('articleType', '')
    master_cat   = query_row.get('masterCategory', '')

    # Determine candidate pool
    target_articles = ARTICLE_COMPLEMENT_MAP.get(article_type)
    if target_articles:
        mask = sample['articleType'].isin(target_articles)
    else:
        target_cats = MASTER_COMPLEMENT_MAP.get(master_cat, [])
        mask = sample['masterCategory'].isin(target_cats)

    mask = mask.copy()
    mask.iloc[query_idx] = False          # exclude self
    cand_indices = np.where(mask)[0]

    # Fallback: any different article type
    if len(cand_indices) == 0:
        cand_indices = np.array([
            i for i in range(len(sample))
            if i != query_idx and sample.iloc[i]['articleType'] != article_type
        ])

    scores     = embeddings[cand_indices] @ query_embed
    top_local  = scores.argsort()[-top_k:][::-1]
    top_global = cand_indices[top_local]

    results = []
    for idx, score in zip(top_global, scores[top_local]):
        r = sample.iloc[idx].to_dict()
        r['score'] = float(score)
        r['idx']   = int(idx)
        results.append(r)
    return query_row.to_dict(), results


def visualize_recommendations(
    query_idx: int,
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 4,
    save_path: str = None,
):
    query, recs = get_recommendations(query_idx, sample, embeddings, top_k=top_k)

    fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 5))
    fig.patch.set_facecolor('#f8f4ff')

    # Query
    ax = axes[0]
    try:
        ax.imshow(Image.open(query['img_path']).convert('RGB'))
    except Exception:
        ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(
        f"🔍 Query\n{str(query.get('productDisplayName',''))[:28]}\n({query.get('articleType','')})",
        fontsize=8, fontweight='bold', color='#4A148C'
    )
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('#4A148C')
        spine.set_linewidth(3)
    ax.set_frame_on(True)

    # Recommendations
    palette = ['#E8F5E9', '#E3F2FD', '#FFF8E1', '#FCE4EC']
    for i, rec in enumerate(recs):
        ax = axes[i + 1]
        try:
            ax.imshow(Image.open(rec['img_path']).convert('RGB'))
        except Exception:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(
            f"#{i+1} {str(rec.get('productDisplayName',''))[:25]}\n"
            f"({rec.get('articleType','')})\nScore: {rec['score']:.3f}",
            fontsize=7.5, color='#1B5E20'
        )
        ax.axis('off')
        ax.patch.set_facecolor(palette[i % len(palette)])

    plt.suptitle(
        f"Complementary Recommendations: {str(query.get('productDisplayName',''))[:40]}",
        fontsize=11, fontweight='bold', color='#4A148C', y=1.02
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
