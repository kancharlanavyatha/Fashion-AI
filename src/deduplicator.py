"""
deduplicator.py
───────────────
Task 2: Near-duplicate detection and unique catalog creation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import AgglomerativeClustering


def build_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance matrix from L2-normalised embeddings."""
    cos_sim = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    return 1.0 - cos_sim


def cluster_products(
    embeddings: np.ndarray,
    distance_threshold: float = 0.15,
) -> np.ndarray:
    """
    Run Agglomerative Clustering.

    Parameters
    ----------
    distance_threshold : float
        Cosine distance cutoff. 0.15 ≈ cosine similarity > 0.85.

    Returns
    -------
    labels : np.ndarray of shape (N,)
    """
    dist_matrix = build_distance_matrix(embeddings)
    clustering  = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='average',
    )
    return clustering.fit_predict(dist_matrix)


def get_representative(cluster_label: int, labels: np.ndarray, embeddings: np.ndarray) -> int:
    """Return index of the product closest to its cluster centroid."""
    member_idx = np.where(labels == cluster_label)[0]
    if len(member_idx) == 1:
        return int(member_idx[0])
    cluster_embeds = embeddings[member_idx]
    centroid = cluster_embeds.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
    return int(member_idx[dists.argmin()])


def build_unique_catalog(
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    distance_threshold: float = 0.15,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full deduplication pipeline.

    Returns
    -------
    (unique_catalog_df, cluster_labels_array)
    """
    labels     = cluster_products(embeddings, distance_threshold)
    n_clusters = len(set(labels))

    rep_indices = [get_representative(c, labels, embeddings) for c in range(n_clusters)]
    catalog     = sample.iloc[rep_indices].copy().reset_index(drop=True)
    cluster_sizes = pd.Series(labels).value_counts()
    catalog['cluster_id']   = range(n_clusters)
    catalog['cluster_size'] = [int(cluster_sizes.get(c, 1)) for c in range(n_clusters)]
    return catalog, labels


def visualize_cluster(
    cluster_label: int,
    labels: np.ndarray,
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    max_show: int = 6,
    save_path: str = None,
):
    member_indices = np.where(labels == cluster_label)[0]
    rep_idx        = get_representative(cluster_label, labels, embeddings)

    n   = min(len(member_indices), max_show)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    fig.patch.set_facecolor('#f8f4ff')
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, member_indices[:n]):
        row = sample.iloc[idx]
        try:
            ax.imshow(Image.open(row['img_path']).convert('RGB'))
        except Exception:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)

        is_rep     = (idx == rep_idx)
        edge_color = '#4CAF50' if is_rep else '#9E9E9E'
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(4 if is_rep else 1)
        ax.set_frame_on(True)
        prefix = '✅ CANONICAL\n' if is_rep else ''
        ax.set_title(
            f"{prefix}{str(row.get('productDisplayName',''))[:22]}\n{row.get('articleType','')}",
            fontsize=7.5, color='#1B5E20' if is_rep else '#424242'
        )
        ax.axis('off')

    plt.suptitle(
        f'Cluster {cluster_label} — {len(member_indices)} near-duplicate(s)',
        fontsize=11, fontweight='bold', color='#4A148C'
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
