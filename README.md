# 🛍️ AI Product Intelligence System
### Gen AI Bootcamp — Day 2 Homework Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/fashion-ai-product-intelligence/blob/main/notebooks/fashion_ai_product_intelligence.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![CLIP](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F32-purple)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered product intelligence system built on **OpenAI CLIP** embeddings, implementing three advanced e-commerce features on the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset.

---

## ✨ Features

| Task | What it does |
|------|-------------|
| **Task 1 — Smart Recommendations** | Given a product, recommends complementary items you'd actually buy together (shoes → socks + jeans, not more shoes) |
| **Task 2 — Unique Catalog Creation** | Detects near-duplicate products across sellers and builds a clean canonical catalog using agglomerative clustering |
| **Task 3 — Reverse Product Search** | Natural language → product images. Type "blue casual shirt" and get matching products — no query image needed |

---

## 🏗️ Architecture

```
Product Images
      │
      ▼
 CLIP ViT-B/32 ◄──── Text Query (Task 3)
      │
      ▼
 512-d Embeddings (L2 normalised)
      │
      ├──► Cosine Similarity + Category Map ──► Complementary Recs (Task 1)
      │
      ├──► Agglomerative Clustering (dist < 0.15) ──► Unique Catalog (Task 2)
      │
      └──► Dot Product with Text Embed ──► Text Search Results (Task 3)
```

**Why CLIP?**  
CLIP is pre-trained on 400M image-text pairs and maps both modalities into a shared 512-d space. This means we get image-image, text-image, and text-text comparison all under one model — no fine-tuning, no separate indexing pipelines.

---

## 🚀 Quickstart

### Run on Google Colab (recommended)
Click the badge above. No local setup needed — GPU is provided free.

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/fashion-ai-product-intelligence.git
cd fashion-ai-product-intelligence
pip install -r requirements.txt
jupyter notebook notebooks/fashion_ai_product_intelligence.ipynb
```

You'll need a Kaggle API token (`kaggle.json`) to download the dataset. Get it from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.

---

## 📁 Project Structure

```
fashion-ai-product-intelligence/
├── notebooks/
│   └── fashion_ai_product_intelligence.ipynb   # Main notebook (all 3 tasks)
├── src/
│   ├── embeddings.py       # CLIP embedding utilities
│   ├── recommender.py      # Task 1: Recommendation engine
│   ├── deduplicator.py     # Task 2: Clustering & deduplication
│   └── search.py           # Task 3: Text search engine
├── outputs/                # Generated visualizations (gitignored)
├── requirements.txt
└── README.md
```

---

## 🧠 Task Details

### Task 1 — Smart Product Recommendation Engine

**Problem:** Visual similarity search returns more shoes when given a shoe. That's not useful. We want complementary products — things you'd buy *with* the item.

**Solution:**
1. Map each `articleType` to a list of complementary article types (e.g., `Tshirts → [Jeans, Casual Shoes, Watches]`)
2. Compute CLIP cosine similarity between the query and all items in those complementary categories
3. Return top-k by similarity score

This hybrid approach (structured domain knowledge + learned embeddings) gives contextually correct recommendations without needing purchase history.

---

### Task 2 — Unique Product Catalog Creation

**Problem:** Same product listed by 15 sellers = 15 duplicate entries. We need one canonical entry per product.

**Solution:**
1. Build pairwise cosine distance matrix from CLIP embeddings
2. Agglomerative clustering with `distance_threshold=0.15` (≈ cosine similarity > 0.85)
3. For each cluster, select the product closest to the cluster centroid as the representative
4. Export clean catalog to CSV

**Why Agglomerative over K-Means?** No need to specify number of clusters upfront. The threshold drives grouping automatically, which is exactly right for deduplication where we don't know how many unique products exist.

---

### Task 3 — Reverse Product Search (Text → Image)

**Problem:** Users know what they want in words but don't have a reference image to upload.

**Solution:**  
CLIP's joint embedding space means text and image embeddings are directly comparable. We encode the query text once and compute dot products against the pre-built image embedding index. Sub-second retrieval.

```python
# All it takes:
text_embed = model.encode_text(clip.tokenize(["blue casual shirt"]))
scores     = image_embeddings @ text_embed.T
top_k      = scores.argsort()[-5:][::-1]
```

Includes an `ipywidgets` interactive search UI built into the notebook.

---

## 📊 Results

- **Task 1:** Complementary recommendations across 20 article-type pairings; tested on Footwear, Apparel, and Accessories anchors
- **Task 2:** ~10-30% catalog size reduction on 1000-product sample; threshold tuned empirically via similarity distribution histogram  
- **Task 3:** Text queries reliably retrieve semantically correct products; alignment visualised via query-category heatmap

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Embeddings | `openai/CLIP` (ViT-B/32) |
| Clustering | `sklearn.cluster.AgglomerativeClustering` |
| Similarity | `numpy` dot product (L2-normed) |
| Visualization | `matplotlib`, `scipy` |
| Interactive UI | `ipywidgets` |
| Data | `pandas` |

---

## 📄 License

MIT
