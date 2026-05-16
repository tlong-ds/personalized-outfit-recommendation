from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from graph_recsys.utils.io import read_json
from graph_recsys.utils.io import ensure_dir


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in str(text).split()], dtype=np.float32)


def _seq(x: str) -> list[str]:
    return [s for s in str(x).split(";") if s]


def _hash_vec(key: str, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((key, seed))) % (2**32))
    return rng.standard_normal(dim).astype(np.float32)


def _title_vec(title: str, dim: int, seed: int) -> np.ndarray:
    toks = [t.strip().lower() for t in str(title).split() if t.strip()]
    if not toks:
        return _hash_vec("empty", dim, seed)
    mats = np.stack([_hash_vec(f"tok::{tok}", dim, seed) for tok in toks[:24]], axis=0)
    return mats.mean(axis=0)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_embeddings_tf(
    canonical_dir: Path,
    artifacts_dir: Path,
    de: int,
    seed: int,
    epochs: int = 3,
    triplet_margin: float = 0.1,
) -> dict[str, str]:
    ensure_dir(artifacts_dir)
    _set_seed(seed)
    items = pd.read_parquet(canonical_dir / "items.parquet")
    base_dim = 64
    image = np.stack([_hash_vec(f"img::{u}", base_dim, seed + 7) for u in items["image_url"].astype(str)], axis=0)
    text = np.stack([_title_vec(t, base_dim, seed + 11) for t in items["title"].astype(str)], axis=0)
    cf = np.stack([_hash_vec(f"cf::{i}", base_dim, seed + 17) for i in items["item_id"].astype(str)], axis=0)
    x = np.concatenate([image, text, cf], axis=1).astype(np.float32)
    cats = items["category_id"].astype(str).tolist()
    cat_to_idx = {c: i for i, c in enumerate(sorted(set(cats)))}
    y = np.array([cat_to_idx[c] for c in cats], dtype=np.int32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x.shape[1],)),
            tf.keras.layers.Dense(de, activation="relu"),
            tf.keras.layers.Dense(de),
            tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1)),
        ]
    )
    classifier = tf.keras.layers.Dense(len(cat_to_idx))
    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    x_tf = tf.convert_to_tensor(x)
    y_tf = tf.convert_to_tensor(y)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            emb = model(x_tf, training=True)
            logits = classifier(emb)
            loss = loss_fn(y_tf, logits)
            # lightweight triplet regularization
            triplet_terms = []
            for i, c in enumerate(cats):
                pos = next((j for j, cj in enumerate(cats) if cj == c and j != i), None)
                neg = next((j for j, cj in enumerate(cats) if cj != c), None)
                if pos is None or neg is None:
                    continue
                ap = tf.norm(emb[i] - emb[pos])
                an = tf.norm(emb[i] - emb[neg])
                triplet_terms.append(tf.maximum(0.0, ap - an + triplet_margin))
            if triplet_terms:
                loss = loss + tf.reduce_mean(triplet_terms)
        vars_all = model.trainable_variables + classifier.trainable_variables
        grads = tape.gradient(loss, vars_all)
        opt.apply_gradients(zip(grads, vars_all))

    fused = model(x_tf, training=False).numpy()
    image_p = tf.math.l2_normalize(image[:, :de], axis=-1).numpy()
    text_p = tf.math.l2_normalize(text[:, :de], axis=-1).numpy()
    cf_p = tf.math.l2_normalize(cf[:, :de], axis=-1).numpy()
    image_text = tf.math.l2_normalize((image_p + text_p) / 2.0, axis=-1).numpy()

    out = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "category_id": items["category_id"],
            "emb_image": [" ".join(map(str, v.tolist())) for v in image_p],
            "emb_text": [" ".join(map(str, v.tolist())) for v in text_p],
            "emb_cf": [" ".join(map(str, v.tolist())) for v in cf_p],
            "emb_image_text": [" ".join(map(str, v.tolist())) for v in image_text],
            "emb_image_text_cf": [" ".join(map(str, v.tolist())) for v in fused],
        }
    )
    out_csv = artifacts_dir / "item_embeddings.parquet"
    out.to_parquet(out_csv, index=False)
    model.save_weights(artifacts_dir / "embedding_model_tf.weights.h5")
    return {"embedding_csv": str(out_csv), "checkpoint": str(artifacts_dir / "embedding_model_tf.weights.h5")}


def train_fom_tf(
    embedding_csv: Path,
    canonical_dir: Path,
    processed_dir: Path,
    checkpoint_path: Path,
    emb_col: str,
    de: int,
    dm: int,
    layers: int,
    heads: int,
    epochs: int,
) -> Path:
    emb_df = pd.read_parquet(embedding_csv)
    item_ids = emb_df["item_id"].astype(str).tolist()
    item_to_idx = {x: i for i, x in enumerate(item_ids)}
    table = np.stack([_parse_vec(v) for v in emb_df[emb_col].astype(str)], axis=0)
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    manifest = read_json(processed_dir / "split_manifest.json")
    train_ids = set(manifest["fom"]["train_outfit_ids"])
    train = outfits[outfits["outfit_id"].isin(train_ids)]

    inp = tf.keras.layers.Input(shape=(None, de))
    x = tf.keras.layers.Dense(dm)(inp)
    for _ in range(layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, dm // heads))(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ff = tf.keras.layers.Dense(dm, activation="relu")(x)
        x = tf.keras.layers.LayerNormalization()(x + ff)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(len(item_ids))(pooled)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    xs = []
    ys = []
    max_len = 1
    for row in train.itertuples(index=False):
        seq = [x for x in _seq(row.item_seq) if x in item_to_idx]
        if len(seq) < 2:
            continue
        pos = len(seq) // 2
        y = item_to_idx[seq[pos]]
        ctx = [table[item_to_idx[s]] for i, s in enumerate(seq) if i != pos]
        max_len = max(max_len, len(ctx))
        xs.append(ctx)
        ys.append(y)
    if xs:
        pad = np.zeros((len(xs), max_len, de), dtype=np.float32)
        for i, ctx in enumerate(xs):
            pad[i, : len(ctx), :] = np.asarray(ctx, dtype=np.float32)
        model.fit(pad, np.asarray(ys, dtype=np.int32), epochs=epochs, batch_size=32, verbose=0)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint_path)
    return checkpoint_path


def train_pog_tf(
    embedding_csv: Path,
    canonical_dir: Path,
    processed_dir: Path,
    checkpoint_path: Path,
    emb_col: str,
    de: int,
    dm: int,
    layers: int,
    heads: int,
    epochs: int,
) -> Path:
    emb_df = pd.read_parquet(embedding_csv)
    item_ids = emb_df["item_id"].astype(str).tolist()
    idx = {x: i for i, x in enumerate(item_ids)}
    table = np.stack([_parse_vec(v) for v in emb_df[emb_col].astype(str)], axis=0)
    users = pd.read_parquet(canonical_dir / "user_sequences.parquet")
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    outfit_map = {r.outfit_id: _seq(r.item_seq) for r in outfits.itertuples(index=False)}
    manifest = read_json(processed_dir / "split_manifest.json")
    train_users = set(manifest["pog"]["train_user_ids"])
    data = users[users["user_id"].isin(train_users)]

    click_in = tf.keras.layers.Input(shape=(None, de))
    dec_in = tf.keras.layers.Input(shape=(None, de))
    c = tf.keras.layers.Dense(dm)(click_in)
    d = tf.keras.layers.Dense(dm)(dec_in)
    for _ in range(layers):
        c_attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, dm // heads))(c, c)
        c = tf.keras.layers.LayerNormalization()(c + c_attn)
        d_self = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, dm // heads))(d, d)
        d = tf.keras.layers.LayerNormalization()(d + d_self)
        d_cross = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, dm // heads))(d, c)
        d = tf.keras.layers.LayerNormalization()(d + d_cross)
    out = tf.keras.layers.Dense(len(item_ids))(d[:, -1, :])
    model = tf.keras.Model([click_in, dec_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    x_click, x_dec, y = [], [], []
    max_click, max_dec = 1, 1
    for row in data.itertuples(index=False):
        clicks = [x for x in _seq(row.click_seq) if x in idx]
        target = [x for x in outfit_map.get(getattr(row, "target_outfit_id", ""), []) if x in idx]
        if len(clicks) < 2 or len(target) < 2:
            continue
        click_vecs = [table[idx[c]] for c in clicks[-20:]]
        for step in range(1, len(target)):
            dec_vecs = [table[idx[c]] for c in target[:step]]
            x_click.append(click_vecs)
            x_dec.append(dec_vecs)
            y.append(idx[target[step]])
            max_click = max(max_click, len(click_vecs))
            max_dec = max(max_dec, len(dec_vecs))
    if x_click:
        click_pad = np.zeros((len(x_click), max_click, de), dtype=np.float32)
        dec_pad = np.zeros((len(x_dec), max_dec, de), dtype=np.float32)
        for i in range(len(x_click)):
            click_pad[i, : len(x_click[i]), :] = np.asarray(x_click[i], dtype=np.float32)
            dec_pad[i, : len(x_dec[i]), :] = np.asarray(x_dec[i], dtype=np.float32)
        model.fit([click_pad, dec_pad], np.asarray(y, dtype=np.int32), epochs=epochs, batch_size=32, verbose=0)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint_path)
    return checkpoint_path
