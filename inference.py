#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from vit.vit.vit_2_cls import VisionTransformer as VisionTransformer_2_cls


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model():
    model = VisionTransformer_2_cls(
        img_size=112,
        patch_size=8,
        num_classes=512,
        embed_dim=512,
        depth=24,
        mlp_ratio=3,
        num_heads=16,
        drop_path_rate=0.1,
        norm_layer="ln",
        mask_ratio=0,
    )
    return model


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        elif new_k.startswith("net."):
            new_k = new_k[len("net."):]
        elif new_k.startswith("module."):
            new_k = new_k[len("module."):]

        if new_k.startswith("head."):
            continue

        new_state_dict[new_k] = v

    result = model.load_state_dict(new_state_dict, strict=False)
    print("[LOAD CKPT]", result)


def collect_images(image_dir):
    image_dir = Path(image_dir)
    image_paths = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(image_paths)


def get_transform(img_size=112):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


@torch.no_grad()
def extract_features(model, image_paths, device, batch_size):
    model.eval()
    model.to(device)

    transform = get_transform(112)

    all_features = []
    valid_paths = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]

        imgs = []
        cur_paths = []

        for p in batch_paths:
            try:
                imgs.append(load_image(p, transform))
                cur_paths.append(p)
            except Exception as e:
                print(f"[WARN] failed to load image {p}: {e}")

        if not imgs:
            continue

        x = torch.stack(imgs, dim=0).to(device)

        out = model(x)

        # Ours ViT-based:
        # emb_main, norm_main, emb_branch, norm_branch, feat
        if isinstance(out, (tuple, list)) and len(out) >= 5:
            emb_main, norm_main, emb_branch, norm_branch, feat_raw = out
            feat = emb_branch
        else:
            feat = out

        feat = torch.nn.functional.normalize(feat.float(), dim=1)

        all_features.append(feat.cpu())
        valid_paths.extend(cur_paths)

        print(f"[Infer] {min(start + batch_size, len(image_paths))}/{len(image_paths)}")

    if len(all_features) == 0:
        raise RuntimeError("No valid images were loaded.")

    features = torch.cat(all_features, dim=0)
    return features, valid_paths


def save_topk_results(features, image_paths, save_dir, top_k):
    os.makedirs(save_dir, exist_ok=True)

    image_paths = np.asarray([str(p) for p in image_paths])

    features = features.detach().float()
    features = torch.nn.functional.normalize(features, dim=1)

    print("[INFO] computing similarity matrix...")
    sim_matrix = features @ features.T
    sim_matrix = sim_matrix.cpu().numpy()

    summary = []
    num_images = len(image_paths)

    for i in range(num_images):
        query_path = image_paths[i]
        query_name = os.path.basename(query_path)

        sims = sim_matrix[i].copy()
        sims[i] = -999.0

        for j in range(num_images):
            if i != j and os.path.basename(image_paths[j]) == query_name:
                sims[j] = -999.0

        topk_indices = np.argsort(-sims)[:top_k]

        query_save_dir = os.path.join(
            save_dir,
            f"query_{i:05d}_{Path(query_name).stem}"
        )
        os.makedirs(query_save_dir, exist_ok=True)

        shutil.copy(query_path, os.path.join(query_save_dir, "query.jpg"))

        query_info = {
            "query_index": int(i),
            "query_path": query_path,
            "topk": [],
        }

        for rank, j in enumerate(topk_indices, start=1):
            ret_path = image_paths[j]
            ret_name = os.path.basename(ret_path)
            sim = float(sims[j])

            save_name = f"top{rank}_sim_{sim:.4f}_{ret_name}"
            shutil.copy(ret_path, os.path.join(query_save_dir, save_name))

            query_info["topk"].append({
                "rank": rank,
                "index": int(j),
                "path": ret_path,
                "similarity": sim,
            })

        summary.append(query_info)

        if i % 100 == 0:
            print(f"[TopK] saved {i}/{num_images}")

    json_path = os.path.join(save_dir, f"top{top_k}_summary.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved results to: {save_dir}")
    print(f"[DONE] summary json: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./topk_results")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[WARN] CUDA not available, using CPU.")
        args.device = "cpu"

    image_paths = collect_images(args.image_dir)
    print(f"[INFO] found {len(image_paths)} images")

    if len(image_paths) == 0:
        return

    model = build_model()
    load_checkpoint(model, args.ckpt_path)

    features, valid_paths = extract_features(
        model=model,
        image_paths=image_paths,
        device=args.device,
        batch_size=args.batch_size,
    )

    save_topk_results(
        features=features,
        image_paths=valid_paths,
        save_dir=args.save_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
