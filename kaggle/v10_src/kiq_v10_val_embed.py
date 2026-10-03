
# ===================== V10-VR Milestones 2-3: VAL region proposals + matched-budget view embeddings
# Appended to the unchanged V5 shared code (load_photo_set / embed_batch / frozen DINOv2) and kiq_v10_regions.py.
# Opens VAL only. TEST is never loaded. No labels are used here; scoring happens locally on frozen Adapter E.
import platform


def run_v10_val_embed() -> None:
    t_start = time.time()
    val, root = load_photo_set({"val"})
    assert all(i["split"] == "val" for i in val)
    status(f"protocol {PROTOCOL['protocol_id']} sha={protocol_sha()} N={PROTOCOL['N']}")
    registry = {}
    sources = {}
    for cls in (FasterRCNNSource, OWLv2Source):
        try:
            src = cls(DEV)
            sources[src.name] = src
            registry[src.name] = {**src.info, "evaluated": True, "reason": None}
        except Exception as e:  # fail loudly for this model; no substitution
            registry[cls.name] = {"model": cls.name, "weights_available": False, "evaluated": False,
                                  "status": "BLOCKED", "reason": f"{type(e).__name__}: {e}"}
            status(f"BLOCKED {cls.name}: {type(e).__name__}: {e}")
    registry["sam3"] = sam3_registry()
    if not sources:
        write_json("v10_region_model_registry.json", registry)
        raise SystemExit("IMPLEMENTATION_BLOCKED: no proposal source could be loaded")
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    per_image, emb_rows, runtime = [], [], {k: [] for k in sources}
    t_embed = 0.0
    for n, it in enumerate(val):
        img = Image.open(root / it["file"]).convert("RGB")
        W, H = img.size
        props = {}
        for k, src in sources.items():
            props[k], dt = timed(src, img)
            runtime[k].append(dt)
        views, arms, padding = build_views(W, H, it["sha256"], props)
        t0 = time.perf_counter()
        E = embed_batch([crop(img, v["crop_box"]) for v in views])
        t_embed += time.perf_counter() - t0
        base = len(emb_rows)
        emb_rows.extend(E)
        per_image.append({"sha256": it["sha256"], "openverse_id": it["openverse_id"], "W": W, "H": H,
                          "emb_offset": base, "views": views, "arms": arms, "padding": padding, "proposals": props})
        if n % 25 == 0:
            status(f"val {n}/{len(val)} views={len(views)}")
    X = np.stack(emb_rows).astype(np.float32)
    np.save(WORK / "v10_val_view_dinov2.npy", X)
    emb_sha = sha_bytes((WORK / "v10_val_view_dinov2.npy").read_bytes())
    for k, src in sources.items():
        rt = np.array(runtime[k])
        registry[k]["runtime_s_per_image"] = {"mean": round(float(rt.mean()), 4), "median": round(float(np.median(rt)), 4)}
        registry[k]["raw_duplicate_rate_iou0.5_mean"] = round(float(np.mean(src.raw_dup)), 4)
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "torchvision": __import__("torchvision").__version__,
           "transformers": __import__("transformers").__version__,
           "cuda": torch.version.cuda, "device": str(DEV),
           "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
           "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if DEV.type == "cuda" else None,
           "dinov2_embed_s_total": round(t_embed, 2), "wall_s": round(time.time() - t_start, 1)}
    reg_sha = write_json("v10_region_model_registry.json", registry)
    out_sha = write_json("v10_val_views.json", {
        "protocol": PROTOCOL, "protocol_sha256": protocol_sha(), "photo_manifest_sha256": PHOTO_MANIFEST_SHA,
        "split": "val", "test_opened": False, "backbone": "DINOv2:dinov2_vitb14 (frozen, x_norm_clstoken, L2)",
        "embedding_file": "v10_val_view_dinov2.npy", "embedding_sha256": emb_sha, "n_images": len(val),
        "n_views": int(X.shape[0]), "environment": env, "registry_sha256": reg_sha, "images": per_image})
    status(f"DONE v10_val_embed images={len(val)} views={X.shape[0]} emb_sha={emb_sha} views_sha={out_sha}")


if __name__ == "__main__":
    status(f"START mode={MODE} device={DEV}")
    run_v10_val_embed()
