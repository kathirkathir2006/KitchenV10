
# ===================== V7 one-shot TEST embedding (appended to unchanged kiq_v5_stage.py shared code)
# Opens TEST only after V7 is frozen (docs/v7/v7_specialists_frozen_config.json sha c29079096ed2...).
# Writes frozen DINOv2 view-0 embeddings + item ids; no model, no labels used for anything here.
V7_CONFIG_SHA = "c29079096ed20ecd78d90bf0b29d86201241f705757d152f1dcef915c1b297b5"


def run_v7_test_embed() -> None:
    test, root = load_photo_set({"test"})
    X = embed_items(test, root, views=1)[:, 0]
    np.save(WORK / "test_dinov2_view0.npy", X)
    write_json("test_items.json", {"v7_config_sha256": V7_CONFIG_SHA, "photo_manifest_sha256": PHOTO_MANIFEST_SHA,
                                   "backbone": "DINOv2:dinov2_vitb14 (frozen)", "n": len(test),
                                   "sha256": [i["sha256"] for i in test],
                                   "embedding_sha256": sha_bytes((WORK / "test_dinov2_view0.npy").read_bytes())})
    status(f"DONE v7_test_embed n={len(test)}")


if __name__ == "__main__":
    status(f"START mode={MODE} device={DEV}")
    run_v7_test_embed()
