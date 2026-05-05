"""Generate per-drug LLM embeddings for the 8981 drugs in the L1000 SDST set.

Mirrors PertDiT upstream preprocessing.ipynb:
  SMILES → MolT5 (molt5-large-smiles2caption, beam=5, max_len=512) → caption
  caption → BioLinkBERT (BioLinkBERT-large, last_hidden_state) → Tensor[L, 1024]

Outputs:
  data/bench_drug_emb.pkl   — dict[int(pert_idx) → Tensor(L, 1024)] + 'negative_ctrl'
  data/bench_dose_emb.pt    — Tensor(11, 1024) for dose=10.0

Verification (via --verify): on 3 known drugs the regenerated embeddings
numerically match the upstream-shipped embeddings to within max_diff < 1e-4.
"""
import os
from pathlib import Path
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

XPERT_ROOT = str(Path(__file__).resolve().parents[3])
IDX2SMI_PATH = os.path.join(XPERT_ROOT, "data/XPert/processed_data/all_drugs_idx2smi_8981.npy")
ORIGINAL_EMB_PATH = os.path.join(
    XPERT_ROOT, "data", "PertDiT", "extracted", "pert_smiles_emb.pkl")
OUT_DIR = os.path.join(XPERT_ROOT, "data", "PertDiT", "bench_drug_emb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: 'cuda:0' if CUDA available, else 'cpu')")
    parser.add_argument("--verify", action="store_true",
                        help="After generation, verify 3 drugs against original embeddings")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ----------------------------------------------------------------
    # 1. Load drug SMILES
    # ----------------------------------------------------------------
    idx2smi = np.load(IDX2SMI_PATH, allow_pickle=True).item()
    all_smiles = list(set(idx2smi.values()))  # unique SMILES
    smi2indices = {}  # SMILES → list of pert_idx (some drugs share SMILES)
    for idx, smi in idx2smi.items():
        smi2indices.setdefault(smi, []).append(idx)
    print(f"Loaded {len(idx2smi)} drugs, {len(all_smiles)} unique SMILES")

    # ----------------------------------------------------------------
    # 2. MolT5: SMILES → caption
    #    Exact same as upstream preprocessing.ipynb cell-7, cell-8.
    #    Cache check runs FIRST so a CPU-only node with a complete
    #    _caption_cache.pkl can finish the rest of this script (BioLinkBERT
    #    is hardcoded to CPU below) without ever loading MolT5.
    # ----------------------------------------------------------------
    caption_cache = os.path.join(OUT_DIR, "_caption_cache.pkl")
    smiles_caption = {}
    if os.path.exists(caption_cache):
        import pickle
        with open(caption_cache, 'rb') as f:
            smiles_caption = pickle.load(f)
        print(f"Loaded {len(smiles_caption)} cached captions from {caption_cache}")

    remaining = [s for s in all_smiles if s not in smiles_caption]
    t0 = time.time()

    if remaining:
        # Generating captions requires CUDA — fail fast on CPU-only nodes.
        if not torch.cuda.is_available():
            sys.exit(
                f"ERROR: {len(remaining)} SMILES still need MolT5 captions, "
                "which require a CUDA GPU. Either run this script on a GPU "
                "node, or copy a complete _caption_cache.pkl into "
                f"{OUT_DIR}/ from a previous GPU run."
            )
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        print("Loading MolT5 (laituan245/molt5-large-smiles2caption)...")
        tokenizer_t5 = T5Tokenizer.from_pretrained(
            "laituan245/molt5-large-smiles2caption", model_max_length=512)
        model_t5 = T5ForConditionalGeneration.from_pretrained(
            "laituan245/molt5-large-smiles2caption")
        model_t5 = model_t5.to(device).eval()

        print(f"Generating captions for {len(remaining)} SMILES ({len(smiles_caption)} cached)...")
        with torch.no_grad():
            for i, smile in enumerate(tqdm(remaining)):
                input_ids = tokenizer_t5(smile, return_tensors="pt").input_ids.to(device)
                outputs = model_t5.generate(input_ids, num_beams=5, max_length=512)
                smiles_caption[smile] = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
                # Save checkpoint every 500 drugs
                if (i + 1) % 500 == 0:
                    import pickle
                    with open(caption_cache, 'wb') as f:
                        pickle.dump(smiles_caption, f)
                    print(f"  Checkpoint: {len(smiles_caption)} captions saved")
        # Final save
        import pickle
        with open(caption_cache, 'wb') as f:
            pickle.dump(smiles_caption, f)
        t1 = time.time()
        print(f"Caption generation done in {t1-t0:.1f}s ({(t1-t0)/len(remaining):.2f}s/drug)")

        # Free MolT5 GPU memory
        del model_t5
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"All {len(smiles_caption)} captions already cached; skipping MolT5 (no GPU needed for the rest of this script).")

    # ----------------------------------------------------------------
    # 3. BioLinkBERT: caption → embedding
    #    Exact same as upstream preprocessing.ipynb cell-9, cell-10
    #    Note: original runs on CPU
    # ----------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModel

    print("Loading BioLinkBERT (michiyasunaga/BioLinkBERT-large)...")
    tokenizer_bert = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    model_bert = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
    # Run on CPU, same as upstream preprocessing.ipynb cell-10
    bert_device = torch.device("cpu")
    model_bert = model_bert.to(bert_device).eval()

    smiles_text_emb = {}
    print(f"Generating BioLinkBERT embeddings for {len(smiles_caption)} captions...")
    t2 = time.time()
    with torch.no_grad():
        for smi, caption in tqdm(smiles_caption.items()):
            inputs = tokenizer_bert(caption, return_tensors="pt").to(bert_device)
            outputs = model_bert(**inputs)
            smiles_text_emb[smi] = outputs.last_hidden_state.detach().squeeze(0).cpu()
    t3 = time.time()
    print(f"Embedding generation done in {t3-t2:.1f}s")

    # ----------------------------------------------------------------
    # 4. Build pert_idx-keyed dict + negative_ctrl
    # ----------------------------------------------------------------
    bench_drug_emb = {}
    for idx in sorted(idx2smi.keys()):
        smi = idx2smi[idx]
        bench_drug_emb[idx] = smiles_text_emb[smi]

    # negative_ctrl: copy from original (special CFG token, not drug-specific)
    original_emb = torch.load(ORIGINAL_EMB_PATH, map_location="cpu")
    bench_drug_emb["negative_ctrl"] = original_emb["negative_ctrl"]
    print(f"negative_ctrl shape: {bench_drug_emb['negative_ctrl'].shape}")

    # ----------------------------------------------------------------
    # 5. Dose embedding
    #    Exact same as upstream preprocessing.ipynb cell-11, cell-12
    # ----------------------------------------------------------------
    dose_prompt = "The dosage is 10.0 micromoles."
    with torch.no_grad():
        inputs = tokenizer_bert(dose_prompt, return_tensors="pt").to(bert_device)
        outputs = model_bert(**inputs)
        dose_emb = outputs.last_hidden_state.detach().squeeze(0).cpu()
    print(f"Dose embedding shape: {dose_emb.shape} for prompt: \"{dose_prompt}\"")

    # ----------------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------------
    drug_emb_path = os.path.join(OUT_DIR, "bench_drug_emb.pkl")
    dose_emb_path = os.path.join(OUT_DIR, "bench_dose_emb.pt")

    torch.save(bench_drug_emb, drug_emb_path)
    torch.save(dose_emb, dose_emb_path)
    print(f"Saved: {drug_emb_path} ({len(bench_drug_emb)-1} drugs + negative_ctrl)")
    print(f"Saved: {dose_emb_path}")

    # ----------------------------------------------------------------
    # 7. Sanity check
    # ----------------------------------------------------------------
    print("\n--- Sanity Check ---")
    # Check all pert_idx present
    missing = [i for i in range(len(idx2smi)) if i not in bench_drug_emb]
    print(f"Missing pert_idx: {len(missing)}")
    assert len(missing) == 0, f"Missing: {missing[:10]}"

    # Check shapes
    shapes = [bench_drug_emb[i].shape for i in range(min(100, len(idx2smi)))]
    print(f"First 100 shapes: all (L, 1024)? {all(s[1]==1024 for s in shapes)}")
    seq_lens = [s[0] for s in shapes]
    print(f"  Seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}")

    # ----------------------------------------------------------------
    # 8. Optional: verify against original embeddings
    # ----------------------------------------------------------------
    if args.verify:
        print("\n--- Verification against original PertDiT embeddings ---")
        # Find 3 drugs with direct SMILES match in original embeddings
        pertdit_keys = {k for k in original_emb.keys() if k != "negative_ctrl"}
        verified = 0
        for idx in sorted(idx2smi.keys()):
            smi = idx2smi[idx]
            if smi in pertdit_keys:
                pre = original_emb[smi]
                regen = bench_drug_emb[idx]
                if pre.shape == regen.shape:
                    max_diff = (pre - regen).abs().max().item()
                    print(f"  idx={idx}: shape={pre.shape}, max_diff={max_diff:.2e}, "
                          f"OK={max_diff < 1e-4}")
                else:
                    print(f"  idx={idx}: SHAPE MISMATCH pre={pre.shape} vs regen={regen.shape}")
                verified += 1
                if verified >= 5:
                    break
        # Verify dose embedding
        original_dose = torch.load(
            os.path.join(XPERT_ROOT,
                         "data/PertDiT/extracted/dosage_prompt_emb_lincs.pkl"),
            map_location="cpu")
        dose_pre = original_dose[np.float64(10.0)]
        dose_diff = (dose_pre - dose_emb).abs().max().item()
        print(f"  dose=10.0: shape={dose_pre.shape}, max_diff={dose_diff:.2e}, OK={dose_diff < 1e-4}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("[DONE]")


if __name__ == "__main__":
    main()
