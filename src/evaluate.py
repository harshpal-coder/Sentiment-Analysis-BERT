import argparse, os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from typing import List

from utils import LABEL2ID, ID2LABEL, plot_confusion_matrix, plot_roc, save_metrics

# ---------------- BERT ---------------- #
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EvalDS(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = [LABEL2ID[l] for l in df["label"].tolist()]
        self.tok = tokenizer; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def is_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

def evaluate_bert(test_csv: str, ckpt_dir: str, outdir: str, max_len: int = 128, wordclouds: bool=False) -> None:
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)

    df = pd.read_csv(test_csv)
    ds = EvalDS(df, tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    y_true=[]; y_prob=[]
    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            y_prob.append(probs)
            y_true.extend(batch["labels"].cpu().numpy().tolist())
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = y_prob.argmax(axis=1)

    # === Dynamically adapt to classes present in y_true === #
    labels_present: List[int] = sorted(np.unique(y_true).tolist())
    target_names = [ID2LABEL.get(i, str(i)) for i in labels_present]

    report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=target_names,
        zero_division=0
    )
    print(report)

    # Confusion matrix with only present classes
    # Draw with utils (it uses full set); for compactness we keep as-is
    plot_confusion_matrix(y_true, y_pred, os.path.join(outdir, "confusion_matrix.png"))

    # ROC: average only over present classes
    oh = pd.get_dummies(y_true)
    for i in range(y_prob.shape[1]):
        if i not in oh.columns:
            oh[i] = 0
    oh = oh[sorted(oh.columns)]
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    aucs = []
    plt.figure(figsize=(5,4))
    for i in labels_present:
        fpr, tpr, _ = roc_curve(oh[i].values, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{ID2LABEL.get(i,str(i))} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (OvR)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=160); plt.close()
    macro_roc_auc = float(np.mean(aucs)) if aucs else 0.0

    save_metrics(report, macro_roc_auc, outdir)

    if wordclouds:
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            for cls, name in [(0,"negative"), (2,"positive")]:
                texts = " ".join([t for t,l in zip(df["text"].tolist(), y_pred.tolist()) if l==cls])
                if not texts.strip(): 
                    continue
                wc = WordCloud(width=800, height=400, background_color="white").generate(texts)
                plt.figure(figsize=(8,4)); plt.imshow(wc); plt.axis("off"); plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"wordcloud_{name}.png"), dpi=160); plt.close()
        except Exception as e:
            print("[WARN] Wordcloud generation skipped:", e)

    print("[OK] Evaluation complete. Outputs saved to", outdir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--checkpoint", required=True, help="Directory with BERT model files (config.json, tokenizer.json).")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--wordclouds", action="store_true")
    args = ap.parse_args()

    if not is_hf_dir(args.checkpoint):
        raise ValueError("For BERT evaluation, pass the model directory (must contain config.json), e.g., --checkpoint outputs")

    evaluate_bert(args.test, args.checkpoint, args.outdir, max_len=args.max_len, wordclouds=args.wordclouds)

if __name__ == "__main__":
    main()
