import argparse, os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from utils import LABEL2ID, ID2LABEL, plot_training_curves

class TweetDS(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = [LABEL2ID[l] for l in df["label"].tolist()]
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): 
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--model", type=str, default="bert-base-uncased")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        num_labels=3, 
        id2label=ID2LABEL, 
        label2id=LABEL2ID
    )

    # Data
    tr = pd.read_csv(args.train)
    va = pd.read_csv(args.val)
    trds = TweetDS(tr, tok, args.max_len)
    vads = TweetDS(va, tok, args.max_len)
    train_loader = DataLoader(trds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(vads, batch_size=args.batch_size, shuffle=False)

    # Optim & sched
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=total_steps)

    history = {"train_loss": [], "val_loss": []}
    best = float("inf")
    best_path = os.path.join(args.outdir, "best_model.pt")

    for ep in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tl = 0.0; n = 0
        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            bs = batch["input_ids"].size(0)
            tl += loss.item() * bs; n += bs
        tr_loss = tl / max(n, 1)

        # ---- Validate ----
        model.eval()
        vl = 0.0; vn = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep}/{args.epochs} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                bs = batch["input_ids"].size(0)
                vl += loss.item() * bs; vn += bs
        val_loss = vl / max(vn, 1)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        print(f"[epoch {ep}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        # ---- Save best ----
        if val_loss < best:
            best = val_loss
            # Save in HF format (dir) + a small pointer file
            model.save_pretrained(args.outdir)
            tok.save_pretrained(args.outdir)
            torch.save({"model_type": "bert", "model_name": args.model}, best_path)

        # Curves
        plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))

    print("[OK] BERT training complete. Best saved to", best_path)

if __name__ == "__main__":
    main()
