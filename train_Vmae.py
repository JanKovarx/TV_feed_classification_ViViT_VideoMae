import os
import json
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_fscore_support
)

# tvoje utility (argumenty, config, W&B init)
from utils.train_utils import parse_args, load_config, init_wandb

# dataset – tvoje třídy
from Vmae_dataset import VideoStreamDataset  # používáme stream dataset (vrací x,y)

# HF VideoMAE
from transformers import VideoMAEForVideoClassification

# ====== DATASET TŘÍDY ======
CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


# ---------------- Vizualizace / metriky ----------------
def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def compute_per_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # per-class accuracy
    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        acc = (cm[i, i] / total) if total > 0 else 0.0
        per_class_accuracy[name] = round(acc, 4)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4)
        } for i in range(len(class_names))
    }
    return cm, per_class, per_class_accuracy


# ---------------- Collate pro VideoMAE ----------------
def collate_videomae(batch):
    xs, ys = [], []
    for item in batch:
        # item = (x, y)  ;  x: (T, C, H, W)
        xs.append(item[0])
        ys.append(item[1])
    x = torch.stack(xs, dim=0)             # (B, T, C, H, W)
    y = torch.tensor(ys, dtype=torch.long) # (B,)
    return {"pixel_values": x, "labels": y}


# ---------------- Eval loop ----------------
@torch.no_grad()
def evaluate(model, data_loader, device, criterion=None, use_hf_loss=True, log_to_wandb=False, step=None, ckpt_dir=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    for _, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)  # logits: (B, num_classes)

        loss = outputs.loss if use_hf_loss else criterion(outputs.logits, batch["labels"])
        total_loss += float(loss)
        n_batches += 1

        preds = outputs.logits.argmax(dim=1)
        y_true.extend(batch["labels"].cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    avg_loss = total_loss / max(1, n_batches)
    acc = correct / max(1, total)

    cm, per_class, per_acc = compute_per_class_metrics(y_true, y_pred, CLASSES)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # W&B log včetně obrázku matic
    if log_to_wandb:
        import wandb
        wandb.log({
            "eval/loss": avg_loss,
            "eval/accuracy": acc,
            "eval/precision": precision,
            "eval/recall": recall,
            "eval/f1": f1
        }, step=step, commit=False)

        # Per-class
        for cname, m in per_class.items():
            wandb.log({
                f"eval/per_class/{cname}/precision": m["precision"],
                f"eval/per_class/{cname}/recall": m["recall"],
                f"eval/per_class/{cname}/f1": m["f1"],
                f"eval/per_class/{cname}/accuracy": per_acc[cname],
            }, step=step, commit=False)

        # Confusion matrix jako obrázek
        if ckpt_dir is not None:
            plot_path = plot_confusion_matrix(cm, CLASSES, os.path.join(ckpt_dir, f"confusion_{step}.png"))
            if plot_path and os.path.exists(plot_path):
                wandb.log({"eval/confusion_matrix": wandb.Image(plot_path)}, step=step, commit=False)

    return avg_loss, acc, precision, recall, f1, cm, per_class, per_acc


# ---------------- Train epoch ----------------
def train_epoch(
    epoch, model, optimizer, lr_sched, grad_accum_steps,
    train_loader, val_loader, device,
    criterion=None, use_hf_loss=True,
    log_step=100, eval_step=-1,
    checkpoint_save_dir=None, metric_name='f1',
    report_to='none'
):
    assert metric_name in ['loss', 'accuracy', 'f1']
    model.train()

    # připrav W&B
    use_wandb = (report_to == 'wandb')
    if use_wandb:
        import wandb

    # checkpoint JSON správa
    ckpt_json_path = os.path.join(checkpoint_save_dir, 'checkpoints.json')
    with open(ckpt_json_path, 'r') as f:
        ckpt_meta = json.load(f)
    best_list = ckpt_meta['3-best']
    ckpt_meta['metric'] = metric_name
    metric_sign = 1 if metric_name == 'loss' else -1
    metric_val = 0 if metric_sign == -1 else 100

    def update_best(checkpoint_path, val):
        ckpt_meta['all'][os.path.basename(checkpoint_path)] = val
        best_list.append((checkpoint_path, val))
        best_list.sort(key=lambda x: x[1] * metric_sign)
        ckpt_meta['best_checkpoint'] = {os.path.basename(best_list[0][0]): best_list[0][1]}

        if len(best_list) > 3:
            rm = best_list.pop(3)[0]
            if rm not in [c[0] for c in best_list]:
                if rm != checkpoint_path and os.path.exists(rm):
                    os.remove(rm)

        last = ckpt_meta['latest_checkpoint']
        if last is not None and last not in [c[0] for c in best_list]:
            if last != checkpoint_path and os.path.exists(last):
                os.remove(last)

        ckpt_meta['latest_checkpoint'] = checkpoint_path
        ckpt_meta['3-best'] = best_list
        with open(ckpt_json_path, 'w') as f:
            json.dump(ckpt_meta, f, indent=2)

    start_time = time.time()
    total_samples = len(train_loader.dataset)

    for it, batch in enumerate(train_loader):
        if it % grad_accum_steps == 0:
            optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if use_hf_loss else criterion(outputs.logits, batch["labels"])
        (loss / grad_accum_steps).backward()

        if ((it + 1) % grad_accum_steps == 0) or (it + 1 == len(train_loader)):
            optimizer.step()
            lr_sched.step()

        # logging
        if lr_sched.last_epoch % log_step == 0:
            if use_wandb:
                wandb.log({
                    "train/loss": float(loss.item()),
                    "train/epoch": epoch,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/time_per_iteration": (time.time() - start_time) / max(1, log_step),
                }, step=lr_sched.last_epoch, commit=True)

            print('[' + '{:5}'.format(it * batch["labels"].size(0)) + '/' + '{:5}'.format(total_samples) +
                  f' ({100 * it / len(train_loader):3.0f}%)]  Loss: {loss.item():6.4f}')
            start_time = time.time()

        # průběžná evaluace
        if lr_sched.last_epoch % eval_step == 0 and eval_step != -1:
            val_loss, acc, prec, rec, f1, *_ = evaluate(
                model, val_loader, device,
                criterion=criterion, use_hf_loss=use_hf_loss,
                log_to_wandb=use_wandb, step=lr_sched.last_epoch, ckpt_dir=checkpoint_save_dir
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/accuracy": acc,
                    "eval/precision": prec,
                    "eval/recall": rec,
                    "eval/f1": f1
                }, step=lr_sched.last_epoch, commit=False)

            metric_val = val_loss if metric_name == 'loss' else acc if metric_name == 'accuracy' else f1

            # checkpoint v průběhu
            ckpt_path = os.path.join(checkpoint_save_dir, f'model_{epoch}-{it}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_sched.state_dict(),
                'loss': float(val_loss),
            }, ckpt_path)
            update_best(ckpt_path, metric_val)

    # epoch-end eval + save
    val_loss, acc, prec, rec, f1, *_ = evaluate(
        model, val_loader, device,
        criterion=criterion, use_hf_loss=use_hf_loss,
        log_to_wandb=use_wandb, step=lr_sched.last_epoch, ckpt_dir=checkpoint_save_dir
    )
    print(f'[Epoch {epoch}] Eval loss: {val_loss:.4f}, acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}')
    metric_val = val_loss if metric_name == 'loss' else acc if metric_name == 'accuracy' else f1

    ckpt_path = os.path.join(checkpoint_save_dir, f'model_{epoch}-final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_sched.state_dict(),
        'loss': float(val_loss),
    }, ckpt_path)
    update_best(ckpt_path, metric_val)


# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['training']

    # W&B
    project_name = 'ViViT'
    if train_cfg['report_to'] == 'wandb':
        init_wandb(project_name, config, name=train_cfg.get('model_name', 'ViViT') + '-' + datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASSES)

    # --- Model (HF) ---
    hf_ckpt = train_cfg.get('hf_checkpoint', None)
    id2label = {i: c for i, c in enumerate(CLASSES)}
    label2id = {c: i for i, c in enumerate(CLASSES)}
    if hf_ckpt:
        print(f'Loading HF checkpoint: {hf_ckpt}')
        model = VideoMAEForVideoClassification.from_pretrained(
            hf_ckpt, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        )
    else:
        print('Initializing VideoMAE from scratch.')
        model = VideoMAEForVideoClassification.from_config(
            VideoMAEForVideoClassification.config_class(num_labels=num_classes, id2label=id2label, label2id=label2id)
        )
    model = model.to(device)

    # --- Datasety (3 JSONy) ---
    print('Loading datasets...')
    train_ds = VideoStreamDataset(
        data_cfg['train_meta_file'], CLASSES,
        load_from_json=data_cfg['train_json'],
        frame_sample_rate=data_cfg['frame_sample_rate'],
        context_size=data_cfg['context_size'],
        overlap=data_cfg['overlap'],
        max_empty_frames=data_cfg['max_empty_frames'],
        num_threads=data_cfg['decord_num_threads'],
        normalize=data_cfg['normalize'],
    )
    val_ds = VideoStreamDataset(
        data_cfg['val_meta_file'], CLASSES,
        load_from_json=data_cfg['val_json'],
        frame_sample_rate=data_cfg['frame_sample_rate'],
        context_size=data_cfg['context_size'],
        overlap=data_cfg['overlap'],
        max_empty_frames=data_cfg['max_empty_frames'],
        num_threads=data_cfg['decord_num_threads'],
        normalize=data_cfg['normalize'],
    )
    test_ds = VideoStreamDataset(
        data_cfg['test_meta_file'], CLASSES,
        load_from_json=data_cfg['test_json'],
        frame_sample_rate=data_cfg['frame_sample_rate'],
        context_size=data_cfg['context_size'],
        overlap=data_cfg['overlap'],
        max_empty_frames=data_cfg['max_empty_frames'],
        num_threads=data_cfg['decord_num_threads'],
        normalize=data_cfg['normalize'],
    )

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg['batch_size'], shuffle=data_cfg['shuffle'],
        drop_last=data_cfg['drop_last'], num_workers=data_cfg['num_workers'],
        collate_fn=collate_videomae, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg['batch_size'], shuffle=False,
        drop_last=data_cfg['drop_last'], num_workers=data_cfg['num_workers'],
        collate_fn=collate_videomae, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=data_cfg['batch_size'], shuffle=False,
        drop_last=False, num_workers=data_cfg['num_workers'],
        collate_fn=collate_videomae, pin_memory=True
    )

    # --- Loss/optimizer/scheduler ---
    use_hf_loss = True
    if train_cfg['loss'] in ['cross_entropy', 'crossentropy']:
        label_smoothing = train_cfg.get('label_smoothing', 0.0)
        if label_smoothing and label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            use_hf_loss = False
        else:
            criterion = nn.CrossEntropyLoss()  # nepoužito, držíme HF loss
            use_hf_loss = True
    else:
        raise ValueError('Only cross-entropy is supported here. (SeeSaw lze doplnit.)')

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate']) \
        if train_cfg['optimizer'] == 'adam' else \
        torch.optim.SGD(model.parameters(), lr=train_cfg['learning_rate'], momentum=0.9)

    steps_per_epoch = len(train_loader)
    if train_cfg['lr_scheduler'] == 'cosine':
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, (train_cfg['epochs'] - train_cfg['warmup_epochs']) * steps_per_epoch)
        )
    elif train_cfg['lr_scheduler'] == 'constant':
        lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    else:  # 'multistep'
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [25 * steps_per_epoch, 50 * steps_per_epoch, 75 * steps_per_epoch]
        )

    if train_cfg['warmup_epochs'] > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=int(train_cfg['warmup_epochs'] * steps_per_epoch)
        )
        lr_sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, lr_sched],
            milestones=[int(train_cfg['warmup_epochs'] * steps_per_epoch)]
        )

    # --- checkpointy (adresář + meta) ---
    model_name = train_cfg.get('model_name', 'VideoMAE') + '-' + datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    ckpt_dir = os.path.join(train_cfg['checkpoint_save_dir'], model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'checkpoints.json'), 'w') as f:
        json.dump({'metric': None, 'best_checkpoint': {}, 'latest_checkpoint': None, '3-best': [], 'all': {}}, f)

    # --- (volitelně) načti checkpoint ---
    start_epoch = 0
    if os.path.exists(train_cfg['load_from_checkpoint']):
        checkpoint = torch.load(train_cfg['load_from_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if not train_cfg.get('load_only_weights', False):
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                lr_sched.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
        print(f'Loaded from {train_cfg["load_from_checkpoint"]}')

    # --- trénování ---
    for e in range(start_epoch, start_epoch + train_cfg['epochs']):
        print('Epoch:', e)
        train_epoch(
            e, model, optimizer, lr_sched,
            grad_accum_steps=train_cfg['gradient_accumulation_steps'],
            train_loader=train_loader, val_loader=val_loader, device=device,
            criterion=criterion, use_hf_loss=use_hf_loss,
            log_step=train_cfg['log_step'], eval_step=train_cfg['eval_step'],
            checkpoint_save_dir=ckpt_dir, metric_name=train_cfg.get('eval_metric', 'f1'),
            report_to=train_cfg['report_to']
        )

    # --- final test ---
    print('Final evaluation on TEST set:')
    test_loss, acc, prec, rec, f1, cm, per_class, per_acc = evaluate(
        model, test_loader, device,
        criterion=criterion, use_hf_loss=use_hf_loss,
        log_to_wandb=(train_cfg['report_to']=='wandb'),
        step=None, ckpt_dir=ckpt_dir
    )
    print(f'Test loss: {test_loss:.4f}, acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}')

    # ulož model
    final_path = os.path.join(ckpt_dir, 'model_final.pt')
    torch.save(model.state_dict(), final_path)
    print('Model saved to', final_path)
