import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from transformers import VideoMAEForVideoClassification
from Vmae_dataset import VideoStreamDataset
from utils.train_utils import parse_args, load_config

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']
EN_CLASSES = ['studio', 'indoor', 'outdoor', 'divide', 'advertisement', 'trailer', 'graphics', 'entertainment']

def plot_confusion_matrix(cm, class_names, save_path=None):
    # Seřadíme třídy abecedně a získáme jejich nové indexy
    sorted_indices = sorted(range(len(class_names)), key=lambda i: class_names[i])
    sorted_class_names = [class_names[i] for i in sorted_indices]

    # Přeskládáme matici podle nového pořadí (řádky i sloupce)
    cm_sorted = cm[np.ix_(sorted_indices, sorted_indices)]

    # Odstraníme diagonálu
    cm_no_diag = cm_sorted.copy()
    np.fill_diagonal(cm_no_diag, 0)

    # Vykreslení
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm_no_diag, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(sorted_class_names)))
    ax.set_yticks(np.arange(len(sorted_class_names)))
    ax.set_xticklabels(sorted_class_names, rotation=45, ha="right")
    ax.set_yticklabels(sorted_class_names)

    for i in range(cm_no_diag.shape[0]):
        for j in range(cm_no_diag.shape[1]):
            ax.text(j, i, str(cm_no_diag[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    return save_path

def plot_diagonal_histogram(cm, class_names, save_path=None):
    diag_counts = np.diag(cm)
    total_counts = cm.sum(axis=1)
    percentages = (diag_counts / total_counts) * 100

    data = list(zip(class_names, diag_counts, percentages))
    data_sorted = sorted(data, key=lambda x: x[1])
    sorted_classes = [x[0] for x in data_sorted]
    sorted_counts = [x[1] for x in data_sorted]
    sorted_percentages = [x[2] for x in data_sorted]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_classes, sorted_counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Correct Predictions')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, count, percent in zip(bars, sorted_counts, sorted_percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(sorted_counts) * 0.01,
            f'{int(count)} ({percent:.1f}%)',
            ha='center', va='bottom', fontsize=9, color='black'
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Diagonal histogram saved to {save_path}")
    else:
        plt.show()

def compute_per_class_metrics(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        acc = (cm[i, i] / total) if total > 0 else 0.0
        per_class_accuracy[name] = round(acc, 4)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        class_names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4)
        } for i in range(len(class_names))
    }
    return cm, per_class, per_class_accuracy

def collate_videomae(batch):
    xs, ys = [], []
    for item in batch:
        xs.append(item[0])
        ys.append(item[1])
    x = torch.stack(xs, dim=0)             
    y = torch.tensor(ys, dtype=torch.long) 
    return {"pixel_values": x, "labels": y}

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs.logits, batch["labels"])
        total_loss += float(loss)
        n_batches += 1

        preds = outputs.logits.argmax(dim=1)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    avg_loss = total_loss / max(1, n_batches)
    accuracy = correct / max(1, total)
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    cm, per_class_metrics, per_class_accuracy = compute_per_class_metrics(y_true, y_pred, CLASSES)
    return avg_loss, accuracy, precision, recall, f1, cm, per_class_metrics, per_class_accuracy

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(CLASSES)
    id2label = {i: c for i, c in enumerate(CLASSES)}
    label2id = {c: i for i, c in enumerate(CLASSES)}

    model = VideoMAEForVideoClassification.from_pretrained(
        config['training'].get('hf_checkpoint', None),
        num_labels=num_classes, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    data_cfg = config['data']
    assert data_cfg['dataset_type'] == 'stream', "Only 'stream' dataset supported here"

    dataset = VideoStreamDataset(
        data_cfg['test_meta_file'], CLASSES,
        load_from_json=data_cfg['test_json'],
        frame_sample_rate=data_cfg['frame_sample_rate'],
        context_size=data_cfg['context_size'],
        overlap=data_cfg['overlap'],
        max_empty_frames=data_cfg['max_empty_frames'],
        num_threads=data_cfg['decord_num_threads'],
        normalize=data_cfg['normalize'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        drop_last=data_cfg['drop_last'],
        num_workers=data_cfg['num_workers'],
        collate_fn=collate_videomae,
        pin_memory=True
    )

    checkpoint_path = config['evaluation']['checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")

    print("Starting evaluation...")
    eval_loss, accuracy, precision, recall, f1, confusion, per_class_metrics, per_class_accuracy = evaluate(model, dataloader, device)

    print(f"Eval loss: {eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("Per-class metrics:")
    for cls in CLASSES:
        m = per_class_metrics[cls]
        acc = per_class_accuracy[cls]
        print(f"  {cls}: Precision={m['precision']}, Recall={m['recall']}, F1={m['f1']}, Accuracy={acc}")

    save_dir = "test_VideoMAE"
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(confusion, EN_CLASSES, cm_path)

    diag_hist_path = os.path.join(save_dir, "diagonal_histogram.png")
    plot_diagonal_histogram(confusion, EN_CLASSES, diag_hist_path)

    results = {
        'eval_loss': eval_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': per_class_metrics,
        'per_class_accuracy': per_class_accuracy
    }
    with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation finished.")
