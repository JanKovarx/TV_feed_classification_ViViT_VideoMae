import torch
from torch import nn
from torch.utils.data import DataLoader
from vivit import ViViT
from dataset import VideoDataset, VideoStreamDataset
from train_vivit import evaluate
from utils.train_utils import *
import json
import matplotlib.pyplot as plt
import numpy as np
import os

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']
EN_CLASSES = ['studio', 'indoor', 'outdoor', 'divide', 'advertisement', 'trailer', 'graphics', 'entertainment']

def plot_confusion_matrix(confusion, class_names, save_path=None):
    # Seřadit podle abecedy
    sorted_indices = np.argsort(class_names)
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_confusion = confusion[np.ix_(sorted_indices, sorted_indices)]

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(sorted_confusion, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(sorted_class_names)))
    ax.set_yticks(np.arange(len(sorted_class_names)))
    ax.set_xticklabels(sorted_class_names, rotation=45, ha="right")
    ax.set_yticklabels(sorted_class_names)

    for i in range(sorted_confusion.shape[0]):
        for j in range(sorted_confusion.shape[1]):
            ax.text(j, i, str(sorted_confusion[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
        plt.close()

    return sorted_confusion, sorted_class_names

def plot_diagonal_histogram(confusion, class_names, save_path=None):
    diagonal = np.diag(confusion)
    totals = np.sum(confusion, axis=1)
    percentages = 100 * diagonal / (totals + 1e-8)

    # Seřadit podle počtu správných klasifikací (diagonála)
    sorted_indices = np.argsort(diagonal)
    diagonal_sorted = diagonal[sorted_indices]
    percentages_sorted = percentages[sorted_indices]
    class_names_sorted = [class_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(np.arange(len(class_names_sorted)), diagonal_sorted, color="skyblue")

    ax.set_xticks(np.arange(len(class_names_sorted)))
    ax.set_xticklabels(class_names_sorted, rotation=45, ha="right")
    ax.set_ylabel("Correct Predictions")


    for i, bar in enumerate(bars):
        count = int(diagonal_sorted[i])
        percent = percentages_sorted[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{count} ({percent:.1f}%)", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Histogram saved to {save_path}")
        plt.close()
    else:
        plt.show()
        
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    eval_config = config['evaluation']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViViT(model_config).to(device)
    model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    checkpoint = torch.load(eval_config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Loading dataset...')
    assert data_config['dataset_type'] in ['one_class', 'stream'], f"Dataset type {data_config['dataset_type']} not supported"
    if data_config['dataset_type'] == 'one_class':
        val_dataset = VideoDataset(
            data_config['test_meta_file'], CLASSES,
            load_from_json=data_config['test_json'],
            frame_sample_rate=data_config['frame_sample_rate'],
            min_sequence_length=data_config['min_sequence_length'],
            max_sequence_length=data_config['max_sequence_length'],
            num_threads=data_config['decord_num_threads'],
            normalize=data_config['normalize'],
        )
    elif data_config['dataset_type'] == 'stream':
        val_dataset = VideoStreamDataset(
            data_config['test_meta_file'], CLASSES,
            load_from_json=data_config['test_json'],
            frame_sample_rate=data_config['frame_sample_rate'],
            context_size=data_config['context_size'],
            overlap=data_config['overlap'],
            max_empty_frames=data_config['max_empty_frames'],
            num_threads=data_config['decord_num_threads'],
            normalize=data_config['normalize'],
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        drop_last=data_config['drop_last'],
        num_workers=data_config['num_workers']
    )

    print(f"Dataset '{data_config['dataset_type']}' successfully loaded.")

    loss_func = nn.CrossEntropyLoss()

    print('Evaluation started.')
    eval_loss, acc, confusion, precision, recall, f1, per_class_metrics, per_class_accuracy = evaluate(
        model, val_dataloader, loss_func, device
    )

    print(f"Eval loss: {eval_loss:.4f}, accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    print("Per-class metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name}: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}, Accuracy={per_class_accuracy[class_name]}")

    save_dir = "test_VIVIT"
    os.makedirs(save_dir, exist_ok=True)

    if args.verbose:
        sorted_confusion, sorted_class_names = plot_confusion_matrix(confusion, EN_CLASSES, os.path.join(save_dir, "confusion_eval.jpg"))
        plot_diagonal_histogram(sorted_confusion, sorted_class_names, os.path.join(save_dir, "diagonal_histogram.jpg"))

    score = {
        'eval_loss': eval_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': per_class_metrics,
        'per_class_accuracy': per_class_accuracy
    }

    with open(os.path.join(save_dir, "score.json"), 'w') as f:
        json.dump(score, f, indent=2)

    print("Confusion Matrix:\n", confusion)
