import os
import time
import argparse

import torch
seed_num = 17
torch.manual_seed(seed_num)
import torch.nn.functional as F

import itertools as it
import matplotlib.pyplot as plt
import numpy as np

# Import validation and metric evaluation tools
from sklearn.metrics import confusion_matrix, roc_curve

from model import RAGCN
from utils import accuracy, load_data_medical, encode_onehot_torch, class_f1, auc_score
from model import my_sigmoid


def train():
    # Configure architectures for networks
    struc_D = {'dropout': args.dropout_D, 'wd': 5e-4, 'lr': args.lr_D, 'nhid': structure_D}
    struc_Ws = n_ws*[{'dropout': args.dropout_W,  'wd': 5e-4, 'lr': args.lr_W, 'nhid': structure_W}]
    
    # Initialize optimization containers
    stats = dict()
    act = my_sigmoid
    
    # Initialize the RA-GCN model
    model = RAGCN(adj=adj, features=features, nclass=nclass, struc_D=struc_D, struc_Ws=struc_Ws, n_ws=n_ws,
                  weighing_output_dim=1, act=act, gamma=args.gamma)

    if use_cuda:
        model.cuda()

    # Track optimal metrics and runtime states
    max_val = dict()
    max_val['f1Macro_val'] = 0
    best_test_probs = None

    # Begin core model optimization loops
    for epoch in range(args.epochs):
        model.train()
        
        # Train optimization components jointly
        model.run_both(epoch_for_D=args.epoch_D, epoch_for_W=args.epoch_W, labels_one_hot=labels_one_hot[idx_train, :],
                       samples=idx_train, args_cuda=use_cuda, equal_weights=False)

        model.eval()
        
        # Calculate performance statistics for training splits
        class_prob, embed = model.run_D(samples=idx_train)
        weights, _ = model.run_W(samples=idx_train, labels=labels[idx_train], args_cuda=use_cuda, equal_weights=False)
        stats['loss_train'] = model.loss_function_D(class_prob, labels_one_hot[idx_train], weights).item()
        stats['nll_train'] = F.nll_loss(class_prob, labels[idx_train]).item()
        stats['acc_train'] = accuracy(class_prob, labels=labels[idx_train]).item()
        stats['f1Macro_train'] = class_f1(class_prob, labels[idx_train], type='macro')
        if nclass == 2:
            stats['f1Binary_train'] = class_f1(class_prob, labels[idx_train], type='binary', pos_label=pos_label)
            stats['AUC_train'] = auc_score(class_prob, labels[idx_train])

        # Evaluate performance on validation and test splits
        test(model, stats)
        
        # Track and cache best performing model state
        if epoch > drop_epochs and max_val['f1Macro_val'] < stats['f1Macro_val']:
            for key, val in stats.items():
                max_val[key] = val
            
            # Cache best model predictions for visualization
            with torch.no_grad():
                best_test_probs, _ = model.run_D(samples=idx_test)
                if use_cuda:
                    best_test_probs = best_test_probs.cpu()

        # Log active training progress to the console
        if (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            print('Epoch: {:04d} | Train Acc: {:.4f} | Train Macro F1: {:.4f}'.format(epoch + 1, stats['acc_train'], stats['f1Macro_train']))

    # Print best recorded metric configurations
    print('\n========Results==========')
    for key, val in max_val.items():
        if 'loss' in key or 'nll' in key or 'test' not in key:
            continue
        print(key.replace('_', ' ') + ' : ' + str(val))

    # Generate visual metric dashboard if valid predictions exist
    if best_test_probs is not None:
        save_evaluation_dashboard(best_test_probs, max_val)


def test(model, stats):
    model.eval()

    # Calculate validation split properties
    class_prob, embed = model.run_D(samples=idx_val)
    weights, _ = model.run_W(samples=idx_val, labels=labels[idx_val], args_cuda=use_cuda, equal_weights=True)
    stats['loss_val'] = model.loss_function_D(class_prob, labels_one_hot[idx_val], weights).item()
    stats['nll_val'] = F.nll_loss(class_prob, labels[idx_val]).item()
    stats['acc_val'] = accuracy(class_prob, labels[idx_val]).item()
    stats['f1Macro_val'] = class_f1(class_prob, labels[idx_val], type='macro')
    if nclass == 2:
        stats['f1Binary_val'] = class_f1(class_prob, labels[idx_val], type='binary', pos_label=pos_label)
        stats['AUC_val'] = auc_score(class_prob, labels[idx_val])

    # Calculate test split properties
    class_prob, embed = model.run_D(samples=idx_test)
    weights, _ = model.run_W(samples=idx_test, labels=labels[idx_test], args_cuda=use_cuda, equal_weights=True)
    stats['loss_test'] = model.loss_function_D(class_prob, labels_one_hot[idx_test], weights).item()
    stats['nll_test'] = F.nll_loss(class_prob, labels[idx_test]).item()
    stats['acc_test'] = accuracy(class_prob, labels[idx_test]).item()
    stats['f1Macro_test'] = class_f1(class_prob, labels[idx_test], type='macro')
    if nclass == 2:
        stats['f1Binary_test'] = class_f1(class_prob, labels[idx_test], type='binary', pos_label=pos_label)
        stats['AUC_test'] = auc_score(class_prob, labels[idx_test])


def save_evaluation_dashboard(probs, max_val):
    # Parse categorical output values
    pred_labels = torch.argmax(probs, dim=1).numpy()
    true_labels = labels[idx_test].cpu().numpy()
    pos_probs = torch.exp(probs)[:, 1].numpy() if probs.min() < 0 else probs[:, 1].numpy()

    # Generate evaluation array states
    cm = confusion_matrix(true_labels, pred_labels)
    fpr, tpr, _ = roc_curve(true_labels, pos_probs)

    # Configure plot canvases to a 3-column layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_text, ax_cm, ax_roc = axes[0], axes[1], axes[2]

    # Dynamic file naming setup using dataset name
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    # --- Panel 1: Training Run Metadata Text Block ---
    ax_text.axis('off')
    summary_text = (
        "=== Run Summary ===\n\n"
        f"Dataset: {dataset_name.upper()}\n"
        f"Total Samples: {len(labels)} Nodes\n"
        f"Test Set Size: {len(true_labels)} Nodes\n"
        f"Total Epochs: {args.epochs}\n\n"
        "=== Best Test Metrics ===\n\n"
        f"Test Accuracy: {max_val['acc_test']:.4f}\n"
        f"Macro F1 Score: {max_val['f1Macro_test']:.4f}\n"
        f"Binary F1 Score: {max_val['f1Binary_test']:.4f}\n"
        f"ROC AUC Score: {max_val['AUC_test']:.4f}"
    )
    ax_text.text(0.1, 0.9, summary_text, fontsize=12, family='sans-serif',
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9f9', edgecolor='#d6dbdf'))

    # --- Panel 2: Confusion Matrix Heatmap ---
    classes = ["Class 0", "Class 1"]
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title(f"{dataset_name.upper()} Confusion Matrix (Test)", fontsize=13, weight='bold', pad=10)
    fig.colorbar(im, ax=ax_cm, shrink=0.7)
    
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(classes, rotation=15, fontsize=10)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        ax_cm.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=12, weight='bold')
    ax_cm.set_ylabel('True Clinical Class', fontsize=11, weight='bold')
    ax_cm.set_xlabel('Predicted Clinical Class', fontsize=11, weight='bold')

    # --- Panel 3: Receiver Operating Characteristic Curve ---
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'RA-GCN Model (AUC = {max_val["AUC_test"]:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=11, weight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=11, weight='bold')
    ax_roc.set_title(f'{dataset_name.upper()} ROC Curve', fontsize=13, weight='bold', pad=10)
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, linestyle=':', alpha=0.6)

    # Export unique visualization using dynamic filename
    plt.tight_layout()
    output_filename = f"{dataset_name}_evaluation_dashboard.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Dashboard Saved Successfully] -> Finished exporting {output_filename}.")


if __name__ == '__main__':
    # Initialize command-line argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/oasis/oasis_data.pkl', help='Path to the dataset pickle file.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--epoch_D', type=int, default=1, help='Discriminator iterations per epoch.')
    parser.add_argument('--epoch_W', type=int, default=1, help='Weighting network iterations per epoch.')
    parser.add_argument('--lr_D', type=float, default=0.01, help='Learning rate for discriminator.')
    parser.add_argument('--lr_W', type=float, default=0.01, help='Learning rate for weighting networks.')
    parser.add_argument('--dropout_D', type=float, default=0.5, help='Dropout rate for discriminator.')
    parser.add_argument('--dropout_W', type=float, default=0.5, help='Dropout rate for weighting networks.')
    parser.add_argument('--gamma', type=float, default=1, help='Entropy coefficient parameter.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU execution.')
    
    # Configure internal topology dimensions
    structure_D = [2]
    structure_W = [4]
    drop_epochs = 500
    args = parser.parse_args()

    # Determine validation device capabilities
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Load dynamically from provided argument path
    adj, features, labels, idx_train, idx_val, idx_test = load_data_medical(dataset_addr=args.dataset,
                                                                           train_ratio=0.6, test_ratio=0.2)

    # Format categorical states into target vectors
    labels_one_hot = encode_onehot_torch(labels)
    nclass = labels_one_hot.shape[1]
    n_ws = nclass
    pos_label = None
    
    # Resolve target label priorities for imbalanced distributions
    if nclass == 2:
        pos_label = 1
        zero_class = (labels == 0).sum()
        one_class = (labels == 1).sum()
        if zero_class < one_class:
            pos_label = 0
            
    # Relocate calculation dependencies to hardware acceleration memory
    if use_cuda:
        for key, val in adj.items():
            if type(val) is list:
                for i in range(len(adj)):
                    adj[key][i] = adj[key][i].cuda()
            else:
                adj[key] = adj[key].cuda()
        features = features.cuda()
        labels_one_hot = labels_one_hot.cuda()
        labels = labels.cuda()

    # Execute core model training sequence
    train()
