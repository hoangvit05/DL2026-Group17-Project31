import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from src.dataset import prepare_splits, get_fast_loader, get_transforms
from src.models import build_model
from src.metrics import ComboLoss, compute_dice_iou


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_eval(model_name, encoder, ratio_pct, device, epochs=25, batch_size=8, lr=3e-4):
    train_loader = get_fast_loader(f'splits/train_{ratio_pct}pct.csv', batch_size=batch_size, is_train=True)
    val_loader = get_fast_loader('splits/val_fixed.csv', batch_size=batch_size, is_train=False)
    test_loader = get_fast_loader('splits/test_fixed_20pct.csv', batch_size=batch_size, is_train=False)

    model = build_model(model_name, encoder).to(device)
    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_dice = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                d, _ = compute_dice_iou(model(imgs), masks)
                dices.append(d)
        val_dice = sum(dices) / max(1, len(dices))
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save Checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/{model_name}_{ratio_pct}pct.pth'
    if best_weights is not None:
        torch.save(best_weights, checkpoint_path)
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Test Evaluation
    model.eval()
    test_dices, test_ious = [], []
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            d, i = compute_dice_iou(model(imgs), masks)
            test_dices.append(d)
            test_ious.append(i)

    test_dice = sum(test_dices) / max(1, len(test_dices))
    test_iou = sum(test_ious) / max(1, len(test_ious))

    print(f"  [{model_name.upper():12s} | Ratio: {ratio_pct:3d}%] -> Test Dice: {test_dice * 100:.2f}% | Test IoU: {test_iou * 100:.2f}% | Saved: {checkpoint_path}")
    return test_dice, test_iou


def plot_scaling_curves(df_res, output_path='data_scaling_comparison.png'):
    print("\n>>> [4/5] Plotting 4-panel data scaling curves...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    colors = {
        'Pretrained U-Net (ResNet-34)': '#1f77b4',
        'Vanilla U-Net (From Scratch)': '#d62728'
    }
    markers = {
        'Pretrained U-Net (ResNet-34)': 'o',
        'Vanilla U-Net (From Scratch)': 's'
    }

    ratios = [10, 25, 50, 100]
    x_tick_labels = ['10%', '25%', '50%', '100%']

    configs = [
        (0, 0, 'test_dice', (0, 100), 'Test Dice Score vs. Training Data Scale (Full Scale: 0% - 100%)', 'Test Dice Score (%)'),
        (0, 1, 'test_iou', (0, 100), 'Test IoU vs. Training Data Scale (Full Scale: 0% - 100%)', 'Test IoU (%)'),
        (1, 0, 'test_dice', (80, 100), 'Test Dice Score vs. Training Data Scale (Zoomed: 80% - 100%)', 'Test Dice Score (%)'),
        (1, 1, 'test_iou', (80, 100), 'Test IoU vs. Training Data Scale (Zoomed: 80% - 100%)', 'Test IoU (%)')
    ]

    for row, col, metric, y_lim, title, ylabel in configs:
        ax = axes[row, col]
        for m_name, group in df_res.groupby('model'):
            group_sorted = group.sort_values('ratio')
            color = colors.get(m_name, '#333333')
            marker = markers.get(m_name, 'o')
            ax.plot(
                group_sorted['ratio'],
                group_sorted[metric] * 100,
                marker=marker,
                linewidth=2.4,
                markersize=8,
                color=color,
                label=m_name
            )

        if y_lim == (80, 100):
            p_data = df_res[df_res['model'].str.contains('Pretrained')].set_index('ratio')[metric] * 100
            v_data = df_res[df_res['model'].str.contains('Vanilla')].set_index('ratio')[metric] * 100
            offset_map = {
                10:  {'higher': (12, 8),   'lower': (12, -15)},
                25:  {'higher': (0, 9),    'lower': (0, -15)},
                50:  {'higher': (0, 9),    'lower': (0, -15)},
                100: {'higher': (0, 9),    'lower': (0, -15)}
            }
            for r in ratios:
                p_val = p_data[r]
                v_val = v_data[r]
                p_pos = 'higher' if p_val >= v_val else 'lower'
                v_pos = 'lower' if p_val >= v_val else 'higher'
                ax.annotate(
                    f"{p_val:.2f}%",
                    (r, p_val),
                    textcoords="offset points",
                    xytext=offset_map[r][p_pos],
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=colors['Pretrained U-Net (ResNet-34)'],
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.75)
                )
                ax.annotate(
                    f"{v_val:.2f}%",
                    (r, v_val),
                    textcoords="offset points",
                    xytext=offset_map[r][v_pos],
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=colors['Vanilla U-Net (From Scratch)'],
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.75)
                )

        ax.set_title(title, fontsize=12.5, fontweight='bold', pad=10)
        ax.set_xlabel('Training Data Scale (%)', fontsize=11, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='semibold')
        ax.set_xticks(ratios)
        ax.set_xticklabels(x_tick_labels, fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_ylim(y_lim)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10.5, loc='lower right', frameon=True, framealpha=0.9)

    plt.suptitle('Data Scaling Benchmark: Pretrained U-Net vs. Vanilla U-Net', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.16)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"    Saved: {output_path}")


def generate_qualitative_matrix(device, output_path='qualitative_comparison.png'):
    print("\n>>> [5/5] Generating qualitative visual prediction matrix...")
    df_test = pd.read_csv('splits/test_fixed_20pct.csv')
    transforms_infer = get_transforms(img_size=256, is_train=False)

    sample_indices = [5, 18, 32, 45, 60]
    ratios = [10, 25, 50, 100]

    models_v = {}
    models_p = {}
    for r in ratios:
        m_v = build_model('vanilla_unet', 'none', pretrained=False).to(device)
        m_v.load_state_dict(torch.load(f'checkpoints/vanilla_unet_{r}pct.pth', map_location=device))
        m_v.eval()
        models_v[r] = m_v

        m_p = build_model('unet', 'resnet34', pretrained=False).to(device)
        m_p.load_state_dict(torch.load(f'checkpoints/unet_{r}pct.pth', map_location=device))
        m_p.eval()
        models_p[r] = m_p

    col_titles = [
        'Raw CXR', 'Ground Truth',
        '10% Data (48 samples)', '25% Data (120 samples)',
        '50% Data (240 samples)', '100% Data (480 samples)'
    ]
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples * 2, 6, figsize=(18, 4.5 * n_samples))

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold', pad=12)

    for i, s_idx in enumerate(sample_indices):
        row_v = i * 2
        row_p = i * 2 + 1

        row_data = df_test.iloc[s_idx]
        img_raw = cv2.imread(row_data['image_path'])
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        mask_gt = cv2.imread(row_data['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask_gt = (mask_gt > 127).astype(np.float32)

        img_disp = cv2.resize(img_rgb, (256, 256))
        mask_disp = cv2.resize(mask_gt, (256, 256))

        t_img = transforms_infer(image=img_disp)['image'].unsqueeze(0).to(device)

        axes[row_v, 0].imshow(img_disp)
        axes[row_v, 1].imshow(mask_disp, cmap='gray')
        axes[row_p, 0].imshow(img_disp)
        axes[row_p, 1].imshow(mask_disp, cmap='gray')

        axes[row_v, 0].set_ylabel(f'Case {i+1}\n[Vanilla]', fontsize=11, fontweight='bold', color='#c0392b')
        axes[row_p, 0].set_ylabel(f'Case {i+1}\n[Pretrained]', fontsize=11, fontweight='bold', color='#2980b9')

        with torch.no_grad():
            for c_idx, r in enumerate(ratios, start=2):
                pred_v = (torch.sigmoid(models_v[r](t_img)) > 0.5).squeeze().cpu().numpy()
                pred_p = (torch.sigmoid(models_p[r](t_img)) > 0.5).squeeze().cpu().numpy()

                axes[row_v, c_idx].imshow(pred_v, cmap='plasma')
                axes[row_p, c_idx].imshow(pred_p, cmap='plasma')

        for c in range(6):
            axes[row_v, c].set_xticks([])
            axes[row_v, c].set_yticks([])
            axes[row_p, c].set_xticks([])
            axes[row_p, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(" END-TO-END DEEP LEARNING BENCHMARK RUNNER")
    print(f" Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # 1. Splits
    print(">>> [1/5] Preparing reproducible dataset splits...")
    prepare_splits(data_dir='Lung Segmentation', output_dir='splits', test_ratio=0.20, seed=42)

    # 2. Benchmark Training (8 configurations: 4 ratios x 2 architectures)
    ratios = [10, 25, 50, 100]
    results = []

    print("\n>>> [2/5] Training Pretrained U-Net (ResNet-34) across data ratios...")
    for r in ratios:
        dice, iou = train_and_eval('unet', 'resnet34', ratio_pct=r, device=device, epochs=25, batch_size=8)
        results.append({
            'model': 'Pretrained U-Net (ResNet-34)',
            'ratio': r,
            'test_dice': dice,
            'test_iou': iou
        })

    print("\n>>> [3/5] Training Vanilla U-Net (From Scratch) across data ratios...")
    for r in ratios:
        dice, iou = train_and_eval('vanilla_unet', 'none', ratio_pct=r, device=device, epochs=25, batch_size=8)
        results.append({
            'model': 'Vanilla U-Net (From Scratch)',
            'ratio': r,
            'test_dice': dice,
            'test_iou': iou
        })

    # Save benchmark CSV
    df_res = pd.DataFrame(results)
    csv_path = 'data_scaling_benchmark_results.csv'
    df_res.to_csv(csv_path, index=False)
    print(f"\n>>> Benchmark results saved to: {csv_path}")
    print(df_res.to_string(index=False))

    # 4. Plots
    plot_scaling_curves(df_res, output_path='data_scaling_comparison.png')

    # 5. Qualitative Matrix
    generate_qualitative_matrix(device, output_path='qualitative_comparison.png')

    print("\n" + "=" * 70)
    print(" ALL WORKFLOW ARTIFACTS GENERATED SUCCESSFULLY!")
    print(" [OK] splits/                          (6 partition manifests)")
    print(" [OK] checkpoints/                     (8 trained model weights)")
    print(" [OK] data_scaling_benchmark_results.csv (Test Dice & IoU metrics)")
    print(" [OK] data_scaling_comparison.png      (4-panel publication plots)")
    print(" [OK] qualitative_comparison.png       (Visual segmentation matrix)")
    print("=" * 70)


if __name__ == '__main__':
    main()
