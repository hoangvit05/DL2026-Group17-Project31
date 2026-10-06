import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import get_loader, prepare_splits
from src.models import build_model
from src.metrics import ComboLoss, compute_dice_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lung Segmentation Models under Limited Data Regimes")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "vanilla_unet"],
                        help="Model architecture: 'unet' (Pretrained ResNet-34) or 'vanilla_unet' (From Scratch)")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Pretrained encoder backbone for SMP Unet (default: resnet34)")
    parser.add_argument("--ratio", type=int, default=100, choices=[5, 10, 25, 50, 100],
                        help="Training data subset proportion (percent): 5, 10, 25, 50, or 100")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate (default: 3e-4)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution (default: 256)")
    parser.add_argument("--splits_dir", type=str, default="splits",
                        help="Directory containing split CSV files")
    parser.add_argument("--data_dir", type=str, default="Lung Segmentation",
                        help="Root directory of raw dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save trained model checkpoints")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker subprocesses")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Model: {args.model.upper()} | Backbone: {args.encoder}")
    print(f" Data Ratio: {args.ratio}% | Epochs: {args.epochs} | Device: {device}")
    print(f"=======================================================\n")

    train_csv = os.path.join(args.splits_dir, f"train_{args.ratio}pct.csv")
    val_csv = os.path.join(args.splits_dir, "val_fixed.csv")
    test_csv = os.path.join(args.splits_dir, "test_fixed_20pct.csv")

    if not os.path.exists(train_csv):
        print(f"[Info] Split files not found in '{args.splits_dir}'. Generating splits...")
        prepare_splits(data_dir=args.data_dir, output_dir=args.splits_dir)

    train_loader = get_loader(train_csv, img_size=args.img_size, batch_size=args.batch_size,
                              is_train=True, num_workers=args.num_workers)
    val_loader = get_loader(val_csv, img_size=args.img_size, batch_size=args.batch_size,
                            is_train=False, num_workers=args.num_workers)
    test_loader = get_loader(test_csv, img_size=args.img_size, batch_size=args.batch_size,
                             is_train=False, num_workers=args.num_workers)

    model = build_model(model_name=args.model, encoder=args.encoder).to(device)
    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                dice, _ = compute_dice_iou(logits, masks)
                val_dices.append(dice)
        val_dice = sum(val_dices) / max(1, len(val_dices))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Val Dice: {val_dice * 100:.2f}% (Best: {best_val_dice * 100:.2f}%)")

    # Save Best Checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"{args.model}_{args.ratio}pct.pth")
    if best_weights is not None:
        torch.save(best_weights, checkpoint_path)
        print(f"\n[Saved Checkpoint] Best model weights saved to: {checkpoint_path}")
        model.load_state_dict(best_weights)

    # Evaluate on Fixed Independent Test Split
    print(f"\nEvaluating on Independent Test Split ({len(test_loader.dataset)} images)...")
    model.to(device)
    model.eval()
    test_dices, test_ious = [], []
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            dice, iou = compute_dice_iou(logits, masks)
            test_dices.append(dice)
            test_ious.append(iou)

    final_test_dice = (sum(test_dices) / max(1, len(test_dices))) * 100
    final_test_iou = (sum(test_ious) / max(1, len(test_ious))) * 100

    print("=" * 60)
    print(f" FINAL TEST RESULT: [{args.model.upper()} | Data Ratio: {args.ratio}%]")
    print(f" >> Test Dice Score: {final_test_dice:.2f}%")
    print(f" >> Test IoU (mIoU): {final_test_iou:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
