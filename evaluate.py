import os
import argparse
import torch

from src.dataset import get_loader
from src.models import build_model
from src.metrics import compute_dice_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Lung Segmentation Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth file)")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "vanilla_unet"],
                        help="Model architecture: 'unet' or 'vanilla_unet'")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Encoder backbone if using SMP unet (default: resnet34)")
    parser.add_argument("--test_csv", type=str, default="splits/test_fixed_20pct.csv",
                        help="Path to CSV containing test set paths")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution (default: 256)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker subprocesses")
    return parser.parse_args()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" EVALUATION ENGINE: {args.model.upper()}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Test Split: {args.test_csv} | Device: {device}")
    print(f"=======================================================\n")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV split not found: {args.test_csv}")

    test_loader = get_loader(
        args.test_csv,
        img_size=args.img_size,
        batch_size=args.batch_size,
        is_train=False,
        num_workers=args.num_workers
    )

    model = build_model(model_name=args.model, encoder=args.encoder, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_dices = []
    test_ious = []

    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            dice, iou = compute_dice_iou(logits, masks)
            test_dices.append(dice)
            test_ious.append(iou)

    mean_dice = (sum(test_dices) / max(1, len(test_dices))) * 100
    mean_iou = (sum(test_ious) / max(1, len(test_ious))) * 100

    print("-------------------------------------------------------")
    print(f" Quantitative Results on Independent Test Set ({len(test_loader.dataset)} cases):")
    print(f" >> Mean Dice Similarity Coefficient (DSC): {mean_dice:.2f}%")
    print(f" >> Mean Intersection over Union (mIoU):    {mean_iou:.2f}%")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate(cli_args)
