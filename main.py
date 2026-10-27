# =============================
# 1. Load Dataset & Augment
# =============================
from logger import get_logger
from data.dataloader import get_dataloaders, load_config
from data.transforms import apply_shortcut_flags
import argparse
import json

# ==========================
# 2. Build Model (CNN + FAGate + FER)
# ==========================
from models.backbones import FIRMNet
import torch

# =============================
# 3. Loss, Optimizer, Scheduler, Callback, Train Loop
# =============================
from train.losses import FocalLoss
from train.optimizer import choose_optimizer
from train.callbacks import ModelCheckpoint
from train.trainer import train_model

logger = get_logger(__name__, log_dir="logs", log_filename="main.log")


def load_data(args):
    # Apply augmentation shortcut flags
    apply_shortcut_flags(args)

    # Log final augmentation config
    logger.info("Final data config:")
    logger.info(json.dumps(vars(args), indent=2))

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        transform_opt=args,
        logger=logger
    )

    return train_loader, val_loader


def build_model(device: torch.device, num_classes=2, pretrained_path=None):
    logger.info("Initializing model: FIRMNet + FAGate + FER")

    model = FIRMNet(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if pretrained_path:
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Pretrained weights loaded.")

    return model


def setup_training(model, train_loader, val_loader, device, args, num_epochs=20):
    # Loss function
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=float(getattr(args, "loss_gamma", 2.0)),
        reduction=getattr(args, "loss_reduction", "mean")
    )

    # Optimizer
    optimizer = choose_optimizer(
        model,
        optimizer_type=args.opt,
        lr=float(getattr(args, "lr", 1e-4)),
        weight_decay=float(getattr(args, "weight_decay", 1e-6))
    )

    # Scheduler
    scheduler_config = {
        'scheduler_type': args.sched,
        'warmup_type': args.warmup,
        'warmup_epochs': args.warmup_epochs,
        'scheduler_params': {}
    }

    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        save_dir=args.checkpoint_dir,
        monitor=args.monitor,
        mode=args.mode,
        save_interval=args.save_interval,
        save_best_only=args.save_best_only,
        verbose=True
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        metric_add=args.metrics,
        checkpoint_callback=checkpoint_callback,
        use_amp=False,
        use_scheduler=True,
        scheduler_config=scheduler_config,
        logger=logger
    )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIRMNet Training Pipeline")

    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to .json or .yaml config file")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights (optional)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (override config)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default="mps",
                        help="Device to train on")

    # Shortcuts
    parser.add_argument('--augment_strong', action='store_true')
    parser.add_argument('--color_jitter', action='store_true')

    # Logging + data paths
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset with FAKE/REAL subfolders")
    parser.add_argument('--log_dir', type=str, default="logs")

    # Data split & loader
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)

    # Optimizer & scheduler
    parser.add_argument('--opt', type=str, default='adam', help="Optimizer type: adam | sgd | adamw")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="Weight decay")

    parser.add_argument('--sched', type=str, default='cosine', help="Scheduler type")
    parser.add_argument('--warmup', type=str, default='linear', help="Warmup type: linear | cosine")
    parser.add_argument('--warmup_epochs', type=int, default=3, help="Number of warmup epochs")

    # Loss & metrics
    parser.add_argument('--loss_gamma', type=float, default=2.0)
    parser.add_argument('--loss_reduction', type=str, default="mean")
    parser.add_argument('--metrics', type=str, nargs='*',
                        default=['AP', 'F1 score', 'precision', 'recall', 'EER'])

    # Callback
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--monitor', type=str, default="val_loss")
    parser.add_argument('--mode', type=str, default="min")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_best_only', type=bool, default=False)

    args = parser.parse_args()

    # Load config file (yaml/json) and merge with CLI
    if args.config and args.config.endswith((".yaml", ".yml", ".json")):
        logger.info(f"Loading config from: {args.config}")
        cfg = load_config(args.config)

        for k, v in cfg.items():
            # Prefer CLI > config.yaml
            if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)

        # logger.info("Final configuration (merged CLI + config):")
        # logger.info(json.dumps(vars(args), indent=2))
    else:
        logger.info("Using manually provided command-line arguments:")

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_loader, val_loader = load_data(args)

    # Optional override
    pretrained_path = args.pretrained or getattr(args, "pretrained_path", None)
    num_epochs = args.epochs or getattr(args, "num_epochs", 20)

    # Build model and train
    model = build_model(device, num_classes=2, pretrained_path=pretrained_path)
    history = setup_training(model, train_loader, val_loader, device, args, num_epochs=num_epochs)

    # Log training history
    logger.info("Training history from main: \n%s", history)
