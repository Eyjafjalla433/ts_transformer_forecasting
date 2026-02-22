from engine.infer import (
    autoregressive_forecast,
    build_model_from_config,
    denormalize_tensor,
    export_predictions_to_csv,
    load_input_window_from_csv,
    load_model_weights,
    load_normalization_stats,
    prepare_source_tensor,
    resolve_device,
)
from utils.config import load_config


def main():
    """Run autoregressive inference from config."""
    cfg = load_config("configs/default.yaml")
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    infer_cfg = cfg.get("infer", {})

    device = resolve_device(infer_cfg.get("device", "cpu"))
    model = build_model_from_config(model_cfg, device)
    load_model_weights(model, infer_cfg["checkpoint_path"], device)

    stats = load_normalization_stats(infer_cfg.get("normalization_stats_path"))
    input_mean = stats["input_mean"] if stats else None
    input_std = stats["input_std"] if stats else None
    target_mean = stats["target_mean"] if stats else None
    target_std = stats["target_std"] if stats else None

    src_window = load_input_window_from_csv(
        input_path=infer_cfg["input_path"],
        src_dim=model_cfg["src_dim"],
        input_length=data_cfg["input_length"],
        has_header=infer_cfg.get("input_has_header", True),
    )

    src, src_mask = prepare_source_tensor(
        src_window=src_window,
        device=device,
        input_mean=input_mean,
        input_std=input_std,
    )

    pred_norm = autoregressive_forecast(
        model=model,
        src=src,
        src_mask=src_mask,
        pred_length=data_cfg["pred_length"],
        tgt_dim=model_cfg["tgt_dim"],
        out_dim=model_cfg["out_dim"],
        start_token_mode=infer_cfg.get("start_token_mode", "zeros"),
    )

    if target_mean is not None:
        target_mean = target_mean.to(device)
    if target_std is not None:
        target_std = target_std.to(device)
    pred = denormalize_tensor(pred_norm, target_mean, target_std)

    output_path = infer_cfg.get("output_path", "outputs/predictions.csv")
    export_predictions_to_csv(pred, output_path)

    print("Inference completed.")
    print("device:", str(device))
    print("src shape:", tuple(src.shape))
    print("pred shape:", tuple(pred.shape))
    print("output:", output_path)


if __name__ == "__main__":
    main()
