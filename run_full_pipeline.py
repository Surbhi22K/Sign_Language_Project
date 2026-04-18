"""
Full pipeline script: Train → Evaluate → Infer both videos → Save results.txt
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_code.training.train import train
from new_code.evaluation.evaluate import evaluate
from new_code.evaluation.metrics import plot_loss_curve, save_results
from new_code.inference.frame_infer import infer_video
from new_code.utils.convert import convert_to_tflite
from new_code.utils.config import CONFIG, get_model_path, get_tflite_path
from new_code.data.dataset import SignLanguageDataset

def run_all():
    # ── 1. Dataset info ──────────────────────────────────────────
    ds = SignLanguageDataset()
    train_gen = ds.train_generator()
    val_gen = ds.val_generator()
    class_labels = [k for k, _ in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]

    dataset_info = {
        "dataset_dir": CONFIG["dataset_dir"],
        "num_classes": train_gen.num_classes,
        "train_samples": train_gen.samples,
        "val_samples": val_gen.samples,
        "class_labels": class_labels,
        "image_size": CONFIG["image_size"],
        "validation_split": CONFIG["validation_split"],
    }

    # Per-class distribution
    import os as _os
    class_dist = {}
    for d in sorted(_os.listdir(CONFIG["dataset_dir"])):
        p = _os.path.join(CONFIG["dataset_dir"], d)
        if _os.path.isdir(p):
            class_dist[d] = len(_os.listdir(p))
    dataset_info["class_distribution"] = class_dist

    print("=" * 60)
    print("  DATASET INFO")
    print("=" * 60)
    for k, v in dataset_info.items():
        if k != "class_distribution":
            print(f"  {k}: {v}")
    print(f"  class_distribution:")
    for k, v in class_dist.items():
        print(f"    {k}: {v}")

    # ── 2. Train ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    start_time = time.time()
    history = train()
    train_time = time.time() - start_time

    # Save training curves
    plot_loss_curve(history)

    training_info = {
        "optimizer": "Adam",
        "learning_rate": CONFIG["learning_rate"],
        "loss_function": "CategoricalCrossentropy",
        "batch_size": CONFIG["batch_size"],
        "max_epochs": CONFIG["epochs"],
        "early_stopping_patience": CONFIG["early_stopping_patience"],
        "actual_epochs": len(history["loss"]),
        "training_time_sec": round(train_time, 2),
        "final_train_loss": round(history["loss"][-1], 6),
        "final_train_accuracy": round(history["accuracy"][-1], 6),
        "final_val_loss": round(history["val_loss"][-1], 6),
        "final_val_accuracy": round(history["val_accuracy"][-1], 6),
        "best_val_loss": round(min(history["val_loss"]), 6),
        "best_val_accuracy": round(max(history["val_accuracy"]), 6),
        "best_epoch": int(history["val_loss"].index(min(history["val_loss"])) + 1),
        "loss_per_epoch": [round(x, 6) for x in history["loss"]],
        "val_loss_per_epoch": [round(x, 6) for x in history["val_loss"]],
        "accuracy_per_epoch": [round(x, 6) for x in history["accuracy"]],
        "val_accuracy_per_epoch": [round(x, 6) for x in history["val_accuracy"]],
    }

    # ── 3. Model info ────────────────────────────────────────────
    from tensorflow.keras.models import load_model
    model = load_model(get_model_path())
    model_info = {
        "architecture": "MobileNetV2 (frozen) + Flatten + Dense(128, ReLU) + Dropout(0.5) + Dense(24, Softmax)",
        "base_model": "MobileNetV2 (ImageNet weights, frozen)",
        "input_shape": str(CONFIG["input_shape"]),
        "output_shape": str(model.output_shape),
        "total_params": model.count_params(),
        "trainable_params": sum(p.numpy().size for p in model.trainable_weights),
        "non_trainable_params": sum(p.numpy().size for p in model.non_trainable_weights),
        "model_path": get_model_path(),
        "model_size_mb": round(os.path.getsize(get_model_path()) / (1024 * 1024), 2),
    }

    # ── 4. Evaluate ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    eval_results = evaluate(history=history)

    # ── 5. Inference on both videos ──────────────────────────────
    print("\n" + "=" * 60)
    print("  INFERENCE")
    print("=" * 60)
    inference_results = {}
    for vname in [CONFIG["test_video"], "signv.mp4"]:
        vpath = os.path.join(CONFIG["video_dir"], vname)
        if os.path.isfile(vpath):
            print(f"\n  Processing {vname}...")
            decoded = infer_video(video_path=vpath, labels=class_labels)
            
            # The decoded string is already a sentence with words and spaces since we added TextBlob.
            # We don't need to rebuild letters manually, just store the output!
            inference_results[vname] = {
                "final_decoded": decoded,
                "video_path": vpath,
            }
            print(f"  {vname} → final text: {decoded}")
        else:
            print(f"  {vname} not found, skipping")

    # ── 6. TFLite conversion ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL CONVERSION")
    print("=" * 60)
    tflite_path = convert_to_tflite()
    tflite_info = {
        "tflite_path": tflite_path,
        "tflite_size_mb": round(os.path.getsize(tflite_path) / (1024 * 1024), 2),
        "quantization": "tf.lite.Optimize.DEFAULT",
    }

    # ── 7. Build results.txt ────────────────────────────────────
    results_txt = os.path.join(CONFIG["reports_dir"], "results.txt")

    with open(results_txt, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(" SIGN LANGUAGE RECOGNITION — COMPLETE RESULTS\n")
        f.write(f" Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Dataset
        f.write("-" * 70 + "\n")
        f.write(" 1. DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f" Directory        : {dataset_info['dataset_dir']}\n")
        f.write(f" Number of classes : {dataset_info['num_classes']}\n")
        f.write(f" Training samples  : {dataset_info['train_samples']}\n")
        f.write(f" Validation samples: {dataset_info['val_samples']}\n")
        f.write(f" Image size        : {dataset_info['image_size']}\n")
        f.write(f" Val split         : {dataset_info['validation_split']}\n")
        f.write(f"\n Class Distribution:\n")
        for cls, cnt in class_dist.items():
            letter = cls.replace("-samples", "")
            f.write(f"   {letter}: {cnt} images\n")

        # Model
        f.write(f"\n{'-' * 70}\n")
        f.write(" 2. MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write(f" Architecture     : {model_info['architecture']}\n")
        f.write(f" Base model       : {model_info['base_model']}\n")
        f.write(f" Input shape      : {model_info['input_shape']}\n")
        f.write(f" Output shape     : {model_info['output_shape']}\n")
        f.write(f" Total params     : {model_info['total_params']:,}\n")
        f.write(f" Trainable params : {model_info['trainable_params']:,}\n")
        f.write(f" Non-trainable    : {model_info['non_trainable_params']:,}\n")
        f.write(f" Saved model size : {model_info['model_size_mb']} MB\n")

        # Training
        f.write(f"\n{'-' * 70}\n")
        f.write(" 3. TRAINING DETAILS\n")
        f.write("-" * 70 + "\n")
        f.write(f" Optimizer         : {training_info['optimizer']}\n")
        f.write(f" Learning rate     : {training_info['learning_rate']}\n")
        f.write(f" Loss function     : {training_info['loss_function']}\n")
        f.write(f" Batch size        : {training_info['batch_size']}\n")
        f.write(f" Max epochs        : {training_info['max_epochs']}\n")
        f.write(f" Actual epochs     : {training_info['actual_epochs']}\n")
        f.write(f" Best epoch        : {training_info['best_epoch']}\n")
        f.write(f" Early stop patience: {training_info['early_stopping_patience']}\n")
        f.write(f" Training time     : {training_info['training_time_sec']} sec\n")
        f.write(f"\n Final Metrics:\n")
        f.write(f"   Train Loss     : {training_info['final_train_loss']}\n")
        f.write(f"   Train Accuracy : {training_info['final_train_accuracy']}\n")
        f.write(f"   Val Loss       : {training_info['final_val_loss']}\n")
        f.write(f"   Val Accuracy   : {training_info['final_val_accuracy']}\n")
        f.write(f"   Best Val Loss  : {training_info['best_val_loss']}\n")
        f.write(f"   Best Val Acc   : {training_info['best_val_accuracy']}\n")

        f.write(f"\n Epoch-by-Epoch:\n")
        f.write(f"   {'Epoch':>5} | {'Train Loss':>12} | {'Train Acc':>10} | {'Val Loss':>12} | {'Val Acc':>10}\n")
        f.write(f"   {'-'*5}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}\n")
        for i in range(training_info["actual_epochs"]):
            f.write(f"   {i+1:>5} | {training_info['loss_per_epoch'][i]:>12.6f} | {training_info['accuracy_per_epoch'][i]:>10.6f} | {training_info['val_loss_per_epoch'][i]:>12.6f} | {training_info['val_accuracy_per_epoch'][i]:>10.6f}\n")

        # Evaluation
        f.write(f"\n{'-' * 70}\n")
        f.write(" 4. EVALUATION METRICS (Validation Set)\n")
        f.write("-" * 70 + "\n")
        f.write(f" Accuracy  : {eval_results['accuracy']}\n")
        f.write(f" Precision : {eval_results['precision']}\n")
        f.write(f" Recall    : {eval_results['recall']}\n")
        f.write(f" F1-Score  : {eval_results['f1_score']}\n")
        f.write(f"\n Classification Report:\n")
        f.write(eval_results["classification_report"])

        # Inference
        f.write(f"\n{'-' * 70}\n")
        f.write(" 5. VIDEO INFERENCE RESULTS\n")
        f.write("-" * 70 + "\n")
        for vname, vres in inference_results.items():
            f.write(f"\n Video: {vname}\n")
            f.write(f"   Final NLP Text             : {vres['final_decoded']}\n")

        # TFLite
        f.write(f"\n{'-' * 70}\n")
        f.write(" 6. MODEL CONVERSION (TFLite)\n")
        f.write("-" * 70 + "\n")
        f.write(f" TFLite model size : {tflite_info['tflite_size_mb']} MB\n")
        f.write(f" Quantization      : {tflite_info['quantization']}\n")
        f.write(f" TFLite path       : {tflite_info['tflite_path']}\n")

        # Sequence decoder config
        f.write(f"\n{'-' * 70}\n")
        f.write(" 7. SEQUENCE DECODER CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f" Sliding window size : {CONFIG['sliding_window_size']}\n")
        f.write(f" Method              : Majority voting + Collapse repeats + TextBlob Spell Check\n")

        f.write(f"\n{'=' * 70}\n")
        f.write(" END OF RESULTS\n")
        f.write("=" * 70 + "\n")

    print(f"\n✅ results.txt saved → {results_txt}")

    # Also save structured JSON
    all_results = {
        "dataset": dataset_info,
        "model": model_info,
        "training": training_info,
        "evaluation": eval_results,
        "inference": inference_results,
        "tflite": tflite_info,
    }
    json_path = os.path.join(CONFIG["reports_dir"], "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"✅ results.json saved → {json_path}")

if __name__ == "__main__":
    run_all()
