import re
import os
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, TrainerCallback,
)

from bytetrl_fixed import GOLDTrainer, ULDLoss

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ============================================================
# CONFIG
# ============================================================

TEACHER_MODEL = "HuggingFaceTB/SmolLM2-360M"   # базовая, без instruct
STUDENT_MODEL = "EleutherAI/pythia-70m"

OUTPUT_DIR    = "./gold_distill_hellaswag_pythia_v3"
TRAIN_SIZE    = 4000
EVAL_SIZE     = 50
MAX_COMP_LEN  = 80
MAX_SEQ_LEN   = 384
EPOCHS        = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 2
LR            = 5e-5   # было 5e-5 — осторожнее, не перезаписываем веса
LMBDA         = 1.0
BETA          = 0.5
USE_BYTE_LOSS = False
LOG_STEPS     = 100
SAVE_STEPS    = 2000
SAVE_TOTAL    = 3


# ============================================================
# Промпт — один для обоих
# ============================================================

def make_prompt(ctx: str) -> str:
    """
    Plain text промпт для учителя и студента.
    Оба базовые модели без instruct-тюнинга — форматы совпадают полностью.
    """
    return f"Continue the following text:\n{ctx.strip()}\n"


# ============================================================
# Оценка на HellaSwag
# ============================================================

def get_label(example) -> int:
    lbl = example["label"]
    if isinstance(lbl, str):
        s = lbl.strip()
        return int(s) if s.isdigit() else -1
    return int(lbl)


def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def evaluate_hellaswag(model, tokenizer, dataset, device,
                        max_new_tokens=80, num_samples=50, desc=""):
    """
    Scoring accuracy (основная): argmax log-prob/token по 4 endings.
    CE loss: CrossEntropyLoss по правильному ending (teacher-forcing).
    Gen accuracy (вспомогательная): word-prefix overlap при greedy decode.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    examples = dataset.select(range(min(num_samples, len(dataset))))
    total = correct_score = correct_gen = 0
    total_loss = 0.0
    sample_results = []

    for ex in examples:
        label = get_label(ex)
        if label < 0 or label >= len(ex["endings"]):
            continue

        ctx     = ex["ctx"]
        endings = ex["endings"]
        correct = endings[label]
        prompt  = make_prompt(ctx)

        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(device)

        # Scoring: argmax log-prob/token по 4 endings
        scores = []
        for ending in endings:
            end_ids = tokenizer(
                ending, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            full   = torch.cat([prompt_ids, end_ids], dim=1)
            labels = full.clone()
            labels[:, :prompt_ids.shape[1]] = -100
            with torch.no_grad():
                out = model(full)
                sl  = out.logits[..., :-1, :].contiguous()
                ll  = labels[..., 1:].contiguous()
                scores.append(-ce(sl.view(-1, sl.size(-1)), ll.view(-1)).item())

        pred_idx = int(np.argmax(scores))
        if pred_idx == label:
            correct_score += 1

        # CE loss по правильному ending
        ref_ids = tokenizer(
            correct, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if tokenizer.eos_token_id is not None:
            eos     = torch.tensor([[tokenizer.eos_token_id]], device=device)
            ref_ids = torch.cat([ref_ids, eos], dim=1)
        full   = torch.cat([prompt_ids, ref_ids], dim=1)
        labels = full.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        with torch.no_grad():
            out = model(full)
            sl  = out.logits[..., :-1, :].contiguous()
            ll  = labels[..., 1:].contiguous()
            total_loss += ce(sl.view(-1, sl.size(-1)), ll.view(-1)).item()

        # Generation accuracy (вспомогательная)
        attn = torch.ones_like(prompt_ids)
        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(
            gen_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True
        )

        gen_words = normalize(gen_text).split()
        best_score, best_idx = -1, -1
        for idx, ending in enumerate(endings):
            end_words = normalize(ending).split()
            common = sum(1 for gw, ew in zip(gen_words, end_words) if gw == ew)
            score  = common / max(len(end_words), 1)
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx == label:
            correct_gen += 1

        total += 1
        if len(sample_results) < 3:
            sample_results.append({
                "ctx":      ctx[:90],
                "correct":  correct[:70],
                "gen":      gen_text[:100],
                "ok_score": pred_idx == label,
                "ok_gen":   best_idx == label,
            })

    acc_score = correct_score / max(total, 1)
    acc_gen   = correct_gen   / max(total, 1)
    avg_loss  = total_loss    / max(total, 1)

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Scoring acc : {acc_score*100:.2f}%  ({correct_score}/{total})  <- основная")
    print(f"  Gen acc     : {acc_gen*100:.2f}%  ({correct_gen}/{total})")
    print(f"  CE loss     : {avg_loss:.4f}")
    print(f"{'='*60}")
    for r in sample_results:
        s = "V" if r["ok_score"] else "X"
        g = "V" if r["ok_gen"]   else "X"
        print(f"  [{s}score/{g}gen] {r['ctx']}...")
        print(f"    correct: {r['correct']}...")
        print(f"    gen    : {r['gen'][:80]}...")

    return {"accuracy": acc_score, "accuracy_gen": acc_gen,
            "avg_loss": avg_loss, "n": total}


# ============================================================
# GOLDArgs
# ============================================================

class GOLDArgs(TrainingArguments):
    def __init__(self, **kwargs):
        self.use_uld_loss                   = kwargs.pop("use_uld_loss", True)
        self.teacher_tokenizer_name_or_path = kwargs.pop("teacher_tokenizer_name_or_path", TEACHER_MODEL)
        self.uld_crossentropy_weight        = kwargs.pop("uld_crossentropy_weight", 0.5)
        self.uld_distillation_weight        = kwargs.pop("uld_distillation_weight", 0.5)
        self.uld_student_temperature        = kwargs.pop("uld_student_temperature", 1.0)
        self.uld_teacher_temperature        = kwargs.pop("uld_teacher_temperature", 1.0)
        self.uld_skip_student_eos           = kwargs.pop("uld_skip_student_eos", False)
        self.uld_skip_teacher_eos           = kwargs.pop("uld_skip_teacher_eos", False)
        self.use_extended_uld               = kwargs.pop("use_extended_uld", True)
        self.uld_use_hybrid_loss            = kwargs.pop("uld_use_hybrid_loss", True)
        self.uld_hybrid_matched_weight      = kwargs.pop("uld_hybrid_matched_weight", None)
        self.uld_hybrid_unmatched_weight    = kwargs.pop("uld_hybrid_unmatched_weight", None)
        self.beta                           = kwargs.pop("beta", BETA)
        self.lmbda                          = kwargs.pop("lmbda", LMBDA)
        self.temperature                    = kwargs.pop("temperature", 0.8)
        self.top_p                          = kwargs.pop("top_p", 0.9)
        self.top_k                          = kwargs.pop("top_k", 50)
        self.num_generations                = kwargs.pop("num_generations", 1)
        self.max_completion_length          = kwargs.pop("max_completion_length", MAX_COMP_LEN)
        self.log_completions                = kwargs.pop("log_completions", True)
        self.log_completion_steps           = kwargs.pop("log_completion_steps", LOG_STEPS * 5)
        self.num_completions_to_print       = kwargs.pop("num_completions_to_print", 2)
        self.disable_dropout                = kwargs.pop("disable_dropout", True)
        self.max_length                     = kwargs.pop("max_length", MAX_SEQ_LEN)
        self.teacher_model_name_or_path     = kwargs.pop("teacher_model_name_or_path", None)
        self.student_model_name_or_path     = kwargs.pop("student_model_name_or_path", None)
        self.use_byte_loss                  = kwargs.pop("use_byte_loss", USE_BYTE_LOSS)
        super().__init__(**kwargs)


# ============================================================
# Коллатор
# ============================================================

class SimpleCollator:
    _tensor_keys = {"input_ids", "attention_mask", "labels",
                    "prompts", "prompt_attention_mask"}

    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            vals = [f[key] for f in features]
            if key in self._tensor_keys:
                batch[key] = torch.stack([
                    torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                    for v in vals
                ])
            else:
                batch[key] = vals
        if "input_ids" not in batch and "prompts" in batch:
            batch["input_ids"] = batch["prompts"]
        return batch


# ============================================================
# Подготовка датасета
# ============================================================

def prepare_dataset(tokenizer, split: str, size: int, max_length: int):
    raw = load_dataset("Rowan/hellaswag", split=split)

    def has_valid_label(ex):
        lbl = ex["label"]
        if isinstance(lbl, str):
            return lbl.strip().isdigit()
        return isinstance(lbl, int) and 0 <= lbl <= 3

    raw = raw.filter(has_valid_label)
    raw = raw.select(range(min(size, len(raw))))

    def preprocess(examples):
        prompts = [make_prompt(ctx) for ctx in examples["ctx"]]
        tok = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        completions = []
        for endings, label in zip(examples["endings"], examples["label"]):
            idx = int(label) if not isinstance(label, str) else int(label.strip())
            completions.append(endings[idx] if idx < len(endings) else "")

        return {
            "prompts":                  tok["input_ids"],
            "prompt_attention_mask":    tok["attention_mask"],
            "original_prompt_text":     prompts,
            "original_completion_text": completions,
        }

    return raw.map(preprocess, batched=True, remove_columns=raw.column_names)


# ============================================================
# Графики
# ============================================================

def save_plots(results, output_dir, loss_history=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    subtitle = (
        f"Student: {results['student_model']}\n"
        f"Teacher: {results['teacher_model']}  |  HellaSwag  |  "
        f"train={results['train_size']}  LR={results['lr']}  CE_w={results['ce_weight']}"
    )
    colors = ["#ac120c", "#510761"]
    labels = ["Before distillation", "After distillation"]

    # График 1: Scoring accuracy
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.suptitle(subtitle, fontsize=8, color="#444444")
    vals_acc = [results["before"]["accuracy"] * 100,
                results["after"]["accuracy"]  * 100]
    bars = ax1.bar(labels, vals_acc, color=colors, width=0.4,
                   edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax1.axhline(25.0, color="#888888", linewidth=1.2, linestyle=":",
                label="random baseline (25%)")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, max(max(vals_acc) * 1.35 + 3, 35))
    ax1.set_ylabel("Scoring Accuracy (%)", fontsize=12)
    ax1.set_title("HellaSwag Scoring Accuracy\n(argmax log-prob/token over 4 endings)",
                  fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()
    p1 = Path(output_dir) / "scoring_accuracy.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {p1}")

    # График 2: Eval CE loss
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.suptitle(subtitle, fontsize=8, color="#444444")
    vals_loss = [results["before"]["avg_loss"],
                 results["after"]["avg_loss"]]
    bars = ax2.bar(labels, vals_loss, color=colors, width=0.4,
                   edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals_loss):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(vals_loss) * 1.35)
    ax2.set_ylabel("Average CE Loss", fontsize=12)
    ax2.set_title("HellaSwag Eval CE Loss\n"
                  "(CrossEntropyLoss on correct ending, teacher-forcing)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    p2 = Path(output_dir) / "eval_ce_loss.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {p2}")

    # График 3: Train loss curve
    if loss_history and len(loss_history) > 1:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        steps  = [x["step"] for x in loss_history]
        losses = [x["loss"] for x in loss_history]
        ax3.plot(steps, losses, color="#337ab7", linewidth=2,
                 marker="o", markersize=3, label="train loss (JSD + CE mix)")
        if len(losses) >= 5:
            w      = min(10, len(losses) // 3)
            smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax3.plot(steps[w - 1:], smooth, color="#d9534f", linewidth=2,
                     linestyle="--", label=f"moving avg (w={w})")
        ax3.legend(fontsize=10)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss — GOLD distillation\n"
                      "(JSD + CE mix — not comparable to eval CE loss)", fontsize=11)
        ax3.grid(alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        p3 = Path(output_dir) / "train_loss_curve.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved -> {p3}")


# ============================================================
# main
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  GOLD + ByteSampler Distillation")
    print(f"  Teacher : {TEACHER_MODEL}")
    print(f"  Student : {STUDENT_MODEL}")
    print(f"  Dataset : HellaSwag  |  {TRAIN_SIZE} examples  |  {EPOCHS} epochs")
    print(f"  LR={LR}  CE_weight=0.5  distil_weight=0.5")
    print("=" * 60)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        raise RuntimeError("XPU not available, but required.")
    print(f"\nDevice: {device}  |  XPU: {torch.xpu.get_device_name()}")

    # Учитель
    print(f"\nLoading teacher : {TEACHER_MODEL}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL, trust_remote_code=True
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # Студент
    print(f"Loading student : {STUDENT_MODEL}")
    student_tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL, trust_remote_code=True
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token    = student_tokenizer.eos_token
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    t_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
    s_params = sum(p.numel() for p in student_model.parameters()) / 1e6
    print(f"  teacher params : {t_params:.0f}M")
    print(f"  student params : {s_params:.0f}M")

    sv      = set(student_tokenizer.get_vocab().keys())
    tv      = set(teacher_tokenizer.get_vocab().keys())
    overlap = sv & tv
    print(f"\nVocab  : student={len(sv)}, teacher={len(tv)}, "
          f"overlap={len(overlap)} ({100*len(overlap)/len(tv):.1f}% of teacher)")
    print("  -> uld_use_hybrid_loss=True обработает matched/unmatched токены раздельно")

    hellaswag_val = load_dataset("Rowan/hellaswag", split="validation")

    print("\n[TEACHER baseline — справка]")
    teacher_baseline = evaluate_hellaswag(
        teacher_model, teacher_tokenizer, hellaswag_val,
        device, max_new_tokens=MAX_COMP_LEN, num_samples=EVAL_SIZE,
        desc=f"Teacher — {TEACHER_MODEL}",
    )

    print("\n[BEFORE distillation]")
    before = evaluate_hellaswag(
        student_model, student_tokenizer, hellaswag_val,
        device, max_new_tokens=MAX_COMP_LEN, num_samples=EVAL_SIZE,
        desc=f"Student BEFORE — {STUDENT_MODEL}",
    )

    print("\nPreparing training data (HellaSwag train)...")
    train_ds = prepare_dataset(student_tokenizer, "train", TRAIN_SIZE, MAX_SEQ_LEN)
    print(f"  Train examples: {len(train_ds)}")

    args = GOLDArgs(
        output_dir                       = OUTPUT_DIR,
        per_device_train_batch_size      = BATCH_SIZE,
        gradient_accumulation_steps      = GRAD_ACCUM,
        learning_rate                    = LR,
        num_train_epochs                 = EPOCHS,
        logging_steps                    = LOG_STEPS,
        save_steps                       = SAVE_STEPS,
        save_total_limit                 = SAVE_TOTAL,
        save_safetensors                 = True,
        warmup_steps                     = 200,
        lr_scheduler_type                = "cosine",
        report_to                        = "none",
        seed                             = 42,
        remove_unused_columns            = False,
        dataloader_pin_memory            = False,
        bf16                             = True,
        use_uld_loss                     = True,
        teacher_tokenizer_name_or_path   = TEACHER_MODEL,
        uld_use_hybrid_loss              = True,
        uld_crossentropy_weight          = 0.1,
        uld_distillation_weight          = 0.9,
        lmbda                            = LMBDA,
        beta                             = BETA,
        use_byte_loss                    = USE_BYTE_LOSS,
        teacher_model_name_or_path       = TEACHER_MODEL,
        student_model_name_or_path       = STUDENT_MODEL,
    )

    loss_history = []

    class LossHistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append({"step": state.global_step, "loss": logs["loss"]})

    trainer = GOLDTrainer(
        model            = student_model,
        teacher_model    = teacher_model,
        args             = args,
        train_dataset    = train_ds,
        processing_class = student_tokenizer,
        data_collator    = SimpleCollator(),
        callbacks        = [LossHistoryCallback()],
    )

    print(f"\n{'='*60}")
    print(f"  Starting distillation")
    print(f"  Examples : {TRAIN_SIZE}  |  Epochs : {EPOCHS}")
    print(f"  LR : {LR}  |  CE_weight : 0.5  |  distil_weight : 0.5")
    print(f"  Checkpoints every {SAVE_STEPS} steps  |  keep last {SAVE_TOTAL}")
    print(f"{'='*60}\n")

    checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    resume_from = str(checkpoints[-1]) if checkpoints else None
    if resume_from:
        print(f"  Resuming from checkpoint: {resume_from}\n")
    else:
        print(f"  No checkpoint found — starting from scratch\n")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min")

    print("\n[AFTER distillation]")
    after = evaluate_hellaswag(
        student_model, student_tokenizer, hellaswag_val,
        device, max_new_tokens=MAX_COMP_LEN, num_samples=EVAL_SIZE,
        desc=f"Student AFTER — {STUDENT_MODEL}",
    )

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    da = (after["accuracy"] - before["accuracy"]) * 100
    dl =  after["avg_loss"] - before["avg_loss"]
    print(f"\n  [справка] Teacher {TEACHER_MODEL}")
    print(f"            scoring acc : {teacher_baseline['accuracy']*100:.2f}%")
    print(f"\n  Student scoring acc  : {before['accuracy']*100:.2f}%  ->  "
          f"{after['accuracy']*100:.2f}%  ({da:+.2f}pp)")
    print(f"  Student eval CE loss : {before['avg_loss']:.4f}  ->  "
          f"{after['avg_loss']:.4f}  ({dl:+.4f})")
    print(f"\n  Training time : {elapsed/60:.1f} min")
    if dl > 0:
        print(f"\n  [!] CE loss вырос — признак forgetting.")
        print(f"      Попробуй снизить LR или увеличить uld_crossentropy_weight.")
    else:
        print(f"\n  [OK] CE loss снизился — forgetting не наблюдается.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    results = {
        "dataset":          "HellaSwag",
        "teacher_model":    TEACHER_MODEL,
        "student_model":    STUDENT_MODEL,
        "train_size":       TRAIN_SIZE,
        "eval_size":        EVAL_SIZE,
        "epochs":           EPOCHS,
        "lr":               LR,
        "ce_weight":        0.5,
        "distil_weight":    0.5,
        "use_byte_loss":    USE_BYTE_LOSS,
        "lmbda":            LMBDA,
        "beta":             BETA,
        "teacher_baseline": teacher_baseline,
        "before":           before,
        "after":            after,
        "training_sec":     elapsed,
    }
    out_path = Path(OUTPUT_DIR) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {out_path}")

    save_plots(results, OUTPUT_DIR, loss_history)

    student_model.save_pretrained(Path(OUTPUT_DIR) / "student_final")
    student_tokenizer.save_pretrained(Path(OUTPUT_DIR) / "student_final")
    print(f"  Student saved -> {OUTPUT_DIR}/student_final")


if __name__ == "__main__":
    main()
