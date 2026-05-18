from data_setup import *
import optuna, os, torch, gc
from huggingface_hub import login

# Load study
study = optuna.load_study(
    study_name="whisper_optimization_v1.6",
    storage="sqlite:///whisper_optuna.db"
)

best = study.best_trial.params

model, processor = load_model(best["r"], best["alpha"])
train_dataset = make_train(processor, best["bs"])

from transformers import TrainerCallback

class FinalTrainingCallback(TrainerCallback):
    """
    Custom callback for final training compatible with latest Transformers.
    Just prints a message when a checkpoint is saved.
    """
    def __init__(self, save_every=100):
        self.save_every = save_every
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1

        if state.global_step % self.save_every == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"[Callback] Checkpoint should have been saved at step {state.global_step}: {checkpoint_dir}")

print("pradedamas mokymas")

trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=SpeechCollator(processor),
    compute_metrics=lambda p: compute_metrics(p, processor),
    args=Seq2SeqTrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=best["bs"],
        gradient_accumulation_steps=best["ga"],
        max_steps=10000,
        learning_rate=best["lr"],
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=20,
        report_to="none",
        save_strategy="steps",
        save_steps=100,      # automatically save every 100 steps
        save_total_limit=5,
        predict_with_generate=True,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
    ),
)

os.makedirs("outputs", exist_ok=True)
with open("outputs/best_params.txt", "w") as f:
    f.write(f"{best}\n")

trainer.train()

#Resume utilities
def resume_optuna_study(storage="sqlite:///whisper_optuna.db", study_name="whisper_optimization_v1.6"):
    return optuna.load_study(study_name=study_name, storage=storage)

def resume_training_from_checkpoint(checkpoint_path, processor, train_dataset, eval_dataset):
    model, processor = FastModel.from_pretrained(checkpoint_path)

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=SpeechCollator(processor),
        compute_metrics=lambda p: compute_metrics(p, processor),
        args=Seq2SeqTrainingArguments(
            output_dir="outputs_resumed",
            save_strategy="steps",
            save_steps=100,
        ),
    )

    trainer.train(resume_from_checkpoint=checkpoint_path)
    return trainer, model

#Push to hub
trainer.save_model("outputs/final_model")
processor.save_pretrained("outputs/final_model")
print("Issaugotas modelis outputs")

login("HF_token")

processor.push_to_hub("domineeka/whisper-large-lt-v1-opt")
trainer.push_to_hub("domineeka/whisper-large-lt-v1-opt")
model.push_to_hub("domineeka/whisper-large-lt-v1-opt")