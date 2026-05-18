from data_setup import *
import os

#Checkpoint Callback for Optuna
class OptunaCheckpointCallback:
    def __init__(self, trial_number, save_every=100):
        self.trial_number = trial_number
        self.save_every = save_every
        self.step = 0

    def __call__(self, trainer, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.save_every == 0:
            checkpoint_dir = f"trial_{self.trial_number}_checkpoint_{self.step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainer.save_model(checkpoint_dir)
            trainer.save_state()
            with open(f"{checkpoint_dir}/trial_info.txt", "w") as f:
                f.write(f"Trial {self.trial_number}, Step {self.step}\n")
            print(f"Saved checkpoint for trial {self.trial_number} at step {self.step}")

def objective(trial):
    lr    = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    r     = trial.suggest_categorical("r", [8, 16])
    alpha = trial.suggest_categorical("alpha", [16, 32])
    bs    = trial.suggest_categorical("bs", [2, 4])
    ga    = trial.suggest_categorical("ga", [2, 4])

    model, processor, trainer = None, None, None

    try:
        model, processor = load_model(r, alpha)
        train_dataset = make_train(processor, bs)

        trial_output_dir = f"trial_{trial.number}"
        os.makedirs(trial_output_dir, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=trial_output_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            max_steps=30,
            learning_rate=lr,
            eval_strategy="steps",
            eval_steps=10,
            logging_steps=20,
            report_to="none",
            bf16=is_bf16_supported(),
            fp16=not is_bf16_supported(),
            predict_with_generate=True,
            generation_max_length=MAX_LABEL_LEN,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            data_collator=SpeechCollator(processor),
            compute_metrics=lambda p: compute_metrics(p, processor),
            args=training_args,
        )

        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(f"{trial_output_dir}/final_model")

        return metrics["eval_wer"]

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 1.0

    finally:
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()


#Run Optuna
study = optuna.create_study(
    direction="minimize",
    storage="sqlite:///whisper_optuna.db",
    study_name="whisper_optimization_v1.6"
)

def save_optuna_study(study, trial):
    checkpoint = f"optuna_checkpoint_trial_{trial.number}"
    os.makedirs(checkpoint, exist_ok=True)

    with open(f"{checkpoint}/study.json", "w") as f:
        f.write(str(study.trials_dataframe()))

    with open(f"{checkpoint}/best_params.txt", "w") as f:
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Best value: {study.best_value}\n")

    print(f"Saved study at trial {trial.number}")

study.optimize(objective, n_trials=4, callbacks=[save_optuna_study])

print("BEST PARAMS:", study.best_trial.params)