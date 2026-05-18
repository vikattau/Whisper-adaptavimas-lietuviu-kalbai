import ast
import matplotlib.pyplot as plt

logs = []
with open("logs213187.out", "r") as f:
    for line in f:
        logs.append(ast.literal_eval(line.strip()))

train_steps, train_loss = [], []
eval_steps, eval_loss = [], []
eval_wer = []

step = 0

for log in logs:
    if "loss" in log:
        train_steps.append(step)
        train_loss.append(log["loss"])
        step += 20

    elif "eval_loss" in log:
        eval_steps.append(step)
        eval_loss.append(log["eval_loss"])
        
        if "eval_wer" in log:
            eval_wer.append(log["eval_wer"])

import matplotlib.pyplot as plt

# Paklaidų grafikas (Loss)
plt.figure()

plt.plot(train_steps, train_loss, label="Mokymo paklaida")
plt.plot(eval_steps, eval_loss, label="Validacijos paklaida")

plt.xlabel("Žingsniai")
plt.ylabel("Paklaida")
plt.title("Mokymo ir validacijos paklaidos pagal žingsnį")
plt.legend()
plt.grid()
plt.savefig("loss_grafikas.pdf", bbox_inches="tight")
plt.show()

# WER grafikas
plt.figure()

plt.plot(eval_steps[:len(eval_wer)], eval_wer, label="Validacijos WER")

plt.xlabel("Žingsniai")
plt.ylabel("WER")
plt.title("Klaidos rodiklis (WER) per žingsnius")
plt.legend()
plt.savefig("wer_grafikas.pdf", bbox_inches="tight")
plt.grid()

plt.show()