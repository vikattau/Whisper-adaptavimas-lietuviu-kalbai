import ast
import matplotlib.pyplot as plt

logs = []
with open("liepa2_further_222464.out", "r") as f:
    for line in f:
        line = line.strip()
        
        if line.startswith("{") and line.endswith("}"):
            logs.append(ast.literal_eval(line))

train_steps, train_loss = [], []
eval_steps, eval_loss = [], []
eval_wer = []

step = 0

for log in logs:
    if "loss" in log:
        train_steps.append(step)
        train_loss.append(log["loss"])
        step += 200

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
plt.title("Antro modelio mokymo ir validacijos paklaidos per žingsnius")
plt.legend()
plt.grid()
plt.savefig("loss_grafikas_m2.pdf", bbox_inches="tight")
plt.show()

# WER grafikas
plt.figure()

plt.plot(eval_steps[:len(eval_wer)], eval_wer, label="Validacijos WER")

plt.xlabel("Žingsniai")
plt.ylabel("WER")
plt.title("Antro modelio klaidos rodiklis (WER) per žingsnius")
plt.legend()
plt.savefig("wer_grafikas_m2.pdf", bbox_inches="tight")
plt.grid()

plt.show()