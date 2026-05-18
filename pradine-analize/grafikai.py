import csv
import matplotlib.pyplot as plt
import pandas as pd

bin_left = []
bin_right = []
counts = []

with open("histogram.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bin_left.append(float(row["bin_left"]))
        bin_right.append(float(row["bin_right"]))
        counts.append(int(row["count"]))

widths = [r - l for l, r in zip(bin_left, bin_right)]

# Bendri šriftų nustatymai visiems grafikams
plt.rcParams.update({
    'axes.titlesize': 18,     # pavadinimo dydis
    'axes.labelsize': 16,     # x ir y ašių pavadinimai
    'xtick.labelsize': 14,    # x ašies reikšmės
    'ytick.labelsize': 14,    # y ašies reikšmės
    'legend.fontsize': 14     # legendos tekstas
})


plt.figure(figsize=(8, 5))
plt.bar(
    bin_left,
    counts,
    width=widths,
    edgecolor="black"
)

plt.xlabel("Trukmė (sek.)")
plt.ylabel("Įrašų skaičius")
plt.title("Garso įrašų trukmės pasiskirstymas")

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("irasu_histograma.pdf", bbox_inches="tight")
plt.show()

colors = ["#28A197", "#801650", "#F46A25", "#FAC205", "#A285D1", "#12436D"]

plt.figure(figsize=(6, 6))

df = pd.read_csv("noise_counts.csv")
labels = df["is_noise"].astype(str)
counts = df["count"]

labels = [
    "Triukšmas" if x == "True"
    else "Ne triukšmas"
    for x in labels
]

plt.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    textprops={'fontsize': 14}
)

plt.title("Triukšmo ir ne triukšmo pasiskirstymas")

plt.savefig("noise_pie10.pdf", bbox_inches="tight")
plt.show()

def autopct_func(pct):
    return f"{pct:.1f}%" if pct > 5 else ""

labels = [
    "Įrašai studijoje",
    "Diktofono įrašai",
    "Audio knygos",
    "TV transliacijos",
    "Radijo transliacijos",
    "Telefoniniai įrašai"
]

counts = [62, 28, 3, 3, 2, 2]

plt.figure(figsize=(8, 8))

plt.pie(
    counts,
    autopct=autopct_func,
    colors=colors,
    startangle=90,
    textprops={'fontsize': 14}
)

plt.title("Įrašų pasiskirstymas pagal įrašymo aplinką")

plt.legend(
    labels,
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.savefig("recording_environments.pdf", bbox_inches="tight")
plt.show()

labels = [
    "Iki 12 metų",
    "13-17 metų",
    "18-25 metų",
    "26-60 metų",
    "Virš 60 metų"
]

counts = [8, 3, 18, 61, 10]

plt.figure(figsize=(8, 8))

plt.pie(
    counts,
    autopct=autopct_func,
    colors=colors,
    startangle=90,
    textprops={'fontsize': 14}
)

plt.title("Įrašų pasiskirstymas pagal kalbėtojo amžių")

plt.legend(
    labels,
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.savefig("amzius.pdf", bbox_inches="tight")
plt.show()