# Whisper adaptavimas lietuvių kalbai

Kursinio darbo tema - Šnekos atpažinimo modelio Whisper tinkamumo tyrimas ir adaptavimas lietuvių kalbai

## Kodo struktūra

- `pradine-analize/` – LIEPA-2 garso įrašų segmentų pradinė analizė
- `pradinis-modelio-mokymas-HPC/` – Whisper modelio mokymas atsititiniais LIEPA-2 duomenimis
- `modelio-tobulinimas-duomenu-atranka/` – Whisper modelio apmokymas atrinktais LIEPA-2 duomenimis
- `modeliu-testavimas/` – bazinio bei adaptuotų modelių testavimas autorių kurtais įrašais, LIEPA-2 duomenimis bei VSSA medicininės diktatūros įrašais

Galutinio modelio nuoroda:
https://huggingface.co/domineeka/whisper-large-lt-v1-further-age-gender-2
