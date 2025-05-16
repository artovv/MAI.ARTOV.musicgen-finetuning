# 🎶 Дообучение MusicGen на жанре Blues

Этот проект демонстрирует дообучение модели [MusicGen](https://huggingface.co/facebook/musicgen-melody) на целевом датасете.

## Основные особенности

- Используется [PEFT (LoRA)](https://github.com/huggingface/peft) для эффективного дообучения.
- Датасет: [`ylacombe/music_genres_Blues`](https://huggingface.co/datasets/ylacombe/music_genres_Blues)
- Аудио автоматически аннотируются с помощью модели [CLAP](https://github.com/LAION-AI/CLAP) (инструменты, настроение, темп, тональность).

##  Результаты
-  Дообученная модель: [artovv/musicgen-melody-blues1](https://huggingface.co/artovv/musicgen-melody-blues1)

##  Содержимое

- `musicgen_finetuning_with_readme.ipynb` — весь pipeline по дообучению
- `after_finetuning.wav` — пример сгенерированного трека
- 
