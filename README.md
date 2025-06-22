# Chaplin

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Runs fully locally.

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) project.

Watch a demo of Chaplin [here](https://youtu.be/qlHi0As2alQ).

## Setup

1. Clone the repository, and `cd` into it:
   ```bash
   git clone https://github.com/amanvirparhar/chaplin
   cd chaplin
   ```
2. Download the required model components: [LRS3_V_WER19.1](https://drive.google.com/file/d/1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6/view) and [lm_en_subword](https://drive.google.com/file/d/1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv/view).
3. Unzip both folders, and place them in their respective directories:
   ```
   chaplin/
   ├── benchmarks/
       ├── LRS3/
           ├── language_models/
               ├── lm_en_subword/
           ├── models/
               ├── LRS3_V_WER19.1/
   ├── ...
   ```
4. Install and run `ollama`, and pull the [`llama3.2`](https://ollama.com/library/llama3.2) model.
5. Install [`uv`](https://github.com/astral-sh/uv).
6. Create a .env file in the home directory. And add your OPENAI_API_KEY=""

## Usage

1. Run the following command:
   ```bash
   uv run --with-requirements requirements.txt --python 3.12 main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
   ```
2. Once the camera feed is displayed, you can start "recording" by pressing the `option` key (Mac) or the `alt` key (Windows/Linux), and start mouthing words.
3. To stop recording, press the `option` key (Mac) or the `alt` key (Windows/Linux) again. You should see some text being typed out wherever your cursor is.
4. To exit gracefully, focus on the window displaying the camera feed and press `q`.
