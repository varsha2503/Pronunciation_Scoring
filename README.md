# 🗣️ Pronunciation Scoring System

This project implements a **context-aware pronunciation scoring system** that evaluates the spoken English of a user by comparing it with native pronunciation using deep learning and signal processing techniques.

It generates detailed feedback on articulation (phoneme-level), fluency (pitch and rhythm), and overall pronunciation quality using a combination of **GOP (Goodness of Pronunciation)** scoring and **DTW (Dynamic Time Warping)**-based fluency analysis.

---

## 🚀 Features

- 🎙️ Real-time user audio recording  
- 🧠 Wav2Vec2-based GOP scoring (traditional + improved)  
- 🎵 DTW-based pitch and intensity fluency analysis  
- 📊 Phoneme, word, and sentence-level scoring  
- 📌 Pronunciation feedback and tips  
- ⚖️ Tunable final score (accuracy + fluency balance via α)

---

## 📁 Project Structure


- `audio/` – Stores user and native audio samples  
- `alignments/` – Contains TextGrid output from Montreal Forced Aligner  
- `dict/` – Pronunciation dictionary (lexicon) used for forced alignment  
- `models/` – Pretrained MFA acoustic models  
- `scoring.py` – Main pronunciation scoring and feedback generation script  
- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation


---

## ⚙️ Installation

### 🔹 1. Clone the repository

```bash
git clone https://github.com/varsha2503/Pronunciation_Scoring.git
cd Pronunciation_Scoring
```

### 🔹 2. Create and activate virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 🔹 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 🔹 4. Install and configure Montreal Forced Aligner (MFA)

Follow official MFA installation instructions:  
📦 [https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)

Ensure MFA is available in your system's PATH by running:

```bash
mfa --version
```

## Usage

```bash
python scoring.py
```
🔧 During execution:
- Press 'r' to start recording
- Press 's' to stop recording
- Native reference audio is auto-generated
- Alignment and scoring are done automatically
- Final report is printed in the terminal


---

## 📈 Output Report Includes:

- ✅ Traditional & Context-aware GOP Scores  
- ✅ DTW Fluency Scores:  
  - Intensity-based  
  - Pitch delta-based  
  - Autocorrelation-based  
- ✅ Accuracy Score (phoneme-wise correctness)  
- ✅ Word and sentence-level scores  
- ⚠️ Detailed phoneme feedback (with tips if weak)  
- 🎯 Final Combined Score (configurable with α)

---

## 🧠 How it Works

- Uses [Wav2Vec2](https://pytorch.org/audio/stable/pipelines.html#wav2vec2-asr) model to compute phoneme emission probabilities  
- Computes GOP using:
  - Traditional posterior probability averaging  
  - Context-aware log-probability smoothing  
- Applies DTW (Dynamic Time Warping) on:
  - **Pitch curves** (intonation)
  - **Pitch deltas** (variation)
  - **Energy/intensity** patterns (to assess fluency)
- Uses Montreal Forced Aligner (MFA) to get precise phoneme and word alignment from `.wav` + `.lab`
- Final score combines accuracy and fluency using a weighted average (`α`)

---

## 📌 Example Output
```bash
💬 Fluency Score DTW (intensity)   : 42.5%
💬 Fluency Score DTW (pitch delta) : 34.1%
💬 Fluency Score DTW (autocorr)    : 38.9%
💬 Accuracy Score                  : 87.6%
🎯 Final Combined Score (α=0.5)     : 61.3%
📄 Sentence Score: 72.4%

🔤 Phoneme Scores:
 - t: 85% ✅
 - ɪ: 58% ⚠️
 - s: 90% ✅

📌 Feedback:
⚠️ Phonemes needing improvement:
 - ɪ: 58%

💡 Tips:
 - Speak slowly
 - Mimic native tone
 - Use mirror practice
```


---

## ✅ Requirements

See [`requirements.txt`](requirements.txt) for the full list.

Key libraries used:

- `torch`, `torchaudio`  
- `librosa`, `parselmouth`, `pydub`  
- `gtts`, `tgt`, `scipy`  
- `sounddevice`, `keyboard`

Ensure you have [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) installed and available in your PATH:

```bash
mfa --version
```

## 📌 Future Enhancements
- 🌐 Add web UI (Streamlit/Flask)
- 📊 Visual pitch and intensity curve plotting
- 🈶 Support tonal language pronunciation scoring (e.g., Mandarin)
- 📱 Mobile and web deployment options
- 🧪 Multi-sentence and paragraph-level evaluation
- 🎧 Add real-time waveform display

## 📜 License
This project is provided for educational and research purposes.
Feel free to use, modify, and extend with attribution.

---
