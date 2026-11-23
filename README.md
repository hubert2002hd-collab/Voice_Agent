# 🎙️ AI Voice Agent – Audio Transcription & Translation

An AI-powered voice transcription application built with **Python, Streamlit, Groq API, and Whisper**. Upload audio or video recordings and generate transcripts with timestamps, language detection, and speaker labels.

## 🚀 Features

- **Multiple File Upload:** Upload multiple audio or video recordings.
- **Long Audio Support:** Split long recordings into smaller chunks for processing.
- **AI Transcription:** Generate transcripts using Groq Whisper.
- **Automatic Language Detection:** Identify the language of uploaded recordings.
- **Multilingual Support:** Transcribe supported languages, including Tamil, English, Hindi, Malayalam, Telugu, and more.
- **Timestamped Transcripts:** Display timestamps alongside spoken text.
- **Speaker Identification:** Assign speaker labels where supported by the application's speaker-detection method.
- **Translation:** Translate transcripts into selected languages when enabled.
- **Batch Processing:** Process multiple files with individual statuses.
- **Error Handling:** Display errors for files that fail during processing.
- **Transcript Export:** Save or export available transcript formats.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Groq API | AI inference and transcription |
| Whisper | Speech-to-text transcription |
| Pydub | Audio processing and splitting |
| FFmpeg | Audio/video conversion |
| Python-dotenv | Environment variable management |
| FPDF | PDF generation, if enabled |

## 📁 Project Structure

```text
Voice_Agent/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
└── error_log.txt
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/hubert2002hd-collab/Voice_Agent.git
```

### 2. Open the project folder

```bash
cd Voice_Agent
```

### 3. Create a virtual environment

```powershell
python -m venv venv
```

### 4. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell activation is restricted, you can run the environment's Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Groq API Configuration

1. Create an API key through the official Groq Console.
2. Create a `.env` file in the project root.
3. Add your API key and the model settings expected by your application.

Example:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_TEXT_MODEL=openai/gpt-oss-120b
```

**Important:** Never upload your real `.env` file or API key to GitHub.

## 🎧 FFmpeg Installation

Pydub may require FFmpeg to process audio and video files.

Install FFmpeg on Windows using:

```powershell
winget install Gyan.FFmpeg
```

Restart PowerShell and verify:

```powershell
ffmpeg -version
```

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🔄 How It Works

1. Open the Voice Agent application.
2. Upload one or more audio/video recordings.
3. Select the available transcription and translation settings.
4. The application processes recordings and splits long audio into chunks.
5. Groq Whisper generates the transcription.
6. The application displays the transcript and timestamps.
7. Speaker labels are assigned using the implemented speaker-detection method.
8. If translation is enabled, the transcript is translated into the selected language.
9. Export the available transcript output.

## 🌐 Language Codes

Groq Whisper expects language codes rather than full language names.

| Language | Code |
|---|---|
| Tamil | `ta` |
| English | `en` |
| Hindi | `hi` |
| Malayalam | `ml` |
| Telugu | `te` |
| Kannada | `kn` |
| Bengali | `bn` |
| Arabic | `ar` |
| French | `fr` |
| Spanish | `es` |

For example, use `ta` instead of `Tamil`.

## 🐛 Troubleshooting

### 1. Groq API Connection Error

Possible causes:
- Unstable internet connection.
- Firewall or antivirus interference.
- Large audio uploads.
- Network connection interruption.

Check connectivity:

```powershell
Test-NetConnection api.groq.com -Port 443
```

### 2. Unsupported Language Error

Use the correct language code.

Incorrect:

```python
kwargs["language"] = "Tamil"
```

Correct:

```python
kwargs["language"] = "ta"
```

### 3. FFmpeg Not Found

Check installation:

```powershell
ffmpeg -version
```

Restart your terminal after installing FFmpeg.

### 4. Missing Python Packages

Install the requirements again:

```powershell
pip install -r requirements.txt
```

### 5. Streamlit Application Error

Restart the application:

```powershell
streamlit run app.py
```

Check the terminal output for the exact error message.

## 🔐 Security

- Keep API keys private.
- Never commit `.env` to GitHub.
- Do not share API credentials in screenshots or error reports.
- Rotate your API key if it is accidentally exposed.

Recommended `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
error_log.txt
```

## 👨‍💻 Developer

**Hubert Donald**

AI Voice Agent Project

GitHub: https://github.com/hubert2002hd-collab/Voice_Agent

## 📜 License

This project is intended for educational and development purposes. Add a formal open-source license before redistributing the project.

---

**Built with Python, Streamlit, Groq API and AI-powered speech recognition.** 🎙️