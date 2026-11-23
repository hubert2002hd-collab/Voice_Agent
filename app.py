import json
import os
import re
import io
import time
from groq import Groq, APIConnectionError
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
from fpdf import FPDF

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Voice Agent",
    page_icon="🎙️",
    layout="wide",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=300.0,
    max_retries=0
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #c471f5, #fa71cd, #ffffff);
    }
    [data-testid="stSidebar"] {
        background: #2e5339;
    }
    .main .block-container {
        padding-top: 2rem;
    }
   div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #2e5339;
}

/* Make all metric box text black */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #000000 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
st.sidebar.title("⚙️ Agent Settings")

language_override = st.sidebar.selectbox(
    "Manual language override",
    [
        "Auto", "English", "Tamil", "Hindi", "Malayalam",
        "Telugu", "Kannada", "French", "Spanish",
    ],
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Maximum output tokens", 256, 8192, 2048, 256)
chunk_minutes = st.sidebar.selectbox(
    "Audio chunk duration (mins)",
    [1, 2, 3],
    index=1
)

if st.sidebar.button("🗑️ Clear session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --------------------------------------------------
# LANGUAGE MAPPING
# --------------------------------------------------

def normalize_language_code(language):
    if not language:
        return None

    language_map = {
        "tamil": "ta",
        "english": "en",
        "hindi": "hi",
        "malayalam": "ml",
        "telugu": "te",
        "kannada": "kn",
        "arabic": "ar",
        "french": "fr",
        "spanish": "es",
        "german": "de",
        "japanese": "ja",
        "chinese": "zh",
        "korean": "ko",
        "urdu": "ur",
        "bengali": "bn",
        "marathi": "mr",
        "gujarati": "gu",
        "punjabi": "pa",
    }

    language = str(language).strip().lower()

    return language_map.get(language, language)


CODE_TO_LANGUAGE = {
    "en": "English",
    "english": "English",
    "ta": "Tamil",
    "tamil": "Tamil",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ml": "Malayalam",
    "malayalam": "Malayalam",
    "te": "Telugu",
    "telugu": "Telugu",
    "kn": "Kannada",
    "kannada": "Kannada",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
}

TRANSLATION_LANGUAGES = ["English", "Tamil", "Hindi", "Malayalam"]


def normalize_language_name(raw_lang, override_choice):
    if override_choice != "Auto":
        return override_choice
    if not raw_lang:
        return "Unknown"
    cleaned = str(raw_lang).strip().lower()
    return CODE_TO_LANGUAGE.get(cleaned, cleaned.capitalize())


# --------------------------------------------------
# COMPATIBILITY HELPERS
# --------------------------------------------------
def get_value(item, key, default=None):
    """Read a field from either a dict or an SDK/Pydantic object."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def safe_filename(name):
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "recording"


def format_timestamp(seconds):
    seconds = max(0.0, float(seconds or 0))
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:05.2f}"


def sanitize_for_core_font(text):
    """Fallback sanitizer only if no Unicode TTF font is found on the system."""
    if not text:
        return ""
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    return text.encode("latin-1", "replace").decode("latin-1")


# --------------------------------------------------
# AUDIO PROCESSING
# --------------------------------------------------

def load_audio(file_bytes, extension):
    return AudioSegment.from_file(
        io.BytesIO(file_bytes),
        format=extension
    )


def split_audio(audio):
    chunk_length_ms = chunk_minutes * 60 * 1000
    chunks = []

    for start_ms in range(0, len(audio), chunk_length_ms):
        end_ms = min(start_ms + chunk_length_ms, len(audio))
        chunks.append((start_ms, audio[start_ms:end_ms]))

    return chunks



def audio_to_wav(audio):
    # Normalize audio to reduce upload size and improve compatibility.
    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
        .set_sample_width(2)
    )

    wav_file = io.BytesIO()
    audio.export(wav_file, format="wav")
    wav_file.seek(0)
    return wav_file


def transcribe_chunk(chunk, language=None):
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    for attempt in range(4):
        wav_file = audio_to_wav(chunk)

        kwargs = {
            "file": ("audio.wav", wav_file, "audio/wav"),
            "model": WHISPER_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }

        if language:
            language_code = normalize_language_code(language)

            if language_code:
                kwargs["language"] = language_code

        try:
            return client.audio.transcriptions.create(**kwargs)

        except APIConnectionError as error:
            if attempt == 3:
                raise RuntimeError(
                    "Groq connection failed after 4 attempts. "
                    "Check your internet connection."
                ) from error

            wait_seconds = 3 * (2 ** attempt)
            time.sleep(wait_seconds)

def transcribe_audio_file(
    file_bytes,
    extension,
    manual_language_code=None,
    override_label="Auto",
    progress_callback=None
):
    audio = load_audio(file_bytes, extension)
    chunks = split_audio(audio)

    segments = []
    detected_raw_language = manual_language_code

    for index, (offset_ms, chunk) in enumerate(chunks):
        response = transcribe_chunk(chunk, detected_raw_language)

        if not detected_raw_language:
            detected_raw_language = get_value(response, "language")

        response_segments = get_value(response, "segments", []) or []

        for item in response_segments:
            try:
                start_seconds = (
                    float(get_value(item, "start", 0))
                    + offset_ms / 1000
                )
                end_seconds = (
                    float(get_value(item, "end", 0))
                    + offset_ms / 1000
                )
            except (TypeError, ValueError):
                continue

            text_value = get_value(item, "text", "") or ""

            segments.append({
                "start": start_seconds,
                "end": end_seconds,
                "text": str(text_value).strip() or "(inaudible)"
            })

        if progress_callback:
            progress_callback((index + 1) / max(1, len(chunks)))

    segments.sort(key=lambda segment: segment["start"])

    speaker_count, segments = detect_and_assign_speakers(segments)

    display_language = normalize_language_name(
        detected_raw_language,
        override_label
    )

    return {
        "language": display_language,
        "duration": len(audio) / 1000,
        "speaker_count": speaker_count,
        "segments": segments,
    }


# --------------------------------------------------
# INTELLIGENT SPEAKER DETECTION & FORMATTING
# --------------------------------------------------
def detect_and_assign_speakers(segments):
    """
    Uses LLM conversational turn analysis to distinguish 1 speaker (monologue)
    from 2 speakers (dialogue), falling back to pause-gap heuristics if needed.
    """
    if not segments:
        return 1, []

    if len(segments) == 1:
        segments[0]["speaker"] = "Speaker 1"
        return 1, segments

    # Prepare compact indexed transcript for speaker turn classification
    indexed_lines = []
    for idx, seg in enumerate(segments):
        gap = 0.0 if idx == 0 else max(0.0, seg["start"] - segments[idx - 1]["end"])
        indexed_lines.append(f"[{idx}] (pause={gap:.1f}s) {seg['text']}")

    prompt = (
        "Analyze the following timestamped speech segments and determine if there is "
        "1 speaker (monologue/narration) or 2 speakers (dialogue/conversation).\n"
        "Look at questions and answers, greetings, topic shifts, and pause durations.\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{"speaker_count": 1, "assignments": [{"index": 0, "speaker": "Speaker 1"}]}\n'
        "Use 'Speaker 1' for all segments if speaker_count is 1. "
        "Use 'Speaker 1' and 'Speaker 2' if speaker_count is 2.\n\n"
        "SEGMENTS:\n" + "\n".join(indexed_lines[:150])
    )

    try:
        if client is not None:
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert audio diarization analyzer. Output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw_json = response.choices[0].message.content or ""
            # Extract JSON block if wrapped in markdown fences
            match = re.search(r"\{.*\}", raw_json, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                count = int(parsed.get("speaker_count", 1))
                count = 1 if count <= 1 else 2

                if count == 1:
                    for seg in segments:
                        seg["speaker"] = "Speaker 1"
                    return 1, segments

                mapping = {
                    int(item["index"]): str(item["speaker"])
                    for item in parsed.get("assignments", [])
                    if "index" in item and "speaker" in item
                }
                last_speaker = "Speaker 1"
                for idx, seg in enumerate(segments):
                    assigned = mapping.get(idx, last_speaker)
                    seg["speaker"] = assigned
                    last_speaker = assigned

                unique_speakers = {seg["speaker"] for seg in segments}
                return len(unique_speakers), segments
    except Exception:
        pass

    # Fallback heuristic based on pause gaps > 1.8s
    current_speaker = 1
    previous_end = segments[0]["end"]
    segments[0]["speaker"] = "Speaker 1"
    used_speakers = {1}

    for seg in segments[1:]:
        gap = max(0.0, seg["start"] - previous_end)
        if gap > 1.8:
            current_speaker = 2 if current_speaker == 1 else 1
        seg["speaker"] = f"Speaker {current_speaker}"
        used_speakers.add(current_speaker)
        previous_end = max(previous_end, seg["end"])

    return len(used_speakers), segments


def create_transcript(segments, speaker_count):
    """
    If 1 speaker: clean timestamped transcription without repetitive speaker labels.
    If 2+ speakers: structured dialogue format with Speaker 1 / Speaker 2 labels.
    """
    lines = []
    if speaker_count == 1:
        for segment in segments:
            start = format_timestamp(segment.get("start", 0))
            end = format_timestamp(segment.get("end", 0))
            text = segment.get("text", "")
            lines.append(f"[{start} - {end}] {text}")
    else:
        for segment in segments:
            start = format_timestamp(segment.get("start", 0))
            end = format_timestamp(segment.get("end", 0))
            speaker = segment.get("speaker", "Speaker 1")
            text = segment.get("text", "")
            lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines)


# --------------------------------------------------
# GROQ TRANSLATION AND SUMMARY
# --------------------------------------------------
def groq_text(prompt):
    if client is None:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional meeting minutes assistant. "
                    "Preserve meaning and never invent dialogue or decisions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def translate_transcript(transcript, target):
    return groq_text(
        f"""Translate the following transcript into {target}.

Preserve all timestamps, speaker labels (if present), original meaning, and order.
Do not omit any dialogue.

TRANSCRIPT:
{transcript}
"""
    )


def generate_summary(transcript):
    return groq_text(
        f"""Create professional meeting minutes from this transcript.

Include:
1. Meeting overview
2. Main discussion points
3. Important decisions
4. Action items
5. Follow-up points

Do not invent information. Mark missing information as Not specified.

TRANSCRIPT:
{transcript}
"""
    )


# --------------------------------------------------
# A4 PDF GENERATION (WITH UNICODE SUPPORT)
# --------------------------------------------------

def find_unicode_font():
    """Find a Unicode TrueType font on Windows or Linux."""
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalaUI.ttf",
        r"C:\Windows\Fonts\arialuni.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path

    return None


class A4TranscriptPDF(FPDF):
    def __init__(self, font_family_name):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.font_family_name = font_family_name

    def header(self):
        self.set_font(self.font_family_name, size=9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "AI Voice Agent - A4 Transcription Report", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font(self.font_family_name, size=9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")


def create_pdf(filename, language, duration, transcript, summary, speaker_count):
    font_path = find_unicode_font()

    if font_path:
        font_name = "CustomUnicode"
        use_unicode = True
    else:
        font_name = "Helvetica"
        use_unicode = False

    pdf = A4TranscriptPDF(font_name)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()

    if use_unicode:
        pdf.add_font(font_name, style="", fname=font_path)

    pdf.add_page()

    def prep(value):
        value = str(value or "")
        return value if use_unicode else sanitize_for_core_font(value)

    printable_width = 180

    # Title
    pdf.set_fill_color(46, 83, 57)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font_name, size=15)
    pdf.cell(
        printable_width,
        12,
        prep("MEETING TRANSCRIPTION REPORT"),
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT"
    )
    pdf.ln(4)

    # Recording information
    speaker_label = (
        "1 Speaker (Monologue)"
        if speaker_count == 1
        else f"{speaker_count} Speakers (Dialogue)"
    )

    pdf.set_text_color(30, 30, 30)
    pdf.set_fill_color(242, 245, 243)
    pdf.set_font(font_name, size=10)

    details = [
        f"Recording File: {filename}",
        f"Detected Language: {language}",
        f"Detected Speakers: {speaker_label}",
        f"Duration: {format_timestamp(duration)}",
        f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
    ]

    for detail in details:
        pdf.cell(
            printable_width,
            7,
            prep(detail),
            border="LR",
            fill=True,
            new_x="LMARGIN",
            new_y="NEXT"
        )

    pdf.cell(printable_width, 1, "", border="T", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Meeting summary
    if summary:
        pdf.set_text_color(46, 83, 57)
        pdf.set_font(font_name, size=13)
        pdf.cell(
            printable_width,
            8,
            prep("MEETING SUMMARY"),
            new_x="LMARGIN",
            new_y="NEXT"
        )

        pdf.set_draw_color(46, 83, 57)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(3)

        pdf.set_text_color(20, 20, 20)
        pdf.set_font(font_name, size=10)

        for line in summary.splitlines():
            clean_line = (
                line.replace("**", "")
                .replace("###", "")
                .replace("##", "")
                .strip()
            )

            if clean_line:
                pdf.multi_cell(
                    printable_width,
                    6,
                    prep(clean_line),
                    new_x="LMARGIN",
                    new_y="NEXT"
                )
            else:
                pdf.ln(2)

        pdf.ln(6)

    # Full transcript
    pdf.set_text_color(46, 83, 57)
    pdf.set_font(font_name, size=13)

    header_title = (
        "TRANSCRIPTION (MONOLOGUE)"
        if speaker_count == 1
        else "DIALOGUE TRANSCRIPTION"
    )

    pdf.cell(
        printable_width,
        8,
        prep(header_title),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_draw_color(46, 83, 57)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_text_color(20, 20, 20)
    pdf.set_font(font_name, size=10)

    for line in transcript.splitlines():
        if line.strip():
            pdf.multi_cell(
                printable_width,
                6,
                prep(line),
                new_x="LMARGIN",
                new_y="NEXT"
            )
            pdf.ln(1)
        else:
            pdf.ln(3)

    return bytes(pdf.output())



# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------
st.title("🎙️ AI Voice Agent")
st.caption(
    "Automatic Language & Speaker Detection | Monologue & Dialogue Formatting | A4 PDF Export"
)
st.info(
    "Upload audio or video files to automatically detect the spoken language, count speakers, and generate an A4 PDF."
)

uploaded_files = st.file_uploader(
    "Upload audio or video recordings",
    type=["mp3", "wav", "mp4", "m4a"],
    accept_multiple_files=True,
)

st.subheader("Output Options")
enable_translation = st.checkbox(
    "Translate transcription after original-language transcription"
)
translation_language = st.selectbox("Select target translation language", TRANSLATION_LANGUAGES)
enable_summary = st.checkbox("Generate meeting summary", value=True)

if uploaded_files and st.button(
    "🚀 Start Transcription",
    type="primary",
    use_container_width=True,
):
    if not client:
        st.error("Add GROQ_API_KEY=your_key to the .env file, then restart the app.")
        st.stop()

    batch_results = []

    for file_index, uploaded in enumerate(uploaded_files):
        safe_stem = safe_filename(uploaded.name)
        extension = Path(uploaded.name).suffix.lower().lstrip(".")

        st.divider()
        st.subheader(f"📁 {uploaded.name}")

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Detecting language, counting speakers, and transcribing audio...")

            manual_language_code = (
                LANGUAGES.get(language_override)
                if language_override != "Auto"
                else None
            )

            result = transcribe_audio_file(
                uploaded.getvalue(),
                extension,
                manual_language_code,
                language_override,
                lambda value: progress.progress(value),
            )

            segments = result["segments"]
            detected_language = result["language"]
            duration = result["duration"]
            speaker_count = result["speaker_count"]
            transcript = create_transcript(segments, speaker_count)

            summary = ""
            if enable_summary and transcript:
                status.info("Generating meeting summary...")
                summary = generate_summary(transcript)

            translated = ""
            if enable_translation and transcript:
                status.info("Translating transcript...")
                translated = translate_transcript(transcript, translation_language)

            final_transcript = transcript
            if translated:
                final_transcript += (
                    f"\n\n--- TRANSLATION ({translation_language}) ---\n\n{translated}"
                )

            txt_bytes = final_transcript.encode("utf-8")
            txt_path = OUTPUT_DIR / f"{safe_stem}_transcribe.txt"
            txt_path.write_bytes(txt_bytes)

            pdf_bytes = create_pdf(
                uploaded.name,
                detected_language,
                duration,
                final_transcript,
                summary,
                speaker_count,
            )
            pdf_path = OUTPUT_DIR / f"{safe_stem}_transcribe.pdf"
            pdf_path.write_bytes(pdf_bytes)

            progress.progress(1.0)
            status.success("Transcription completed!")

            # Display clear detection metrics
            speaker_display = "1 Speaker" if speaker_count == 1 else f"{speaker_count} Speakers"
            mode_display = "Monologue Format" if speaker_count == 1 else "Dialogue Format"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detected Language", detected_language)
            m2.metric("Detected Speakers", speaker_display)
            m3.metric("Transcript Format", mode_display)
            m4.metric("Duration", format_timestamp(duration))

            transcript_heading = (
                "### Transcribed Audio (1 Speaker)"
                if speaker_count == 1
                else f"### Dialogue Transcript ({speaker_count} Speakers)"
            )
            st.markdown(transcript_heading)
            st.text_area(
                "Transcription Output",
                final_transcript,
                height=300,
                key=f"transcript_{file_index}",
            )

            if summary:
                st.markdown("### Meeting Summary")
                st.write(summary)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download A4 PDF",
                    data=pdf_bytes,
                    file_name=f"{safe_stem}_A4_transcript.pdf",
                    mime="application/pdf",
                    key=f"pdf_{file_index}",
                )
            with col2:
                st.download_button(
                    "📝 Download TXT",
                    data=txt_bytes,
                    file_name=f"{safe_stem}_transcript.txt",
                    mime="text/plain",
                    key=f"txt_{file_index}",
                )

            batch_results.append(
                {
                    "File": uploaded.name,
                    "Status": "Done",
                    "Language": detected_language,
                    "Speakers": speaker_display,
                    "Format": mode_display,
                }
            )


        except Exception as error:
            status.error(f"Failed: {type(error).__name__}: {error}")
            st.exception(error)

            with open("error_log.txt", "a", encoding="utf-8") as log:
                log.write(
                    f"{datetime.now()} | {uploaded.name} | "
                    f"{type(error).__name__}: {error}\n"
                )

            batch_results.append(
                {
                    "File": uploaded.name,
                    "Status": "Failed",
                    "Language": "Unknown",
                    "Speakers": "0",
                    "Format": "N/A",
                }
            )

    st.divider()
    st.subheader("📊 Batch Processing Report")

    headers = ["File", "Status", "Language", "Speakers", "Format"]
    table_html = (
        "<table style='width:100%;border-collapse:collapse;background:#ffffff;'>"
        "<thead><tr>"
        + "".join(
            f"<th style='text-align:left;padding:10px;border:1px solid #cccccc;color:#000000;background:#eeeeee;'>{h}</th>"
            for h in headers
        )
        + "</tr></thead><tbody>"
    )
    for row in batch_results:
        table_html += "<tr>" + "".join(
            "<td style='padding:10px;border:1px solid #cccccc;color:#000000;'>"
            + str(row.get(header, ""))
            + "</td>"
            for header in headers
        ) + "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.success("Batch processing finished.")

else:
    st.caption("Upload one or more recordings to begin.")