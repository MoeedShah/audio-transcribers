import streamlit as st
import whisper
import tempfile
import shutil
import os
import sys
import time
import datetime
from pathlib import Path

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="Audio → English Transcript (Whisper)", layout="wide")
st.title("🎙️ Audio → English Transcript (Whisper, no pydub)")

# ---------------- Helpers ----------------
def is_ffmpeg_available() -> bool:
    """Check if ffmpeg exists on PATH."""
    return shutil.which("ffmpeg") is not None

def set_ffmpeg_path():
    """Force add custom ffmpeg bin path if not already available."""
    ffmpeg_dir = r"C:\ffmpeg-8.0-full_build\bin"
    if not is_ffmpeg_available():
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def segments_to_srt(segments: list) -> str:
    """Convert whisper segments list to SRT formatted string."""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end   = format_timestamp(seg["end"])
        text  = seg.get("text", "").strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line
    return "\n".join(lines)

@st.cache_resource
def load_whisper_model(model_size: str):
    """Load and cache the Whisper model (takes time on first load)."""
    try:
        model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model '{model_size}': {e}")
        raise

# ---------------- Sidebar - user options ----------------
st.sidebar.header("Settings")
model_size = st.sidebar.selectbox("Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1)
st.sidebar.markdown(
    "- `tiny`, `base`: faster on CPU, less accurate.\n"
    "- `small`, `medium`: better accuracy; use GPU if available.\n"
    "- `large`: best accuracy but heavy on RAM/GPU."
)
st.sidebar.write("")
st.sidebar.markdown("**Outputs**")
show_segments = st.sidebar.checkbox("Show segments with timestamps", value=True)
enable_txt_download = st.sidebar.checkbox("Enable TXT download", value=True)
enable_srt_download = st.sidebar.checkbox("Enable SRT download", value=True)

# ---------------- ffmpeg check ----------------
set_ffmpeg_path()
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    st.warning(
        "⚠️ ffmpeg not detected. Whisper may fail to decode audio/video. "
        "Please check your ffmpeg installation."
    )
else:
    st.success(f"✅ ffmpeg found: {ffmpeg_path}")

# ---------------- File uploader ----------------
uploaded = st.file_uploader(
    "Upload an audio or video file (mp3, wav, m4a, mp4, ogg, flac, webm, mov, etc.)",
    type=["mp3","wav","m4a","flac","ogg","webm","mp4","mov","avi","mkv"],
)

if uploaded is None:
    st.info("Upload a file to begin transcription. For long files prefer 'small' or 'medium' model and GPU.")
    st.divider()

# ---------------- Main flow ----------------
if uploaded is not None:
    file_size_mb = round(len(uploaded.getbuffer()) / (1024*1024), 2)
    st.write(f"**File:** {uploaded.name} — **{file_size_mb} MB**")

    suffix = Path(uploaded.name).suffix or ".tmp"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp_file.write(uploaded.getbuffer())
        tmp_file.flush()
        tmp_file.close()
        temp_path = tmp_file.name
    except Exception as e:
        st.error(f"Failed to save uploaded file to disk: {e}")
        raise

    st.info("File saved to temporary path. Loading model...")

    try:
        model = load_whisper_model(model_size)
    except Exception:
        try: os.remove(temp_path)
        except Exception: pass
        st.stop()

    transcribe_btn = st.button("🚀 Start Transcription (English)")
    if transcribe_btn:
        start_time = time.time()
        try:
            with st.spinner("Transcribing — this can take a while depending on model and hardware..."):
                result = model.transcribe(temp_path, language="en", temperature=0.0)
        except RuntimeError as re:
            st.error(f"Runtime error during transcription: {re}")
            result = None
        except Exception as exc:
            st.error(f"An error occurred during transcription: {exc}")
            result = None

        elapsed = time.time() - start_time
        if result is None:
            st.warning("Transcription failed. Check ffmpeg installation and model compatibility.")
        else:
            st.success(f"Transcription finished in {elapsed:.1f} seconds.")
            transcript_text = result.get("text", "").strip()
            segments = result.get("segments", [])

            st.subheader("Full Transcript (English)")
            st.text_area("Transcript", value=transcript_text, height=360)

            if show_segments and segments:
                st.subheader("Segments")
                for seg in segments:
                    start = str(datetime.timedelta(seconds=int(seg["start"])))
                    end = str(datetime.timedelta(seconds=int(seg["end"])))
                    st.markdown(f"**{start} → {end}**  \n{seg.get('text','').strip()}")

            if enable_txt_download:
                try:
                    st.download_button(
                        label="📄 Download TXT",
                        data=transcript_text.encode("utf-8"),
                        file_name=f"{Path(uploaded.name).stem}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Failed to create TXT download: {e}")

            if enable_srt_download and segments:
                try:
                    srt_content = segments_to_srt(segments)
                    st.download_button(
                        label="🎬 Download SRT",
                        data=srt_content.encode("utf-8"),
                        file_name=f"{Path(uploaded.name).stem}.srt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Failed to create SRT download: {e}")

        try: os.remove(temp_path)
        except Exception: pass

# ---------------- Footer ----------------
st.divider()
st.markdown(
    "**Tips:**\n\n"
    "- If transcription is very slow on CPU, try `tiny` or `base` for testing.\n"
    "- For higher accuracy on long/complex audio, use `small` or `medium` and a machine with a GPU.\n"
    "- Make sure `ffmpeg -version` works in your terminal before running the app.\n"
)
print("ffmpeg path detected:", ffmpeg_path)
