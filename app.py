
import queue
import time
import torch
import numpy as np
import sounddevice as sd
from pydub import AudioSegment, silence
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import logging as hf_logging

import cv2
import textwrap
from PIL import Image, ImageDraw, ImageFont

KB_MODEL = "KBLab/kb-whisper-tiny" #tiny / base / small / medium / large


#####################################################
# Subtitle Display Configuration
#####################################################

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CHROMA_KEY_GREEN = (0, 255, 0)  # BGR for OpenCV, but we'll do the same in Pillow as (R,G,B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RECT_HEIGHT = 220

SUBTITLE_MAX_LINES = 2
CHARS_PER_LINE = 52
LINE_SPACING = 52
BOTTOM_MARGIN = 130

# Path to a TTF font that supports å, ä, ö. DejaVuSans is just an example.
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"  # or adjust for your system
FONT_SIZE = 52  # Adjust to taste

# This list will store up to 2 lines of text
subtitle_lines = []


def add_subtitle_text(new_text):
    """
    Breaks new_text into wrapped lines (max 42 chars),
    adds them to subtitle_lines, and ensures we keep only 2 lines total.
    """
    global subtitle_lines
    wrapped = textwrap.wrap(new_text, width=CHARS_PER_LINE)
    for line in wrapped:
        subtitle_lines.append(line)
        while len(subtitle_lines) > SUBTITLE_MAX_LINES:
            subtitle_lines.pop(0)


def create_subtitle_frame(lines):
    # 1) Create the 1920x1080 image in green
    pil_image = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 255, 0))
    draw = ImageDraw.Draw(pil_image)

    # 2) Draw black rectangle at bottom
    black_rect_top = SCREEN_HEIGHT - RECT_HEIGHT
    draw.rectangle([(0, black_rect_top), (SCREEN_WIDTH, SCREEN_HEIGHT)], fill=(0, 0, 0))

    # 3) Set up font & margins
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    y_start = SCREEN_HEIGHT - BOTTOM_MARGIN

    # 4) Draw each line from bottom to top
    for i in range(len(lines)):
        text = lines[len(lines) - 1 - i]
        y = y_start - i * LINE_SPACING

        # Use textbbox to measure text width/height
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top

        # Center horizontally
        x = (SCREEN_WIDTH - text_width) // 2

        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # 5) Convert Pillow (RGB) -> NumPy (BGR) for OpenCV
    open_cv_image = np.array(pil_image)[:, :, ::-1].copy()
    return open_cv_image


#####################################################
# Setup Device and Load Model
#####################################################

hf_logging.set_verbosity_error()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = KB_MODEL

print("Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    cache_dir="cache"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


#####################################################
# Transcription Helper
#####################################################

def transcribe_chunk(chunk: AudioSegment) -> str:
    raw_samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
    raw_samples /= 32767.0

    audio_input = {"array": raw_samples, "sampling_rate": 16000}
    result = asr_pipe(
        audio_input,
        chunk_length_s=30,
        generate_kwargs={"task": "transcribe", "language": "sv"}
    )
    return result["text"]


#####################################################
# Audio Streaming from Microphone
#####################################################

sample_rate = 16000
block_size_ms = 200
block_size = int(sample_rate * (block_size_ms / 1000))

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=block_size,
    callback=audio_callback
)

buffered_segment = AudioSegment.empty()

def is_chunk_silent(chunk: AudioSegment, threshold_db: float = -35) -> bool:
    return chunk.max_dBFS < threshold_db

def transcribe_if_not_silent(chunk):
    if is_chunk_silent(chunk, threshold_db=-35):
        return ""
    else:
        return transcribe_chunk(chunk)



#####################################################
# Main Loop: collect audio, split on silence, transcribe,
# display subtitles
#####################################################

print("Starting audio stream. Press Ctrl+C to stop.")
stream.start()

cv2.namedWindow("Subtitles", cv2.WINDOW_NORMAL)
# If desired, go fullscreen:
# cv2.setWindowProperty("Subtitles", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    print("\n--- Transcription + Subtitles Running ---\n")
    while True:
        # 1) If we have data
        if not audio_queue.empty():
            data_block = audio_queue.get()
            block_int16 = (data_block * 32767).astype(np.int16).tobytes()

            segment = AudioSegment(
                data=block_int16,
                sample_width=2,
                frame_rate=sample_rate,
                channels=1
            )
            buffered_segment += segment

            # 2) Silence splitting
            silence_thresh_db = -40
            min_silence_len_ms = 400
            keep_silence_ms = 100

            chunks = silence.split_on_silence(
                buffered_segment,
                min_silence_len=min_silence_len_ms,
                silence_thresh=silence_thresh_db,
                keep_silence=keep_silence_ms
            )

            # Word that Whisper inserts when silent
            EXCLUDED_TEXTS = {"Tack.", "Tack!", "Ja.", "Musik"}

            if len(chunks) > 1:
                finished_chunks = chunks[:-1]
                for c in finished_chunks:
                    text = transcribe_chunk(c)
                    if text.strip() and text.strip() not in EXCLUDED_TEXTS:
                        print(text)
                        add_subtitle_text(text)
                buffered_segment = chunks[-1]

            # 3) Forced chunk if buffer is > 8 sec
            if len(buffered_segment) > 8000:
                text = transcribe_if_not_silent(buffered_segment)
                if text.strip() and text.strip() not in EXCLUDED_TEXTS:

                    print(text)
                    add_subtitle_text(text)
                buffered_segment = AudioSegment.empty()

        # 4) Render subtitles in OpenCV window
        frame = create_subtitle_frame(subtitle_lines)
        cv2.imshow("Subtitles", frame)
        if cv2.waitKey(10) == 27:  # ESC to exit
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping transcription...")

finally:
    stream.stop()
    stream.close()
    cv2.destroyAllWindows()