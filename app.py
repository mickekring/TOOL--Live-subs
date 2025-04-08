import queue
import time
import os
import argparse
import threading
import torch
import numpy as np
import sounddevice as sd
from pydub import AudioSegment, silence
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import logging as hf_logging

import cv2
import textwrap
from PIL import Image, ImageDraw, ImageFont
import platform
import gc
import logging
from colorama import init, Fore, Style
init() # Initialize colorama

app_version = "0.1.1"

# Setup logging
def setup_colored_logger():
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.WARNING:
                record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
            elif record.levelno == logging.ERROR:
                record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
            elif record.levelno == logging.INFO:
                record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
            return super().format(record)
    
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('subtitle_generator')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = setup_colored_logger()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Real-time subtitle generator')
parser.add_argument('--model', type=str, default="KBLab/kb-whisper-tiny",
                    help='Whisper model to use (tiny/base/small/medium/large)')
parser.add_argument('--language', type=str, default="sv", 
                    help='Language code for transcription (e.g., sv, en, etc.)')
parser.add_argument('--width', type=int, default=1920, help='Width of output window')
parser.add_argument('--height', type=int, default=1080, help='Height of output window')
parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
parser.add_argument('--buffer_size', type=int, default=200, 
                    help='Character buffer size for continuous text')
parser.add_argument('--max_lines', type=int, default=2, help='Maximum lines to display')
parser.add_argument('--chars_per_line', type=int, default=52, 
                    help='Maximum characters per line')
parser.add_argument('--silence_threshold', type=float, default=-40, 
                    help='Silence threshold in dB')
parser.add_argument('--min_silence', type=int, default=400, 
                    help='Minimum silence duration in ms')
parser.add_argument('--save_transcript', action='store_true', 
                    help='Save transcript to a file')
parser.add_argument('--output', type=str, default="transcript.txt", 
                    help='Output file for transcript')
args = parser.parse_args()

#####################################################
# Subtitle Display Configuration
#####################################################

SCREEN_WIDTH = args.width
SCREEN_HEIGHT = args.height
CHROMA_KEY_GREEN = (0, 255, 0)  # BGR for OpenCV
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RECT_HEIGHT = 220

SUBTITLE_MAX_LINES = args.max_lines
CHARS_PER_LINE = args.chars_per_line
LINE_SPACING = 52
BOTTOM_MARGIN = 130
SENTENCE_BUFFER_SIZE = args.buffer_size

# Font selection based on platform
if platform.system() == "Windows":
    FONT_PATH = os.path.join(os.environ["WINDIR"], "Fonts", "Arial.ttf")
elif platform.system() == "Darwin":  # macOS
    FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
else:  # Linux and others
    # Common locations for fonts on Linux
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    FONT_PATH = next((path for path in font_paths if os.path.exists(path)), None)
    if not FONT_PATH:
        logger.warning("No suitable font found. Using default.")
        FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

FONT_SIZE = 52

# State variables
subtitle_lines = []
recent_transcriptions = []
sentence_buffer = ""
full_transcript = []
is_paused = False
is_running = True
show_controls = False
mic_level = 0
processing_status = "Ready"

# Create a lock for thread safety when updating shared state
state_lock = threading.Lock()

#####################################################
# Helper Functions
#####################################################

def get_display_text():
    """Get text for display from the current buffer."""
    with state_lock:
        # Wrap the current buffer into lines of appropriate width
        lines = textwrap.wrap(sentence_buffer, width=CHARS_PER_LINE)
        
        # Keep only the last SUBTITLE_MAX_LINES for display
        return lines[-SUBTITLE_MAX_LINES:] if lines else []

def add_subtitle_text(new_text):
    """
    Improved function that maintains a continuous buffer of recent speech
    and intelligently breaks it into visible lines.
    """
    global sentence_buffer, full_transcript
    
    with state_lock:
        # Add new text to the full transcript
        full_transcript.append(new_text.strip())
        
        # Add new text to our sentence buffer
        sentence_buffer += " " + new_text.strip()
        
        # Keep only the most recent portion of speech (last N characters)
        if len(sentence_buffer) > SENTENCE_BUFFER_SIZE:
            # Find a good break point (period, question mark, etc.) if possible
            good_break_points = ['.', '!', '?', ';']
            break_points = []
            
            cutoff_start = len(sentence_buffer) - SENTENCE_BUFFER_SIZE
            for char in good_break_points:
                positions = [pos for pos in range(cutoff_start, len(sentence_buffer)) 
                             if sentence_buffer[pos] == char]
                break_points.extend(positions)
                
            if break_points:
                # Use the latest good break point that's in the early part of the buffer
                cutoff = max(min(break_points) + 1, cutoff_start)
                sentence_buffer = sentence_buffer[cutoff:].strip()
            else:
                # If no good break point, try to break at a word boundary
                text_to_keep = sentence_buffer[cutoff_start:]
                first_space = text_to_keep.find(" ")
                if first_space > 0:
                    sentence_buffer = text_to_keep[first_space:].strip()
                else:
                    # If no word boundary, just keep the last N characters
                    sentence_buffer = sentence_buffer[-SENTENCE_BUFFER_SIZE:].strip()


def create_subtitle_frame():
    global mic_level, processing_status, show_controls
    
    # 1) Create the image in green
    pil_image = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 255, 0))
    draw = ImageDraw.Draw(pil_image)

    # 2) Draw black rectangle at bottom
    black_rect_top = SCREEN_HEIGHT - RECT_HEIGHT
    draw.rectangle([(0, black_rect_top), (SCREEN_WIDTH, SCREEN_HEIGHT)], fill=(0, 0, 0))

    # 3) Set up font & margins
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        logger.error(f"Font error: {e}")
        # Fallback to default font
        font = ImageFont.load_default()
    
    y_start = SCREEN_HEIGHT - BOTTOM_MARGIN

    # 4) Get current text to display
    lines = get_display_text()

    # 5) Draw each line from bottom to top
    for i in range(len(lines)):
        text = lines[len(lines) - 1 - i]
        y = y_start - i * LINE_SPACING

        # Use textbbox to measure text width/height
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left

        # Center horizontally
        x = (SCREEN_WIDTH - text_width) // 2

        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # 6) Add status overlay if controls are showing
    if show_controls:
        # Draw semi-transparent overlay at top
        overlay_height = 80
        draw.rectangle([(0, 0), (SCREEN_WIDTH, overlay_height)], 
                      fill=(0, 0, 0, 180))
        
        # Show mic level
        level_width = int(mic_level * 200)  # Scale to 200px max
        draw.rectangle([(20, 20), (20 + level_width, 40)], fill=(0, 255, 0))
        draw.rectangle([(20, 20), (220, 40)], outline=(255, 255, 255))
        
        # Show status
        status_text = f"Status: {processing_status}"
        if is_paused:
            status_text += " (PAUSED)"
        draw.text((250, 20), status_text, font=font, fill=(255, 255, 255))
        
        # Show help text
        help_text = "P: Pause | S: Save | ESC: Exit | H: Hide Controls"
        draw.text((20, 50), help_text, font=font, fill=(255, 255, 255))

    # 7) Convert Pillow (RGB) -> NumPy (BGR) for OpenCV
    open_cv_image = np.array(pil_image)[:, :, ::-1].copy()
    return open_cv_image


def save_transcript():
    """Save the full transcript to a file."""
    with state_lock:
        if not full_transcript:
            return False
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_transcript))
        return True


def is_chunk_silent(chunk, threshold_db=-35):
    """Check if an audio chunk is silent based on dB threshold."""
    try:
        return chunk.dBFS < threshold_db
    except AttributeError:
        # Handle case where chunk doesn't have dBFS attribute
        return chunk.max_dBFS < threshold_db


def update_mic_level(audio_chunk):
    """Update the microphone level indicator from audio chunk."""
    global mic_level
    try:
        # Calculate normalized mic level (0-1)
        db_level = audio_chunk.dBFS
        # Map from typical dB range (-60 to 0) to 0-1
        norm_level = max(0, min(1, (db_level + 60) / 60))
        with state_lock:
            mic_level = norm_level
    except Exception:
        pass  # Ignore errors in level calculation


def log_audio_devices():
    """Log information about available audio devices."""
    devices = sd.query_devices()
    #logger.info("Available audio devices:")
    #for i, device in enumerate(devices):
    #    logger.info(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
    
    default_input = sd.query_devices(kind='input')
    logger.info(f"Using input device: {default_input['name']}")

def log_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


#####################################################
# Setup Device and Load Model
#####################################################

def setup_model():
    """Set up and load the transcription model."""
    global processing_status
    
    with state_lock:
        processing_status = "Loading model..."
    
    hf_logging.set_verbosity_error()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning("Using CPU for processing, this may be slow.")
    if device == "cuda:0":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = args.model

    logger.info(f"Loading model {model_id} on {device}...")
    
    try:
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
        
        with state_lock:
            processing_status = "Ready"
        
        return asr_pipe
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        with state_lock:
            processing_status = f"Error: {str(e)[:30]}..."
        raise


def transcribe_chunk(chunk, asr_pipe):
    """Transcribe an audio chunk using the provided ASR pipeline."""
    with state_lock:
        processing_status = "Transcribing..."
    
    try:
        raw_samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
        raw_samples /= 32767.0  # Normalize to [-1.0, 1.0]

        audio_input = {"array": raw_samples, "sampling_rate": 16000}
        result = asr_pipe(
            audio_input,
            chunk_length_s=30,
            generate_kwargs={"task": "transcribe", "language": args.language}
        )
        
        with state_lock:
            processing_status = "Ready"
        
        return result["text"]
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        with state_lock:
            processing_status = "Error in transcription"
        return ""


def transcribe_if_not_silent(chunk, asr_pipe, threshold_db=-35):
    """Transcribe a chunk only if it's not silent."""
    if is_chunk_silent(chunk, threshold_db=threshold_db):
        return ""
    else:
        return transcribe_chunk(chunk, asr_pipe)


#####################################################
# Audio Processing Thread
#####################################################

def audio_processing_thread(audio_queue, asr_pipe):
    global is_running, is_paused, processing_status
    
    buffered_segment = AudioSegment.empty()
    silence_start_time = None
    
    # Words that may be falsely detected during silence
    excluded_texts = set(["Tack.", "Tack!", "Ja.", "Musik"])
    
    # For different languages, you might need different exclusions
    if args.language != "sv":
        excluded_texts = set(["Tack.", "Tack!", "Ja.", "Musik", "Um.", "Uh."])
    
    logger.info("Audio processing thread started")
    
    try:
        while is_running:
            if is_paused:
                time.sleep(0.1)
                continue
                
            # 1) If we have data
            if not audio_queue.empty():
                try:
                    data_block = audio_queue.get(timeout=0.1)
                    block_int16 = (data_block * 32767).astype(np.int16).tobytes()

                    segment = AudioSegment(
                        data=block_int16,
                        sample_width=2,
                        frame_rate=16000,
                        channels=1
                    )
                    buffered_segment += segment
                    
                    # Update mic level indicator
                    update_mic_level(segment)

                    # 2) Silence splitting with adaptive parameters
                    chunks = silence.split_on_silence(
                        buffered_segment,
                        min_silence_len=args.min_silence,
                        silence_thresh=args.silence_threshold,
                        keep_silence=100
                    )

                    if len(chunks) > 1:
                        finished_chunks = chunks[:-1]
                        for c in finished_chunks:
                            if not is_running:
                                break
                                
                            text = transcribe_chunk(c, asr_pipe)
                            if text.strip() and text.strip() not in excluded_texts:
                                logger.info(f"Transcribed: {text}")
                                
                                # Add to recent transcriptions and subtitle
                                with state_lock:
                                    recent_transcriptions.append(text)
                                    if len(recent_transcriptions) > 5:
                                        recent_transcriptions.pop(0)
                                        
                                add_subtitle_text(text)
                        buffered_segment = chunks[-1]
                        silence_start_time = None  # Reset silence timer

                    # 3) Forced chunk if buffer is too long
                    if len(buffered_segment) > 8000:
                        text = transcribe_if_not_silent(buffered_segment, asr_pipe, 
                                                       args.silence_threshold)
                        if text.strip() and text.strip() not in excluded_texts:
                            logger.info(f"Transcribed (forced): {text}")
                            add_subtitle_text(text)
                        buffered_segment = AudioSegment.empty()
                        silence_start_time = None  # Reset silence timer
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    with state_lock:
                        processing_status = "Error processing audio"
            else:
                # Check for silence duration
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > args.min_silence / 1000.0:
                    # Force transcription if silence duration exceeds threshold
                    text = transcribe_if_not_silent(buffered_segment, asr_pipe, 
                                                    args.silence_threshold)
                    if text.strip() and text.strip() not in excluded_texts:
                        logger.info(f"Transcribed (silence): {text}")
                        add_subtitle_text(text)
                    buffered_segment = AudioSegment.empty()
                    silence_start_time = None  # Reset silence timer
                    if buffered_segment.empty():
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                time.sleep(0.01)
                
    except Exception as e:
        logger.error(f"Audio processing thread error: {e}")
    finally:
        logger.info("Audio processing thread ending")


#####################################################
# Main Function
#####################################################

def main():
    global is_running, is_paused, show_controls
    
    # Load the ASR model
    try:
        asr_pipe = setup_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Setup audio queue and callback
    sample_rate = 16000
    block_size_ms = 200
    block_size = int(sample_rate * (block_size_ms / 1000))
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        audio_queue.put(indata.copy())

    # Start audio stream
    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=block_size,
            callback=audio_callback
        )
        stream.start()
    except Exception as e:
        logger.error(f"Audio stream error: {e}")
        return
    
    log_audio_devices()
    log_memory_usage()
    
    # Start processing thread
    processing_thread = threading.Thread(
        target=audio_processing_thread,
        args=(audio_queue, asr_pipe),
        daemon=True
    )
    processing_thread.start()
    
    # Setup display window
    cv2.namedWindow("Subtitles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Subtitles", SCREEN_WIDTH, SCREEN_HEIGHT)
    
    if args.fullscreen:
        cv2.setWindowProperty("Subtitles", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    logger.info("Starting subtitle display. Press ESC to exit, P to pause/resume, H to show/hide controls.")
    
    try:
        while is_running:
            # Create and display the subtitle frame
            frame = create_subtitle_frame()
            cv2.imshow("Subtitles", frame)
            
            # Process key presses
            key = cv2.waitKey(10)
            if key == 27:  # ESC to exit
                is_running = False
            elif key == ord('p') or key == ord('P'):  # P to pause/resume
                is_paused = not is_paused
                logger.info(f"{'Paused' if is_paused else 'Resumed'} transcription")
            elif key == ord('h') or key == ord('H'):  # H to show/hide controls
                show_controls = not show_controls
            elif key == ord('s') or key == ord('S'):  # S to save transcript
                if args.save_transcript or save_transcript():
                    logger.info(f"Saved transcript to {args.output}")
                    with state_lock:
                        processing_status = f"Saved to {args.output}"
                else:
                    logger.warning("No transcript to save")
            
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Cleanup resources
        is_running = False
        stream.stop()
        stream.close()
        cv2.destroyAllWindows()
        
        # Wait for processing thread to end
        processing_thread.join(timeout=1.0)
        
        # Save transcript if enabled
        if args.save_transcript:
            save_transcript()
            logger.info(f"Saved transcript to {args.output}")
        
        logger.info("Subtitle generator shutdown complete")


if __name__ == "__main__":
    main()