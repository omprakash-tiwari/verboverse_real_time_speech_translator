import os
import time
import whisper
import pygame
import tempfile
import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from googletrans import LANGUAGES, Translator
import logging
from datetime import datetime
import langdetect

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="VerboVerse",
    page_icon="🗣️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.output-text {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.big-font {
    font-size: 24px;
}
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🎤 VerboVerse")
st.markdown("Speak in any language and get real-time translation")

# Constants
SAMPLE_RATE = 16000
RECORD_SECONDS = 5  # Fixed recording duration

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.translations = []
    st.session_state.current_translation = None
    st.session_state.debug_info = []
    st.session_state.initialized = True

# Debug container
debug_container = st.empty()

# Hidden debug mode toggle
debug_mode = False

# Whisper model selection
WHISPER_MODELS = {
    "tiny": "Fastest, least accurate",
    "base": "Fast, more accurate",
    "small": "Balanced speed/accuracy",
    "medium": "Slow, high accuracy",
    "large": "Very slow, highest accuracy"
}

# Load Whisper model
@st.cache_resource
def load_whisper_model(model_name="base"):
    try:
        model = whisper.load_model(model_name)
        logger.info(f"Whisper {model_name} model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

# Initialize translator with retry mechanism
def get_translator():
    max_retries = 3
    for i in range(max_retries):
        try:
            return Translator()
        except Exception as e:
            logger.error(f"Error initializing translator (attempt {i+1}/{max_retries}): {str(e)}")
            if i == max_retries - 1:
                st.error("Failed to initialize translator. Please restart the app.")
                return None
            time.sleep(1)  # Wait before retrying

translator = get_translator()

# Initialize pygame for audio playback
if not pygame.mixer.get_init():
    pygame.mixer.init()

# Find all available input devices
def get_input_devices():
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if isinstance(device, dict) and device.get('max_input_channels', 0) > 0:
            input_devices.append({
                'index': i,
                'name': device['name'],
                'channels': device['max_input_channels']
            })
    return input_devices

# Find Intel microphone
def find_intel_mic():
    devices = get_input_devices()
    for device in devices:
        if "intel" in device['name'].lower() or "smart sound" in device['name'].lower():
            logger.info(f"Found Intel microphone: {device['name']}")
            return device
    return None

# Simple recording function
def record_audio(device_index, duration=5):
    """Record audio for a fixed duration"""
    try:
        # Show recording progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Record audio
        status_text.text("🎙️ Recording...")
        
        # Direct recording without callback to avoid threading issues
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_index,
            dtype='float32'
        )
        
        # Show progress while recording
        for i in range(10):
            progress_bar.progress((i+1)/10)
            time.sleep(duration/10)
        
        # Wait for recording to complete
        sd.wait()
        
        # Check if we got audio
        if recording is None or len(recording) == 0:
            status_text.error("No audio recorded!")
            return None
        
        # Check if audio contains actual speech (not just silence)
        audio_level = np.max(np.abs(recording))
        if audio_level < 0.01:  # Threshold for detecting actual speech
            status_text.warning("Audio level too low. Please speak louder.")
            if debug_mode:
                logger.info(f"Audio level: {audio_level:.6f} - too low")
            return None
            
        if debug_mode:
            logger.info(f"Audio recorded successfully. Max level: {audio_level:.6f}")
            
        status_text.text("✅ Recording complete!")
        return recording
        
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        logger.error(f"Error recording audio: {str(e)}")
        return None

# Verify language detection
def verify_language(text, expected_lang=None):
    """Verify detected language with langdetect as a backup"""
    try:
        detected = langdetect.detect(text)
        confidence = langdetect.detect_langs(text)[0].prob
        
        if debug_mode:
            logger.info(f"Language detection: {detected} with confidence {confidence:.4f}")
            
        if expected_lang and detected != expected_lang and confidence > 0.8:
            logger.info(f"Language mismatch: expected {expected_lang}, got {detected}")
            
        return detected, confidence
    except:
        return None, 0.0

# Process audio
def process_audio(audio_data, model_name="base"):
    """Process recorded audio"""
    temp_file = None
    try:
        # Create a unique temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        temp_file.close()  # Close immediately to avoid file access issues
        
        # Save audio to the temporary file
        sf.write(temp_filename, audio_data, SAMPLE_RATE)
        
        # Load model
        model = load_whisper_model(model_name)
        if not model:
            return None
        
        # Transcribe with more options for accuracy
        st.text("🧠 Transcribing...")
        result = model.transcribe(
            temp_filename, 
            fp16=False,
            language=None,  # Auto-detect language
            task="transcribe",
            temperature=0.0,  # Use greedy decoding for more consistent results
            best_of=1
        )
        
        # Get detected language and transcript
        detected_lang = result.get("language", "")
        transcript = result.get("text", "").strip()
        
        if not transcript:
            st.warning("No speech detected. Please try again.")
            return None
            
        # Double-check language detection
        backup_lang, confidence = verify_language(transcript)
        
        # Translate
        st.text("🌐 Translating...")
        try:
            translation = translator.translate(
                transcript, 
                src=detected_lang if detected_lang else backup_lang, 
                dest=st.session_state.target_language
            )
            
            # Fallback if translation fails
            if not translation or not translation.text:
                st.warning("Translation failed. Trying again with auto-detected language...")
                translation = translator.translate(
                    transcript,
                    dest=st.session_state.target_language
                )
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            st.error("Translation failed. Please try again.")
            return None
        
        # Create translation entry
        translation_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'detected_language': LANGUAGES.get(detected_lang, detected_lang).title(),
            'original_text': transcript,
            'translation': translation.text,
            'pronunciation': getattr(translation, 'pronunciation', None)
        }
        
        # Update state
        st.session_state.translations.append(translation_entry)
        st.session_state.current_translation = translation_entry
        
        # Text to speech
        st.text("🔊 Playing translation...")
        speech_temp = None
        try:
            speech_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            speech_temp_name = speech_temp.name
            speech_temp.close()  # Close immediately
            
            tts = gTTS(text=translation.text, lang=st.session_state.target_language, slow=False)
            tts.save(speech_temp_name)
            
            # Play audio
            pygame.mixer.music.load(speech_temp_name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")
            logger.error(f"Error playing audio: {str(e)}")
        finally:
            # Clean up speech file
            if speech_temp and os.path.exists(speech_temp_name):
                try:
                    os.unlink(speech_temp_name)
                except:
                    pass
                
        return translation_entry
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        logger.error(f"Error processing audio: {str(e)}")
        return None
    finally:
        # Always clean up the temp file
        if temp_file and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except Exception as e:
                logger.error(f"Error deleting temp file: {str(e)}")
                pass

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Whisper model selection
    model_name = st.selectbox(
        "Whisper Model",
        list(WHISPER_MODELS.keys()),
        index=1,  # Default to "base"
        format_func=lambda x: f"{x} - {WHISPER_MODELS[x]}"
    )
    
    # Target language selection
    available_languages = sorted(list(LANGUAGES.values()))
    target_lang_name = st.selectbox(
        "Target Language",
        available_languages,
        index=available_languages.index('hindi') if 'hindi' in available_languages else available_languages.index('english'),
        help="Choose the language you want to translate to"
    )
    language_mapping = {name.lower(): code for code, name in LANGUAGES.items()}
    st.session_state.target_language = language_mapping.get(target_lang_name.lower(), "hi")
    
    # Recording duration
    record_duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
    
    # Silently handle microphone selection
    intel_mic = find_intel_mic()
    if intel_mic:
        selected_device = intel_mic
    else:
        input_devices = get_input_devices()
        if not input_devices:
            st.error("No input devices found!")
            st.stop()
        selected_device = input_devices[0]  # Select first available device

# Main content area
col1, col2 = st.columns(2)

with col1:
    if st.button("🎤 Record and Translate", type="primary"):
        # Record audio
        audio_data = record_audio(selected_device['index'], record_duration)
        
        # Process if we got audio
        if audio_data is not None and len(audio_data) > 0:
            process_audio(audio_data, model_name)

with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.translations = []
        st.session_state.current_translation = None
        st.rerun()

# Display current translation
if st.session_state.current_translation:
    st.markdown("### Current Translation")
    st.markdown(f"""
    <div class="output-text">
        <span class="big-font">🔊 {st.session_state.current_translation['translation']}</span><br><br>
        🕒 Time: {st.session_state.current_translation['timestamp']}<br>
        🗣️ Detected: {st.session_state.current_translation['detected_language']}<br>
        📝 Original: "{st.session_state.current_translation['original_text']}"
    </div>
    """, unsafe_allow_html=True)

# Display translation history
if st.session_state.translations:
    with st.expander("📚 Translation History", expanded=False):
        for entry in reversed(st.session_state.translations[-10:]):
            st.markdown(f"""
            <div class="output-text">
                🕒 {entry['timestamp']}<br>
                🗣️ {entry['detected_language']}: "{entry['original_text']}"<br>
                🔄 Translation: "{entry['translation']}"
            </div>
            """, unsafe_allow_html=True)

# Display debug info if enabled
if debug_mode and st.session_state.debug_info:
    with st.expander("🐞 Debug Info", expanded=True):
        for info in st.session_state.debug_info[-20:]:
            st.text(info)