import streamlit as st
import requests
import os
import tempfile
import json
import anthropic
from pytube import YouTube
import whisper
from langdetect import detect
import time
import re
import yt_dlp

# Page configuration
st.set_page_config(
    page_title="Elkheta Video Summarizer",
    page_icon="🎬",
    layout="wide"
)

# App title and description
st.title("Elkheta Video Summarizer")
st.markdown("Generate structured explanatory summaries from educational videos")

# Initialize Anthropic client
@st.cache_resource
def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Anthropic API key not found. Please set it in the app secrets or .env file.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

# Initialize Whisper model for transcription
@st.cache_resource
def load_whisper_model():
    with st.spinner("Loading Whisper model (first run only)..."):
        return whisper.load_model("base")

# Function to download video using yt-dlp (more robust method)
def download_video_ytdlp(url):
    st.info("Downloading video...")
    progress_bar = st.progress(0)
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_filename = temp_file.name
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_filename,
            'progress_hooks': [lambda d: progress_bar.progress(d['downloaded_bytes'] / d['total_bytes'] if 'total_bytes' in d and d['total_bytes'] > 0 else 0)],
            'quiet': True,
            'no_warnings': True
        }
        
        # Try to download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        progress_bar.progress(1.0)
        return temp_filename
    except Exception as e:
        st.error(f"Error downloading video with yt-dlp: {str(e)}")
        return None

# Fallback function to download video using direct requests
def download_video_direct(url):
    st.info("Attempting direct download...")
    progress_bar = st.progress(0)
    
    try:
        # Add a user agent to help with some sites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 200:
            # Create a temporary file to store the video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))
                
                progress_bar.progress(1.0)
                return temp_file.name
        else:
            st.error(f"Failed to download: HTTP status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error with direct download: {str(e)}")
        return None
# Add this function before transcribe_video
def verify_audio_exists(video_path):
    try:
        import subprocess
        # Check if the file contains an audio stream
        result = subprocess.run(
            ['ffmpeg', '-i', video_path, '-af', 'volumedetect', '-f', 'null', '-'],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        # Check if audio stream is found in the output
        return "Stream #0:1" in result.stderr or "Audio" in result.stderr
    except Exception as e:
        st.error(f"Error verifying audio: {str(e)}")
        return False

# Then modify your transcribe_video function:
def transcribe_video(video_path):
    st.info("Transcribing video (this may take a few minutes)...")
    progress_bar = st.progress(0)
    
    try:
        # First verify the file has audio
        if not verify_audio_exists(video_path):
            st.error("No audio stream found in the video file. Cannot transcribe.")
            return None
            
        # Get the Whisper model
        model = load_whisper_model()
        
        # Extract audio to a temporary WAV file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_path = temp_audio.name
        
        # Use ffmpeg to extract audio
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-q:a', '0', '-map', 'a', 
            '-f', 'wav', audio_path
        ], check=True)
        
        # Verify the audio file has data
        if os.path.getsize(audio_path) < 1000:  # If file is too small
            st.error("Extracted audio file is too small or empty.")
            return None
        
        # Start transcription on the extracted audio
        progress_bar.progress(0.5)
        result = model.transcribe(audio_path)
        progress_bar.progress(1.0)
        
        # Clean up temp file
        os.unlink(audio_path)
        
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing video: {str(e)}")
        return None
        
# Function to transcribe video
def transcribe_video(video_path):
    st.info("Transcribing video (this may take a few minutes)...")
    progress_bar = st.progress(0)
    
    try:
        # Get the Whisper model
        model = load_whisper_model()
        
        # Start transcription
        result = model.transcribe(video_path)
        progress_bar.progress(1.0)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing video: {str(e)}")
        return None

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails

# Function to generate structured summary
def generate_structured_summary(transcript, lang):
    st.info("Generating structured summary using Claude...")
    
    client = get_anthropic_client()
    
    prompt = f"""
    Please analyze this educational video transcript and create a well-structured 
    explanatory summary that could be used for an animated educational video.
    
    The summary should include:
    1. Introduction of the main topic
    2. Key concepts organized in logical sequence
    3. Important definitions and formulas (if applicable)
    4. Examples and applications
    5. Summary of critical points
    
    Format the output as a JSON structure with these sections clearly defined.
    The transcript language is {lang}.
    
    Transcript:
    {transcript}
    """
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",  # Using a model that should be available
            max_tokens=4000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the structured content from Claude's response
        return response.content[0].text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Function to format for animation
def format_for_animation(structured_content):
    try:
        # Try to parse as JSON if Claude returned JSON
        content = json.loads(structured_content)
        return content
    except json.JSONDecodeError:
        # If not JSON, return as is
        return structured_content

# Main app UI
with st.form("video_form"):
    video_url = st.text_input("Video URL (YouTube, Bunny.net, or direct link):", 
                             placeholder="https://www.youtube.com/watch?v=example")
    
    # Optional file upload for local videos
    uploaded_file = st.file_uploader("Or upload a video file:", type=["mp4", "mov", "avi", "mkv"])
    
    submit_button = st.form_submit_button("Process Video")

# Process the video when the form is submitted
if submit_button and (video_url or uploaded_file):
    # Create a container for results
    result_container = st.container()
    video_path = None
    
    # Handle file upload if provided
    if uploaded_file is not None:
        with st.spinner("Saving uploaded video..."):
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                video_path = temp_file.name
    # Handle URL if provided
    elif video_url:
        # First try with yt-dlp (most robust method)
        video_path = download_video_ytdlp(video_url)
        
        # If that fails, try direct download as fallback
        if not video_path:
            st.warning("Primary download method failed. Trying alternative method...")
            video_path = download_video_direct(video_url)
    
    if video_path:
        # Step 2: Transcribe the video
        transcript = transcribe_video(video_path)
        
        if transcript:
            # Show the transcript in an expandable section
            with st.expander("Video Transcript", expanded=False):
                st.write(transcript)
            
            # Step 3: Detect language
            language = detect_language(transcript)
            language_name = {
                'en': 'English',
                'ar': 'Arabic',
                'fr': 'French',
                'es': 'Spanish',
                'de': 'German'
            }.get(language, language)
            
            # Step 4: Generate structured summary
            structured_content = generate_structured_summary(transcript, language)
            
            if structured_content:
                # Step 5: Format for animation
                animation_script = format_for_animation(structured_content)
                
                # Display results
                with result_container:
                    st.success("✅ Processing complete!")
                    
                    st.subheader("Generated Summary")
                    st.markdown(f"**Detected Language:** {language_name}")
                    
                    # Display the formatted content
                    st.subheader("Animation-Ready Script")
                    
                    if isinstance(animation_script, dict):
                        # If it's a dictionary, format it nicely
                        formatted_json = json.dumps(animation_script, indent=2)
                        st.json(animation_script)
                    else:
                        # If it's a string, display as text
                        st.text_area("Script Content", animation_script, height=400)
                    
                    # Download button
                    script_str = json.dumps(animation_script, indent=2) if isinstance(animation_script, dict) else animation_script
                    st.download_button(
                        label="Download Script",
                        data=script_str,
                        file_name="animation-script.json",
                        mime="application/json"
                    )
                
                # Clean up the temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
    else:
        st.error("Unable to process the video. Please check the URL or try uploading the file directly.")

# Add instructions at the bottom
with st.expander("Need help?"):
    st.markdown("""
    ### Troubleshooting
    
    If you're experiencing download issues:
    
    1. **Try YouTube links** - These typically work more reliably
    2. **Upload the video directly** - If downloading fails, you can upload the file
    3. **Check URL format** - Make sure the URL is complete and accessible
    4. **Special Characters** - Some URLs with special characters may cause issues
    
    For Bunny.net URLs, make sure they're publicly accessible.
    """)
