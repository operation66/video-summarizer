import streamlit as st
import requests
import os
import tempfile
import json
import anthropic
from pytube import YouTube
import whisper
from langdetect import detect
from dotenv import load_dotenv
import time

# Load environment variables from .env file if present
load_dotenv()

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

# Function to download video (now supports YouTube)
def download_video(url):
    st.info("Downloading video...")
    progress_bar = st.progress(0)
    
    try:
        if "youtube.com" in url or "youtu.be" in url:
            # Download from YouTube
            yt = YouTube(url)
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            
            # Create a temporary file to store the video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                video_stream.download(output_path=os.path.dirname(temp_file.name), 
                                     filename=os.path.basename(temp_file.name))
                progress_bar.progress(100)
                return temp_file.name
        else:
            # Download from direct URL (Bunny.net or other)
            response = requests.get(url, stream=True)
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
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

# Function to transcribe video
def transcribe_video(video_path):
    st.info("Transcribing video (this may take a few minutes)...")
    progress_bar = st.progress(0)
    
    try:
        # Get the Whisper model
        model = load_whisper_model()
        
        # Start transcription
        result = model.transcribe(video_path, progress_callback=lambda progress: progress_bar.progress(progress))
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
            model="claude-3-sonnet-20240229",  # You can change to claude-3-7-sonnet-20250219 if available
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
    video_url = st.text_input("Video URL (YouTube or direct link):", 
                             placeholder="https://www.youtube.com/watch?v=example")
    
    submit_button = st.form_submit_button("Process Video")

# Process the video when the form is submitted
if submit_button and video_url:
    # Create a container for results
    result_container = st.container()
    
    # Step 1: Download the video
    video_path = download_video(video_url)
    
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
