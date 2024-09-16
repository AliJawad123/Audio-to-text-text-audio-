import streamlit as st
import whisper
import nltk
import numpy as np
from io import BytesIO

# Make sure to download the punkt tokenizer for sentence splitting
nltk.download('punkt')

# Transcription function
def transcribe_audio(audio_file, model_type="base"):
    # Load the Whisper model
    model = whisper.load_model(model_type)
    result = model.transcribe(audio_file)

    # Save transcription
    output_file = audio_file.split('.')[0] + '_transcription.txt'
    with open(output_file, 'w') as f:
        f.write(result["text"])

    return result["text"], output_file

# Text-to-Speech function
def text_to_speech(text, sample_rate=22050, speaker="v2/en_speaker_6"):
    from bark import generate_audio  # Assuming you are using Bark library
    
    # Prepare text for sentence tokenization
    story_1 = text.replace("\n", " ")
    sentences = nltk.sent_tokenize(story_1)
    
    # Silence between sentences
    silence = np.zeros(int(0.25 * sample_rate))
    pieces = []
    
    # Generate audio for each sentence
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=speaker)
        pieces += [audio_array, silence.copy()]

    # Concatenate all audio pieces
    final_audio = np.concatenate(pieces)
    return final_audio

# Streamlit app
def main():
    st.set_page_config(page_title="Audio Transcription and TTS App", page_icon=":microphone:", layout="wide")
    
    st.title("Audio Transcription and Text-to-Speech with Whisper and Bark")
    st.markdown("Upload an MP3 audio file to transcribe its content to text and convert the text back into speech.")

    # File uploader
    audio_file = st.file_uploader("Upload Audio File", type=["mp3"])

    # Transcription model type
    model_type = st.selectbox("Select Model Type", ["base", "small", "medium", "large"])

    # TTS settings
    speaker = st.selectbox("Select Speaker", ["v2/en_speaker_6"])  # Add more speakers if available

    if st.button("Transcribe and Convert to Speech"):
        if audio_file is not None:
            with st.spinner("Processing..."):
                # Transcribe the audio file
                transcription, output_file = transcribe_audio(audio_file.name, model_type=model_type)

                # Convert text to speech
                sample_rate = 22050  # Set sample rate for generated audio
                tts_audio = text_to_speech(transcription, sample_rate=sample_rate, speaker=speaker)

                # Convert numpy array to bytes
                audio_buffer = BytesIO()
                audio_buffer.write(tts_audio.tobytes())
                audio_buffer.seek(0)
                
            st.success("Processing Completed!")
            
            # Display the transcription
            st.write("Transcription:")
            st.write(transcription)

            # Option to download the transcription
            with open(output_file, 'r') as file:
                transcription_text = file.read()
            st.download_button(label="Download Transcription", data=transcription_text, file_name=output_file)

            # Play the generated audio
            st.audio(tts_audio, format='audio/wav', sample_rate=sample_rate)

            # Option to download the generated audio
            st.download_button(
                label="Download Generated Audio",
                data=audio_buffer,
                file_name='generated_audio.wav',
                mime='audio/wav'
            )
        else:
            st.warning("Please upload an audio file first.")

if __name__ == "__main__":
    main()
