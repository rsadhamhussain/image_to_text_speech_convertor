from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import os
import streamlit as st

# Load environment variables from .env file
load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def image2text(url):
    """
    Converts an image into a descriptive caption using a pre-trained Hugging Face image captioning model.

    Args:
        url (str): Path or URL to the image.

    Returns:
        str: Generated image caption text.
    """
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text


def generate_story(scenario):
    """
    Generates a short story (max 20 words) based on the provided image caption.

    Args:
        scenario (str): Description of the image (caption).

    Returns:
        str: Short generated story.
    """
    template = """
    You are a skilled storyteller. You can generate a story based on the given scenario.
    Story should not be more than 20 words.

    CONTEXT:
    {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_LLM = LLMChain(
        llm=OpenAI(temperature=0.7, model="gpt-3.5-turbo"),
        prompt=prompt,
        verbose=True
    )
    story = story_LLM.predict(scenario=scenario)
    print(story)
    return story


def text_to_speech(message):
    """
    Converts the given text message to speech using Hugging Face's ESPnet model.

    Args:
        message (str): Text to convert into speech.

    Returns:
        None: Saves the audio as a local 'audio.flac' file.
    """
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    """
    Streamlit app for uploading an image, generating a caption, creating a story, and converting the story to audio.
    """
    st.set_page_config(page_title="Image to Caption to Summary", page_icon="😊")
    st.header("Image → Caption → Summary")

    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        st.text('Processing image to caption...')
        caption = image2text(uploaded_file.name)
        with st.expander("Caption"):
            st.write(caption)

        st.text('Generating story...')
        story = generate_story(caption)
        with st.expander("Story"):
            st.write(story)

        st.text('Converting story to speech...')
        text_to_speech(story)
        st.audio("audio.flac")


if __name__ == '__main__':
    main()
