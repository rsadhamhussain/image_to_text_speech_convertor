# image_to_text_speech_convertor
An AI-powered Streamlit app that converts images into short stories and reads them aloud. It uses Hugging Face for image captioning and text-to-speech, and OpenAI (via LangChain) for creative story generation.

# 🧠 Image to Caption to Story (with Text-to-Speech)

This is a simple AI-powered Streamlit app that:

📸 → Converts an **image** into a **caption**  
📝 → Turns the **caption** into a **short story** (via OpenAI LLM)  
🔊 → Converts the **story** into **speech** (via HuggingFace TTS)

---

## 💡 Features

- **Image Captioning**: Uses BLIP (`Salesforce/blip-image-captioning-base`) to describe uploaded images.
- **Story Generation**: Uses OpenAI's `gpt-3.5-turbo` to write a creative story (limited to 20 words).
- **Text-to-Speech**: Uses HuggingFace’s ESPnet TTS model to convert story to audio.

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

