from mlx_lm import load, generate
import mlx_whisper
from melo.api import TTS
import numpy as np
import librosa
import gradio as gr
from gradio_webrtc import WebRTC, StreamHandler, AdditionalOutputs

llm, tokenizer = load("mlx-community/SmolLM-360M-Instruct")

tts = TTS(language="EN", device="cpu")


def speech_to_text(audio: tuple[int, np.ndarray]):
    sampling_rate, audio_np = audio
    audio_np = audio_np.astype(np.float32) / 32768.0
    audio_np = librosa.resample(audio_np, orig_sr=sampling_rate, target_sr=16000)
    text = mlx_whisper.transcribe(audio=audio_np.squeeze(),
                                  path_or_hf_repo="mlx-community/whisper-small.en-mlx")['text']
    print("text:", text)
    return text


def generate_llm(prompt: str, messages: list[dict]):
    messages.append({"role": "user", "content": prompt})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("prompt:", prompt)
    response = generate(llm, tokenizer, prompt, max_tokens=128)
    messages.append({"role": "assistant", "content": response})
    print("response:", response)
    return response, messages


def text_to_speech(text: str):
    speaker_id = tts.hps.data.spk2id["EN-US"]
    audio_chunk = tts.tts_to_file(text, speaker_id, quiet=False)
    print("audio_chunk:", audio_chunk.shape)
    audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
    yield (16000, audio_chunk.reshape(1, -1))


def response(audio: tuple[int, np.ndarray], messages: list[dict]):
    text = speech_to_text(audio)
    response, messages = generate_llm(text, messages)
    yield from text_to_speech(response)
    yield AdditionalOutputs(messages)


with gr.Blocks() as demo:
    gr.HTML(
    """
<div style='display: flex; gap: 15px; align-items: center'>
    <!-- Left column -->
    <div style='flex: 1; display: flex; justify-content: flex-end'>
        <img src="/gradio_api/file=logo.png" alt="Logo" style="height: 100px; width: auto;">
    </div>
    
    <!-- Right column -->
    <div style='flex: 2'>
        <h1>
            Hugging Face Speech To Speech (Powered by WebRTC ⚡️)
        </h1>
        <p>
            Each conversation is limited to 90 seconds. Once the time limit is up you can rejoin the conversation.
        </p>
    </div>
</div>
    """
    )
    with gr.Row():
       with gr.Group():
            transcript = gr.Chatbot(label="transcript", type="messages",
                                    value=[{"role": "system",
                                            "content": "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words."}],)
            audio = WebRTC(
                rtc_configuration=None,
                label="Stream",
                mode="send-receive",
                modality="audio",
            )
    audio.stream(ReplyOnPause(response), inputs=[audio, transcript], outputs=[audio], time_limit=90)
    audio.on_additional_outputs(lambda s: s, outputs=[transcript])


if __name__ == "__main__":
    demo.launch(allowed_paths=["logo.png"])