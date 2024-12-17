import gradio as gr
from gradio_webrtc import WebRTC, StreamHandler
import numpy as np
import os
import websockets.sync.client
from typing import cast
import io
from pydub import AudioSegment
from huggingface_hub import get_token
import logging

# Configure the root logger to WARNING to suppress debug messages from other libraries
logging.basicConfig(level=logging.WARNING)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Configure the logger for your specific library
logger = logging.getLogger("gradio_webrtc")
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

class SpeechToSpeechHandler(StreamHandler):

    def __init__(self, url: str, hf_token: str | None = None):
        self.url = url
        self.ws_url = url.replace("http", "ws") + "/ws"
        self.auth_token = hf_token or os.getenv("HF_TOKEN") or get_token()

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        self.ws = websockets.sync.client.connect(self.ws_url,
                                                 additional_headers=self.headers,
                                                 )
        super().__init__(expected_layout= "mono",
                         output_sample_rate=24000,
                         output_frame_size=960)
     
    def emit(self) -> tuple[int, np.ndarray]:
        data = self.ws.recv(decode=False)
        print("data:", data)
        audio = AudioSegment.from_file(io.BytesIO(cast(bytes, data)), format="mp3")
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(1, -1)
        return (audio.frame_rate, audio_array.reshape(1, -1))

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, audio = frame
        self.ws.send(audio.tobytes(), text=False)
    
    def copy(self) -> StreamHandler:
        return SpeechToSpeechHandler(self.url, self.auth_token)


def make_app(url, token): 

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
        webrtc = WebRTC(
            label="Conversation",
            modality="audio",
            mode="send-receive",
                rtc_configuration=None,
                )
        webrtc.stream(SpeechToSpeechHandler(url, token), inputs=[webrtc], outputs=[webrtc], time_limit=90)

    return demo

if __name__ == "__main__":
    demo = make_app("https://qgf95updzk4p96w4.us-east-1.aws.endpoints.huggingface.cloud", None)
    demo.launch(allowed_paths=["logo.png"])