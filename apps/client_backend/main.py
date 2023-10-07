import io
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List

import typing
from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, \
    TimeEndpointingConfig
from vocode.streaming.models.websocket import AudioConfigStartMessage, ReadyMessage, WebSocketMessage, \
    WebSocketMessageType, AudioMessage
from vocode.streaming.output_device.websocket_output_device import WebsocketOutputDevice
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.client_backend.conversation import ConversationRouter
from vocode.streaming.models.message import BaseMessage

from dotenv import load_dotenv
from vocode.streaming.transcriber import DeepgramTranscriber
from pydub import AudioSegment

load_dotenv()

app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class SynthesizedAudioChunk:
    audio: bytes
    time_delay_milliseconds: int


class WebsocketOutputDeviceWithRecording(WebsocketOutputDevice):
    """
    WebsocketOutputDevice implementation that also stores audio chunks in output_audio_bytes_list.

    We include the time delay when we append audio chunks so that the caller knows where to append the given audio
    chunk.
    """
    def __init__(
            self, ws: WebSocket, sampling_rate: int, audio_encoding: AudioEncoding,
            output_audio_bytes_list: List[SynthesizedAudioChunk], start_time: datetime
    ):
        super().__init__(ws, sampling_rate, audio_encoding)
        self.output_audio_bytes_list = output_audio_bytes_list
        self.start_time = start_time

    def consume_nonblocking(self, chunk: bytes):
        if self.active:
            audio_message = AudioMessage.from_bytes(chunk)
            self.queue.put_nowait(audio_message.json())

            self.output_audio_bytes_list.append(SynthesizedAudioChunk(
                audio=chunk,
                time_delay_milliseconds=int((datetime.now() - self.start_time).total_seconds() * 1000)
            ))


class ConversationRouterWithRecording(ConversationRouter):
    """
    Overrides the parent class conversation() function. As audio chunks come in from the client and the synthesizer,
    we combine them into a single audio file and save to /code/combined_audio.wav.
    """

    async def conversation(self, websocket: WebSocket):
        await websocket.accept()
        # a continuous audio stream coming in from the client.
        client_audio_bytes_list: List[bytes] = []
        # discrete audio chunks that are outputted from the synthesizer.
        synthesized_audio_chunks_list: List[SynthesizedAudioChunk] = []
        try:
            start_message: AudioConfigStartMessage = AudioConfigStartMessage.parse_obj(
                await websocket.receive_json()
            )
            self.logger.debug(f"Conversation started")
            start_time = datetime.now()
            output_device = WebsocketOutputDeviceWithRecording(
                websocket,
                start_message.output_audio_config.sampling_rate,
                start_message.output_audio_config.audio_encoding,
                # pass in synthesized_audio_chunks_list, so the output device can add synthesized audio chunks as they
                # come in
                synthesized_audio_chunks_list,
                start_time
            )
            conversation = self.get_conversation(output_device, start_message)
            await conversation.start(lambda: websocket.send_text(ReadyMessage().json()))

            while conversation.is_active():
                message: WebSocketMessage = WebSocketMessage.parse_obj(
                    await websocket.receive_json()
                )
                if message.type == WebSocketMessageType.STOP:
                    break
                audio_message = typing.cast(AudioMessage, message)
                audio_bytes = audio_message.get_bytes()
                # append audio bytes from client as they come in
                client_audio_bytes_list.append(audio_bytes)
                conversation.receive_audio(audio_bytes)
            output_device.mark_closed()
        except WebSocketDisconnect:
            self.logger.debug("Websocket disconnected")
        finally:
            client_audio_data = b''.join(client_audio_bytes_list)
            client_audio_file = io.BytesIO(client_audio_data)
            client_audio = AudioSegment.from_raw(client_audio_file, sample_width=2,
                                                 frame_rate=start_message.input_audio_config.sampling_rate, channels=1)

            # Initialize an empty audio segment for synthesized audio
            synthesized_audio_overlay = AudioSegment.silent(duration=len(client_audio))

            # Logic for overlaying discrete synthesized audio chunks on top of continuous client audio.
            for synthesized_chunk in synthesized_audio_chunks_list:
                synthesized_audio_file = io.BytesIO(synthesized_chunk.audio)
                synthesized_audio = AudioSegment.from_raw(synthesized_audio_file, sample_width=2,
                                                          frame_rate=start_message.output_audio_config.sampling_rate,
                                                          channels=1)
                synthesized_audio_overlay = synthesized_audio_overlay.overlay(
                    # crossfading to smooth out the audio chunk boundaries
                    synthesized_audio.fade_in(10).fade_out(10),
                    # offset the chunk so it plays at the right time
                    position=synthesized_chunk.time_delay_milliseconds)

            # overlay synthesized audio on top of client audio
            combined_audio = client_audio.overlay(synthesized_audio_overlay)
            # save in a file
            combined_audio.export("./combined_audio.wav", format="wav")

            await conversation.terminate()


conversation_router = ConversationRouterWithRecording(
    agent_thunk=lambda: ChatGPTAgent(
        ChatGPTAgentConfig(
            end_conversation_on_goodbye=True,
            initial_message=BaseMessage(text="Hi!"),
            prompt_preamble="Your preamble..."
        )
    ),
    synthesizer_thunk=lambda output_audio_config: AzureSynthesizer(
        AzureSynthesizerConfig.from_output_audio_config(
            output_audio_config, voice_name="en-US-SteffanNeural",
            # important: need to encode as wav
            **{"should_encode_as_wav": True}
        ),
    ),
    transcriber_thunk=lambda input_audio_config: DeepgramTranscriber(
        DeepgramTranscriberConfig.from_input_audio_config(
            input_audio_config=input_audio_config,
            endpointing_config=TimeEndpointingConfig(
                time_cutoff_seconds=2.0
            )
        )
    ),
    logger=logger,
)

app.include_router(conversation_router.get_router())
