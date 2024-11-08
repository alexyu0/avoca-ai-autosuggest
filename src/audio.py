"""Audio utilities."""
from loguru import logger
from typing import Callable
import numpy as np
import pyaudio
import soundcard as sc
import soundfile as sf
import speech_recognition as sr
import time
import wave

from src.llm import transcribe_audio_realtime
from src.constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE, TranscriptionModes


SPEAKER_ID = str(sc.default_speaker().name)
MIC_ID = str(sc.default_microphone().name)


def record_background(
    audio_data: list[bytes],
    transcription_mode: TranscriptionModes,
    record_sec: int = RECORD_SEC
) -> Callable:
    """
    Records an audio batch for a specified duration.

    Args:
        record_sec (int): The duration of the recording in seconds. Defaults to the value of
        RECORD_SEC.

    Returns:
        np.ndarray: The recorded audio sample.

    Example:
        ```python
        audio_sample = record_batch(5)
        print(audio_sample)
        ```
    """

    """
    def callback(_, audio):
        audio_data.append(audio.get_wav_data())

    def realtime_callback(_, audio):
        transcribe_audio_realtime(audio.get_wav_data())
        audio_data.append(audio.get_wav_data())

    r = sr.Recognizer()
    r.non_speaking_duration = 1
    r.pause_threshold = 1
    s = sr.Microphone()

    if transcription_mode == TranscriptionModes.Live:
        return r.listen_in_background(s, realtime_callback)
    else:
        return r.listen_in_background(s, callback)
    """

    p = pyaudio.PyAudio()
    device_info = p.get_default_input_device_info()
    sample_rate = device_info.get("defaultSampleRate")
    m = sr.Microphone()
    with m as s:
        audio_samples = []
        for _ in range(0, int(sample_rate / 1024 * record_sec)):
            audio_samples.append(s.stream.read(1024))

        transcribe_audio_realtime(b''.join(audio_samples))
        audio_data.append(b''.join(audio_samples))


def save_audio_file(
    audio_data: list[bytes],
    output_file_name: str = OUTPUT_FILE_NAME
) -> None:
    """
    Saves an audio data array to a file.

    Args:
        audio_data (np.ndarray): The audio data to be saved.
        output_file_name (str): The name of the output file. Defaults to the value of
        OUTPUT_FILE_NAME.

    Returns:
        None

    Example:
        ```python
        audio_data = np.array([0.1, 0.2, 0.3])
        save_audio_file(audio_data, "output.wav")
        ```
    """
    logger.debug(f"Saving audio file to {output_file_name}...")

    p = pyaudio.PyAudio()
    device_info = p.get_default_input_device_info()
    sample_width = int(p.get_sample_size(pyaudio.paInt16))
    channels = device_info.get("maxInputChannels")
    sample_rate = device_info.get("defaultSampleRate")

    with wave.open(output_file_name, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        for d in audio_data:
            w.writeframes(d)
    logger.debug("...Saved!")
