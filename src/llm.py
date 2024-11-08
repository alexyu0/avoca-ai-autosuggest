from collections import deque
from datetime import datetime, timedelta
import threading

from loguru import logger
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    FileSource,
    LiveOptions,
    LiveTranscriptionEvents,
    PrerecordedOptions,
)
from openai import OpenAI

from src.constants import (
    OPENAI_API_KEY,
    OUTPUT_FILE_NAME,
    DEEPGRAM_API_KEY,
    SYSTEM_PROMPT,
    SHORTER_INSTRUCTION,
    LONGER_INSTRUCTION
)


client = OpenAI(api_key=OPENAI_API_KEY)
dg_connection = None
transcript_queue = deque()


def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribes an audio file into text.

    Args:
        path_to_file (str, optional): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the audio file fails to transcribe.
    """

    try:
        deepgram = DeepgramClient()
        with open(path_to_file, "rb") as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        return response.results.channels[0].alternatives[0].transcript
    except Exception as e:
        logger.error(f"Can't transcribe audio: {e}")
        raise e


def start_dg_connection():
    global dg_connection  # pylint: disable=global-statement

    def on_open(self, open, **kwargs):
        logger.info("Connection Open")


    def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return
        if result.is_final:
            logger.debug(f"Transcription: {sentence}")
            if len(transcript_queue) > 0 and not transcript_queue[-1][0]:
                transcript_queue.pop()
            transcript_queue.append((True, sentence))
        else:
            logger.debug(f"Interim transcription: {sentence}")
            if len(transcript_queue) > 0:
                transcript_queue.pop()
            transcript_queue.append((False, sentence))

    def on_error(self, error, **kwargs):
        logger.error(f"Error: {error}")

    def on_speech_started(self, speech_started, **kwargs):
        logger.debug("Speech Started")

    try:
        deepgram = DeepgramClient(
            api_key=DEEPGRAM_API_KEY,
            config=DeepgramClientOptions(options={"keepalive": "true"})
        )
        dg_connection = deepgram.listen.websocket.v("1")
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=48000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=100,
        )
        dg_connection.start(options)
    except Exception as e:
        logger.error(f"Can't start Deepgram connection: {e}")
        raise e


def close_dg_connection():
    dg_connection.finish()


def transcribe_audio_realtime(audio_data: bytes) -> str:
    """
    Transcribes a chunk of audio data into text.

    Args:
        audio_data (bytes): The audio data to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the audio file fails to transcribe.
    """
    try:
        dg_connection.send(audio_data)
    except Exception as e:
        logger.error(f"Can't transcribe audio: {e}")
        raise e


def generate_answer(transcript: str, short_answer: bool = True, temperature: float = 0.7) -> str:
    """
    Generates an answer based on the given transcript using the OpenAI GPT-3.5-turbo model.

    Args:
        transcript (str): The transcript to generate an answer from.
        short_answer (bool): Whether to generate a short answer or not. Defaults to True.
        temperature (float): The temperature parameter for controlling the randomness of the
        generated answer.

    Returns:
        str: The generated answer.

    Example:
        ```python
        transcript = "Can you tell me about the weather?"
        answer = generate_answer(transcript, short_answer=False, temperature=0.8)
        print(answer)
        ```

    Raises:
        Exception: If the LLM fails to generate an answer.
    """
    now = datetime.now()
    schedule_date = now + timedelta(days=2)
    if schedule_date.strftime('%a') == 'Sat':
        schedule_date += timedelta(days=2)
    elif schedule_date.strftime('%a') == 'Sun':
        schedule_date += timedelta(days=1)

    call_date = now + timedelta(days=1)
    if call_date.strftime('%a') == 'Sat':
        call_date += timedelta(days=2)
    elif call_date.strftime('%a') == 'Sun':
        call_date += timedelta(days=1)
    schedule = (
        'The current time and date is {} so the first day you can schedule is {} morning. '
            .format(now.strftime('%I:%M %p %A, %B %d'), schedule_date.strftime('%A, %B %d')),
        'A live agent can still call between 7:30 AM to 8:30 AM {} though.'
            .format(schedule_date.strftime('%A, %B %d'))
    )

    if short_answer:
        system_prompt = SYSTEM_PROMPT.format(scheduling_prompt=schedule) + SHORTER_INSTRUCTION
    else:
        system_prompt = SYSTEM_PROMPT.format(scheduling_prompt=schedule) + LONGER_INSTRUCTION
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error
    return response.choices[0].message.content
