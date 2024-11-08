import textwrap
import time

import numpy as np
import PySimpleGUI as sg
from loguru import logger

from src import audio, llm
from src.constants import (
    TranscriptionModes,
    APPLICATION_WIDTH,
    OFF_IMAGE,
    ON_IMAGE,
    OUTPUT_FILE_NAME
)


OUTPUT_AUDIO_FILE = OUTPUT_FILE_NAME
AUDIO_SAVED = False
AUDIO_DATA = []
TRANSCRIPTION_MODE = TranscriptionModes.Live
CURRENT_TRANSCRIPT = ''
MESSAGE_HISTORY = []


def get_text_area(text: str, size: tuple) -> sg.Text:
    """
    Create a text area widget with the given text and size.

    Parameters:
        text (str): The initial text to display in the text area.
        size (tuple): The size of the text area widget.

    Returns:
        sg.Text: The created text area widget.
    """
    return sg.Text(
        text,
        size=size,
        background_color=sg.theme_background_color(),
        text_color="white",
        font=('Arial', 12)
    )


class BtnInfo:
    def __init__(self, state=False):
        self.state = state


# All the stuff inside your window:
sg.theme("DarkAmber")  # Add a touch of color
record_status_button = sg.Button(
    image_data=OFF_IMAGE,
    k="-TOGGLE-RECORDING-",
    border_width=0,
    button_color=(sg.theme_background_color(), sg.theme_background_color()),
    disabled_button_color=(sg.theme_background_color(), sg.theme_background_color()),
    metadata=BtnInfo(),
    size=(int(APPLICATION_WIDTH * 0.1), 1)
)
analyzed_text_label = get_text_area("", size=(int(APPLICATION_WIDTH * 0.3), 2))
transcribed_message = get_text_area("", size=(int(APPLICATION_WIDTH * 0.5), 6))
quick_chat_gpt_answer = get_text_area("", size=(int(APPLICATION_WIDTH * 0.5), 6))
full_chat_gpt_answer = get_text_area("", size=(int(APPLICATION_WIDTH * 0.5), 6))
message_history = sg.Text('')

layout = [
    [sg.Column([[
        sg.Column([
            [
                sg.Text("Run transcription: "),
                record_status_button
            ]
        ]),
        sg.Column([[sg.Text('', size=(int(APPLICATION_WIDTH * 0.1), 0))]]),
        sg.Column([
                [
                    sg.Text("Suggest response: "),
                    sg.Button(
                        'Run',
                        size=(int(APPLICATION_WIDTH * 0.1), 1),
                        k="-RUN-TRANSCRIPTION-"
                    )
                ]
        ])
    ]], element_justification='c', expand_x=True)],
    [
        [sg.Text('', size=(int(APPLICATION_WIDTH * 2), 0))],
        sg.Column([
            [sg.Text(
                "Previous messages:", font=('Arial', 18), size=(int(APPLICATION_WIDTH * 0.6), 1)
            )],
            [message_history],
        ], expand_y=True, element_justification='top'),
        sg.Column([
            [analyzed_text_label],
            [sg.Text("Current message:", font=('Arial', 18))],
            [transcribed_message],
            [sg.Text("Suggested responses:", font=('Arial', 18))],
            [sg.Text("Short answer:", font=('Arial', 14))],
            [quick_chat_gpt_answer],
            [sg.Text("Full answer:", font=('Arial', 14))],
            [full_chat_gpt_answer],
        ], expand_y=True, element_justification='top')
    ],
    [sg.Button("Cancel")],
]
WINDOW = sg.Window("Keyboard Test", layout, return_keyboard_events=True, use_default_focus=False)


def background_recording_loop() -> None:
    global AUDIO_SAVED, AUDIO_DATA  # pylint: disable=global-statement

    AUDIO_SAVED = False
    llm.transcript_queue.clear()
    AUDIO_DATA = []
    # stop_recording = audio.record_background(AUDIO_DATA, TRANSCRIPTION_MODE)
    # while record_status_button.metadata.state:
    #     time.sleep(0.1)
    # stop_recording()
    while record_status_button.metadata.state:
        audio.record_background(AUDIO_DATA, TRANSCRIPTION_MODE, 1)
    audio.save_audio_file(AUDIO_DATA, OUTPUT_AUDIO_FILE)
    AUDIO_SAVED = True


def background_transcription_loop():
    global CURRENT_TRANSCRIPT  # pylint: disable=global-statement

    while record_status_button.metadata.state:
        interim_message = None
        while len(llm.transcript_queue) > 0:
            sentence = llm.transcript_queue[0]
            if sentence[0]:
                interim_message = None
                llm.transcript_queue.popleft()
                if not CURRENT_TRANSCRIPT.endswith(' '):
                    CURRENT_TRANSCRIPT += f" {sentence[1]}"
                else:
                    CURRENT_TRANSCRIPT += f"{sentence[1]}"

                transcribed_message.update(CURRENT_TRANSCRIPT)
            elif not interim_message or interim_message.strip() != sentence[1]:
                if not CURRENT_TRANSCRIPT.endswith(' '):
                    interim_message = f" {sentence[1]}"
                else:
                    interim_message = f"{sentence[1]}"

                transcribed_message.update(CURRENT_TRANSCRIPT + interim_message)

        time.sleep(0.1)


def run_ui():
    global CURRENT_TRANSCRIPT  # pylint: disable=global-statement

    if TRANSCRIPTION_MODE == TranscriptionModes.Live:
        llm.start_dg_connection()
        while not llm.dg_connection.is_connected():
            logger.debug("Waiting for connection...")
            time.sleep(0.1)

    while True:
        event, values = WINDOW.read()
        if event in ["Cancel", sg.WIN_CLOSED]:
            if TRANSCRIPTION_MODE == TranscriptionModes.Live:
                llm.close_dg_connection()
            logger.debug("Closing...")
            break

        if event in ("r", "R", "-TOGGLE-RECORDING-"):  # start recording
            record_status_button.metadata.state = not record_status_button.metadata.state
            if record_status_button.metadata.state:
                CURRENT_TRANSCRIPT = ''
                WINDOW.perform_long_operation(background_recording_loop)
                if TRANSCRIPTION_MODE == TranscriptionModes.Live:
                    WINDOW.perform_long_operation(background_transcription_loop)
            else:
                MESSAGE_HISTORY.append(CURRENT_TRANSCRIPT)
                if TRANSCRIPTION_MODE == TranscriptionModes.Live:
                    analyzed_text_label.update("Start analyzing...")
                    WINDOW.write_event_value("-TRANSCRIPTION COMPLETE-", None)

                message_history.update('\n\n\n'.join(
                    [textwrap.fill(m, 150) for m in MESSAGE_HISTORY]
                ))

            record_status_button.update(
                image_data=ON_IMAGE if record_status_button.metadata.state else OFF_IMAGE
            )

        elif event in ("a", "A", "-RUN-TRANSCRIPTION-"):
            if TRANSCRIPTION_MODE == TranscriptionModes.Prerecorded:
                # send audio to OpenAI Whisper model
                if record_status_button.metadata.state:
                    record_status_button.metadata.state = not record_status_button.metadata.state
                record_status_button.update(
                    image_data=ON_IMAGE if record_status_button.metadata.state else OFF_IMAGE
                )

                if not AUDIO_SAVED:
                    audio.save_audio_file(audio_data, OUTPUT_AUDIO_FILE)

                logger.debug("Analyzing audio...")
                analyzed_text_label.update("Start analyzing...")
                WINDOW.perform_long_operation(llm.transcribe_audio, "-TRANSCRIPTION COMPLETE-")
            else:
                analyzed_text_label.update("Start analyzing...")
                while len(llm.transcript_queue) > 0:
                    sentence = llm.transcript_queue[0]
                    if llm.transcript_queue[0][0]:
                        llm.transcript_queue.popleft()
                        if not CURRENT_TRANSCRIPT.endswith(' '):
                            CURRENT_TRANSCRIPT += f" {sentence[1]}"
                        else:
                            CURRENT_TRANSCRIPT += f"{sentence[1]}"

                WINDOW.write_event_value("-TRANSCRIPTION COMPLETE-", None)

        elif event == "-TRANSCRIPTION COMPLETE-":
            if TRANSCRIPTION_MODE == TranscriptionModes.Prerecorded:
                audio_transcript = values["-TRANSCRIPTION COMPLETE-"]
                transcribed_message.update(audio_transcript)
            else:
                audio_transcript = CURRENT_TRANSCRIPT

            analyzed_text_label.update("...done")

            # Generate quick answer:
            quick_chat_gpt_answer.update("Chatgpt is working...")
            WINDOW.perform_long_operation(
                lambda: llm.generate_answer(audio_transcript, short_answer=True, temperature=0),
                "-CHAT_GPT SHORT ANSWER-",
            )

            # Generate full answer:
            full_chat_gpt_answer.update("Chatgpt is working...")
            WINDOW.perform_long_operation(
                lambda: llm.generate_answer(audio_transcript, short_answer=False, temperature=0.7),
                "-CHAT_GPT LONG ANSWER-",
            )
        elif event == "-CHAT_GPT SHORT ANSWER-":
            quick_chat_gpt_answer.update(values["-CHAT_GPT SHORT ANSWER-"])
        elif event == "-CHAT_GPT LONG ANSWER-":
            full_chat_gpt_answer.update(values["-CHAT_GPT LONG ANSWER-"])
