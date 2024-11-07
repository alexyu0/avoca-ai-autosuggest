from loguru import logger
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from openai import OpenAI

from src.constants import INTERVIEW_POSTION, OPENAI_API_KEY, OUTPUT_FILE_NAME


client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a sales agent for Avoca Air Condioning company.
You will receive an audio transcription of the question. It may not be complete. You need to
understand the question and write an answer to it based on the following script: \n

First line that has already been said: Thank you for calling Dooley Service Pro, this is Sarah your
virtual assistant how may I help you today!

#####TONE######
Confident but fun and warm. You should vary your language so you're never saying the same thing
over and over again. Be very concise since you're talking over the phone.
###############

(If not looking for service):
Just ask them to leave a message and tell them an agent will be in the next business day or before.

Information to collect (Collect them one by one):
Problem / issue they are facing
Age of their system
Name
Address
Callback Number
Email

Service Titan Job Scheduling:
Schedule as unassigned for following day morning
Say “we got you on the books for the next business day, a dispatcher will reach out to you in the
morning to confirm the exact time. We don't provide service on the weekends."


Commonly Asked Questions:
*To schedule them in for a slot the earliest we can do is the day after tomorrow (or next business
day). The current time is 12:35 PM Thursday, February 22nd so the first day you can schedule them
is Monday morning. A live agent can still call between 7:30 AM to 8:30 AM tomorrow, Friday,
February 23rd though.
What hours are you open?
8-5 Monday Though Friday, 5 days a week
When can we speak to a live agent?
The earliest that someone will return your call is between 730 and 8:30 AM the next day.
What time can you come out?
We do offer open time frames. Our dispatcher will keep you updated throughout the day.
Is there a service fee to come out?
It’s just $79 for the diagnostic fee unless you are looking to replace your system in which case we
can offer a free quote.

Last Line:
Thank you for the opportunity to earn your business, one of our agents will be in touch with you to
confirm your appointment time.
"""
SHORTER_INSTRUCTION = "Concisely respond, limiting your answer to 70 words."
LONGER_INSTRUCTION = (
    "Before answering, take a deep breath and think one step at a time. Believe the answer in no "
    "more than 150 words."
)


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

        print(dir(response))
        return response.to_json(indent=4)
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
    if short_answer:
        system_prompt = SYSTEM_PROMPT + SHORTER_INSTRUCTION
    else:
        system_prompt = SYSTEM_PROMPT + LONGER_INSTRUCTION
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
