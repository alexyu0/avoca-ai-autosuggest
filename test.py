import soundcard as sc

def a():
    # get a list of all speakers:
    speakers = sc.all_speakers()
    # get the current default speaker on your system:
    default_speaker = sc.default_speaker()
    # get a list of all microphones:
    mics = sc.all_microphones()
    # get the current default microphone on your system:
    default_mic = sc.default_microphone()

    datas = []
    with default_mic.recorder(samplerate=48000) as mic, \
      default_speaker.player(samplerate=48000) as sp:
        for _ in range(100):
            data = mic.record(numframes=48000)
            sp.play(data)
            datas.append(data)
            print(data.any())
            print('looping')

    print([d.any() for d in datas])



import pyaudio
import numpy as np
import wave
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 48000
RECORD_SEC = 2

def b():
    p = pyaudio.PyAudio()

    device_info = p.get_default_input_device_info()
    assert (
        isinstance(device_info.get("defaultSampleRate"), (float, int)) and
        device_info["defaultSampleRate"] > 0
    ), f"Invalid device info returned from PyAudio: {device_info}"

    sample_width = p.get_sample_size(FORMAT)
    sample_rate = int(device_info["defaultSampleRate"])

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK,
    )

    wf = wave.open('test.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)

    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SEC)):
        data = stream.read(CHUNK)
        wf.writeframes(data)
        frames.append(np.frombuffer(data, dtype=np.float32))


    stream.stop_stream()
    stream.close()
    p.terminate()


import speech_recognition as sr
import soundfile as sf

asdf = []

def callback(recognizer, audio):
    global asdf
    print(len(audio.get_wav_data()))
    asdf.append(audio.get_wav_data())
    with open(f'test{len(asdf)}.wav', 'wb') as f:
        f.write(audio.get_wav_data())

def c():
    r = sr.Recognizer()
    r.pause_threshold = 0.5
    s = sr.Microphone()
    stop = r.listen_in_background(s, callback)
    import time
    time.sleep(10)
    stop()
    # with sr.Microphone() as source:
    #     print("Say something!")
    #     audio = r.listen(source)


    w = wave.open('test.wav', 'wb')
    for i in asdf:
        w.writeframes(i)


c()
