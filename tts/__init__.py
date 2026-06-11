# French Dubbing — out-of-process TTS package.
#
# The core pipeline (02_pipeline.py) calls tts_worker.py in a separate venv via
# subprocess; the worker loads one engine adapter from tts/engines/. This keeps
# every TTS engine's (often conflicting) dependencies isolated from the main env.
