import asyncio
from google import genai
import os
import wave

# geminiAPI = os.environ.get("GOOGLE_API_KEY")
# client = genai.Client(api_key=geminiAPI, http_options={'api_version': 'v1alpha'})
client = genai.Client( http_options={'api_version': 'v1alpha'}) # api_key直接从环境变量中名称为GOOGLE_API_KEY获取
model_id = "gemini-2.0-flash-exp"

import logging

logger = logging.getLogger('Live')
logger.setLevel('INFO')

async def txt2txt():
    """
    gemini 2.0 multi-modal live api simple text to text with google_search
    """
    search_tool = {'google_search': {}}
    config = {"response_modalities": ["TEXT"],
              "tools": [search_tool]}
    async with client.aio.live.connect(model=model_id, config=config) as session:
        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break
            await session.send(message, end_of_turn=True)

            async for response in session.receive():
                if response.text is None:
                    continue
                print(response.text, end="")

def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """
    is to write it out to a .wav file. So here is a simple wave file writer
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
async def async_enumerate(it):
  n = 0
  async for item in it:
    yield n, item
    n +=1
class AudioLoop:
    """
    run - The main loop

    This method:

    Opens a websocket connecting to the Live API.
    Calls the initial setup method.
    Then enters the main loop where it alternates between send and recv until send returns False.

    send - Sends input text to the api

    The send method collects input text from the user, wraps it in a client_content message (an instance of BidiGenerateContentClientContent), and sends it to the model.

    If the user sends a q this method returns False to signal that it's time to quit.

    recv - Collects audio from the API and plays it

    The recv method collects audio chunks in a loop and writes them to a .wav file. It breaks out of the loop once the model sends a turn_complete method, and then plays the audio.
    """
  def __init__(self, turns=None,  config=None):
    self.session = None
    self.index = 0
    self.turns = turns
    if config is None:
      config={
          "generation_config": {
              "response_modalities": ["AUDIO"]}}
    self.config = config

  async def run(self):
    logger.debug('connect')
    async with client.aio.live.connect(model=model_id, config=self.config) as session:
      self.session = session

      async for sent in self.send():
        # Ideally send and recv would be separate tasks.
        await self.recv()

  async def _iter(self):
    if self.turns:
      for text in self.turns:
        print("message >", text)
        yield text
    else:
      print("Type 'q' to quit")
      while True:
        text = await asyncio.to_thread(input, "message > ")

        # If the input returns 'q' quit.
        if text.lower() == 'q':
          break

        yield text

  async def send(self):
    async for text in self._iter():
      logger.debug('send')

      # Send the message to the model.
      await self.session.send(text, end_of_turn=True)
      logger.debug('sent')
      yield text

  async def recv(self):
    # Start a new `.wav` file.
    file_name = f"audio_{self.index}.wav"
    with wave_file(file_name) as wav:
      self.index += 1

      logger.debug('receive')

      # Read chunks from the socket.
      turn = self.session.receive()
      async for n, response in async_enumerate(turn):
        logger.debug(f'got chunk: {str(response)}')

        if response.data is None:
          logger.debug(f'Unhandled server message! - {response}')
        else:
          wav.writeframes(response.data)
          if n == 0:
            print(response.server_content.model_turn.parts[0].inline_data.mime_type)
          print('.', end='')

      print('\n')

    # display(Audio(file_name, autoplay=True))
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(txt2txt())