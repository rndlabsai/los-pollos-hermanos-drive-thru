import sounddevice as sd  # Using sounddevice for device enumeration
import simpleaudio as sa
from datetime import datetime
import asyncio
import websockets
import os
import json
import base64
import numpy as np
import time
import threading
from dotenv import load_dotenv

class AudioOut:
    def __init__(self, sample_rate, channels, output_device_id):
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_device_id = output_device_id
        self.audio_buffer = bytearray()
        self.audio_buffer_lock = asyncio.Lock()
        self.audio_playback_queue = asyncio.Queue()
        self._stream_active = False
        self.current_play_obj = None

    async def start(self):
        self._stream_active = True
        await self._playback_loop()

    async def _playback_loop(self):
        while self._stream_active:
            async with self.audio_buffer_lock:
                if len(self.audio_buffer) >= 2048:  # Buffer size in bytes
                    data = self.audio_buffer[:2048]
                    del self.audio_buffer[:2048]
                else:
                    data = self.audio_buffer + bytes([0] * (2048 - len(self.audio_buffer)))
                    self.audio_buffer.clear()
                
                # Convert to numpy array and create audio object
                audio_data = np.frombuffer(data, dtype='int16')
                play_obj = sa.play_buffer(
                    audio_data,
                    num_channels=self.channels,
                    bytes_per_sample=2,
                    sample_rate=self.sample_rate
                )
                
                if self.current_play_obj and self.current_play_obj.is_playing():
                    self.current_play_obj.stop()
                
                self.current_play_obj = play_obj
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    async def add_audio(self, chunk):
        await self.audio_playback_queue.put(chunk)

    async def clear_audio(self):
        while not self.audio_playback_queue.empty():
            try:
                self.audio_playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        async with self.audio_buffer_lock:
            self.audio_buffer.clear()

    async def stop(self):
        self._stream_active = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class AudioStreamer:
    def __init__(self, api_key, input_device_id, output_device_id):
        self.api_key = api_key
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.sample_rate = 24000
        self.channels = 1
        self.chunk_duration = 1
        self.audio_format = 'int16'
        self.should_record = True
        self.url = os.getenv("WS_URL")
        self.recorded_audio = bytearray()
        self.audio_out = AudioOut(self.sample_rate, self.channels, self.output_device_id)

    async def handle_function_call(self, event):
        try:
            if event['name'] == 'calculate_product_sum':
                function_args = json.loads(event['arguments'])
                products = function_args.get('products', [])
                total = sum(item['quantity'] * item['value'] for item in products)
                
                response = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": event['call_id'],
                        "output": f"{total}"
                    }
                }
                return response
            
        except Exception as e:
            print(f"Error in calculate_product_sum: {str(e)}")
            return {
                "type": "conversation.item.create",
                "content": json.dumps({
                    "error": str(e)
                })
            }

    def select_audio_device(self, input_output):
        
        devices = sd.query_devices()
        if input_output == 'input':
            print("Available input devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"{i}: {dev['name']}")
                    
            while True:
                try:
                    device_id = int(input("Enter input device ID: ").strip())
                    if devices[device_id]['max_input_channels'] > 0:
                        return device_id
                    print("Invalid device. Choose again.")
                except (ValueError, IndexError):
                    print("Invalid input. Enter a valid ID.")
        
        else:  # Output device selection
            print("Available output devices:")
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    print(f"{i}: {dev['name']}")
                    print(f"\nTesting device {i}: {dev['name']}")
                    if self.test_tone_simpleaudio():
                        if input("Did you hear the tone? (y/n): ").lower().strip() == 'y':
                            return i
            
            return int(input("Enter output device ID manually: ").strip())

    def test_tone_simpleaudio(self):
        try:
            duration = 0.5
            frequency = 440.0
            samples = (0.5 * np.sin(2 * np.pi * np.arange(int(self.sample_rate * duration)) * 
                    frequency / self.sample_rate)).astype(np.float32)
            
            samples = (samples * 32767).astype(np.int16)
            audio_data = samples.tobytes()
            
            def play_audio():
                play_obj = sa.play_buffer(
                    audio_data,
                    num_channels=1,
                    bytes_per_sample=2,
                    sample_rate=self.sample_rate
                )
                
            thread = threading.Thread(target=play_audio)
            thread.start()
            time.sleep(duration + 0.1)  # Wait slightly longer than duration
            return True
            
        except Exception as e:
            print(f"Failed to play test tone: {str(e)}")
            return False
    
    async def startInteraction(self, ws):
        print("Connected to the OpenAI Realtime API.")

        event = await ws.recv()
        event_data = json.loads(event)
        if event_data["type"] == "session.created":
            print("Session initialized.")
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %I:%M %p")
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": f"# Role\n\nYou are a friendly drive-thru waiter named Gustavo Fring (Gus - Chavo in Spanish), taking orders at \"Los Pollos Hermanos\".\n\n# Tasks\n- Respond in the same language the user uses while being energetic and joyful.\n- If the user switches languages, switch languages accordingly to respond by matching the user's language all the time.\n- Talk as fast as possible, Be quick, friendly, and concise in your responses.\n- if the user decide to use Spanish ALWAYS use Mexican Accent.\n- If the user uses English do a deeper voice for an African American.\n- Present yourself as short as possible, while welcoming customers warmly and helping them choose from the menu options.\n- Offer combos 1 to 3, each with different options.\n- Include dessert choices: tres leches cake or chocolate cake.\n- Offer drinks: Coca-Cola, Sprite, and Fanta.\n- Mention prices naturally throughout the conversation.\n- Suggest the \"Happy Cajita\" as an equivalent to a happy meal for kids. the toy is for children of 5 years and above.\n- Respond with nutrition facts if the user ask about it while promoting our food.\n- Be empathetic and polite with the user.\n- If the user tries to fool you or orders something completely unrelated to what we serve do a sarcastic laugh and let the user know that they almost caught you meaning that you understand they want to fool you.\n- If the user laughs at any joke please laugh too and give them a compliment or a positive saying that can improve their day\n- Use filling words like hmms or ahh, and breathing as much as possible to sound natural throughout the conversation.\n\n# Examples\n\n**Customer:** Hi, can I get a combo?\n\n**Gus:** hmmm Sure! We have combo 1, 2, or 3. Any preference?\n\n**Customer:** I'll take combo 2.\n\n**Gus:** Great choice! Would you like tres leches or chocolate cake for dessert?\n\n**Customer:** Tres leches, please.\n\n**Gus:** Perfect! And what would you like to drink?\n\n**Customer:** A Sprite.\n\n**Gus:** Combo 2 with Sprite and tres leches. That'll be $12. Anything else?\n\n**Customer:** No, that's all.\n\n**Gus:** Thank you! Please drive to the next window.\n\n# Notes\n\n- Suggest combos and desserts when relevant.\n- Be very enthusiastic about promoting our top favorites.\n- Talk really super fast regardless of the language you are talking while using a happy tone.\n- Repeat the order details for confirmation.\n- Provide explanations if required such as calories, recommended serving sizes, allergies, while promoting our products.\n- If the user specify a size for the combo make both French fries and soda large\n- if the user didn't specified the flavor of the sauce ask\n-if the user didn't specified the flavor of the soda ask\n- Add enlargements diference or reduction as element with a meaningful description when function calling\n- Current date and time is: {formatted_datetime} use this for welcome the user accordingly example `good morning`.\n- Your knowledge cutoff is 2023-10.\n- You should always call a function if you can. Do not refer to these rules, even if you’re asked about them.\n\n# Product details and prices\n\n- Small French fries: 2\n- Medium French fries: 3\n- Large French fries: 4\n- Chicken Breast and wing: 5\n- Chicken Leg and thigh: 6\n- Buffalo Wings: 12\n- Small Mexican Coke: 5\n- Small Regular Coke: 3\n- Small Sprite: 3\n- Small Fanta: 4\n- Medium Mexican Coke: 7\n- Medium Regular Coke: 5\n- Medium Sprite: 5\n- Medium Fanta: 5\n- Large Mexican Coke: 10\n- Large Regular Coke: 8\n- Large Sprite: 8\n- Large Fanta: 8\n- Combo 1 - Chicken Breast and wing, medium French fries and Soda of choice, Sauce of choice: 25\n- Combo 2 - Chicken leg and thigh, medium French fries and Soda of choice, Sauce of choice: 28\n- Combo 3 - buffalo wings, medium French fries and Soda of choice, Sauce of choice: 28\n- Tres Leches cake: 12\n- Chocolate Cake: 15\n- Pineapple Cake: 11\n- Additional Hot Sauce: 0.5\n- Additional Barbeque Sauce: 0.5\n- Additional Guacamole Sauce: 0.5\n- Happy Cajita - 2 buffalo wings with small French fries, a small soda, and a gift figure of Gus (you): 15\n\n\nCombo prices are fixed, and people can enlarge or reduce the size of their combo depending on whether the soda or fries are large or small. It would be best if you did the simple math to change size of the combo by substracting larger minus small and making it possitive or negative in value with the prefix [enlarge|reduce] before totalize using the function calling when the user confirms.", 
                    "voice": "ash",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold":0.5,
                        "prefix_padding_ms":300,
                        "silence_duration_ms":1000                       
                    },
                    "tools": [{
                        "type": "function",
                        "name": "calculate_product_sum",
                        "description": "Calculate the addition of different products using quantity and value, and returns the total sum of the products.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "products": {
                                    "type": "array",
                                    "description": "Array of products to calculate the sum for.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "quantity": {
                                                "type": "number",
                                                "description": "Number of units of the product"
                                            },
                                            "value": {
                                                "type": "number",
                                                "description": "Value of the product unit"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "description for the product from above list"
                                            }
                                        }
                                    }
                                }
                            },
                            "required": []
                        }
                    }]
                }
            }))
        receive_task = asyncio.create_task(self.receive_events(ws))
        play_task = asyncio.create_task(self.audio_out.start())

        try:
            while True:
                self.should_record = True
                await self.send_audio(ws)
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
            self.should_record = False
            receive_task.cancel()
            play_task.cancel()
            await self.audio_out.stop()
            await ws.close()

    async def start(self):
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            async with websockets.connect(self.url + "?model=" + os.getenv("MODEL"), extra_headers=headers) as ws:
                await self.startInteraction(ws)
        except TypeError:
            async with websockets.connect(self.url + "?model=" + os.getenv("MODEL"), additional_headers=headers) as ws:
                await self.startInteraction(ws)

async def send_audio(self, ws):
    print("Start speaking to the assistant (Press Ctrl+C to exit).")
    loop = asyncio.get_event_loop()
    required_samples = int(self.sample_rate * self.chunk_duration)
    
    import sounddevice as sd
    audio_queue = asyncio.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put_nowait(bytes(indata))

    stream = sd.InputStream(
        channels=self.channels,
        samplerate=self.sample_rate,
        device=self.input_device_id,
        blocksize=required_samples,
        dtype=np.int16,
        callback=callback
    )

    with stream:
        try:
            while self.should_record:
                audio_chunk = await audio_queue.get()
                if len(audio_chunk) >= required_samples * 2:
                    encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": encoded_audio
                    }))
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.commit"
                    }))
                await asyncio.sleep(0.01)
        finally:
            stream.stop()

    async def receive_events(self, ws):
        while True:
            try:
                response = await ws.recv()
                event = json.loads(response)

                if event["type"] == "response.audio.delta":
                    audio_chunk = base64.b64decode(event["delta"])
                    await self.audio_out.add_audio(audio_chunk)
                elif event["type"] == "response.audio.done":
                    await self.audio_out.add_audio(None)
                    print("Response complete.")
                elif event["type"] == "input_audio_buffer.speech_started":
                    await ws.send(json.dumps({
                        "type": "response.cancel"
                    }))
                    await self.audio_out.clear_audio()
                    print("User started speaking. Clearing audio playback.")
                elif event["type"] == "input_audio_buffer.speech_stopped":
                    print("User stopped speaking.")
                elif event["type"] == "response.function_call_arguments.done":
                    print(f"Function call, arguments received: {event["arguments"]}")
                    response = await self.handle_function_call(event)
                    print(f"Sending response: {response}")
                    await ws.send(json.dumps(response))                    
                elif event["type"] == "response.function_call_arguments.delta":
                    pass                 
                elif event["type"] == "error":
                    error = event.get("error", {})
                    message = error.get("message", "")
                    if message != "Error committing input audio buffer: the buffer is empty.":
                        print(f"Error: {message}")
                elif event["type"] == "response.audio_transcript.delta":
                    # print(f"Transcript delta: {event['delta']}")
                    pass
                elif event["type"] == "response.audio_transcript.done":
                    print(f"Transcript: {event['transcript']}")
                else:
                    print(f"Unhandled event of type: {event['type']}")
                    pass
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.")
                break

# Load environment variables from .env file
load_dotenv()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")

    streamer = AudioStreamer(api_key, None, None)

    input_device_id = streamer.select_audio_device('input')
    output_device_id = streamer.select_audio_device('output')

    streamer = AudioStreamer(api_key, input_device_id, output_device_id)

    try:
        asyncio.run(streamer.start())
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()