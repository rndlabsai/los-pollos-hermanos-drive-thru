from datetime import datetime
import asyncio
import websockets
import os
import json
import base64
import sounddevice as sd
import numpy as np
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
        self.stream = None

    async def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=self._audio_callback,
            device=self.output_device_id,
            latency='low'
        )
        self.stream.start()
        await self._playback_loop()

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        bytes_to_read = frames * self.channels * 2
        with threading.Lock():
            if len(self.audio_buffer) >= bytes_to_read:
                data = self.audio_buffer[:bytes_to_read]
                del self.audio_buffer[:bytes_to_read]
            else:
                data = self.audio_buffer + bytes([0] * (bytes_to_read - len(self.audio_buffer)))
                self.audio_buffer.clear()
        outdata[:] = np.frombuffer(data, dtype='int16').reshape(-1, self.channels)

    async def _playback_loop(self):
        while True:
            chunk = await self.audio_playback_queue.get()
            if chunk is None:
                continue
            async with self.audio_buffer_lock:
                self.audio_buffer.extend(chunk)

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
        if self.stream:
            self.stream.stop()
            self.stream.close()

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
        self.extra_headers = True
        self.url = os.getenv("WS_URL")
        self.recorded_audio = bytearray()
        self.audio_out = AudioOut(self.sample_rate, self.channels, self.output_device_id)

    async def handle_function_call(self, event, ws):
        try:
            print(f"Handling function call: {event['name']}")
            if event['name'] == 'sub_total_order_not_final' or event['name'] == 'place_order':
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
                if event['name'] == 'sub_total_order_not_final':
                    ws.send(json.dumps({ "type": "conversation.item.create", "content": json.dumps({ "text": "ok" }) }))
                elif event['name'] == 'place_order':
                    await self.audio_out.stop()
                    await ws.close()
                    headers = {
                        "Authorization": "Bearer " + self.api_key,
                        "OpenAI-Beta": "realtime=v1",
                    }
                    if(self.extra_headers):
                        async with websockets.connect(self.url + "?model=" + os.getenv("MODEL"), extra_headers=headers) as ws:
                            await self.startInteraction(ws)
                    else:
                        async with websockets.connect(self.url + "?model=" + os.getenv("MODEL"), additional_headers=headers) as ws:
                            await self.startInteraction(ws)
                return response
            
        except Exception as e:
            print(f"Error in calculate_product_sum: {str(e)}")
            return {
                "type": "conversation.item.create",
                "content": json.dumps({
                    "error": str(e)
                })
            }

    def test_tone(self, output_device_id):
        """Plays a short test tone on the selected output device."""
        duration = 0.5  # seconds
        frequency = 440.0  # A4 note

        try:
            # Query the selected device for its maximum output channels
            device_info = sd.query_devices(output_device_id)
            max_output_channels = device_info['max_output_channels']

            if max_output_channels < 1:
                raise ValueError(f"Selected device does not support audio output.")

            t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
            tone = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate a sine wave

            # Play the test tone
            print(f"Playing test tone on device '{device_info['name']}'...")
            sd.play(tone, samplerate=self.sample_rate, device=output_device_id)  # No 'channels'
            sd.wait()  # Wait until the tone finishes playing
            print("Test tone completed.")
            return True
        except Exception as e:
            print(f"Failed to play test tone: {str(e)}")
            return False

    def select_audio_device(self, input_output):
        """Loops through available devices until a working one is found and confirmed by the user."""
        devices = sd.query_devices()
        print(f"Available {input_output} audio devices:")
        for i, device in enumerate(devices):
            if (input_output == 'input' and device['max_input_channels'] > 0) or \
               (input_output == 'output' and device['max_output_channels'] > 0):
                print(f"{i}: {device['name']}")

        if input_output == 'input':
            # For input, ask the user to select a device
            while True:
                try:
                    device_id = int(input("Enter the ID of the input device to use: ").strip())
                    return device_id
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
        else:
            try:
                # For output, iterate through devices and test each one
                for i, device in enumerate(devices):
                    if device['max_output_channels'] > 0:
                        print(f"Testing device {i}: {device['name']}")
                        if self.test_tone(i):  # Now calling the method of the class
                            response = input("Could you hear the test tone? (y/n): ").strip().lower()
                            if response == 'y':
                                print(f"Selected output device: {device['name']} (ID: {i})")
                                return i
                            else:
                                print("Moving to the next device...")
                print("No suitable output device found.")
            except sd.PortAudioError as e:
                print(f"Error testing device {i}: {str(e)}")
                print("Moving to the next device...")
                
            # If no suitable device is found, ask the user to manually select one
            print("Please select one manually.")
            while True:
                try:
                    device_id = int(input("Enter the ID of the output device to use: ").strip())
                    if devices[device_id]['max_output_channels'] > 0:
                        print(f"Selected output device: {devices[device_id]['name']} (ID: {device_id})")
                        return device_id
                    else:
                        print("Invalid device ID. Please enter a valid output device ID.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter a valid integer.")

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
                    1"voice": "ash",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold":0.5,
                        "prefix_padding_ms":300,
                        "silence_duration_ms":1000                       
                    },
                    "tools": [{
                        "type": "function",
                        "name": "sub_total_order_not_final",
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
                                                "quantity": "Number of units of the product"
                                            },
                                            "value": {
                                                "type": "number",
                                                "value": "Value of the product unit without $ sign"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "name for the product EXACTLY as it appears on the menu"
                                            },
                                            "special instructions": {
                                                "type": "string",
                                                "description": "special instructions for the product paying attention to alergies, elements that need to be removed or added that are not extras (extras need to be on a different item)"
                                            }
                                        }
                                    }
                                },
                            },
                            "required": []
                        }
                    }, {
                        "type": "function",
                        "name": "place_order",
                        "description": "Send the order to the kitchen for preparation and payment",
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
                                                "quantity": "Number of units of the product"
                                            },
                                            "value": {
                                                "type": "number",
                                                "value": "Value of the product unit without $ sign"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "name for the product EXACTLY as it appears on the menu"
                                            },
                                            "special instructions": {
                                                "type": "string",
                                                "description": "special instructions for the product paying attention to alergies, elements that need to be removed or added that are not extras (extras need to be on a different item)"
                                            }
                                        }
                                    }
                                },
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
            self.extra_headers = True
        except TypeError:
            async with websockets.connect(self.url + "?model=" + os.getenv("MODEL"), additional_headers=headers) as ws:
                await self.startInteraction(ws)
            self.extra_headers = False
                
    async def send_audio(self, ws):
        print("Start speaking to the assistant (Press Ctrl+C to exit).")
        loop = asyncio.get_event_loop()
        required_samples = int(self.sample_rate * self.chunk_duration)

        def callback(indata, frames, time, status):
            if not self.should_record:
                return
            if status:
                print(status, flush=True)
            audio_bytes = indata.tobytes()
            self.recorded_audio.extend(audio_bytes)

            if len(self.recorded_audio) >= required_samples * 2:
                audio_chunk = self.recorded_audio[:required_samples * 2]
                self.recorded_audio = self.recorded_audio[required_samples * 2:]

                encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                message_event = {
                    "type": "input_audio_buffer.append",
                    "audio": encoded_audio
                }

                asyncio.run_coroutine_threadsafe(ws.send(json.dumps(message_event)), loop)

        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype=self.audio_format, callback=callback, blocksize=int(self.sample_rate * self.chunk_duration), device=self.input_device_id):
            while self.should_record:
                await asyncio.sleep(1)
                if len(self.recorded_audio) >= required_samples * 2:
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.commit"
                    }))

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
                    response = await self.handle_function_call(event, ws)
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
                elif event["type"] == "response.done":
                    print(f"Response: {event['response']}")
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