import pygame
import numpy as np

def play_tone_with_pygame():
    try:
        # Initialize the pygame mixer with 24 kHz, 16-bit, stereo (2 channels) audio
        pygame.mixer.init(frequency=24000, size=-16, channels=2)  # Stereo, 16-bit, 24kHz
        
        sample_rate = 24000  # 24 kHz sample rate
        duration = 0.5  # Duration in seconds
        frequency = 440.0  # Frequency in Hz (440Hz = A4 tone)

        # Generate a sine wave (mono audio)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)  # Scale to 16-bit PCM

        # Duplicate mono audio to stereo (2 channels)
        stereo_samples = np.stack((samples, samples), axis=-1)  # Duplicate across 2 channels

        # Create the pygame sound object directly from the stereo numpy array
        sound = pygame.sndarray.make_sound(stereo_samples)

        # Play the sound
        sound.play()

        # Wait for the sound to finish
        pygame.time.wait(int(duration * 1000))  # Wait for duration in milliseconds
        print("Test tone played successfully with pygame.")
    
    except Exception as e:
        print(f"Error playing tone with pygame: {e}")

# Call the function to play the tone
play_tone_with_pygame()
