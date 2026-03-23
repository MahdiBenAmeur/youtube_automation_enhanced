from pydub import AudioSegment

from pydub import AudioSegment

def crop_mp3(input_path, output_path, start_ms, end_ms):
    audio = AudioSegment.from_mp3(input_path)
    
    cropped = audio[start_ms:end_ms]
    cropped.export(output_path, format="mp3")

# example (10s → 30s = 10000 → 30000 ms)
# example
path = r"C:\disque d\ai_stuff\projects\enhanced_youtube_automation\src\builders\ai_car_driving_builder\assets\Sound Effect Whoosh Sound (No Copyright).mp3"
crop_mp3(path, "output.mp3", 0, 900)