import os
from pydub import AudioSegment

def extract_audio_segments(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    segment_number = 448

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
            input_path = os.path.join(input_folder, filename)
            
            try:
                audio = AudioSegment.from_file(input_path)
                total_duration = len(audio)
                
                for start_time in range(0, total_duration, 30 * 1000): 
                    end_time = start_time + 60 * 1000
                    
                    segment = audio[start_time:end_time]
                    
                    if len(segment) > 0:
                        output_filename = f"music-{segment_number}.mp3"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        segment.export(output_path, format="mp3")
                        
                        segment_number += 1
                
                print(f"Processed {filename}: Extracted {segment_number - 1} segments")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    input_folder = "/mnt/c/Users/vmuog/OneDrive/Desktop/Audio/"  
    output_folder = "/mnt/c/Users/vmuog/OneDrive/Desktop/instrumentals/" 
    
    extract_audio_segments(input_folder, output_folder)

if __name__ == "__main__":
    main()
