import os
import librosa
from midiutil import MIDIFile


def convert_mp3_to_midi(mp3_path, midi_path):

    y, sr = librosa.load(mp3_path)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    for time_index in range(pitches.shape[1]):
        index = magnitudes[:, time_index].argmax()
        pitch = pitches[index, time_index]
        if pitch > 0:
            midi_note = int(librosa.hz_to_midi(pitch))
            midi.addNote(0, 0, midi_note, time_index * (1.0 / sr), 1, 100)
 
    with open(midi_path, "wb") as output_file:
        midi.writeFile(output_file)



def process_files_mp3(source_folder,output_folder):
    
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(source_folder) if f.endswith('.mp3')]

    for file in files:
        input_file_path = os.path.join(source_folder, file)
        output_file_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".mid")
        
        convert_mp3_to_midi(input_file_path, output_file_path)
        print(f"Converted and saved: {output_file_path}")
        
    
def main():
    source_folder = "/mnt/c/Users/vmuog/OneDrive/Desktop/instrumentals/" 
    output_folder = "dataset/instrumentals"
    process_files_mp3(source_folder,output_folder)

if __name__ == "__main__":
    main()