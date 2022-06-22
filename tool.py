import os

def check_files():
    path_to_audios = './audiosWAV'
    path_to_krn = './krn'

    krn_list = len(os.listdir(path_to_krn))
    audio_list = len(os.listdir(path_to_audios))

    print(f"krn_list size {krn_list}, audio {audio_list}")

    for file in os.listdir(path_to_krn):
        krn_name = file.split('.')[0]
        aux = False
        for audio_file in os.listdir(path_to_audios):
            audio_name = audio_file.split('.')[0]

            if krn_name == audio_name:
                aux = True
                break
            
        if not aux:
            print(f" ============= No he encontrado el audio {file}")


    

if __name__ == "__main__":
    check_files()