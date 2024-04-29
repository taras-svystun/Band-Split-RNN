from glob import glob
import torchaudio
from tqdm import tqdm
import os
from glob import glob
from time import perf_counter
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as TF
from tqdm import tqdm

print(torch.__version__, torchaudio.__version__)

# root = './musdb18hq_augmented/valid/'
os.chdir('../../datasets/musdb18hq_augmented/')
# root = './'
# os.listdir(root)


for folder in tqdm(sorted(glob(f'train/*'))):
    if 'ipynb' in folder:
        pass
    else:
        try:
            _, song = folder.split('/')
            print(song)


            pitches = [-3, -2, -1, 1, 2, 3]
            speed_ups = [.81, .93, 1.07, 1.23]
            shifts = [1, 2]

            

            # for idx, stem in os.listdir()
            
            
            augmented_folder = 'augmented_train'
        
            for pitch in pitches:
                pitch_dir_name = f'{augmented_folder}/{song}_{pitch}_pitch'
                # print(pitch_dir_name)
                if not os.path.isdir(pitch_dir_name):
                    os.makedirs(pitch_dir_name)

            for speedup in speed_ups:
                speed_dir_name = f'{augmented_folder}/{song}_{speedup}_speed'
                # print(speed_dir_name)
                if not os.path.isdir(speed_dir_name):
                    os.makedirs(speed_dir_name)

            for shift in shifts:
                shift_dir_name = f'{augmented_folder}/{song}_{shift}_shift'
                # print(shift_dir_name)
                if not os.path.isdir(shift_dir_name):
                    os.makedirs(shift_dir_name)


            # print(song)
            for stem in glob(f'{folder}/*'):
                y, sr = torchaudio.load(stem)
                stem_name = stem.split('/')[-1]


                for pitch in pitches:
                    pitch_filename = f'{augmented_folder}/{song}_{pitch}_pitch/{stem_name}'
                    if not os.path.isfile(pitch_filename):
                        torchaudio.save(
                                pitch_filename, 
                                F.pitch_shift(y, sr, pitch), 
                                sr
                        )
                    # print(pitch_filename)

                for speedup in speed_ups:
                    speed_filename = f'{augmented_folder}/{song}_{speedup}_speed/{stem_name}'
                    if not os.path.isfile(speed_filename):
                        torchaudio.save(
                                speed_filename,
                                T.Speed(44100, speedup)(y)[0],
                                sr
                        )
                    # print(speed_filename)

                for shift in shifts:
                    shift_filename = f'{augmented_folder}/{song}_{shift}_shift/{stem_name}'
                    if not os.path.isfile(shift_filename):
                        torchaudio.save(
                                shift_filename,
                                y[:, shift * sr:],
                                sr
                        )
                    # print(shift_filename)
        
            # os.makedirs(f'{root}/augmented_train/{folder}_+3_pitch')
            # for stem in os.listdir(root + song):
            #     track, sr = torchaudio.load(root + song + '/' + stem)
            #     new_track = F.pitch_shift(track, sr, 2)
            #     torchaudio.save(root + song + '_higher_pitch/' + stem, new_track, sr)
    
            # os.makedirs(root + song + '_lower_pitch')
            # for stem in os.listdir(root + song):
            #     track, sr = torchaudio.load(root + song + '/' + stem)
            #     new_track = F.pitch_shift(track, sr, -2)
            #     torchaudio.save(root + song + '_lower_pitch/' + stem, new_track, sr)
        
        
        
            # os.makedirs(root + song + '_higher_speed')
            # time_stretcher = T.Speed(44100, 1.1)
            # for stem in os.listdir(root + song):
            #     track, sr = torchaudio.load(root + song + '/' + stem)
            #     new_track, _ = time_stretcher(track)
            #     torchaudio.save(root + song + '_higher_speed/' + stem, new_track, sr)
            
            # os.makedirs(root + song + '_lower_speed')
            # time_stretcher = T.Speed(44100, .9)
            # for stem in os.listdir(root + song):
            #     track, sr = torchaudio.load(root + song + '/' + stem)
            #     new_track, _ = time_stretcher(track)
            #     torchaudio.save(root + song + '_lower_speed/' + stem, new_track, sr)
        
        
        
            # os.makedirs(root + song + '_shift_left')
            # for stem in os.listdir(root + song):
            #     track, sr = torchaudio.load(root + song + '/' + stem)
            #     torchaudio.save(root + song + '_shift_left/' + stem, track[:, 44100:], sr)
        
        
            # os.makedirs(root + song + '_shift_right')
            # for stem in os.listdir(root + song):
            #     track, sr = torchaudio.load(root + song + '/' + stem)
            #     new_track = TF.pad(track, (44100, 0))
            #     torchaudio.save(root + song + '_shift_right/' + stem, new_track, sr)

        except:
            print(f'Fuck this {song}')

