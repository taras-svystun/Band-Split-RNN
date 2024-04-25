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
root = '../../datasets/musdb18hq_augmented/'
# os.listdir(root)


for folder in tqdm(sorted(glob(f'{root}train/*'))):
    if 'ipynb' in folder:
        pass
    else:
        try:
            
            # for idx, stem in os.listdir()
            
            
            print(folder)
        
        
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
            pass

