import random
import typing as tp
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from PIL import Image
from tqdm import tqdm
from glob import glob


class SourceSeparationDataset(Dataset):
    """
    Dataset class for working with train/validation data from MUSDB18 dataset.
    """
    TARGETS: tp.Set[str] = {'vocals', 'bass', 'drums', 'other'}
    EXTENSIONS: tp.Set[str] = {'.wav', '.mp3'}

    def __init__(
            self,
            file_dir: str,
            txt_dir: str = None,
            txt_path: str = None,
            target: str = 'vocals',
            preload_dataset: bool = False,
            is_mono: bool = False,
            is_training: bool = True,
            sr: int = 44100,
            silent_prob: float = 0.1,
            mix_prob: float = 0.1,
            remixing_ratio: float = 0.5,
            mix_tgt_too: bool = False,
    ):
        self.file_dir = Path(file_dir)
        self.is_training = is_training
        self.target = target
        self.sr = sr

        if txt_path is None and txt_dir is not None:
            mode = 'train' if self.is_training else 'valid'
            self.txt_path = Path(txt_dir) / f"{target}_{mode}.txt"
        elif txt_path is not None and txt_dir is None:
            self.txt_path = Path(txt_path)
        else:
            raise ValueError("You need to specify either 'txt_path' or 'txt_dir'.")

        self.preload_dataset = preload_dataset
        self.is_mono = is_mono
        self.filelist = self.get_filelist()

        # augmentations
        self.silent_prob = silent_prob
        self.mix_prob = mix_prob
        self.remixing_ratio = remixing_ratio
        self.spectrogram = T.Spectrogram(2048, 2048, 512)
        self.mix_tgt_too = mix_tgt_too

    def get_filelist(self) -> tp.List[tp.Tuple[str, tp.Tuple[int, int]]]:
        filename2label = {}
        filelist = []
        i = 0
        for line in tqdm(open(self.txt_path, 'r').readlines()):
            file_name, start_idx, end_idx = line.split('\t')
            if file_name not in filename2label:
                filename2label[file_name] = i
                i += 1
            filepath_template = self.file_dir / "train" / f"{file_name}" / "{}.wav"
            if self.preload_dataset:
                mix_segment, tgt_segment = self.load_files(
                    str(filepath_template), (int(start_idx), int(end_idx))
                )
                filelist.append((mix_segment, tgt_segment))
            else:
                filelist.append(
                    (str(filepath_template), (int(start_idx), int(end_idx)))
                )
        return filelist

    def load_file(
            self,
            file_path: str,
            indices: tp.Tuple[int, int]
    ) -> torch.Tensor:
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."

        offset = indices[0]
        num_frames = indices[1] - indices[0]
        y, sr = torchaudio.load(
            file_path,
            frame_offset=offset,
            num_frames=num_frames,
            channels_first=True
        )
        assert sr == self.sr, f"Sampling rate should be equal {self.sr}, not {sr}."
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        return y

    def load_files(
            self, fp_template: str, indices: tp.Tuple[int, int],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        mix_segment = self.load_file(
            fp_template.format('mixture'), indices
        )
        tgt_segment = self.load_file(
            fp_template.format(self.target), indices
        )
        max_norm = max(
            mix_segment.abs().max(), tgt_segment.abs().max()
        )
        mix_segment /= max_norm
        tgt_segment /= max_norm
        return (
            mix_segment, tgt_segment
        )

    @staticmethod
    def imitate_silent_segments(
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return (
            mix_segment - tgt_segment,
            torch.zeros_like(tgt_segment)
        )

    def mix_segments(
            self,
            tgt_segment: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Creating new mixture and new target from target file and random multiple sources
        """
        # decide how many sources to mix
        if not self.mix_tgt_too:
            self.TARGETS.discard(self.target)
        n_sources = random.randrange(1, len(self.TARGETS) + 1)
        # decide which sources to mix
        targets_to_add = random.sample(
            self.TARGETS, n_sources
        )
        # create new mix segment
        mix_segment = tgt_segment.clone()
        for target in targets_to_add:
            # get random file to mix source from
            fp_template_to_add, indices_to_add = random.choice(self.filelist)
            segment_to_add = self.load_file(
                fp_template_to_add.format(target), indices_to_add
            )
            mix_segment += segment_to_add
            if target == self.target:
                tgt_segment += segment_to_add
        return (
            mix_segment, tgt_segment
        )
    
    def get_speech_filelist(self):
        return [filename for filename in glob(str(self.file_dir / '../LibriSpeech/**/*.wav'), recursive=True)]
    
    def add_noise(
        self, waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Copied as is from newer torch version.
        """

        if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
            raise ValueError("Input leading dimensions don't match.")

        L = waveform.size(-1)

        if L != noise.size(-1):
            raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

        # compute scale
        if lengths is not None:
            mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
                -1
            )  # (*, L) < (*, 1) = (*, L)
            masked_waveform = waveform * mask
            masked_noise = noise * mask
        else:
            masked_waveform = waveform
            masked_noise = noise

        energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
        energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
        original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
        scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

        # scale noise
        scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

        return waveform + scaled_noise  # (*, L)

    def remix(
        self,
        mix_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        vocal_lengths = 0
        vocal_samples = []
        if not hasattr(self, "speech_filelist"):
            self.speech_filelist = self.get_speech_filelist()
            
        
        while vocal_lengths < mix_segment.shape[1]:
            vocal_sample_filename = random.choice(self.speech_filelist)
            assert Path(vocal_sample_filename).is_file(), f"There is no such file - {vocal_sample_filename}."

            vocal_sample, sr = torchaudio.load(vocal_sample_filename, channels_first=True)
            assert sr == self.sr, f"Sampling rate should be equal {self.sr}, not {sr}."
            if self.is_mono:
                vocal_sample = torch.mean(vocal_sample, dim=0, keepdim=True)
            
            max_norm = vocal_sample.abs().max()
            vocal_sample /= max_norm

            vocal_samples.append(vocal_sample)
            vocal_lengths += vocal_sample.shape[1]



        vocals = torch.cat(vocal_samples, 1)[:, :mix_segment.shape[1]]
        # SNR = random.uniform(-5, 15)
        # SNRs = torch.tensor([SNR] * 2)

        SNRs = torch.tensor([random.uniform(-5, 15)] * 2)
        
        # torchaudio.save(f'../../datasets/tests/mix_{SNR:.1f}.wav', mix_segment, sr)
        
        mix_segment = self.add_noise(vocals, mix_segment, SNRs)
        max_norm = mix_segment.abs().max()
        mix_segment /= max_norm
    
        # torchaudio.save(f'../../datasets/tests/vocals_{SNR:.1f}.wav', vocals, sr)
        # torchaudio.save(f'../../datasets/tests/mix_with_vocals_{SNR:.1f}.wav', mix_segment, sr)
        
        

        spectrogram = self.spectrogram(mix_segment.mean(0))
        print(spectrogram.shape)
        spec_db = T.AmplitudeToDB(stype="magnitude", top_db=80)(spectrogram)
        print(spec_db.shape)
        # _ = plt.imshow(spec_db, aspect="auto", origin="lower")
        
        img = Image.fromarray(spec_db.numpy())
        img.save('../../datasets/tests/img.png')
        
        
        sys.exit()
        
        return (mix_segment, vocals)

    def augment(
            self,
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if self.is_training:
            # dropping target
            if random.random() < self.silent_prob:
                mix_segment, tgt_segment = self.imitate_silent_segments(
                    mix_segment, tgt_segment
                )
            # mixing with other sources
            if random.random() < self.mix_prob:
                mix_segment, tgt_segment = self.mix_segments(
                    tgt_segment
                )
            
            if random.random() < self.remixing_ratio:
                mix_segment, tgt_segment = self.remix(
                    mix_segment - tgt_segment
                )
        return mix_segment, tgt_segment

    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Each Tensor's output shape: [n_channels, frames_in_segment]
        """
        # load files
        if self.preload_dataset:
            mix_segment, tgt_segment = self.filelist[index]
        else:
            mix_segment, tgt_segment = self.load_files(*self.filelist[index])

        # augmentations related to mixing/dropping sources
        mix_segment, tgt_segment = self.augment(mix_segment, tgt_segment)

        return (
            mix_segment, tgt_segment
        )

    def __len__(self):
        return len(self.filelist)


class EvalSourceSeparationDataset(Dataset):
    """
    Dataset class for working with test data from MUSDB18 dataset.
    """
    EXTENSIONS: tp.Set[str] = {'.wav', '.mp3'}

    def __init__(
            self,
            mode: str,
            in_fp: str,
            out_fp: tp.Optional[str] = None,
            target: str = 'vocals',
            is_mono: bool = False,
            sr: int = 44100,
            win_size: float = 3,
            hop_size: float = 0.5,
            *args, **kwargs
    ):
        self.mode = mode

        # files params
        self.in_fp = Path(in_fp)
        self.out_fp = Path(out_fp) if out_fp is not None else None
        self.target = target

        # audio params
        self.is_mono = is_mono
        self.sr = sr
        self.win_size = int(win_size * sr)
        self.hop_size = int(hop_size * sr)
        self.pad_size = self.win_size - self.hop_size

        self.filelist = self.get_filelist()

    def get_test_filelist(self) -> tp.List[tp.Tuple[str, str]]:
        filelist = []
        test_dir = self.in_fp / self.mode

        for fp in test_dir.glob('*'):
            fp_template = str(fp / "{}.wav")
            fp_mix = fp_template.format('mixture')
            fp_tgt = fp_template.format(self.target)
            filelist.append((fp_mix, fp_tgt))

        return filelist

    def get_inference_filelist(self) -> tp.List[tp.Tuple[str, str]]:
        filelist = []
        if self.in_fp.is_file() and self.in_fp.suffix in self.EXTENSIONS:
            self.out_fp = self.out_fp / f"{self.in_fp.stem}_{self.target}.wav"
            filelist.append((self.in_fp, self.out_fp))
        elif self.in_fp.is_dir():
            for in_fp in self.in_fp.glob("*"):
                if in_fp.suffix in self.EXTENSIONS:
                    out_fp = self.out_fp / f"{in_fp.stem}_{self.target}.wav"
                    filelist.append((str(in_fp), str(out_fp)))
        else:
            raise ValueError(f"Can not open the path {self.in_fp}")
        return filelist

    def get_filelist(self) -> tp.List[tp.Tuple[str, str]]:
        if self.mode == 'test':
            filelist = self.get_test_filelist()
        elif self.mode == 'inference':
            filelist = self.get_inference_filelist()
        else:
            raise ValueError(f"Selected mode = '{self.mode}' is invalid")
        return filelist

    def load_file(self, file_path: str) -> torch.Tensor:
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."
        y, sr = torchaudio.load(
            file_path,
            channels_first=True
        )
        if sr != self.sr:
            y = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sr
            )(y)
        # setting to mono if necessary
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        elif y.shape[0] == 1:
            y = y.repeat(2, 1)
        return y

    def __getitem__(
            self, index: int
    ) -> tp.Tuple[torch.Tensor, tp.Union[torch.Tensor, str]]:

        fp_mix, fp_tgt = self.filelist[index]

        y_mix = self.load_file(fp_mix)

        if self.mode == 'test':
            return y_mix, self.load_file(fp_tgt)
        else:
            return y_mix, fp_tgt

    def __len__(self):
        return len(self.filelist)
