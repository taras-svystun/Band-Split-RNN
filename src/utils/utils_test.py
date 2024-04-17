import numpy as np
import torch
from museval.metrics import bss_eval
import typing as tp
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from sys import exit


def compute_uSDR(
        y_hat: np.ndarray,
        y_tgt: np.ndarray,
        delta: float = 1e-7,
) -> float:
    """
    Computes SDR metric as in https://arxiv.org/pdf/2108.13559.pdf.
    Taken and slightly rewritten from
    https://github.com/AIcrowd/music-demixing-challenge-starter-kit/blob/master/evaluator/music_demixing.py
    """
    # compute SDR for one song
    num = np.sum(np.square(y_tgt), axis=(1, 2))
    den = np.sum(np.square(y_tgt - y_hat), axis=(1, 2))
    num += delta
    den += delta
    print(num, den)
    return 10 * np.log10(num / den)


def compute_SDRs(
        y_hat: torch.Tensor, y_tgt: torch.Tensor
) -> tp.Tuple[float, float]:
    """
    Computes cSDR and uSDR as defined in paper
    """
    print(y_hat.shape)
    print(y_tgt.shape)
    print('=' * 40)
    
    si_sdr = ScaleInvariantSignalDistortionRatio()
    siSDR = si_sdr(y_hat, y_tgt).item()
    sdr = SignalDistortionRatio()
    SDR = sdr(y_hat, y_tgt).item()
    # print(f'{siSDR=}\t{SDR=}')
    print(f'siSDR={si_sdr(y_hat.unsqueeze(0), y_tgt.unsqueeze(0)).item()}\tSDR={sdr(y_hat.unsqueeze(0), y_tgt.unsqueeze(0)).item()}')
    
    
    y_hat = y_hat.T.unsqueeze(0).numpy()
    y_tgt = y_tgt.T.unsqueeze(0).numpy()
    print(y_hat.shape)
    print(y_tgt.shape)
    print('=' * 40)
    
    
    # bss_eval way
    cSDR, *_ = bss_eval(
        y_tgt,
        y_hat
    )
    cSDR = np.nanmedian(cSDR)

    # as in music demixing challenge
    uSDR = compute_uSDR(
        y_hat,
        y_tgt
    )
    print(cSDR, uSDR, siSDR, SDR)
    exit()
    return cSDR, uSDR, siSDR