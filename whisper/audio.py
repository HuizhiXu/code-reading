import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    mono waveform是指左右声道声音均匀混合成一个单声道的音频。stereo audio是指立体声，声音来自左右两个声道
    声音中的 sample rate 是指在数字音频中每秒钟采集或播放的样本数。它表示了音频的采样率或取样率。采样率决定了数字音频的质量和频谱范围。

    在数字音频中，声音被转换为数字形式，通过在一段时间内对声音进行多次采样来捕捉声音的细节。采样率表示每秒钟采样的次数，以赫兹（Hz）为单位。
    常见的采样率包括 44.1 kHz（CD质量）、48 kHz（广播和视频）、96 kHz（高保真音频）等。
    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout # execute a command and captire ts output
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    #字节缓存转为int16格式，flatten将其变为一维数组，然后变为float32类型，除以32768是归一化，将range限制在-1到1.
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array): # 判断array是不是tensor 
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            ) #dim表示从第几维挑选数据，index表示从前面说的维度中哪个位置挑选

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis) # take返回前indices个元素

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    STFT是指Short-Time Fourier Transform 
    Mel滤波器组是一组三角滤波器，用于将短时傅里叶变换（STFT）投影到Mel频谱图中。Mel刻度是一种感知刻度，近似人类听觉系统对不同频率的敏感性。
    Mel滤波器组通过将一组重叠的三角滤波器应用于频谱来模拟人类听觉系统的频率分辨率。滤波器组中的每个三角滤波器都以特定的Mel频率为中心，覆盖一定范围的频率。
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    log-Mel频谱是音频信号处理任务中常用的特征表示方法
    得到的log-Mel频谱是一个二维矩阵，其中x轴表示时间（对应帧），y轴表示Mel频带。矩阵中的每个元素表示特定Mel频带在特定时间帧的能量的对数。
    计算log-Mel频谱可以捕捉音频信号的重要频率成分，同时降低特征表示的维度。这种紧凑的表示专注于感知相关的信息。
    Compute the log-Mel spectrogram of
    
    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
