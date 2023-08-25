import torch
import librosa
import numpy as np

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2
    
class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        fft_size=512,
        hop_size=128,
        win_length=512,
        window="hann",
        num_mels=80,
        fmin=30,
        fmax=12000,
        eps=1e-6,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = window
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps
        # get mel basis
        fmin = 0 if fmin == -1 else fmin
        fmax = sample_rate / 2 if fmax == -1 else fmax
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.numpy().T

        # Load waveform and compute spectrogram
        x_stft = librosa.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            pad_mode="constant",
        )
        spc = np.abs(x_stft)

        # Compute mel spectrogram
        mel = self.mel_basis @ spc

        # Apply log transformation
        mel = np.log10(np.maximum(self.eps, mel)) # (n_mel_bins, T)

        # Pad waveform
        l_pad, r_pad = librosa_pad_lr(waveform, self.fft_size, self.hop_size, 1)
        waveform = np.pad(waveform, (l_pad, r_pad), mode='constant', constant_values=0.0)
        waveform = waveform[:mel.shape[1] * self.hop_size] # (n_mel_bins,T) 
        # mel = mel.T # (T,n_mel_bins) 

        return torch.from_numpy(waveform),torch.from_numpy(mel)
    
def wav2mel(wav):
    fft_size=512
    hop_size=128
    win_length=512
    window="hann"
    num_mels=80
    fmin=30
    fmax=12000
    eps=1e-6
    sample_rate=24000

    audio_pipeline = AudioPipeline(sample_rate=sample_rate,
                                    fft_size=fft_size,
                                    num_mels=num_mels,
                                    win_length=win_length,
                                    hop_size=hop_size)
    wav_ = torch.from_numpy(wav)
    _,mel = audio_pipeline(wav_) # (n_mel_bins, T)

    return mel.unsqueeze(0)