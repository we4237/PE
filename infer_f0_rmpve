from typing import Union
from matplotlib import pyplot as plt
import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
import torchaudio
from F0Predictor import F0Predictor
import parselmouth
from fcpe.model import FCPEInfer
from rmvpe.inference import RMVPE

class FCPEF0Predictor(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sampling_rate=44100,
                 threshold=0.05):
        self.fcpe = FCPEInfer(model_path="pretrain/fcpe.pt", device=device, dtype=dtype)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "fcpe"

    def repeat_expand(
            self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[
                0]).cpu().numpy(), vuv_vector.cpu().numpy()

        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

        return f0, vuv_vector.cpu().numpy()

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0,:,0]
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process(x, self.sampling_rate, f0, p_len)[0]

    def compute_f0_uv(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0,:,0]
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process(x, self.sampling_rate, f0, p_len)


class PMF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "pm"
    
    def interpolate_f0(self,f0):
        '''
        对F0进行插值处理
        '''
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
    
        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0,vuv_vector
    

    def compute_f0(self,wav,p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0,uv = self.interpolate_f0(f0)
        return f0

    def compute_f0_uv(self,wav,p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0,uv = self.interpolate_f0(f0)
        return f0,uv


class RMVPEF0Predictor(F0Predictor):
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100, dtype=torch.float32, device=None,sampling_rate=44100,threshold=0.05):
        self.rmvpe = RMVPE(model_path="pretrain/rmvpe.pt",dtype=dtype,device=device)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "rmvpe"

    def repeat_expand(
        self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)
        
        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate
        
        vuv_vector = F.interpolate(vuv_vector[None,None,:],size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(),vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy() ,vuv_vector.cpu().numpy()
    
        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        #vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        return f0,vuv_vector.cpu().numpy()

    def compute_f0(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)[0]
    
    def compute_f0_uv(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)
    
    def compute_f0_uv_mel(self,mel,p_len=None):
        f0 = self.rmvpe.infer_from_mel(mel,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(mel,self.sampling_rate,f0,p_len)

def read_npy_files_in_directory(directory_path):
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('_raw.npy')]
    npy_data_list = []
    
    for npy_file in npy_files:
        npy_file_path = os.path.join(directory_path, npy_file)
        # npy_data = np.load(npy_file_path,allow_pickle=True)
        npy_data_list.append(npy_file_path)
    
    return npy_data_list
   
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampling_rate = 24000
    hop_length = 128

    out_folder = "outf0_rmvpe"
    wav_folder = "/mnt/users/chenmuyin/diffsinger/vocoder/NSF-HiFiGAN-cmy/dataset/data_24k"
    npy_list = read_npy_files_in_directory(wav_folder)
    f0_predictor_rm = RMVPEF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,dtype=torch.float32 ,device=device,threshold=0.05,f0_min=80,f0_max=750)


    for npy in npy_list:
        out_name = os.path.basename(npy).split('.')[0]
        out_name = out_name + '.pt'
        out_path = os.path.join(out_folder,out_name)
        mel = np.load(npy,allow_pickle=True)[1]
        # mel = torch.from_numpy(mel).to(device)
        f0_rm, uv_rm = f0_predictor_rm.compute_f0_uv_mel(mel)
        
        # 将张量保存为.pt文件
        data_to_save = {
            'mel': mel,
            'f0_rm': f0_rm
        }
        torch.save(data_to_save, out_path)


