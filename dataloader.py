import json
import os
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL
from transformers import BertTokenizer
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
    ele = batch[0]
    if isinstance(ele, dict):
        return {key: collate_fn([d[key] for d in batch]) for key in ele}
    elif isinstance(ele, (tuple, list)):
        return [collate_fn(x) for x in zip(*batch)]
    else:
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if any(c < m for c, m in zip(b.shape, max_shape)):
                        b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                        if b.dim() == 1:
                            b2[:b.shape[0]] = b
                        elif b.dim() == 2:
                            b2[:b.shape[0], :b.shape[1]] = b
                        elif b.dim() == 3:
                            b2[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                        else:
                            raise NotImplementedError
                        b = b2
                    batch2.append(b)
                batch = batch2
        return default_collate(batch)


class AVCapDataset(Dataset):
    def __init__(self, dataset_json_file, base_path, num_frames, audio_conf):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.num_frames = num_frames
        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')

        self.audio_base_path = os.path.join(base_path, 'waveforms') + '/'
        self.video_base_path = os.path.join(base_path, 'frames') + '/'

        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            self.data = json.load(fp)

        self.data = self.pro_data(self.data)

        self.num_samples = self.data.shape[0]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_text_len = 40

        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)

        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)

        self.target_length = self.audio_conf.get('target_length')

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        self.total_frame = self.audio_conf.get('total_frame', 20)

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [self.audio_base_path + data_json[i]['audio_path'], 
                            data_json[i]['labels'], 
                            data_json[i]['video_path'],
                            data_json[i]['caption'],
                            data_json[i]['torch_id']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['audio_path'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_path'] = np_data[2]
        datum['caption'] = np_data[3]
        datum['torch_id'] = int(np_data[4])
        return datum

    def get_image(self, filename, mix_lambda=1):
        img = Image.open(filename)
        image_tensor = self.preprocess(img)
        return image_tensor


    def _wav2fbank(self, filename):

        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_img(self, video_path):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 19)

        while os.path.exists(self.video_base_path + 'frame_' + str(frame_idx) + '/' + video_path) == False and frame_idx >= 1:
            print('frame {:s} {:d} does not exist'.format(video_path, frame_idx))
            frame_idx -= 1
        out_path = self.video_base_path + 'frame_' + str(frame_idx) + '/' + video_path
        #print(out_path)
        return out_path

    def get_video(self, video_path):
        if self.mode == 'eval':
            frame_idx = int((self.total_frame-self.num_frames)/2)
        else:
            frame_idx = random.randint(0, self.total_frame-self.num_frames)

        imgs = []
        for i in range(frame_idx, frame_idx+self.num_frames):
            while os.path.exists(self.video_base_path + 'frame_' + str(i) + '/' + video_path) == False and i >= 1:
                print('frame {:s} {:d} does not exist'.format(video_path, i))
                i -= 1
            out_path = self.video_base_path + 'frame_' + str(i) + '/' + video_path
            img = Image.open(out_path)
            image_tensor = self.preprocess(img)
            imgs.append(image_tensor.unsqueeze(1))
        video_tensor = torch.cat(imgs, dim=1)
        return video_tensor


    def get_fbank(self, fbank):
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        return fbank

    def get_text(self, caption):
        target_encoding = self.tokenizer(caption, padding='do_not_pad',
                        add_special_tokens=False,
                        truncation=True, max_length=self.max_text_len)
        need_predict =  [1] * len(target_encoding['input_ids'])
        payload = target_encoding['input_ids']
        if len(payload) > self.max_text_len:
            payload = payload[-(self.max_text_len - 2):]
            need_predict = need_predict[-(self.max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        return input_ids, need_predict
    # def get_attention_mask(self, N_a = , N_v = , N_t =):

    #     return attn_mask

    def __getitem__(self, index):

        datum = self.data[index]
        datum = self.decode_data(datum)

        fbank = self._wav2fbank(datum['audio_path'])
        fbank = self.get_fbank(fbank)
        if self.num_frames == 1:
            visual = self.get_image(self.randselect_img(datum['video_path']))
        elif self.num_frames > 1:
            visual = self.get_video(datum['video_path'])
        else:
            raise NotImplementedError
        input_ids, need_predict = self.get_text(datum['caption'])


        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        batch={'visual': visual, 'audio': fbank, 
               'caption_tokens': torch.tensor(input_ids), 
               'need_predict': torch.tensor(need_predict),
               'torch_id': torch.tensor(datum['torch_id'])}
        return batch

    def __len__(self):
        return self.num_samples