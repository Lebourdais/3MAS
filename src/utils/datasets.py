import torch
import torch.nn.functional as F
import pandas
import torchaudio
import torchaudio.transforms as T
import os
from datasets import load_dataset,Audio
import numpy as np
import soundata
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
torchaudio.set_audio_backend("soundfile")
import tqdm

class GTZAN(Dataset):
    
    def __init__(self,csv_file,root_dir,mode="train",prop=0.8,seed=42,target_samplerate=16000,overwrite=False):
        super().__init__()
        
        self.samplerate = 22050
        self.target_samplerate = target_samplerate
        
        if not os.path.exists(f"{root_dir}/train.csv") or overwrite:
            data_full = pandas.read_csv(f"{root_dir}{csv_file}")
            
            data_full = data_full[~data_full['filename'].str.contains("jazz.00054")]
            
            data_train = data_full.sample(frac=prop,random_state=seed)
            data_test = data_full.loc[data_full.index.difference(data_train.index)]
            
            data_valid = data_train.sample(frac=0.1,random_state=seed)
            data_train = data_train.loc[data_train.index.difference(data_valid.index)]
            data_train.to_csv(f"{root_dir}/train.csv")
            data_valid.to_csv(f"{root_dir}/valid.csv")
            data_test.to_csv(f"{root_dir}/test.csv")
        
        if mode == "train":
            self.data = pandas.read_csv(f"{root_dir}/train.csv")
        elif mode == "valid":
            self.data = pandas.read_csv(f"{root_dir}/valid.csv")
        else:
            self.data = pandas.read_csv(f"{root_dir}/test.csv")

        self.genre_list = os.listdir(f"{root_dir}/genres_original/")
        self.genre_list.sort()
        
        self.label_encode = {}
        for ii,genre in enumerate(self.genre_list):
            self.label_encode[genre] = ii
            
        
        num_win = 10
        length = self.data.iloc[0]["length"]
        
        self.windows = [{"idx_start":t*length, "idx_stop":(t+1)*length} for t in range(num_win)]
        self.root_dir = root_dir+"genres_original/"
        self.seg_length = length
        self.resampler = T.Resample(self.samplerate, self.target_samplerate)
        
    def __getitem__(self,idx):
        seg_info = self.data.iloc[idx]
        label = seg_info["label"]
        name_split = seg_info["filename"].split(".")
        win_idx = int(name_split[2])
        
        uri = f"{self.root_dir}{label}/{label}.{name_split[1]}.wav"
        audio,_ = torchaudio.load(uri,
                                frame_offset=self.windows[win_idx]["idx_start"],
                                num_frames=self.seg_length)
        audio = self.resampler(audio)
        
        return audio, self.label_encode[label]
        
    def __len__(self):
        return len(self.data)
    
    def get_classes(self):
        return self.genre_list

class CommonVoiceDataset(Dataset):
    def __init__(self,mode="train",samplerate=16000,length=2.0,label="gender"):
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()
        super().__init__()
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split=mode)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=samplerate))
        dataset = dataset.remove_columns(["sentence","up_votes","down_votes","age","accent","locale","segment"])
        print("Cleaning...",end="")
        dataset = dataset.filter(lambda example: example,input_columns=label)
        print("..",end="")
        dataset = dataset.map(self.to_index,input_columns=label)
        dataset.set_format(type='torch', columns=['audio', 'gender'])
        print("..OK")
        self.samplerate = samplerate
        self.set = dataset
        self.length=length
        self.sample_length=int(length*samplerate)
        
        
    def to_index(self,example):
        example = 1 if example=='female' else 0
        dicti={"gender":example}
        return dicti
    
    def __getitem__(self,idx):
        sample = self.set.__getitem__(idx)
        label = sample['gender']
        audio = sample['audio']['array']
        
        
        audio = F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0)
        return audio,label
    def __len__(self):
        return len(self.set)

class UrbanSound8k(Dataset):
    def __init__(self,mode="train",samplerate=16000,length=4.0):
        
        super().__init__()
        dataset = soundata.initialize('urbansound8k')
        splits = {"train":[1,2,3,4,5,6],
                 "validation":[7,8],
                 "test":[9,10]}
        resize = T.Resample(44100, samplerate)
        #dataset.download()  # download the dataset
        #dataset.validate()  # validate that all the expected files are there
        self.map = {l:ii for ii,l in enumerate(['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])}
        self.set = dataset
        self.ids = dataset.load_clips()
        self.sample_length = int(samplerate*length)
        
        self.examples = []
        for e in tqdm.tqdm(self.ids):
            clip = self.set.clip(e)
            if clip.fold in splits[mode]:
                audio,sr = clip.audio
                
                self.examples.append({'audio':resize(torch.from_numpy(audio).unsqueeze(0)),'label':torch.tensor(self.map[clip.class_label])})
            
    def __getitem__(self,idx):
        sample = self.examples.__getitem__(idx)
        
        label = sample['label']
        audio = sample['audio']
        
        audio = F.pad(audio[:,:self.sample_length],(0,max(0,self.sample_length-audio.shape[1])), "constant", 0)
        #print(audio.shape,label.shape)
        return audio,label
    def __len__(self):
        return len(self.examples)
    
class TimitDataset(Dataset):

    def __init__(self,mode="train",samplerate=16000,length=0.5,dataset_utterance=None):
        from datasets.utils.logging import disable_progress_bar
        
        disable_progress_bar()
        super().__init__()
        dataset = load_dataset('timit_asr', data_dir='/lium/raid01_b/mlebour/data/timit/TIMIT/',split=mode)
        #dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split=mode)
        self.mapping_phone = {
                'iy':0,
                'ih':1,
                'ix':1,
                'eh':2,
                'ae':3,
                'ax':4,
                'ah':4,
                'ax-h':4,
                'uw':5,
                'ux':5,
                'uh':6,
                'ao':7,
                'aa':7,
                'ey':8,
                'ay':9,
                'oy':10,
                'aw':11,
                'ow':12,
                'er':13,
                'axr':13,
                'l':14,
                'el':14,
                'r':15,
                'w':16,
                'y':17,
                'm':18,
                'em':18,
                'n':19,
                'en':19,
                'nx':19,
                'ng':20,
                'eng':20,
                'v':21,
                'f':22,
                'dh':23,
                'th':24,
                'z':25,
                's':26,
                'zh':27,
                'sh':27,
                'jh':28,
                'ch':29,
                'b':30,
                'p':31,
                'd':32,
                'dz':33,
                't':34,
                'g':35,
                'k':36,
                'hh':37,
                'hv':37,
        }
        
        dataset = dataset.cast_column("audio", Audio(sampling_rate=samplerate))
        
        dataset = dataset.remove_columns(["text","word_detail","dialect_region","sentence_type","speaker_id","id"])
        print("Cleaning...",end="")
        print("..",end="")
        
        phonemes = []
        for detail in dataset['phonetic_detail']:
            for p in detail['utterance']:
                if p not in phonemes:
                    phonemes.append(p)

        self.list_phone = ['iy','ih','eh','ae','ax','uw','uh','ao','ey','ay','oy','aw','ow','er','l','r','w','y','m','n','ng','v','f','dh','th','z','s','zh','jh','ch','b','p','d','dz','t','g','k','hh','sil']
        
        
        dataset = dataset.map(self.to_index,input_columns='phonetic_detail')
        
        dataset.set_format(type='torch', columns=['audio', 'phonetic_detail','start','stop'])
        
        print("..OK")
        self.samplerate = samplerate
        self.length=length
        self.sample_length=int(length*samplerate)
        print("Dataset preparation")
        self.phone = []
        for e in dataset:
            
            for label,start,stop in zip(e['phonetic_detail'],e['start'],e['stop']):
#                 self.phone.append(
#                     {'audio':np.resize(e['audio']['array'][start:stop],int(length*samplerate)),
#                      'label':label
#                     }
#                 )
                audio = e['audio']['array'][start:stop]
                self.phone.append(
                    {'audio':F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0),
                     'label':label
                    }
                )
        
        
        
    def to_index(self,example):
        dicti = {"phonetic_detail":[]}
        
        dicti = {'start':example['start'],'stop':example['stop'],'phonetic_detail': [self.mapping_phone[x] if x in self.mapping_phone else 38 for x in example['utterance']]}
        return dicti
    
    def __getitem__(self,idx):
        sample = self.phone.__getitem__(idx)
        label = sample['label']
        audio = sample['audio']
        
        #audio = F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0)
        return audio,label
    def __len__(self):
        return len(self.phone)
