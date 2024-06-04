import pandas
import torch
import numpy
import torchaudio
from src.utils import SpecFeat
from nmf_pretrain import snmf_pretrain
import argparse


def get_ref(rttm_file):
    ref = pandas.read_csv(rttm_file,
                  names=['type',"show","chan","start",'duration','na1',"na2","cluster","na3","na4"],sep='\s',engine="python")

    return ref


def build_train_matrix(train_list,
                       audio_path,
                       rttm_path,
                       N_speech=50,
                       N_music=50,
                       N_noise=50,
                       spec_kw={"win_length":1024, "hop_length":320,"n_fft":1024},
                       seg_limits_sec=(0.5,3.0),
                       seed=1234):
    shows = [s.strip('\n') for s in open(train_list,'r').readlines()]
    full_df = pandas.DataFrame(columns=['type',"show","chan","start",'duration','na1',"na2","cluster","na3","na4"])
    for show in shows:
        full_df = pandas.concat([full_df,get_ref(f"{rttm_path}{show}.rttm")])

    # log spectrogram extractor
    spec = SpecFeat(**spec_kw)
    # total number of segments
    tot = N_speech+N_noise+N_music
    fs=16e3
    min_duration,max_duration = seg_limits_sec
    #filter shorter segments
    full_df = full_df[full_df["duration"] >= min_duration]
    
    # Shuffle evrything
    full_df=full_df.sample(frac=1,random_state=numpy.random.default_rng(seed=seed))
    all_feat = []
    max_duration_samples=int(max_duration*fs)
    for i,seg in full_df.iterrows():
        if len(all_feat) < tot:
            if seg["cluster"] == "sp" and N_speech > 0:
                audio,_ = torchaudio.load(f"{audio_path}{seg['show']}.wav",frame_offset=int(seg["start"]*fs),num_frames=int(seg["duration"]*fs))
                N_speech-=1
            elif  seg["cluster"] == "no" and N_noise > 0:
                audio,_ = torchaudio.load(f"{audio_path}{seg['show']}.wav",frame_offset=int(seg["start"]*fs),num_frames=int(seg["duration"]*fs))
                N_noise-=1
            elif  seg["cluster"] == "mu" and N_music > 0:
                audio,_ = torchaudio.load(f"{audio_path}{seg['show']}.wav",frame_offset=int(seg["start"]*fs),num_frames=int(seg["duration"]*fs))
                N_music-=1
            else:
                audio=None
            if audio is not None:
                if audio.shape[-1] > max_duration_samples:
                    audio=audio[:,:max_duration_samples]
                feat=spec(audio)
                all_feat.append(feat)
        else:
            break
    # training matrix composed of concatenation of segments spectrograms
    X = torch.cat(all_feat, dim=-1)
    
    return X


def build_valid_set(test_list,
                    audio_path,
                    rttm_path,
                    num_seg=20,
                    label="sp",
                    spec_kw={"win_length":400, "hop_length":320,"n_fft":512},
                    seed=1234,):
    shows = [s.strip('\n') for s in open(test_list,'r').readlines()]
    full_df = pandas.DataFrame(columns=['type',"show","chan","start",'duration','na1',"na2","cluster","na3","na4"])
    for show in shows:
        full_df = pandas.concat([full_df,get_ref(f"{rttm_path}{show}.rttm")])

    # log spectrogram extractor
    spec = SpecFeat(**spec_kw)
    # total number of segments
    fs=16e3
    #filter shorter segments
    full_df = full_df[full_df["cluster"] == label].sample(n=num_seg,random_state=numpy.random.default_rng(seed=seed))
    
    all_feat=[]
    for i,seg in full_df.iterrows():
        audio,_ = torchaudio.load(f"{audio_path}{seg['show']}.wav",frame_offset=int(seg["start"]*fs),num_frames=int(seg["duration"]*fs))
        feat=spec(audio)
        all_feat.append(feat)
        
    return all_feat


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_music",default=50,help="Number of segments containing music in the training matrix",type=int)
    parser.add_argument("--n_speech",default=50,help="Number of segments containing speechin the training matrix",type=int)
    parser.add_argument("--n_noise",default=50,help="Number of segments containing noise in the training matrix",type=int)
    parser.add_argument("--beta",default=2,help="Divergence type (0: IS div, 1: KL div, 2: euclidean distance)",type=int)
    parser.add_argument("--mu",default=1,help="Sparsity factor",type=int)
    parser.add_argument("--min_duration",default=0.5,help="Minimum duration of the segments",type=float)
    parser.add_argument("--max_duration",default=3.0,help="Maximum duration of the segments",type=float)
    parser.add_argument("--nmf_components",default=256,help="Number of components used for factorization",type=int)
    parser.add_argument("--win_length",default=1024,help="Length of spectrogram window",type=int)
    parser.add_argument("--n_fft",default=1024,help="Number of frquencies in the spectrogram",type=int)
    parser.add_argument("--out_path",default="./",help="Where to save the learnt dictionary",type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1234)
    data_path="/gpfswork/rech/wcp/commun/datasets/Albayzin_10_16_pyannote/"
    dataset="AragonRadio"
    train_list=f"{data_path}lists/{dataset}/train.txt"
    dev_list=f"{data_path}lists/{dataset}/dev1.txt"
    audio_path=f"{data_path}audio/{dataset}/"
    rttm_path=f"{data_path}rttms_segmentation/{dataset}/"

    fname = f"W_win_{args.win_length}_nfft_{args.n_fft}_{args.nmf_components}_beta_{args.beta}_mu_{args.mu}_{dataset}_sp_{args.n_speech}_mus_{args.n_music}_no_{args.n_noise}_max_seg_{int(args.max_duration)}.pt"

    X = build_train_matrix(train_list=train_list,
                           audio_path=audio_path,
                           rttm_path=rttm_path,
                           N_speech=args.n_speech,
                           N_music=args.n_music,
                           N_noise=args.n_noise,
                           spec_kw={"win_length":args.win_length, "hop_length":320,"n_fft":args.n_fft},
                           seg_limits_sec=(args.min_duration,args.max_duration))

    W,H,snmf=snmf_pretrain(X.squeeze(),
                            n_components=args.nmf_components,
                            device=device,
                            n_iter=200,
                            mu_fact=args.mu,
                            beta=args.beta,
                            save_measure=False,
                            save_dictionary=f"{args.out_path}{fname}")
