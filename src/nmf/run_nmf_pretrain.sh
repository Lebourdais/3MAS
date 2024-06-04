#!/bin/bash
out_path="/gpfswork/rech/wcp/uho93vn/src/1.explainability/nmf/dict/"
mkdir -p $out_path
python AragonRadio_nmf_pretrain.py --nmf_components 128\
			           --win_length 400 \
				   --n_fft 512 \
				   --n_speech 200 \
				   --n_music 500 \
				   --n_noise 500 \
				   --out_path $out_path \
				   --mu 1 \
				   --beta 0 \
				   --min_duration 1.0 \
				   --max_duration 4.0 


