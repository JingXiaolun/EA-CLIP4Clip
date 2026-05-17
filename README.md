# 【NeurCom'2024 :fire:】An empirical study of excitation and aggregation design adaptions in CLIP4Clip for video–text retrieval

<p align="center">
    <a href="https://doi.org/10.1016/j.neucom.2024.127905"><img src="https://img.shields.io/badge/NeurCom-2024-yellow.svg" alt="Build Status"></a>
    <a href="https://arxiv.org/abs/2406.01604"><img src="https://img.shields.io/badge/Paper-arxiv.2406.01604-b31b1b.svg" alt="Build Status"></a>
</p>

The implementation of NeurCom 2024 paper [An empirical study of excitation and aggregation design adaptions in CLIP4Clip for video–text retrieval](https://arxiv.org/abs/2406.01604). 

## :pushpin: Citation
If you find our method useful in your work, please cite:
```bibtex
@article{jing2024empirical,
  title={An empirical study of excitation and aggregation design adaptions in CLIP4Clip for video-text retrieval},
  author={Jing, Xiaolun and Yang, Genke and Chu, Jian},
  journal={Neurocomputing},
  pages={127905},
  year={2024},
  publisher={Elsevier}
}
```

## :closed_book: Overview
CLIP4Clip model transferred from the CLIP has been the de-factor standard to solve the video clip retrieval task
from frame-level input, triggering the surge of CLIP4Clip-based models in the video–text retrieval domain. In
this work, we rethink the inherent limitation of widely-used mean pooling operation in the frame features
aggregation and investigate the adaptions of excitation and aggregation design for discriminative video
representation generation. We present a novel excitation-and-aggregation design, including (1) The excitation
module is available for capturing non-mutually-exclusive relationships among frame features and achieving
frame-wise features recalibration, and (2) The aggregation module is applied to learn exclusiveness used for
frame representations aggregation. Similarly, we employ the cascade of sequential module and aggregation
design to generate discriminative video representation in the sequential type. Besides, we adopt the excitation
design in the tight type to obtain representative frame features for multi-modal interaction. The proposed
modules are evaluated on three benchmark datasets of MSR-VTT, ActivityNet and DiDeMo, achieving MSRVTT (43.9 R@1), ActivityNet (44.1 R@1) and DiDeMo (31.0 R@1). They outperform the CLIP4Clip results
by +1.2% (+0.5%), +4.5% (+1.9%) and +9.5% (+2.7%) relative (absolute) improvements, demonstrating the
superiority of our proposed excitation and aggregation designs. We hope our work will serve as an alternative
for frame representations aggregation and facilitate future research.

## :books: Method

![image](https://ars.els-cdn.com/content/image/1-s2.0-S0925231224006763-gr2_lrg.jpg)

## :rocket: Quick Start
### Setup code environment
```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```

### Datasets
We train our model on MSR-VTT, ActivityNet and DiDeMo datasets respectively. Please refer to this [repo](https://github.com/ArrowLuo/CLIP4Clip) for data preparation.

| Datasets  | Google Cloud    | Baidu Yun | Peking University Yun|
|:------:|:------:|:------:|:------:|
| MSR-VTT  | [Download](https://drive.google.com/drive/folders/1LYVUCPRxpKMRjCSfB_Gz-ugQa88FqDu_)  | [Download](https://pan.baidu.com/share/init?surl=Gdf6ivybZkpua5z1HsCWRA&pwd=enav) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6A028EE7EF5C48A788118B82D6ABE0C5?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=MSRVTT&type=anonymous) |
| ActivityNet  | TODO  | [Download](https://pan.baidu.com/s/1tI441VGvN3In7pcvss0grg?pwd=2ddy) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AAE744E6488E2049BD9412738E14AAA8EA?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=activitynet&type=anonymous) |
| DiDeMo  | TODO  | [Download](https://pan.baidu.com/share/init?surl=Tsy9nb1hWzeXaZ4xr7qoTg&pwd=c842) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA14E48D1333114022B736291D60350FA5?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=didemo&type=anonymous) |

### Download CLIP Model
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

### Compress Video
```bash
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```

### Train on MSRVTT (Squeeze Excitation-and-Aggregation)
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29505' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ../DataSet/MSRVTT/data/file/MSRVTT_train.9k.csv \
--val_csv ../DataSet/MSRVTT/data/file/MSRVTT_JSFUSION_test.csv \
--data_path ../DataSet/MSRVTT/data/file/MSRVTT_data.json \
--features_path ../DataSet/MSRVTT/data/compressed/clip4clip_video_frame_input \
--output_dir ../Model/Experience/eaclip4clip_exc_agg_msrvtt_vit32 \
--log_dir ../Log/Experience/eaclip4clip_exc_agg_msrvtt_vit32 \
--visualize_dir ../Visualize/Experience/eaclip4clip_exc_agg_msrvtt_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--se_block --se_type excitation_aggregation --reduction_ratio 4 \
--pretrained_clip_name ViT-B/32
```

### Train on MSRVTT (Expansion Aggregation)
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29506' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ../DataSet/MSRVTT/data/file/MSRVTT_train.9k.csv \
--val_csv ../DataSet/MSRVTT/data/file/MSRVTT_JSFUSION_test.csv \
--data_path ../DataSet/MSRVTT/data/file/MSRVTT_data.json \
--features_path ../DataSet/MSRVTT/data/compressed/clip4clip_video_frame_input \
--output_dir ../Model/Experience/eaclip4clip_agg_msrvtt_seqLSTM_vit32 \
--log_dir ../Log/Experience/eaclip4clip_agg_msrvtt_seqLSTM_vit32 \
--visualize_dir ../Visualize/Experience/eaclip4clip_agg_msrvtt_seqLSTM_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--se_block --se_type aggregation --reduction_ratio 0.25 \
--pretrained_clip_name ViT-B/32
```

### Train on MSRVTT (Expansion Aggregation)
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29507' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ../DataSet/MSRVTT/data/file/MSRVTT_train.9k.csv \
--val_csv ../DataSet/MSRVTT/data/file/MSRVTT_JSFUSION_test.csv \
--data_path ../DataSet/MSRVTT/data/file/MSRVTT_data.json \
--features_path ../DataSet/MSRVTT/data/compressed/clip4clip_video_frame_input \
--output_dir ../Model/Experience/eaclip4clip_agg_msrvtt_seqTransf_vit32 \
--log_dir ../Log/Experience/eaclip4clip_agg_msrvtt_seqTransf_vit32 \
--visualize_dir ../Visualize/Experience/eaclip4clip_agg_msrvtt_seqTransf_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--se_block --se_type aggregation --reduction_ratio 0.25 \
--pretrained_clip_name ViT-B/32
```

### Train on MSRVTT (Squeeze Excitation)
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29508' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ../DataSet/MSRVTT/data/file/MSRVTT_train.9k.csv \
--val_csv ../DataSet/MSRVTT/data/file/MSRVTT_JSFUSION_test.csv \
--data_path ../DataSet/MSRVTT/data/file/MSRVTT_data.json \
--features_path ../DataSet/MSRVTT/data/compressed/clip4clip_video_frame_input \
--output_dir ../Model/Experience/eaclip4clip_exc_msrvtt_tightTransf_vit32 \
--log_dir ../Log/Experience/eaclip4clip_exc_msrvtt_tightTransf_vit32 \
--visualize_dir ../Visualize/Experience/eaclip4clip_exc_msrvtt_tightTransf_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--linear_patch 2d --sim_header tightTransf \
--se_block --se_type excitation --reduction_ratio 4 \
--pretrained_clip_name ViT-B/32
```

## :reminder_ribbon: Acknowledgments
Our code is based on [CLIP](https://github.com/openai/CLIP) and [UniVL](https://github.com/microsoft/UniVL).
