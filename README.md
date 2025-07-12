# Fast3D
Fast3D is a plug-and-play visual token pruning framework for accelerating 3D MLLMs (*e.g.*, Chat-Scene). It could reach 90% visaul token pruning ratio with negligible performance drop through two technical innovations: *(1) Global Attention Prediction (GAP)*, where a lightweight neural network is trained to predict the aggregated attention map from all layers of the target model, enabling efficient token importance estimation for precise pruning guidance; and *(2) Sample-Adaptive visual token Pruning (SAP)*, which dynamically adjusts token budgets based on input complexity to achieve improved overall accuracy-efficiency trade-offs.

## Performance Comparison

  | Method | [ScanRefer](https://github.com/daveredrum/ScanRefer)  | [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) | [Scan2Cap](https://github.com/daveredrum/Scan2Cap) | [ScanQA](https://github.com/ATR-DBI/ScanQA) | [SQA3D](https://github.com/SilongYong/SQA3D) | Score Ratio |
  | :----:	|:---------:	|:-------:	|:------:	|:------:	|:---------:	|:---------:	|
  | | Acc@0.5 | F1@0.5 | B-4@0.5 | B-4 | EM-R | |
  | [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene)   | 50.40 | 53.21 | 35.92 | 13.55 | 56.83 | 100 % |
  | w/ [FastV](https://github.com/pkunlp-icler/FastV) 35% | 49.65 | 52.64 | 35.77 | 13.80 | 56.74 | 99.74 % |
  | w/ [FastV](https://github.com/pkunlp-icler/FastV) 65% | 22.91 | 28.26 | 29.36 | 12.84 | 56.01 | 74.72 % |
  | w/ [FastV](https://github.com/pkunlp-icler/FastV) 90% | 3.69 | 8.91 | 21.92  | 10.47 | 50.64 | 50.29 % |
  | w/ Fast3D(GAP) 35% | 50.84 | 53.55 | 35.32 | 13.29 | 56.86 | 99.60 % |
  | w/ Fast3D(GAP) 65% | 50.89 | 53.68 | 35.01 | 13.44 | 56.34 | 99.53 % |
  | w/ Fast3D(GAP) 90% | 50.02 | 51.09 | 32.64 | 12.79 | 55.42 | 95.61 % |
  | w/ Fast3D(GAP+SAP) ~90% | 50.94 | 53.06 | 34.60 | 13.29 | 56.22 | 98.82 % |

<small>\* *w/ method x%* indicates results with a *x%* average visual token pruning ratio. </small>


## Preparation
- Prepare the environment:

```shell
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n fast3d python=3.9.17
conda activate fast3d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
wget --quiet https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
cd transformers
pip install -e .
```

- Download LLM backbone:
  -  We use Vicuna-7B v1.5 in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5).

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the path of `vicuna-7b-v1.5`.
  

- Annotations and extracted features:
  
  Please follow the instructions in [preprocess](preprocess/README.md).


- Download Chat-Scene's pretrained checkpoint:

  We provide the pretrained checkpoint in Google Drive. Download it from either [Link 1](https://drive.google.com/file/d/1whwCaq0YfPGWX1VxB_gVn_BoLYasc_nb/view?usp=sharing) or [Link 2](https://drive.google.com/file/d/1Ziz7Be9l6MEbn3Qmlyr9gv42C0iJQgAn/view?usp=sharing).

## Chat-Scene Vanilla Inference

- Modify [eval.sh](scripts/eval.sh):

```python
val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
evaluate=True
pretrained_path="/path/to/pretrained_model.pth"
```

- Run: `bash scripts/eval.sh`

## Inference with FastV 

- Modify [batch_eval_fastv.sh](scripts/batch_eval_fastv.sh): 

```python
val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
evaluate=True
pretrained_path="/path/to/pretrained_model.pth"
# batch eval pruning ratios: 90%, 65%, 35%
rank_list=(15 60 90) # keep from 300 visual tokens
Ks=(2 6 16) # from which layer of 32 layers
```

- Run: `bash scripts/batch_eval_fastv.sh`


## Inference with Fast3D (GAP)

<details>
<summary>
1. extract global attention maps from Chat-Scene as the training target of the GAP network.
</summary>
  
  (You can skip this step and use our provided infer_attn_maps in [Google Drive](https://drive.google.com/drive/folders/16yQARetNqhp8hos2PgtHpnC5dMrCXnYe?usp=drive_link))

  - Modify [extract_attn_maps.sh](scripts/extract_attn_maps.sh)

  ```python
  train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
  val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
  evaluate=True
  pretrained_path="/path/to/pretrained_model.pth"
  ```

  - Run: `bash scripts/extract_attn_maps.sh`
</details>

<details>
<summary>
2. train the GAP network. 
</summary>
  
  (You can skip this step and use our provided trained_gap_network in [Google Drive](https://drive.google.com/drive/folders/16yQARetNqhp8hos2PgtHpnC5dMrCXnYe?usp=drive_link))

  - Modify [train.yaml](fast3d/config/train.yaml)

  ```yaml
  attn_maps_root: /path/to/infer_attn_maps(extracted_attn_maps)
  train_tags: scanrefer#scan2cap#scanqa#sqa3d#multi3dref
  val_tags: scanrefer#scan2cap#scanqa#sqa3d#multi3dref
  ```

  -  We use roberta-base in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/FacebookAI/roberta-base). Then modify Fast3dNetConfig.roberta_path in [modeling_fast3d.py](fast3d/modeling_fast3d.py).

  - We use 4× NVIDIA RTX 3090 GPUs to train the GAP network. Run: 
  ```bash
  accelerate config
  cd fast3d
  bash train_fast3d.sh
  ```
</details>

<details>
<summary>
3. get predicted attention maps from the trained GAP network. 
</summary>

  (You can skip this step and use our provided pred_attn_maps in [Google Drive](https://drive.google.com/drive/folders/16yQARetNqhp8hos2PgtHpnC5dMrCXnYe?usp=drive_link))

  - Modify [test.yaml](fast3d/config/test.yaml)

  ```yaml
  eval_only: True
  pretrained_model_path: /path/to/checkpoint_best.pth
  save_attn_maps: True
  attn_maps_root: /path/to/infer_attn_maps(extracted_attn_maps)
  val_tags: scanrefer#scan2cap#scanqa#sqa3d#multi3dref
  ```

  - Run: 
  ```bash
  accelerate config
  cd fast3d
  bash test_fast3d.sh
  ```
</details>
 
<details open>
<summary>
4. Chat-Scene inference with predicted attention maps. 
</summary>
  
  (Quick start: You can skip the above steps and use our provided predicted attention maps in [Google Drive](https://drive.google.com/drive/folders/16yQARetNqhp8hos2PgtHpnC5dMrCXnYe?usp=drive_link))

  - Modify [batch_eval_fast3d_pred_attn.sh](scripts/batch_eval_fast3d_pred_attn.sh)

  ```shell
  val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
  evaluate=True
  pretrained_path="/path/to/pretrained_model.pth"
  use_fast_v=False
  use_fast_v_oracle=True
  use_external_attn_maps=True
  use_a_map_ori=False
  val_attn_maps_path="/path/to/predicted_attn_maps"
  # batch eval pruning ratios: 90%, 65%, 35%
  rank_list=(15 60 90)
  Ks=(2 6 16)
  ```

  - Run: `bash scripts/batch_eval_fast3d_pred_attn.sh`
</details>


## Inference with Fast3D (GAP+SAP)

- manual search total attention score threshold alpha: modify [search_alpha.py](tasks/search_alpha.py)

  ```python
  alpha = 0.21
  target_pruning_ratio = 90
  tolerance = 2
  pred_attn_maps_path = "/path/to/predicted_attn_maps"
  ```

  then run `python tasks/search_alpha.py` to check if the alpha is valid.

- Modify [batch_eval_fast3d_pred_attn_adaptive.sh](scripts/batch_eval_fast3d_pred_attn_adaptive.sh): (We provide the predicted attention maps in [Google Drive](https://drive.google.com/drive/folders/16yQARetNqhp8hos2PgtHpnC5dMrCXnYe?usp=drive_link))

  ```shell
  val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
  evaluate=True
  pretrained_path="/path/to/pretrained_model.pth"
  use_fast_v=False
  use_fast_v_oracle=True
  use_external_attn_maps=True
  use_a_map_ori=False
  val_attn_maps_path="/path/to/predicted_attn_maps"
  alpha_list=(0.21)
  Ks=(0)
  ```

- Run: `bash scripts/batch_eval_fast3d_pred_attn_adaptive.sh`


## 📄 Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@inproceedings{huang2025fast3d,
  title={Fast3D: Accelerating 3D Multi-modal Large Language Models for Efficient 3D Scene Understanding},
  author={Huang, Wencan and Liu, Daizong and Hu, Wei},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025}
}
```

## 😊 Acknowledgement

Thanks to the open source of the following projects: [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene), [FastV](https://github.com/pkunlp-icler/FastV), and [Vil3dref](https://github.com/cshizhe/vil3dref)
