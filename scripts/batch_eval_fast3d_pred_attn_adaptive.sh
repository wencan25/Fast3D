# nohup bash scripts/batch_eval_fast3d_pred_attn_adaptive.sh > "outputs/batch_eval_fast3d_pred_attn_adaptive_run1.log" 2>&1 &
which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=32
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config=""
max_grad_norm=0.01
seed=42
use_location_token=False

llama_model_path="llm/vicuna-7b-v1.5"

train_tag=""
val_tag="scanrefer#scan2cap#multi3dref#scanqa#sqa3d"
evaluate=True
debug=False
enable_wandb=False
gpu_num=4
do_save=True
pretrained_path="ckpt_01_3446.pth"
num_workers=4


use_fast_v=False
use_fast_v_oracle=True
use_external_attn_maps=True
use_a_map_ori=False
val_attn_maps_path="fast3d/outputs/pred_attn_maps"
alpha_list=(0.21)
Ks=(0)

len=${#Ks[@]}
for ((i=0; i<len; i++)); do
    k=${Ks[i]}
    alpha=${alpha_list[i]}

    other_info="fast3d_pred_attn_ada_${use_external_attn_maps}_layer${k}_alpha${alpha}_oracle${use_fast_v_oracle}_a_map_ori${use_a_map_ori}"
    tag="${val_tag}__${other_info}"

    OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
    mkdir -p ${OUTPUT_DIR}
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${gpu_num} \
        tasks/inference_fast3d_pred_attn_ada.py \
        "$(dirname $0)/${config}config.py" \
        output_dir "$OUTPUT_DIR" \
        scheduler.epochs "$epoch" \
        optimizer.lr "$lr" \
        model.add_scene_token "$add_scene_token" \
        model.add_img_token "$add_img_token" \
        pretrained_path "$pretrained_path" \
        evaluate "$evaluate" \
        wandb.enable "$enable_wandb" \
        gpu_num "$gpu_num" \
        do_save "$do_save" \
        batch_size "$batch_size" \
        num_workers "$num_workers" \
        model.train_emb "$train_emb" \
        model.train_img_proj "$train_img_proj" \
        train_tag "$train_tag" \
        val_tag "$val_tag" \
        use_fast_v "$use_fast_v" \
        use_fast_v_oracle "$use_fast_v_oracle" \
        use_external_attn_maps "$use_external_attn_maps" \
        use_a_map_ori "$use_a_map_ori" \
        val_attn_maps_path "$val_attn_maps_path" \
        fast_v_agg_layer "$k" \
        adaptive_rank_alpha "$alpha" \
        model.no_obj "$no_obj" \
        segmentor "$segmentor" \
        pc_encoder "$pc_encoder" \
        model.input_dim "$input_dim" \
        model.bidirection "$bidirection" \
        optimizer.different_lr.enable "$different_lr" \
        model.max_obj_num "$max_obj_num" \
        lora.lora_r "$lora_r" \
        lora.lora_alpha "$lora_alpha" \
        model.add_pos_emb "$add_pos_emb" \
        model.feat_fusion "$feat_fusion" \
        optimizer.max_grad_norm "$max_grad_norm" \
        seed "$seed" \
        model.fuse_with_id "$fuse_with_id" \
        model.llama_model_path "$llama_model_path" \
        model.use_location_token "$use_location_token"
    done
done
