import torch


def find_min_rank_sum_larger_than_alpha(attn_map, alpha=0.95):
    """
    attn_map: torch.FloatTensor (100, )
    Find the minimum rank so that the attention sum of the top rank is greater than or equal to alpha
    """
    sorted_attn = torch.sort(attn_map, descending=True).values
    cum_sum = torch.cumsum(sorted_attn, dim=0)
    mask = cum_sum >= alpha

    if mask.any():
        # Find the first index with True
        idx = mask.nonzero(as_tuple=True)[0][0].item()
        return idx + 1
    else:
        return attn_map.size(0)


import numpy as np
from tqdm import tqdm

setnames = "scanrefer#scan2cap#multi3dref#scanqa#sqa3d".split("#")
alpha = 0.21
target_pruning_ratio = 90
tolerance = 2
pred_attn_maps_path = "fast3d/outputs/pred_attn_maps"
average_pruning_ratios = []
for setname in setnames:
    a = torch.load(f"{pred_attn_maps_path}/infer_attn_maps_val_{setname}.pt")
    stat1 = []
    stat2 = []
    for k, v in tqdm(a.items()):
        tmp = v["a_map"].softmax(-1)
        t1 = tmp.topk(10).values.sum()
        t2 = find_min_rank_sum_larger_than_alpha(tmp, alpha)
        stat1.append(t1)
        stat2.append(t2)
    average_pruning_ratio = 100 - np.mean(stat2)
    average_pruning_ratios.append(average_pruning_ratio)
    print(
        setname,
        "average pruning ratio =",
        average_pruning_ratio,
        "reference alpha =",
        np.mean(stat1),
    )

x = np.mean(average_pruning_ratios)
valid = abs(x - target_pruning_ratio) / target_pruning_ratio <= tolerance / 100
print("=" * 10)
print("current alpha =", alpha)
print("target_pruning_ratio =", target_pruning_ratio)
print("average_pruning_ratio =", x)
print(
    f"|{target_pruning_ratio}-{x}|/{target_pruning_ratio} {'<=' if valid else '>'} {tolerance}%"
)
print("is_valid =", valid)
