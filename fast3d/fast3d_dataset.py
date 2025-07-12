import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from torch.nn.utils.rnn import pad_sequence
import re


class Fast3dDataset(Dataset):
    cached_feats = None

    def __init__(
        self,
        anno_file,
        attn_maps_file,
        attibutes_file,
        use_ori_attn_maps=False,
        use_mentioned_oids_in_answers=True,
        feat_file="../annotations/scannet_mask3d_uni3d_feats.pt",
        img_feat_file="../annotations/scannet_mask3d_videofeats.pt",
        max_obj_num=100,
        feat_dim=1024,
        img_feat_dim=1024,
    ):
        self.max_obj_num = max_obj_num
        self.feat_dim = feat_dim
        self.img_feat_dim = img_feat_dim
        self.feat_file = feat_file
        self.img_feat_file = img_feat_file
        self.anno_file = anno_file
        with open(anno_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        self.annotations = []
        for index, anno in enumerate(annotations):
            anno.update({"index": index})
            self.annotations.append(anno)

        self.attn_maps = preprocess_attn_maps(
            torch.load(attn_maps_file, map_location="cpu")
        )
        self.use_ori_attn_maps = use_ori_attn_maps
        self.use_mentioned_oids_in_answers = use_mentioned_oids_in_answers

        self.attibutes = torch.load(attibutes_file, map_location="cpu")

        if Fast3dDataset.cached_feats is not None:
            self.scene_features_dict = Fast3dDataset.cached_feats
        else:
            self.scene_features_dict = self.prepare_scene_features()
            Fast3dDataset.cached_feats = self.scene_features_dict

    def prepare_scene_features(self):
        self.feats = torch.load(self.feat_file, map_location="cpu", weights_only=True)
        self.img_feats = torch.load(
            self.img_feat_file, map_location="cpu", weights_only=True
        )
        scan_ids = set(["_".join(x.split("_")[:-1]) for x in list(self.feats.keys())])
        scan_ids = [sid for sid in scan_ids if sid.endswith("_00")]

        scene_features_dict = {}
        for scan_id in scan_ids:
            # 每个场景限制加载 max_obj_num 个物体
            obj_num = self.max_obj_num
            obj_ids = [_ for _ in range(obj_num)]

            scene_feat = []
            scene_img_feat = []
            for oid in obj_ids:
                item_id = "_".join([scan_id, f"{oid:02}"])
                # 场景中物体个数少于 max_obj_num 时添加全0特征
                if item_id not in self.feats:
                    scene_feat.append(torch.zeros(self.feat_dim))
                else:
                    scene_feat.append(self.feats[item_id].float())

                if item_id not in self.img_feats:
                    scene_img_feat.append(torch.zeros(self.img_feat_dim))
                else:
                    scene_img_feat.append(self.img_feats[item_id].float())

            scene_feat = torch.stack(scene_feat, dim=0)  # max_obj_num * self.feat_dim
            scene_img_feat = torch.stack(
                scene_img_feat, dim=0
            )  # max_obj_num * self.img_feat_dim
            scene_features_dict[scan_id] = dict(
                scene_feat=scene_feat,
                scene_img_feat=scene_img_feat,
            )
        return scene_features_dict

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        scene_id = self.annotations[index]["scene_id"]
        if scene_id not in self.scene_features_dict:
            print(f"{scene_id} not in scene_features_dict")
            return self.__getitem__(random.randint(0, len(self.annotations) - 1))
        scene_feature_dict = self.scene_features_dict[scene_id]
        scene_locs = self.attibutes[scene_id]["locs"]

        prompt = self.annotations[index]["prompt"]

        if "ref_captions" in self.annotations[index]:
            ref_captions = self.annotations[index]["ref_captions"]
        else:
            ref_captions = [""]

        caption = self.annotations[index].get(
            "caption", random.choice(ref_captions) if ref_captions != [] else ""
        )
        if not isinstance(caption, str):
            caption = ""

        a_map_key = "a_map" if not self.use_ori_attn_maps else "a_map_ori"
        try:
            attn_map = self.attn_maps[self.annotations[index]["index"]][a_map_key]
        except:
            attn_map = self.attn_maps[self.annotations[index]["index"]]["a_map"]

        if self.use_mentioned_oids_in_answers:
            mentioned_oids_list = find_oids_in_str(prompt + caption)
        else:
            mentioned_oids_list = find_oids_in_str(prompt)

        mentioned_oids = torch.zeros(self.max_obj_num, dtype=torch.float32)
        mentioned_oids[mentioned_oids_list] = 1
        return (
            self.annotations[index]["index"],
            mentioned_oids,
            attn_map,
            scene_feature_dict["scene_feat"],
            scene_feature_dict["scene_img_feat"],
            scene_locs,
            prompt,
        )


def find_oids_in_str(s):
    id_format = "<OBJ\\d{3}>"
    oids = []
    for match in re.finditer(id_format, s):
        start, end = match.span()
        oid = int(s[start:end][4:-1])
        oids.append(oid)
    return oids


def fast3d_collator(batch):
    index, mentioned_oids, attn_map, scene_feat, scene_img_feat, scene_locs, prompt = zip(
        *batch
    )
    batch_scene_feat = pad_sequence(scene_feat, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feat, batch_first=True)
    mentioned_oids = torch.stack(mentioned_oids, dim=0)
    attn_map = torch.stack(attn_map, dim=0)
    scene_locs = torch.stack(scene_locs, dim=0)

    ret = {
        "index": index,
        "mentioned_oids": mentioned_oids,
        "attn_maps": attn_map,
        "object_features": batch_scene_feat,
        "object_img_features": batch_scene_img_feat,
        "object_locations": scene_locs,
        "prompt": prompt,
    }
    return ret


def preprocess_attn_maps(attn_maps):
    if isinstance(attn_maps, list):
        new_attn_maps = {}
        for a_map in attn_maps:
            new_attn_maps[a_map["index"]] = a_map
    elif isinstance(attn_maps, dict):
        new_attn_maps = attn_maps
    else:
        raise Exception
    return new_attn_maps
