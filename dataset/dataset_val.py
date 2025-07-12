import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import BaseDataset, update_caption
import glob
import random
from prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class ValDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, dataset_name, config, attn_maps_file=None, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num
        self.config = config

        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        self.anno_file = anno_file
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        anno = json.load(open(anno_file, 'r'))

        self.anno = []
        for index, a in enumerate(anno):
            a.update({
                'index': index
            })
            self.anno.append(a)

        self.attn_maps = torch.load(attn_maps_file, map_location='cpu') if attn_maps_file is not None else None

        if feat_file in ValDataset.cached_feats and img_feat_file in ValDataset.cached_feats:
            self.scene_feats, self.scene_masks = ValDataset.cached_feats[feat_file]
            self.scene_img_feats = ValDataset.cached_feats[img_feat_file]
        else:
            if feat_file is not None and os.path.exists(feat_file):
                self.feats = torch.load(feat_file, map_location='cpu')
            else:
                self.feats = None
            if img_feat_file is not None and os.path.exists(img_feat_file):
                self.img_feats = torch.load(img_feat_file, map_location='cpu')
            else:
                self.img_feats = None
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks = self.prepare_scene_features()
            ValDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            ValDataset.cached_feats[img_feat_file] = self.scene_img_feats

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids = self.get_anno(index)
        obj_id = int(self.anno[index].get('obj_id', 0))
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = int(self.anno[index].get('sqa_type', 0))
        if 'sqa_type' in self.anno[index]:
            type_info = self.anno[index]['sqa_type']
        elif 'eval_type' in self.anno[index]:
            type_info = self.anno[index]['eval_type'] 
        elif 'type_info' in self.anno[index]:
            type_info = self.anno[index]['type_info']
        if 'prompt' not in self.anno[index]:
            prompt = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            prompt = self.anno[index]["prompt"]
        ref_captions = self.anno[index]["ref_captions"].copy() if "ref_captions" in self.anno[index] else []
        qid = self.anno[index]["qid"] if "qid" in self.anno[index] else 0
        
        if 'multi3dref_mask3d_val' in self.anno_file:
            caption = ''
        elif 'sqa3d_val' in self.anno_file or 'scanqa_val' in self.anno_file:
            caption = random.choice(ref_captions)
            caption = caption.capitalize() + '.'
        elif 'scanrefer_mask3d_val' in self.anno_file or 'scan2cap_mask3d_val' in self.anno_file:
            caption = random.choice(ref_captions)
        else:
            caption = ''

        # for oracle debug
        assigned_ids = torch.arange(self.max_obj_num)
        ret_index = self.anno[index]['index']
        
        if self.attn_maps is not None:
            if self.config.use_a_map_ori:
                try:
                    attn_map = self.attn_maps[self.anno[index]['index']]['a_map_ori']
                except:
                    attn_map = self.attn_maps[self.anno[index]['index']]['a_map']
            else:
                attn_map = self.attn_maps[self.anno[index]['index']]['a_map']
            assigned_ids = torch.arange(self.max_obj_num)
            return caption, ret_index, scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, prompt, ref_captions, scene_id, qid, pred_id, type_info, attn_map
        
        return caption, ret_index, scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, prompt, ref_captions, scene_id, qid, pred_id, type_info
        # return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, prompt, ref_captions, scene_id, qid, pred_id, type_info


def val_collate_fn(batch, use_external_attn_maps=False):
    if use_external_attn_maps:
        caption, ret_index, scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, assigned_ids, prompts, ref_captions, scene_ids, qids, pred_ids, type_infos, attn_maps = zip(*batch)
        batch_attn_maps = torch.stack(attn_maps, dim=0)
    else:
        caption, ret_index, scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, assigned_ids, prompts, ref_captions, scene_ids, qids, pred_ids, type_infos = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    obj_ids = torch.tensor(obj_ids)
    pred_ids = torch.tensor(pred_ids)
    ret_dict = {
        'caption': caption,
        'captions': caption,
        "prompts": prompts,
        'index': ret_index,
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        "obj_ids": obj_ids,
        "custom_prompt": prompts,
        "ref_captions": ref_captions,
        "scene_id": scene_ids,
        "qid": qids,
        "pred_ids": pred_ids,
        "type_infos": type_infos
        # "ids": index
    }
    if use_external_attn_maps:
        ret_dict.update({
            'attention_maps': batch_attn_maps
        })
    return ret_dict
