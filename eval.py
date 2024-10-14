from typing import List, Literal, Union
import os
import argparse 
import glob 
import json

import clip
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ClipEvaluator(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-L/14@336px": 336,
        }.get(name, 224)

        self.model, self.preprocess = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    @property
    def device(self):
        return next(self.parameters()).device

    def normalize_feature(self, feature):
        """
        features: [*, D]
        """
        feature = feature / feature.norm(dim=-1, keepdim=True)
        return feature

    @torch.no_grad()
    def encode_text(self, text: str):
        """
        Return unnormalized text feature.
        """
        text = clip.tokenize(text, truncate=True).to(self.device)
        text_features = self.model.encode_text(text)
        return text_features

    @torch.no_grad()
    def encode_image(self, pil_or_img_path: Union[Image.Image, str, os.PathLike]):
        """
        Return unnormalized image feature.
        """
        if isinstance(pil_or_img_path, str) or isinstance(pil_or_img_path, os.PathLike):
            image = (
                self.preprocess(Image.open(pil_or_img_path))
                .unsqueeze(0)
                .to(self.device)
            )
        elif isinstance(pil_or_img_path, Image.Image):
            image = self.preprocess(pil_or_img_path).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"{type(pil_or_img_path)}")

        image_features = self.model.encode_image(image)
        return image_features
    
    @torch.no_grad()
    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        """
        Return CLIP similarity matrix.
        Input:
            img_features: [Nimg,D]
            text_features: [Ntxt,D]
        Output:
            cos_sim_mat: [Nimg, Ntxt]
        """
        D = img_features.shape[-1]
        img_features = img_features.reshape(-1, 1, D)
        text_features = text_features.reshape(1, -1, D)

        cos_sim_mat = F.cosine_similarity(img_features, text_features, dim=-1) #[Nimg, Ntxt]
        return cos_sim_mat

    @torch.no_grad()
    def measure_clip_sim_from_img_and_text(self, pil_or_img_path: Union[Image.Image, str, os.PathLike], text: str):
        img_f = self.encode_image(pil_or_img_path)
        txt_f = self.encode_text(text)
        cos_sim = self(img_f, txt_f).squeeze()
        return cos_sim

    @torch.no_grad()
    def measure_visual_anagram_metrics(
        self, image_0_path: str, image_1_path: str, text_0: str, text_1: str
    ):
        """
        image_0_path: view 0 image path
        image_1_path: view 1 image path
        text_0: view 0 prompt
        text_1: view 1 prompt
        """
        image_features_0 = self.encode_image(image_0_path)
        image_feature_1 = self.encode_image(image_1_path)
        text_feature_0 = self.encode_text(text_0)
        text_feature_1 = self.encode_text(text_1)

        image_features = torch.cat([image_features_0, image_feature_1], 0)
        text_features = torch.cat([text_feature_0, text_feature_1], 0)

        logit_scale = self.model.logit_scale.exp()
        cos_sim_mat = self(image_features, text_features)
        logit_mat = cos_sim_mat / logit_scale

        alignment_score = min(torch.diag(cos_sim_mat))
        concealment_score = (
            torch.diag(logit_mat.softmax(dim=0)).mean()
            + torch.diag(logit_mat.softmax(dim=1)).mean()
        ) / 2

        return alignment_score, concealment_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir1", type=str, required=True)
    
    args = parser.parse_args()
    
    name = "ViT-B/32" # Change from Vit-B/14 -> ViT-B/32
    clip_evaluator = ClipEvaluator(name=name).cuda()

    fdir1 = args.fdir1  # imgae directory 1

    prompt_view_pairs = defaultdict(list)
    for img_path in glob.glob(f"{fdir1}/*.png"):
        fname = img_path.split("/")[-1].split(".")[0]
        prompt = fname
        prompt_key = prompt.replace(" ", "_")
        prompt_view_pairs[prompt_key].append(img_path)

    metric_dict = {
        "source_dir": fdir1,
    }
    
    final_score = []
    for prompt, img_path_list in prompt_view_pairs.items():
        prompt = prompt.replace("_", " ")
        txt_f = clip_evaluator.encode_text(prompt)
        img_f = []
        for img_path in img_path_list:
            print("Procesing", prompt, img_path)
            # 1. Encode image and text first and measure clip sim.
            img_f.append(clip_evaluator.encode_image(img_path))
        img_f = torch.cat(img_f, dim=0)
        
        cos_sim = clip_evaluator(img_f, txt_f).squeeze().mean().item()

        metric_dict[img_path_list[0]] = cos_sim
        final_score.append(cos_sim)

    metric_dict["score"] = sum(final_score) / len(final_score)
 
    print("Cosine similarity ", metric_dict["score"])
    
    save_path = os.path.join(args.fdir1, "eval.json")
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
    
