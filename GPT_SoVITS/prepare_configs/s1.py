# -*- coding: utf-8 -*-

import yaml
import os


exp_root = "/data1/xiepengyuan/exp/audio/gpt_sovits"
exp_name = "base_mihoyo"
is_half = True

batch_size = 4
total_epoch = 500
pretrained_s1 = ""
save_every_epoch = 4
if_save_every_weights = True
if_save_latest = True
if_dpo = False
GPT_weight_root = "/data1/xiepengyuan/workspace/audio/GPT-SoVITS/GPT_weights"


def run():
    with open("/data1/xiepengyuan/workspace/audio/GPT-SoVITS/GPT_SoVITS/configs/s1longer.yaml") as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    s1_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["optimizer"]["lr"] = 1
    data["optimizer"]["lr_init"] = 1e-2
    data["pretrained_s1"] = pretrained_s1
    data["data"]["max_sec"] = 15
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = GPT_weight_root
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
    data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
    data["output_dir"] = "%s/logs_s1" % s1_dir

    config_path = "%s/config_s1.yaml" % s1_dir
    with open(config_path, "w") as f:
        f.write(yaml.dump(data, default_flow_style=False))


if __name__ == '__main__':
    run()
