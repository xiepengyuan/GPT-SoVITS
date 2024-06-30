# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import soundfile as sf
import librosa
import LangSegment

from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from tools.i18n.i18n import I18nAuto
from my_utils import load_audio
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

cnhubert_base_path = "/data1/xiepengyuan/.cache/huggingface/GPT-SoVITS/chinese-hubert-base"
bert_path = "/data1/xiepengyuan/.cache/huggingface/GPT-SoVITS/chinese-roberta-wwm-ext-large"

cnhubert.cnhubert_base_path = cnhubert_base_path

is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = False

i18n = I18nAuto()
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)
ssl_model = cnhubert.get_model().to(device)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


dtype = torch.float16 if is_half == True else torch.float32


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)
    return bert


def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones, bert.to(dtype), norm_text


def change_sovits_weights(sovits_path):
    import json
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, ref_free=False, pred_semantic=None):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)

    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)

        prompt_semantic = codes[0, 0]

    audio_opt = []

    if (text[-1] not in splits):
        text += "。" if text_language != "en" else "."
    print(i18n("实际输入的目标文本(每句):"), text)
    phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
    print(i18n("前端处理后的文本(每句):"), norm_text2)

    # 推理
    prompt = prompt_semantic.unsqueeze(0).to(device)

    # print(pred_semantic.shape,idx)
    if pred_semantic is None:
        pred_semantic = prompt[:, :].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次

    refer = get_spepc(hps, ref_wav_path)  # .to(device)
    if is_half == True:
        refer = refer.half().to(device)
    else:
        refer = refer.to(device)

    # codes转audio
    audio = (
        vq_model.decode(
            pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
        )
        .detach()
        .cpu()
        .numpy()[0, 0]
    )

    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
    if max_audio > 1:
        audio /= max_audio
    audio_opt.append(audio)
    audio_opt.append(zero_wav)

    return hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


def get_semantic(ref_wav_path):
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        # if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
        #     raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)

        semantic = codes[0, 0].unsqueeze(0).unsqueeze(0).to(device)
        # print(semantic)
        # semantic = torch.tensor(
        # [[[520, 72, 722, 272, 102, 415, 352,  73,  388,  836,  551,
        #        337,    4,  685,  483,  460,  135,  915,   77,  136,  261,  891,
        #        117,  240,  185,  190,  596,  609,  140,  360,  913,  937,  574,
        #        161,  138,   34,  438,   34,  808,  463,  766,  308,  195,  947,
        #        692,  851,  576,  751, 1009,  783,  871,  659,  867,  535,  570,
        #        242,  103,  623,   41,  149,  149,  290,  124,  477,   69,   77,
        #        957,  777,   51,  502,  462,  242,  915,   77,  741,  771,  172,
        #        298,  850,  674,  501,  808,  152,  297,  947,  379,   95,  812,
        #        738,  638,   90,  140,  360,  140,  140, 149,  824,  150,   33,
        #        195,  450,  468,  467,  949,  363,  639,  144,  489,  232,  333,
        #        376,  187,  762,  639,   87,  321,  777,  537,  682,  926, 1001,
        #        714,  242,   53,  280,  486,  486,  486,  486,  486,  486,  486,
        #        486,  486,  280,  280,  105]]], device='cuda:0')
    return semantic


def synthesize():
    # SoVITS_model_path = "/data1/xiepengyuan/workspace/audio/GPT-SoVITS/SoVITS_weights/base_aishell3_e458_s265640.pth"
    # SoVITS_model_path = "/data1/xiepengyuan/.cache/huggingface/GPT-SoVITS/s2G488k.pth"
    SoVITS_model_path = "/data1/xiepengyuan/workspace/audio/GPT-SoVITS/SoVITS_weights/base_mihoyo_e500_s432500.pth"
    # ref_audio_path = target_audio_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/5-wav32k/SSB02730412.wav"
    # ref_text = target_text = "它位于厄瓜多尔首都,基,多南部约一百,三,十,千米处."
    ref_audio_path = target_audio_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/5-wav32k/SSB00050001.wav"
    ref_text = target_text = "广州女大学生,登,山失联四天警方找到疑似女尸."
    # ref_audio_path = target_audio_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/samples/TaoXing0600.wav"
    # ref_text = target_text = "修正局的警员为什么搜查,你有头绪吗?是不是隐瞒了什么事情?"
    # output_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/output"
    output_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_mihoyo/output"

    change_sovits_weights(sovits_path=SoVITS_model_path)
    # target_audio_path = "/data1/xiepengyuan/data/audio_datas/tts/OC8145/train_gpt-sovits_output/slicer_opt/哈利有礼包-主流程3.1.wav_0_609920.wav"
    # target_text = "恩恩,目前游戏幸运轮盘上新,传说品质时装宴会风雅,魔杖外观黑车闪击,猫头鹰雪橇上新,现在呢还有决斗俱乐部全新自定义玩法开启,可以自由编辑决斗赛场,有你做主,有空呢记得回霍格沃茨看看吧."
    # target_audio_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/samples/YeBieZhi0270.wav"
    # target_text = "蔬菜我洗好咯,鸡翅可以冻冰箱,草莓的保质期很短,记得今晚就吃完哦."
    # target_audio_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/samples/TaoXing0630.wav"
    # target_text = "修正员身穿制服，佩戴瑞恩梅克徽记时，是不需要出示证件的."
    # target_audio_path = "/data1/xiepengyuan/exp/audio/gpt_sovits/base_aishell3/5-wav32k/SSB00050001.wav"
    # target_text = "广州女大学生,登,山失联四天警方找到疑似女尸."
    semantic = get_semantic(target_audio_path)

    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path,
                                   prompt_text=ref_text,
                                   prompt_language="zh",
                                   text=target_text,
                                   text_language="zh",
                                   pred_semantic=semantic)

    last_sampling_rate, last_audio_data = synthesis_result
    output_wav_path = os.path.join(output_path, "output.wav")
    sf.write(output_wav_path, last_audio_data, last_sampling_rate)

    print("Audio saved to " + output_wav_path)


def run():
    synthesize()


if __name__ == '__main__':
    run()
