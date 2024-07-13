import torch
from torch import nn
from models.cav_mae import CAVMAEFT
from models.decoder import TransformerDecoderTextualHead, GeneratorWithBeamSearch, CaptioningModel

def get_avcap_model(tokenizer, avencoder, param):
    audio_visual_encoder = avencoder# non meaningful label dim
    text_decoder = TransformerDecoderTextualHead(
        av_feature_size=param.get('av_feature_size', 768),
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768* 4,
        max_caption_length=1024,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        av_projection_type='linearLn',
    )

    decoder = GeneratorWithBeamSearch(
        eos_index=tokenizer.sep_token_id,
        #max_steps=40,
        max_steps=1024,
        beam_size=4,
        length_penalty=0.6,
    )

    model = CaptioningModel(
        audio_visual_encoder,
        text_decoder,
        decoder=decoder,
        sos_index=tokenizer.cls_token_id,
        eos_index=tokenizer.sep_token_id,
        tokenizer=tokenizer,
        use_history_for_infer=True,
        loss_type='smooth',
        num_image_with_embedding=param.get('num_image_with_embedding'),
        mode=param['mode']
    )
    return model

    