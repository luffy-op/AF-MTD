from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertForMaskedLM
from models.cross_transformer import CrossTransformer
import timm
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

from transformers import AutoModel
from models.tiny_vit import tiny_vit_21m_224

# TBPS-CLIP T1
from models.tbps_model_teacher import clip_vitb
from misc.utils import parse_config
import os
from text_utils.tokenizer import tokenize
from loss_distill import MultiTeacherDistillLoss

config_tbpsclip_path = 'configs/config_tbpsclip.yaml'
config_tbpsclip = parse_config(config_tbpsclip_path)

# MARS T2
from models.model_person_search import ALBEF as ALBEF_TCH
import ruamel_yaml as yaml
config_marstch_path = 'configs/PS_cuhk_pedes.yaml'
config_marstch = yaml.load(open(config_marstch_path, 'r'), Loader=yaml.Loader)

from models.tokenization_bert import BertTokenizer
text_encoder_t2 = '/workspace/zl/t2i-reid/APTM/data/bert-base-uncased'
tokenizer_t2 = BertTokenizer.from_pretrained(text_encoder_t2)


# Teacher 
from .pca_layer import pca_layers

class ALBEF(nn.Module):
    def __init__(self,
                 path_to_pretrained_weights,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 tbpsclip_numclasses=11003,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        # self.visual_encoder = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=512, depth=6, num_heads=8,
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
        # self.visual_encoder =  timm.create_model('tinyvit_21m_224', pretrained=True)
        self.visual_encoder = tiny_vit_21m_224(pretrained=True)
        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.visual_decoder = CrossTransformer(768, 8, 96, 
                                depth = 4, context_dim=768)
        self.decoder_pred = nn.Linear(768, 16**2 * 3, bias=True) # decoder to patch
        scale = 768 ** -0.5 # 1/sqrt(768)
        self.visual_decoder_pos_embed = nn.Parameter(scale * torch.randn(577, 768))
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, 768))
        
        self.text_encoder_cross = BertForMaskedLM.from_pretrained('/workspace/zl/t2i-reid/APTM/data/bert-base-uncased', config=bert_config)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(self.text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.mlm_probability = config['mlm_probability']
        self.mrtd_mask_probability = config['mrtd_mask_probability']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(self.text_width, 2)
        # self.itm_head_attr = nn.Linear(self.text_width, 2)
        self.prd_head = nn.Linear(self.text_width, 2)
        self.mrtd_head = nn.Linear(self.text_width, 2)
        # create momentum models
        # self.visual_encoder_m = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=512, depth=6, num_heads=8,
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
        # self.visual_encoder_m =  timm.create_model('tinyvit_21m_224', pretrained=True)
        self.visual_encoder_m = tiny_vit_21m_224(pretrained=True)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_cross_m = BertForMaskedLM.from_pretrained('/workspace/zl/t2i-reid/APTM/data/bert-base-uncased', config=bert_config)
        self.text_encoder_m = AutoModel.from_pretrained(text_encoder)
        self.text_proj_m = nn.Linear(self.text_width, embed_dim)
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        
        # teacher1:   TBPS-CLIP
        self.model_t1 = clip_vitb(config=config_tbpsclip, num_classes=tbpsclip_numclasses)
        checkpoint_t1 = torch.load(os.path.join("/workspace/zl/t2i-reid/TBPS-CLIP-master/ckpts/baseline_224_224/CUHK-PEDES", "checkpoint_best.pth"))
        self.model_t1.load_state_dict(checkpoint_t1['model'])  # 加载模型权重
        # self.visual_encoder_t1 = model_t1.image_encode
        for param in self.model_t1.parameters():
            param.requires_grad = False
        self.vision_proj_t1 = nn.Linear(512, embed_dim)
        self.text_proj_t1 = nn.Linear(512, embed_dim)

        # teacher2:   MARS
        self.model_t2 = ALBEF_TCH(config=config_marstch, text_encoder='/workspace/zl/t2i-reid/APTM/data/bert-base-uncased', tokenizer=tokenizer_t2)

        checkpoint_t2 = torch.load("/workspace/zl/t2i-reid/MARS-main/output/cuhk-pedes/train224/checkpoint_best.pth", map_location='cpu')
        state_dict_t2 = checkpoint_t2['model']
        msg = self.model_t2.load_state_dict(state_dict_t2, strict=False)
        print('load MARS(T2) checkpoint from /workspace/zl/t2i-reid/MARS-main/output/cuhk-pedes/train224/checkpoint_best.pth')
        print(msg)
        for param in self.model_t2.parameters():
            param.requires_grad = False

        # self.model_t2 = APTM_Retrieval(config=config_aptm)
        # self.model_t2.load_pretrained('/workspace/zl/t2i-reid/APTM/output/ft_cuhk/checkpoint_best.pth', config_aptm, is_eval=True)
        # # self.visual_encoder_t1 = model_t1.image_encode
        # for param in self.model_t2.parameters():
        #     param.requires_grad = False
        

        # self.vision_proj_t2 = nn.Linear(embed_dim, 512)
        # self.text_proj_t2 = nn.Linear(embed_dim, 512)

        encoders_t = list()
        encoders_t.append(self.model_t1)
        encoders_t.append(self.model_t2)
        self.encoders_t = nn.ModuleList(encoders_t)
        self.embed_dims_t = [encoder_t.embed_dim for encoder_t in self.encoders_t]

        norm_layers_t = list()
        norm_layers_t.append(pca_layers['TBPS'](path_to_pretrained_weights, embed_dim=embed_dim))
        norm_layers_t.append(pca_layers['MARS'](path_to_pretrained_weights, embed_dim=embed_dim))
        self.norm_layers_t = nn.ModuleList(norm_layers_t)
        self.embed_dims_t = [norm_layer_t.dim2 for norm_layer_t in self.norm_layers]


        self.loss_fn_distill = MultiTeacherDistillLoss(
            st=config['st'], tt=config['tt'],
            s=config['strategy'], teachers=config['teachers']
        ).to('cuda')

        self.copy_params()
        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        

        self.mask_ratio = config["mask_ratio"]

        self.dbert2bert_proj = nn.Linear(576, 768)
        nn.init.xavier_uniform_(self.dbert2bert_proj.weight)
        nn.init.zeros_(self.dbert2bert_proj.bias)

        self.norm_db = nn.LayerNorm(768)

    def forward(self, image1, image2, text1, text2, text_tbpsclip, alpha, idx, replace):
        # print('text1.input_ids[0].shape: ', text1.input_ids.shape)
        # print('text2.input_ids[0].shape: ', text2.input_ids.shape)
        # extract image features
        image_embeds = self.visual_encoder(image1) # (bs 49 576)
        # print('image_embeds.shape:  ', image_embeds.shape) 
        # print('image_embeds[0, 0]:  ', image_embeds[0, 0, 0:10]) # -1.0779e+00,  2.7532e-01,  1.3575e-01, -1.8741e-02, -3.4870e-01,
        # print('image_embeds[0, 1]:  ', image_embeds[0, 1, 0:10])
        image_embeds = self.norm_db(self.dbert2bert_proj(image_embeds))
        # print('dbimage_embeds.shape:  ', image_embeds.shape)
        # print('dbimage_embeds[0, 0]:  ', image_embeds[0, 0, 0:10]) # -1.7645e-01, -6.6626e-01, -1.0512e+00, -6.9024e-01, -1.4435e-01,
        # print('dbimage_embeds[0, 1]:  ', image_embeds[0, 1, 0:10])
        # print('image_embeds.shape: ', image_embeds.shape)
        # import sys
        # sys.exit()
        N, L, D = image_embeds.size()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # print('image_feat.shape:  ', image_feat.shape)
        # print('image_feat[0, 0:10]: ', image_feat[0, 0:10])

        # extract text features
        # text_output = self.text_encoder.bert(text2.input_ids, attention_mask=text2.attention_mask,
        #                                      return_dict=True, mode='text')
        text_output = self.text_encoder(text2.input_ids, attention_mask=text2.attention_mask,
                                             return_dict=True)
        
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        # Contrastive loss
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image2)
            # image_embeds_db_m = self.dbert2bert_proj(image_embeds_m)
            image_embeds_m = self.dbert2bert_proj(image_embeds_m)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            # text_output_m = self.text_encoder_m.bert(text2.input_ids, attention_mask=text2.attention_mask,
            #                                          return_dict=True, mode='text')
            text_output_m = self.text_encoder_m(text2.input_ids, attention_mask=text2.attention_mask,
                                                     return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            sim_i2i_m = image_feat_m @ image_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        out_img_t = list()
        out_text_t = list()
        # T1
        out_img_t.append(self.encoders_t[0].encode_image(image1))
        out_text_t.append(self.encoders_t[0].encode_text(text_tbpsclip))
        # T2
        text_output_t2 = self.encoders_t[1].text_encoder.bert(text2.input_ids, attention_mask=text2.attention_mask, mode='text')
        out_img_t.append(self.encoders_t[1].vision_proj(self.encoders_t[1].visual_encoder(image1)[:, 0, :]))
        out_text_t.append(text_output_t2.last_hidden_state[:, 0, :])

        out_img_t = [nn.functional.normalize(o, p=2, dim=-1) for o in out_img_t]
        out_text_t = [nn.functional.normalize(o, p=2, dim=-1) for o in out_text_t]
        out_t = [norm_layer(o) for norm_layer, o in zip(self.norm_layers_t, out_t)]

        with torch.no_grad():
            tch1_image_feat = F.normalize(self.model_t1.encode_image(image1), dim=-1)
            tch1_text_feat = F.normalize(self.model_t1.encode_text(text_tbpsclip), dim=-1)

        t1_proj_image_feat = F.normalize(self.vision_proj_t1(tch1_image_feat), dim=-1)
        t1_proj_text_feat = F.normalize(self.text_proj_t1(tch1_text_feat), dim=-1)

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()
        loss_cl = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4

        loss_distill = self.loss_fn_distill(image_feat, text_feat, t1_proj_image_feat, t1_proj_text_feat)

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        # Relation-aware Learning: Probabilistic Image-Text Matching + Positive Relation Detection
        # Probabilistic Image-Text Matching
        # forward the positve image-text pairs

        # output_pos = self.text_encoder.bert(text2.input_ids,
        #                                     attention_mask=text2.attention_mask,
        #                                     encoder_hidden_states=image_embeds,
        #                                     encoder_attention_mask=image_atts,
        #                                     return_dict=True, 
        #                                     mode='multi_modal')
        # print('text2.input_ids.shape: ', text2.input_ids.shape) # (bs 43)
        # print('text2.attention_mask.shape: ', text2.attention_mask.shape) # (bs 43)
        # print('image_embeds: ', image_embeds.shape) # (bs 49 576)
        # print('image_atts: ', image_atts.shape) # (bs 49)
        # import sys
        # sys.exit()
        output_pos = self.text_encoder_cross.bert(text2.input_ids,
                                        attention_mask=text2.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True,
                                        mode='multi_modal')

        with torch.no_grad():
            bs = image1.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)
            
        # select a negative image for each text
        image_neg_idx = torch.multinomial(weights_t2i, 1).flatten()
        image_embeds_bp_neg = image_embeds[image_neg_idx]

        # select a negative text for each image
        text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()

        text_inputs_ids_neg = text2.input_ids[text_neg_idx]

        text_atts_neg = text2.attention_mask[text_neg_idx]
        
        # forward the negative image-text pairs
        text_inputs_ids_all = torch.cat([text2.input_ids, text_inputs_ids_neg], dim=0)
        # text_attribute_masks_neg=torch.cat([text2.attribute_masks, text2.attribute_masks[text_neg_idx]], dim=0)
        
        text_atts_all = torch.cat([text2.attention_mask, text_atts_neg], dim=0)
        image_embeds_db_all = torch.cat([image_embeds_bp_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        
        output_neg_cross = self.text_encoder_cross.bert(text_inputs_ids_all,
                                            attention_mask=text_atts_all,
                                            encoder_hidden_states=image_embeds_db_all,
                                            encoder_attention_mask=image_atts_all,
                                            return_dict=True,
                                            mode='multi_modal'
                                            )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg_cross.last_hidden_state[:, 0, :]],
                                  dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image1.device)
        loss_pitm = F.cross_entropy(vl_output, itm_labels)

        # MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS MARS

        # loss_attribute = 0
        # count=0
        # for attribute_mask, text_emb in zip(text2.attribute_masks, output_pos.last_hidden_state):
        #     max_attribute_value=torch.max(attribute_mask).to(torch.int32)
            
        #     # P-itm with attribute
        #     averaged_attribute=[]

        #     for i in range(1,max_attribute_value+1):
        #         mask=attribute_mask==i

        #         averaged_attribute.append(text_emb[mask].mean(0))
            
        #     if len(averaged_attribute)>0:
        #         vl_avg_output=self.itm_head(torch.stack(averaged_attribute))
        #         loss_attribute += F.cross_entropy(vl_avg_output, torch.ones(vl_avg_output.size(0), dtype=torch.long, device=image1.device))
        #         count += 1

        # for attribute_mask, text_emb in zip(text_attribute_masks_neg, output_neg_cross.last_hidden_state):
        #     max_attribute_value=torch.max(attribute_mask).to(torch.int32)

        #     # P-itm with attribute
        #     averaged_attribute=[]


        #     for i in range(1,max_attribute_value+1):
        #         mask=attribute_mask==i

        #         averaged_attribute.append(text_emb[mask].mean(0))
            
        #     if len(averaged_attribute)>0:
        #         vl_avg_output=self.itm_head(torch.stack(averaged_attribute))
        #         loss_attribute += F.cross_entropy(vl_avg_output, torch.zeros(vl_avg_output.size(0), dtype=torch.long, device=image1.device))
        #         count += 1
        
        # loss_attribute=loss_attribute/count

        loss_pitm=loss_pitm

        # # Positive Relation Detection
        # prd_output = self.prd_head(output_pos.last_hidden_state[:, 0, :])
        # loss_prd = F.cross_entropy(prd_output, replace)

        # Sensitivity-aware Learning: Masked Language Modeling + Momentum-based Replaced Token Detection
        input_ids = text1.input_ids.clone()
        labels = input_ids.clone()
        mrtd_input_ids = input_ids.clone()
        # Masked Language Modeling
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder_cross.config.vocab_size, targets=labels, probability_matrix=probability_matrix)
        with torch.no_grad():
            logits_m = self.text_encoder_cross_m(input_ids,
                                           attention_mask=text1.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           )
            prediction = F.softmax(logits_m, dim=-1)
        mlm_output = self.text_encoder_cross(input_ids,
                                       attention_mask=text1.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=prediction,
                                       alpha=alpha
                                       )
        loss_mlm = mlm_output.loss
        # # Momentum-based Replaced Token Detection
        # with torch.no_grad():
        #     probability_matrix = torch.full(labels.shape, self.mrtd_mask_probability)
        #     mrtd_input_ids = self.mask(mrtd_input_ids, self.text_encoder.config.vocab_size, probability_matrix=probability_matrix)
        #     # momentum module is used as generator
        #     mrtd_logits_m = self.text_encoder_m(mrtd_input_ids,
        #                                        attention_mask=text1.attention_mask,
        #                                        encoder_hidden_states=image_embeds_m,
        #                                        encoder_attention_mask=image_atts,
        #                                        return_dict=True,
        #                                        return_logits=True,
        #                                        )
        #     weights = F.softmax(mrtd_logits_m, dim=-1)
        #     mrtd_input_ids, mrtd_labels = self.mrtd_mask_modeling(mrtd_input_ids, text1.input_ids, text1.attention_mask, weights)
        # output_mrtd = self.text_encoder.bert(mrtd_input_ids,
        #                                     attention_mask=text1.attention_mask,
        #                                     encoder_hidden_states=image_embeds,
        #                                     encoder_attention_mask=image_atts,
        #                                     return_dict=True,
        #                                     )
        # mrtd_output = self.mrtd_head(output_mrtd.last_hidden_state.view(-1, self.text_width))
        # loss_mrtd = F.cross_entropy(mrtd_output, mrtd_labels.view(-1))

        # # mae loss
        # x, mask, ids_restore = self.visual_encoder(image1, mask_ratio=self.mask_ratio)

        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # x = x + self.visual_decoder_pos_embed[:x.size(1)]

        # x = self.visual_decoder(x, text_output.last_hidden_state)
        # x = self.decoder_pred(x)

        # x = x[:,1:,:]

        # loss_mae = self.forward_loss(image1, x, mask)

        return loss_cl, loss_pitm, loss_mlm, loss_distill
        # return loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd, loss_mae, loss_attribute
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.visual_encoder.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        empty = self.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.idx_queue[:, ptr:] = idxs[:empty].T
            self.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.idx_queue[:, :batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mrtd_mask_modeling(self, mrtd_input_ids, ori_input_ids, attention_mask, weights):
        bs = mrtd_input_ids.size(0)
        weights = weights.view(-1, weights.size(-1))
        pred = torch.multinomial(weights, 1).view(bs, -1)
        pred[:, 0] = self.tokenizer.cls_token_id
        # pad_token_id is 0
        mrtd_input_ids = pred * attention_mask
        mrtd_labels = (pred != ori_input_ids) * attention_mask
        mrtd_labels[mrtd_input_ids == self.tokenizer.pad_token_id] = -100
        mrtd_labels[mrtd_input_ids == self.tokenizer.cls_token_id] = -100
        return mrtd_input_ids, mrtd_labels

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
