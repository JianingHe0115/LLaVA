#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

'''
LLaVA的顶层“可训练/可生成”的语言模型类
LlavaLlamaForCausalLM 同时继承 LlamaForCausalLM（HF表征的LLama因果语言模型） 和 LlavaMetaForCausalLM（LLava自己定义的多模态“mixin”）
LlamaForCausalLM：Hugging Face 标准 LLaMA 因果语言模型：Transformer 前向传播、词表预测、训练 loss、generate() 解码生成等
LlavaMetaForCausalLM：LLaVA 自己定义的多模态“Mixin”：图像编码、图文 token/embedding 拼接等
'''
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    #构建语言与多模态模型
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        '''
        post_init() 是 Hugging Face PreTrainedModel 的标准后处理步骤，通常包括：
            初始化尚未从 checkpoint 加载的参数；
            进行权重绑定（如果配置要求，例如输入 embedding 与输出 lm_head 共享权重）；
            执行框架层面的收尾逻辑。
            如果从预训练模型加载，随后 checkpoint 权重会覆盖相应参数；如果从零开始创建模型，这一步尤其重要。
        '''
        self.post_init()

    def get_model(self):
        return self.model

    #训练与普通前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  #文本token的整数编号,[batch_size, text_seq_len]
        attention_mask: Optional[torch.Tensor] = None, #注意力掩码
        position_ids: Optional[torch.LongTensor] = None, #每个token的位置编号
        past_key_values: Optional[List[torch.FloatTensor]] = None, #KVcache 
        inputs_embeds: Optional[torch.FloatTensor] = None, #直接传入的token embedding
        labels: Optional[torch.LongTensor] = None, #训练标签
        use_cache: Optional[bool] = None, #是否返回/使用 KV Cache
        output_attentions: Optional[bool] = None, #是否返回每层 attention 权重
        output_hidden_states: Optional[bool] = None, #是否返回每层隐藏状态
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None, #图片原始尺寸
        return_dict: Optional[bool] = None, #return_dict 控制输出格式：为 True：返回结构化对象 CausalLMOutputWithPast；为 False：返回 tuple。
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        '''
        是否进行图文拼接
        如果调用者没有提前传入 embedding，就由 LLaVA 负责：
            将文本 token 转为 embedding；
            将图片编码为视觉 embedding；
            把图像 embedding 插进文本序列；
            同时对齐 labels、attention_mask、position_ids。
            如果已经传入了 inputs_embeds，则认为调用者已完成这些准备，不再重复处理。
        '''
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            
        #调用原始llama的前向传播
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
    #图文条件下生成文本
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            
        #复用HF的生成框架
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    #让图片跨生成步骤继续传递
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        #从参数中取出LLaVA自己扩展的图像信息
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        #先调用HF原始LLaMA的输入准备逻辑
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

#注册到HF自动加载机制
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
