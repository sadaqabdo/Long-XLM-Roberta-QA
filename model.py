from transformers import XLMRobertaForQuestionAnswering
from transformers.models.longformer import LongformerSelfAttention


class XLMRobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)

        # is index masked or global attention
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())

        return super().forward(
            hidden_states,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


class XLMRobertaLongForQuestionAnswering(XLMRobertaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_xlmroberta.XLMRobertaAttention` object with `XLMRobertaLongSelfAttention`
            layer.attention.self = XLMRobertaLongSelfAttention(config, layer_id=i)
