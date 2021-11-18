
from transformers import BartTokenizer, BartForConditionalGeneration
from torch import nn
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss, MSELoss


from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

@dataclass
class MultiTaskArgGenModelOutput(ModelOutput):

    total_loss: Optional[torch.FloatTensor] = None
    wp_loss : Optional[torch.FloatTensor] = None
    lm_loss : Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    logits: torch.FloatTensor=None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
        

class BartForMultiTaskArgGeneration(BartForConditionalGeneration):
    
    def __init__(self, config, without_classification_head=False, wp_weight=0.5):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels
        self.with_classification_head = not without_classification_head
        self.wp_weight = wp_weight
        
        print(self.with_classification_head)

        if self.with_classification_head:
            #Weak premises extraction head (similar to the question answer task)
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
            self.model._init_weights(self.qa_outputs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if start_positions is not None and end_positions is not None:
            use_cache = False

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        wp_loss = 0
        total_loss=None
        masked_lm_loss = None
        
        start_logits = None
        end_logits = None

        if self.with_classification_head:
            #Compute the logits of the weak premise identification (similar to qa task)
            encoder_output = outputs[1] if not return_dict else outputs.encoder_last_hidden_state
            logits = self.qa_outputs(encoder_output) # the item in index one is supposed to be the encoder_outputs
            start_logits, end_logits = logits.split(1, dim=-1)
            
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits   = end_logits.squeeze(-1).contiguous()
        
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct1 = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct1(start_logits, start_positions)
                end_loss = loss_fct1(end_logits, end_positions)
                wp_loss  = (start_loss + end_loss) / 2


        #Compute the LM loss
        lm_logits = self.lm_head(sequence_output) + self.final_logits_bias
        if labels is not None:
            loss_fct2 = CrossEntropyLoss()
            masked_lm_loss = loss_fct2(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))



            if self.with_classification_head:
                if wp_loss is not None:
                    total_loss = (1- self.wp_weight) * masked_lm_loss + self.wp_weight * wp_loss
                else:
                    total_loss = masked_lm_loss
            else:
                total_loss = masked_lm_loss

        if not return_dict:
            output = (
                start_logits,
                end_logits,
                lm_logits
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskArgGenModelOutput(
            total_loss=total_loss,
            lm_loss=masked_lm_loss,
            wp_loss=wp_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    

    
if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForMultiTaskArgGeneration.from_pretrained('facebook/bart-base')
    encoding = tokenizer("I love argument generation so much.")
    outputs = model(torch.tensor([encoding['input_ids']]), attention_mask=torch.tensor([encoding['attention_mask']]), start_positions=torch.tensor([1]), end_positions=torch.tensor([2]), return_dict=True)
    print(outputs.encoder_last_hidden_state)