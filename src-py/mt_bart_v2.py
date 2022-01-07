from transformers import BartModel, BartTokenizer, BartForConditionalGeneration
from torch import nn
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss, MSELoss


from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import *

@dataclass
class MultiTaskArgGenModelOutput(ModelOutput):

    counter_last_hidden_state: torch.FloatTensor = None
    counter_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    counter_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    counter_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    conclusion_last_hidden_state: torch.FloatTensor = None
    conclusion_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    conclusion_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    conclusion_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    counter_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    conclusion_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
        
class BartModelV2(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.conclusion_decoder = BartDecoder(config, self.shared)
        self.counter_decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        conclusion_decoder_input_ids=None,
        conclusion_decoder_attention_mask=None,
        counter_decoder_input_ids=None,
        counter_decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        conclusion_decoder_inputs_embeds=None,
        counter_decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        conclusion_labels=None,
        counter_labels=None
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if conclusion_decoder_input_ids is None and conclusion_decoder_inputs_embeds is None:
            conclusion_decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            
        if counter_decoder_input_ids is None and counter_decoder_inputs_embeds is None:
            counter_decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # First decode conclusion
        conclusion_decoder_outputs = self.decoder(
            input_ids=conclusion_decoder_input_ids,
            attention_mask=conclusion_decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=conclusion_decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Second extend the encoder_hidden_states with the conclusion_decoder output_states
        encoder_hidden_states= torch.cat([encoder_outputs[0], conclusion_decoder_outputs[0]], axis=1)
        
        conclusion_decoder_attention_mask = torch.ones(conclusion_decoder_input_ids.shape[0:2]) if conclusion_decoder_attention_mask is None else conclusion_decoder_attention_mask
        
        attention_mask = torch.cat([attention_mask, conclusion_decoder_attention_mask], axis=1) if attention_mask != None else None
        
        
        print(conclusion_decoder_outputs[0].shape)
        print(encoder_outputs[0].shape)
        print(encoder_hidden_states.shape)
        
        if attention_mask != None:
            print(attention_mask.shape)
            print(attention_mask)

        # Third decode the counter
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        counter_decoder_outputs = self.decoder(
            input_ids=counter_decoder_input_ids,
            attention_mask=counter_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,#TODO figure out whether this needs to be changed
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=counter_decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return counter_decoder_outputs + conclusion_decoder_outputs + encoder_outputs

        
        
        return MultiTaskArgGenModelOutput(
            counter_last_hidden_state=counter_decoder_outputs.last_hidden_state,
            counter_past_key_values=counter_decoder_outputs.past_key_values,
            counter_decoder_hidden_states=counter_decoder_outputs.hidden_states,
            counter_decoder_attentions=counter_decoder_outputs.attentions,
            counter_cross_attentions=counter_decoder_outputs.cross_attentions,
            
            conclusion_last_hidden_state=conclusion_decoder_outputs.last_hidden_state,
            conclusion_past_key_values=conclusion_decoder_outputs.past_key_values,
            conclusion_decoder_hidden_states=conclusion_decoder_outputs.hidden_states,
            conclusion_decoder_attentions=conclusion_decoder_outputs.attentions,
            conclusion_cross_attentions=conclusion_decoder_outputs.cross_attentions,
            
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model     = BartModelV2.from_pretrained('facebook/bart-base')
    original_bart_model = BartModel.from_pretrained('facebook/bart-base')
    
    #load the weights of the two decoders
    model.conclusion_decoder.load_state_dict(original_bart_model.decoder.state_dict())
    model.counter_decoder.load_state_dict(original_bart_model.decoder.state_dict())
    
    encoding  = tokenizer("I love argument generation so much.")
    outputs   = model(torch.tensor([encoding['input_ids']]), return_dict=True)
    print(outputs.encoder_last_hidden_state)