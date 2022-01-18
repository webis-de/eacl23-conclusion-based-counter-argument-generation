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
from transformers.generation_utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class MultiTaskArgGenModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    conc_loss : Optional[torch.FloatTensor] = None
    count_loss : Optional[torch.FloatTensor] = None

    conc_lm_logits:torch.FloatTensor = None
    count_lm_logits:torch.FloatTensor = None
    logits: torch.FloatTensor = None
        
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
        
class BartModelV2(BartPretrainedModel):
    def __init__(self, config: BartConfig, conc_loss_weight=0.5, counter_loss_weight=0.5):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.conclusion_decoder = BartDecoder(config, self.shared)
        self.counter_decoder = BartDecoder(config, self.shared)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.conc_lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        self.count_lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        
        self.conc_loss_weight = conc_loss_weight
        self.counter_loss_weight = counter_loss_weight

        self.init_weights()
        
    def get_encoder(self):
        return self.encoder


    def prepare_inputs_for_conclusion_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        
        
        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]


        return {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs[0],
            "past_key_values": past,
            "head_mask": head_mask,
            "head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

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
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if conclusion_labels is not None:
            if conclusion_decoder_input_ids is None and conclusion_decoder_inputs_embeds is None:
                conclusion_decoder_input_ids = shift_tokens_right(
                    conclusion_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        if counter_labels is not None:
            if counter_decoder_input_ids is None and counter_decoder_inputs_embeds is None:
                counter_decoder_input_ids = shift_tokens_right(
                    counter_labels, self.config.pad_token_id, self.config.decoder_start_token_id
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
        conclusion_decoder_outputs = self.conclusion_decoder(
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
        
        conclusion_decoder_attention_mask = torch.ones(conclusion_decoder_input_ids.shape[0:2]).to(device) if conclusion_decoder_attention_mask is None else conclusion_decoder_attention_mask
        
        attention_mask = torch.cat([attention_mask, conclusion_decoder_attention_mask], axis=1) if attention_mask != None else None
        

        # Third decode the counter
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        counter_decoder_outputs = self.counter_decoder(
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

        loss_fct = CrossEntropyLoss()
        
        #print(conclusion_decoder_outputs['last_hidden_state'].size())
        #print(conclusion_decoder_input_ids.size())
        #print(conclusion_labels.size())
        conc_lm_logits = self.conc_lm_head(conclusion_decoder_outputs['last_hidden_state']) + self.final_logits_bias
        conc_lm_loss = loss_fct(conc_lm_logits.view(-1, self.config.vocab_size), conclusion_labels.view(-1)) if conclusion_labels is not None else 0

        count_lm_logits = self.count_lm_head(counter_decoder_outputs['last_hidden_state']) + self.final_logits_bias
        count_lm_loss = loss_fct(count_lm_logits.view(-1, self.config.vocab_size), counter_labels.view(-1))  if counter_labels is not None else 0

        loss = self.counter_loss_weight * count_lm_loss + self.conc_loss_weight * conc_lm_loss
            
        if not return_dict:
            return counter_decoder_outputs + conclusion_decoder_outputs + encoder_outputs

        
        
        return MultiTaskArgGenModelOutput(
            loss = loss,
            conc_lm_logits=conc_lm_logits,
            count_lm_logits=count_lm_logits,
            logits = count_lm_logits,
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
    
    @torch.no_grad()
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:


        conclusion_input_ids = input_ids.clone()
        counter_input_ids = input_ids.clone()
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        conclusion_hidden_states =  None
        
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        #FIRST DECODE A CONCLUSION.....
        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_conclusion_generation(input_ids, **model_kwargs)
            
            
            outputs = self.conclusion_decoder(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            #next_token_logits =  outputs.logits[:, -1, :]
            next_token_logits = self.conc_lm_head(outputs['last_hidden_state']) + self.final_logits_bias
            next_token_logits = next_token_logits[:, -1, :]
            
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
                
                #We need to save the last conclusion_decoder_hidden_state anyway
                conclusion_hidden_states = outputs.hidden_states

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        generated_conclusion_ids = input_ids.clone().detach()
        #Finished decoding the conclusion....       

        
        #we start again with input_ids equal to bos
        input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.config.decoder_start_token_id
        
        #Second DECODE the Counter
        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_counter_generation(input_ids, conclusion_decoder_outputs=conclusion_hidden_states,  **model_kwargs)
            
            
            outputs = self.conclusion_decoder(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            #next_token_logits =  outputs.logits[:, -1, :]
            next_token_logits = self.conc_lm_head(outputs['last_hidden_state']) + self.final_logits_bias
            next_token_logits = next_token_logits[:, -1, :]
            
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

        
    


    def prepare_inputs_for_counter_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            conclusion_decoder_outputs=None,
            conclusion_decoder_attention_mask=None,
            **kwargs
        ):

            
        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_hidden_states= torch.cat([encoder_outputs[0], conclusion_decoder_outputs[0]], axis=1)

        conclusion_decoder_attention_mask = torch.ones(conclusion_decoder_outputs.shape[0:2]).to(device) if conclusion_decoder_attention_mask is None else conclusion_decoder_attention_mask

        attention_mask = torch.cat([attention_mask, conclusion_decoder_attention_mask], axis=1) if attention_mask != None else None


        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]


        return {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "past_key_values": past,
            "head_mask": head_mask,
            "head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


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