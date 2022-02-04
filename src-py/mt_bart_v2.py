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

from transformers.trainer_seq2seq import *

from transformers.trainer_pt_utils import nested_detach

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class MultiTaskArgGenModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    dynamic_weight_loss: Optional[torch.FloatTensor] = None
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


logger = logging.get_logger(__name__)

class Seq2TwoSeqTrainer(Seq2SeqTrainer):

    def dummy_func(self):
        return


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": 200,
            "num_beams": 1,
            "synced_gpus": False
        }

        
        generation_inputs = inputs[self.model.main_input_name]

        counter_generated_tokens, conclusion_generated_tokens = self.model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if counter_generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            counter_generated_tokens = self._pad_tensors_to_max_len(counter_generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            outputs = model(**inputs)
            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            
        if self.args.prediction_loss_only:
            return (loss, None, None)

        counter_labels = inputs["counter_labels"]
        if counter_labels.shape[-1] < gen_kwargs["max_length"]:
            counter_labels = self._pad_tensors_to_max_len(counter_labels, gen_kwargs["max_length"])

        return (loss, counter_generated_tokens, counter_labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        
        #We dont perform label smoothing for now
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     loss = self.label_smoother(outputs, labels)
            
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if model.conc_decoder:
            self.log({'eval_conclusionLoss': outputs['conc_loss'].item(),
                    'eval_counterLoss': outputs['count_loss'].item()})

            if model.compute_dynamic_weights:
                self.log({'eval_LogVar1': model.log_vars[0].item(),
                    'eval_modelLogVar2': model.log_vars[1].item()})


        return (loss, outputs) if return_outputs else loss



class BartModelV2(BartPretrainedModel):
    
    main_input_name = "input_ids"

    def __init__(self, config: BartConfig, compute_dynamic_weights=False, conc_loss_weight=0.5, counter_loss_weight=0.5, attention_to_conc=False, conc_decoder=False):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        
        self.encoder = BartEncoder(config, self.shared)
        
        if conc_decoder:
            self.conclusion_decoder = BartDecoder(config, self.shared)
            self.conc_lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        
        self.counter_decoder = BartDecoder(config, self.shared)
        self.count_lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        

        self.conc_loss_weight = conc_loss_weight
        self.counter_loss_weight = counter_loss_weight
        self.attention_to_conc=attention_to_conc
        self.conc_decoder=conc_decoder
        self.compute_dynamic_weights=compute_dynamic_weights

        if self.compute_dynamic_weights:
            #Add a layer for the dynamic loss unit
            #slim.fully_connected(input_lossweights_embedings, 2, activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_lossweights', reuse=False)
            #self.dynamic_loss_unit = nn.utils.weight_norm(nn.Linear(config.d_model, 2))
            self.log_vars = torch.nn.Parameter(torch.zeros(2))

        
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

        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


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


        #if self.compute_dynamic_weights:
            #shared_output=encoder_outputs.last_hidden_state.detach() #because we don't want to backpopagate over the whole encoder
            #Now compute the dynamic weights of the two tasks
            #dynamic_weights = nn.functional.softmax(self.dynamic_loss_unit(shared_output))

        
        if self.conc_decoder:
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
        
        # Second extend the encoder_hidden_states with the conclusion_decoder output_states if attention to conclusion is set to True
        encoder_hidden_states= torch.cat([conclusion_decoder_outputs[0], encoder_outputs[0]], axis=1) if self.attention_to_conc else encoder_outputs[0]
        encoder_attention_mask = torch.cat([conclusion_decoder_attention_mask, attention_mask], axis=1) if self.attention_to_conc else attention_mask
        
        # Third decode the counter
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        counter_decoder_outputs = self.counter_decoder(
            input_ids=counter_decoder_input_ids,
            attention_mask=counter_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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
        if self.conc_decoder:
            
            
            conc_lm_logits = self.conc_lm_head(conclusion_decoder_outputs['last_hidden_state']) + self.final_logits_bias
            conc_lm_loss = loss_fct(conc_lm_logits.view(-1, self.config.vocab_size), conclusion_labels.view(-1)) if conclusion_labels is not None else 0
        
            count_lm_logits = self.count_lm_head(counter_decoder_outputs['last_hidden_state']) + self.final_logits_bias
            count_lm_loss = loss_fct(count_lm_logits.view(-1, self.config.vocab_size), counter_labels.view(-1))  if counter_labels is not None else 0
            
            if self.compute_dynamic_weights:
                losses = torch.stack([conc_lm_loss, count_lm_loss])
                #Retriev loss weights from the dynamic weights unit
                #conc_loss_weight  = dynamic_weights[:,0].mean()
                #counter_loss_weight = dynamic_weights[:,1].mean()
                dtype  = conc_lm_loss.dtype
                device = conc_lm_loss.device
                stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
                #self.is_regression = self.is_regression.to(device).to(dtype)
                coeffs = 1 /(stds**2)
                multi_task_losses = coeffs*losses + torch.log(stds)
                loss = multi_task_losses.mean()

            else:
                conc_loss_weight  = self.conc_loss_weight
                counter_loss_weight = self.counter_loss_weight
                loss = counter_loss_weight * count_lm_loss + conc_loss_weight * conc_lm_loss


            # if self.compute_dynamic_weights:
            #     #compute the loss of the dynamic loss weight unit
            #     sigma = 0.01
            #     weights_loss = (conc_lm_loss / conc_loss_weight + sigma) + (count_lm_loss / counter_loss_weight +sigma)
            

        else:
            count_lm_logits = self.count_lm_head(counter_decoder_outputs['last_hidden_state']) + self.final_logits_bias
            count_lm_loss   = loss_fct(count_lm_logits.view(-1, self.config.vocab_size), counter_labels.view(-1))  if counter_labels is not None else 0
            loss = count_lm_loss

            
        
        #print(count_lm_loss.item(), conc_lm_loss.item())
        
        if not return_dict:
            if self.conc_decoder:
                return counter_decoder_outputs + conclusion_decoder_outputs + encoder_outputs
            else:
                return counter_decoder_outputs + encoder_outputs
                

        
        
        return MultiTaskArgGenModelOutput(
            loss = loss,
            conc_loss= conc_lm_loss if self.conc_decoder else None,
            count_loss= count_lm_loss,

            logits = count_lm_logits,
            conc_lm_logits=conc_lm_logits if self.conc_decoder else None,
            count_lm_logits=count_lm_logits,
            
            counter_last_hidden_state=counter_decoder_outputs.last_hidden_state,
            counter_past_key_values=counter_decoder_outputs.past_key_values,
            counter_decoder_hidden_states=counter_decoder_outputs.hidden_states,
            counter_decoder_attentions=counter_decoder_outputs.attentions,
            counter_cross_attentions=counter_decoder_outputs.cross_attentions,
            
            conclusion_last_hidden_state=conclusion_decoder_outputs.last_hidden_state if self.conc_decoder else None,
            conclusion_past_key_values=conclusion_decoder_outputs.past_key_values if self.conc_decoder else None,
            conclusion_decoder_hidden_states=conclusion_decoder_outputs.hidden_states if self.conc_decoder else None,
            conclusion_decoder_attentions=conclusion_decoder_outputs.attentions if self.conc_decoder else None,
            conclusion_cross_attentions=conclusion_decoder_outputs.cross_attentions if self.conc_decoder else None,
            
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

        conclusion_input_ids = input_ids.clone()
        #To be used later for the counter_decoder
        conclusion_hidden_states =  None
        
        if self.conc_decoder:

            # keep track of which sequences are already finished
            unfinished_sequences = conclusion_input_ids.new(conclusion_input_ids.shape[0]).fill_(1)
            cur_len = conclusion_input_ids.shape[-1]

            #FIRST DECODE A CONCLUSION.....
            this_peer_finished = False  # used by synced_gpus only
            while True:

                if synced_gpus:
                    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                    # The following logic allows an early break if all peers finished generating their sequence
                    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(conclusion_input_ids.device)
                    # send 0.0 if we finished, 1.0 otherwise
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    # did all peers finish? the reduced sum will be 0.0 then
                    if this_peer_finished_flag.item() == 0.0:
                        break

                # prepare model inputs
                model_inputs = self.prepare_inputs_for_conclusion_generation(conclusion_input_ids, **model_kwargs)

                outputs = self.conclusion_decoder(
                    **model_inputs,
                    return_dict=True
                )

                if synced_gpus and this_peer_finished:
                    cur_len = cur_len + 1
                    continue  # don't waste resources running the code we don't need

                next_token_logits = self.conc_lm_head(outputs['last_hidden_state']) + self.final_logits_bias
                next_token_logits = next_token_logits[:, -1, :]

                #We need to save the last conclusion_decoder_hidden_state anyway
                conclusion_hidden_states = outputs.last_hidden_state

                # pre-process distribution
                next_tokens_scores = logits_processor(conclusion_input_ids, next_token_logits)

                # argmax
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                conclusion_input_ids = torch.cat([conclusion_input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                cur_len = cur_len + 1

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

                # stop when each sentence is finished, or if we exceed the maximum length
                if unfinished_sequences.max() == 0 or stopping_criteria(conclusion_input_ids, scores):
                    if not synced_gpus:
                        break
                    else:
                        this_peer_finished = True
        
        

        model_kwargs['past_key_values'] = None #reset the past_key_values
        model_kwargs['past'] = None #reset the past_key_values
        counter_input_ids = input_ids.clone()        

        # keep track of which sequences are already finished
        unfinished_sequences = counter_input_ids.new(counter_input_ids.shape[0]).fill_(1)
        cur_len = counter_input_ids.shape[-1]
        
        #Second DECODE the Counter
        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(counter_input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_counter_generation(counter_input_ids, conclusion_decoder_last_hidden_state=conclusion_hidden_states,  **model_kwargs)
            
            
            outputs = self.counter_decoder(
                **model_inputs,
                return_dict=True
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = self.count_lm_head(outputs['last_hidden_state']) + self.final_logits_bias
            next_token_logits = next_token_logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(counter_input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            counter_input_ids = torch.cat([counter_input_ids, next_tokens[:, None]], dim=-1)
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(counter_input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        return counter_input_ids, conclusion_input_ids


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
            conclusion_decoder_last_hidden_state=None,
            conclusion_decoder_attention_mask=None,
            **kwargs
        ):

            
        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        
        encoder_hidden_states= torch.cat([conclusion_decoder_last_hidden_state, encoder_outputs[0]], axis=1) if self.attention_to_conc else encoder_outputs[0]
        conclusion_decoder_attention_mask = torch.ones(conclusion_decoder_last_hidden_state.shape[0:2]).to(device) if conclusion_decoder_attention_mask is None and self.conc_decoder != False else conclusion_decoder_attention_mask

        attention_mask = torch.cat([conclusion_decoder_attention_mask, attention_mask], axis=1) if self.attention_to_conc else attention_mask


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


    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
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

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        conclusion_input_ids = input_ids.clone()
        #To be used later for the counter_decoder
        conclusion_hidden_states =  None

        # keep track of which sequences are already finished
        unfinished_sequences = conclusion_input_ids.new(conclusion_input_ids.shape[0]).fill_(1)
        cur_len = conclusion_input_ids.shape[-1]


        #FIRST DECODE A CONCLUSION.....
        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(conclusion_input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(conclusion_input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.conclusion_decoder(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = self.conc_lm_head(outputs['last_hidden_state']) + self.final_logits_bias
            next_token_logits = next_token_logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(conclusion_input_ids, next_token_logits)
            next_token_scores = logits_warper(conclusion_input_ids, next_token_scores)

            #We need to save the last conclusion_decoder_hidden_state anyway
            conclusion_hidden_states = outputs.last_hidden_state


            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            conclusion_input_ids = torch.cat([conclusion_input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(conclusion_input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True



        model_kwargs['past_key_values'] = None #reset the past_key_values
        model_kwargs['past'] = None #reset the past_key_values
        counter_input_ids = input_ids.clone()
      

        # keep track of which sequences are already finished
        unfinished_sequences = counter_input_ids.new(counter_input_ids.shape[0]).fill_(1)
        cur_len = counter_input_ids.shape[-1]


        #FIRST DECODE A Counter.....
        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(counter_input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_counter_generation(counter_input_ids, conclusion_decoder_last_hidden_state=conclusion_hidden_states, **model_kwargs)

            # forward pass to get next token
            outputs = self.counter_decoder(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = self.count_lm_head(outputs['last_hidden_state']) + self.final_logits_bias
            next_token_logits = next_token_logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(counter_input_ids, next_token_logits)
            next_token_scores = logits_warper(counter_input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            counter_input_ids = torch.cat([counter_input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(counter_input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True


        return counter_input_ids, conclusion_input_ids

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