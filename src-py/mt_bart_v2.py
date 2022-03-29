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
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
        
    #     #We dont perform label smoothing for now
    #     # if self.label_smoother is not None and "labels" in inputs:
    #     #     labels = inputs.pop("labels")
    #     # else:
    #     #     labels = None

    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     # if labels is not None:
    #     #     loss = self.label_smoother(outputs, labels)
            
    #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     if model.conc_decoder:
    #         self.log({'eval_conclusionLoss': outputs['conc_loss'].item(),
    #                 'eval_counterLoss': outputs['count_loss'].item()})

    #         if model.compute_dynamic_weights:
    #             self.log({'eval_LogVar1': model.log_vars[0].item(),
    #                 'eval_modelLogVar2': model.log_vars[1].item()})


    #     return (loss, outputs) if return_outputs else loss

@dataclass
class MultiTaskArgGenModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    dynamic_weight_loss: Optional[torch.FloatTensor] = None
    conc_loss : Optional[torch.FloatTensor] = None
    count_loss : Optional[torch.FloatTensor] = None

    conc_lm_logits:torch.FloatTensor = None
    count_lm_logits:torch.FloatTensor = None
    logits: torch.FloatTensor = None
        
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    conclusion_last_hidden_state: torch.FloatTensor = None
    conclusion_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    conclusion_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    conclusion_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    conclusion_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


logger = logging.get_logger(__name__)

class BartModelV2(BartPretrainedModel):
    
    main_input_name = "input_ids"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"conc_lm_head\.weight", r"count_lm_head\.weight", r"conclusion_decoder\.layers"]

    def __init__(self, config: BartConfig, compute_dynamic_weights=False, conc_loss_weight=0.5, counter_loss_weight=0.5, attention_to_conc=False, conc_decoder=False):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        
        
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        if conc_decoder:
            self.conclusion_decoder = BartDecoder(config, self.shared)
            self.conc_lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        
        
        self.count_lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))


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
    
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.count_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.count_lm_head = new_embeddings

    def prepare_inputs_for_generation(
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
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
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
        decoder_input_ids=None,
        decoder_attention_mask=None,
        conclusion_decoder_input_ids=None,
        conclusion_decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        conclusion_decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        conclusion_labels=None
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
                
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
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

        
        if self.conc_decoder and conclusion_labels != None: #Run conclusion decoder only when training
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
        
        
        # Second decode the counter
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        counter_decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        count_lm_logits = self.count_lm_head(counter_decoder_outputs['last_hidden_state']) + self.final_logits_bias

        loss = None
        conc_lm_loss = None
        count_lm_loss = None
        conc_lm_logits = None

        if labels != None and conclusion_labels != None:

            conc_lm_logits = self.conc_lm_head(conclusion_decoder_outputs['last_hidden_state']) + self.final_logits_bias if self.conc_decoder else None

            #Compute the Loss
            loss_fct = CrossEntropyLoss()
            
            if self.conc_decoder:    
                
                conc_lm_loss = loss_fct(conc_lm_logits.view(-1, self.config.vocab_size), conclusion_labels.view(-1)) if conclusion_labels is not None else None
                count_lm_loss = loss_fct(count_lm_logits.view(-1, self.config.vocab_size), labels.view(-1))  if labels is not None else None
                
                if self.compute_dynamic_weights and conc_lm_loss != 0: #if 0 then its prediction mode
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

            else:
                count_lm_loss = loss_fct(count_lm_logits.view(-1, self.config.vocab_size), labels.view(-1))  if labels is not None else 0
                loss = count_lm_loss

        
        if not return_dict:
            return counter_decoder_outputs + encoder_outputs
                

        return MultiTaskArgGenModelOutput(
            loss = loss,
            conc_loss= conc_lm_loss,
            count_loss= count_lm_loss,

            logits = count_lm_logits,
            #conc_lm_logits=conc_lm_logits if self.conc_decoder else None,
            #count_lm_logits=count_lm_logits,
            
            decoder_hidden_states=counter_decoder_outputs.hidden_states,
            past_key_values=counter_decoder_outputs.past_key_values,
            last_hidden_state=counter_decoder_outputs.last_hidden_state,
            decoder_attentions=counter_decoder_outputs.attentions,
            cross_attentions=counter_decoder_outputs.cross_attentions,
            
            # conclusion_last_hidden_state=conclusion_decoder_outputs.last_hidden_state if self.conc_decoder else None,
            # conclusion_past_key_values=conclusion_decoder_outputs.past_key_values if self.conc_decoder else None,
            # conclusion_decoder_hidden_states=conclusion_decoder_outputs.hidden_states if self.conc_decoder else None,
            # conclusion_decoder_attentions=conclusion_decoder_outputs.attentions if self.conc_decoder else None,
            # conclusion_cross_attentions=conclusion_decoder_outputs.cross_attentions if self.conc_decoder else None,
            
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