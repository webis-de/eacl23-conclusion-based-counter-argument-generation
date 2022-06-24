from transformers import BartTokenizer, BartForConditionalGeneration
from torch import nn
import torch
import transformers
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss, MSELoss

import os

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


class MultiTaskBart(BartForConditionalGeneration):
    def __init__(self, config, bart_encoder=None, classification_head=None):
        super().__init__(config)
        
        if classification_head == None:
            self.classification_head = BartClassificationHead(
                config.d_model*2,
                config.d_model*2,
                2,
                config.classifier_dropout,
            )

            self.model._init_weights(self.classification_head.dense)
            self.model._init_weights(self.classification_head.out_proj)

        else:
            self.classification_head = classification_head
    

        if bart_encoder != None:
            self.model.encoder=bart_encoder


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
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        #print(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        # modified: here we change the output of encoder_last_hidden_state to decoder's last hidden state
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs[0],
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class OurModel(transformers.PreTrainedModel):
    
    def __init__(self, counter_claim_model, counter_argument_model, conclusion_model, encoder, stance_classifier_head, teacher_model_path=None):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.stance_classifier_head = stance_classifier_head
        self.counter_claim_model    = counter_claim_model
        self.counter_argument_model = counter_argument_model
        self.conclusion_model = conclusion_model

    @classmethod
    def create(cls, model_name, model_config, teacher_model_path=None):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        
        #create the first encoder-decoder for generating counter-claim
        counter_claim_model = MultiTaskBart.from_pretrained(model_name, config=model_config)
        
        #extract the encoder and classification head to share them with the other model
        encoder = counter_claim_model.model.encoder
        stance_classifier_head = counter_claim_model.classification_head


        #create the second encoder-decoder for generation counter-argument
        counter_argument_model = MultiTaskBart.from_pretrained(model_name, config=model_config, bart_encoder=encoder, classification_head=stance_classifier_head)

        #create the second encoder-decoder for generation conclusion
        conclusion_model = MultiTaskBart.from_pretrained(model_name, config=model_config, bart_encoder=encoder, classification_head=stance_classifier_head)



        return cls(counter_claim_model, counter_argument_model, conclusion_model, encoder, stance_classifier_head, teacher_model_path=teacher_model_path)
    
    @classmethod
    def load(cls, model_folder, model_type, model_config):
        """
        This loads a MultitaskModel using the model class and config objects
        from single-task models.
        """

        #create the first encoder-decoder for generating counter-claim
        counter_claim_model = MultiTaskBart.from_pretrained(f"{model_folder}/counter_claim_model", config=model_config)
        
        #extract the encoder and classification head to share them with the other model
        encoder = counter_claim_model.model.encoder
        stance_classifier_head = counter_claim_model.classification_head


        #create the second encoder-decoder for generation counter-argument
        counter_argument_model = MultiTaskBart.from_pretrained(f"{model_folder}/counter_argument_model", config=model_config)

        #create the second encoder-decoder for generation counter-argument
        conclusion_model = MultiTaskBart.from_pretrained(f"{model_folder}/conclusion_model", config=model_config)


        return cls(counter_claim_model, counter_argument_model, conclusion_model, encoder, stance_classifier_head)
    
    def save_model(self, model_folder):

        os.makedirs(model_folder + '/counter_claim_model', exist_ok=True)
        os.makedirs(model_folder + '/counter_argument_model', exist_ok=True)
        os.makedirs(model_folder + '/conclusion_model', exist_ok=True)

        self.counter_claim_model.config.to_json_file(model_folder + "/counter_claim_model/config.json")
        torch.save(self.counter_claim_model.state_dict(), model_folder + "/counter_claim_model/pytorch_model.bin")

        self.counter_argument_model.config.to_json_file(model_folder + "/counter_argument_model/config.json")
        torch.save(self.counter_argument_model.state_dict(), model_folder + "/counter_argument_model/pytorch_model.bin")

        self.conclusion_model.config.to_json_file(model_folder + "/conclusion_model/config.json")
        torch.save(self.conclusion_model.state_dict(), model_folder + "/conclusion_model/pytorch_model.bin")



    def generate_conclusion(self, input_ids, attention_mask, gen_kwargs, logits_processor=None):
       
        return self.conclusion_model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor = logits_processor,
            **gen_kwargs,
        )


    def generate_counter_claim(self, input_ids, attention_mask, gen_kwargs, logits_processor=None):
       
        return self.counter_claim_model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor = logits_processor,
            **gen_kwargs,
        )


    def generate_counter_argument(self, input_ids, attention_mask, gen_kwargs, logits_processor = None):
       
        return self.counter_argument_model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor = logits_processor,
            **gen_kwargs,
        )

    def generate_counter(self, input_ids, attention_mask, claim_gen_kwargs, argument_gen_kwargs, logits_processor=None):

        generated_calims = self.counter_claim_model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor = logits_processor,
            **claim_gen_kwargs,
        )

        generated_arguments = self.counter_argument_model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor = logits_processor,
            **argument_gen_kwargs,
        )

        return generated_calims, generated_arguments