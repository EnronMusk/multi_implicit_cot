import os
import sys
import math
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')

#For safe imports
file_directory = os.getcwd()
parent_directory = os.path.dirname(file_directory)
sys.path.insert(False, parent_directory)

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import tqdm
import logging
import inspect

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
from torch.utils.data import DataLoader

from data.data import DatasetHandler
from data.data import CoTDataCollator
from data.data import extractAnswer

from source.configurations import EmulatorConfig
from source.utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor, createAccuracyPlot, createLossPlot
from source.gpt2_implicit import GPT2LMHeadImplicitModel

from source.Teacher import Teacher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

class Emulator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size
        
        #We turn each layer into a verticle set of layers.
        self.verticle_model = nn.ModuleList([nn.Sequential(
             nn.Linear(2*hidden_size, 4*hidden_size),
             nn.ReLU(),
             nn.Linear(4*hidden_size, hidden_size),
             ) for _ in range(num_layers)])

        self.mixture_components = nn.Embedding(config.mixture_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, \
                batch_first=False, dropout=0, bidirectional=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, input_ids, requires_backward=False):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        input_ids = input_ids[:, :sep_positions.max()+1]
        outputs = self.base_model.forward(mode='forward_emulator', \
                input_ids=input_ids, \
                positions_to_take=sep_positions, \
                softmax_temperature=self.config.softmax_temperature, \
                requires_backward=requires_backward, \
                rnn=self.rnn, \
                mlps=self.verticle_model, \
                mixture_components=self.mixture_components, \
                key_proj=self.key_proj, \
                query_proj=self.query_proj, \
                out_proj=self.out_proj)
        emulated_teacher_states = outputs.f_h_cs
        return emulated_teacher_states

    def computeLoss(self, input_ids, teacher_states):
        emulated_teacher_states = self.forward(input_ids=input_ids, requires_backward=True)
        batch_size = input_ids.shape[0]

        loss_function = nn.MSELoss(reduction='none')
        loss = 0
        for teacher_state, emulated_teacher_state in zip(teacher_states, emulated_teacher_states):
            loss += loss_function(teacher_state, emulated_teacher_state).sum(-1) / 2
        loss = loss.mean()

        #Calculate training accuracy
        correct_predictions = (teacher_state == emulated_teacher_states).sum()
        total = teacher_state.count()
        train_accuracy = correct_predictions / total


        outputs = CausalLMOutputWithCrossAttentions(loss=loss)
        outputs.total_loss = loss * batch_size
        outputs.train_accuracy = train_accuracy
        return outputs

    def __generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True) -> list:
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]

        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None

        if sep_positions.eq(sep_positions[0]).all():
            input_ids = input_ids[:, :sep_positions[0]+1]
            beam_output = self.base_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
            beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i:i+1]
                sep_positions_i = sep_positions[i:i+1]
                input_ids_i = input_ids_i[:, :sep_positions_i+1]
                beam_output_i = self.base_model.generate(
                    input_ids=input_ids_i,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
                beam_output.append(beam_output_i)
        return beam_output

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = EmulatorConfig.from_pretrained(pretrained_path)
        model = Emulator(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory) -> None:
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

    @torch.no_grad() #Freeze gradients.
    def evaluate(self, dataloader : DataLoader, ctx, teacher : Teacher):
        '''
        Calculates accuracy metrics on test data and can also generate predictions.
        '''
        self.base_model.eval() #Freeze loss function, gradients etc.

        self.total_instances = 0
        self.total_tokens = 0
        self.total_correct_tokens = 0
        self.total_correct = 0
        self.total_loss = 0

        self.__sub_iteration = 0

        for batch in tqdm.tqdm(dataloader):
            input_ids_cot = batch['input_ids_cot'].to(device)
            labels = batch['input_ids_nocot'].to(device)
            # Remove answer part
            sep_positions = get_sep_position(input_ids_cot, self.tokenizer.eos_token_id)
            input_ids = input_ids_cot[:, :sep_positions.max()+1]
            batch_size = input_ids_cot.shape[0]
            with ctx:
                teacher_states = teacher.extract_states(input_ids=input_ids_cot, delta=self.config.delta, subset=self.config.subset)
                outputs = self.compute_loss(input_ids=input_ids_cot, teacher_states=teacher_states)
                loss = outputs.loss
            total_loss += outputs.total_loss.item()
            total_instances += batch_size

            loss = total_loss / total_instances

            # Generate
            beam_output = self.__generate(
                input_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens,
            )
            for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_cot, beam_output)):
                self.__sub_iteration += 1
                sep_position = sep_positions[i].item()
                tgt = input_ids_all_i[sep_position+1:]
                tgt_text = self.tokenizer.decode(tgt, skip_special_tokens=True)
                pred_text = self.tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)

                if i == 0 and self.__sub_iteration <= 100: # to limit spam of prediction examples.
                    print (f'Input: {self.tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                    print (f'Target: {tgt_text}')
                    print (f'Predicted: {pred_text}')
                    print ('')

        loss = self.total_loss / self.total_tokens
        return outputs.training_accuracy, loss
    
    def predict(self, custom_data_handler : DatasetHandler) -> None:
        '''
        Used for custom test cases for fun. You can create custom test cases using the generateDataset using a DatasetHandler.
        '''

        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

        teacher = self.to(device).to(ptdtype)
        # Load data
        tokenizer = teacher.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        custom_data_handler = DataLoader(custom_data_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        self.evaluate(custom_data_handler, ctx)


    def train(self, train_handler : DatasetHandler, test_handler : DatasetHandler, teacher : Teacher, limit : float) -> None:
        '''
        Trains the model and automatically evaluates. 
        @limit hard caps the desired accuracy to stop training early if the threshold is meet.
        '''
        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

        # Create Emulator 
        emulator  = self.to(device).to(ptdtype)

        # Load data
        tokenizer = teacher.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        train_dataloader = DataLoader(train_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=True)
        val_dataloader = DataLoader(test_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        # Create Optimizer
        trainable_params = list(emulator.parameters())
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(trainable_params, lr = self.config.eta, **extra_args)

        self.base_model.train() #Put model in training mode

        for p in teacher.parameters():
            p.requires_grad = False

        train_losses = []

        train_accs = []

        # Train
        iteration = 0
        for batch in tqdm.tqdm(train_dataloader):
            self.base_model.train()
        

            input_ids_cot_only  = batch['input_ids_cot'].to(device)
            input_w_nocot  = batch['input_ids_nocot'].to(device)
            with ctx:
                with torch.no_grad():
                    teacher_states = teacher.extractStates(input_ids=input_ids_cot_only, delta=self.config.delta, subset=self.config.subset)
                outputs = emulator.computeLoss(input_ids=input_w_nocot, teacher_states=teacher_states)
            loss = outputs.loss
            train_accuracy = outputs.train_accuracy

            #Stop training early to save resources.
            if train_accuracy > limit:
                print(f"Accuracy limit reached, stopping training at training accuracy: {train_accuracy:.6f}.")
                break

            loss.backward() #Calculates gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step() #Subtracts gradients.
            optimizer.zero_grad() #Set gradients to zero.

            if iteration % 250 == 0:
                print (f"Step: {iteration}. Loss: {loss:.6f}. Training Accuracy: {train_accuracy:.6f}.")
            iteration += 1

            train_losses.append(loss.item())
            train_accs.append(train_accuracy)

        accuracy, loss = evaluate(val_dataloader, ctx, teacher)

        print (f'Loss: {loss:.6f}; Accuracy: {accuracy:.6f}; Training Accuracy: {train_accuracy:.6f}.')
        emulator.save_pretrained(os.path.join(train_handler.path, f'emulator_model'))

        createLossPlot(train_losses) #Plots the lsos and accuracy information over batches, so we can gage training performance/overfitting.
        createAccuracyPlot(train_accs)