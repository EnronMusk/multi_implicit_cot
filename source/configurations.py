from transformers import PretrainedConfig
import os

class EmulatorConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        softmax_temperature=0.05,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        self.softmax_temperature = softmax_temperature
        super().__init__(**kwargs)

class StudentConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        super().__init__(**kwargs)

class TeacherConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        super().__init__(**kwargs)

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
