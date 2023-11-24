<<<<<<< HEAD
from transformers import PretrainedConfig
i = 0
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
=======
from transformers import PretrainedConfig

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
>>>>>>> 850a79755722d2031ad88091480c2062961bb66f
        super().__init__(**kwargs)