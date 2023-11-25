import torch
import matplotlib as plt
from transformers import StoppingCriteria, LogitsProcessor

def get_sep_position(input_ids, sep_id, skip=0):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        sep_position = mask.nonzero()[0, -1].item()
        for _ in range(skip):
            mask[sep_position] = False
            sep_position = mask.nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions


# Stop generation only after generating two EOSs, such as  z <eos> y <eos>
class DoubleEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False
    
    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id] = 0
        return scores
    
def createAccuracyAndLossPlots(train_losses, train_accs, test_accs):
    #Plots training summary results on loss function
    plt.plot(train_losses, color='red', label='train')
    plt.grid(alpha=0.3)
    plt.ylabel('loss',fontweight='bold')
    plt.xlabel('batch',fontweight='bold')
    plt.title("Training Loss Distribution")
    plt.legend()
    plt.show()

    #Plots training summary results on accuracy
    plt.plot(train_accs, color='red', label='train')
    plt.plot(test_accs, color='black', label='test')
    plt.grid(alpha=0.3)
    plt.ylabel('accuracy',fontweight='bold')
    plt.xlabel('batch',fontweight='bold')
    plt.title("Training and Test Accuracy Distribution")
    plt.legend()
    plt.show()