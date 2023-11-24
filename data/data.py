import pandas as pd
import secrets as s
from torch.utils.data import Dataset

class DatasetGenerator(Dataset):
    """
    Can generate both train and test datasets.
    Can also generate tokenzied data for model training.
    """
    
    def __init__(self, tokenizer, path : str, max_len : int):
        super().__init__()
        self.path = path

    def __generateDataset(self, size : int, type : str) -> pd.DataFrame:
        
        data = []
        delimiter_input = '||'
        delimter_output = '####'
        delmiter_problem = '$$$'

        def __generateProblem() -> None:

            i = s.randbelow(100)
            j = s.randbelow(100)

            prod = i*j

            j_digit_1 = i // 10
            j_digit_2 = i % 10

            assert (j_digit_1*i + i*j_digit_2) == prod

            #Convert all numbers to strings and reverse them.
            prod = __addSpaces(str(prod)[::-1])
            z = __addSpaces(str(j_digit_2*i)[::-1] + "+" + str(j_digit_1*i)[::-1])
            input = __addSpaces((str(i)[::-1] + "*" + str(j)[::-1]))

            return [input, z, prod]

        def __addSpaces(str : str) -> str:
            return " ".join(str)
        
        #Generate entries here
        for _ in range(size):
            problem_1 = __generateProblem()
            problem_2 = __generateProblem()

            entry = (" " + __addSpaces(problem_1[0] + delmiter_problem + problem_2[0] + delimiter_input + problem_1[1] + delmiter_problem + problem_2[1] + delimter_output + problem_1[2] + delmiter_problem + problem_2[2]) + " ")

            data.append(entry)

        #Save the dataset
        data = pd.DataFrame(data)
        file_path = self.path + type + r"_dataset.txt"
        data.to_csv(file_path, index = False)
        print(f'Generated raw dataset of type {type} saved at {file_path}')

        return data
    
    '''
    Generates a test dataset
    
    '''
    def generateTestDataset(self, size : int) -> pd.DataFrame:
        
        test_data = self.__generateDataset(self, size, "test")
        self.test_data = test_data
        return test_data
    
    '''
    Generates a train dataset
    
    '''

    def generateTrainDataset(self, size : int) -> pd.DataFrame:
        
        test_data = self.__generateDataset(self, size, "train")
        self.test_data = test_data
        return test_data
    
    '''
    Generates a CoT dataset with all needed features and labels tokenzied for training the various models
    
    '''
    