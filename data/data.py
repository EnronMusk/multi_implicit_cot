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
        delimiter_input = ' || '
        delimter_output = ' #### '
        delmiter_problem = ' $$$ '

        def __generateProblem() -> None:

            i = s.randbelow(100)
            j = s.randbelow(100)

            j_digit_1 = j // 10
            j_digit_2 = j % 10

            z1 = i*j_digit_2 #First intermediate sum.
            z2 = j_digit_1*i*10 #Second intermediate sum.

            prod = i*j
            
            assert z1 + z2 == prod

            #Convert all numbers to strings then reverse them. Also ensure appropriate length using zfill. z is our CoT.
            i = str(i).zfill(2)
            j = str(j).zfill(2)
            z1 = str(z1).zfill(3)
            z2 = str(z2).zfill(4)
            prod = str(prod).zfill(4)

            input = __addSpaces((i[::-1] + "*" + j[::-1]))
            z = __addSpaces(z1[::-1] + "+" + z2[::-1])
            prod = __addSpaces(prod[::-1])

            return [input, z, prod]

        def __addSpaces(str : str) -> str:
            return " ".join(str)
        
        #Generate entries here by assembling each problem into a full entry.
        for _ in range(size):
            problem_1 = __generateProblem()
            problem_2 = __generateProblem()

            entry = problem_1[0] + delmiter_problem + problem_2[0] + delimiter_input + problem_1[1] + delmiter_problem + problem_2[1] + delimter_output + problem_1[2] + delmiter_problem + problem_2[2]

            data.append(entry)

        #Save the dataset
        data = pd.DataFrame(data)
        file_path = self.path + r"\data\raw_" + type + r"_dataset.txt"
        data.to_csv(file_path, index = False, header = False)
        print(f'Generated raw {type} dataset saved at {file_path}')


        return data
    
    def generateTestDataset(self, size : int) -> None:
        '''
        Generates a test dataset of a given size and stores it in the DG object.
    
        '''
        
        test_data = self.__generateDataset(size = size, type = "test")
        self.test_data = test_data
        return test_data

    def generateTrainDataset(self, size : int) -> None:
        '''
        Generates a train dataset of a given size  and stores it in the DG object.
    
        '''

        test_data = self.__generateDataset(size = size, type = "train")
        self.test_data = test_data
        return test_data
    
    