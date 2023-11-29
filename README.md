**Model demo** notebook is stored with instructions and visuals: [here](https://github.com/EnronMusk/multi_implicit_cot/blob/main/demo/model_demo.ipynb)

Simply run the notebook to explore results.

## **Data Format**
The format of training and test datasets follow this format:
``` 
[input 1a] $ [input 1b]||[CoT 1a] $ [CoT 1b] #### [output 1a] $ [output 1b]
[input 2a] $ [input 2b]||[CoT 2a] $ [CoT 2b] #### [output 2a] $ [output 2b]
[input 3a] $ [input 3b]||[CoT 3a] $ [CoT 3b] #### [output 3a] $ [output 3b]
```
Example entry:
``` 
1 7 * 1 3 $ 6 2 * 6 3||1 7 0 + 0 3 1 2 $ 6 5 1 + 0 8 7 0 #### 1 0 2 2 $ 6 3 9 0
```
Each multiplication is delimited by `$`. The `1 7 * 1 3` corresponds to `31 * 71` and `1 7 0 + 0 3 1 2` corresponds to `2130 + 71` and `1 0 2 2 ` corresponds to `2201`

Dataset is dynamically generated and saved: [here](https://github.com/EnronMusk/multi_implicit_cot/tree/main/data)

Referenced paper: [here](https://arxiv.org/pdf/2311.01460.pdf)

Results:
| |||
|----------|----------|
|Teacher|Perplexitity: 1.000465| Test Accuracy: 0.997169| Training Accuracy: 0.999882|
