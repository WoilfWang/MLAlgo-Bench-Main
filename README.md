# MLAlgo-Bench-Main

This is the official repo of the EMNLP2025 findings paper:  **MLAlgo-Bench: Can Machines Implement Machine Learning Algorithms?**

MLAlgo-Bench is a novel benchmark designed to evaluate whether LLMs can autonomously generate ML code for sophisticated, human-designed algorithms or solutions. It includes two challenging tasks:

1. Setting1: Generating code for machine learning algorithms, including both traditional ML and modern deep learning methods.
2. Setting2: Providing solution sketches and writing ML code for practical tasks such as Kaggle competitions.

It is unique in focusing on the challenges of interpreting intricate human instructions and producing multi-step, high-complexity code, offering a rigorous test for current Large Language Model (LLM) capabilities.


## Usage

### Setting 1

#### Traditional ML Algorithms

If you want to evaluate the ability of LLMs to generate code for **traditional machine learning algorithms**, you should first generation the corresponding code. 

Note that you should not forget to modify the code generation function in `setting 1/ml/generation.py`.

```bash
cd setting 1/ml
python generate.py --instruct_file instruction/reduction --answer llm_answer/gpt/reduction
python generate.py --instruct_file instruction/regression --answer llm_answer/gpt/regression
python generate.py --instruct_file instruction/classification --answer llm_answer/gpt/classification
python generate.py --instruct_file instruction/cluster --answer llm_answer/gpt/cluster
```

After all the code has been generated, run the evaluation script to assess the generated code. 

```bash
sh run_evaluation.sh
```

#### Deep Learning Algorithms

Similarly, after you finish modifying the generated code in `setting 1/dl/generation.py`, you first need to generate the code corresponding to each algorithm.

```bash
cd setting 1/dl
python generation.py --llm_name llm
```

Then, download the dataset from google drive https://drive.google.com/drive/folders/1qUWlvXX-VCZL9p3vn70a1svfOiZjsYCk?usp=drive_link. Put the downloaded dataset into the folder of `setting 1/dl/dataset`.

Finial, run the evaluation script to assess the generated code.

```bash
sh run_evalution.sh llm
```

### Setting 2

First, modify the function in `/setting2/generate.py` that generates code, so it produces the code corresponding to all tasks.

```bash
cd setting 2
python generation.py
```

Then, download the dataset from google drive https://drive.google.com/drive/folders/13cDdEZ0tS_bWeWWc1_bZdWik_tinpcM1?usp=drive_link. Put the downloaded dataset into the folder of `setting 2/dataset`. 

After all the code has been generated, run the evaluation script to assess the generated code. 

This script can evaluate the pass rate of tasks.

```bash
sh run_evaluation.sh solution/gpt
```

## Will Come Soon

1. The complete evaluation code and dataset for Setting 2 is currently at a completion level of 28 / 97. Continuous updates will be provided.
2. The evaluation framework for Setting 1 will be restructured by integrating ML and DL evaluations into a unified codebase, improving the usability of the framework.
