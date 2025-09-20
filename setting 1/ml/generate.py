import argparse
import os
from pathlib import Path
import re
from tqdm import tqdm
import time

Instruction = '''
Given an instruction about a machine learning algorithm, implement the relevant code based on this instruction.
You should implement the algorithm by using Python, Numpy or Scipy from scratch. You can't use any functions or classes from scikit-learn.
You only need to implement the algorithm module, and you don't need to generate test cases.
Just output the code of the algorithm. Don't output the description of the algorithm.
Prepend your answer with code:\n.
'''.strip()


def generate_code(task_prompt):
    '''
    Define your code hear
    '''
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruct_file', type=str, default='problem-detailed/reduction')
    parser.add_argument('--code_save', type=str, default='answer/gpt/reduction')
    args = parser.parse_args()
    return args

args = get_args()


os.makedirs(args.code_save, exist_ok=True)
init_file = Path(args.code_save) / '__init__.py'
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        pass

with open(init_file, 'r') as f:
    init_code = f.read()

for item in tqdm(os.listdir(args.instruct_file)[:]):
    if 'instruct' not in item: continue
    name = item.split('.')[0]
    with open(Path(args.instruct_file) / item, 'r') as f:
        instruct = f.read().strip()
        
        match = re.search(r'The module should be named (.*).', instruct)
        if match:
            module_name = match.group(1).replace('.', '').split()[0]
        
        code = generate_code(instruct)
        code_file = Path(args.code_save) / f'{name}.py'
        with open(code_file, 'w') as f:
            f.write(code)
            
        init_add_code = f'from .{name} import {module_name}'
        init_code += '\n' + init_add_code

        with open(init_file, 'w') as f:
            f.write(init_code.strip())
    
    time.sleep(2)
        