import os
from tqdm import tqdm
from pathlib import Path
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--llm_name', default='llm', type=str)
args = parser.parse_args()


def generate_code(prompt):
    '''
    Define your function hear
    '''

            
if __name__ == '__main__':
    model_name = args.llm_name
    
    for instruct_file in tqdm(os.listdir('instruction')):
        if '.md' not in instruct_file: continue
    
        code_save = f'llm_module/{model_name}'
        os.makedirs(code_save, exist_ok=True)
        init_file = Path(code_save) / '__init__.py'
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
            
        with open(init_file, 'r') as f:
            init_code = f.read()

        name = instruct_file.split('.')[0]
        with open(Path('llm_module') / instruct_file, 'r') as f:
            instruct = f.read().strip()
                
            match = re.findall(r"should be named (\S+).", instruct)
            if match:
                module_name = ', '.join([item.replace('.', '').strip() for item in match])
                
                code = generate_code(instruct)
                code_file = Path(code_save) / f'{name}.py'
                with open(code_file, 'w') as f:
                    f.write(code)
                    
                init_add_code = f'from .{name} import {module_name}'
                init_code += '\n' + init_add_code

                with open(init_file, 'w') as f:
                    f.write(init_code.strip())    
