import os
from tqdm import tqdm

def generation_code(task):
    '''
    Define your code here.
    '''

if __name__ == '__main__':
    llm_name = 'llm'
    
    solution_save = f'solution/{llm_name}'
    os.makedirs(solution_save, exist_ok=True)
    
    os.makedirs('logs', exist_ok=True)
    
    problem_folder = 'instruction'
    
    for file in tqdm(os.listdir(problem_folder)):
        problem_name = file.replace('.md', '')  
        name = '_'.join(problem_name.split())
        name = name.replace('/', '_')
        
        file_path = problem_folder + file
        with open(file_path, 'r') as f:
            task = f.read()
        
        code = generation_code(task)
        
        with open(f'{solution_save}/{name}.py', 'w') as f:
            f.write(code)
            
            