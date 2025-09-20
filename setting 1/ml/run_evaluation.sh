llm_name=${1:-llm} 

python evaluation/test_class --evaluate_llm ${llm_name}
python evaluation/test_cluster --evaluate_llm ${llm_name}
python evaluation/test_reduction --evaluate_llm ${llm_name}
python evaluation/test_regress --evaluate_llm ${llm_name}