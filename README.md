# IESTA - Ineffective-effective Style Transfer for Arguments
This repository is the code base for the paper "**Improving Argument Effectiveness Across Ideologies using Instruction-tuned Large Language Models**" Accepted at EMNLP2024 Findings.


## Data Links

- **Generated effective arguments**: [[link](data/llms_out/new/)]
    - Format: `{ideology}_{model-type}_{n}shot{_steered}.jsonl` (for example: conservative_chatgpt_0shot.jsonl)
      - **ideology**: refers to the target reader's ideology - it has one of these values: _liberal_ or _conservative_
      - **model-type**: refers to the Instruction-tuned LLM used: either _chatgpt_ or _llamav2_ referring to Llama-2-7b-chat-hf
      - **n**: refers to Zero- or One-shot prompting with values _0_ or _1_ resepctively
      - **_steered**: Empty if the LLM is not steered; otherwise it has the value of _steered_meanl0.5_ and  _steered_meanl0.2_ where 0.5 and 0.2 refers to the $lambda$ value
- **Dismissed arguments list**: [[link](data/dismiss_text.txt)]
- **Evaluation**:[[link](data/llm_ideology/)]
  - **LLM**:  
    - [Impersonation Exp] PEW Quiz Results [[raw](data/llm_ideology/pew_quiz_results), [processed](data/llm_ideology/pew_quiz_results_processed)]
    - [Impersonation Exp] Tested Role templates for PEW: [[link](data/llm_ideology/role_templates.json)]
    - [Evaluation] [prompt](data/llm_ideology/evaluation_prompt.txt), [results](data/llms_out/llm_evaluation/) 
  - **Human**: [data](data/human_evaluation/)  
  

## Code Overview

### 1. Installation and Requirements

Install the required packages from [the requirements file](requirements.txt) using `pip install -r requirements.txt`.

The code is a Python package that can be downloaded locally with the following command when you access the code folder locally:
`pip install .`


### 2. General Structure
The main folder contains the following:
- `requirements.txt`: contains all dependencies
- `pyproject.toml`: contains the Python package information (e.g., version, dependencies, metadata)
- `LICENSE`: MIT license
- `iesta`: the python package that contains the main code for the paper (see below for more details)
- `data`: contains the original and generated data
- `scripts` and `notebooks`: execute the `iesta` code


### 3. Mapping Code to Paper Sections

#### PART I: Data Curation

- Paper Section 3. _Data_ and Appendix A:   `package`:**iesta.data**
  - debateorg data exploration: `iesta.data.debateorg.py` [[link](iesta/data/debateorg.py)]
  - From debateorg to IESTA:  `iesta.data.debateorg_processor.py` [[link](iesta/data/debateorg_processor.py)]
  - Iesta to huggingface: `iesta.data.huggingface_loader.py` [[link](iesta/data/huggingface_loader.py)]
  - HuggingFace links: `pending paper acceptance`


#### PART II: Data Genration (ineffective -> effective)

- Paper Section 5.1 _Experiments_: `package`:**iesta.llms**, **iesta.evaluator**
  - Steering the LLM: `iesta.llms.steering.*` [[link](iesta/llms/steering)] - Code taken from the Steering Vectors paper in the publication process.
  - ineffective-effective transfer using Zero- One-shot and steered LLMs:  `iesta.llms.generate.py` [[link](iesta/llms/generate.py) ]
- Paper Section 5.2 _Experiments_ POSTPROCESSING: `iesta.evaluator.generation_processor.py` [[link](iesta/evaluator/generation_processor.py)]  

#### PART III: Evaluation

- Paper Section 5.3 _Automatic Evaluation_: `iesta.evaluator.evaluator.py` [[link](iesta/evaluator/evaluator.py)]
- Paper Section 6.1 _LLM-Based Evaluation_: `notebooks/llm_evaluator.ipynb` [[link](notebooks/llm_evaluator.ipynb)]
- Paper Section 6.2 _Human-Based Evaluation_: `notebooks/human_evaluation.ipynb` [[link](notebooks/human_evaluation.ipynb)]


#### APPENDIX

- Appendix B. _Style Analysis_:  `package`:**iesta.data**, **iesta.stats**
  - Features extraction: `iesta.data.feature_extraction.py` [[link](iesta/data/feature_extraction.py)]
  - Style features significance:`iesta.data.feature_score.py` [[link](iesta/data/feature_score.py)]
  - Style feature score:  `iesta.stats.significance.py` [[link](iesta/stats/significance.py)]

