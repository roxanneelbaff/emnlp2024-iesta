# IESTA - Ineffective-effective Style Transfer for Arguments
This repository is the code base for the paper "**Improving Argument Effectiveness Across Ideologies using Instruction-tuned Large Language Models**" submitted to ARR JUNE 2024.


## Data Links

- **Generated effective arguments**: [[link](https://anonymous.4open.science/r/iesta-june2024/data/llms_out/new/)]
  

  > for each each jsonl we share 50 generated arguments (a total of 600 arguments). 
  
  > The rest will be shared upon acceptance.
    
    - Format: `{ideology}_{model-type}_{n}shot{_steered}.jsonl` (for example: conservative_chatgpt_0shot.jsonl)
      - **ideology**: refers to the target reader's ideology - it has one of these values: _liberal_ or _conservative_
      - **model-type**: refers to the Instruction-tuned LLM used: either _chatgpt_ or _llamav2_ referring to Llama-2-7b-chat-hf
      - **n**: refers to Zero- or One-shot prompting with values _0_ or _1_ resepctively
      - **_steered**: Empty if the LLM is not steered; otherwise it has the value of _steered_meanl0.5_ and  _steered_meanl0.2_ where 0.5 and 0.2 refers to the $lambda$ value
- **Dismissed arguments list**: [[link](https://anonymous.4open.science/r/iesta-june2024/data/dismiss_text.txt)]
- **Evaluation**:[[link](https://anonymous.4open.science/r/iesta-june2024/data/llm_ideology/)]
  - **LLM**:  
    - [Impersonation Exp] PEW Quiz Results [[raw](https://anonymous.4open.science/r/iesta-june2024/data/llm_ideology/pew_quiz_results), [processed](https://anonymous.4open.science/r/iesta-june2024/data/llm_ideology/pew_quiz_results_processed)]
    - [Impersonation Exp] Tested Role templates for PEW: [[link](https://anonymous.4open.science/r/iesta-june2024/data/llm_ideology/role_templates.json)]
    - [Evaluation] [prompt](https://anonymous.4open.science/r/iesta-june2024/data/llm_ideology/evaluation_prompt.txt), [results](https://anonymous.4open.science/r/iesta-june2024/data/llms_out/llm_evaluation/) 
  - **Human**: [data](https://anonymous.4open.science/r/iesta-june2024/data/human_evaluation/)  
  

## Code Overview

### 1. Installation and Requirements

Install the required packages from [the requirements file](https://anonymous.4open.science/r/iesta-june2024/requirements.txt) using `pip install -r requirements.txt`.

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
  - debateorg data exploration: `iesta.data.debateorg.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/data/debateorg.py)]
  - From debateorg to IESTA:  `iesta.data.debateorg_processor.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/data/debateorg_processor.py)]
  - Iesta to huggingface: `iesta.data.huggingface_loader.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/data/huggingface_loader.py)]
  - HuggingFace links: `pending paper acceptance`


#### PART II: Data Genration (ineffective -> effective)

- Paper Section 5.1 _Experiments_: `package`:**iesta.llms**, **iesta.evaluator**
  - Steering the LLM: `iesta.llms.steering.*` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/llms/steering)] - Code taken from the Steering Vectors paper in the publication process.
  - ineffective-effective transfer using Zero- One-shot and steered LLMs:  `iesta.llms.generate.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/llms/generate.py) ]
- Paper Section 5.2 _Experiments_ POSTPROCESSING: `iesta.evaluator.generation_processor.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/evaluator/generation_processor.py)]  

#### PART III: Evaluation

- Paper Section 5.3 _Automatic Evaluation_: `iesta.evaluator.evaluator.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/evaluator/evaluator.py)]
- Paper Section 6.1 _LLM-Based Evaluation_: `notebooks/llm_evaluator.ipynb` [[link](https://anonymous.4open.science/r/iesta-june2024/notebooks/llm_evaluator.ipynb)]
- Paper Section 6.2 _Human-Based Evaluation_: `notebooks/human_evaluation.ipynb` [[link](https://anonymous.4open.science/r/iesta-june2024/notebooks/human_evaluation.ipynb)]


#### APPENDIX

- Appendix B. _Style Analysis_:  `package`:**iesta.data**, **iesta.stats**
  - Features extraction: `iesta.data.feature_extraction.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/data/feature_extraction.py)]
  - Style features significance:`iesta.data.feature_score.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/data/feature_score.py)]
  - Style feature score:  `iesta.stats.significance.py` [[link](https://anonymous.4open.science/r/iesta-june2024/iesta/stats/significance.py)]

