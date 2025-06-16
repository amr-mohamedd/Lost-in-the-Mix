# Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text

This repository contains all code, data-processing notebooks, and helper scripts for the paper
> **Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text**  

The project provides a reproducible pipeline that
1. **Builds linguistically grounded or heuristic code-switched (CSW) versions** of three popular reasoning/comprehension benchmarks (Belebele, MMLU, XNLI);
2. **Runs large-scale evaluations** of open-source LLMs on those CSW benchmarks.

---
## Repository layout
```
Lost-in-the-Mix-Evaluating-LLM-Understanding-of-Code-Switched-Text/
├── requirements.txt
├── datasets/
└── scripts/
    ├── Main Experiments/
    │   ├── Linguistically motivated CSW/
    │   └── Non-linguistically motivated code-switching/
    └── Ablations/
```
---
## Quick start
### 1. Clone & create an environment
```bash
$ git clone https://github.com/amr-mohamedd/Lost-in-the-Mix.git
$ cd Lost-in-the-Mix
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt  
```

### 2. Download & unify raw benchmarks
Each sub-experiment ships a notebook `1.prepare_datasets.ipynb`, running it results in one file per benchmark containing the parallel versions of the benchmark of in language.

### 3. Generate code-switched benchmarks
Below is an example for English→Arabic noun-token CSW on Belebele passages.
```bash
python scripts/Main\ Experiments/Linguistically\ motivated\ CSW/noun-token/2.code_switching.py \
  --input_csv datasets/belebele.csv \
  --source_column eng_flores_passage \
  --target_column arb_flores_passage \
  --target_language Arabic \
  --csw_column_name en2ar_noun_token \
  --output_dir outputs/belebele/en2ar_noun_token
```
*Change `target_language`, `source/target_column`, and `output_dir` to cover the other language pairs.*

For the heuristic **ratio-token** baseline, invoke the analogous script:
```bash
python scripts/Main\ Experiments/Non-linguistically\ motivated\ code-switching/2.code_switching.py ...
```
---
<!-- ## Citation
If you find this repository useful, please cite:
```bibtex
``` 