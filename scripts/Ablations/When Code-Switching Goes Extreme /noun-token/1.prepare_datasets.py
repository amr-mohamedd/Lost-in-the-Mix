from datasets import load_dataset
import pandas as pd
import tqdm
from datasets import load_dataset
from tqdm.notebook import tqdm
import os
import ast
tqdm.pandas()
# Belebele
subsets = ['eng_Latn', 'fra_Latn', 'deu_Latn', 'arb_Arab', 'zho_Hans']
datasets = [(load_dataset("facebook/belebele", subset), subset) for subset in subsets if subset != 'All']

combined_df = pd.DataFrame()
for dataset, subset in datasets:
    if subset == "eng_Latn":
        df = pd.DataFrame(dataset['test']).sort_values(by='link').reset_index(drop=True)[['flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4', 'correct_answer_num']]
        df[f"{subset.split('_')[0]}_flores_passage"] = df['flores_passage']
        df[f"{subset.split('_')[0]}_question"] = df['question']
        df.drop(columns=['flores_passage', 'question'], inplace=True)
        combined_df = pd.concat([combined_df, df], axis=1)
    else:
        df = pd.DataFrame(dataset['test']).sort_values(by='link').reset_index(drop=True)[['flores_passage', 'question']]
        df[f"{subset.split('_')[0]}_flores_passage"] = df['flores_passage']
        df[f"{subset.split('_')[0]}_question"] = df['question']
        df.drop(columns=['flores_passage', 'question'], inplace=True)
        combined_df = pd.concat([combined_df, df], axis=1)
combined_df.reset_index(drop=True, inplace=True)
os.makedirs('datasets', exist_ok=True)
combined_df.to_csv('./datasets/belebele.csv', index=False)
# MMLU
# Load English questions from the MMLU dataset
eng_mmlu = load_dataset("cais/mmlu", "all")['test'].to_pandas()
# Load the non-English versions (questions) from other provided splits.
fra_mmlu = load_dataset("openai/MMMLU", "FR_FR")['test'].to_pandas()
deu_mmlu = load_dataset("openai/MMMLU", "DE_DE")['test'].to_pandas()
arb_mmlu = load_dataset("openai/MMMLU", "AR_XY")['test'].to_pandas()
zho_mmlu = load_dataset("openai/MMMLU", "ZH_CN")['test'].to_pandas()

combined_mmlu = pd.DataFrame()
combined_mmlu['eng_mmlu_question'] = eng_mmlu['question']
combined_mmlu['fra_mmlu_question'] = fra_mmlu['Question']  # Note: column name might be different (capitalized).
combined_mmlu['deu_mmlu_question'] = deu_mmlu['Question']
combined_mmlu['arb_mmlu_question'] = arb_mmlu['Question']
combined_mmlu['zho_mmlu_question'] = zho_mmlu['Question']

combined_mmlu.reset_index(drop=True, inplace=True)
combined_mmlu.to_csv('datasets/mmlu.csv', index=False)
print("MMLU dataset saved to 'datasets/mmlu.csv'.")
# XNLI
xnli = load_dataset("facebook/xnli", "all_languages")['test'].to_pandas()

# For XNLI, the premise and hypothesis columns may be stored as strings representing dictionaries.
# Convert them to dictionaries (if needed).
xnli['premise'] = xnli['premise'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
xnli['hypothesis'] = xnli['hypothesis'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Sometimes the hypothesis field contains dictionaries with separate 'language' and 'translation' keys.
# Convert such dictionaries into a more usable format.
def process_translation(d):
    if isinstance(d, dict):
        if 'language' in d and 'translation' in d:
            return dict(zip(d['language'], d['translation']))
    return d

xnli['hypothesis'] = xnli['hypothesis'].apply(process_translation)

# Define a language map to extract specific language translations.
language_map = {
    'en': 'eng',
    'fr': 'fra',
    'de': 'deu',
    'ar': 'arb',
    'zh': 'zho'
}

# Extract both the premise and hypothesis for each language.
for lang, prefix in language_map.items():
    xnli[f'{prefix}_premise'] = xnli['premise'].apply(lambda d: d.get(lang, None) if isinstance(d, dict) else None)
    xnli[f'{prefix}_hypothesis'] = xnli['hypothesis'].apply(lambda d: d.get(lang, None) if isinstance(d, dict) else None)

xnli.reset_index(drop=True, inplace=True)
xnli.to_csv('datasets/xnli.csv', index=False)
print("XNLI dataset saved to 'datasets/xnli.csv'.")