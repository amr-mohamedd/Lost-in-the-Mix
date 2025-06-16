#!/usr/bin/env python3
"""
Randomized code-switching script: selects ~20% of words in each English sentence at random,
replaces them with placeholders, then invokes Claude to fill placeholders from a parallel text.
"""
import argparse
import pandas as pd
from tqdm import tqdm
import os
import logging
import random
from claude import invoke_claude

# Enable progress bars for pandas operations
tqdm.pandas()


def insert_placeholders_random(text: str, rate: float = 0.2) -> str:
    """
    Split the sentence into words, randomly pick ~rate fraction of them,
    and replace each picked word with a placeholder '#######'.
    """
    words = text.split()
    count = max(1, int(len(words) * rate))
    indices = random.sample(range(len(words)), count)
    for i in indices:
        words[i] = '#######'
    return ' '.join(words)


def code_switch_text(placeholder_text: str, target_text: str, target_language: str) -> str:
    """
    Replace each placeholder in the English text with words from the parallel target text.
    """
    prompt = f"""
You will be given an English sentence with placeholders (#######) and its parallel sentence in {target_language}.
Replace each placeholder with the corresponding segment from the {target_language} text, ensuring:
- The inserted text matches the target-language phrasing (inflections, gender, number).
- The final sentence reads naturally as mixed English and {target_language}.
- Preserve the original sentence order.
Return only the filled sentence, no extra comments.

[English with placeholders]
{placeholder_text}

[{target_language} parallel text]
{target_text}

[Mixed code-switched result]
"""
    response = invoke_claude(prompt)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Randomly placeholder ~20% of words, then code-switch using parallel text."
    )
    parser.add_argument("--input_csv", required=True,
                        help="CSV with English and target text columns.")
    parser.add_argument("--source_column", required=True,
                        help="Name of the English source column.")
    parser.add_argument("--target_column", required=True,
                        help="Name of the parallel target-language column.")
    parser.add_argument("--target_language", required=True,
                        help="Target language name (e.g., Arabic). ")
    parser.add_argument("--csw_column_name", required=True,
                        help="Name for the new mixed column.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save outputs.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Process only first N rows.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"random_csw_{args.target_language}.log")
    logger = logging.getLogger("random_code_switch")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Loading {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} rows")

    if args.sample_size:
        df = df.head(args.sample_size)
        logger.info(f"Sampling first {args.sample_size} rows")

    # Step 1: placeholder insertion
    logger.info("Inserting random placeholders (~20% of words)")
    df['placeholder_text'] = df[args.source_column].progress_apply(
        lambda txt: insert_placeholders_random(txt, rate=0.2)
    )

    # Step 2: placeholder replacement
    logger.info("Filling placeholders via Claude using parallel text")
    df[args.csw_column_name] = df.progress_apply(
        lambda row: code_switch_text(
            row['placeholder_text'], row[args.target_column], args.target_language
        ), axis=1
    )

    # Save
    base = os.path.splitext(os.path.basename(args.input_csv))[0]
    out_path = os.path.join(args.output_dir, f"{base}_rnd_{args.csw_column_name}.csv")
    logger.info(f"Saving to {out_path}")
    df.to_csv(out_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()