#!/usr/bin/env python3
"""
Multi-language code-switching script: identifies code-switch points in an English sentence
and replaces them with corresponding words from multiple target-language sentences, distributing
the switches evenly across languages.
"""
import argparse
import pandas as pd
from tqdm import tqdm
import os
import logging
from claude import invoke_claude

# Enable progress bars for pandas operations
tqdm.pandas()


def get_switching_points(text: str) -> str:
    """
    Identify nouns in the English sentence that can serve as code-switching points,
    replacing each noun with a sequence of hashtags (#######).
    """
    prompt = f"""
You are an expert linguist and code-switching analyst. Based on the Equivalence Constraint Theory, identify nouns in the input English sentence that would serve as appropriate code-switching points.

– Input variable: text (a single English sentence)
– Task: Find every noun (as a free content morpheme) that can be switched under the theories above.
– Transformation: Replace each identified noun in the sentence with "#######".
– Output: Return only the transformed sentence with nouns replaced by "#######".
- Do not include any summary or extra commentary.

[English text]
{text}
"""
    response = invoke_claude(prompt)
    return response.strip()


def code_switch_multi(placeholder_text: str, target_texts: list, languages: list) -> str:
    """
    Replace each hashtag-sequence (#######) in the placeholder_text with words taken from
the parallel texts in different target languages, distributing replacements evenly.
    """
    # Prepare the parallel language sections
    parallel_section = "\n".join(
        [f"[{lang} text]\n{text}" for lang, text in zip(languages, target_texts)]
    )

    prompt = f"""
You are a code-switching specialist. Given an English sentence with placeholder markers (#######) and parallel sentences in multiple target languages, produce a single mixed-language code-switched English sentence by replacing each placeholder with the appropriate word or phrase from one of the target-language sentences.

Guidelines:
- Replace each ####### with text from exactly one of the provided target-language sentences.
- The text should be grounded with the principles of the Equivalence Constraint Theory and the Matrix Language Frame model.
- Distribute replacements as evenly as possible across the set of languages; do not use any language more often than the others.
- Maintain the original meaning and grammatical flow.
- Use the exact form (including inflections) from the parallel text.
- Return only the final code-switched sentence without any additional commentary.

[English text with placeholders]
{placeholder_text}

{parallel_section}

[Mixed-language code-switched sentence]
"""
    response = invoke_claude(prompt)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Perform code-switching on English sentences using multiple parallel texts and distribute switches evenly."
    )
    parser.add_argument("--input_csv", required=True,
                        help="Path to the input CSV file containing English and parallel text columns.")
    parser.add_argument("--source_column", required=True,
                        help="Name of the English source text column.")
    parser.add_argument("--target_columns", required=True,
                        help="Comma-separated list of column names for parallel texts.")
    parser.add_argument("--target_languages", required=True,
                        help="Comma-separated list of target language names corresponding to the parallel text columns.")
    parser.add_argument("--csw_column_name", required=True,
                        help="Name of the new column for the mixed code-switched text.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where the updated CSV and logs will be saved.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional number of rows to process (defaults to all rows).")

    args = parser.parse_args()

    # Parse lists
    target_cols = [c.strip() for c in args.target_columns.split(',')]
    languages = [l.strip() for l in args.target_languages.split(',')]
    if len(target_cols) != len(languages):
        raise ValueError("The number of target_columns and target_languages must match.")

    # Prepare output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "multi_language_code_switch.log")
    logger = logging.getLogger("multi_code_switch")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    if args.sample_size:
        df = df.drop_duplicates(subset=args.source_column).head(args.sample_size)
        logger.info(f"Processing first {args.sample_size} unique rows from {args.source_column}")

    # Step 1: generate placeholders (if not already present)
    if 'placeholder_text' not in df.columns:
        logger.info("Generating code-switching placeholders")
        df['placeholder_text'] = df[args.source_column].progress_apply(
            lambda txt: get_switching_points(txt)
        )
    else:
        logger.info("Found existing placeholders; skipping placeholder generation.")

    # Step 2: multi-language replacement
    logger.info("Replacing placeholders with mixed-language code switches")
    df[args.csw_column_name] = df.progress_apply(
        lambda row: code_switch_multi(
            row['placeholder_text'],
            [row[col] for col in target_cols],
            languages
        ), axis=1
    )

    # Save results
    input_base = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_path = os.path.join(
        args.output_dir,
        f"{input_base}_with_{args.csw_column_name}.csv"
    )
    logger.info(f"Saving updated DataFrame to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Save completed successfully")


if __name__ == "__main__":
    main()