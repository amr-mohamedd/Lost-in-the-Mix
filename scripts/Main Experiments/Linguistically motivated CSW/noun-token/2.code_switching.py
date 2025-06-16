#!/usr/bin/env python3
"""
Combined code-switching script: identifies code-switch points in an English sentence
and replaces them with the corresponding text from a target-language sentence.
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
You are an expert linguist and code‑switching analyst. Based on the Equivalence Constraint Theory and the Matrix Language Frame model, identify nouns in the input English sentence that would serve as appropriate code‑switching points.

– Input variable: text (a single English sentence)  
– Task: Find every noun (as a free content morpheme) that can be switched under the theories above.  
– Transformation: Replace each identified noun in the sentence with "#######".  
– Output: Return only the transformed sentence with nouns replaced by "#######".
- The substituted words blend seamlessly into the text, following natural bilingual speech patterns.
- Adjust the target language words as needed (e.g., inflection, gender, number) so that the text remains syntactically correct.
- Ensure that nouns in common expressions are not code-switched.
- Don't return any summary or introduction, just the processed text

[English text]
{text}
"""
    response = invoke_claude(prompt)
    return response.strip()


def code_switch_text(placeholder_text: str, target_text: str, target_language: str) -> str:
    """
    Replace each hashtag-sequence (#######) in the placeholder_text with the 
    corresponding word(s) from the parallel target_text, producing a code-switched sentence.
    """
    prompt = f"""
You will be given a pair of parallel texts in English and {target_language}.

Your goal is to produce a code-switched version of the English text by replacing each of the hashtag-sequences (#######) in the English text with their {target_language} counterparts from the {target_language} text, ensuring that:
- The substituted words blend seamlessly into the text, following natural bilingual speech patterns.
- The text should be grounded with the principles of the Equivalence Constraint Theory and the Matrix Language Frame model.
- Adjust the target language words as needed (e.g., inflection, gender, number) so that the text remains syntactically correct.
- The original meaning and flow of the text are maintained.
- All the hashtag-sequences (#######) have to be replaced with text from the {target_language} text.
- Use only the words from the {target_language} text.
- Return only the code-switched text, without any additions or explanations.

[English text with placeholders]
{placeholder_text}

[{target_language} text]
{target_text}

[Code-switched English and {target_language}]
"""
# .format(
#         target_language=target_language,
#         placeholder_text=placeholder_text,
#         target_text=target_text
    # )
    response = invoke_claude(prompt)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Combine placeholder insertion and replacement for code-switching English text using parallel target-language text."
    )
    parser.add_argument("--input_csv", required=True,
                        help="Path to the input CSV file containing English and parallel text columns.")
    parser.add_argument("--source_column", required=True,
                        help="Name of the English source text column.")
    parser.add_argument("--target_column", required=True,
                        help="Name of the parallel target-language text column.")
    parser.add_argument("--target_language", required=True,
                        help="Name of the target language (e.g., Arabic, French). ")
    parser.add_argument("--csw_column_name", required=True,
                        help="Name of the new column for the final code-switched text.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where the updated CSV and logs will be saved.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional number of rows to process (defaults to all rows).")

    args = parser.parse_args()

    # Prepare output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{args.target_language}.log")
    logger = logging.getLogger("code_switching")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Loading data from {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise

    if args.sample_size:
        df = df.drop_duplicates(subset=args.source_column).head(args.sample_size)
        logger.info(f"Processing first {args.sample_size} unique rows from {args.source_column}")

    # Step 1: generate placeholders
    logger.info("Generating code-switching placeholders")
    df['placeholder_text'] = df[args.source_column].progress_apply(
        lambda txt: get_switching_points(txt)
    )

    # Step 2: replace placeholders with target-language text
    logger.info("Replacing placeholders with target-language text")
    df[args.csw_column_name] = df.progress_apply(
        lambda row: code_switch_text(
            row['placeholder_text'], row[args.target_column], args.target_language
        ), axis=1
    )

    # Save results
    input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_path = os.path.join(
        args.output_dir,
        f"{input_basename}_with_{args.csw_column_name}.csv"
    )
    logger.info(f"Saving updated DataFrame to {output_path}")
    try:
        df.to_csv(output_path, index=False)
        logger.info("Save completed successfully")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        raise


if __name__ == "__main__":
    main()