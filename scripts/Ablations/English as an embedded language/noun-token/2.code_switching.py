#!/usr/bin/env python3
'''
Reverse code-switching script: masks code-switch points in a target-language sentence
and then replaces those masked spots with corresponding words from the English sentence.
'''
import argparse
import pandas as pd
from tqdm import tqdm
import os
import logging
from claude import invoke_claude

# Enable progress bars for pandas operations
tqdm.pandas()

def mask_target_points(text: str, target_language: str) -> str:
    '''
    Identify nouns in the target-language sentence that can serve as code-switching points,
    replacing each noun with a sequence of hashtags (#######).
    '''
    prompt = f"""
You are an expert linguist and code‑switching analyst. Based on the Equivalence Constraint Theory and the Matrix Language Frame model, identify nouns in the input {target_language} sentence that would serve as appropriate code‑switching points.

– Input variable: text (a single {target_language} sentence)
– Task: Find every noun (as a free content morpheme) that can be switched under the theories above.
– Transformation: Replace each identified noun in the sentence with "#######".
– Output: Return only the transformed sentence with nouns replaced by "#######".
- The substituted words blend seamlessly into the text, following natural bilingual speech patterns.
- Adjust the English words as needed so that the text remains syntactically correct when inserted later.
- Ensure that nouns in common expressions are not code-switched.
- Don't return any summary or introduction, just the processed text

[{target_language} text]
{text}
"""
    response = invoke_claude(prompt)
    return response.strip()

def insert_english_words(placeholder_text: str, english_text: str, target_language: str) -> str:
    '''
    Replace each hashtag-sequence (#######) in the placeholder_text with the
    corresponding words from the English text, producing a reverse code-switched sentence.
    '''
    prompt = f"""
You will be given parallel texts in {target_language} and English.

Your goal is to produce a reverse code-switched version of the {target_language} text by replacing each of the hashtag-sequences (#######) in the {target_language} placeholder text with their English counterparts from the English text, ensuring that:
- The substituted English words fit naturally into the {target_language} sentence structure.
- The text should be grounded with the principles of the Equivalence Constraint Theory and the Matrix Language Frame model.
- Inflect English words if needed to maintain grammatical correctness.
- The original meaning and flow of the {target_language} text are preserved.
- All the hashtag-sequences (#######) have to be replaced with text from the English text.
- Use only the words from the English text.
- Return only the final mixed sentence, without any additions or explanations.

[{target_language} text with placeholders]
{placeholder_text}

[English text]
{english_text}

[Reverse code-switched {target_language} and English]
"""
    response = invoke_claude(prompt)
    return response.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Mask then insert English words into a target-language text for reverse code-switching."
    )
    parser.add_argument("--input_csv", required=True,
                        help="Path to the input CSV file containing source English and target-language columns.")
    parser.add_argument("--source_column", required=True,
                        help="Name of the English source text column.")
    parser.add_argument("--target_column", required=True,
                        help="Name of the target-language text column.")
    parser.add_argument("--target_language", required=True,
                        help="Name of the target language (e.g., Arabic, French). ")
    parser.add_argument("--csw_column_name", required=True,
                        help="Name of the new column for the final reverse code-switched text.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where the updated CSV and logs will be saved.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional number of rows to process (defaults to all rows).")

    args = parser.parse_args()

    # Prepare output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"reverse_{args.target_language}.log")
    logger = logging.getLogger("reverse_code_switching")
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
        df = df.drop_duplicates(subset=args.target_column).head(args.sample_size)
        logger.info(f"Processing first {args.sample_size} unique rows from {args.target_column}")

    # Step 1: mask target-language nouns
    logger.info("Masking target-language code-switching points")
    df['masked_target'] = df[args.target_column].progress_apply(
        lambda txt: mask_target_points(txt, args.target_language)
    )

    # Step 2: insert English words into masked placeholders
    logger.info("Inserting English words into masked placeholders")
    df[args.csw_column_name] = df.progress_apply(
        lambda row: insert_english_words(
            row['masked_target'], row[args.source_column], args.target_language
        ), axis=1
    )

    # Save results
    input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_path = os.path.join(
        args.output_dir,
        f"{input_basename}_reverse_{args.csw_column_name}.csv"
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