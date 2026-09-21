import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from src.config import (
    DATA_FACT_PATH,
    DATA_INT_PATH,
    DATA_PATH,
    RESULTS_PATH,
    MODELS,
    TEMPERATURE,
    TOP_K,
    NOISE_RATIO,
    SEED,
    MAX_NEW_TOKENS,
)
from src.data import load_data, get_query_card
from src.models import load_local_model, load_api_model
from src.pipeline import run_single_query
from src.evaluation import load_prompts


def run_phase1():
    """
    Executes Phase 1 of the experiment:
    - loads all 300 queries
    - processes each query with all model configurations (5)
    - stores results after every fifth query-block
    - marks failed queries with error
    """
    # Setup
    data = load_data()
    prompts = load_prompts()
    instruction = prompts["system_prompt"]

    # Conditions
    CONDITIONS = {
        "noise": {"path": DATA_PATH, "only_negative": False},
        "rejection": {"path": DATA_PATH, "only_negative": True},
    }
    TESTBEDS = {
        "noise": {
            "path": DATA_PATH,
            "prompt": prompts["system_prompt"],
            "pos_key": "positive",
            "only_negative": False,
        },
        "rejection": {
            "path": DATA_PATH,
            "prompt": prompts["system_prompt"],
            "pos_key": "positive",
            "only_negative": True,
        },
        "integration": {
            "path": DATA_INT_PATH,
            "prompt": prompts["system_prompt"],
            "pos_key": "positive",
            "only_negative": False,
        },
        "fact": {
            "path": DATA_FACT_PATH,
            "prompt": prompts["system_prompt_fact"],
            "pos_key": "positive_wrong",
            "only_negative": False,
        },
    }
    # Create result order
    results_dir = Path(RESULTS_PATH)
    results_dir.mkdir(exist_ok=True)

    # RAGAS evaluator client
    evaluator_client = load_api_model("openai/gpt-4o-mini")["client"]
    total_queries = len(data)

    # Iterate over all model configurations
    for model_name, model_path in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 60}\n")

        if model_name == "gpt-4o-mini":
            # API model - only one configuration
            model_configs = [(model_path, False)]
        else:
            # Local models - test both FP16 and Q4
            model_configs = [(model_path, False), (model_path, True)]

        for model_path_config, quantize in model_configs:
            quant_str = "Q4" if quantize else "FP16"
            print(f"\nLoading {model_name} {quant_str}...")

            # Load model
            if model_name == "gpt-4o-mini":
                model = load_api_model(model_path_config)
            else:
                model = load_local_model(model_path_config, quantize=quantize)

            # CSV path for this configuration
            csv_path = results_dir / f"phase1_results_{model_name}_{quant_str}.csv"

            # Check whether CSV already exists
            if csv_path.exists():
                existing_df = pd.read_csv(csv_path)
                processed_ids = set(existing_df["query_id"].tolist())
                print(f"Found: {len(processed_ids)} already processed queries")
            else:
                existing_df = None
                processed_ids = set()

            checkpoint_buffer = []

            print(f"Start processing of {total_queries} queries...")

            for query_id in tqdm(
                range(total_queries), desc=f"{model_name} {quant_str}"
            ):
                # Skip if already processed
                if query_id in processed_ids:
                    continue

                try:
                    # load query card
                    query_card = get_query_card(query_id, data)

                    # execute pipeline
                    results = run_single_query(
                        query_card=query_card,
                        model=model,
                        instruction=instruction,
                        evaluator_client=evaluator_client,
                    )

                    csv_row = {
                        "query_id": results["query_id"],
                        "question": results["question"],
                        "prediction": results["prediction"],
                        "ground_truth": results["ground_truth"],
                        "faithfulness": results.get("faithfulness", np.nan),
                        "answer_relevance": results.get("answer_relevance", np.nan),
                        "context_relevance": results.get("context_relevance", np.nan),
                        "noise_robustness": results.get("noise_robustness", np.nan),
                        "negative_rejection": results.get("negative_rejection", np.nan),
                        "information_integration": results.get(
                            "information_integration", np.nan
                        ),
                        "counterfactual_robustness": results.get(
                            "counterfactual_robustness", np.nan
                        ),
                        "error": "",
                    }
                except Exception as e:
                    # In failure case: prediction empty, set error, metrics to NaN
                    csv_row = {
                        "query_id": query_id,
                        "question": data[query_id]["query"],
                        "prediction": "",
                        "ground_truth": data[query_id]["answer"],
                        "faithfulness": np.nan,
                        "answer_relevance": np.nan,
                        "context_relevance": np.nan,
                        "noise_robustness": np.nan,
                        "negative_rejection": np.nan,
                        "information_integration": np.nan,
                        "counterfactual_robustness": np.nan,
                        "error": str(e),
                    }
                    print(f"\nError for query {query_id}: {e}")

                checkpoint_buffer.append(csv_row)

                # Checkpoint: save after every fifth query
                if len(checkpoint_buffer) >= 5:
                    existing_df = save_checkpoint(
                        checkpoint_buffer, csv_path, existing_df
                    )
                    checkpoint_buffer = []

            # save last queries
            if checkpoint_buffer:
                existing_df = save_checkpoint(checkpoint_buffer, csv_path, existing_df)

            print(f"\nCompleted {model_name} {quant_str}")
            print(f"Saved results in: {csv_path}")

            # Error statistics
            final_df = pd.read_csv(csv_path)
            error_count = (final_df["error"] != "").sum()
            print(f"Failed queries: {error_count}/{total_queries}")


def save_checkpoint(buffer, csv_path, existing_df):
    """
    Saves checkpoint buffer to CSV file.

    Args:
        buffer: List of csv_row dictionaries to save
        csv_path: Path to the CSV file
        existing_df: Existing DataFrame or None if new file

    Returns:
        Updated DataFrame (for next checkpoint)
    """
    new_df = pd.DataFrame(buffer)

    if existing_df is not None:
        # Append to existing DataFrame
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
        print(f"Checkpoint saved: {len(buffer)} queries (total: {len(combined_df)})")
        return combined_df
    else:
        # Create new CSV
        new_df.to_csv(csv_path, index=False)
        print(f"Checkpoint saved: {len(buffer)} queries")
        return new_df


if __name__ == "__main__":
    run_phase1()
