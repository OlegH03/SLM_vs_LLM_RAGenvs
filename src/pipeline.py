import torch
from src.evaluation import (
    ragas_faithfulness,
    ragas_answer_relevance,
    ragas_context_relevance,
    chen_noise_robustness,
    chen_negative_rejection,
    chen_negative_rejection_chatgpt,
    chen_information_integration,
    chen_counterfactual_robustness,
)
from src.config import MAX_NEW_TOKENS, TEMPERATURE


def build_prompt(query, contexts, instruction):
    docs = "\n".join(contexts)
    return instruction.format(context=docs, question=query)


def generate_answer(prompt, model):
    if "tokenizer" in model:
        tokenizer = model["tokenizer"]
        model_obj = model["model"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)

        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return answer

    elif "client" in model:
        client = model["client"]
        model_name = model["model_name"]

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content

    else:
        raise ValueError("Unbekannter Modell-Typ")


def run_single_query(
    query_card, model, instruction, evaluator_client=None, testbed="noise"
):
    # 1. Build prompt
    prompt = build_prompt(query_card["query"], query_card["context"], instruction)

    # 2. Generate answer
    prediction = generate_answer(prompt, model)

    # 3. Evaluieren
    results = {
        "query_id": query_card["id"],
        "question": query_card["query"],
        "prediction": prediction,
        "ground_truth": query_card["answer"],
        "contexts": query_card["context"],
    }

    # RAGAS metrics
    if evaluator_client:
        if testbed == "noise":
            results["faithfulness"] = ragas_faithfulness(
                prediction, query_card["context"]
            )
            results["answer_relevance"] = ragas_answer_relevance(
                query_card["query"], prediction
            )
            results["context_relevance"] = ragas_context_relevance(
                query_card["query"], query_card["context"]
            )

            # Chen metrics
            results["noise_robustness"] = chen_noise_robustness(
                prediction, query_card["answer"]
            )
        elif testbed == "rejection":
            results["negative_rejection"] = chen_negative_rejection(prediction)
            results["negative_rejection_llm"] = chen_negative_rejection_chatgpt(
                query_card["query"], prediction, evaluator_client
            )
        elif testbed == "integration":
            results["information_integration"] = chen_information_integration(
                prediction, query_card["answer"]
            )
        elif testbed == "fact":
            cf = chen_counterfactual_robustness(prediction, query_card["answer"])
            results["counterfactual_detection"] = cf["error_detected"]
            results["counterfactual_correction"] = cf["error_corrected"]

    return results
