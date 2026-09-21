import yaml
from pathlib import Path
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, ContextRelevance


def load_prompts(yaml_path="src/instruction.yaml"):
    # Loads instruction prompts from YAML file
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def ragas_faithfulness(answer, contexts):
    """
    Faithfulness measures whether the answer is grounded in the retrieved context.
    The evaluator decomposes the answer into individual claims, and the score is
    the fraction of claims that can be inferred from the context
    """
    data = {"answer": [answer], "contexts": [contexts]}
    dataset = Dataset.from_dict(data)
    results = ragas_evaluate(dataset, metrics=[faithfulness])
    return results["faithfulness"]


def ragas_answer_relevance(question, answer):
    """
    Answer Relevance measures whether the answer addresses the question. The
    evaluator generates candidate questions from the answer alone, and the score
    reflects their similariy to the original question.
    """
    data = {"question": [question], "answer": [answer]}
    dataset = Dataset.from_dict(data)
    results = ragas_evaluate(dataset, metrics=[answer_relevancy])
    return results["answer_relevancy"]


def ragas_context_relevance(question, contexts):
    """
    Context Relevance measures whether the retrieved context is focused.
    The evaluator extracts the sentences needed to answer the question, and
    the score is the fraction of such sentences over all context sentences.
    """
    data = {"question": [question], "contexts": [contexts]}
    dataset = Dataset.from_dict(data)
    results = ragas_evaluate(dataset, metrics=[context_relevancy])
    return results["context_relevancy"]


def chen_checkanswer(prediction, ground_truth):
    """
    Check if the prediction contains the ground truth answer (Exact match)
    From github.com/chen700564/RGB
    """
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    labels = []
    for instance in ground_truth:
        if isinstance(instance, list):
            flag = False
            for variant in instance:
                if variant.lower() in prediction:
                    flag = True
                    break
        else:
            flag = instance.lower() in prediction
        labels.append(int(flag))
    return labels


def chen_noise_robustness(prediction, ground_truth):
    """
    Noise Robustness checks whether the model can extract the correct answer from a noisy context
    Scored by Exact Match accuracy against the gold answer.
    """
    labels = chen_checkanswer(prediction, ground_truth)
    return max(labels)


def chen_negative_rejection(prediction):
    """
    Negative Rejection measures whether the model declines to answer
    when no relevant information is provided. Scored by rejection rate
    whether the model outputs the specific rejection phrase.
    """
    rejection_phrases = [
        "insufficient information",
        "i can not answer",
        "can't answer",
        "not enough information",
        "unable to answer",
    ]
    prediction_lower = prediction.lower()
    for phrase in rejection_phrases:
        if phrase in prediction_lower:
            return 1
    return 0


def chen_information_integration(prediction, ground_truth):
    """
    Information Integration checks whether the model combines information from
    multiple documents into a single answer. Scored by Exact Match accuracy (same
    logic as noise robustness).
    """
    labels = chen_checkanswer(prediction, ground_truth)
    return max(labels)


def chen_counterfactual_robustness(prediction, ground_truth):
    """
    Counterfactual Robustness checks whether the model can detect and correct factual errors
    in retrieved documents. Returns both error detection and error correction status.
    """
    prediction_lower = prediction.lower()

    error_detected = 1 if "factual errors" in prediction_lower else 0

    error_corrected = max(chen_checkanswer(prediction, ground_truth))

    return {"error_detected": error_detected, "error_corrected": error_corrected}


def evaluate_all_ragas(questions, answers, contexts):
    data = {"question": questions, "answer": answers, "contexts": contexts}
    dataset = Dataset.from_dict(data)
    results = ragas_evaluate(
        dataset, metrics=[faithfulness, answer_relevancy, context_relevancy]
    )
    return {
        "faithfulness": results["faithfulness"],
        "answer_relevance": results["answer_relevancy"],
        "context_relevance": results["context_relevancy"],
    }


def evaluate_all_chen(predictions, ground_truths, testbed, questions=None, client=None):
    if testbed == "noise_robustness":
        scores = [
            chen_noise_robustness(pred, gt)
            for pred, gt in zip(predictions, ground_truths)
        ]
        return {"accuracy": sum(scores) / len(scores)}

    elif testbed == "negative_rejection":
        scores = [
            chen_negative_rejection_chatgpt(q, pred, client)
            for q, pred in zip(questions, predictions)
        ]
        return {"rejection_rate": sum(scores) / len(scores)}

    elif testbed == "information_integration":
        scores = [
            chen_information_integration(pred, gt)
            for pred, gt in zip(predictions, ground_truths)
        ]
        return {"accuracy": sum(scores) / len(scores)}

    elif testbed == "counterfactual_robustness":
        results = [chen_counterfactual_robustness(pred) for pred in predictions]
        error_detected = sum(r["error_detected"] for r in results) / len(results)
        return {"error_detection_rate": error_detected}
    else:
        raise ValueError(f"Unknown testbed: {testbed}")


def chen_negative_rejection_chatgpt(question, prediction, client):
    """
    Uses GPT-4o-mini to check, whether the model rejected.
    Adapted from Chen et al. (2024)
    Returns 1 if rejected, 0 if not
    """

    prompt = f"""I will give you a question and an answer generated through 
    document retrieval. Please use this answer to determine if the retrieved document can solve the question.

    Demonstrations:
    Question: Who is the champion of Australian Open 2023 Women's Singles?
    Answer: Serena Williams
    Yes, the question is addressed by the documents.

    Question: Where is ACL2023 held?
    Answer: Location of ACL2023 has not been confirmed.
    No, the question is not addressed by the documents.

    Question: What is the GDP of China in 2023?
    Answer: I can not answer this question.
    No, the question is not addressed by the documents.

    Begin to generate:
    Question: {question}
    Answer: {prediction}"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        evaluation = response.choices[0].message.content

        if "not addressed" in evaluation.lower():
            return 1
        return 0

    except Exception as e:
        print(f"Error in ChatGPT rejection check: {e}")
        return 0
