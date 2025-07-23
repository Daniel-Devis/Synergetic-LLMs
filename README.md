# 🔍 Self-Reviewing LLMs for Unit Test Generation

This project explores the capabilities of large language models (LLMs) — specifically **GPT-3.5 Turbo** and **GPT-4o Mini** — to generate and **self-review** unit test cases for focal methods using a Synergetic-LLM framework. It assesses whether LLMs can reliably generate, critique, and improve unit tests.

---

## 🧠 Project Overview

- Fine-tune LLMs on paired Java focal methods and reference unit tests from the [Methods2Test dataset](https://github.com/Software-Systems-Lab/Methods2Test).
- Use the fine-tuned models to generate unit tests for unseen Java methods.
- Employ a Synergetic LLM to evaluate and review the generated tests.
- Measure effectiveness using metrics like **CodeBLEU**, **Precision**, **Recall**, and **F1 Score**.

---

## 🗂️ File Descriptions

| Filename               | Description |
|------------------------|-------------|
| `3.5finetune.py`        | Fine-tunes GPT-3.5 Turbo using JSONL data via OpenAI API. |
| `4ominifinetune.py`     | Fine-tunes GPT-4o Mini using JSONL data via OpenAI API. |
| `3.5generator.py`       | Uses the fine-tuned GPT-3.5 to generate unit tests for Java focal methods. |
| `4ogptmini.py`          | Uses the fine-tuned GPT-4o Mini to generate unit tests. |
| `3.5testgencheck.py`    | Evaluates test cases from GPT-3.5 using CodeBLEU, Precision, Recall, F1 Score, and dual-LLM review. |
| `4ogpttestgencheck.py`  | Evaluates test cases from GPT-4o Mini similarly. |
| `jasonifier.py`         | Creates JSONL fine-tuning files with focal methods and reference test cases. |
| `fulevalcsv.py`         | Converts evaluation results into a CSV with CodeBLEU, Precision, Recall, and F1 Score columns. |


---

## 📊 Dataset

This project uses the [Methods2Test dataset](https://github.com/Software-Systems-Lab/Methods2Test), which includes:
- Java class files
- Focal methods
- Associated reference unit tests

