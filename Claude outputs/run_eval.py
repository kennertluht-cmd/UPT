"""
Main experiment script for the UPT project.

Runs DeepSeek-R1-Distill models on MATH-500 in three conditions and grades the
answers the same way as Barua et al. (2026), eval/eval.py:

  en_only    : English question,  English reasoning   (En-Only)
  en_cot     : Estonian question, English reasoning   (En-CoT)
  target_cot : Estonian question, Estonian reasoning  (Target-CoT)

Language forcing (instruction + prefilled start of the reasoning) is used in all
three conditions, so En-Only and En-CoT differ only in the question language.

Backends:
  ollama : quick tests on the Mac (quantized model, short token limit)
  vllm   : main runs on a rented NVIDIA GPU (original precision, 32768 tokens)

Examples:
  python run_eval.py --backend ollama --model 1.5B --condition all --limit 3
  python run_eval.py --backend vllm --model 7B --condition all

Results: results/<backend>/<model>/<condition>.jsonl (one line per problem).
If a run stops, start the same command again and it continues where it left off.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "harjutused")             # hendrycks_utils.py, language_validation.py (Estonian added)
sys.path.insert(0, "Multilingual-Long-CoT")  # Barua's repository (prompts)

from hendrycks_utils import last_boxed_only_string, remove_boxed, is_equiv
from language_validation import validate_language
from utils.prompts import LF_prefix_dict, LF_suffix_dict, verify_answer_template

# -------------------------
# Settings (same as Barua et al.)
# -------------------------

TEMPERATURE = 0.6
TOP_P = 0.95
MAX_MODEL_LEN = 32768          # prompt + answer, as in Barua's eval.py
CONFIDENCE_THRESHOLD = 0.9     # language check
COMPLIANCE_THRESHOLD = 0.9     # language check

BARUA_DATA = "Multilingual-Long-CoT/eval/benchmarks/mt-math-500"
ESTONIAN_DATA = "translated_et_fable.json"

# Hugging Face name (vLLM) and Ollama name for each model size
MODELS = {
    "1.5B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-r1:1.5b"),
    "7B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-r1:7b"),
    "14B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-r1:14b"),
    "32B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-r1:32b"),
}

# condition -> (question language column, reasoning language)
CONDITIONS = {
    "en_only": ("en", "English"),
    "en_cot": ("et", "English"),
    "target_cot": ("et", "Estonian"),
}

# Language forcing: English strings come straight from Barua's prompts.py,
# Estonian strings were checked by an English teacher (informal "sina" form).
PREFIX = {
    "English": LF_prefix_dict["English"],
    "Estonian": "Palun arutle samm-sammult ja kirjuta lõppvastus \\boxed{} sisse. "
                "Sinu mõttekäik ja lõpplahendus peavad olema eesti keeles.",
}
SUFFIX = {
    "English": LF_suffix_dict["English"],
    "Estonian": "Olgu, proovin selle välja mõelda.",
}


# -------------------------
# Data
# -------------------------

def load_problems(limit):
    """Join Barua's MATH-500 file (answer, level, subject) with the Estonian translation."""
    from datasets import load_from_disk

    barua = load_from_disk(BARUA_DATA)
    with open(ESTONIAN_DATA, encoding="utf-8") as f:
        estonian = json.load(f)

    problems = []
    for row in estonian[:limit]:
        nr = row["nr"]
        # Safety check: the translation must belong to the same problem
        if barua[nr]["en"] != row["en"]:
            raise ValueError(f"Problem {nr}: English text does not match Barua's file")
        problems.append({
            "row_index": nr,
            "en": row["en"],
            "et": row["et"],
            "answer": str(barua[nr]["answer"]),
            "level": str(barua[nr]["level"]),
            "subject": str(barua[nr]["subject"]),
        })
    return problems


# -------------------------
# Prompt (same as Barua's PromptGenerator.generate_message)
# -------------------------

def build_prompt(tokenizer, question, reasoning_language):
    message = [{"role": "user", "content": f"{question}\n{PREFIX[reasoning_language]}"}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return f"{prompt} {SUFFIX[reasoning_language]}"


# -------------------------
# Generation
# -------------------------

_llm = None  # the model is loaded once and reused for all three conditions


def generate_vllm(prompts, hf_name, tokenizer):
    """All prompts at once on the GPU. Returns a list of (text, number of output tokens)."""
    global _llm
    import torch
    from vllm import LLM, SamplingParams

    if _llm is None:
        _llm = LLM(
            model=hf_name,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=max(1, torch.cuda.device_count()),
            revision="main",
            trust_remote_code=True,
            enforce_eager=True,
        )
    llm = _llm
    # Prompt + answer must fit into MAX_MODEL_LEN, so the answer limit depends on the prompt length
    params = []
    for p in prompts:
        n_prompt = len(tokenizer(p)["input_ids"])
        params.append(SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P,
                                     max_tokens=MAX_MODEL_LEN - n_prompt))
    outputs = llm.generate(prompts, params)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]


def generate_ollama_one(prompt, ollama_name, max_tokens):
    """One prompt on the Mac. Returns (text, number of output tokens)."""
    import requests

    request = {
        "model": ollama_name,
        "prompt": prompt,
        "raw": True,          # we already built the full chat template ourselves
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": max_tokens,
            "num_ctx": max_tokens + 1024,   # room for the prompt
        },
    }
    r = requests.post("http://localhost:11434/api/generate", json=request, timeout=7200)
    r.raise_for_status()
    data = r.json()
    return data["response"], data.get("eval_count", 0)


# -------------------------
# Grading (same as Barua's ResponseProcessor + _label_raw_file)
# -------------------------

_openai_client = None


def ask_gpt5_mini(prompt):
    """Barua's settings: gpt-5-mini, flex tier first, then auto; 3 retries each."""
    global _openai_client
    from openai import OpenAI
    if _openai_client is None:
        _openai_client = OpenAI()  # reads OPENAI_API_KEY

    for tier in ("flex", "auto"):
        for attempt in range(4):
            try:
                resp = _openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024,
                    temperature=1.0,
                    reasoning_effort="medium",
                    service_tier=tier,
                )
                text = resp.choices[0].message.content or ""
                if text:
                    return text
                break  # empty answer: try the next tier
            except Exception as e:
                print(f"  gpt-5-mini ({tier}) attempt {attempt + 1} failed: {e}")
                time.sleep(min(0.75 * 2 ** attempt, 8.0))
    return ""


def check_answer(response, gold, question):
    """Returns (correct?, how it was decided)."""
    try:
        boxed = (last_boxed_only_string(response) or "").strip()
    except Exception:
        boxed = ""
    if not boxed:
        return False, "no_boxed"

    try:
        if is_equiv(gold, remove_boxed(boxed)):
            return True, "is_equiv"
    except Exception:
        pass  # unusual \boxed format: let gpt-5-mini decide

    prompt = verify_answer_template.format(problem=question, gt_solution=gold, model_solution=boxed)
    text = ask_gpt5_mini(prompt)
    if not text:
        return False, "gpt5mini_failed"
    if "Correct:" in text:
        text = text.split("Correct:")[1].strip().lower()
    return "true" in text.lower(), "gpt5mini"


def label(record, reasoning_language):
    """Add answer check, language check and final 'correct' to one record."""
    response = record["response"]
    lang_ok, lang_score, n_chunks, _valid, invalid = validate_language(
        response, reasoning_language, CONFIDENCE_THRESHOLD, COMPLIANCE_THRESHOLD
    )
    ans_ok, how = check_answer(response, record["answer"], record["question"])

    record["answer_alone"] = 1 if ans_ok else 0
    record["verified_by"] = how
    record["language_compliance"] = lang_ok
    record["language_compliance_score"] = lang_score
    record["language_filtered_chunks"] = n_chunks
    record["language_invalid_chunks"] = invalid
    record["correct"] = 1 if (ans_ok and lang_ok == 1) else 0
    return record


# -------------------------
# One condition
# -------------------------

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_condition(args, condition, problems, tokenizer):
    question_col, reasoning_language = CONDITIONS[condition]
    hf_name, ollama_name = MODELS[args.model]

    out_dir = os.path.join(args.output, args.backend, args.model)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{condition}.raw.jsonl")      # answers only
    final_path = os.path.join(out_dir, f"{condition}.jsonl")        # answers + grades

    # 1) Generate answers (skip problems that already have one)
    done = {r["row_index"] for r in read_jsonl(raw_path)}
    todo = [p for p in problems if p["row_index"] not in done]
    print(f"\n[{condition}] {len(done)} done, {len(todo)} to generate")

    def make_record(p, text, n_tokens, limit):
        return {
            "row_index": p["row_index"],
            "question": p[question_col],
            "response": text,
            "answer": p["answer"],
            "level": p["level"],
            "subject": p["subject"],
            "condition": condition,
            "query_language": "English" if question_col == "en" else "Estonian",
            "reasoning_language": reasoning_language,
            "model": hf_name if args.backend == "vllm" else ollama_name,
            "backend": args.backend,
            "n_output_tokens": n_tokens,
            "hit_limit": n_tokens >= limit,
        }

    if todo and args.backend == "vllm":
        prompts = [build_prompt(tokenizer, p[question_col], reasoning_language) for p in todo]
        results = generate_vllm(prompts, hf_name, tokenizer)
        with open(raw_path, "a", encoding="utf-8") as f:
            for p, prompt, (text, n) in zip(todo, prompts, results):
                limit = MAX_MODEL_LEN - len(tokenizer(prompt)["input_ids"])
                f.write(json.dumps(make_record(p, text, n, limit), ensure_ascii=False) + "\n")

    if todo and args.backend == "ollama":
        for i, p in enumerate(todo, 1):
            prompt = build_prompt(tokenizer, p[question_col], reasoning_language)
            text, n = generate_ollama_one(prompt, ollama_name, args.max_tokens)
            with open(raw_path, "a", encoding="utf-8") as f:   # saved after every problem
                f.write(json.dumps(make_record(p, text, n, args.max_tokens), ensure_ascii=False) + "\n")
            print(f"  {i}/{len(todo)} problem {p['row_index']}: {n} tokens")

    # 2) Grade all answers (always redone from the raw file, so it is cheap to rerun)
    raw = read_jsonl(raw_path)
    print(f"[{condition}] grading {len(raw)} answers")
    labeled = [label(r, reasoning_language) for r in raw]
    with open(final_path, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(labeled)
    acc = sum(r["correct"] for r in labeled) / n if n else 0
    print(f"[{condition}] accuracy {acc:.1%} ({sum(r['correct'] for r in labeled)}/{n}), "
          f"answers correct {sum(r['answer_alone'] for r in labeled)}, "
          f"language ok {sum(r['language_compliance'] for r in labeled)}, "
          f"hit limit {sum(r['hit_limit'] for r in labeled)}")


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="UPT: DeepSeek-R1-Distill on MATH-500 in English and Estonian")
    parser.add_argument("--backend", choices=["ollama", "vllm"], required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--condition", choices=list(CONDITIONS) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=500, help="number of problems (test with a few first)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="answer limit for ollama tests only")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    hf_name = MODELS[args.model][0]
    tokenizer = AutoTokenizer.from_pretrained(hf_name)  # same chat template as Barua

    problems = load_problems(args.limit)
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for condition in conditions:
        run_condition(args, condition, problems, tokenizer)


if __name__ == "__main__":
    main()
