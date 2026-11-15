# UPT: Reasoning language and LLM accuracy on Estonian math problems

Student research project (Hugo Treffneri Gümnaasium, 2026/2027).

Do large language models solve math problems less accurately when the question
and the reasoning are in Estonian instead of English? The project replicates the
MATH-500 part of Barua et al. (2026), *Long Chain-of-Thought Reasoning Across
Languages* (ICLR 2026), and extends it to Estonian.

## Experiment

| Condition | Question | Reasoning |
|---|---|---|
| En-Only | English | English |
| En-CoT | Estonian | English |
| Target-CoT | Estonian | Estonian |

- Models: DeepSeek-R1-Distill-Qwen 1.5B, 7B, 14B and 32B
- Data: MATH-500 (500 problems), translated into Estonian with Claude Fable 5.2
  (`claude-fable-5-2`) using Barua et al.'s translation prompt unchanged
- Settings (as in Barua et al.): temperature 0.6, top_p 0.95, up to 32768 tokens,
  one answer per problem (pass@1), language forcing in all conditions
- Grading (as in Barua et al.): last `\boxed{}` answer checked with `is_equiv`,
  then gpt-5-mini; language checked with Lingua. An answer counts as correct only
  if both the answer and the language are correct.

## Files

| File | What it is |
|---|---|
| `run_eval.py` | Main experiment script (Ollama for local tests, vLLM on a rented GPU) |
| `translate_fable.py` | Translates the 500 problems into Estonian |
| `check_translation.py` | Checks that all translations exist and formulas did not change |
| `translated_et_fable.json` | The Estonian translation (500 problems) |
| `exercises/` | Practice scripts written while learning (yl1–yl9) and two helper files used by `run_eval.py` |
| `archive/` | Old files no longer used (first Gemini translation, earlier plan to check 150 translations by hand) |

## How to run

```
python run_eval.py --backend ollama --model 1.5B --condition all --limit 3   # test on a Mac
python run_eval.py --backend vllm --model 7B --condition all                 # main run on a GPU
```

Needs `OPENAI_API_KEY` (gpt-5-mini) and Barua et al.'s repository in
`Multilingual-Long-CoT/` (not included here).

## Credits

- `run_eval.py` is adapted from Barua et al. (2026), `eval/eval.py`,
  https://github.com/Berkeley-NLP/Multilingual-Long-CoT (MIT License).
- `exercises/hendrycks_utils.py` is from EleutherAI lm-evaluation-harness.
- `exercises/language_validation.py` is from Barua et al., with Estonian added.
- Scripts were written with help from Claude (Anthropic).