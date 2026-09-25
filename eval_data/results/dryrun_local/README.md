# Local dry-run outputs — NOT reportable results

Produced with `qwen3:4b` as the generator (Groq `gpt-oss-20b` as judge) to
exercise every audit configuration for free and catch code bugs before spending
scarce Groq tokens. They are kept for provenance only.

Do not quote them: the model is far weaker than the shipped `gpt-oss-120b`,
Ollama's default 4,096-token context is smaller than the ~4.6K-token audit
prompt, and several runs are marked `contaminated` (the judge's daily token
budget ran out, so findings skipped a check). `audit_local_A.json` is an
interrupted run.
