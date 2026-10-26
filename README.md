# LLM prompt lab: support-ticket triage with the Gemini API

A small script, a test suite and an experiment notebook for one task: turn a free-text customer-support ticket into
schema-validated JSON.

```json
{
  "reasoning": "The customer paid via M-Pesa but has not received their order after two weeks. ...",
  "category": "shipping",
  "priority": "high",
  "sentiment": "negative",
  "summary": "Customer has not received their order two weeks after payment via M-Pesa."
}
```
(Real output of `python triage.py "Nimelipa kwa M-Pesa lakini oda yangu haijafika wiki mbili sasa."`.)

| file | what it is |
|---|---|
| [triage.py](triage.py) | the API script / CLI: config from env, Gemini call with timeout + bounded retries, strict JSON-Schema validation |
| [experiments.ipynb](experiments.ipynb) | 14 prompt experiments (11 on the main set + 3 on a hold-out set), executed, with conclusions |
| [results/runs.jsonl](results/runs.jsonl) | every real API reply behind the notebook (195 calls); also the notebook's cache |
| [data/tickets.json](data/tickets.json), [data/holdout.json](data/holdout.json) | 15 + 10 hand-labelled tickets |
| [tests/test_triage.py](tests/test_triage.py) | 57 mocked unit tests + 1 opt-in live test |
| [.env.example](.env.example) | config template (no secrets) |

## Install

Tested on Windows 11 with Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp; then put your key in .env
```

Get a free key at <https://aistudio.google.com/apikey>. `.env` is git-ignored; a real environment variable
`GEMINI_API_KEY` works too and takes precedence over `.env`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `GEMINI_API_KEY` | (required) | sent in the `x-goog-api-key` header, never in the URL, never printed (`Config.__repr__` hides it) |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | any Gemini model id |
| `GEMINI_TIMEOUT` | `30` | seconds per HTTP request |
| `GEMINI_MAX_RETRIES` | `3` | retries after the first attempt (so at most 4 requests per ticket) |

## Usage

```powershell
python triage.py "I was charged twice for my March subscription"
python triage.py --file ticket.txt
echo "My package never arrived" | python triage.py -
python triage.py --model gemini-3.6-flash --temperature 0.2 "..."
```

JSON goes to stdout, errors to stderr. Exit codes:

| code | meaning |
|---|---|
| 0 | success |
| 2 | bad config or input (missing/placeholder key, bad `GEMINI_TIMEOUT`, empty or >5000-char ticket, unreadable file) |
| 3 | API/network failure after retries, or a non-retryable HTTP error (400/401/403/404) |
| 4 | the model replied, but the reply is blocked, empty, not JSON, or violates the schema |

Output schema (`triage.SCHEMA`, JSON Schema 2020-12, `additionalProperties: false`): `category` ∈ billing / technical / account / shipping / other,
`priority` ∈ low / medium / high / urgent, `sentiment` ∈ negative / neutral / positive, `summary` (string, 1-300 chars), and an optional `reasoning` string.
The parser is strict: a reply wrapped in a Markdown code fence or with prose around the JSON is rejected, not repaired.

**Retry policy.** Retried with exponential backoff (1 s, 2 s, 4 s, ... capped at 60 s; a `Retry-After` header is honoured, also capped): HTTP 429/500/502/503/504,
request timeouts, connection errors. Not retried: other 4xx, blocked/empty replies, and invalid model output (it is reported as exit code 4 rather than re-asked,
so one bad reply costs one call).

## Tests

```powershell
python -m pytest -q                           # 57 mocked tests, no network, <1 s
$env:RUN_LIVE="1"; python -m pytest -q -m live # 1 real API call (needs a valid key)
```

Last run: `57 passed, 1 skipped` (the skipped one is the live test); with `RUN_LIVE=1`, the live test `1 passed` (one real call).

**Mocked** (`requests.post` and `time.sleep` are replaced; nothing leaves the machine): success and request shape (header, timeout, JSON mode, system prompt),
`thought` parts ignored, invalid outputs (not JSON, code fence, wrong type, bad enum x2, missing field, extra field, empty summary), blocked/empty replies,
missing / blank / placeholder key, bad numeric settings, 429 with and without `Retry-After`, 503, backoff sequence, retry cap, `Retry-After` cap, zero retries,
timeouts (recovered and persistent), connection errors, truncated / undecodable response bodies, 400/401/403/404 not retried, HTTP 200 with a non-JSON body or a JSON body of the wrong shape (reported, never an uncaught traceback), API key never in a URL or error message,
empty / over-long ticket rejected before any call, spoofed `</ticket>` stripped, and the CLI exit codes 0/2/3/4.

**Real, not mocked:** the single `live` test; the 195 calls behind the notebook; and by hand, the CLI with no key (exit 2, no network), a bogus key
(`HTTP 400: API key not valid`, exit 3, one request), and `GEMINI_TIMEOUT=0.001 GEMINI_MAX_RETRIES=1` (`giving up after 2 attempts: ConnectTimeout`, exit 3).
Real 429s were not provoked on purpose; rate-limit handling is covered by mocks only. In the notebook run, 4 of the 195 calls were retried by the client (max 4 attempts), and none ultimately failed.

As a one-off check that the tests are not vacuous, I broke `triage.py` in 9 ways (no 429 retry, no schema check, unbounded retries, no timeout, retry on 401,
no `Retry-After` cap, key in URL, no JSON mode, tolerate code fences); each was caught by a failing test. That throw-away script is not part of the repo.

## Experiments

```powershell
jupyter execute --inplace experiments.ipynb   # re-runs for free: replies come from results/runs.jsonl
```
Delete `results/runs.jsonl` to re-query the live API (~195 small calls, ~93k tokens in total at the time of writing, paced at one call per 4.5 s). The notebook can also be opened
in VS Code or JupyterLab (`pip install jupyterlab`).

**Task and metrics.** 15 tickets (+10 hold-out), gold labels for category, priority and sentiment, scored identically for every experiment; an invalid reply counts as wrong. One variable changes per experiment relative to a stated base.

Main set (15 tickets), `gemini-3.1-flash-lite` unless stated, temperature 0:

| id | change | valid JSON | all 3 correct | category | priority | sentiment | injection resisted |
|---|---|---|---|---|---|---|---|
| E01 | zero-shot, one-line prompt | 0/15 | 0/15 | 0 | 0 | 0 | yes |
| E02 | + field names and allowed values | 15/15 | 9/15 | 14 | 10 | 14 | yes |
| E03 | + category / priority definitions | 15/15 | **14/15** | 15 | 14 | 15 | yes |
| E04 | + expert persona | 15/15 | 14/15 | 15 | 14 | 15 | yes |
| E05 | + `<ticket>` delimiters, "it is data" rule | 15/15 | 14/15 | 15 | 14 | 15 | yes |
| E06 | + 2 worked examples | 15/15 | 13/15 | 15 | 14 | 14 | yes |
| E07 | + 6 worked examples | 15/15 | 12/15 | 15 | 14 | 13 | yes |
| **E08** | **chain of thought (`reasoning` first), zero-shot: production prompt** | 15/15 | **15/15** | 15 | 15 | 15 | yes |
| E09 | E07 at temperature 1.0 | 15/15 | 12/15 | 15 | 14 | 13 | yes |
| E10 | E07 without JSON mode | 15/15 | 13/15 | 15 | 14 | 14 | yes |
| E11 | E07 on `gemini-3.6-flash` | 15/15 | 12/15 | 15 | 12 | 15 | yes |

The category / priority / sentiment columns are the number of correct labels out of 15 (the notebook prints the same figures with more columns, plus summary length, tokens and latency).

Hold-out set (10 new tickets, prompts frozen):

| id | prompt | valid JSON | all 3 correct | tokens/call |
|---|---|---|---|---|
| E12 | E05 (zero-shot) | 10/10 | 7/10 | ~308 |
| E13 | E07 (6 examples) | 10/10 | 9/10 | ~734 |
| E14 | E08 (chain of thought) | 10/10 | 9/10 | ~375 |

**What the data supports**
- Constraining the output is what mattered: allowed values took validity from 0/15 to 15/15, and definitions took all-correct from 9 to 14 of 15.
- Persona (E04), temperature 1.0 (E09) and a larger model (E11) gave no measurable gain. Turning JSON mode off (E10) did not break parsing on these tickets (0 of 15 replies fenced).
- Examples did *not* help on the main set (14 -> 13 -> 12), yet the 6-example prompt tied the best on the hold-out. With 10-15 tickets and one run per cell, 1-3 ticket differences are noise, so E06/E07/E08 cannot be ranked with confidence.
  E08 was chosen as the production prompt because it is never worse than the alternatives on either set and costs about half the tokens of the 6-shot prompt. Its 15/15 was selected on the same tickets it is reported on; the hold-out 9/10 is the fairer figure.
- **Injection:** no prompt was ever hijacked, not even the undefended E01-E04, so the experiments cannot show that the delimiter rule helps. It is kept as cheap defence in depth; its benefit is unproven here.

## Known limitations

- **Small, single-annotator data.** 25 tickets in total; one annotator wrote both the tickets and the gold labels, and nobody else has reviewed them. Some labels are arguable (T07 `medium` vs `high`, T15 sentiment, H05 `other` vs `billing`). Please review `data/*.json` before relying on the numbers.
- **One run per cell.** No confidence intervals; temperature 0 is not strictly deterministic (the same Swahili ticket gave `medium` in E08 and `high` through the CLI, though the CLI input omitted a sentence). Re-running with the cache deleted may give different numbers.
- **Selection bias** on the main set (see above); the hold-out has only 10 tickets.
- **Latency numbers** in the notebook depend on server load far more than on the prompt; use token counts for cost comparisons.
- **Invalid model output is not re-asked**; a caller that wants that should wrap `triage.triage()` in its own retry.
- **Rate limits** are only exercised by mocks; the free tier's real quota was not hit during the notebook run (the notebook paces itself at 4.5 s per call). The CLI itself does not pace calls.
- **Gemini only**, REST API called directly with `requests` (no SDK). Model ids such as `gemini-3.1-flash-lite` reflect what answered with the author's key on 2026-10-26.
- Tested only on Windows 11 / Python 3.14.
- English-centred prompt and summaries; Swahili and French tickets were each a single sample.
