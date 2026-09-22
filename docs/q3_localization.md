# Question 3 — localisation, not translation

Two markets, one engine. `agent/flow.py` contains no product name, no script and
no business rule; everything market-specific is in `agent/markets/*.yaml`. Adding
a market is a YAML file and, if the sector differs, a corpus entry in
`kb/sources.py`.

| | Philippines | Indonesia |
|---|---|---|
| Market file | `agent/markets/fil_PH.yaml` | `agent/markets/id_ID.yaml` |
| Sector | Life insurance and bancassurance | Multifinance and consumer finance |
| Language | Taglish (Filipino with English code-switching) | Bahasa Indonesia, colloquial register |
| Flow | Lead qualification | Instalment reminder and collections support |
| Corpus | `prulife_ph` (English) | `fifgroup_id` (Bahasa) |
| ASR hint | `tl` | `id` |
| TTS | `fil-PH-BlessicaNeural` | `id-ID-GadisNeural` |
| Retrieval gate | 0.499 | 0.383 |

Recorded calls: `evidence/calls/q3_*.mp3`, transcripts in
`evidence/transcripts/`, results in `evidence/q3_call_results.json`.

---

## Philippines: three adaptations

### 1. Financial nouns stay in English

A translated script produces correct Filipino that nobody speaks. Real
bancassurance calls run in Taglish, where English carries the financial nouns
and Filipino carries the grammar around them.

- Translation: *"Ang inyong buwanang bayad ay nakatakda sa susunod na linggo."*
- What the agent says: *"Due na po yung premium payment ninyo next week."*

`premium`, `policy`, `rider`, `coverage`, `sum assured` and `grace period` are
never translated. A caller who hears *"buwanang bayad"* has to work out that it
means premium; a caller who hears *"premium"* does not.

### 2. `po` is structural, not decorative

`po` and `opo` mark respect in Filipino and their absence reads as rudeness, not
neutrality. Dropping them from a sales call to an older customer ends the call.
It appears in every agent turn, which is why it is a style rule rather than a
suggestion: *"Sino po ang kausap ko?"*, never *"Sino ang kausap ko?"*

### 3. Deferring to family is respected, not overcome

A Philippine objection with no direct equivalent in the English script is
*"kailangan ko munang itanong sa asawa ko"* — I need to ask my spouse first.
Sales training in other markets treats this as a stall to work through. In a
Philippine household a major financial decision genuinely is a family one, and
pushing against it costs the lead.

The `asking_family` objection exists only in the Philippine market file, and its
stance is to offer materials the family can discuss together rather than to
counter the objection at all.

---

## Indonesia: three adaptations

### 1. Register carries more than vocabulary

A collections call in Indonesia is not a demand. It opens by apologising for
disturbing the person — *"Mohon maaf mengganggu waktunya"* — uses *Bapak* or
*Ibu* throughout, and frames a late payment as a shared problem. Getting this
wrong offends in a way a mistranslated noun does not.

`kamu` is prohibited outright in the style rules. It is the informal "you" and
is close to insulting from a company to a customer.

### 2. Finance terms stay Indonesian, and they are not interchangeable

*cicilan, angsuran, tenor, denda, DP, jatuh tempo, pembiayaan, plafond, BPKB.*

Translating *tenor* to *jangka waktu* is technically correct and sounds wrong
from a finance company. *cicilan* and *angsuran* both mean instalment, but
*angsuran* is the formal one used in documents and *cicilan* is what customers
say. The agent opens with *angsuran* and follows the customer into *cicilan*.

### 3. Hardship is handled, not scripted past

The `cannot_afford` stance forbids the agent from offering a discount and
directs it to an approved hardship option. There is also a hard rule against
mentioning legal consequences, because that is both a compliance risk and the
fastest way to lose a customer who intends to pay.

This is visible in the recorded hardship call. Asked about a job loss, the agent
answers from the configured stance in Bahasa rather than refusing, because a
customer who has just been laid off should not hear *"I don't have that detail."*

---

## Fallback and escalation stay in language

The brief names unexpected English switching as a failure, and the first
recording of these markets did it. Escalation, refusal, slot-skip and decline
were hardcoded English strings in `flow.py`, so a call conducted entirely in
Bahasa answered in English at the handover — the single worst moment for it.

All four now live in each market file. `scripts/check_language_leak.py` guards
against the next one: it scans every configured line and every spoken literal in
`flow.py` for English markers, and drives a scripted call in each market to check
the turns actually produced. Fixing four strings would not have prevented a fifth
being added later.

```
python -m scripts.check_language_leak
```

---

## Cross-lingual retrieval

The Taglish market reads from an English corpus, which degrades retrieval in two
ways at once. BM25 can only match the English loanwords that survive in the
question — one token in a Filipino sentence — and the multilingual embedding
carries across less sharply than a same-language pair.

Measured on *"what happens if I miss a premium payment"*:

| | Top score | Chunk retrieved |
|---|---|---|
| English | 0.591 | `kb_payments_050_c25` (correct) |
| Taglish, direct | 0.510 | `kb_education_insurance_077_c05` (wrong) |
| Taglish, cross-lingual | **0.629** | `kb_payments_050_c25` (correct) |

The failure mode is worth naming: the direct Taglish score still cleared the
gate, so the agent did not refuse for lack of confidence. It answered from the
wrong passage, and the generation gate then refused a question the corpus could
answer. A confidence threshold cannot catch this, because the confidence was
fine — it was pointed at the wrong text.

`Retriever.search_cross_lingual` asks the question in the corpus language as
well and merges on the better score per chunk. Cost is one extra model call and
one extra embedding per question.

---

## Retrieval thresholds are per corpus

`MIN_RETRIEVAL_SCORE` was calibrated on English insurance prose and applying it
to Indonesian multifinance content was wrong: the same semantic match simply
scores lower there.

| Corpus | Weakest in-scope query | Gate |
|---|---|---|
| `prulife_ph` | 0.529 | 0.499 |
| `fifgroup_id` | 0.413 | 0.383 |

Derived by `python -m scripts.calibrate_thresholds`. A single global threshold
would have refused fair Indonesian questions outright. Neither corpus separates
answerable from out-of-scope by score alone, which is why the generation gate
exists in both.

---

## Known gaps

**No native-speaker validation.** This is the significant one. Terminology and
register are sourced from the markets' own public material and from documented
usage, and neither market's output has been reviewed by a native speaker. The
Taglish reads as plausible agent speech and the Bahasa register follows
collections convention, but plausible is not validated. Before any real
deployment both scripts need a native reviewer, and that is a person, not a
model.

**Gendered address is not tracked.** The Indonesian agent picks *Bapak* or *Ibu*
per turn without carrying the customer's gender through the call, so it can
address the same person both ways. Visible in the recorded hardship call, where
a caller named Siti Rahayu is addressed as *Bapak*. The fix is a gender slot
inferred at the name turn and carried in state; it is not implemented.

**Regional accent degrades sharply.** Measured in `docs/q3_asr_report.md`:
Javanese-accented speech scores 0.301 word error rate against 0.020 for Jakarta
standard, roughly fifteen times worse, concentrated in colloquial and
emotionally loaded speech. A customer in Yogyakarta explaining a job loss is the
call this system transcribes worst, which is the opposite of where accuracy can
be spared.

**The Indonesian corpus is thin.** Four records and 168 chunks, because the
source is a client-rendered site that ships the same state payload on every
page, so 20 of 24 pages deduplicated to exact copies. It covers products and
FAQs but has little on collections or hardship, which is why the hardship path
answers from the configured stance rather than from retrieval.

**Synthetic speech throughout.** Every recording is TTS. There is no packet
loss, background noise, crosstalk or microphone variation, so the ASR figures
are a floor rather than a prediction.
