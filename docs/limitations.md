# Limitations

What this system does not do, and what I could not test. Written plainly because
a reviewer will find these anyway, and it is better that I found them first.

---

## The big one: I don't speak the languages

I don't speak Tagalog or Bahasa Indonesia. Not a little — not at all.

Everything in the Q3 markets is researched: terminology taken from the real
companies' own public material, register modelled on how those markets actually
run collections and bancassurance calls, politeness conventions read up on rather
than known.

What that means in practice:

- The Taglish **reads** like plausible agent speech. I cannot confirm it
  **sounds** like it.
- I could not judge whether a phrase is natural, stiff, or accidentally rude. A
  native speaker would catch in ten seconds what I cannot catch at all.
- My test cases are limited by the same gap. I wrote what I believed a caller
  would say, which is not the same as knowing what they say. Real calls contain
  slang, regional variation and elisions I have no way to anticipate.
- I verified what I could verify mechanically: that the ASR transcribes the
  audio back correctly, that terminology appears where it should, that the bot
  never switches to English.

Before any of this went near a real customer, both scripts need a native
reviewer. That is a person, not a model, and not me.

I'd rather say this clearly than have it look like I tested something I didn't.

---

## Regional accents degrade badly

Measured, in [`q3_asr_report.md`](q3_asr_report.md):

| Accent | Word error rate |
|---|---|
| Jakarta standard | 0.020 |
| Sundanese | 0.134 |
| Javanese | 0.301 |

Javanese-accented speech is roughly **fifteen times worse**, and the damage
concentrates in colloquial and emotional speech rather than formal speech.

So the call this system transcribes worst is a customer in Yogyakarta explaining
they have lost their job. That is exactly backwards from where accuracy matters,
and it is the single thing I would fix first in the Indonesian market.

---

## Everything is synthetic speech

Every recording is text-to-speech. There is no background noise, no packet loss,
no codec compression, no crosstalk, no cheap phone microphone.

The word error rates are therefore a **floor**, not a prediction. Real audio will
be worse and I don't know by how much.

The specific risk isn't that transcription gets a bit worse. It's that a partly
wrong transcript still reads as fluent, so a compliance rule can match a phrase
nobody actually said.

---

## No real telephony

The brief allowed "a callable number or a web calling interface" and I chose the
browser.

Honest reason: every free telephony trial needs a credit card, and I wanted a
demo you can reproduce without one. The trade-off is real — no PSTN leg means
jitter, latency variation and codec degradation are all untested.

---

## The Indonesian corpus is thin

Four records, 169 chunks, against 116 records and 814 chunks for the Philippines.

The source site renders in the browser and ships the same state payload on every
page, so 20 of 24 pages deduplicated to exact copies. That deduplication was
correct — the pages really were identical — but it leaves a small corpus.

It covers products and FAQs well and collections poorly, which is why the
hardship path answers from the configured stance rather than from retrieval.

---

## Gendered address is not tracked

The Indonesian agent picks *Bapak* or *Ibu* per turn without carrying the
customer's gender through the call, so it can address the same person both ways.

You can hear this in `evidence/calls/q3_id_02_hardship_javanese_accent.mp3` —
a caller named Siti Rahayu gets called *Bapak*.

The fix is a gender slot inferred at the name turn and held in state. It isn't
implemented. In the Indonesian market this is not a cosmetic bug.

---

## Free tier is the real ceiling

At ten concurrent calls the architecture is fine and the free tier is not: two
speech-to-text requests per four-second chunk works out at roughly five requests
per second against a limit of twenty per minute.

The fix is a streaming connection per call instead of discrete uploads, which
removes the per-request overhead entirely. Rules and nudge control don't move —
they're local and measured in microseconds.

---

## Speech synthesis is slow

About 1,700 ms, against 38 ms for retrieval and 415 ms for generating the answer.
It dominates everything else.

A caller notices this as a pause before the agent starts talking. Streaming
synthesis — starting to speak before the sentence is finished — is the first
performance work I would do.

---

## Retrieval thresholds are calibrated on small sets

Each corpus has its own gate, derived from its own evaluation set:

| Corpus | Weakest answerable query | Gate |
|---|---|---|
| Philippines | 0.529 | 0.499 |
| Indonesia | 0.454 | 0.424 |

Those sets are small — a dozen or so queries each. They are enough to show the
thresholds must differ per corpus, which was the point. They are not enough to
pin the numbers precisely, and I would expect both to move with more data.

---

## Things I know are approximations

- **Speaker separation** works because the test recordings are stereo with one
  speaker per channel, which is how contact centres record. Mono input is handled
  but marked unattributed rather than guessed, because every agent-side
  compliance rule depends on knowing who spoke.
- **The silence gate** that stops Whisper hallucinating on empty audio is tuned
  against clean recordings. On a real line it would admit line noise and need
  recalibrating.
- **Lead scoring** is deliberately simple arithmetic, not a model. A sales team
  has to be able to see why a lead ranked where it did.
- **The escalation webhook** writes locally when no endpoint is configured. A
  dropped escalation means a caller who asked for a human never reached one, so
  it fails loudly rather than silently.
