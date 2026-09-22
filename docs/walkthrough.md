# Video walkthrough guide

This file is for you, not for them. It is your script, what to click, what to
say, and what to avoid.

**Target length: 8–10 minutes.** Do not go over 12. They are reviewing many of
these.

---

## Before you start recording

```bash
cd "Darwix Voice Agent"
.venv/bin/python -m uvicorn agent.server:app --port 8000
```

Wait for it to finish loading (about ten seconds), then:

- Open **http://127.0.0.1:8000** in Chrome
- Open a second tab on **http://127.0.0.1:8000/kb**
- Open a third tab on **http://127.0.0.1:8000/live**
- Test your mic once before recording
- Close Slack, email, anything that pops up
- Have `docs/retrieval_results.md` open in an editor as a fourth tab

Record your screen with your voice. You do not need your face.

---

## The one-sentence version

If you say nothing else, say this:

> "The interesting problem here wasn't making it talk. It was making it refuse."

Everything in this submission supports that sentence. Come back to it at the end.

---

## Part 1 — Open (45 seconds)

**Show:** the call console landing page, the three market cards.

**Say:**

> "This is one voice engine serving three markets. Philippines in English,
> Philippines in Taglish, and Indonesia in Bahasa for consumer finance. The
> engine code has no product names, no scripts and no business rules in it —
> those all live in config files, one per market. Adding a fourth market is a
> YAML file."

Don't click anything yet. Let them see the three cards.

---

## Part 2 — The voice agent (2 minutes)

**Click:** the first card, Pru Life UK Philippines, English.

Wait for the greeting to play. Let them hear it.

**Then say these, one at a time, into the mic:**

| Say | What to point out |
|---|---|
| "Yes, I have a few minutes" | Watch the Name slot light up on the right |
| "My name is Marco" | Slot fills, moves to the next question |
| "What happens if I miss a premium payment?" | **Stop here.** Point at the green *grounded* badge and the chunk id |

**Say:**

> "That answer came out of the knowledge base, not the model's memory. The green
> tag is the exact passage it used. Thirty-one day grace period — that is a real
> number from their real site, not something the model invented."

**Then say:**

> "My wife and two kids depend on me."

A nudge appears in the **Live coaching** panel on the right.

**Say:**

> "That's the Q4 nudge engine running on this live call, not a recording. It
> heard the customer mention a family and flagged it as an opening the agent
> talked past. Same detectors as the recorded pipeline — and here it knows
> exactly who said what, because the agent's words are generated locally and
> mine came back from the transcriber already labelled."

**Then say:**

> "What is my current policy account balance?"

**Say:**

> "It refuses. It cannot know that, so it says so and offers a person. That is
> the behaviour I spent the most time on."

**Then:**

> "Can I speak to a real person please?"

It escalates. Click **End call** and let the summary sheet rise.

**Say:**

> "Lead created, scored, and an escalation record written for whoever picks up.
> Latency per stage is measured on every call."

---

## Part 3 — The knowledge base and the two gates (2.5 minutes)

This is the strongest part of the submission. Slow down here.

**Switch to the `/kb` tab.**

**Say:**

> "This is the knowledge base behind that answer. A hundred and thirty-nine pages
> scraped from Pru Life UK, cleaned down to a hundred and sixteen records and
> eight hundred and fourteen chunks. Boilerplate removed, duplicates dropped,
> personal data redacted."

**Click the example:** *"What happens if I miss a premium payment?"*

Point at the two green gate boxes.

**Say:**

> "Two gates. The first is a score threshold. The second has the model actually
> read the passages and decide whether the answer is in them."

**Now click:** *"What is my current policy account balance?"*

**Point at the score. This is your moment.**

**Say:**

> "Look at that. It scored 0.498. The gate is 0.499. One thousandth below.
>
> And here's the thing I found when I measured this properly — the scores for
> questions it can answer and questions it can't **overlap**. Asking for the
> weather in Cebu pulls up a typhoon advisory and scores higher than some real
> questions about premiums, because the words genuinely do overlap.
>
> So no threshold separates them. That is why there are two gates. A score can't
> read. The model can. Neither one is enough on its own."

**Scroll down** to the score bars.

**Say:**

> "Each bar is the semantic score and the keyword score, which is why it's hybrid
> — embeddings catch paraphrasing, keyword search catches product names people
> say word for word. The amber line is the gate."

---

## Part 4 — Localisation (1.5 minutes)

**Back to the call console.** Click **Pru Life UK Philippines · Taglish**.

**Say:**

> "Same engine. Different config."

**Type or say:** *"Ano po ang mangyayari kung hindi ko mabayaran ang premium?"*

**Say:**

> "That's the same question in Taglish, and note it kept 'premium' in English.
> That's deliberate — real agents in Manila don't translate financial terms.
> A translated script would say 'buwanang bayad' and no one talks like that.
>
> One thing worth flagging: the knowledge base here is in English and the caller
> is speaking Taglish. When I first built this it retrieved the wrong passage —
> not because the score was low, but because it was confidently pointed at the
> wrong text. So the question gets asked in the corpus language too and the
> results are merged."

**Switch market to FIFGROUP Indonesia.** Point at the qualification panel.

**Say:**

> "Different sector entirely — consumer finance, not insurance. So the questions
> change: instalment status, obstacle, plan to pay. And it reads from a different
> knowledge base, because a Philippine life insurance answer has no business on
> an Indonesian collections call."

---

## Part 5 — Live nudges (2 minutes)

**Switch to the `/live` tab.**

**Say:**

> "This is the fourth piece. Analysis while the call is still running, not after."

**Click:** *Skipped disclosure and risky statement*. **Let it play.** Don't talk
over the first few seconds — let them see the transcript filling in.

When the first nudge appears:

**Say:**

> "That fired four seconds in. The agent said 'investment' and never gave the
> disclosure, so it tells them to. The red one there — the agent promised
> guaranteed returns, and that correction landed in about a quarter of a second.
> They're still mid-sentence."


The exact millisecond figure shows on the nudge card and changes slightly every
run, so read it off the screen rather than quoting a number from memory.

Point at the suppression panel.

**Say:**

> "The number I care about is this one. Twenty-seven signals found, three shown.
> Detecting things is easy. Not burying the agent is the hard part."

**Click New run**, then scroll down and click **"Or speak into it yourself"**.

**Say:**

> "You can also just talk to it."

Set the role to **I am the agent** and say:

> "Our investment funds have guaranteed returns and there is no risk at all."

Two red nudges appear.

**Say:**

> "Missing disclosure, and guarantee language. Both on my own voice, in real
> time. Now switch roles."

Click **I am the customer** and say:

> "My wife and two kids depend on me."

**Say:**

> "Different speaker, different detector. That's why the role is a button rather
> than something the system guesses — a missing disclosure only counts against
> the agent, and a family mentioned is only an opening when the customer says it.
> Guessing would fire half the rules on the wrong person."

**Click New run, then:** *Noisy and ambiguous call*.

**Say:**

> "This one is the real test. Forty-three seconds of small talk and hesitation
> and nothing actionable. It has to stay silent — and it does."

Let it finish so they see the green **Pass**.

---

## Part 6 — What I got wrong (1.5 minutes)

**Do not skip this.** It is the part that makes the rest credible.

**Say:**

> "Three things I got wrong and fixed.
>
> One — I picked the refusal threshold by intuition. Then I measured it and it
> was wrong, and the fix wasn't a better number, it was a second gate.
>
> Two — I added a language hint to the speech recognition because I assumed
> auto-detect misfires on code-switched speech. I tested it across eighty-five
> trials and it barely moved the numbers. I corrected the comment in my own code
> to say so rather than leave a claim the evidence doesn't support.
>
> Three — my nudge system told an agent to quote a projected return, four seconds
> after a compliance rule told them returns are never guaranteed. The model was
> being helpful about sales and had no idea it was recommending the exact thing
> the rule above it exists to prevent. Model-written nudges now get checked
> before a human sees them."

**Then, on limitations:**

> "The honest gap: I don't speak Tagalog or Bahasa. The terminology and register
> are researched from the companies' own material, and the bot never switches to
> English, but I can't tell you it sounds natural to a native speaker. My test
> cases are limited by the same thing. That needs a native reviewer before it
> goes anywhere near a customer.
>
> And regional accents degrade badly — Javanese-accented speech is fifteen times
> worse than Jakarta standard, concentrated in exactly the emotional calls where
> you need accuracy most. It's measured in the repo."

---

## Part 7 — Close (30 seconds)

**Say:**

> "Everything is on free tiers. The recordings, transcripts and reports are all in
> the repo, and every number in the docs is regenerated by a script — none of them
> are typed by hand.
>
> Like I said at the start: the hard part wasn't making it talk. It was making it
> refuse."

---

## Show this

- The green **grounded** badge with a chunk id — proves it isn't making things up
- **0.498 against a gate of 0.499** — your single best moment
- The **noisy call staying silent** — harder than making nudges fire
- The **Taglish answer** keeping "premium" in English
- The **Indonesian market** having different qualification questions
- The **suppression counter** — 27 signals, 3 shown
- A **nudge firing on your own live voice**, in both roles

## Don't show this

- **Your `.env` file.** Ever. Your API key is in it.
- The terminal while the server is starting — it is ten seconds of nothing
- Code files. They will read the code themselves. The video is for behaviour.
- The Indonesian hardship call unless you flag the *Bapak*/*Ibu* bug yourself
- Anything you cannot explain if asked

## If something breaks on camera

Don't restart. Type the reply instead of speaking it — same engine, and the
transcript still shows everything. If a page is stuck, the recordings in
`evidence/calls/` cover every scenario and you can talk over those instead.

---

## Questions they will probably ask

**"How do you stop it hallucinating?"**
Two gates. Retrieval score first, then the model reads the passages and declines
if the answer isn't there. The scores overlap, so neither works alone.

**"Why this model?"**
Benchmarked four on latency, multilingual quality, JSON reliability and grounded
refusal. Qwen was three times faster and matched on quality. It's in
`docs/model_selection.md` and the script regenerates it.

**"Why not Llama?"**
Not available on my Groq account. I picked from what the key could actually reach.

**"What breaks at 10x?"**
The free tier, not the architecture. Two speech requests per four-second chunk is
about five per second against a twenty-per-minute limit. Fix is a streaming
connection per call.

**"Why no phone number?"**
Every free telephony trial needs a card. I chose something reproducible over
something impressive. No PSTN leg means jitter and codec loss are untested, and
that's in the limitations.

**"How would you improve it?"**
Streaming speech synthesis first — it's 1,700 ms and dominates everything else.
Then a native speaker review of both localised scripts.
