# Architecture

One voice engine. Three market configurations. Two knowledge bases. The live
nudge pipeline taps the same conversation the agent is already handling.

It runs two ways. On a recorded call it splits speakers by stereo channel and cuts
each channel into utterances with a voice activity detector - an utterance opens
when someone starts talking and closes when they pause - never looking at audio
that has not played yet. On a live call it consumes turns, where attribution is
exact: the agent's words are generated here and the caller's come back from the
transcriber already labelled. Same detectors, same suppression, same safety
check; only the segmentation differs.

That reuse is the main design decision. Q1, Q3-Philippines and Q3-Indonesia are
not three systems — they are one system reading three config files.

---

## The whole thing

```mermaid
flowchart TB
    subgraph input [" "]
        MIC["Browser mic<br/><i>or a recorded call</i>"]
    end

    ASR["<b>Speech to text</b><br/>Whisper large v3 turbo<br/><i>language hinted per market</i>"]

    subgraph agent ["AGENT PATH — Q1 and Q3"]
        direction TB
        FLOW["<b>State machine</b><br/>greet · qualify · answer<br/>object · escalate"]
        MKT["<b>Market config</b><br/>en_PH · fil_PH · id_ID<br/><i>script, voice, rules, slots</i>"]
        RET["<b>Hybrid retrieval</b><br/>embeddings + BM25"]
        G1{"<b>Gate 1</b><br/>score vs threshold"}
        G2{"<b>Gate 2</b><br/>does the text<br/>answer it?"}
        ANS["<b>Grounded answer</b><br/><i>cited</i>"]
        REF["<b>Refuse</b><br/><i>offer a human</i>"]
    end

    subgraph kb ["KNOWLEDGE BASE — Q2"]
        direction TB
        SCRAPE["scrape<br/><i>robots.txt honoured</i>"]
        CLEAN["clean<br/><i>boilerplate, dupes, PII</i>"]
        CHUNK["chunk<br/><i>on headings</i>"]
        INDEX[("index<br/>814 + 169 chunks")]
        SCRAPE --> CLEAN --> CHUNK --> INDEX
    end

    subgraph live ["LIVE PATH — Q4"]
        direction TB
        SPLIT["<b>Split channels</b><br/><i>agent left, customer right</i>"]
        RULES["<b>Rules</b><br/>disclosures, risky phrases<br/><i>0.2 ms</i>"]
        JUDGE["<b>Model judge</b><br/>frustration, intent<br/><i>~215 ms</i>"]
        SUPP["<b>Nudge control</b><br/>dedupe · cooldown · expiry<br/>safety check · priority"]
        DASH["<b>Dashboard</b><br/><i>nudge in ~600 ms</i>"]
        SPLIT --> RULES --> SUPP
        SPLIT --> JUDGE --> SUPP --> DASH
    end

    TTS["<b>Text to speech</b><br/>native Filipino and<br/>Indonesian voices"]
    OUT["Caller hears the reply"]
    ACT["<b>Business actions</b><br/>lead created · escalation written"]

    MIC --> ASR
    ASR --> FLOW
    ASR --> SPLIT
    MKT -.-> FLOW
    FLOW --> RET
    INDEX -.-> RET
    RET --> G1
    G1 -->|below| REF
    G1 -->|clears| G2
    G2 -->|not in the text| REF
    G2 -->|found it| ANS
    ANS --> TTS
    REF --> TTS
    TTS --> OUT
    FLOW --> ACT

    classDef gate fill:#2a1f08,stroke:#F2C14E,stroke-width:2px,color:#F2C14E
    classDef good fill:#0d2419,stroke:#35D6A4,stroke-width:1.5px,color:#35D6A4
    classDef warn fill:#2a1410,stroke:#FF8A3D,stroke-width:1.5px,color:#FF8A3D
    classDef store fill:#141a2e,stroke:#6E8BFF,stroke-width:1.5px,color:#9AB0FF
    class G1,G2 gate
    class ANS good
    class REF warn
    class INDEX store
```

---

## The two gates

This is the part worth understanding, because it is the answer to "how do you
stop it hallucinating".

```mermaid
flowchart LR
    Q["Question"] --> R["Retrieve<br/>top 8 passages"]
    R --> S{"Fused score<br/>above the<br/>threshold?"}
    S -->|"no"| X["Refuse"]
    S -->|"yes"| M{"Model reads the<br/>passages. Is the<br/>answer in them?"}
    M -->|"no"| X
    M -->|"yes"| A["Answer, with a citation"]

    classDef gate fill:#2a1f08,stroke:#F2C14E,stroke-width:2px,color:#F2C14E
    classDef good fill:#0d2419,stroke:#35D6A4,stroke-width:1.5px,color:#35D6A4
    classDef warn fill:#2a1410,stroke:#FF8A3D,stroke-width:1.5px,color:#FF8A3D
    class S,M gate
    class A good
    class X warn
```

Why two and not one: the scores for answerable and unanswerable questions
**overlap**. Asking for tomorrow's weather in Cebu retrieves a typhoon advisory
with a higher score than several legitimate questions about premiums, because
the vocabulary genuinely does overlap.

No threshold separates them. A score cannot read. The model can. Both are needed,
and neither is sufficient alone. The measurements are in
[`retrieval_results.md`](retrieval_results.md).

---

## One engine, three markets

```mermaid
flowchart LR
    subgraph cfg ["Config files — the only thing that differs"]
        A["en_PH.yaml<br/><i>English · insurance</i>"]
        B["fil_PH.yaml<br/><i>Taglish · bancassurance</i>"]
        C["id_ID.yaml<br/><i>Bahasa · multifinance</i>"]
    end

    E["<b>agent/flow.py</b><br/>no product names<br/>no scripts<br/>no business rules"]

    A --> E
    B --> E
    C --> E

    E --> K1[("prulife_ph<br/>814 chunks")]
    E --> K2[("fifgroup_id<br/>169 chunks")]

    classDef store fill:#141a2e,stroke:#6E8BFF,stroke-width:1.5px,color:#9AB0FF
    class K1,K2 store
```

Adding a fourth market is a YAML file. If it needs a different corpus, that is
one entry in `kb/sources.py`.

---

## Where the time goes

Measured, not estimated. Full numbers in
[`latency_report.md`](latency_report.md).

| Stage | Typical |
|---|---|
| Speech to text | ~550 ms |
| Retrieval | 38 ms |
| Answer generation | ~415 ms |
| Rule detectors | 0.2 ms |
| Nudge control | 0.1 ms |
| Speech synthesis | ~1,700 ms |

Speech synthesis dominates everything else by a wide margin. It is the first
thing I would replace, with a streaming voice that starts speaking before the
sentence is finished.

---

## Repository map

```
core/      shared services — model access, speech, timing
kb/        Q2 — scrape, clean, chunk, index, retrieve, answer
agent/     Q1 and Q3 — state machine, markets, actions, web server
live/      Q4 — signals, nudge control, real-time streaming
web/       the three browser consoles
scripts/   every script that produced a number in docs/
evidence/  recordings, transcripts, results
docs/      the reports those scripts generate
```
