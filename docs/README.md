# Engineering notes

Notes for people working *on* Tempo9 rather than with it. If you are running
it, you want [`../manual/`](../manual/) and [`../examples/`](../examples/).

- [`swift-sdk.md`](swift-sdk.md) — the Swift package: products, what links
  the engine, and how to build against it
- [`support_matrix.md`](support_matrix.md) — which model/quant combinations
  were run, and what each one did
- [`agent_prompt_crash.md`](agent_prompt_crash.md) — a stack overflow in the
  tokenizer that only a real agent's prompt could reach, and the two
  convincing wrong answers that came first

The split is deliberate. A page explaining how to connect an app and a page
recording why a crash happened have different readers, and a directory that
holds both serves neither: the user wades through post-mortems, and the
post-mortem gets edited for tone it does not need.

Nothing here is a promise. These files record what was true when they were
written, including the wrong turns — that is most of their value.
