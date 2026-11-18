# Looped transformer replay

Start with the [remote reproduction journal](../../docs/docs/developer_guide/training_free_looped_transformer_journal.md).
It contains checkout/bootstrap commands, the experiment matrix, all recovered
historical scores, exact output contracts, and unresolved historical inputs.

From the checkout root:

```bash
python benchmark/looped_transformer/reproduce.py list
python benchmark/looped_transformer/reproduce.py run --dry-run
python benchmark/looped_transformer/reproduce.py prepare
PYTHONPATH=python python benchmark/looped_transformer/reproduce.py run \
  --experiments qwen3-8b-suitefull llama31-8b-suitefull --gpu 0 --port 30100
python benchmark/looped_transformer/reproduce.py summarize
```

Use the committed Linux/Python 3.12 hash lock and asset lock. Model and dataset
revisions in the latter were resolved for this handoff; historical revisions
were not archived. Replaying the inputs generates new results to compare with
`historical-results.json`; it does not establish exact historical equivalence.

Two early experiments lack their original loop JSON. They fail before startup
unless an explicit reconstructed override is supplied. The author HF reference
additionally requires an archived patch and its SHA256 because the linked
repository is currently unavailable.

No GPU evaluation was repeated when preparing this package. The dataset and
orchestration checks described in the journal do not measure model accuracy.
