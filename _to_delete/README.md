# FunASR configs

Training and finetuning are done with [FunASR](https://github.com/modelscope/FunASR).
No FunASR source is vendored here; only the configs that reproduce our runs.

Drop your run YAMLs here, e.g.:

- `paraformer_offline.yaml` — offline topline (750h + aug)
- `paraformer_streaming.yaml` — causal streaming baseline, chunk sweep 600 ms → 3 s
- `dynamic_block_paraformer.yaml` — proposed VAD-aligned block-online model
- `domain_menstrual.yaml` — medical-domain adaptation

```bash
pip install -U funasr
funasr-train --config configs/dynamic_block_paraformer.yaml
```

Key knobs for the block-online model:

| Setting | Value | Why |
|---|---|---|
| VAD pause threshold | 200 ms | Marks a communicative boundary |
| Max block length | 3.0 s | Cap for dense agglutinative chains |
| Encoder attention | full bidirectional within block | Restores hindsight over suffixes |
| LID tokens | off | Discrete `<en>`/`<bn>` tags destabilise CIF |
