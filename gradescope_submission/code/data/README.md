# Data

No datasets are tracked. Regenerate SD datasets via the training CLI, e.g.:

```bash
python scripts/train.py dataset --episodes 250 --horizon 200 --output data/SD_dataset_clean.pt
```

Tune episodes/horizon/seed as needed; metadata is stored alongside the saved file.
