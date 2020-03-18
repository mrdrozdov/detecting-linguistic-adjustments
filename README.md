# Make Data

```
# Creates Raw Data.
python make_dataset_0.py

# Split into train/dev/test.
python split_data.py
```

# Make Data with DIORA (exploratory)

```
python make_dataset_from_diora.py
```

# Train Model

Only GRU baseline works right now. Larger batch size helps performance.

```
python train_baseline.py --cuda --batch_size 128
```

# Dependencies

```
pip install -r requirements.txt
```
