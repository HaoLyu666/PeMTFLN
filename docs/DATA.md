# Data Notes

The model expects preprocessed platoon arrays saved as NumPy `.npz` files.

## Expected Keys

- `train_data`
- `val_data`
- `test_data`

Each split should have shape:

```text
(num_samples, 6, 41, 10)
```

The default setting uses 21 history steps and 20 prediction steps.

## Feature Order

| Index | Feature |
| ---: | --- |
| 0 | Platoon id |
| 1 | Gap |
| 2 | Speed |
| 3 | Speed difference |
| 4 | Acceleration |
| 5 | PET |
| 6 | SSDD |
| 7 | Vehicle length |
| 8 | Preceding vehicle length |
| 9 | Preceding vehicle speed |

The loader builds:

- History input: `[gap, speed, speed difference, acceleration, SSDD, vehicle length, preceding vehicle length]`
- Future target: `[gap, speed, SSDD]`
- External input: preceding vehicle speed of the leader

## Included Files

- `data/sample_highsim.npz`: tiny HIGH-SIM-format split for smoke tests.
- `data/pemtfln_prediction_sample.npz`: saved prediction/parameter arrays for plotting or inspection.

The full processed dataset is intentionally excluded from GitHub. Place it at `data/platoons_data_split.npz` and pass `--data data/platoons_data_split.npz` when running evaluation or training.
