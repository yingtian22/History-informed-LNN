# HiLNN Data

## Layout

```text
data/
├── raw/                          # optional raw exports
└── processed/
    ├── pendulum/
    ├── damped_pendulum/
    └── variable_pendulum/
```

## Trajectory `.npz` fields

| Field    | Shape           | Description              |
|----------|-----------------|--------------------------|
| states   | [N, T, 2]       | [theta, theta_dot]       |
| q        | [N, T, 1]       | generalized coordinate   |
| qdot     | [N, T, 1]       | generalized velocity     |
| times    | [T]             | time grid                |
| params   | [N, 4]          | [g, l, m, c] (analysis)  |
| energy   | [N, T]          | mechanical energy        |

## Window `.npz` fields (L=8, H=32)

`hist_q`, `hist_qdot`, `hist_state`, `future_q`, `future_qdot`, `future_state`, `params`, `energy_hist`, `energy_future`.
