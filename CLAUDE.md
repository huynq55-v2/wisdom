# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

### Build, format, lint, test
- `cargo build`
- `cargo build --release`
- `cargo fmt --all`
- `cargo fmt --all --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- `cargo test --test repetition_tests`
- `cargo test test_perft_depth_4 -- --nocapture`
- `cargo test --test repetition_tests test_diagram_4_mutual_perpetual_check -- --nocapture`

### Engine / protocol / search
- `cargo run --bin wisdom` (UCCI engine, CPU backend)
- `cargo run --bin wisdom -- gpu` (UCCI engine, WGPU backend)
- `cargo run --bin wisdom -- perft 4` (perft from start position)

### GUI
- `cargo run --bin gui`
- `src/bin/gui.rs` spawns `./wisdom gpu` as a subprocess. If `./wisdom` does not exist in repo root, build/link it first:
  - `cargo build --bin wisdom`
  - `ln -sf target/debug/wisdom wisdom`

### Training / data pipeline binaries
- `cargo run --bin selfplay` (full self-play + training loop; long-running)
- `cargo run --bin build_dataset`
- `cargo run --bin augment -- <input_csv>`
- `cargo run --bin clean_dataset -- <input_csv>`
- `cargo run --bin inspect`
- `cargo run --bin convert_mpk`

## Architecture overview

### Core engine model (board + moves + rules)
- `src/board.rs` stores board state on a 256-cell 16-stride array (0x88-style validity checks), plus king squares, side-to-move, and Zobrist key.
- `src/move.rs` encodes moves as `u16` using dense 0..89 square indices (`from`, `to`, capture bit).
- `src/movegen.rs` generates **pseudo-legal** captures/quiets. Legality is enforced by callers with make/unmake + `kings_facing` + `is_in_check`.
- Repetition handling is rule-heavy and shared across engine/self-play/GUI:
  - `HistoryEntry { hash, is_check, chased_set, is_reversible }`
  - `judge_repetition` distinguishes perpetual idle/chase/check and returns `Win/Loss/Draw/Undecided`.

### Search + NN inference stack
- `src/mcts.rs` implements preallocated atomic MCTS with PUCT, virtual loss, sparse child priors, optional Dirichlet root noise, and multithreaded playout support.
- `src/tt.rs` is a transposition table keyed by Zobrist; each entry stores `(value, sparse policy Vec<(move, logit)>)`.
- `src/eval_queue.rs` runs a background batching worker: many search threads submit `EvalRequest`, then one NN forward pass serves a batch.
- `src/nn.rs` defines a ResNet-style model (input conv + 7 residual blocks, policy head, value head). Action space is `90 * 90 = 8100`.

### Canonical perspective conventions (important)
- `board_to_tensor` always maps to side-to-move perspective (planes 0..6 = side to move, 7..13 = opponent).
- For black-to-move, board coordinates are rotated 180° before tensorization.
- Policy index conversion for search uses `move_to_index_perspective`; raw absolute mapping uses `move_to_index`.
- Any change in move indexing or tensor orientation must be kept consistent across:
  - `src/nn.rs` (mapping)
  - `src/mcts.rs` (policy lookup)
  - `src/bin/selfplay.rs` (training target generation/augmentation)
  - dataset utilities in `src/bin/*`

### UCCI and binary roles
- `src/bin/wisdom.rs` is the main engine entrypoint:
  - default: CPU (`NdArray`)
  - `gpu`: WGPU backend
  - `perft <depth>`: perft mode
- `src/ucci.rs` implements the UCCI loop (`position`, `go`, `bestmove`, etc.) and calls MCTS.
- `src/bin/gui.rs` is a Macroquad board UI that talks to the engine subprocess over UCCI text I/O.
- `src/bin/selfplay.rs` is the unified training pipeline:
  - self-play game generation via MCTS + EvalQueue
  - replay buffer persistence at `wisdom_models/replay_buffer.csv`
  - iterative training/checkpointing into `wisdom_models/xiangqi_net_ckpt_*` and `wisdom_models/xiangqi_net_*.mpk`

## Data/model utility expectations
- `build_dataset` expects `pikafish_FEN.csv` and writes `pikafish_FEN_processed.csv`.
- `augment` defaults to `xiangqi_dataset.csv`, outputs `xiangqi_dataset_augmented.csv`.
- `clean_dataset` defaults to `xiangqi_dataset.csv`, outputs `xiangqi_dataset_cleaned.csv`.
- `convert_mpk` expects `xiangqi_net_weights_latest.safetensors`, outputs `xiangqi_net_latest.mpk`.
- Runtime engine model loading in `wisdom.rs` looks for `xiangqi_net_weights.mpk` (via `MODEL_PATH = "xiangqi_net_weights"`).

## Repo note
- No `README.md`, `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` were found at the time this file was created.