# Per-species pipeline（種別モデル）

このディレクトリは、論文（「@@@」第「@@@」節）で説明している **per-species model（種別モデル）** を実装したものです。  
この設定では、**各樹種ごとに独立したモデルを個別に学習**し、その樹種から得られた画像パッチのみを使用します。  
交差種汎化を必要としないため、**同一ドメイン内性能の上限（upper-bound）**を与えるベンチマークとして位置づけられます。

## このパイプラインが行うこと

- 樹種ごとに **1つの Mask R-CNN モデル**を学習
- 対応する樹種専用チェックポイントを用いて推論を実行
- すべての出力を共通の `runs/` 構造に保存し、`evaluation/` による一貫した評価を可能にする

## 実行方法

### 1) 設定
`config.yaml` を編集し、パスや実験パラメータを設定してください。

確認すべき主な項目（実際のキー名は設定により異なる場合があります）：
- `data_root`：画像パッチのパス
- `ann_path` / `coco_json`：COCO形式アノテーションのパス（使用する場合）
- `species_list`：処理対象とする樹種ID／名称のリスト
- `out_dir`：出力ディレクトリ（推奨：`runs/per_species/`）
- `seed`：乱数シード
- `device`：`cuda` または `cpu`

### 2) 学習
レポジトリのルートから以下を実行します：

    python pipelines/per_species/train.py --config pipelines/per_species/config.yaml

期待される挙動：
- `species_list` に含まれる樹種を順に処理
- 各樹種ごとにモデルを学習
- 以下の構造でチェックポイントを保存：

    runs/per_species/<experiment_id>/<species_id>/ckpt/

### 3) 推論
レポジトリのルートから以下を実行します：

    python pipelines/per_species/inference.py --config pipelines/per_species/config.yaml

期待される挙動：
- 各樹種に対応するチェックポイントを読み込み
- 以下の構造で予測結果を保存：

    runs/per_species/<experiment_id>/<species_id>/preds/

## 出力（共通規約）

本パイプラインは、以下の共通 `runs/` 構造に従って出力を生成します：

    runs/
      per_species/
        <experiment_id>/
          <species_id>/
            ckpt/      学習済みモデル
            preds/     推論されたインスタンスマスク（および／または COCO 形式出力）
            scores/    評価結果（通常は evaluation/ により生成）

`evaluation/score_all.py` は、すべてのパイプラインでこの共通構造が維持されていることを前提としています。

## 注意事項

- このパイプラインは **同一樹種内での性能上限**を評価するためのベンチマークです。
- 樹種間の汎化性能を目的としたものではありません。
- `notebooks/per_species/` 以下の Jupyter Notebook は、確認・可視化・図表作成用です。実装の正本ではありません。

## 論文との対応関係

- 論文中のモデル名：**Per-species model**
- 対応するディレクトリ：`pipelines/per_species/`
- 関連する図表： "@@@"






# Per-species pipeline

This directory implements the **per-species model** described in the paper ("@@@": Section "@@@").  
In this setting, **separate models are trained independently for each species**, using only image patches derived from that species. This serves as an upper-bound benchmark for within-domain performance (no cross-species generalisation is required).

## What this pipeline does

- Trains one Mask R-CNN model **per species**
- Runs inference using the corresponding species-specific checkpoint
- Writes outputs to the common `runs/` structure so that `evaluation/` can score results consistently

## How to run

### 1) Configure
Edit `config.yaml` to set paths and experiment parameters.

Key items to check (names may differ depending on your config):
- `data_root`: path to image patches
- `ann_path` / `coco_json`: path to COCO annotations (if used)
- `species_list`: list of species IDs/names to iterate over
- `out_dir`: output base directory (recommended: `runs/per_species/`)
- `seed`: random seed
- `device`: `cuda` or `cpu`

### 2) Train
From the repository root:

    python pipelines/per_species/train.py --config pipelines/per_species/config.yaml

Expected behaviour:
- Iterates over species in `species_list`
- Trains one model per species
- Saves checkpoints under:

    runs/per_species/<experiment_id>/<species_id>/ckpt/

### 3) Inference
From the repository root:

    python pipelines/per_species/inference.py --config pipelines/per_species/config.yaml

Expected behaviour:
- Loads the checkpoint for each species
- Writes predictions under:

    runs/per_species/<experiment_id>/<species_id>/preds/

## Outputs (convention)

This pipeline follows the shared output convention under `runs/`:

    runs/
      per_species/
        <experiment_id>/
          <species_id>/
            ckpt/      trained checkpoints
            preds/     predicted instance masks (and/or COCO-style outputs)
            scores/    evaluation outputs (optional; usually written by evaluation/)

`evaluation/score_all.py` assumes outputs are organised consistently across pipelines.

## Notes

- This pipeline is designed as a **within-domain upper-bound** benchmark: it maximises within-species fit but is not intended to generalise across species.
- Jupyter notebooks under `notebooks/per_species/` are provided for inspection and figure generation; they are not the authoritative implementation.

## Paper mapping

- Model name in paper: **Per-species model**
- Corresponding directory: `pipelines/per_species/`
- Related figures/tables: "@@@"

