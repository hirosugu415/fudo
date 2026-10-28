# fudo — 都市創発シミュレーター

仕様は [SPEC.md](SPEC.md)、フェーズ1の計画は [docs/phase1_plan.md](docs/phase1_plan.md)。

## 準備（初回だけ）

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"      # macOS/Linux は .venv/bin/python
```

## ビューア

```
.venv\Scripts\fudo view
```

ブラウザで http://127.0.0.1:8765/ を開く（ポートは `config/viewer.yaml`）。ネット接続は不要（Three.js は同梱）。

1. 「マップを生成」：シード欄が空ならランダムなシードで世界を生成する（初回は Numba のコンパイルを含めて数十秒）。
2. 生成された世界の説明を読み、気に入らなければシード欄を空にしてもう一度押す。
3. 「シミュレート開始」：別プロセスで計算が進み、スライダーと再生ボタンで、計算済みの月まで流量と積雪の変化を見られる。

ドラッグで回転、右ドラッグで移動、ホイールで拡大縮小。マスをクリックすると数値が表示される。

## コマンドライン

```
fudo generate --seed 3                     # 世界を生成（world_id が表示される）
fudo describe --world <world_id>           # 世界の説明
fudo run --world <world_id>                # 既定の物理法則で実行
fudo run --world <world_id> --physics my_physics.yaml --run-seed 5   # 物理法則・実行シードを変えて比較
```

## テスト

```
.venv\Scripts\python -m pytest
```
