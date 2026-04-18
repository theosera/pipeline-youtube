# pipeline-youtube

[![CI](https://github.com/theosera/pipeline-youtube/actions/workflows/main.yml/badge.svg)](https://github.com/theosera/pipeline-youtube/actions/workflows/main.yml)

YouTube プレイリスト → Obsidian Vault 自学自習用学習レポート生成パイプライン。

NotebookLM に動画 URL を 1 本ずつ手動で貼り付けて要約を Obsidian に転記する運用を自動化する。プレイリスト URL を 1 本投げると、プレイリスト内の全動画を以下 5 工程に通して、学習しやすいレポートと分野横断のハンズオンを Vault に保存する。

**動画単位 (01〜04)**
- **01_Scripts**: タイムスタンプ付き文字起こし (YouTube 純正字幕 → 自動生成字幕 → ローカル Whisper の 3 段フォールバック)
- **02_Summary**: 意味単位タイムスタンプ範囲付き要約
- **03_Capture**: 要点タイムスタンプに対応する動画フレーム抽出 (WebP アニメーション)
- **04_Learning_Material**: 上記 3 工程を「時系列 → キャプチャ画像 → 要点」3 点セットでテーマ単位に再構成

**プレイリスト単位 (05)**
- **05_Synthesis**: プレイリストの動画数 ≥ 3 本の時、Agent Teams (α/β/γ/leader) で 04 全体を横断統合し、章別ハンズオン md 群 + 00_MOC.md + 重複度スコア JSON を出力する

AI 呼び出し (stage 02/04/05) は **`claude -p` headless CLI を subprocess で呼び出し**、Claude Pro/Max の OAuth セッションを使います。`ANTHROPIC_API_KEY` は不要。`claude login` 済であればそのまま動作します。

### 実行環境の前提 (重要)

このパイプラインは **ローカル実行専用** の設計です。理由は:

- stage 02/04/05 は `claude` CLI を subprocess で呼び、OAuth セッション (`~/.claude/`) に依存
- GitHub Actions などの CI 環境では `claude login` できないため、これら 3 段は実行不可
- API キー (`ANTHROPIC_API_KEY`) を使う設計には意図的に切り替えていません (Claude Pro/Max 定額を活かすため)

CI で動かすのは lint / format / type-check / 単体テストのみです。
プレイリスト処理は手元の macOS/Linux で `uv run python -m pipeline_youtube.main ...` を実行してください。

このトレードオフを受け入れずにクラウド実行したい場合は、`pipeline_youtube/providers/claude_cli.py` を Anthropic SDK 呼び出しに差し替える必要があります (未実装)。

## セットアップ

```bash
# 前提: Python 3.13, uv, yt-dlp, ffmpeg, claude CLI が PATH にあること
cd __skills/pipeline-youtube
uv sync
# Whisper フォールバック (3次) を使う場合は別途インストール (torch 含み ~2GB):
#   uv sync --extra whisper

# 編集可能インストール (main.py の import エラー防止)
uv pip install -e .

cp config.example.json config.json
# config.json で設定できるフィールド:
#   - vault_root (必須): Obsidian Vault のルートパス
#   - models (任意): ステージ/エージェント別モデル
#     {"stage_02","stage_04","alpha","beta","gamma","leader"} のキーを任意に設定。
#     未設定キーは CLI の --model にフォールバック。
#   - filler_words (任意): Stage 02 の transcript から除去する日本語フィラー語リスト。
#     未設定ならデフォルト (えー/えっと/あのー/まあ/なんか 等) を使用。
# 他のノブ (capture-format / concurrency / min-playlist-size / max-chapters 等) は
# 都度変わる運用値なので CLI フラグで渡す設計。
```

## 使い方

```bash
# 通常実行: プレイリスト全体を 01〜05 まで処理
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx"

# dry-run (Vault 書き込みなし、stdout に全 md を出力)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx" --dry-run

# Stage 05 をスキップ (01〜04 だけ処理したい場合)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx" --skip-synthesis

# Stage 05 だけ再実行 (既存の 04 md を読み込んでハンズオンを作り直す)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx" --synthesis-only

# キャプチャフォーマット指定 (デフォルトは auto = 可能なら WebP)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx" --capture-format gif

# 単一動画のみ (05 Synthesis は自動スキップ)
uv run python -m pipeline_youtube.main "https://www.youtube.com/watch?v=VIDEO_ID"
```

### CLI フラグ一覧

| フラグ | 概要 |
|---|---|
| `--dry-run` | Vault 書き込み無しで stdout に md を出力 |
| `--skip-synthesis` | 01〜04 の後、Stage 05 (Agent Teams) をスキップ |
| `--synthesis-only` | 01〜04 をスキップし、今日の日付のプレイリストフォルダ内にある既存 04 md から Stage 05 のみ再実行 |
| `--capture-format {auto,webp,gif}` | キャプチャ出力形式。`auto` は ffmpeg の libwebp / gif2webp 検出結果に応じて WebP を優先、なければ GIF |
| `--model` | stage 02/04/05 の Claude モデル (sonnet / haiku / opus / フル ID) |
| `--min-playlist-size N` | Stage 05 を起動する最低動画数 (デフォルト 3)。N 未満なら 05 は `[skip]` |
| `--max-chapters N` | β エージェントに章数上限をプロンプト経由で指示 (デフォルト未指定 = β が自動判断) |
| `--concurrency N` | 1〜5 本並列処理 (デフォルト 1)。Whisper 段は内部ロックで常に 1 本に保たれる |
| `--force-video ID` | checkpoint 完了済み動画を強制再処理。繰り返し指定可 |
| `--config PATH` | 代替 `config.json` パス |
| `--stop-after-capture` | Phase 1 実行 (01〜03 のみ)。Obsidian で 02_Summary.md を校閲して `reviewed: true` に書き換えてから Phase 3 を回す運用 |
| `--resume-reviewed` | Phase 3 実行。`reviewed: true` が付いた動画だけ Stage 04〜05 を走らせる |

詳細は [`docs/cli.md`](docs/cli.md) と [`docs/sample-run.md`](docs/sample-run.md) を参照。

## フォルダ構成

Vault 出力は Obsidian vault 側の `Permanent Note/` 配下に書き込まれます (config.json の `vault_root` の相対パス)。

```
Permanent Note/08_YouTube学習/
├── 01_Scripts_Processing_Unit/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
├── 02_Summary_Processing_Unit/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
├── 03_Capture_Processing_Unit/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
├── 04_Learning_Material/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
└── 05_Synthesis/{YYYY-MM-DD-HHmm} {playlist}/
    ├── 00_MOC.md                      ← ハブノート (章構成・動画一覧・学習順序)
    ├── 01_{章名}.md 〜 NN_{章名}.md   ← β が動的に章数を決定
    └── _meta/duplicate_score.json     ← α が算出した重複度スコア
```

- **フォルダ命名**: 日付と時刻はハイフンで連結し、その後ろに半角スペース 1 つ + プレイリスト/動画タイトル。`format_video_note_base` / `format_playlist_folder_name` で一貫。
- **プレイリスト名の `/` 扱い**: YouTube プレイリスト名に ASCII `/` が含まれる場合、`/` はカテゴリ区切りとみなして **最後のセグメントのみ** をタイトルとして採用する (例: `2026Agent Teams/AI駆動経営` → `AI駆動経営`)。全角 `／` (U+FF0F) は日本語タイトル内の正規句読点として保持。
- **画像配置**: `Permanent Note/_assets/2026/pipeline-youtube/pyt_{video_id}_{idx}.webp` に直接書き込み。Obsidian Attachment Management プラグインの `${notename}` リネームパターンとは故意に衝突させない命名を使用。

## 文字起こしフォールバック階層

1. **純正字幕** (`youtube-transcript-api` の manually-created)
2. **自動生成字幕** (`youtube-transcript-api` の auto-generated)
3. **ローカル Whisper** (`openai-whisper`、グローバルロックで常に 1 本ずつ実行)

各段階の適用結果は `logs/transcript_stats_{YYYY-MM-DD}.jsonl` に JSONL で記録され、どの動画/チャンネルで Whisper フォールバックが多発しているかの統計に使える。

**言語方針**: 文字起こし (stage 01) は YouTube の字幕を**元言語のまま**保存する。stage 02 (要約) / stage 04 (learning material) / stage 05 (synthesis) はすべて **日本語固定** で出力する (英語動画の入力でも出力は日本語)。

## Stage 05 Agent Teams の内部構造

Stage 05 はプレイリストに 3 本以上の動画があるときだけ自動起動する。4 つの Claude エージェントを逐次呼び出す:

| ロール | 役割 | 入力 | 出力 |
|---|---|---|---|
| **α** (alpha) | トピック抽出 + 重複度スコア付与 | 全動画の 04 md 本文 | `topics[]` JSON (topic_id / label / source_videos / duplication_count / category) |
| **β** (beta) | 章立て設計 (章数を動的決定) | α の topics | `chapters[]` JSON (index / label / category / topic_ids) |
| **γ** (gamma) | カバレッジ検証 | α + β | `CoverageReport` (covered / missing / notes) |
| **leader** | 章別 md + MOC 生成 | α + β + γ + 04 md 本文 | `{moc, chapters[]}` JSON |

- **重複度カテゴリ**: `duplication_count >= 3` → `core` (章冒頭 `> [!important]`)、`== 2` → `supporting` (本文中で **太字**)、`== 1` → `unique` (通常記述)
- **章数**: β が動的に決定 (3 章以上を目安、内容に応じて増減)
- **章本文の画像埋め込み**: leader は、概念理解に特に有用な箇所だけに絞って `![[...webp]]` を挿入する (各章 0〜3 枚目安)。ファイル名は入力の 04 md 本文にある記法からそのままコピー (新規ファイル名の創作は禁止)。
- **プロンプトインジェクション対策**: 全ロール入力は `<untrusted_content>` でラップし、`sanitize_untrusted_text` を経由。

## セキュリティ層

- **パストラバーサル防御 (7 段階)** — `path_safety.py:ensure_safe_path`
  URL デコード迂回 / 絶対パス拒否 / 制御文字除去 / Unicode NFC 正規化 / `..` 即時拒否 / `realpath` 逃げ道検証 / パス長制限
- **プロンプトインジェクション緩和** — `sanitize.py:sanitize_untrusted_text` + `wrap_untrusted`
  制御文字・ゼロ幅 Unicode 除去、`<untrusted_content>` デリミタで囲む、system prompt で「指示文に従わない」を明示

既存 `__skills/pipeline/` (TypeScript, OneTab 分類) と同じ設計パターンを Python に移植したもの。

## 開発

### フォーマットと Lint

`ruff` (lint + format) と `mypy` (型チェック) を CI で強制しています。ローカルでは pre-commit フックを入れると commit 時に ruff が自動修正 + フォーマットします。

```bash
uv sync                              # ruff / mypy / pre-commit が dev group に入っています
uv run pre-commit install            # 初回のみ: フックを git に登録
uv run ruff check pipeline_youtube/ tests/
uv run ruff format pipeline_youtube/ tests/
uv run mypy pipeline_youtube/ --ignore-missing-imports
```

### テスト

```bash
uv run pytest tests/ -q
# 352 passed (2026-04-18 時点)
```

主な対象:
- `test_path_safety.py` — 7 段階防御の回帰
- `test_sanitize.py` — プロンプトインジェクション緩和
- `test_obsidian.py` — ファイル名規約 / フォルダ名規約 / `/` 分割ルール
- `test_transcript_*` — 字幕フォールバック
- `test_scripts_stage.py` / `test_summary_stage.py` / `test_capture_stage.py` / `test_learning_stage.py` — stage 01〜04
- `test_synthesis_scoring.py` / `test_synthesis_agents.py` / `test_synthesis_stage.py` — stage 05 Agent Teams (claude_cli モック)

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `config.json vault_root is not configured` | `cp config.example.json config.json` してから `vault_root` を実在するパスに書き換える |
| `claude: command not found` (stage 02/04/05 が失敗) | Claude Code CLI (`claude` コマンド) を `npm i -g @anthropic-ai/claude-code` 等でインストール → `claude login` で OAuth を通す |
| stage 03 の capture が `format_unavailable` | ffmpeg が PATH に無い、または libwebp / gif2webp 両方が欠落。`--capture-format gif` を明示するか `brew install ffmpeg webp` |
| stage 01 が `all transcript tiers failed` | 字幕なし動画では Whisper extra が必要: `uv sync --extra whisper`。それでも NG なら動画が非公開 / リージョンブロック |
| stage 05 が `[skip] only N videos succeeded` | プレイリストの stage 04 成功数が `--min-playlist-size` (デフォルト 3) 未満。成功数を増やすか `--min-playlist-size 2` で緩和 |
| Templater が 04 の md を空ファイルと誤認してリネーム | stage 04 は placeholder を作らず直接書き込むよう設計済み。該当フォルダの Templater テンプレート指定を解除するのが確実 |
| `--synthesis-only` で `04 folder not found` | 指定日付のフォルダに該当プレイリストの 04 md が存在しないときに出る。まず 01〜04 を通す |

## 関連ドキュメント

- [`docs/cli.md`](docs/cli.md) — `--help` の完全出力 + 各フラグの挙動
- [`docs/sample-run.md`](docs/sample-run.md) — 実行時コンソール出力の読み方
- 設計プラン: `~/.claude/plans/wondrous-bouncing-feigenbaum.md`
- 兄弟プロジェクト (OneTab 分類): `../pipeline/README.md`
