# pipeline-youtube

YouTube プレイリスト → Obsidian Vault 自学自習用学習レポート生成パイプライン。

NotebookLM に動画 URL を 1 本ずつ手動で貼り付けて要約を Obsidian に転記する運用を自動化する。プレイリスト URL を 1 本投げると、プレイリスト内の全動画を以下 5 工程に通して、学習しやすいレポートと分野横断のハンズオンを Vault に保存する。

**動画単位 (01〜04)**
- **01_Scripts**: タイムスタンプ付き文字起こし (YouTube 純正字幕 → 自動生成字幕 → ローカル Whisper の 3 段フォールバック)
- **02_Summary**: 意味単位タイムスタンプ範囲付き要約
- **03_Capture**: 要点タイムスタンプに対応する動画フレーム抽出 (WebP アニメーション)
- **04_Lerning_Material**: 上記 3 工程を「時系列 → キャプチャ画像 → 要点」3 点セットでテーマ単位に再構成

**プレイリスト単位 (05)**
- **05_Synthesis**: プレイリストの動画数 ≥ 3 本の時、Agent Teams (α/β/γ/leader) で 04 全体を横断統合し、章別ハンズオン md 群 + 00_MOC.md + 重複度スコア JSON を出力する

AI 呼び出し (stage 02/04/05) は **`claude -p` headless CLI を subprocess で呼び出し**、Claude Pro/Max の OAuth セッションを使います。`ANTHROPIC_API_KEY` は不要。`claude login` 済であればそのまま動作します。

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
# config.json の vault_root を編集 (Obsidian Vault のルートパス)
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
| `--config` | 代替 `config.json` パス |
| `--concurrency` | 将来用。現状は sequential にフォールバック (Step 11) |

## フォルダ構成

Vault 出力は Obsidian vault 側の `Permanent Note/` 配下に書き込まれます (config.json の `vault_root` の相対パス)。

```
Permanent Note/08_YouTube学習/
├── 01_Scripts_Processing_Unit/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
├── 02_Summary_Processing_Unit/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
├── 03_Capture_Processing_Unit/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
├── 04_Lerning_Material/{YYYY-MM-DD-HHmm} {playlist}/{YYYY-MM-DD-HHmm} {title}.md
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

## テスト

```bash
uv run pytest tests/ -q
# 245 passed (2026-04-16 時点)
```

主な対象:
- `test_path_safety.py` — 7 段階防御の回帰
- `test_sanitize.py` — プロンプトインジェクション緩和
- `test_obsidian.py` — ファイル名規約 / フォルダ名規約 / `/` 分割ルール
- `test_transcript_*` — 字幕フォールバック
- `test_scripts_stage.py` / `test_summary_stage.py` / `test_capture_stage.py` / `test_learning_stage.py` — stage 01〜04
- `test_synthesis_scoring.py` / `test_synthesis_agents.py` / `test_synthesis_stage.py` — stage 05 Agent Teams (claude_cli モック)

## 関連ドキュメント

- 設計プラン: `~/.claude/plans/wondrous-bouncing-feigenbaum.md`
- 兄弟プロジェクト (OneTab 分類): `../pipeline/README.md`
