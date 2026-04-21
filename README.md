# pipeline-youtube

[![CI](https://github.com/theosera/pipeline-youtube/actions/workflows/main.yml/badge.svg)](https://github.com/theosera/pipeline-youtube/actions/workflows/main.yml)

YouTube プレイリスト → Obsidian Vault 自学自習用学習レポート生成パイプライン。

NotebookLM に動画 URL を 1 本ずつ手動で貼り付けて要約を Obsidian に転記する運用を自動化する。プレイリスト URL を 1 本投げると、プレイリスト内の全動画を以下の工程に通して、学習しやすいレポートと分野横断のハンズオンを Vault に保存する。

**プレイリスト単位の前処理 (00.5)**
- **Router**: プレイリスト全体のジャンルを 1 回の haiku 呼び出しで分類 (`coding` / `humanities` / `business` / `science` / `lifestyle` / `entertainment` / `other`)。下流の「コードを含む動画向けの特別処理」はこの結果でゲートする

**動画単位 (01〜04)**
- **01_Scripts**: タイムスタンプ付き文字起こし (YouTube 純正字幕 → 自動生成字幕 → ローカル Whisper の 3 段フォールバック)。`coding` ジャンルの時は動画概要欄から GitHub/Gist URL を検出 → raw コードを fetch → `## 関連コード` セクションとして追記
- **02_Summary**: 意味単位タイムスタンプ範囲付き要約
- **03_Capture**: 要点タイムスタンプに対応する動画フレーム抽出 (WebP アニメーション)
- **04_Learning_Material**: 上記 3 工程を「時系列 → キャプチャ画像 → 要点」3 点セットでテーマ単位に再構成。`coding` ジャンル時は `# 概念` (Concepts) / `# 実践` (Practice) の 2 階層に分割して出力

**プレイリスト単位 (05)**
- **05_Synthesis**: プレイリストの動画数 ≥ 3 本の時、Agent Teams (α→β→Leader) で 04 全体を横断統合し、章別ハンズオン md 群 + 00_MOC.md + 重複度スコア JSON を出力する。カバレッジ検証は Python 集合演算で決定論的に実施し、β の漏れトピックは Reflexion リトライで自己修復する

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
#     {"router","stage_02","stage_04","alpha","beta","leader"} のキーを任意に設定。
#     router は未設定時に haiku (他は --model にフォールバック)。
#     "gamma" は後方互換で受理するが無視。
#   - filler_words (任意): Stage 02 の transcript から除去する日本語フィラー語リスト。
#     未設定ならデフォルト (えー/えっと/あのー/まあ/なんか 等) を使用。
#   - capture_backend (任意): "host" (デフォルト) または "docker"
#   - synthesis_timeout (任意): "auto" (デフォルト, 300 + 60×動画数) または秒数の整数
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

# Stage 05 タイムアウトを固定指定 (デフォルトは auto = 300 + 60×動画数)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx" --synthesis-timeout 3600

# 単一動画のみ (05 Synthesis は自動スキップ)
uv run python -m pipeline_youtube.main "https://www.youtube.com/watch?v=VIDEO_ID"

# macOS で長時間実行する場合は caffeinate で sleep 防止
caffeinate -i uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx"
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
| `--capture-backend {host,docker}` | Stage 03 実行バックエンド。`docker` はハードニングしたコンテナで yt-dlp/ffmpeg を隔離実行 |
| `--synthesis-timeout N` | Stage 05 の per-agent タイムアウト (秒)。未指定 = auto (`300 + 60 × 動画数`) |

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

## ジャンル判定 (Router) とジャンル別分岐

パイプラインの最上流 (Stage 00.5) で、プレイリスト全体のジャンルを 1 回の haiku 呼び出しで判定します。コストは 1 プレイリストあたり ~$0.01 (数百トークン × haiku)。

**判定結果による分岐:**

| Genre | Stage 01 `## 関連コード` 自動追記 | Stage 04 `# 概念` / `# 実践` 分割 |
|---|---|---|
| `coding` | ✅ GitHub/Gist 概要欄 URL から raw コードを fetch | ✅ 理論と実装を分離 |
| `business` / `humanities` / `science` / `lifestyle` / `entertainment` / `other` | ❌ | ❌ (flat 構造のまま) |

**ゲート基準**: `CODE_BEARING_GENRES = {Genre.CODING}` (genres.py)。将来 `mixed` や `tutorial` を追加したい場合はこの frozenset を拡張。

**エラー耐性**: Router の API 失敗・JSON パース失敗・未知のジャンル値は `Genre.OTHER` に collapse され、下流はデフォルト動作 (コード特別処理なし) に fallback します。ルーター呼び出しはパイプライン全体のブロッカーにはなりません。

**GitHub URL 抽出の制約**:
- blob URL (`https://github.com/owner/repo/blob/ref/path`) → raw をそのまま fetch
- Gist URL → public API (`api.github.com/gists/<id>`) でファイル群をまとめて取得
- リポジトリ URL (`github.com/owner/repo` のみ) → スキップ (README 自動 fetch は情報量的にノイズが多い)
- 1 動画あたり最大 5 URL / ファイルあたり最大 50KB
- 認証なし (rate limit: 60 req/h per IP)

## 文字起こしフォールバック階層

1. **純正字幕** (`youtube-transcript-api` の manually-created)
2. **自動生成字幕** (`youtube-transcript-api` の auto-generated)
3. **ローカル Whisper** (`openai-whisper`、グローバルロックで常に 1 本ずつ実行)

各段階の適用結果は `logs/transcript_stats_{YYYY-MM-DD}.jsonl` に JSONL で記録され、どの動画/チャンネルで Whisper フォールバックが多発しているかの統計に使える。

**言語方針**: 文字起こし (stage 01) は YouTube の字幕を**元言語のまま**保存する。stage 02 (要約) / stage 04 (learning material) / stage 05 (synthesis) はすべて **日本語固定** で出力する (英語動画の入力でも出力は日本語)。

## Stage 05 Agent Teams の内部構造

Stage 05 はプレイリストに 3 本以上の動画があるときだけ自動起動する。3 つの Claude エージェント + 決定論的カバレッジ検証を逐次実行する:

| ロール | 役割 | 入力 | 出力 |
|---|---|---|---|
| **α** (alpha) | トピック抽出 + 重複度スコア付与 | 全動画の 04 md 本文 | `topics[]` JSON (topic_id / label / source_videos / duplication_count / category) |
| **β** (beta) | 章立て設計 (章数を動的決定) | α の topics | `chapters[]` JSON (index / label / category / topic_ids) |
| **coverage** | カバレッジ検証 (Python 集合演算) | α + β | `CoverageReport` (covered / missing) — LLM 不要 |
| **β retry** | Reflexion リトライ (漏れがあれば 1 回) | β + missing IDs | 修正版 `chapters[]` |
| **leader** | 章別 md + MOC 生成 | α + β + coverage + 04 md 本文 | `{moc, chapters[]}` JSON |

- **カバレッジ検証**: 旧 γ エージェント (LLM) を Python 集合演算 (`compute_coverage`) に置換。α の topic_ids と β の chapter.topic_ids の set diff でミクロ秒で結果が出る。漏れがあれば β に missing IDs をフィードバックして最大 `MAX_BETA_REFLEXION_RETRIES=3` 回リトライ。3 回でも残る残存ミスは Leader が「残存ミス補完ポリシー」で最も関連性の高い章末尾に `### 補足` として組み込む。
- **動的タイムアウト**: 各エージェントのタイムアウトは `300 + 60 × 動画数` で自動計算。CLI `--synthesis-timeout` または config.json `synthesis_timeout` で固定値に上書き可。β は入力が軽量なため最大 600s にキャップ。
- **プリフライトログ**: Stage 05 開始前に入力充填率・トランケーション有無・タイムアウト値をログ出力。プレイリスト fetch 直後にも早期見積もりを表示。
- **重複度カテゴリ**: `duplication_count >= 3` → `core` (章冒頭 `> [!important]`)、`== 2` → `supporting` (本文中で **太字**)、`== 1` → `unique` (通常記述)
- **章数**: β が動的に決定 (3 章以上を目安、内容に応じて増減)
- **章本文の画像埋め込み**: leader は、概念理解に特に有用な箇所だけに絞って `![[...webp]]` を挿入する (各章 0〜3 枚目安)。ファイル名は入力の 04 md 本文にある記法からそのままコピー (新規ファイル名の創作は禁止)。
- **プロンプトインジェクション対策**: 全ロール入力は `<untrusted_content>` でラップし、`sanitize_untrusted_text` を経由。

## 耐障害性

- **Transient エラー自動リトライ** — `invoke_claude` は `Stream idle timeout` / `ConnectionRefused` / `FailedToOpenSocket` / 5xx 等の既知のネットワーク一時障害を検出し、最大 3 回まで指数バックオフ (30s → 60s → 120s) でリトライする。laptop の sleep 復帰後の接続断や API の瞬断に対応。
- **Checkpoint リカバリ** — Stage 04 完了済みの動画は再実行時に自動スキップ。途中で Ctrl+C しても完了分は保持される。
- **動的タイムアウト** — Stage 05 のタイムアウトが動画数に応じて自動伸長 (`300 + 60 × N`)。大規模プレイリストでの timeout 失敗を防止。

## セキュリティ層

- **パストラバーサル防御 (7 段階)** — `path_safety.py:ensure_safe_path`
  URL デコード迂回 / 絶対パス拒否 / 制御文字除去 / Unicode NFC 正規化 / `..` 即時拒否 / `realpath` 逃げ道検証 / パス長制限
- **プロンプトインジェクション緩和** — `sanitize.py:sanitize_untrusted_text` + `wrap_untrusted`
  制御文字・ゼロ幅 Unicode 除去、`<untrusted_content>` デリミタで囲む、system prompt で「指示文に従わない」を明示
- **Stage 03 Docker 隔離 (オプション)** — `stages/capture_backend.py:DockerCaptureBackend`
  `yt-dlp` / `ffmpeg` / `gif2webp` を `--cap-drop=ALL --read-only --user=1000:1000` のハードニングしたコンテナで実行。Threat Model §11 R1 (ffmpeg/yt-dlp のホスト実行) への対策。詳細: [docs/docker.md](docs/docker.md)

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
# 572 passed (2026-04-21 時点)
# 572 passed (2026-04-21 時点)
# 572 passed (2026-04-21 時点)
# 572 passed (2026-04-21 時点)
# 572 passed (2026-04-21 時点)
# 572 passed (2026-04-21 時点)
```

主な対象:
- `test_path_safety.py` — 7 段階防御の回帰
- `test_sanitize.py` — プロンプトインジェクション緩和
- `test_obsidian.py` — ファイル名規約 / フォルダ名規約 / `/` 分割ルール
- `test_transcript_*` — 字幕フォールバック
- `test_scripts_stage.py` / `test_summary_stage.py` / `test_capture_stage.py` / `test_learning_stage.py` — stage 01〜04
- `test_synthesis_scoring.py` / `test_synthesis_agents.py` / `test_synthesis_stage.py` — stage 05 Agent Teams (claude_cli モック)
- `test_synthesis_timeout.py` — 動的タイムアウト計算 + プリフライトログ + config.json 読み込み
- `test_claude_cli_retry.py` — transient エラーリトライ (パターン検出 / 指数バックオフ / 非 transient 即時伝播)
- `test_capture_backend.py` — Docker 隔離バックエンド (コマンド形状 / パス変換 / プリフライト)
- `test_genres.py` — Router ジャンル分類 (JSON 解析 / エラー fallback / プロンプト形状)
- `test_code_fetch.py` — GitHub URL 抽出 + raw コード取得 (yt-dlp / urllib モック)
- `test_learning_code_bearing.py` — Stage 04 の `# 概念` / `# 実践` プロンプト分割
- `test_synthesis_prompt_rules.py` — Leader/β プロンプトの品質ルール回帰 (核心要素出典必須、矢印圧縮禁止、章最低 5 トピック、MOC 概念索引、時間別学習コース)

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
| `Stream idle timeout` / `ConnectionRefused` で動画が FAIL | ネットワーク一時障害。リトライ機構 (3 回、計 210s) で吸収するが、laptop sleep 中の長時間断はリトライでもカバーできない。`caffeinate -i` で sleep 防止して再実行 |
| Stage 05 が timeout | `--synthesis-timeout 3600` で延長するか、auto (300+60×動画数) が適切か確認。50 本超は auto で 3300s |
| iCloud `Operation not permitted` | macOS のプライバシー設定 → フルディスクアクセスにターミナルアプリを追加 → ターミナル再起動 |

## 関連ドキュメント

- [`docs/cli.md`](docs/cli.md) — `--help` の完全出力 + 各フラグの挙動
- [`docs/sample-run.md`](docs/sample-run.md) — 実行時コンソール出力の読み方
- 設計プラン: `~/.claude/plans/wondrous-bouncing-feigenbaum.md`
- 兄弟プロジェクト (OneTab 分類): `../pipeline/README.md`
