# pipeline-youtube → pipeline-youtube-SDK 移行 設計サマリ（ハンドオフ）

> **重要:** 根本的な再設計・実装は本リポジトリ `theosera/pipeline-youtube` ではなく、
> 新リポジトリ **`pipeline-youtube-SDK`** 側で行う。本リポジトリには変更を加えない。
> 本ドキュメントは、ここまでの調査・議論で固めた設計案の引き継ぎメモ。

---

## 1. 目的

現行 pipeline-youtube（YouTube プレイリスト → Obsidian Vault 学習レポート生成）の
**実行時間を大幅に削減**する。10本逐次で15〜25分かかる現状を、キャッシュ・並列化・
in-process 化で短縮する。SDK 版はグリーンフィールドで Claude Agent SDK を基盤に作り直す。

## 2. 現行アーキテクチャ（調査で確定した処理方法）

```text
プレイリスト取得(yt-dlp) → [00.5 Router] → 動画ごと[01→02→03→04] → [05 Synthesis]
```

| ステージ | 処理 | LLM | 主コスト |
|---|---|---|---|
| 00.5 Router | プレイリストのジャンル分類（coding等） | haiku ×1/playlist | 軽 |
| 01 Scripts | 文字起こし（純正→自動→Whisper の3段）＋coding時GitHubコード取得 | **なし(純Python)** | 字幕/ Whisper |
| 02 Summary | 意味単位タイムライン要約 | **sonnet ×1/動画** | LLM 30–60s |
| 03 Capture | 要点範囲の動画フレームをWebP抽出 | **なし(yt-dlp/ffmpeg)** | DL+ffmpeg |
| 04 Learning | 時系列→画像→要点の3点セット再構成 | **sonnet ×1/動画** | LLM 20–40s |
| 05 Synthesis | α→β→coverage(Python)→leader→reviewer で横断統合 | **sonnet ×3〜** | LLM 直列 |

LLM はすべて `claude -p` を **subprocess 起動**（OAuth Pro/Max 定額、`providers/claude_cli.py:298`）。

## 3. 確定したボトルネック（実装で潰す対象）

1. **永続キャッシュ皆無** — 文字起こし・動画DL・LLM出力すべて毎回再計算。
   checkpoint は Stage04 md の有無＝動画単位の全/無のみ（`checkpoint.py:147`）、途中再開不可。
2. **LLM 毎回 subprocess 起動**（0.5〜2s固定×呼び出し数）。`subprocess.run` は blocking。
   `resume_session` 引数はあるが**未使用**＝キャッシュ再利用制御なし。
3. **並列デフォルト1**（`main.py:572`）。02/04 が動画間で直列。
4. **Whisper グローバル file lock＋model 毎回ロード**（`whisper_fallback.py:60,224`）。
5. **Stage 05 で 04本文を α と Leader に二重送信**（冗長トークン）。
6. **モジュールグローバル状態**（`config.py` `_vault_root/_dry_run`、`official.py:27` `_api`）が
   真の並列化を阻む。

時間配分: LLM推論55–60% / ffmpeg15–20% / ネットワーク10–15% / I/O5–10%。

## 4. 新 SDK 版の設計方針（合意済み）

### 4-1. 基盤: Claude Agent SDK（in-process 非同期）

- `claude -p` subprocess を撤廃し、`ClaudeSDKClient`/`query()` の **in-process・asyncio ネイティブ**へ。
  → プロセス起動コスト消滅、`asyncio.gather` で多動画を真に並列、session resume で prompt cache 再利用。
- **認証/課金の注意（要意思決定）:** Agent SDK は **OAuth 不可・`ANTHROPIC_API_KEY` 必須**。
  2026/6/15 以降はサブスク経由の SDK/`claude -p` 利用も別枠の従量クレジット化。
  → SDK 版は**従量課金前提**。定額維持が必須なら、LLM 呼び出しを抽象インターフェース化して
  CLI バックエンド(OAuth)も差し替え可能に残すのが保険（後述）。

### 4-2. 混成型ステージ構成（ユーザー採用）

- **01 Scripts / 03 Capture = LLM を呼ばない高速決定論ツールのまま**（LLM化すると逆に遅くなるため）。
- **02 / 04 / 05 = LLM サブエージェント**（`AgentDefinition` で stage 別の system prompt・tool セット・
  model を定義。サブエージェントは並列実行可・ネスト不可）。
- **skill.md は複雑な 04・05 のみ**に付与。現状 system prompt にインラインの巨大な整形/出典/
  画像マッピング規約（`learning.py` の `LEARNING_SYSTEM_PROMPT`＋addendum、`agents.py` の
  α/β/leader/reviewer）を `.claude/skills/*/SKILL.md` へ切り出し、progressive disclosure で
  必要時のみロード → 毎回の cache-creation トークン削減。軽量な 01/02 は skill 化しない。

### 4-3. フル永続キャッシュ（ユーザー採用・課金方式に依らず最大の効果）

content-addressed キャッシュを**リポジトリ外・Vault 外**（`~/.cache/pipeline-youtube/`、
`--cache-dir`/`--no-cache` 可）に保存し、動画由来物を git 追跡しない security posture を順守。
- `transcript/{video_id}/{tier}/{lang}` / `video/{video_id}/{fmt}.mp4`（削除せず LRU 退避）/
  `llm/{sha256(model+system+prompt)}` / `code_fetch/{sha256(url)}`。
- 効果: 同一プレイリストの再実行・部分再実行がほぼ即時に。

### 4-4. 真の並列化・段階別チェックポイント

- async スケジューラ: 動画内は 01→02→(03DLは02と重複)→04 の依存を保ちつつ、動画間で 02/04 を
  並列（bounded semaphore、既定 concurrency 引き上げ）。
- Whisper: グローバル lock → **bounded semaphore（既定1可変）** ＋ model プロセス内キャッシュ。
- Stage 05: α-batch 並列は維持し、session resume で α→β→leader の 04本文二重送信をキャッシュ再利用化。
- 段階別 checkpoint で動画の途中から再開（02成功/03失敗なら03から）。
- グローバル状態（`_vault_root/_dry_run/_api`）をコンテキスト受け渡し or スレッド安全化。

### 4-5. 保険: LLM バックエンド抽象化（課金判断を後回しにする口）

`LLMBackend` Protocol（`invoke(prompt, *, system, model, resume_session, timeout)`）を定義し、
`SdkBackend`(API キー・既定) と `CliBackend`(`claude -p` async化・OAuth定額) を差し替え可能に。
ステージは抽象経由でのみ呼ぶ。→ SDK 版でも定額運用へ退避できる。

## 5. 移行スコープと注意

- **本リポジトリ(`theosera/pipeline-youtube`)は変更しない。** 実装は `pipeline-youtube-SDK` で。
- 移植すべき資産: 01/03 の決定論ロジック（transcript 3段・chunking・code_fetch・capture）、
  path_safety/sanitize のセキュリティ層、Stage05 の coverage 集合演算（`synthesis/scoring.py`）、
  Obsidian 命名規約（`obsidian.py`）、各 system prompt（→ skill.md 化）。
- 再設計する箇所: LLM 呼び出し層（SDK化）、スケジューラ（async並列）、キャッシュ層（新規）、
  checkpoint（段階別）、グローバル状態の除去。

## 6. 推奨実装順序（SDK リポジトリ内）

1. キャッシュ層（transcript/動画/code_fetch）— LLM非依存・即効・低リスク。
2. `LLMBackend` 抽象＋`SdkBackend`＋LLM出力キャッシュ＋session resume。
3. async スケジューラ・並列既定引き上げ・Whisper semaphore/model キャッシュ・グローバル除去。
4. 段階別 checkpoint。
5. 04/05 の skill.md・サブエージェント定義切り出し。
6.（保険）`CliBackend` で OAuth 定額バックエンドを差し替え可能に。

## 7. 期待効果

- **再実行**: キャッシュヒットでほぼ即時（最大の体感改善）。
- **単発実行**: subprocess 起動消滅＋02/04 並列＋Whisper 並列で大幅短縮（推定、逐次比で数倍）。
- **Stage 05**: session resume で冗長送信削減・トークン/時間減。

---

## 8. 意思決定が必要な質問（要ユーザー回答）

| # | 質問 | 選択肢 / 論点 | 影響 |
|---|---|---|---|
| Q1 | **認証/課金** をどうするか | B(SDK・APIキー従量)で確定 / A(OAuth定額)も保険で残す / 既定はどちら | アーキ最上流。SDKはOAuth不可 |
| Q2 | `pipeline-youtube-SDK` リポジトリ | 既存? 新規作成? 言語は Python 継続? | 着手の前提 |
| Q3 | **出力互換性** | 生成 md 構造を現行と完全互換にするか / SDK版で刷新可か | 移植 vs 再設計の線引き |
| Q4 | **Vault 連携** | 同じ Obsidian Vault 構造に書くか | 出力先・命名規約の流用範囲 |
| Q5 | **並列度の既定** | concurrency 既定値 / Whisper 同時数 / APIレート上限の織り込み | 速度と安定性のトレードオフ |
| Q6 | **キャッシュ運用** | 保存先 `~/.cache` で可か / サイズ上限・TTL / 既定で有効か | 肥大化・再現性 |
| Q7 | **テスト/CI** | 現行605テストを移植? / CIでAPIキー・SDKを動かすか | 品質ゲート |
| Q8 | **モデル方針** | stage別モデル(haiku/sonnet/opus)の現行方針を継続するか | コスト/品質 |

## 9. 未解決・要検証事項（技術リスク）

| # | 項目 | 内容・懸念 |
|---|---|---|
| R1 | OAuth 不可の最終確認 | Agent SDK が OAuth 定額を使えない件と 2026/6/15 課金変更の正確な影響を一次情報で再確認 |
| R2 | skill のスコープ | skill はサブエージェント単位にスコープできず session グローバル。04用と05用の skill が干渉しないか |
| R3 | サブエージェントのネスト不可 | Stage05 の α/β/leader/reviewer を subagent 化する構成が SDK 制約内で成立するか |
| R4 | prompt cache と skill 変更 | skill リスト変更で prompt cache が壊れる仕様。段階で skill を出し入れする運用設計が必要 |
| R5 | レート制限/同時実行 | SDK 側に pooling なし。30本並列が API レート制限内で現実的か、バックオフ設計 |
| R6 | Whisper 並列 | in-process model キャッシュとマルチ並列(GPU/RAM)の両立。semaphore 既定値の根拠 |
| R7 | 動画キャッシュ肥大 | mp4 を削除せず保持 → LRU 上限・eviction 設計（GB単位になり得る） |
| R8 | checkpoint キー設計 | 段階別 checkpoint を content-hash で持つか video_id で持つか（プレイリスト改訂への追従） |
| R9 | コスト試算 | B案従量化での 1プレイリストあたり想定コスト試算（定額からの乖離を可視化） |
