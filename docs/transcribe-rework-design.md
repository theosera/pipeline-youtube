# 設計: Stage 01 文字起こし刷新（高速文字起こし＋agentic web search 訂正）

> ステータス: **非SDK 実装済み**（このリポ）。SDK へは後続 PR で移植する。
> 実装後に判明した差分は §6・§4 の注記参照（非SDK には永続 transcript キャッシュが無く、
> claude CLI には拡張思考フラグが無い）。ユーザー向け手順は `docs/transcribe.md`。

## 1. 背景・目的

現状の Stage 01 は字幕取得→（無ければ）Whisper という流れで、**長尺で遅い**のが 01–04 のボトルネック。
方針を「**速度優先で粗く文字起こし → LLM＋web search で誤変換を事実確認して訂正**」に転換し、
速度と精度を両立する（Obsidian の YTranscriptor の発想。ただし時間軸の刻みは現リポ形式へ正規化）。

**本 PR のスコープ = Stage 01 のみ**（訂正済み・タイムスタンプ付きトランスクリプトの生成まで）。
まとめ・知見抽出（ユーザープロンプトの Phase 2–3）は **Stage 02 に時間軸対応を加える別 PR**で扱う。

## 2. アーキテクチャ

```
01a 高速文字起こし        01b 誤変換訂正(LLM+web search)        出力
─────────────────        ──────────────────────────          ──────────
YouTube: auto-captions ─┐
                        ├─→ 粗トランスクリプト(snippets) ─→ Opus/Sonnet(thinking)+WebSearch ─→ 訂正済み 01 md
local : 高速 Whisper  ─┘     (start/duration 保持)            ・固有名詞/専門用語を自律 web 検索で事実確認
                                                              ・タイムスタンプは保持（要点）
```

- **01a**: 取得は既存の fallback を流用（official→auto→whisper）。`--local-media` は Whisper。
  「精度度外視で速い」一次ソースを使い、誤りは 01b で回収する前提。
- **01b**: 訂正は **`stage_01_correct` モデル（既定 Opus／Sonnet 拡張思考）＋ WebSearch ツール**で実行。
  ユーザープロンプト Phase 1（データクレンジング＋web 検索ファクトチェック）相当。**要約はしない**。
- **出力**: 既存の `01_Scripts_Processing_Unit` の md（`[MM:SS](url&t=) text` 形式）を**そのまま流用**し、
  中身を訂正済みに差し替える。新フォルダは作らない（02 が時間レンジを解析する契約を壊さないため）。

## 3. タイムスタンプ保持つき訂正（最重要設計）

訂正で時刻がズレると 02/03 が壊れる。LLM に**自由形式で書き直させない**。

- 入力: チャンク（`chunk_by_window`）を **`[idx] (MM:SS) text`** 形式の番号付き行で渡す。
- 指示: 「**各 idx の行を 1:1 で訂正して返す。行の増減・統合・並べ替え・時刻改変は禁止。**
  固有名詞/専門用語に不確かさがあれば WebSearch で事実確認。判断不能な深刻欠落は `[聴取不能]`。」
- 出力: **JSON**（`[{"idx": int, "text": str}, ...]`）で受け取り、idx で元 snippet にマッピングして
  タイムスタンプを再付与。idx 欠落・件数不一致は**訂正失敗扱い**で原文フォールバック（壊すより原文）。
- 長尺対策: 一定トークン/行数ごとに**バッチ分割**して複数回呼ぶ。各バッチ独立で失敗は局所化。

## 4. プロバイダ層の変更（claude_cli）

`invoke_claude` / `_invoke_claude_once` に web search と拡張思考を通す:

- 新パラメータ `allowed_tools: list[str] | None`（例 `["WebSearch"]`）。
  指定時は `--tools <names>` ＋ `--allowedTools <names>`（自動承認）を付与。`disallow_tools` と排他。
- **拡張思考**: claude CLI には拡張思考の専用フラグが無い（`claude --help` で確認）。非SDK 版は
  **モデル指定（既定 opus）**で対応し、`thinking` パラメータの配線は **SDK 移植時**（Anthropic API の
  `thinking={"type":"enabled","budget_tokens":...}`）に行う。
- WebSearch は OAuth/プラン課金で動く（API キーは既存どおり環境から除去）。

## 5. 設定（config.json）

- `_MODEL_KEYS` に **`stage_01_correct`** を追加。既定は `opus`（thinking 有効）。
- 任意で `whisper_model` を「速い小モデル」に倒す運用を docs に明記（例 local の一次パスは turbo/ small）。
- 訂正の上限系（バッチサイズ、最大 web 検索回数の目安）は config で調整可能にする。

## 6. transcript キャッシュの再設計（Codex #3） — **非SDK では不要**

- **実装時の判明事項**: 非SDK 版には永続 transcript キャッシュが存在しない（`cache.py` 無し）。
  よってモデル変更でのサイレント再利用問題は**非SDK では起きず**、本 PR でキャッシュ変更は不要。
- Codex #3 は **SDK 固有**（SDK は `cache.py` の `(video_id, tier, lang)` キャッシュを持つ）。
  SDK 移植時に「訂正済みを `(video_id, source_tier, correct_model)` で別レイヤ保存」して解消する。

## 7. ステージ境界・契約

- 02（要約）・03（キャプチャ）・04・05 は**無変更**。01 の出力 md 形式・パスも不変。
- 02 は引き続き `[MM:SS ~ MM:SS]` レンジを生成（03 が解析）。
- ユーザープロンプト（Phase 2–3）の 02 統合・時系列補強・粒度調整は**別 PR**。

## 8. リスク / 留意

- **コスト・レイテンシ**: 長尺×Opus×web search は高コスト。バッチ＋（任意）検索回数上限で抑制。
  既定モデルや thinking 既定値は保守的にし、docs に費用感を明記。
- **claude CLI の web search 有効化**は実機で最終確認（フラグ・ツール名・自動承認挙動）。
- **JSON 整合**: idx 不一致時は原文フォールバックで「壊さない」を最優先。

## 9. テスト方針

- 訂正レスポンスのパース＆idx マッピング（件数一致/欠落/不正 JSON → 原文フォールバック）。
- タイムスタンプ不変の検証（訂正後も snippet の start/duration が保持される）。
- web search 呼び出しはモック（`invoke_claude` を差し替え）。実 API は叩かない。
- 既存 01/02/03 のレンジ契約が壊れていない回帰。
- ruff / ruff format / mypy strict / pytest を従来どおり通す。

## 10. SDK への移植（後続）

- web search: `AnthropicProvider.invoke()` に `tools=[{"type":"web_search_...","name":"web_search"}]`、
  拡張思考は `thinking={"type":"enabled","budget_tokens":...}` を追加。config に `stage_01_correct`（provider=anthropic）。
- OpenAI 互換プロバイダは web search 非対応 → 訂正ステージは Anthropic 固定（`--hybrid` の heavy 同様）。
- 非SDK で確定した「タイムスタンプ保持訂正」「キャッシュ再設計」をそのまま移植。
