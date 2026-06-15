# Stage 01 文字起こし — 高速取得＋LLM校正（誤変換訂正）

Stage 01 は2段構成:

1. **01a 高速文字起こし（LLM 非介在）**: YouTube は auto-captions、`--local-media` は Whisper。
   速度優先で、誤変換はこの段では直さない。
2. **01b 誤変換訂正（任意・LLM＋web search）**: チャンク化したトランスクリプトを LLM（既定 Opus）に渡し、
   固有名詞・専門用語の誤変換を **WebSearch で事実確認**して訂正する。**要約はしない**。

01b は **タイムスタンプを保持**する（チャンクの `start` は不変）。02/03 が依存する
`[MM:SS ~ MM:SS]` レンジ契約は壊れない。

## 有効化（オプトイン）

01b は**有料・低速**なので既定で OFF。`config.json` で有効化する:

```jsonc
{
  "transcript_correction": true,
  "models": {
    "stage_01_correct": "opus"   // 既定 opus。sonnet 等も可
  }
}
```

- `transcript_correction: true` で 01b を実行。`false`（既定）なら 01a の生トランスクリプトをそのまま出力。
- 校正モデルは `models.stage_01_correct`（既定 `opus`）。WebSearch ツールが自動で有効化される。

## 挙動・安全性

- **ベストエフォート**: LLM エラー・不正 JSON・件数不一致は**その場の生テキストにフォールバック**し、
  Stage 01 を止めない／タイムスタンプをずらさない。
- **バッチ処理**: 長尺は一定チャンク数ごとに分割して校正（既定 40 チャンク/回）。失敗は局所化。
- **コスト/レイテンシ**: 長尺 × Opus × web search は高コスト。費用が気になる場合は
  `stage_01_correct` を `sonnet` に下げる、または `transcript_correction: false` のままにする。
- `--dry-run` では 01b はスキップ（課金回避）。

## 仕組み（タイムスタンプ保持）

LLM には `[idx] (MM:SS) text` 形式で番号付きチャンクを渡し、`[{"idx", "text"}]` の **JSON で 1:1 訂正**を
返させる。行の統合・分割・並べ替え・時刻改変は禁止。idx で元チャンクへ写し戻し、`start` を再付与する。

## 注意（claude CLI / OAuth）

- WebSearch は `claude -p` の組込みツール。OAuth/プラン枠で実行され、API キーは環境から除外される（既存挙動）。
- claude CLI には拡張思考の専用フラグが無いため、非SDK 版は**モデル指定（opus 等）**で対応する。
  SDK 版では Anthropic API の `thinking` を別途配線する（移植時）。
