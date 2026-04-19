---
date: 2026-04-19 21:11
title: "自律実行とGUI自動化による作業委譲"
URL: ""
playlist: "2026ClaudeCode Settings Only"
chapter: "2"
category: "unique"
sources: "E9tjcRjqE4c, AL_7VqZEqD4"
tags: [memo, youtube, synthesis]
---

## 概念定義

**定期タスク・夜間自律実行**とは、特定時刻に特定の作業をClaude Codeが無人で自動実行する機能である。下書き生成・トレンド抽出・返信文作成・ファイル整理などを夜間に回すことで、人間は翌朝に確認と微調整を行う「承認者」として振る舞える。設定に必要なのは「名前・やってほしいこと・頻度」の3項目のみで、Claude Codeとの対話で自動生成される。

出典:
- [[I woke up this morning, opened my PC, and found that the AI had finished my work overnight. [Cl...#^04-15]] (04:15〜)

**Computer Use・GUI自動化**とは、MCP経由でClaude CodeがSwiftUIアプリやElectronビルドなどCLIを持たないGUIアプリを直接操作できる機能である。テスト・デバッグを単一プロンプトで完結させ、従来ヒューマン・イン・ザ・ループが必要だった検証フローをエージェントループ内に取り込める。

出典:
- [[NEW Hidden Features You MUST Enable In Your Claude Code Setup!#^05-39]] (05:39〜)

## 核心要素

### 定期タスクの設定

1. **設定の簡便さ**: 「名前・やってほしいこと・頻度」の3項目を指定し、Claude Codeとの対話で自動生成される
2. **承認者モデルの実現**: 朝起きたら確認と微調整だけで済む「人間＝承認者」のワークフローへ移行できる
3. **知識資産の蓄積**: 作業結果がフォルダーに積み上がり、継続的な知識ベースが形成される

![[pyt_E9tjcRjqE4c_04.webp]]

### Computer Use・GUI自動化の活用

1. **有効化方法**: `/mcp` コマンドからComputer Use MCPを有効化する
2. **対応範囲**: SwiftUIアプリ・Electronビルドなど、CLIを持たないGUIアプリを直接操作できる
3. **制限事項**: 現在はProおよびMaxプランのmacOS限定の研究プレビュー版

![[pyt_AL_7VqZEqD4_05.webp]]

## 補足とまとめ

定期タスク（夜間実行）とComputer Use（GUI自動化）はともに「人間が離れた状態でのエージェント自律実行」という共通軸を持つ。定期タスクで繰り返し業務を委譲し、Computer UseでGUIアプリのテスト・デバッグも自動化することで、人間の介在が必要なポイントを最小化できる。「実行はエージェント・確認と判断は人間」という役割分担を設計することが、作業委譲の成功パターンである。
