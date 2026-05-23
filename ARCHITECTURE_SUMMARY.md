# Urban Safety App 架构与数据结构总结

本文档用于整理当前系统的实现结构，方便毕业论文第3章「提案システム」和第4章「実装」使用。

## 1. 研究定位

当前系统不是城市危险预测系统，也不是 SNS 多源融合系统，而是：

```text
基于官方都市安全信息的 AI 回答验证系统
```

系统目标是检查 AI 对城市安全相关问题的回答是否：

- 基于官方信息
- 与官方信息一致
- 不包含危险建议
- 不在信息不足时断言安全
- 不产生过度安心表达

因此，系统核心不是预测风险，而是：

```text
Official Evidence に基づく AI draft answer の検証
```

## 2. 当前整体架构

系统由后端和前端两部分组成。

```text
官方信息源
  ↓
official_sources.py
  ↓
official_service.py
  ↓
Evidence DB: official_area_observations
  ↓
用户问题
  ↓
answer_service.py
  ↓
Answer Generator
  ↓
draft_answer
  ↓
Verifier Agent
  ↓
PASS / NEEDS_REVIEW / FAIL
  ↓
SHOW / SHOW_WITH_WARNING / DO_NOT_SHOW
  ↓
前端展示
```

论文中可以简化成：

```text
Official Data Fetch
  → Evidence DB
  → Answer Generator
  → Verifier Agent
  → Final Display
```

## 3. 后端模块结构

后端目录：

```text
urban-safety-backend/
  main.py
  config.py
  db.py
  official_sources.py
  official_service.py
  answer_service.py
  ai_client.py
  prompts.py
  json_utils.py
  schemas.py
  init_db.py
  urban_safety.db
```

各文件职责如下。

| 文件 | 作用 |
|---|---|
| `main.py` | FastAPI 路由层。只负责 API endpoint、startup/shutdown、health check。 |
| `config.py` | 环境变量、AI 配置、DB 路径、官方信息同步间隔。 |
| `db.py` | SQLite 表结构和 DB 操作。论文中的 Evidence DB 主要在这里。 |
| `official_sources.py` | 官方信息源的定义与原始数据获取。 |
| `official_service.py` | 官方信息同步、LLM 正规化、summary 构建。 |
| `answer_service.py` | `/ask` 的核心流程：Generator → Verifier → 保存结果。 |
| `ai_client.py` | Ollama / API 形式的 LLM 调用封装。 |
| `prompts.py` | 官方信息正规化、回答生成、Verifier 的 prompt。 |
| `json_utils.py` | 从 LLM 输出中提取 JSON。 |
| `schemas.py` | FastAPI request model。 |
| `init_db.py` | 初始化 SQLite DB。 |

## 4. 前端模块结构

前端目录：

```text
urban-safety-app/
  App.js
  src/
    api/urbanSafetyApi.js
    config/backend.js
    utils/official.js
    components/
      AskPanel.js
      SummaryPanel.js
      OfficialCard.js
      OfficialInfoModal.js
      ConnectionErrorPanel.js
      SystemStatus.js
    styles/styles.js
```

前端职责是展示官方信息和 Verifier 判定结果。研究核心逻辑放在后端。

| 文件 | 作用 |
|---|---|
| `App.js` | 页面状态管理和整体布局。 |
| `src/api/urbanSafetyApi.js` | 后端 API 调用。 |
| `src/config/backend.js` | 后端 URL 自动判断。 |
| `src/utils/official.js` | status 显示配置和 list key 生成。 |
| `src/components/AskPanel.js` | 用户提问、AI 回答、Verifier 结果显示。 |
| `src/components/SummaryPanel.js` | 当前官方状态概要。 |
| `src/components/OfficialCard.js` | 官方信息卡片。 |
| `src/components/OfficialInfoModal.js` | 官方信息详情弹窗。 |
| `src/styles/styles.js` | 共通样式。 |

## 5. 数据流 1：官方信息同步

对应 API：

```text
POST /official/sync
```

流程：

```text
1. main.py 接收 /official/sync 请求
2. official_service.py 调用 official_sources.py
3. official_sources.py 从气象厅、自治体、铁路、道路等官方页面获取原始数据
4. official_service.py 使用 LLM 或 fallback 将原始数据正规化
5. db.py 保存到 official_area_observations
6. 前端通过 /dashboard 或 /official/live 显示
```

这部分在论文中对应：

```text
Official Data Fetch
Evidence DB Construction
```

## 6. 数据流 2：用户提问与 AI 回答验证

对应 API：

```text
POST /ask
```

流程：

```text
1. 用户在前端 AskPanel 输入问题
2. 前端调用 POST /ask
3. answer_service.py 从 official_area_observations 读取 evidence
4. Answer Generator 根据 question + evidence 生成 draft_answer
5. Verifier Agent 检查 draft_answer 是否符合 evidence
6. Verifier 输出 PASS / NEEDS_REVIEW / FAIL
7. 系统映射为 SHOW / SHOW_WITH_WARNING / DO_NOT_SHOW
8. db.py 保存到 answer_verifications
9. 前端显示最终结果
```

判定映射：

| Verifier verdict | display_policy | 前端行为 |
|---|---|---|
| `PASS` | `SHOW` | 正常显示回答 |
| `NEEDS_REVIEW` | `SHOW_WITH_WARNING` | 显示回答，但带 warning |
| `FAIL` | `DO_NOT_SHOW` | 不显示回答正文，只显示阻止理由 |

## 7. 当前 DB 表结构

当前 SQLite DB 主要有两张表：

```text
official_area_observations
answer_verifications
```

### 7.1 official_area_observations

这张表是当前系统的 Evidence DB。

论文中可以写：

```text
本研究では、公式情報を official_area_observations テーブルに保存し、
AI回答検証の Evidence DB として利用する。
```

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | TEXT | 记录 ID。由 source、area、observed_at 等生成。 |
| `source` | TEXT | 官方信息来源，例如 気象庁、名鉄、名古屋市。 |
| `source_url` | TEXT | 官方信息 URL。 |
| `area` | TEXT | 对象地区，例如 愛知県、名古屋市。 |
| `label` | TEXT | 官方信息标题或短摘要。 |
| `severity` | REAL | 旧字段。当前实现不再使用数值评分，新写入记录固定为 0.0。 |
| `status` | TEXT | 状态，例如 通常 / 情報 / 注意 / 警戒 / 運休 / 支障 / 取得不可。 |
| `detail` | TEXT | 详细说明、判断依据、官方 URL 等。 |
| `observed_at` | TEXT | 官方发布时间或抓取时间。 |
| `created_at` | TEXT | DB 保存时间。 |

典型记录示意：

```json
{
  "source": "気象庁防災情報XML",
  "source_url": "https://www.data.jma.go.jp/...",
  "area": "愛知県",
  "label": "気象警報・注意報",
  "severity": 0.0,
  "status": "注意",
  "detail": "大雨注意報に関する公式発表",
  "observed_at": "2026-05-22 10:00:00"
}
```

### 7.2 answer_verifications

这张表保存用户问题、AI 草稿回答和 Verifier 判定。

论文中可以写：

```text
answer_verifications テーブルには、AI の draft answer と
Verifier Agent の判定結果を保存する。
```

字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | TEXT | 回答记录 ID。 |
| `question` | TEXT | 用户问题。 |
| `draft_answer` | TEXT | Answer Generator 生成的草稿回答。 |
| `visible_answer` | TEXT | 最终允许显示的回答。若被阻止则为空。 |
| `verdict` | TEXT | Verifier 判定：PASS / NEEDS_REVIEW / FAIL。 |
| `display_policy` | TEXT | 前端显示策略：SHOW / SHOW_WITH_WARNING / DO_NOT_SHOW。 |
| `warning` | TEXT | 注意提示。 |
| `reasons_json` | TEXT | Verifier 理由和 checked_claims。 |
| `model` | TEXT | 使用的 LLM 模型。 |
| `provider` | TEXT | AI provider，例如 ollama / api。 |
| `ai_error` | TEXT | AI 调用错误信息。 |
| `created_at` | TEXT | 保存时间。 |

典型记录示意：

```json
{
  "question": "地下通路を使えば安全ですか？",
  "draft_answer": "地下通路なら雨を避けられるため安全です。",
  "visible_answer": null,
  "verdict": "FAIL",
  "display_policy": "DO_NOT_SHOW",
  "warning": "公式情報と矛盾する可能性があります。",
  "reasons_json": {
    "reasons": ["公式情報では地下通路の利用回避が示されている"],
    "checked_claims": ["地下通路は安全である"]
  }
}
```

## 8. 官方信息来源

当前 `official_sources.py` 中包含的来源包括：

- 気象庁防災情報XML
- 愛知県 災害関連情報ポータル
- 名古屋市 防災・危機管理
- 名古屋鉄道 運行情報
- 名古屋市交通局 運行情報
- JR東海 運行情報
- JARTIC 道路交通情報Now!!
- NEXCO中日本 交通情報
- 愛知県警察 交通事故発生状況

这些来源主要用于证明系统可以获取 live official evidence。

## 9. Prompt 结构

当前有三类 prompt。

### 9.1 Official Normalization Prompt

位置：

```text
prompts.py / build_official_normalization_prompt()
```

作用：

```text
将官方页面/API 的原始文本整理成统一 evidence record。
```

### 9.2 Answer Generator Prompt

位置：

```text
prompts.py / build_answer_prompt()
```

作用：

```text
根据用户问题和 Evidence DB 生成 draft_answer。
```

约束：

- 不编造官方信息中没有的事故、灾害、延迟
- 不明时回答不明
- 明示应确认的官方信息

### 9.3 Verifier Agent Prompt

位置：

```text
prompts.py / build_verifier_prompt()
```

作用：

```text
检查 draft_answer 是否被官方 evidence 支持。
```

检查重点：

- 是否有官方 evidence 支持
- 是否与官方信息矛盾
- 是否包含危险建议
- 是否包含过度安心表达
- 是否在信息不足时断言安全

## 10. API 一览

| API | 方法 | 作用 |
|---|---|---|
| `/health` | GET | DB 和 AI 连接状态确认。 |
| `/system/overview` | GET | 系统整体 pipeline 和 DB 状态。 |
| `/dashboard` | GET | 前端首页需要的官方信息概要。 |
| `/official/sources` | GET | 当前官方信息源和 status 一览。 |
| `/official/live` | GET | Evidence DB 中保存的官方信息。 |
| `/official/live/weather` | GET | 直接从气象厅 XML 获取天气信息。 |
| `/official/sync` | POST | 同步官方信息到 Evidence DB。 |
| `/official/live/sync` | POST | 旧接口兼容，同步官方信息。 |
| `/ask` | POST | 用户提问，执行 Generator + Verifier。 |
| `/answers/cache` | DELETE | 删除回答验证缓存。 |

## 11. 论文中的写法建议

### 第3章 提案システム

可以写：

```text
本システムは、公式都市安全情報を取得する Official Data Fetch、
取得結果を保存する Evidence DB、
ユーザ質問に回答する Answer Generator、
回答内容を公式情報と照合する Verifier Agent、
判定結果に基づいて表示制御を行う Final Display から構成される。
```

### 第4章 実装

可以写：

```text
Backend には FastAPI、Database には SQLite、Frontend には React Native / Expo を用いた。
LLM 呼び出しは ai_client.py に集約し、公式情報取得は official_sources.py、
公式情報の保存と検索は db.py、回答生成と検証は answer_service.py に分離した。
```

### 数据库说明

可以写：

```text
公式情報は official_area_observations テーブルに保存される。
このテーブルは、本研究における Evidence DB として機能する。
AI回答と検証結果は answer_verifications テーブルに保存され、
Verifier Agent の判定理由、表示ポリシー、使用モデルを記録する。
```

## 12. 当前完成状态

已经完成：

- 官方信息源定义
- 官方信息获取
- Evidence DB
- Answer Generator
- Verifier Agent
- SHOW / SHOW_WITH_WARNING / DO_NOT_SHOW
- 前端 Dashboard
- 前端 AskPanel
- 后端模块化
- 前端模块化

也就是说：

```text
应用演示系统：基本完成
```
