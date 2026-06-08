---
name: deepvision
description: 当用户提供图片(截图、UI、文档、流程图等),需要精确的元素坐标、稳定 id 或空间关系时,用 DeepVision 把图片解析成带坐标锚点的结构化视觉表示。需要可复用的结构化坐标时,即使能直接读图也适用;自身无法读图时同样适用。
---

# DeepVision 视觉解析 skill

DeepVision 把图片转成机器可精确引用的结构化表示:每个元素带稳定 id、归一化坐标 `[0,1]`、以及元素间的空间逻辑关系。

## 何时使用

- 用户给了图片(路径、截图,或提到"刚截了图""图在剪贴板里"),问"这是什么""X 在哪""布局如何"
- 需要稳定的元素 id、精确坐标或空间关系,用于后续引用(UI 自动化、表格抽取、流程图、文档解析)——即使你能直接读图,需要可复用的结构化坐标时仍应使用
- 自身读图能力不可用,或读图结果不够精确

## 怎么用

```bash
deepvision <图片路径>              # 结构化解析,输出带坐标锚点的文本
deepvision <图片路径> --json       # 输出结构化 JSON,供程序化处理
deepvision -c                      # 解析剪贴板:位图截图,或复制的图片文件(多张则批量)
```

核查(剔除模型编造的元素)。默认不核查;在以下情况加 `--verify`:解析结果将用于关键决策(如自动化点击、数据提取),或怀疑输出中存在实际不存在的元素。

```bash
deepvision <图片路径> --verify        # 核查结构可疑的基元(坐标退化/越界/低置信)
deepvision <图片路径> --verify full   # 核查全部基元;更严格,请求数等于基元数,较慢
```

响应默认缓存到 `~/.deepvision/cache/`,同图同模型同 prompt 只解析一次,后续直接复用;超过条目上限会自动按最近最少使用淘汰。改了 prompt、换了模型或要看新输出时加 `--no-cache` 强制重新请求;`deepvision cache` 查看缓存、`deepvision cache --clear` 清空。

## 输出怎么读

输出为锚点文本,每个元素形如 `[id] 标签 bbox=(x0,y0,x1,y1) text="..."`,坐标归一化到 `[0,1]`,原点在左上;关系形如 `a --above--> b`。基于这些 id 和坐标回答,不要凭空猜测位置。

## 前置

需先配置一个 OpenAI 兼容、支持图片输入的端点。运行:

```bash
deepvision init --api-key <key> --base-url <端点> --model <模型id>
```

或设置环境变量 `DEEPVISION_API_KEY` / `DEEPVISION_BASE_URL` / `DEEPVISION_MODEL`。详见项目 README。
