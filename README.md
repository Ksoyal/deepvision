# DeepVision

让没有视觉能力的文本模型也能"看懂"图片 —— 但不是把图翻译成一段模糊的散文,而是产出**带坐标锚点的结构化视觉表示**。

## 它和普通"图转文字"工具的区别

普通方案把图片描述成一段话:"左上角有个按钮,旁边是输入框"。这段散文落进了所谓的 **Reference Gap** —— 下游模型无法精确推理元素的相对位置、对齐、包含关系,密集布局(UI、表格、流程图)下尤其容易逻辑断裂、产生幻觉。

DeepVision 受 [Thinking with Visual Primitives](https://github.com/ailuntx/Thinking-with-Visual-Primitives) 的"point-while-reasoning"思想启发,把图片解析成**机器可精确引用的层级视觉表示**:既有可读的语义单元,也有带稳定 id 和归一化坐标的视觉基元,下游模型可以先理解结构,再"指着坐标推理"。

```
# 语义单元
- [login_form] 登录表单 role=form bbox=(0.08,0.32,0.64,0.88) text="Email 输入框和 Submit 按钮" children=[email_input, submit_button]
# 视觉基元(归一化坐标,原点左上)
- [email_input] 邮箱输入框 bbox=(0.10,0.40,0.60,0.46) text="Email"
- [submit_button] 提交按钮 bbox=(0.12,0.80,0.28,0.86) text="Submit"
# 关系
- email_input --above--> submit_button
```

## 核心能力

- **结构化解析**:输出语义单元(composites)+ 视觉基元(primitives)+ 空间逻辑关系,而非扁平描述
- **坐标自适应**:模型不管返回 `[0,1]` / `0~1000` / 像素坐标,自动归一化
- **抗幻觉自检**:可选的视觉证据核查,剔除模型凭空编造的元素(力度可选)
- **多种接入**:CLI、Python API、skill、MCP server

## 快速开始

```bash
# 安装
git clone https://github.com/Ksoyal/deepvision.git
cd deepvision
pip install -e .

# 配置:把 api_key、base_url、model 写入 ~/.deepvision/config.json
deepvision init --api-key <key> --base-url <端点> --model <模型id>
# 示例(OpenAI):deepvision init --api-key sk-xxx --base-url https://api.openai.com/v1 --model gpt-4o

# 解析一张图
deepvision 图片.png
```

最小配置(`.deepvision.json`):

```json
{
  "api_key": "<key>",
  "base_url": "<端点>",
  "model": "<模型id>"
}
```

示例(OpenAI):`api_key` 填 `sk-xxx`,`base_url` 填 `https://api.openai.com/v1`,`model` 填 `gpt-4o`。

DeepVision 通过 `base_url` + `model` + `api_key` 调用模型,兼容 OpenAI `/chat/completions` 协议、且模型支持图片输入的端点均可使用。可选端点见下方[选模型](#选模型)。

卸载:

```bash
pip uninstall deepvision
rm -rf ~/.deepvision        # 删除配置(Windows: rmdir /s %USERPROFILE%\.deepvision)
```

## 升级

如果是从 GitHub clone 后用 `pip install -e .` 安装:

```bash
cd deepvision
git pull
pip install -e .
```

如需 MCP 依赖:

```bash
pip install -e ".[mcp]"
```

如果是直接从 GitHub 安装:

```bash
pip install --upgrade --force-reinstall "git+https://github.com/Ksoyal/deepvision.git"
```

注意:Python 包升级不会自动更新已复制到工具目录的 skill。Claude Code 用户如安装了 `skills/deepvision/`,升级后请重新复制:

```bash
cp -r skills/deepvision ~/.claude/skills/
```

## CLI 用法

位置参数是图片路径(相对或绝对均可,也接受图片 URL);可一次传多个批量处理。下文 `图.png` 仅为示例路径。

```bash
deepvision 图.png                # 结构化解析,输出带坐标锚点的文本
deepvision 图.png --json         # 输出原始结构化 JSON
deepvision -c                    # 解析剪贴板:位图截图,或复制的图片文件(多张则批量)
cat 图.png | deepvision -         # 从 stdin 读取

# 批量:多张图各自独立解析
deepvision a.png b.png c.png       # 文本输出,每张图前带 # === 路径 === 分隔头
deepvision *.png --json            # JSON 数组,每项含 source(来源)和 scene(结果)

# 覆盖参数
deepvision 图.png -m google/gemma-4-31b-it:free   # 临时换模型
deepvision 图.png -t 0.2 -s 1280                   # 温度 / 长边像素上限
deepvision 图.png --intent count --target 手指      # 精确计数指定目标
deepvision 图.png --intent ocr                     # 精确转写 / OCR
deepvision 图.png --detail fine                    # 高级控制:精细输出粒度
deepvision 图.png --no-cache                       # 禁用缓存,强制重新请求
```

`deepvision -c` 会先使用标准剪贴板读取；Windows 上遇到 Pillow 无法读取的位图剪贴板格式时，会自动用 STA 剪贴板 fallback 在内存中转成 PNG，不需要接入工具自行保存临时文件。

`--intent` 是推荐入口,用于告诉 DeepVision 当前任务类型:

- `general`: 默认通用解析。
- `count`: 精确计数,先建立候选实例列表并逐个核查边界、遮挡、重叠、局部可见和相似干扰,避免被常识默认数量带偏。
- `ocr`: 精确转写文字、代码、表格文本。
- `locate`: 精确定位,用于回答位置、方向、第几个、靠近什么。
- `inspect`: 仔细复核、找异常、找差异或检查不一致。

`count` 模式的主流程是候选实例核查,不依赖图片里有无标记。图中存在重复小型彩色标记点时会附加 `visual_markers` 本地视觉线索,用于提醒下游逐点核对。该线索不绑定手指等特定对象,也不是最终答案;如果标记点和目标对象不对应,应忽略这条线索。

`--target` 是可选的短目标词,用于告诉 `count/ocr/locate/inspect` 关注什么,例如 `--target 手指`、`--target 公式`、`--target 表格`。不要把完整用户问题放进 `--target`。`target` 会按 `intent` 解释:`count` 下用于逐个计数,`ocr` 下用于完整转写目标区域并保留公式上下标/分式等结构,`locate` 下用于定位,`inspect` 下用于复核异常。

`--detail` 是高级粒度控制,默认是 `standard`:

- `brief`: 低噪声概览,只保留主要对象、区域和关键文本。
- `standard`: 默认模式,输出语义结构和必要视觉基元,不拆单字符或单 token。
- `fine`: 精细 OCR / 精确结构化。非 `general` 的 `--intent` 会自动使用 fine 级约束。

### 响应缓存

默认缓存模型响应到 `~/.deepvision/cache/`,同图同请求不重复调用 API,省额度也更快。缓存键由模型 + 提示词 + 图片内容决定；不同 `--detail`、`--intent` 或 `--target` 会使用不同提示词,因此 `standard`、`fine`、`count` 等模式不会互相复用旧结果。结构化解析默认是问题无关的客观全量表示,所以同一张图在相同模型、相同 prompt 下只会解析一次,后续直接复用。

缓存目录不会无限增长:条目数超过 `cache_max_entries`(默认 1000,设 0 不限)时,按最近最少使用自动淘汰最旧的。也可手动管理:

```bash
deepvision cache            # 查看缓存位置、条目数、占用大小
deepvision cache --clear    # 清空缓存
```

改了 prompt、换了模型,或想看模型的新输出时,加 `--no-cache` 强制重新请求。

### 核查力度

模型有时会凭常识先验编造不存在的元素(且给高置信度)。默认不核查,`--verify` 提供两档:

```bash
deepvision 图.png                 # 不核查(默认):最快,完全信任模型
deepvision 图.png --verify        # 只核查"可疑"基元:退化框/越界/
                                  #   异常长宽比/低置信度。请求少,避开限流
deepvision 图.png --verify full   # 全量逐个核查:最严格,能抓"坐标正常但内容
                                  #   是幻觉"的元素;但请求数=基元数,慢/易触发限流
```

核查过程全程记录在输出 JSON 的 `meta.verification` 里,可审计每个基元为何被保留或剔除。

## Python API

```python
from deepvision import describe_structured, Config

cfg = Config.load()                       # 读全局/项目配置与环境变量
scene = describe_structured("图.png", cfg=cfg)
scene = describe_structured("图.png", cfg=cfg, intent="count", target="手指")
scene = describe_structured("图.png", cfg=cfg, detail="fine")   # 高级控制:brief/standard/fine

print(scene.to_anchored_text())           # 给下游文本模型的锚点文本
print(scene.to_json())                    # 结构化 JSON
for p in scene.primitives:                # 遍历视觉基元
    print(p.id, p.label, p.box or p.point)
for c in scene.composites:                # 遍历聚合语义单元
    print(c.id, c.role, c.text, c.children)

# 抗幻觉核查(可选)
from deepvision.verify import verify_scene
scene = verify_scene("图.png", scene, level="suspicious")  # 或 "full"
```

## 配置优先级

显式传参 > 环境变量 > 项目配置(`./.deepvision.json`) > 全局配置(`~/.deepvision/config.json`)。环境变量:

| 变量 | 含义 |
|---|---|
| `DEEPVISION_API_KEY` | API key(也兼容 `OPENAI_API_KEY`) |
| `DEEPVISION_BASE_URL` | OpenAI 兼容端点 |
| `DEEPVISION_MODEL` | 模型 id |

配置加载时先读全局默认,再用当前目录的 `.deepvision.json` 覆盖;如果配置文件里仍是 init 生成的占位 API key,`OPENAI_API_KEY` 也会作为兜底。

## 选模型

DeepVision 调用 OpenAI 风格的 `/chat/completions` 接口。兼容该协议、且模型支持图片输入(multimodal)的端点均可使用,包括商用 API、聚合平台、本地推理(Ollama / vLLM / LM Studio)。配置项为 `base_url`、`model`、`api_key`。

> 纯文本模型无法使用,模型必须能接收图片。

`config.example.json` 附有几组常见端点的 `base_url` / `model` 写法。

## 接入工具

DeepVision 可通过三种方式接入 AI 工具:skill、CLI、MCP server。

### Skill

仓库的 [`skills/deepvision/`](skills/deepvision/) 目录是一个 skill,内容为一段说明,引导 AI 在遇到图片时调用 `deepvision` 命令。安装好 deepvision(见上方「快速开始」)后,把它复制到工具识别的 skills 目录即可:

```bash
# Claude Code:复制到项目或用户级 skills 目录
cp -r skills/deepvision ~/.claude/skills/
```

AI 在遇到图片时会调用 `deepvision` 解析,把结构化结果用于回答。其他支持 skill / 自定义指令的工具同理:把 `SKILL.md` 的内容接入它们各自的指令机制。

### 直接用 CLI

在能执行命令的环境中直接调用:

```bash
deepvision 图.png > scene.txt    # 结构化表示存成文本,粘贴给文本模型
```

### MCP server

供只支持 MCP 的工具使用:

```bash
pip install -e ".[mcp]"
python -m deepvision.mcp_server
```

它暴露两个工具:

- `describe_image_structured(path, detail="standard", intent="general", target="", refine=false)`: 解析明确路径图片。
- `describe_clipboard_image(detail="standard", intent="general", target="", refine=false)`: 直接读取剪贴板图片,不需要 agent 自行保存临时文件。

`detail`、`intent`、`target` 的含义与 CLI 一致。例如计数手指可用 `intent="count", target="手指"`;识别公式可用 `intent="ocr", target="公式"`。

在工具的 MCP 配置里(如 Claude Desktop 的 `claude_desktop_config.json`)添加:

```json
{
  "mcpServers": {
    "deepvision": {
      "command": "python",
      "args": ["-m", "deepvision.mcp_server"],
      "env": {
        "DEEPVISION_API_KEY": "<key>",
        "DEEPVISION_BASE_URL": "<端点>",
        "DEEPVISION_MODEL": "<模型id>"
      }
    }
  }
}
```

## 工作原理

```
图片 ─▶ 多模态 API ─▶ 层级结构 ─▶ 坐标归一化 ─▶ 排序+几何关系推导 ─▶ [可选]核查 ─▶ 锚点文本/JSON
      (任意视觉模型)  (语义单元+基元+关系) (压回[0,1])  (父容器内算空间关系)  (剔除幻觉)    (给下游模型)
```

1. **解析**:把图片(自动缩放到长边上限)发给多模态模型,要求输出结构化 JSON(语义单元 + 视觉基元 + 语义关系),而非散文。模型只产语义关系(labels/points_to/part_of)。
2. **归一化**:模型常无视坐标约定,返回 `0~1000` 或像素坐标 —— 自动识别标度并把语义单元和视觉基元都压回 `[0,1]`。
3. **排序 + 几何关系推导**:基元按阅读顺序(上→下、左→右)聚簇,缓解下游 Lost-in-the-Middle;above/left_of/contains 等几何关系直接从坐标确定性推导,并限制在同一父语义单元内,既消冗余也避免跨题、跨表格、跨面板关系污染上下文。
4. **核查(可选)**:把可疑(或全部)基元区域裁剪出来,用"无暗示"提问独立核对,剔除模型编造的元素。
5. **输出**:锚点文本(给文本模型)或 JSON(给程序)。

## 测试

```bash
python -m pytest -q                # 离线测试,不需要 API key
```

## 已知限制

- **核查力度的权衡**:`--verify full` 能核查"坐标正常但内容是幻觉"的元素,成本随基元数线性增长;`--verify` 只核查可疑基元,成本低但覆盖窄。
- **模型能力决定上限**:参数量较小的视觉模型在复杂、密集的图上容易整体误判,解析质量取决于所用模型。

## 致谢

- [OpenVL](https://github.com/scp3500/openvl) —— "图转文字"接入层的工程形态参考
- [Thinking with Visual Primitives](https://github.com/ailuntx/Thinking-with-Visual-Primitives) —— point-while-reasoning 思想来源

## License

MIT
