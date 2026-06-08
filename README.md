# DeepVision

让没有视觉能力的文本模型也能"看懂"图片 —— 但不是把图翻译成一段模糊的散文,而是产出**带坐标锚点的结构化视觉表示**。

## 它和普通"图转文字"工具的区别

普通方案把图片描述成一段话:"左上角有个按钮,旁边是输入框"。这段散文落进了所谓的 **Reference Gap** —— 下游模型无法精确推理元素的相对位置、对齐、包含关系,密集布局(UI、表格、流程图)下尤其容易逻辑断裂、产生幻觉。

DeepVision 受 [Thinking with Visual Primitives](https://github.com/ailuntx/Thinking-with-Visual-Primitives) 的"point-while-reasoning"思想启发,把图片解析成**机器可精确引用的视觉基元**:每个元素带稳定 id 和归一化坐标,下游模型可以"指着坐标推理",而不是面对模糊散文猜位置。

```
# 视觉基元(归一化坐标,原点左上)
- [email_input] 邮箱输入框 bbox=(0.10,0.40,0.60,0.46) text="Email"
- [submit_button] 提交按钮 bbox=(0.12,0.80,0.28,0.86) text="Submit"
# 关系
- email_input --above--> submit_button
```

## 核心能力

- **结构化解析**:输出视觉基元(点 / bbox)+ 空间逻辑关系,而非扁平描述
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

## CLI 用法

位置参数是图片路径(相对或绝对均可,也接受图片 URL);可一次传多个批量处理。下文 `图.png` 仅为示例路径。

```bash
deepvision 图.png                # 结构化解析,输出带坐标锚点的文本
deepvision 图.png --json         # 输出原始结构化 JSON
deepvision 图.png --flat         # 退回扁平描述(简单用法)
deepvision 图.png -q "提交按钮在哪"   # 带具体问题
deepvision -c                    # 解析剪贴板:位图截图,或复制的图片文件(多张则批量)
cat 图.png | deepvision -         # 从 stdin 读取

# 批量:多张图各自独立解析
deepvision a.png b.png c.png       # 文本输出,每张图前带 # === 路径 === 分隔头
deepvision *.png --json            # JSON 数组,每项含 source(来源)和 scene(结果)

# 覆盖参数
deepvision 图.png -m google/gemma-4-31b-it:free   # 临时换模型
deepvision 图.png -t 0.2 -s 1280                   # 温度 / 长边像素上限
```

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

cfg = Config.load()                       # 读 .deepvision.json / 环境变量
scene = describe_structured("图.png", cfg=cfg)

print(scene.to_anchored_text())           # 给下游文本模型的锚点文本
print(scene.to_json())                    # 结构化 JSON
for p in scene.primitives:                # 遍历视觉基元
    print(p.id, p.label, p.box or p.point)

# 抗幻觉核查(可选)
from deepvision.verify import verify_scene
scene = verify_scene("图.png", scene, level="suspicious")  # 或 "full"
```

## 配置优先级

显式传参 > 环境变量 > 配置文件。环境变量:

| 变量 | 含义 |
|---|---|
| `DEEPVISION_API_KEY` | API key(也兼容 `OPENAI_API_KEY`) |
| `DEEPVISION_BASE_URL` | OpenAI 兼容端点 |
| `DEEPVISION_MODEL` | 模型 id |

配置文件查找顺序:`~/.deepvision/config.json` → `./.deepvision.json`。

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

它暴露 `describe_image_structured` 和 `describe_image_flat` 两个工具。在工具的 MCP 配置里(如 Claude Desktop 的 `claude_desktop_config.json`)添加:

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
图片 ──▶ 多模态 API ──▶ 结构化基元 ──▶ 坐标归一化 ──▶ [可选]抗幻觉核查 ──▶ 锚点文本/JSON
       (任意视觉模型)   (id+坐标+关系)   (压回[0,1])    (剔除幻觉元素)        (给下游模型)
```

1. **解析**:把图片(自动缩放到长边上限)发给多模态模型,要求输出结构化 JSON(视觉基元 + 关系),而非散文。
2. **归一化**:模型常无视坐标约定,返回 `0~1000` 或像素坐标 —— 自动识别标度并压回 `[0,1]`。
3. **核查(可选)**:把可疑(或全部)基元区域裁剪出来,用"无暗示"提问独立核对,剔除模型编造的元素。
4. **输出**:锚点文本(给文本模型)或 JSON(给程序)。

## 测试

```bash
python tests/test_schema.py        # 离线测试,不需要 API key
```

## 已知限制

- **核查力度的权衡**:`--verify full` 能核查"坐标正常但内容是幻觉"的元素,成本随基元数线性增长;`--verify` 只核查可疑基元,成本低但覆盖窄。
- **模型能力决定上限**:参数量较小的视觉模型在复杂、密集的图上容易整体误判,解析质量取决于所用模型。

## 致谢

- [OpenVL](https://github.com/scp3500/openvl) —— "图转文字"接入层的工程形态参考
- [Thinking with Visual Primitives](https://github.com/ailuntx/Thinking-with-Visual-Primitives) —— point-while-reasoning 思想来源

## License

MIT



