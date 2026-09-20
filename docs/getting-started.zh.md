# 安装包上手指南

*[English](getting-started.md)*

这一页讲的是用发布的安装包运行 Strix Llama。这条路不需要 Python、ROCm、Visual Studio，也不需要
源码——安装包自带整套运行时。自己编译是另一页：[install.md](install.md)。

全部流程是：装应用，把五个文件下载到同一个文件夹，把应用指向这个文件夹，点"加载模型"，开聊。

## 需要什么

| | |
|---|---|
| 机器 | **Ryzen AI Max+ 395**——Radeon 8060S，`gfx1151`，128 GB 统一内存。自带的内核只为 gfx1151 编译，其他 GPU 上模型加载不起来。 |
| 显存划分 | **96 GB**，在 BIOS 里设置（选项一般叫 *UMA Frame Buffer Size* 或"专用显存"）。64 GB 下 93.7 GB 的模型会溢出到共享内存，解码慢 28%，没有任何软件设置能补回来。 |
| 系统 | Windows 11 |
| 磁盘 | 模型文件约 100 GB，应用 0.5 GB |
| 下载 | 五个文件，约 97 GB，来自 Hugging Face |

## 1. 装应用

从 [Releases 页面](https://github.com/rulith-dev/strixllama/releases)下载
`Strix-Llama_<版本>_x64-setup.exe`，运行。安装到此为止：应用本体、用我们的 fork 编译的
`llama-server`、它需要的 ROCm 库、进程管理器和它自己的 Python 都在里面，不用再装任何东西。

升级时直接用新安装包覆盖安装，聊天记录和设置都保留。

## 2. 下载模型文件——全部放进同一个文件夹

应用要加载三类文件，第二、三类是**按文件名在第一类旁边找**的，所以五个文件必须在同一个文件夹里，
文件名保持原样。

推荐用 LM Studio 的目录布局：

```
D:\models\unsloth\Qwen3.8-Flash-Next-GGUF\
```

如果 LM Studio 已经下载过这个模型，大部分文件你已经有了，可以直接跳到下一节——应用会原地读
LM Studio 的目录，不复制。

五个文件都来自同一个 Hugging Face 仓库
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)：

| # | 仓库内路径 | 大小 | 作用 |
|---|---|---|---|
| 1 | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 11 MB | 模型分片 1/3（元数据） |
| 2 | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49.8 GB | 模型分片 2 |
| 3 | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43.8 GB | 模型分片 3 |
| 4 | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 1.9 GB | MTP 草稿：推测解码，解码速度约 +60% |
| 5 | `mmproj-F16.gguf` | 904 MB | 视觉投影：图像输入 |

**为什么正好是这几个。** 本项目公布的所有数字都是在 `UD-IQ4_XS` 这个量化上测的；同一模型的其他
量化也能加载，但没有针对它们调过、也没验证过。第 1–3 个是模型本体，必须有。第 4、5 个是可选的：
草稿和投影在文件夹里就自动开启，不在就保持关闭，"模型配置"页会说明没找到哪一个。没有草稿，
解码慢约 40%。（第 4 个换成 `mtp-…-shared-Q8_0.gguf` 也行，两者相差不到 1%。）

**再加一个，可选：草稿头**——`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`，349 MB，在本项目的
[Releases 页面](https://github.com/rulith-dev/strixllama/releases)。放到第 4 个文件所在的同一个
文件夹。下次重新扫描时应用会把两者合并成 `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf`
（几秒钟，只做一次）并优先使用：英文解码快 5%，中文快 9%，接受率不变，下面的数字就是这个配置测的。
见[第 5 节](#5-可选更快的草稿头解码-5–9)。

### 怎么下

用 [aria2](https://aria2.github.io/) 在 PowerShell 里跑——五个文件平铺进一个文件夹，断点续传，
这么大的文件我们知道的最快办法：

```powershell
$base = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main"
$dir  = "D:\models\unsloth\Qwen3.8-Flash-Next-GGUF"
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($f in
    "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
    "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
    "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf",
    "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf",
    "mmproj-F16.gguf") {
  aria2c -x 8 -s 8 -c -d $dir -o (Split-Path $f -Leaf) "$base/$f"
}
```

国内访问 Hugging Face 慢或不通的话用镜像：把 `$base` 里的 `huggingface.co` 换成 `hf-mirror.com`。

用 Hugging Face 命令行也可以（`pip install huggingface_hub`）：

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir D:\models\unsloth\Qwen3.8-Flash-Next-GGUF \
  --include "UD-IQ4_XS/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" "mmproj-F16.gguf"
```

这种方式会保留仓库的子目录（`UD-IQ4_XS\`、`MTP\`），下完之后**把那四个文件上移**到
`mmproj-F16.gguf` 所在的文件夹。用镜像时先设 `HF_ENDPOINT=https://hf-mirror.com`。
（旧版本的命令叫 `huggingface-cli download`。）

用浏览器也行：在仓库页面逐个打开这五个路径下载，全部存到同一个文件夹。

## 3. 告诉应用文件夹在哪

打开应用。侧栏 **Strix Llama** 下面打开**模型**页。

- 装了 LM Studio 的话，它的下载目录会自动登记，模型已经在列表里。
- 否则点**模型目录**，填上级目录——`D:\models`——每行一个，点**保存并扫描**。目录是递归扫描的，
  填上级目录就够。

模型会显示成一行，标着 *3 分片*。显示"文件不完整"说明缺分片。列表有缓存：加了或移动了文件之后点
**重新扫描**。把文件类型筛选切到 *MTP 草稿* 或 *视觉投影*，可以确认第 4、5 个文件被识别到了。

## 4. 加载，开聊

在模型那一行点**配置**，再点**加载模型**。不用先设任何东西：默认值就是所有公布数字的测量配置——
上下文 262144、batch 和 ubatch 8192、Flash Attention、稀疏注意力、MTP 草稿、图像输入全开，思考深度
"中"。开发者日志里能看到加载进度。第一次加载要从磁盘读 93.7 GB，之后会快一些。

状态变成"模型已加载"后打开聊天，模型已经选好了，直接打字。图像输入就是在消息里附一张图。

在参考机器上的实测预期：

| | |
|---|---|
| 短上下文 | 约 37 tok/s |
| 85K token 上下文 | 约 35 tok/s，预填充约 890 t/s |
| 同机同模型的 LM Studio | 约 18 tok/s |

如果加载时提示改用了共享显存，说明专用显存对这个配置不够：去 BIOS 加大划分，或者在配置页调小
上下文长度。

模型加载着的时候，`http://127.0.0.1:8080/v1` 也是一个普通的 OpenAI 兼容接口，其他客户端都能接。

## 5. 可选：更快的草稿头（解码 +5–9%）

上面的解码数字是用一个自带 IQ4_XS 输出投影的草稿文件测的，它不再借用模型的输出层，英文快 5%，
中文快 9%，接受率不变。这一页其他所有内容不需要它。

Unsloth 的 `shared-*` 草稿没有自己的输出投影，每一步草稿都要去读模型那份 521 MB 的。这个投影只是
一张张量，所以单独发布：`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`，349 MB，在
[Releases 页面](https://github.com/rulith-dev/strixllama/releases)。下载到模型文件夹里、第 4 个文件
旁边，然后在模型页点**重新扫描**。应用会在旁边写出
`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf`（两个文件逐字节拼接，几秒钟，只做一次；这是
应用唯一会写进模型文件夹的文件），弹一条提示，之后所有没手动选过草稿的配置都用它。配置页的草稿列表
里三个都能看到。

想自己生成草稿头，需要带工具链的源码树：[install.md 第 5 节](install.md#5-model-files)。

## 出问题了

- **模型页是空的。** 检查"模型目录"：只列出已登记目录下的 `.gguf`，分片模型按 `-00001-of-` 那个
  分片显示。点"重新扫描"。
- **MTP 或图像输入是关的，配置页说没找到文件。** 第 4 或第 5 个文件不在模型文件夹里（或改过名）。
  补上并重新扫描，开关会自己打开。只有当保存过的配置里手动开了开关、而文件后来没了，加载才会因此
  拒绝启动。
- **开始加载了，然后挂掉。** 看开发者日志。常见原因：GPU 不是 gfx1151，或者显存划分对这个上下文长度
  不够（应用会用共享显存重试一次并提示）。
- **8080 端口被占用。** 本应用早前实例启动的 `llama-server` 还在跑；状态页会认领它并可以卸载。
  别的程序占了这个端口就得手动停掉。
- **东西都在哪。** 应用在 `%LOCALAPPDATA%\Strix Llama`；管理器的设置、模型目录缓存和最近一次服务日志在
  它下面的 `runtime\config\jan\`；聊天记录在 `%APPDATA%\strixllama\data`。卸载会删掉应用文件夹连同
  这些设置，聊天记录保留。
