# Strix Llama

在一台 AMD Strix Halo 机器上、在 Windows 下，把 125B 的 MoE 模型跑快。

目标平台：**Ryzen AI Max+ 395**（Radeon 8060S，gfx1151，128 GB 统一内存），模型
**Qwen3.8-Flash-Next**（125B-A6B，Unsloth UD-IQ4_XS，93.7 GB），经打过补丁的 llama.cpp 服务。

*[English](README.md)*

## 当前水平

本机实测，85K token 真实散文，推测解码开启：

| | |
| --- | --- |
| 预填充 | **886 t/s** |
| 解码，85K 上下文 | **28.7 ms/token**（34.9 tok/s，草稿接受率 68%） |
| 解码，短上下文 | **26.8 ms/token**（37.4 tok/s，接受率 60%） |
| 图像输入 | 支持（Qwen3-VL 投影模型） |

作为对照，同一台机器上 LM Studio 跑同一个模型约 18 tok/s。

本仓库里的每个数字都附带产生它的命令，见 [docs/results.md](docs/results.md)。凡是改动无法在噪声
之上分辨的，就直接写明分辨不出，而不是算作收益；解码数字一律带上草稿接受率——开着推测解码时，脱离
接受率谈吞吐，描述的是那段提示词，不是这套运行时。

## 只想跑起来？

**[docs/getting-started.zh.md](docs/getting-started.zh.md)**——从
[Releases 页面](https://github.com/rulith-dev/strixllama/releases)装安装包、要下载的五个模型文件和
放在哪、点什么。不涉及 Python、ROCm 或任何编译工具。

## 快速开始（从源码）

```bash
python bootstrap/bootstrap.py --toolchain          # 把 ROCm SDK 和 ninja 装进 toolchain/，约 5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` 按固定版本号克隆 `pwilkin/llama.cpp`、应用补丁集、对着 ROCm SDK 构建。
`tools/manager.py` 是一个从 stdin 读 JSON 的进程管理器：启动参数、环境开关和运行时都由它掌握，
所以一套配置是可复现的，而不是靠记忆。

**先读 [docs/install.md](docs/install.md)。** 有三件事不是可选的，也不显然：显存划分必须是 96 GB
（64 GB 下模型装不下，解码慢 28%，没有任何软件设置能补回来）；ROCm SDK 必须用 TheRock 10.1 而不是
系统的 7.1（同样的源码，预填充 +60%）；而实测所用的 SDK 版本来自一个约 27 天滚动窗口的 nightly
索引——也就是说这个钉死的版本号迟早会失效，那篇文档写了届时怎么办。模型文件是另外约 95 GB 的下载。

可选：`integrations/jan/apply.py` 把三个管理页面叠加进
[Jan](https://github.com/menloresearch/jan) 的源码树，中英双语。

## 里面到底有什么

相对上游 llama.cpp 的全部改动是 **27 个文件**——3610 个里改了 23 个、新增 4 个。主要几项：

| | |
| --- | --- |
| **IQ3_S 走矩阵核** | 这个模型 52% 的主体以 IQ3_S 存储，而 MMB 反量化 GEMM 不接受这种格式，于是一半权重从未到达矩阵核。预填充 +5.6%。 |
| **解码期稀疏注意力** | 稀疏内核需要只有预填充才构建的打包布局，所以解码一个 token 要在整个缓存上做稠密注意力。改为聚集被选中的单元：97K 解码 59.0 → 49.3 ms/token，上下文斜率从每千 token 0.188 降到 0.067 ms。 |
| **MTP 推测解码调优** | 草稿头配自己的 IQ4_XS 输出投影，三个草稿 token，n-gram 草稿关闭。85K 解码 21.5 → 34.9 tok/s。 |
| **两个正确性修复** | 推测验证批次跑了无 causal mask 的稠密注意力，长答案会跑偏并提前结束。图像输入曾以三种不同方式在 QSA 块机制里让服务端崩溃。 |
| **测量仪表** | 逐图、逐派发、逐阶段计时，全部默认关闭，需设环境变量才启用。 |

`patches/MANIFEST.md` 列出每个补丁、它动了哪些文件，以及必须的应用顺序。

## 构建怎么验证

**无法重放的补丁集会悄悄腐烂。** 两个工具保证这件事诚实：

```bash
python bootstrap/bootstrap.py --record    # 记录补丁集产出的哈希
python bootstrap/bootstrap.py --verify    # 当前树是否仍然完全一致
```

`--verify` 回答的是弱问题：这棵树是不是配方产出的？手改完再重新记录一遍哈希，它照样通过。强问题
是配方今天还能不能从零重建出这棵树，这有单独的工具：

```bash
python tools/replay_bootstrap.py          # 干净上游 + 补丁集 == 那 27 个文件，逐字节相同
```

它按 `bootstrap/UPSTREAM.json` 里记的 blob 哈希，从克隆自带的 git 对象里取回那 23 个文件的上游版
本——干净上游是重建出来的，不是信来的——然后重放整套快照和脚本再比对。结果是 **27 / 27**。

一开始并非如此。24 个里有 5 个没有任何脚本负责，其中就包括项目里最大的单项收益（干净重建会把它
静默丢掉）；另有 3 个脚本已经漂移到打不上去。`tools/make_patch_script.py` 从两棵树生成补丁脚本、
锚点自动扩展到唯一，这些缺口就是这样补上的——手写锚点正是最初腐烂的原因。

## 适用范围与限制

- **仅 Windows。** 管理器用 Win32 进程 API；构建目标是 gfx1151。
- **单一模型族。** 注意力部分专属于 `qwen4exp`。内核部分（IQ3_S MMB、MMVQ 每工作组行数）适用面更广。
- **不是 llama.cpp 的分叉。** 本仓库只放补丁、工具和实测数据，不含模型权重、上游源码或二进制。

## 目录结构

```
bootstrap/   拉取、打补丁、构建运行时；UPSTREAM.json 是这份 delta 的哈希记录
patches/     补丁集、清单，以及两处大到无法用锚点表达的整文件替换
tools/       管理器、测量工具、验证工具
integrations/jan/   可选的 Jan 管理页面
docs/        install.md · results.md · measuring.md · decode-budget.md · dead-ends.md
```

在相信本仓库任何数字（包括上面那几个）之前，先读 [docs/measuring.md](docs/measuring.md)——那是一份
这个项目把自己测错的方式清单。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方署名见 [NOTICE.md](NOTICE.md)——特别是本项目修改的是
[pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp)（MIT），Jan 覆盖层用于
[Jan](https://github.com/menloresearch/jan) 的源码树（Apache-2.0）。
