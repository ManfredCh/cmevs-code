# H100 数据快照 SHA256 清单

> 生成时间：2026-05-02 06:48 UTC（H100 节点 `node96`）
> 数据根：`/data/data_wr/data_shushu/` (7 TB 数据盘)
> **范围：仅包含本论文相关的 6 个数据子目录**

---

## 这些清单是什么

这是 H100 上**本论文相关数据**的逐文件 SHA-256 留底。每行格式：

```
<sha256-hex>  <relative-path-from-data-root>
```

可用 `shasum -a 256 -c` / `sha256sum -c` 在数据所在机器上原地校验。

`ARCHIVE_DIGESTS.txt` 是每份 `SHA256SUMS_*.txt` 自身的 SHA-256，作为各 archive 的"内容指纹"。

---

## 论文 → H100 数据映射

V4 §4.3 表 4 列出 5 个数据来源；H100 上 Blender indoor 因为采样轮次不同分成两个目录存放：

| 论文来源（V4 §4.3） | V4 声称规模 | H100 子目录 | 实际数量（manifest 行数 = 文件数）|
| --- | --- | --- | --- |
| **Blender indoor**（CC-BY 4.0）| 326 场景 / 11,583 帧 | `combined/blender_indoor_FOU_threshold-0.2_round1+2`（第 1 轮 + 第 1+2 轮合并）| 19,271 |
| | | `combined/blender_indoor_FOU_threshold-0.2_round2`（第 2 轮独立提取）| 20,673 |
| **HM3D**（adapter，不再分发）| 401 rooms / 14,475 帧 | `combined/HM3D` | 43,431 |
| **ScanNet++**（adapter，不再分发）| 500 scans / 8,267 帧 | `combined/scannetpp_threshold0.2` | 24,801 |
| **OB3D**（户外，按上游许可）| 12 场景 / 2,400 帧 | `continuous/OB3D_shushu`（12 场景 × 2 视角 = 24 份）| 7,246 |
| **TartanGround**（户外，**不再分发**）| 63 environments / 783,944 帧 | `continuous/Tartanground`（11 个环境，762 parts；其余环境仍在用户本地）| 342,064 |

**Blender 只有室内**，没有 outdoor — 论文中的 outdoor 部分由 TartanGround + OB3D 提供。

---

## 为什么没有写到 `croissant.json`

`croissant.json` 里 `cr:FileObject.sha256` 字段的语义是：**用户从 `contentUrl` 下载到的 .tar 文件的 SHA-256**。我们目前还**没有**把数据打包成最终发布 archive，因此那些字段保持 `TODO_SHA256` 占位。

H100 上的目录结构与 `croissant.json` 里的发布产物**不是一一对应**：

- `blender-indoor-archive.tar` 应是 **curator 筛选过的 11,583 帧 / 326 场景**，需要从 `round1+2` 与 `round2` 两个 H100 子目录中合并去重 + curator 选帧后再 tar，**这一步还没做**。
- `*-adapter.tar`（HM3D / ScanNet++ / TartanGround / OB3D）应是 **adapter 脚本 + 元数据包**，**不**包含数据帧。脚本目前在 `cmevs_anonymous_code_release/scripts/` + `adapters/` 下，也还没单独打包成 tar。

这里的 SHA256SUMS 只是当前 H100 数据状态的"内部留底"，**不会上传到 HF / OpenReview**，仅供你后续：
- 增量补 TartanGround 时比对哪些 part 已处理
- 重组 / 重新打包发布 archive 时确认源数据没被改动

---

## 何时把哈希填进 `croissant.json`

当你**实际打包发布 archive** 后：

```bash
# 1. 在打包好的 .tar 上重算
shasum -a 256 blender_indoor_v1.tar
# 2. 把哈希写入 croissant.json 中 blender-indoor-archive.tar 的 sha256 字段
#    或运行：tools/update_croissant_with_real_hashes.py
```

reviewer 下载验证时用的就是这个哈希。

---

## 文件清单

| 文件 | 行数（=文件数）| 大小 | H100 路径 |
| --- | ---: | ---: | --- |
| `ARCHIVE_DIGESTS.txt` | 6 | 568 B | `~/cmevs_release_hashing/manifests/` |
| `SHA256SUMS_blender_indoor_round1+2.txt` | 19,271 | 2.9 MB | `combined/blender_indoor_FOU_threshold-0.2_round1+2` |
| `SHA256SUMS_blender_indoor_round2.txt` | 20,673 | 3.0 MB | `combined/blender_indoor_FOU_threshold-0.2_round2` |
| `SHA256SUMS_HM3D.txt` | 43,431 | 5.2 MB | `combined/HM3D` |
| `SHA256SUMS_scannetpp.txt` | 24,801 | 3.0 MB | `combined/scannetpp_threshold0.2` |
| `SHA256SUMS_OB3D.txt` | 7,246 | 930 KB | `continuous/OB3D_shushu` |
| `SHA256SUMS_tartanground.txt` | 342,064 | 47 MB | `continuous/Tartanground` |

合计：**457,486 行**（一行 = 一个文件的 sha256），约 **62 MB**（不含 README）。

---

## 排除的目录

H100 数据根下还有以下子目录，但**与本论文无关**，因此**未生成 manifest**：

- `combined/data_离散_1m间距` — 1m 间距实验性采样，未进入论文最终发布
- `continuous/shushu_line_连续` — 室内连续直线轨迹，未进入论文
- `continuous/shushu_circle_连续` — 室内连续圆形轨迹，未进入论文
- `Robustness/` — 鲁棒性评估数据（洋红坏帧筛选用），未进入论文

如果未来这些目录进入发布范围，再单独跑 `~/cmevs_release_hashing/run_hashes.sh` 补哈希即可。

---

## 上游许可重要说明

- **TartanGround 不再分发为 ERP 重新编码帧**——与 V4 §4.2 表 3 `Direct release: no` 一致。这里的 sha256 仅作 H100 内部留底；最终发布只发 adapter 脚本 + 场景 id + 元数据。
- **HM3D / ScanNet++ 同理**：上游 EULA / ToS 不允许再分发衍生 RGB-D 帧。
- **OB3D**：按上游许可范围内可再分发；具体由发布前的许可核查决定。
- **可直接发布的部分**：仅 Blender indoor（CC-BY 4.0 资产渲染）。
