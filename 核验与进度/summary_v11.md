# bench10 + 三篇测试论文：v11 逐图精度与跑时（Asia/Seoul）

对照：`/workspace/xrd_user/bench10/summary.md`（v9，before）、`summary_v10.md`（v10，after）。本文件是同一套文献的 **v11** 审计。官方结果在：

- 三篇测试论文：`/workspace/xrd_user/results_{dee,mat,co}_v11` 与 `*_v11_fit`
- 十篇 bench：`/workspace/xrd_user/bench10/results_v11/`（未覆盖 `results/` 或 `results_v10/`）

时间一律为 **Asia/Seoul (UTC+9)**。官方 no-fit / `--fit` / 批量跑于 **2026-11-25 12:59–13:24**。

「整图正确」标准与 v9/v10 相同：找到并正确裁切；主峰相对原图刻度/论文已给 2θ 约 **0.3° 以内**，主要实验曲线非全零、未把卡片竖线描成主峰；物相与论文一致（多报不被支持的相记为 partial，不能算整图正确）。不以流水线自己的 pass 为金标准。

---

## 头条：v9 → v10 → v11

**bench10 的 N = 18 保持不变（Fig 6a+b 仍按一幅计）。整图正确 M = 5（5/18 = 28%），与 v10 持平，但成员变了：保住 MoS2、ZIF、BSM20、LDH Fig 9b，新增 LDH Fig 4；唇膏 Fig 3 不再算整图正确。**

若把三篇测试论文的 6 幅图一并计入（N = 24）：v9 约 **4/24**（bench 2 幅 + 测试论文中 materials p5_02、Co p4_01 在 v9 轴/描线可用），v10 **5/24**（测试论文两幅回归，bench 增至 5），v11 **5/24**（测试论文两幅回归已修但仍未达整图正确；bench 仍为 5）。

| 阶段（bench10 N=18） | v9 | v10 | v11 | 说明 |
| --- | --- | --- | --- | --- |
| 找到（found） | 18/18 | 18/18 | **18/18** | Fig 6 在 v11 只裁出 6a（v10 曾拆成 6a+6b 两张裁图，仍计为同一幅） |
| 裁切可接受（crop ok） | 17/18 | 18/18 | **17/18** | LiFePO4 2b 仍分开。LDH Fig 6 只框了 (a)，(b) 未裁 |
| 数字化/峰位（peaks ok） | 2/18 | 5/18 | **5/18** | 保住 MoS2、ZIF、BSM20、LDH 9b；**新增 LDH Fig 4**（轴 10–70°）；**唇膏丢失 ~7.26° 绿泥石峰并描进 10–13° 卡片竖线** |
| 物相正确（不含 partial） | 15/18 | 17/18 | **18/18** | Fig 6 现写出 MgGa 氧化物（v10 为 partial）；HEA 仍不报 Y2O3 |
| **整图正确** | **2/18（11%）** | **5/18（28%）** | **5/18（28%）** | 见上：换掉唇膏、补上 LDH Fig 4 |
| 假阳性（不计入 N） | 1（BSM20 Fig 4c 中子 d/Å） | 0 | **0** | Fig 4c 仍未进入数字化 |

数字化仍是短板（5/18）。物相 18/18 并不等于整图正确。

---

## 测试论文 6 幅（回归项）

| paper | fig | v9 | v10 | v11 crop | v11 peaks | v11 phase | v11 accurate | v11 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dee-zn-batteries | Fig 4c | 可用但不整图正确 | 无明显致命回归 | yes | no | yes（Zn + 碱式硫酸锌） | **no** | 3 条非零。DEE 峰呈平顶/方框，BE 低角杂质区有竖线伪迹 |
| materials-15-07378 | Fig 1（p5_01） | 大体可用 | 0.5 K 几乎只剩右端尖峰 | yes（20–90°，残差 0.08°） | no | yes（赤铁矿） | **no** | 6 条都在；0.5 K 已找回，但 1 K / 3 K 在 ~88° 有边沿假峰 |
| materials-15-07378 | Fig 2（p5_02） | **轴 20–90° 可用** | **回归：残差 30°，范围约 −49–117°** | **yes（20–90°，残差 ~0）** | no | yes（赤铁矿 + 高温钾铁氧化物） | **no** | **轴回归已修**。叠加上看 5 条都有迹，但写出的 `.xy` 只有 800/700/600/500 °C，缺 400 °C |
| Co-nanorods | Fig 1（p4_01） | **6 条非零、轴 30–105° 可用** | **回归：裁到左空白轴，6/6 全零** | 部分（堆叠已框住，但右边界约 95–97°，**(103) 被切掉**） | no | yes（hcp Co） | **no** | **全零回归已修**，主峰 50/52/55/74/90 与刻度对齐；仍有方框伪迹，保存 5 条 |
| Co-nanorods | Fig 2（p4_02）Rietveld | 2 条，19–91° | 外推到 ~129° | yes | no | yes（hcp Co） | **no** | 只描了 Yobs（1 条）；高角刻度乱到 119°，~109° 有假峰。已识别为发表精修并在 `--fit` 中跳过 |
| Co-nanorods | Fig 7（p8_01） | 大体可用 | 大体可用 | 部分（同样切掉 100–105° 的 103） | no | yes（hcp Co） | **no** | 框内 100/102/55/74/90 对齐较好 |

结论：v10 对 materials p5_02 和 Co p4_01 的致命回归已经按目标修掉，但还没有回到「整图正确」。

---

## bench10 逐图 v9 → v10 → v11

| paper | fig | crop v9→v10→v11 | peaks v9→v10→v11 | phase v9→v10→v11 | accurate v9→v10→v11 | v11 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| oxide-tio2 | Fig 3（450–700 °C + P25） | y→y→y | n→n→**n** | y→y→y | n→n→**n** | 框 20–70° 正确，残差 0.039°。**P25 已单独描出（5 条）**，但 P25/600/700 仍箱状 |
| alloy-hea | Fig 1a+b | y→y→y | n→n→**n** | p→y→y | n→n→**n** | 4 条；烧结前 3%Y2O3 一条塌成水平线；烧结后 0% 仍严重方框。主相 fcc+B2，不再多报 Y2O3 |
| ldh-mgga | Fig 1 | y→y→y（只裁了 (b)，v10 只裁了 (a)） | n→n→**n** | y→y→y | n→n→**n** | (a)(b) 仍未同框。(b) 三条大体在，仍有方框 |
| ldh-mgga | Fig 3（原位变温） | y→y→y | n→n→**n** | y→y→y | n→n→**n** | 14 条，轴 10–70° 好一些，仍有缺口/伪迹 |
| ldh-mgga | Fig 4 | y→y→**y** | n→n→**yes** | y→y→y | n→n→**yes** | **定标已修好**（残差 0，10–70°，不再是 −15–96° / 11.7°）。三条实验/模拟主峰与刻度对齐。物相脱水 LDH + Si |
| ldh-mgga | Fig 5 | y→y→y | n→n→**n** | y→y→y | n→n→**n** | Si 与卡片竖线仍被描进曲线。物相 MgO / MgGa2O4 + Si |
| ldh-mgga | Fig 6a+b（Rietveld） | y→y（拆两张）→**只裁 6a** | n→n→**n** | y→p→**y** | n→n→**n** | 6b 未再单独裁出。6a 计算谱仍不完整。物相现写出 Si + MgGa 氧化物。`--fit` 正确跳过发表精修 |
| ldh-mgga | Fig 7 | y→y→y | n→n→**n** | y→y→y | n→n→**n** | 实验谱大致在；模拟仍有矩形缺口 |
| ldh-mgga | Fig 8 | y→y→y | n→n→**n** | y→y→y | n→n→**n** | 第 1、2 条仍有方框/截断 |
| ldh-mgga | Fig 9a | y→y→y | n→n→**n** | p→y→y | n→n→**n** | 5 条峰位大致对，仍有缺口。不再写成 Mg–Al |
| ldh-mgga | Fig 9b | y→y→y | n→y→**y** | y→y→y | n→y→**y** | 叠加仍贴合，~11° 最强峰在。物相 LDH 正确 |
| mof-zif8-zif90 | Fig 1 | y→y→y | y→y→**y** | y→y→y | y→y→**y** | 刻度 5–50°。ZIF-8 实验谱 7.30/10.36/12.66/18.00°。写出 5 条（v10 为 6，ZIF-90 计算谱仍弱）。物相 ZIF-8 + ZIF-90 |
| zeolite-lta-zn | Fig 4A | y→y→y | n→n→**n** | y→y→y | n→n→**n** | **7.20° 的 200 仍在**（6.48–90.20°），右侧 10–15° inset 已屏蔽。多条计算/观测叠线仍乱。物相 NaA+ZnO+Zn(OH)2（另列了 ZnO2）。`--fit` 正确跳过发表 Rietveld |
| battery-lifepo4 | Fig 2a+b | n→y→**y** | n→n→**n** | y→y→y | n→n→**n** | 2b 已排除，主框 10–70°。橄榄石主峰（17.16/20.80/25.60/29.70/32.26/35.60）在 0.3° 内，但 LFP-N 仍把 PDF 卡片竖线描成方波 |
| her-mos2 | Fig 2a | y→y→y | y→y→**y** | y→y→y | y→y→**y** | 叠加贴合。14.20/24.56/32.64/39.44/58.40°。物相 MoS2 + rGO |
| perovskite-lsco | Fig 1A | y→y→y | n→n→**n** | p→y→y | n→n→**n** | 框 20–60°。主峰大致在 23/33/41/47/59，多条仍有平台/方框；LCOS0 仍差。物相 LaCoO3 |
| hematite-lipstick-co | Fig 3 | y→y→y | n→y→**no** | y→y→y | n→y→**no** | Co Kα 仍对。石英 31.04、赤铁矿 104 **38.70**（对 38.72）、110 41.66 仍准，**但 ~7.26° 绿泥石峰从重绘图中消失，10–13° 出现卡片竖线方波**（对照 v10 重绘图是有 7.26、没有这组竖线的）。这是相对 v10 的退步 |
| perovskite-bascmo | Fig 3a+b | y→y→y | n→y→**y** | y→y→y | n→y→**y** | 主峰压在 **30.02°** 刻度上。其余约 21.30/37.50/43.50/54.00/63.02。范围止于 69.42°，叠加上未见 70° 满量程尖峰。物相 BSM20 |

假阳性（不计入 N）：perovskite-bascmo Fig 4c 中子 d/Å 仍被拒绝。

---

## 物相为什么像是空的（phase_empty）

`run_times_v11.json` 里每篇的 `phase_empty` 等于裁图数，这是**统计脚本的假阳性**，不是物相步骤失败。

1. 默认模型 `qwen3.5-omni-flash` 在本账户上仍然欠费。日志里每篇都有：「已改用 qwen3.8-flash 重试（不会打印密钥）」。没有一篇需要再落到 `qwen3.6-flash`（无 403）。
2. 识图是在官方墙钟里做完的，不是事后补跑。例如 dee 识图 **26.81 s**，写在 42.2 s 总墙钟里；materials 两幅各约 13–14 s；bench 单图约 10–19 s。`phase_ok` 与裁图数一致（顺序 13 篇全部 rc=0，识图失败 0）。
3. 物相正文在 `<crop>.png.json` 的 `result.phase`（candidates + summary）以及 `report.md` 里，共 **24/24** 篇顺序 no-fit 裁图都有非空物相。`figures_index.json` 当时**没有复制** `phase` / `phase_model` 字段（v10 索引里的这些字段来自后来的补写）。v11 代码已改：以后会把 `phase` 写入索引。`png.json` 的 `model` 字段仍可能显示默认名 `qwen3.5-omni-flash`，但日志表明实际重试用的是 3.8。

批量（`_batch_all`）是第二次跑 10 篇：墙钟 141.45 s，**rc=2**，因为 `perovskite-lsco_p3_01_panel.png` 有一次 3.8 输出解析失败（17 成功 / 1 失败）。同一篇在顺序 no-fit 里是成功的。不要用这次批量的 1 例失败否定顺序跑的物相。

---

## 跑时（含真实 3.8 物相时间，Asia/Seoul）

与 v9/v10 第一遍「物相约 1 s 即失败、再事后补写」不同，下面是端到端、含成功 3.8 识图的墙钟。

### 顺序 no-fit（13 篇）

| 短名 | 墙钟 s | 定位 s | 裁图 | 曲线 / 全零 | 结束（UTC+9） |
| --- | --- | --- | --- | --- | --- |
| dee-zn-batteries | 42.83 | 6.82 | 1 | 3 / 0 | 13:00:03 |
| materials-15-07378 | 36.53 | 11.14 | 2 | 10 / 0 | 13:00:39 |
| Co-nanorods | 30.42 | 11.13 | 3 | 7 / 0 | 13:01:10 |
| oxide-tio2 | 29.96 | 8.30 | 1 | 5 / 0 | 13:01:40 |
| alloy-hea | 24.17 | 6.46 | 1 | 4 / 0 | 13:02:04 |
| ldh-mgga | 67.86 | 13.12 | 9 | 37 / 0 | 13:03:12 |
| mof-zif8-zif90 | 32.00 | 9.51 | 1 | 5 / 0 | 13:03:44 |
| zeolite-lta-zn | 33.96 | 7.52 | 1 | 8 / 0 | 13:04:18 |
| battery-lifepo4 | 31.78 | 6.49 | 1 | 4 / 0 | 13:04:49 |
| her-mos2 | 25.23 | 6.81 | 1 | 1 / 0 | 13:05:15 |
| perovskite-lsco | 24.85 | 6.72 | 1 | 5 / 0 | 13:05:39 |
| hematite-lipstick-co | 34.46 | 8.76 | 1 | 1 / 0 | 13:06:14 |
| perovskite-bascmo | 29.41 | 11.85 | 1 | 2 / 0 | 13:06:43 |

- bench10 十篇：中位数 **30.87 s**，最大 **67.86 s**（ldh，9 裁图），合计 333.68 s。**9/10 在 60 s 以内**；只有 LDH 因图多超过一分钟。
- 全部 13 篇：中位数 **32.00 s**，最大 67.86 s，合计 453.46 s。零全零曲线。
- 对照：v10 十篇第一遍中位数 18.65 s、最大 35.76 s，但那次物相约 1 s 失败，**不能直接和本次比快慢**。带成功 3.8 物相后，单图论文仍多在 25–35 s，符合「大约一分钟」的目标。

### 一次文件夹批量（10 篇 → `results_v11/_batch_all`）

13:06:43–13:09:05，墙钟 **141.45 s**（pipeline 140.9 s），18 裁图、72 条曲线、0 全零。比 v10 批量 92.65 s 慢，主要是因为这次包含了成功的 3.8 物相（且 18 幅图都要识图）。rc=2：LSCO 有 1 幅识图解析失败，见上。

### `--fit`（13 篇都跑了）

| 短名 | 墙钟 s | DARA 阶段 s | 结果 |
| --- | --- | --- | --- |
| dee-zn-batteries | 100.71 | 67.23 | 超时放弃（60 s），无 fit.png |
| materials-15-07378 | 71.94 | 30.84 | Fig 1：Hematite COD 1011240，Rwp=33.18%；Fig 2 有 report 但摘要写无候选 |
| Co-nanorods | 60.24 | 26.47 | Fig 1 / Fig 7：hcp Co COD 9008492，Rwp=19.51% / 20.72%；Fig 2 正确跳过发表精修 |
| oxide-tio2 | 37.91 | 0 | 未下载到 CIF，跳过 |
| alloy-hea | 34.99 | 1.44 | 未下载到 CIF，跳过 |
| ldh-mgga | 162.32 | 80.76 | Fig 6 正确跳过发表精修；若干图未下到 CIF 或 60 s 超时；Fig 3 误用了 hcp Co 参考且无候选 |
| mof-zif8-zif90 | 35.47 | 0 | 未下载到 CIF，跳过 |
| zeolite-lta-zn | 36.14 | 0 | **正确跳过**已发表 Rietveld |
| battery-lifepo4 | 97.09 | 64.67 | 代表曲线 LFP/C-N，LiFePO4 100 wt%，Rwp=21.74%，GOF=2.353，有 fit.png |
| her-mos2 | 99.11 | 67.43 | MoS2 100 wt%，Rwp=9.05%，GOF=1.513，有 fit.png |
| perovskite-lsco | 64.97 | 30.58 | LaCoO3 100 wt%，Rwp=22.78%，有 fit.png（与 v10 同类） |
| hematite-lipstick-co | 59.87 | 19.81 | 只拟合了赤铁矿，Rwp=37.24%，解读写明解释不了图中的多相 |
| perovskite-bascmo | 40.03 | 1.59 | 未下载到 CIF，跳过 |

13 篇 `--fit`：中位数 **60.24 s**，最大 **162.32 s**（ldh），合计 901.79 s。超过 60 s 的主要是 DARA/超时，不是 no-fit 定位。

拟合图（有写出的）：

- `/workspace/xrd_user/bench10/results_v11/battery-lifepo4_fit/fits/battery-lifepo4_p4_01_00_LFP_C-N/fit.png`
- `/workspace/xrd_user/bench10/results_v11/her-mos2_fit/fits/her-mos2_p5_01_00_MoS2_NSs_rGO/fit.png`
- `/workspace/xrd_user/bench10/results_v11/perovskite-lsco_fit/fits/perovskite-lsco_p3_01_00_LCOS30/fit.png`

---

## 仍失败的主要问题（如实）

1. **堆叠/多色描线仍不稳**：TiO2 虽找回 P25 但仍箱状；HEA 一条塌缩、烧结后 0% 方框；LSCO / 多数 LDH / LFP-N 卡片竖线仍在。
2. **唇膏退步**：丢失 ~7.26° 峰，10–13° 描进卡片竖线，不再整图正确。
3. **裁切仍会切掉右端**：Co Fig 1 / Fig 7 把 100–105° 的 (103) 切在框外；LDH Fig 6b 未裁。
4. **materials p5_02** 轴已对，但 400 °C 未写入 `.xy`。
5. **CIF / DARA**：若干相下不到 CIF；dee 与部分 LDH 60 s 超时；超时或跳过是预期行为，不是 no-fit 失败。

---

## 最有用的对照图

- materials p5_02 轴从 v10 的 −49–117° 回到 20–90°：
  - v11 validate：`/workspace/xrd_user/results_mat_v11/digitize/materials-15-07378_p5_02/validate.png`
- Co p4_01 不再全零（右端仍切到约 97°）：
  - v11 validate：`/workspace/xrd_user/results_co_v11/digitize/Co-nanorods-10-00334_p4_01/validate.png`
- 新的整图正确：LDH Fig 4（轴 10–70°）：
  - v11 validate：`/workspace/xrd_user/bench10/results_v11/ldh-mgga/digitize/ldh-mgga_p7_02/validate.png`
- 对照：唇膏 v11 重绘图缺 7.26°，v10 重绘图有该峰：
  - v11：`/workspace/xrd_user/bench10/results_v11/hematite-lipstick-co/digitize/hematite-lipstick-co_p3_01/replot.png`
  - v10：`/workspace/xrd_user/bench10/results_v10/hematite-lipstick-co/digitize/hematite-lipstick-co_p3_01/replot.png`

跑时 JSON：`/workspace/xrd_user/bench10/run_times_v11.json`。进度备忘：`/workspace/xrd_user/bench10/fix_progress.md`。
