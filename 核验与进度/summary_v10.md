# bench10：v10 修复后逐图精度与跑时（Asia/Seoul）

对照基线见 `/workspace/xrd_user/bench10/summary.md`（v9，「before」）。本文件是同一 10 篇文献、同一 N=18 定义的 **after** 审计。结果在 `/workspace/xrd_user/bench10/results_v10/`（未覆盖 `results/`）。时间一律为 **Asia/Seoul (UTC+9)**。

「整图正确」标准与 before 相同：找到并正确裁切；主峰相对原图刻度/论文已给 2θ 约 **0.3° 以内**，主要实验曲线非全零、未把卡片竖线描成主峰；物相与论文一致（多报不被支持的相记为 partial，不能算整图正确）。不以流水线自己的 pass 为金标准。

---

## 头条：before → after

**N = 18 保持不变。整图正确 M = 5（5/18 = 28%；before 是 2/18 = 11%）。漏检 0。假阳性 0（before 有 1 幅）。**

| 阶段 | before (v9) | after (v10) | 说明 |
| --- | --- | --- | --- |
| 找到（found） | 18/18 | **18/18** | N 内无漏检 |
| 裁切可接受（crop ok） | 17/18 | **18/18** | LiFePO4 Fig 2 不再把 2a 全谱与 2b 放大图裁在同一 2θ 系 |
| 数字化/峰位（peaks ok） | 2/18 | **5/18** | 新增：唇膏 Fig 3、BSM20 Fig 3、LDH Fig 9b；MoS2 与 ZIF 仍过 |
| 物相正确（不含 partial） | 15/18 | **17/18** | HEA 不再多报 Y2O3；LDH Fig 9a 不再写成 Mg–Al。Fig 6a/b 改为 partial（未点名 MgGa 氧化物） |
| **整图正确** | **2/18（11%）** | **5/18（28%）** | her-mos2 Fig 2a；mof Fig 1；hematite-lipstick Fig 3；perovskite-bascmo Fig 3；ldh Fig 9b |
| 假阳性（不计入 N） | 1（BSM20 Fig 4c 中子 d/Å） | **0** | Fig 4c 被判定为非 2θ，未进入数字化 |

数字化仍是短板（5/18），不要把「物相多数正确 / 都跑通了」当成整图精度。

---

## 逐图 before / after

| paper | fig | crop b→a | peaks b→a | phase b→a | accurate b→a | after 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| oxide-tio2 | Fig 3（450–700 °C + P25） | yes→yes | no→**no** | yes→yes | no→**no** | 框 20–70° 正确。仍只有 4 条迹线（P25 未单独描出）；500/600/700 °C 仍有平台和竖框。物相锐钛矿+金红石正确 |
| alloy-hea | Fig 1a+b | yes→yes | no→**no** | partial→**yes** | no→**no** | 不再多报 Y2O3（主相 fcc+B2）。仍只有 3/4 条；烧结后 0% 曲线平台/方框伪迹严重 |
| ldh-mgga | Fig 1 | yes→yes | no→**no** | yes→yes | no→**no** | (a)(b) 同为 5–80°，可同框。迹线仍箱状、不完整，右端 80° 尖峰。物相 LDH + 2H1/3R1 正确 |
| ldh-mgga | Fig 3（原位变温） | yes→yes | no→**no** | yes→yes | no→**no** | 16 条多数只描到一两个峰或箱状。物相 LDH + Si + 高温非晶合理 |
| ldh-mgga | Fig 4 | yes→yes | no→**no** | yes→yes | no→**no** | **定标失败**（残差 11.7°，范围 −15–96°）。原图轴标注本身较乱。物相 LDH + Si 仍对 |
| ldh-mgga | Fig 5 | yes→yes | no→**no** | yes→yes | no→**no** | Si 标记竖线仍被描进曲线。物相 MgO / MgGa2O4 + Si 正确 |
| ldh-mgga | Fig 6a+b（Rietveld） | yes→yes（现拆成两幅裁图） | no→**no** | yes→**partial** | no→**no** | 6a 实验谱已不是全零，Si 与 111/200/220 宽峰大致在位，但计算谱未描、200 被截断；6b 在 ~48° 截断，后半缺失。物相写成 Si +「未知/非晶」，未点名 MgGa 氧化物 |
| ldh-mgga | Fig 7 | yes→yes | no→**no** | yes→yes | no→**no** | 实验谱几乎只留下 ~42.7° 与 ~62.3°；模拟有矩形缺口。物相尖晶石 MgGa2O4 正确 |
| ldh-mgga | Fig 8 | yes→yes | no→**no** | yes→yes | no→**no** | 第 3 条较好，第 1 条几乎只剩 003，第 2 条右端假峰。物相 MgGa-OH LDH 正确 |
| ldh-mgga | Fig 9a | yes→yes | no→**no** | partial→**yes** | no→**no** | 不再写成 Mg–Al。5 条峰位大致对，但仍有缺口/裂峰 |
| ldh-mgga | Fig 9b | yes→yes | no→**yes** | yes→yes | no→**yes** | 叠加贴合较好，~11° 最强峰已描出（before 被水平线截断）。物相 LDH 正确 |
| mof-zif8-zif90 | Fig 1 | yes→yes | yes→yes | yes→yes | yes→**yes** | 刻度 5–50°，ZIF-8 实验谱 7.30/10.34/12.64/18.0° 与常见值在 0.3° 内。ZIF-90 活化/计算谱仍不完整（v9 已有类似问题）。物相 JSON 曾解析失败，已从 raw_text 挽救：ZIF-8；摘要中同时提到 ZIF-90 |
| zeolite-lta-zn | Fig 4A | yes→yes | no→**no** | yes→yes | no→**no** | **~7.2° 的 200 峰已找回**（NaA 实验：7.18/16.1/21.62/23.96/27.1/29.92）。inset 已标记屏蔽，但第 5 条在 ~90.8° 仍有边沿伪迹，ZnO 迹线中高角偏弱。物相 NaA+ZnO+Zn(OH)2 正确 |
| battery-lifepo4 | Fig 2a+b | no→**yes** | no→**no** | yes→yes | no→**no** | 放大图 2b 已屏蔽，主框 10–70°。橄榄石主峰（17.2/20.8/25.6/29.7/32.3/35.6）在较强曲线上多在 0.3° 内，但 LFP-N 从 ~40° 起把 PDF 卡片竖线描成方波，LFP 峰形箱状。物相 LiFePO4 正确 |
| her-mos2 | Fig 2a | yes→yes | yes→yes | yes→yes | yes→**yes** | 叠加贴在原峰上。相对论文值：14.20、24.56、32.64、39.44、58.40°（110 为宽峰，叠加仍压在原图 (110) 上）。物相 MoS2 + graphene/rGO 正确 |
| perovskite-lsco | Fig 1A | yes→yes | no→**no** | partial→**yes** | no→**no** | 框 20–60° 正确，不再多报 La2O3。主峰大致在 23/33/41/47/59，但多条有平台/方框；LCOS0 几乎只剩 ~32.9°。物相钙钛矿 La–Sr–Co–O（另列 Co3O4 为次相） |
| hematite-lipstick-co | Fig 3 | yes→yes | no→**yes** | yes→yes | no→**yes** | Co Kα 识别仍对。主峰：绿泥石 ~7.26，14.92，石英 31.04（对 31.06），赤铁矿 104 **38.70**（对 38.72，before 偏 0.64°），110 41.76（对 41.63）。叠加贴合主峰。物相赤铁矿+石英+锰氧化物等与图例一致 |
| perovskite-bascmo | Fig 3a+b | yes→yes | no→**yes** | yes→yes | no→**yes** | 叠加主峰压在 30° 刻度上（before 重绘图写在 ~32°，系统偏约 2°）。其余峰约 21.3/37.5/43.5/54/63。xy/重绘图在 70° 轴脊仍有满量程尖峰，但叠加图上主峰与刻度对齐。物相 BSM20 正确 |

假阳性（不计入 N）：

| paper | fig | before | after |
| --- | --- | --- | --- |
| perovskite-bascmo | Fig 4c 中子 Rietveld（d / Å） | 被当成 2θ 数字化 | **已拒绝**（figures_index 中不再出现） |

---

## 逐篇无拟合跑时（v10 顺序，Asia/Seoul 04:40–04:43）

这些墙钟是**第一遍**跑出来的。当时默认物相模型 `qwen3.5-omni-flash` 全部返回账户欠费，相位步骤约 1 s 即失败（rc=2），但定位/定标/描线已写成结果。随后用代码里的 3.8 回退把物相补写进同一批 no-fit 目录（补跑结束 04:57）。**因此下表不是「一次成功跑通含 3.8 物相」的端到端时间。**

| 短名 | 墙钟 s | 裁图 | 曲线 / 全零 | 对照 v9 墙钟 |
| --- | --- | --- | --- | --- |
| oxide-tio2 | 18.64 | 1 | 4 / 0 | 24.81 |
| alloy-hea | 18.41 | 1 | 3 / 0 | 22.99 |
| ldh-mgga | 35.76 | 10 | 41 / 0 | 41.09（当时 9 裁图） |
| mof-zif8-zif90 | 23.38 | 1 | 6 / 0 | 19.96 |
| zeolite-lta-zn | 29.52 | 1 | 5 / 0 | 23.81 |
| battery-lifepo4 | 18.65 | 1 | 4 / 0 | 21.70 |
| her-mos2 | 16.75 | 1 | 1 / 0 | 17.83 |
| perovskite-lsco | 17.96 | 1 | 5 / 0 | 19.73 |
| hematite-lipstick-co | 17.34 | 1 | 1 / 0 | 22.59 |
| perovskite-bascmo | 18.89 | 1 | 2 / 0 | 19.44 |

- 顺序 10 篇（含失败的约 1 s 物相）：中位数 **18.65 s**，平均 **21.53 s**，最大 **35.76 s**（ldh），**10/10 都在 60 s 以内**，合计 215.30 s（v9 合计 233.95 s）。描线本身没有变慢，部分还略快。
- 一次文件夹批量（`results_v10/_batch_all`，04:43:54–04:45:27）：墙钟 **92.65 s**（pipeline 92.2 s），19 裁图、71 条曲线、2 条全零。v9 批量是 94.42 s / 19 裁图 / 80 条。批量同样是按 PDF 串行。
- 带工作物相的端到端参考：后来重跑的 3 个 `--fit` 在进入 DARA 前（定位+数字化+3.8 物相）分别是 zeolite 34.4 s、LFP 28.8 s、LSCO 30.5 s，均 < 60 s。LDH（10 图）没有再做一次带成功物相的整篇计时；物相按图与复核并行，若 10 图并发，预期仍低于 60 s，但这是推断不是新测值。

---

## --fit 三篇（相位回退已生效后重跑，05:00:36–05:03:50）

旧的 `results_v10/*_fit` 是在物相欠费时跑的，DARA 为 0、未下载 CIF，已被这次重跑替换。

| 短名 | 墙钟 s | DARA 阶段 s | 结果 |
| --- | --- | --- | --- |
| zeolite-lta-zn | 39.92（05:01:16） | 0 | **正确跳过**：识别为已发表 Rietveld，未拟合 |
| battery-lifepo4 | 64.46（05:02:21） | 30.46（其中 DARA 本体 17.3 s，CIF 1.59 s） | 代表曲线 LFP/C-N。下载 LiFePO4 CIF COD 4001845、4001846。`report.json` **candidates 非空**，Rwp=23.26%，GOF=2.573。两条 CIF 被当成同一相（各约 43.8 wt%）。解读写明这是对数字化曲线的符合程度，不是真正晶体学精修 |
| perovskite-lsco | 89.62（05:03:50） | 54.3（CIF 4.75 s，DARA 本体 14.32 s） | **已下载到 LaCoO3 CIF**（COD 1533633），不再是「未下载到 CIF」。`report.json` candidates 非空，LaCoO3 100 wt%，Rwp=27.56%，GOF=3.09。日志提示峰位偏差、Rwp 偏高，符合度一般 |

LFP / LSCO 的 `--fit` 墙钟超过 60 s，是 DARA/CIF 阶段，不是 no-fit 超时。两次都写出了 `fit.png` 与非空 `report.json`（这是 v9 没有做到的）。

拟合图：

- `/workspace/xrd_user/bench10/results_v10/battery-lifepo4_fit/fits/battery-lifepo4_p4_01_00_LFP_C-N/fit.png`
- `/workspace/xrd_user/bench10/results_v10/perovskite-lsco_fit/fits/perovskite-lsco_p3_01_00_LCOS30/fit.png`

---

## 三篇原测试论文（回归检查，未覆盖 v8/v9 目录）

| 论文 | 目录 | 墙钟 s（对照 v9） | 结论 |
| --- | --- | --- | --- |
| dee-zn-batteries | `/workspace/xrd_user/results_dee_v10` | 15.4（v9 20.73） | **无明显致命回归**。仍 1 裁 3 条、无全零。轴约 3.5–80°（v9 为 9.8–70°）。DEE 高角 55/70° 峰偏弱或缺失，BE 低角杂质峰只描到约 7° |
| materials-15-07378 | `/workspace/xrd_user/results_mat_v10` | 25.17（v9 27.59） | **有回归**。Fig（p5_01，20–90°）0.5 K 几乎只剩右端尖峰；p5_02 定标残差 30°，范围约 −49–117°（v9 是 20–90°）。根因像是图下额外数字标注被当成刻度 |
| Co-nanorods | `/workspace/xrd_user/results_co_v10` | 22.14（v9 22.37） | **有回归**。p4_01（6 条堆叠）裁框缩到左侧空白纵轴区，6 条 `.xy` **全部为零**（v9 这 6 条都有峰）。p4_02 右端外推到 ~129° 并出现轴外竖线；p8_01 大体仍可用 |

这与「不要回归三篇测试论文」的目标不符，主要是新的缩框/刻度覆盖逻辑在「左侧大段空白轴」和「轴下另有一排数字」时过激。未再改代码重跑 bench10，以免撤掉沸石/LFP/BSM20 的改进。

---

## 仍失败的主要问题

1. **堆叠/多色描线仍不稳**：TiO2 丢 P25、HEA 箱状平台、LSCO 的 LCOS0 塌缩、LFP-N 描进 PDF 卡片竖线、多数 LDH 图不完整。
2. **轴脊/右端假峰**：BSM20、沸石、若干 LDH 在 70/80/90° 出现满量程尖峰。
3. **难轴仍会定标崩溃**：LDH Fig 4 残差 11.7°；测试论文 materials p5_02、Co p4_01 被新逻辑误伤。
4. **物相**：Fig 6 未写出 MgGa 氧化物；MOF 一条 JSON 缺右括号，已从 raw_text 挽救。

---

## 最有用的前后对比图

- BSM20 定标（主峰从 ~32° 回到 30° 刻度）：
  - after overlay：`/workspace/xrd_user/bench10/results_v10/perovskite-bascmo/digitize/perovskite-bascmo_p4_01/overlay_check.png`
  - after replot：`/workspace/xrd_user/bench10/results_v10/perovskite-bascmo/digitize/perovskite-bascmo_p4_01/replot.png`
  - before validate：`/workspace/xrd_user/bench10/results/perovskite-bascmo/digitize/perovskite-bascmo_p4_01/validate.png`
- 唇膏 104 峰（38.70° vs 38.72°）：
  - after overlay：`/workspace/xrd_user/bench10/results_v10/hematite-lipstick-co/digitize/hematite-lipstick-co_p3_01/overlay_check.png`
  - after replot：`/workspace/xrd_user/bench10/results_v10/hematite-lipstick-co/digitize/hematite-lipstick-co_p3_01/replot.png`
- LiFePO4 裁切（2b 放大图已分开）：
  - after validate：`/workspace/xrd_user/bench10/results_v10/battery-lifepo4/digitize/battery-lifepo4_p4_01/validate.png`
  - after replot：`/workspace/xrd_user/bench10/results_v10/battery-lifepo4/digitize/battery-lifepo4_p4_01/replot.png`
  - before validate：`/workspace/xrd_user/bench10/results/battery-lifepo4/digitize/battery-lifepo4_p4_01/validate.png`
- 沸石 7.2° 200：
  - after replot：`/workspace/xrd_user/bench10/results_v10/zeolite-lta-zn/digitize/zeolite-lta-zn_p10_01/replot.png`
- MoS2（仍整图正确）：
  - after overlay：`/workspace/xrd_user/bench10/results_v10/her-mos2/digitize/her-mos2_p5_01/overlay_check.png`
- LDH Fig 9b（新的整图正确）：
  - after validate：`/workspace/xrd_user/bench10/results_v10/ldh-mgga/digitize/ldh-mgga_p11_02/validate.png`
  - after replot：`/workspace/xrd_user/bench10/results_v10/ldh-mgga/digitize/ldh-mgga_p11_02/replot.png`

跑时 JSON：`/workspace/xrd_user/bench10/run_times_v10.json`。进度备忘：`/workspace/xrd_user/bench10/fix_progress.md`。
