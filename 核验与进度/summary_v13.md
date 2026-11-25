# v13 after-fix audit (accuracy follow-up after official v12)

Times in Asia/Seoul (UTC+9). Official v12 dirs were not overwritten.
This is an offline redigitize of existing official v12 crops + stored calib into new dirs, after the accuracy patches in `/workspace/xrd_qwen/trace_xrd.py`.

## Counts

| 口径 | v11 official | official v12 (BSM corrected) | v13 after-fix |
| --- | --- | --- | --- |
| bench10 整图正确 | 5/18 | 4/18 | **5/18** |
| 含 6 篇原始论文 | 5/24 | 4/24 | **5/24** |

v12 丢失的只有 LDH Fig 9b；BSM20 的 “主峰 +0.56°” 是相对 v11 xy 的误判（v11 主峰顶同样在 ~30.50–30.54°），已在 `summary_v12.md` 纠正为准确。v13 收回 Fig 9b，bench10 回到 5/18。lipstick 有明显改善（7.26 氯泥石峰回来、10.8–11.2 卡片竖线消失），但 104 尖锐峰顶比官方 v12 弱（argmax 38.42°，38.70° 处强度约 158 vs v12 的 272），暂不把整图翻成准确。

## Recovered

- **LDH Fig 9b 实验谱 ~11° 峰**：轴下印刷数字重定标（残差 0.000→0.091°，8 个标签 10–80）。实验谱低角峰在 **11.34°**（11.0/11.4 = 96，9.9 = 0）。模拟谱仍独立，保留 ~11°、22°、34°、37°、38.3°、60°。模拟主峰从官方 v12 的 35.4°（v11 约 37.1°）移到 38.3°，与 10° 标签纠正后的整轴平移一致；目视高峰紧挨印刷 40° 左侧。
- **lipstick ~7.26° 氯泥石峰**：强度 57.5（v11/v12 均为 0）。9.9 / 10.8 / 11.2 / 12.5 卡片竖线为 0。石英 31.04 仍在。104 在 38.70±0.3° 窗内仍有强度。

## Protected controls (held)

- MoS2：14.20 / 32.64 / 58.40，范围 10.50–89.50°
- ZIF：6 条，7.30 / 12.66 / 18.00，范围 5.22–49.76°
- LDH Fig 4：轴 10.44–69.62°，实验谱 11.08 仍在；主峰约 12.52 / 14.46（与官方 v12 相同两峰）
- Co p4_01：6 条具名曲线，29.98–107.68°，主峰 ~55.5°
- TiO2：P25 仍在（25.3 = 59），5 条，20.44–69.50°
- LFP：2a/2b 分开，4 条，橄榄石 17.16 / 25.60 / 29.70 / 35.60 仍在
- BSM20：30.02–30.54 已达 ymax（与官方 v12 / v11 xy 一致），未再改轴
- lipstick 104：38.70° 仍有强度（见上，尖顶变弱）

## New damage / caveats

- lipstick 104 尖锐峰顶弱于官方 v12（见上）。未把卡片竖线重新引入。
- Fig 9b 模拟主峰随轴标签纠正从 ~35–37° 移到 ~38.3°（不是实验谱塌到模拟谱）。
- 未再跑一轮官方 13 篇 no-fit / `--fit`；v13 数字来自离线红描，约 42 s / 9 张对照图，无 API。

## Paths

- Redig: `/workspace/xrd_user/bench10/compare_v13/redig/{ldh9b,lip,ldh4,mos2,zif,lfp,tio2,bsm,co}/`
- PNGs:
  - `/workspace/xrd_user/bench10/compare_v13/ldh-mgga_p11_02_v11_v12_v13_zoom.png`
  - `/workspace/xrd_user/bench10/compare_v13/ldh-mgga_p11_02_v11_v12_v13_full.png`
  - `/workspace/xrd_user/bench10/compare_v13/hematite-lipstick-co_p3_01_v11_v12_v13_zoom.png`
  - `/workspace/xrd_user/bench10/compare_v13/hematite-lipstick-co_p3_01_v11_v12_v13_full.png`
  - also Fig 4 / BSM / MoS2 zooms in the same directory
