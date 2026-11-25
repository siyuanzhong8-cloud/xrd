# vendor 说明

这里存放从「文献 XRD 工作台」（XRD-Workbench-Dev）原样复制的描线与 COD 查询脚本，供 `digitize_crop.py` / `phase_fit.py` 调用。核心算法未改，只作为依赖放入本工具，避免运行时还要去找工作台目录。

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `digitize_xrd.py` | `skill/scripts/digitize_xrd.py` | 按 x 像素–2θ 锚点把堆叠谱描成 `.xy` |
| `color_traces.py` | 工作台根目录 | 按颜色/分格面板分离多条曲线 |
| `cod_refs.py` | 工作台根目录 | 查询并下载 COD 晶体库的 CIF |

描线时的横坐标定标不再使用 Windows OCR，而改用百炼视觉模型在裁图上读出的刻度位置（见 `calib_read.py`）。

调用 `digitize_xrd.py` 时必须加 `--refs none`，否则会套用脚本里写死的锐钛矿/金红石参考峰，对其他物相会误导质检。
