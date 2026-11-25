# XRD 识图（通义千问视觉模型）

批量调用阿里云百炼通义千问视觉模型，识读 XRD 图谱（版式、峰位、物相、合成路线建议）。

**完整中文说明（安装、申请 API Key、PowerShell 设置环境变量、示例命令、费用与注意事项）请看：[使用说明.md](使用说明.md)。**

有原始 `.xy` 数据时，请先用 `xy_peaks.py` 寻峰，并用 `xrd_vl.py --data` 把峰表与图片一起送给模型——读数据比读图准。

快速检查（无需 API Key）：

```powershell
pip install -r requirements.txt
python xrd_vl.py .\examples --dry-run
python xy_peaks.py .\examples\TiO2_anatase.xy --plot
```

从文献 PDF 找出 XRD 图、数字化曲线并识图：

```powershell
python pdf_xrd.py .\examples\papers --dry-run
python pdf_xrd.py .\examples\papers
```

完整说明见 [使用说明.md](使用说明.md) 第 4.1 节。
