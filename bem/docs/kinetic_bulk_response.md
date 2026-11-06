# 黏著／滑動的獨立空間響應對照

2026-11-04：三個實際 CQ 複數模式、兩種 frozen 接觸切線的 FEM／BEM 對照已完成。
在這六個控制中，壓力、剪力與 platen 的相對差均低於 5%。
**完整非線性事件仍[未通過原場門檻](recovered_full_event.md)**；
這些抽樣結果不能界定整段頻帶或非線性破裂軌跡的誤差。
[完整結果、來源與資源紀錄](kinetic_bulk_response_results.json)及
[六組複數 profile](diagnostics/kinetic_bulk_response/)已保存。

## 實際模型與結果

FEM 使用原 float64 模型：h=2 mm、端點 0.015625 mm，157506 個 DOF，
支承與 rigid platen 消去後為 156902 個座標、285 個接觸節點。
BEM 使用原已完成核：bulk h=10 mm、接觸 h=2 mm、端點／角落 0.015625 mm，
531 個 retained DOF、265 個接觸節點。材料、接觸 penalty、μk 與總 spring stiffness 皆核對。

頻率欄為 `Im(s)/(2π)`；第一點使用正實數 `s≈175.6733 s⁻¹`。
另外兩點的 `Re(s)` 分別約為 175.6735、177.4042 s⁻¹，取自原 Radau contour。
profile 誤差對完整共同接觸區間的複數 P1 場精確積分，保留幅值、相位與全部端點。

| 模式頻率 | frozen 接觸 | 壓力相對 P1 L2 差 | 剪力相對 P1 L2 差 | platen 相對差 |
|---|---|---:|---:|---:|
| 0 Hz | 閉合黏著 | 1.07083% | 0.44277% | 0.002500% |
| 0 Hz | 閉合、正向完全弱化滑動 | 0.63081% | 0.63081% | 約 `2.1e-10%` |
| 999.451 Hz | 閉合黏著 | 1.10889% | 0.40514% | 0.015691% |
| 999.451 Hz | 閉合、正向完全弱化滑動 | 0.56339% | 0.56339% | 0.000328% |
| 10002.142 Hz | 閉合黏著 | 3.45960% | 3.18710% | 4.56943% |
| 10002.142 Hz | 閉合、正向完全弱化滑動 | 2.18741% | 2.18741% | 0.44849% |

FEM 的各次線性方程相對 residual 皆低於 `1e-8`。
快取座標中的 BEM residual 也通過相同門檻。

## 保留滑動耦合並獨立核對

完全弱化、正向滑動時，局部切線包含 `dτ/dδn=μk Kn`、`dτ/dδs=0`。
此 nonsymmetric 切線與黏著的 `diag(Kn,Kt)` 分開組裝。
原 contact map 的 2×2 JAX `jacfwd` 與獨立 NumPy return map 均核對通過，
沒有用對稱黏著 Hessian 代替滑動切線。

令 `L` 為保存的 work-coordinate 黏著 resolvent、`D` 為滑動相對黏著的切線差，
求解 `(I+L D)x=L b`。不同 line weights 的直接 nodal-force 方程另作獨立控制。
第一個實際模式重新組裝完整原 BEM，黏著／滑動解與上述快取路徑的相對差
分別為 **1.086e-12／2.596e-13**，門檻保持 `1e-6`。

本次對抽樣矩陣的原始 bytes 計算 SHA，核對來源大小、mtime、package／case SHA；
沒有在此重讀兩個合計約 295 GB 的完整頻譜 SHA。
正在進行的原始因果稽核已另行核對完整來源頻譜。
這項控制仍未定位滑動系統的 poles，亦未證明各取樣點都在其 causal Laplace 收斂域。

## 能得出的結論與後續

這六個點支持黏著與完全滑動的抽樣空間響應一致性。
更高頻率、獨立空間／macro-time 加密、變動的接觸集合、
載入停止與破裂時求根分支仍需驗證。
目前 FEM 保存資料的[張開、彈簧分離與塑性路徑](contact_state_divergence.md)
和首組 BEM 有明顯差異；尚未由此指定唯一原因。

本次管理 wall time **1206.620 s**、實際 CPU **120.661 s**，
平均約 **0.100 核心**，與主稽核共用原四核心預算。
這些時間包含模型重建、JAX 編譯、fresh BEM 與序列控制，
不構成同精度的 FEM／BEM 完整模擬 benchmark。

工具為 `tools/compare_kinetic_bulk_response.py`，保留原案例與 kernel：

```bash
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true \
python -B bem/tools/compare_kinetic_bulk_response.py KERNEL \
  --output bem/outputs/kinetic-bulk-response-new \
  --fem-mesh 2 --fem-end-size 0.015625 \
  --mode-indices 0 131 1311 --fresh-check-index 0
```

仍需外部 CPU 管理；此命令本身不設定 hard affinity。
原 `/Volumes/Gauss-T7/tatva` 全程只讀。
