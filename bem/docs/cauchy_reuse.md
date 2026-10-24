# 重用已建立的 CQ 頻譜：研究原型

2026-10-24。此路徑針對相同幾何／材料／積分的時間網格比較，
尚未產生新的完整事件或證明加速。動機是完整2 us／Q12建核已很昂貴，
下一組4 us／Q12比較應盡量重用已有計算。

令已取樣的 work-resolvent 為
`F(z) = D [Z(delta(z)/dt_source)+C+K]^-1 D`，
其中 `delta(z)=1.5-2z+z²/2`。新頻率 `s` 對應原平面的座標為

```text
h(s) = 2 - sqrt(1 + 2 dt_source s)
     = (3 - 2 dt_source s) / (2 + sqrt(1 + 2 dt_source s)).
```

在原等角 circle 上，用權重與節點成正比的 barycentric Cauchy 比值，
計算樣本的 DFT 插值多項式；常數 density 提供分母。
這是本實作的代數重用方法，並非把
[Parsimonious CQ](https://arxiv.org/html/2410.15079v1)
的誤差定理套用到另一條 contour。

`tools/cauchy_reuse_experiment.py` 提供小型直接加總、任意 normalization 的
批次 Cauchy backend、可選的 [FMM2D](https://github.com/flatironinstitute/fmm2d)
多 density 後端，以及對新座標的解析導數。另可直接呼叫官方 Fortran wrappers，
避免依賴特定 Python／NumPy ABI；該介面限定官方預設的32-bit INTEGER。
FMM2D未列為套件必要依賴，尚未在本機進行大型重用量測。

半頻譜按 Hermitian symmetry 補全，兩個實軸端點採與 IRFFT 相同的實數投影。
剛好落在原節點時直接取樣本，導數使用節點處的 barycentric 公式。
新座標必須滿足 `abs(h)<1`，並限制
`max(1,abs(h)/radius)^N <= 4`；後者只是外推放大 guard，沒有保證 transfer 誤差。
因此不能直接用這條路徑外推所有更細時間網格的高頻值。

`tools/audit_cauchy_reuse.py` 讀取唯讀、經 SHA256 核對的原頻譜，
在指定新 contour 點重建完整矩陣，再以新算出的原始 BEM 矩陣比較。
實際算子核對通過前，不會建立或宣稱有效的新時間歷史。
例如在其他重計算完成後可執行：

```sh
FDBEM_CORES=1 bem/run.sh cauchy-reuse-audit \
  bem/outputs/direct-resolvent-fine2-q12 --dt 4e-6 --fft-size 32768 \
  --output bem/outputs/cauchy-fine2-to4-audit
```

若使用現有 Python直接執行，需設 `PYTHONPATH=bem/src`。
大型 circle 重用須使用FMM等快速加總，
直接加總只保留作少量目標點的獨立參考。

這個導數也為固定幾何的材料 AD 提供一條後續路徑。固定 Poisson ratio 時，
`Z_new(s)=alpha*Z_old(s*sqrt(beta/alpha))`，
`alpha=E_new/E_old`、`beta=rho_new/rho_old`。但保存的是 resolvent，
故不能把它直接乘上 `1/alpha`。令
`T=D^-1(C+K)D^-1`、`H0`是變換頻率處的原resolvent，則

```text
H_new = solve(alpha*I + (1-alpha)*H0*T, H0).
```

這是由矩陣恆等式推導的後續設計，尚未接成完整材料 AD。
未來須同時檢查頻率插值、矩陣導數、時間歷史及初始場／載入依賴。
現有[條件式接觸 AD](autodiff.md)的範圍並未因此自動擴充。
