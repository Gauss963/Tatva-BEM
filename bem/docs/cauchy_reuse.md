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
FMM2D未列為套件必要依賴。本機已完成一個真實275-density row的完整目標circle量測，
尚未完成全部275 rows的新kernel與非線性驗證。
此處FMM作用在複數頻率平面的Cauchy加總，並未實作
[2026 elastodynamics論文](https://arxiv.org/html/2607.15292v1)的3D-ACA或空間FMM。

可選建置工具固定FMM2D revision `550dae5b77b1e006c8ffae37fc832f8c2b536871`，
在新資料夾保存source archive、編譯器／flags、log與library SHA256，
並以小型直接加總核對實際Cauchy值／導數。它不安裝系統套件或Python extension。
本機已有gfortran與C compiler時可執行：

```sh
FDBEM_CORES=4 bem/run.sh fmm2d-library --cores 4 \
  --output bem/outputs/fmm2d-local-library
```

半頻譜按 Hermitian symmetry 補全，兩個實軸端點採與 IRFFT 相同的實數投影。
剛好落在原節點時直接取樣本。對極靠近節點的導數，改用同一DFT多項式的
局部Taylor展開，避免Cauchy商法則的奇異項互相抵消。
normalized falling factorials避免高次N冪；只需求原節點導數時改用FFT，
避免逐節點全circle加總。獨立解析函數測試涵蓋此分支，
仍沒有據此保證真實BEM transfer導數的插值誤差。
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

另提供 `reused-resolvent-kernel`，限定已有matching且容差不大於1e-6的獨立operator audit。
它先以小型直接加總檢查實際FMM library的值／導數，再逐row重建完整新circle，
核對audit點的新BEM矩陣，最後以既有分row IFFT寫出所有correction weights及完整elastic startup。
每row結束即關閉來源／輸出的mmap，避免把整個數十GB頻譜保留為程序resident mapping。
它還會以新算出的瞬時BEM矩陣檢查CQ inverse composition；沒有修改摩擦律或刪除歷史。
這條producer已有解析矩陣的完整權重／startup回歸測試，真實完整kernel仍待執行。

```sh
FDBEM_CORES=4 bem/run.sh reused-resolvent-kernel \
  bem/outputs/direct-resolvent-fine2-q12 --dt 4e-6 \
  --operator-audit bem/outputs/cauchy-fine2-to4-audit \
  --fmm-library /path/to/libfmm2d.so --threads 4 --max-memory-mb 8192 \
  --output bem/outputs/reused-q12-dt4-kernel
```

FMM外部workspace目前採顯式reserve估計，沒有硬RSS保證；OpenMP thread數亦非硬CPU affinity。
大型結果及完整非線性／AD對照通過前，不能把此prototype列為已達成速度或精度優勢。

## 真實頻譜的抽樣核對（2026-10-25）

來源為2 us／Q12、65536點circle，目標為4 us／Q12、32768點circle。
在目標indices 0、1、64、4096、16384，重新組裝的五個完整275×275 BEM矩陣
與Cauchy重用的最大相對Frobenius差為6.46e-12，通過原1e-6門檻。
直接Cauchy加總89.19 s，新BEM組裝87.89 s；五點測試本身沒有展示加速。
讀取採最多64個完整頻率矩陣的bounded block，避免frequency-major檔案的逐row重複掃描。
大型檔案SHA256以串流核對，避免一次載入39.65 GB頻譜。

另以實際編譯的FMM2D library，在4個OpenMP threads下計算一個真實row：
65536來源節點、275個densities、16385個目標頻率，讀取5.22 s、FMM1.66 s，
程序峰值RSS約6000 MiB。五個抽樣row與新BEM的最大相對差6.78e-13。
這只量測一個row，沒有包括全部新頻譜、IFFT、因果startup或完整事件；
不得乘275後當作已量測的建核時間。
實際library的值／導數另與小型直接加總核對至約3.4e-15。
數據、library與來源指紋見[cauchy_reuse_results.json](cauchy_reuse_results.json)。

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
