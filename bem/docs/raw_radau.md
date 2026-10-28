# 不依賴 BDF2 頻譜的 Radau raw 取樣核心

2026-10-28。[實際細網格時間響應](bulk_time_symbol.md)支持進一步測試 2 µs Radau，
但既有 BDF2 circle 無法安全重用到這個 RK contour。
`tools/prepare_raw_radau_resolvent.py` 現已完成兩 branch 的 fresh kernel producer，
並通過完整粗網格 circle、非線性滑移及獨立原始 RK 方程檢查。
fine 45 ms 事件與完整 Radau AD 尚未完成，仍屬實驗工具，沒有 FEM 速度優勢結論。

Radau symbol 的兩個無因次 eigenvalues 滿足

```text
w² - (4+2*z)*w + 6*(1-z) = 0.
```

在下半圓選擇連續的 square-root 分支，從兩個實際 `s=w/dt` 各自重新組裝
兩個 body 的 V／H／D，再做原 trace Schur elimination 與 work-coordinate compliance。
靠近 `z=1` 時，較慢 mode 使用 eigenvalue product 形式求值，避免相近數相減。
測試另核對其 causal limit，不以一般 eigensolver 的小 eigenvalue 當高精度 reference。

兩 branch 依序取樣，共用唯讀 geometry，維持單一 branch 的 interpolation workspace。
原 sampler 的 raw-family tolerance、Chebyshev tail、獨立 output checks、
細分及 direct fallback 全部保留。原 BDF2 symbol 在 adapter 內只作可反解的角度座標，
沒有使用任何 BDF2 frequency matrices 或時間權重。

各 branch 在半圓 endpoint 可以是複數，不能個別使用 Hermitian mirroring／irfft。
必須先得到兩個完整 branch，再以 Cayley–Hamilton 形成

```text
F0 = (w1*L0-w0*L1)/(w1-w0)
F1 = (L1-L0)/(w1-w0)
L_RK = I*F0 + Delta(z)*F1.
```

這裡的 w 為無因次 eigenvalue，與既有 compact history 的 API 一致。
組合後才可檢查 real endpoints、保存完整兩族時間權重與 finite-DFT shift wrap。
沒有刪除快 mode、非零 lag 或接觸狀態。

已完成的控制包括：連續 branch 與完整 stage symbol、所有 indices 恰好 emit 一次、
原 sampler binding 不變、dimensionless derivative scaling，以及近 causal limit 的精度。
另對真實小型兩有限體重新組裝全部 frequency matrices，將兩族完整 CQ 權重展開後，
與另一套原 compliance-condensation 路徑逐點建立的 dense stage weights 核對。
測試另保留 Q4 在 2 µs 下無法通過 inverse-composition 門檻的負例；
同一幾何與 dt 提升至 Q12／Q24／Q48 後均通過，未放寬容差。

## 實際完整 circle 與非線性控制

[量測紀錄](raw_radau_results.json) 對應 producer commit `cd1429c`，並核對該版本
所有被記錄的數值 helper hashes。這個 h160／Q12 小型兩有限體有 11 個 retained
DOFs、22 個 stage DOFs；45 ms、2 µs 的完整兩族頻譜各包含 32769 個矩陣。
lag-zero inverse-composition 殘差為 7.32e-12；獨立原 compliance 路徑抽查與
兩 mode／完整 stage 組合差異均低於 1e-12。

接著使用原始 float64 seed、完整 body healed active-set 初始化，讓診斷載入
在 128 步內增加 2 mm，以實際經過滑移與弱化。最大累積滑移為 1.03382 mm，
原始 RK 方程殘差為 9.42e-8 MPa，低於未放寬的 1e-7 MPa 門檻。
另一套重新組裝的 original-Z RK-CQ 路徑，最大剪力差為 3.72e-8 MPa。
該短程 reference circle 明確使用 FFT=2048；完整 kernel 為 65536。
此診斷不是 event-16 的載入，也不是細網格 FEM 比較。

初始化保留實測的 `Kr*u-F`，包含其有限積分殘差；其當前項與嚴格過去項
均進入 RK 方程，沒有將小殘差歸零。實際 fine h10／contact2／end0.015625、Q12
初始化也已通過：265 個接觸點的塑性／累積滑移完全未變，原始 retained
平衡殘差為 7.55e-9 MPa。這仍只驗證初態。

## 執行與資源

現在可直接從隨套件附帶的 case TOML 建立，無須任何舊 BDF2 kernel：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FDBEM_CORES=4 \
  python -B bem/tools/prepare_raw_radau_resolvent.py \
  bem/src/finiteblock_bem/resources/cases/pmma_event16.toml \
  --dt 2e-6 --duration .045 --workers 2 --primitive-partition \
  --weight-layout fortran --max-workspace-mb 10240 --max-memory-mb 18432 \
  --output /path/to/fresh-radau-kernel
```

加上 `--preflight-only` 只做資源檢查。此 fine 設定保守預估陣列峰值
18051.5 MiB、所需磁碟 417.88 GiB；1 µs 需要約 835.75 GiB，且需
`--fft-rows 2` 控制 IFFT workspace。兩個建構不能同時在本機執行。
四核心為總預算；raw 建構只開兩 workers，以符合 24 GiB 主記憶體限制。
macOS 此環境沒有硬性 CPU affinity。

新的有界寫入緩衝會分開報告 committed 與 pending matrices，並要求全部落盤才可
完成。兩族完整粗網格頻譜經新 writer 複製後均逐位元相同；這不是速度 benchmark。
coarse kernel 約耗時 1037 s，期間與舊兩 worker 建構重疊，不能與 FEM 作速度比。
後續仍需完整 fine event、時間／接觸／bulk 空間收斂、原始因果 RK audit、
initialized AD 與同精度四核心總耗時比較。
