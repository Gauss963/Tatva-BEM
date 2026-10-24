# 固定幾何近遠場積分

2026-10-24。這是降低每次 Laplace BEM 組裝成本的獨立實驗。
細網格的積分點數約減半，抽樣算子及短程非線性／AD 核對已通過。
完整 45 ms 的新建核驗證正在進行；尚未證明與 FEM 等精度的速度或穩定性優勢。
數字、來源與 SHA-256 收錄在 [結果紀錄](mixed_quadrature_results.json)。

## 規則與保留的物理模型

對每一組面板，若為同面板、拓撲鄰接，或最短線段距離不大於
`2 * max(length_i, length_j)`，維持原積分階數；其他面板採較低階數。
分類只在幾何初始化時計算一次，不隨複數頻率改變。
如此可以減少 Bessel 函式計算，又避免頻率切換積分規則引入新的不連續。

沿用原 P1/P0 對稱 Galerkin 的 Green tensor、source traction、弱 hypersingular 形式、
singular subtraction 與面板加總。完整 CQ 頻譜、每項非零歷史、摩擦律、
penalties 及載入停止規則不變。這沒有實作 FMM 或 H-matrix，
也沒有沿用相關論文的複雜度或誤差定理。
近遠場降低積分成本是本專案的實作選擇；後續空間低秩求解仍依
[文獻紀錄](literature.md) 另行驗證。

原數值套件及預設 backend 保持原樣。實驗使用私有 operator factory，
頻譜與 prepared kernel 都記錄近／遠階數、幾何分類、程式指紋，並在完成前重查。
兩個階數相同，以及所有面板皆歸為近場的極限，V/H/D/M 與原組裝逐位元相同。

## 細網格 2 us 的實際 contour 檢查

幾何為 h=20、contact=5、end=0.5、corner=0.5 mm，含原 seed 的局部加密，
保留 275 個 DOFs。contour 對應完整 45 ms、22500 steps、FFT size 65536。
九個角度為 `0, .025, .05, .1, .2, .5, 1, 2, pi`。
下表是各點最大相對 Frobenius 差，並非連續頻率或完整軌跡的誤差上界。
Z 先轉為 traction rows；L 使用線權重平方根的 work coordinates。

|規則|Z 相對原 uniform Q12|work L 相對原 uniform Q12|
|---|---:|---:|
|近 Q12／遠 Q6|2.59e-9|9.73e-10|
|近 Q12／遠 Q8|2.14e-12|2.52e-13|

原 uniform Q12 相對 Q18 的最大差為 Z 8.48e-7、work L 6.46e-7。
近 Q12／遠 Q8 的額外差遠低於這組抽樣的原積分誤差，因此優先保留它作細網格候選。

兩個 block 的 uniform Q12 共需 9,254,016 個積分點；近 Q12／遠 Q8 共需
4,336,736 點，減少約 53.1%。近 Q12／遠 Q6 則減少約 71.7%。
這是確定的積分工作量縮減，不是整體 solver 的加速倍數。
上述量測與原四-worker fine2 建核重疊，wall time 不作受控效能比較。

另在 h80 與粗 h40 幾何抽樣，證實不能只憑一個幾何選遠場階數：
h80 的遠 Q6 相對 Q12 不通過 1e-6 算子門檻；遠 Q8 通過。
粗 h40 的近 Q8／遠 Q6 相對原 Q8 則約 3.53e-9（Z）、2.32e-9（work L）。
但粗 h40 的 uniform Q8 相對 Q12 本身約 9.85e-6，**沒有**通過 1e-6 門檻。
完整粗網格測試因此只驗證加速前後等價，不把它當作空間／積分收斂證據。

三組幾何全部保存的算子，另檢查 work-conjugate `Z(s)/s` 的 Hermitian 部分。
目前這些點未見相對 1e-8 以下的負方向；這仍不是所有頻率的被動性定理，
也不是非單調 slip-weakening 接觸的穩定性證明。

## 短程接觸與 JAX 核對

h80、近 Q12／遠 Q8、4 us／64 步、2 mm actuator 增量的診斷，
直接以新 BEM 核建立 native resolvent，再獨立推進非線性方程。
這是加速滑動診斷，不是原載入的 45 ms 事件。

相對獨立 dense Z 解，壓力最大差 4.09e-9 MPa、剪力 3.57e-9 MPa、
累積滑移 2.72e-11 mm；全部七個欄位通過原 1e-5 絕對差 guard。
滑動、開閉、weakening clip 及 clip 等值分支變更皆為零。
解代回原完整 Z 歷史，加上初始靜態誤差的 residual 界為 4.43e-9 MPa，
小於原 1e-7 MPa 平衡容差。

五參數條件式 adjoint 最大相對差 6.35e-9；actuator cotangent 最大絕對差
4.67e-11，兩種初始狀態 cotangent 差均小於 6e-10，通過原容差。
仍不含 E／rho／幾何、static lift 與 hold 事件時間的導數。

另修正 `adjoint_resolvent.py`：prepared candidate 與原 Z 比較時，
梯度使用 candidate 前向步驟實際採用的瞬時 impedance，並核對檔案指紋，
避免把原 Z 的瞬時矩陣混入新算子的導數。
比較入口也會記錄全部七個前向欄位差；梯度吻合不能掩蓋前向 guard 失敗。
standalone AD 沒有獨立軌跡時，該欄位標為 null，不把自己與自己比較當成驗證。

## 執行

從 repository 根目錄執行；每個輸出指定新的空資料夾。
SOURCE_RUN 必須是與目前套件／case 相符的已完成 Galerkin run。

```sh
FDBEM_CORES=4 bem/run.sh mixed-quadrature-audit SOURCE_RUN \
  --near-order 12 --far-orders 6 8 --reference-order 18 \
  --dt 2e-6 --duration .045 --workers 4 --output NEW_AUDIT

# near order 沿用 SOURCE_RUN 的 quadrature，不可把 Q8 來源誤當成近 Q12。
FDBEM_CORES=4 bem/run.sh mixed-resolvent-kernel SOURCE_RUN \
  --far-order 8 --separation 2 --dt 2e-6 --duration .045 \
  --correction-duration .020 --workers 4 --max-memory-mb 8192 \
  --max-spectrum-mb 16384 --output NEW_KERNEL

bem/run.sh resolvent-simulate NEW_KERNEL --fft-cache-mb 128 \
  --max-memory-mb 8192 --output NEW_RUN

# 獨立原 Z 須具相同網格／時間步與已驗證的參考梯度。
bem/run.sh resolvent-adjoint ORIGINAL_Z_RUN NEW_RUN --fft-cache-mb 128 \
  --max-memory-mb 8192
PYTHONPATH=bem/src python -B bem/tools/audit_replay_residual.py ORIGINAL_Z_RUN NEW_RUN
```

若原 Z 沒有 `adjoint.json`，可用 `--reference-gradient` 明確指定已驗證的同一 run 梯度。
只有五參數的參考無法驗證 actuator／初始狀態 cotangent，須另列驗證範圍。
核對解、原方程、分支與 AD 以後，才進行無其他重工作的四-worker 計時。
