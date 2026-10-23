# 對稱 Galerkin 動態 BEM

2026-10-23。本次把文獻中的能量一致空間離散接回既有 BDF2-CQ、
兩有限 block、LSW、JAX Newton 與完整歷史 adjoint。
**完整 FEM event-16 的空間／時間收斂與同精度速度優勢仍待驗證。**
下列線性與短程檢查不代表完成這個驗收。

## 方法與來源

[Aimi、Di Credico、Gimperlein (2024)](https://arxiv.org/html/2405.08566v1)
以 Poincaré–Steklov 算子建立動態摩擦接觸的 energetic Galerkin 形式。
本實作採用其對稱 bulk 結構，時間積分仍使用既有 BDF2-CQ，
接觸保留原 Tatva penalty 與 previous-state slip weakening。
這不是該文 space-time Uzawa／Tresca 演算法的逐式實作，亦不沿用其誤差定理。

連續位移以 P1 nodal hats 表示，single-layer 的輔助密度使用 P0：

```text
V = <P0, single layer P0>
H = <P0, (1/2 I + double layer) P1>
S = H.T solve(V, H) + D
```

`D = -T_x double_layer_y` 使用
[Bao、Xu、Yin 的二維 regularization，式 (3.6)](https://arxiv.org/html/1608.08759v1)，
以 `omega = i*s` 延拓至 Laplace 域；它等於 Aimi 文獻中的 `-W`。
複頻率使用雙線性轉置 `.T`。程式不裁掉負特徵值，也不事後對稱化矩陣。

`S` 輸出與 nodal displacement 共軛的 **nodal force**，與舊 midpoint
traction 分開處理。接觸採正的 P1 lumped line weights，包含半個端點權重，
總長 480 mm；下端與剛性 platen 共用同一位移自由度。
兩體的 nodal forces 在頻率域精確縮減為 contact jumps 與 platen，再建立完整 CQ。

奇異積分使用 self／adjacent Duffy 轉換與解析 log moments。
`galerkin.py` 保留逐對參考組裝器；`galerkin_fast.py` 共用唯讀幾何、
批次組裝並重用 Bessel 值，積分公式與階數一致。
資源規劃包含共享幾何與每個 worker 的有界暫存，沒有截短時間歷史。

## 獨立驗證

- 弱式 hypersingular form 對照獨立的 traction 有限差分。
- 非正交多邊形的平移／轉動模態、仿射應變與積分 nodal forces。
- P 波與 S 波製造解的空間加密。
- 頻率縮減前後的功率、支承與互易性。
- 有限桿波到達、固定端反射與同一物理 probe 的加密。
- nodal 靜態 lift、端點、NumPy／JAX 軌跡與輸出積分權重。
- 全歷史 reverse-CQ、scan AD 與中央差分梯度核對。

已取樣的 `s=200+2*pi*f*i`、`f=1,5,10 kHz`，幾組均勻與角落加密網格
的 velocity impedance Hermitian part 均為正。
接觸 5 mm、角落 2 mm、5 kHz 的正規化最小值約 `+1.53e-5`，
同幾何舊 collocation 約 `-7.00e-4`。
有限取樣仍不能證明整個非單調摩擦問題穩定。

有限桿 loading-face 相對 L2 誤差由 0.875% 降到 0.360%，
同一位置 `x=0.5` 的 probe 誤差由 1.479% 降到 0.570%。
空間與時間同時加密，此處不單獨宣稱兩者的收斂階數。

## 成本與 FEM 差異

同一 102+112 面板模型、八個相同頻率、4 workers、BLAS=1，交換執行順序：

|順序|逐對參考組裝|批次組裝，含幾何準備|
|---|---:|---:|
|reference → prepared|76.32 s|9.97 s|
|prepared → reference|77.71 s|9.92 s|

約 7.7 倍是相同離散算子的組裝加速，尚非 FEM 同精度的總速度優勢。
數值、執行環境與程式指紋見 [galerkin_results.json](galerkin_results.json)。

靜態 lift 的中段 traction 已接近 FEM，角落仍有明顯差異。
接觸 5 mm、角落 2 mm 的 frozen-plastic 屈服預估約 24.25 ms、位置 y=0，
FEM 首次保存滑移約 25.73 ms。靜態預估不能當成動態成核時間。
後續要獨立解析端點、process zone 與時間步，不能只把曲線調到吻合。

以原版本 FEM 原始碼在本 checkout 做加密的靜態初始化，保留同一份
分段線性塑性滑移 seed 與 actuator，得到：

|FEM 網格|下端 p|上端 p|接觸加權平均 tau|platen|
|---|---:|---:|---:|---:|
|2 mm，原紀錄|38.07 MPa|344.62 MPa|8.82654 MPa|9.958879 mm|
|1 mm，本機|45.08 MPa|479.08 MPa|8.82068 MPa|9.959424 mm|
|0.5 mm，本機|52.55 MPa|653.84 MPa|8.81805 MPa|9.959684 mm|

端點峰值仍隨 FEM 網格改變。加密 Galerkin 的平均 tau 約 8.81680 MPa、
platen 9.959814 mm，積分反應比端點峰值更接近；這仍不是動態破裂等價的證明。
FEM 對照只初始化並推進極短時間，沒有完成加密網格的 45 ms 事件。
工具沿用舊名稱 `restart_projected_to_coarse_mesh`，此處實際是加密網格。

![解析波與端點網格依賴](figures/galerkin_validation.png)

同一 event-16 CQ contour 的九個取樣點，積分階數 8／12 相對階數 16 的
traction-scaled 矩陣差最大約 `3.45e-6`／`2.86e-7`。
階數 8 的完整動態對照仍須核對更高階積分與更嚴格的頻率插值。
這是取樣檢查，不是全頻誤差界。

目前本機 58 項 BEM tests 與 Ruff 通過。短程 Galerkin 滑動診斷的
mu_s 梯度與中央差分相對差約 `2.21e-9`，兩側差分沒有改變保存的接觸分支；
該診斷使用加速載入，不是 event-16 的物理參數識別。

## 完整 45 ms：已完成一組四 worker 對照

102+112 面板、35 接觸節點、dt=8 us、Gauss 8、adaptive tolerance 1e-8，
使用原載入及停止政策，完成 5625 步。此組是較粗空間解析度的結果。

|量|Galerkin BEM|原 FEM h=2 mm|
|---|---:|---:|
|首次保存滑移 >1e-7 mm|24.216 ms|25.733 ms|
|停止載入|29.024 ms|29.788 ms|
|末最大累積滑移|0.400258 mm|0.644123 mm|

最大滑移偏低 37.86%，共同 P1 插值網格的末滑移分布相對 L2 差為 39.25%；
壓力／剪力分布差約 10.03%／10.39%。**尚未達成 FEM 等價驗收。**
末界面的 [滑移、壓力與剪力分布](figures/galerkin_profiles.png) 另存供比較。
這也顯示修正先前的負 bulk 方向，並未單獨消除這組載入下的主要滑移差異。

最大接觸平衡殘差約 1.21e-10 MPa，保存狀態皆有限，累積滑移非遞減，
按當步實際摩擦係數計算的最大強度比為 1。塑性摩擦功為 1823.17 MPa mm²。
另以獨立的整段 zero-padded FFT 重建 bulk 力，和直接歷史加總的最大差約
1.50e-11；BDF2 速度的累積 bulk 擾動功最小值為 0，末值 56.27 MPa mm²。
這是固定靜態 lift 周圍的擾動功檢查，沒有把它當作整個非線性系統的能量證明。

冷權重建立 2100.24 s，實際推進約 2.75 s；重用相同權重後，全程約 5.7 s。
冷建立期間有少量報告整理與打包，列為探索性 wall time；
它與未知核心數的 FEM 歷史耗時不構成同精度四核心速度比。
3336 次 transfer 評估取代 8193 個 dense 頻率點；這個新方法的完整 dense／
更嚴格 tolerance 與時間加密軌跡仍需核對。

同一完整軌跡的條件式 adjoint 已檢查 mu_s、mu_k、Dc，
中央差分相對差分別約 1.8e-7、1.5e-5、3.3e-7，兩側差分均未改變保存的接觸分支。
微分範圍維持固定 bulk、lift 與記錄的載入；並未包含停止事件時間。

![完整時間歷程對照](figures/galerkin_event16.png)

## 端點、初始場與下一組網格

使用同一 archived FEM solver 與同一 piecewise-linear 塑性滑移 seed，
frozen-plastic 靜態屈服預估並不隨網格單調收斂：

|FEM 網格|預估時間|預估位置|
|---|---:|---:|
|2 mm|25.803 ms|y=0 mm|
|1 mm|28.123 ms|y=22 mm|
|0.5 mm|23.227 ms|y=2 mm|

這些是靜態預估，不是加密 FEM 的完整動態事件。原 2 mm 曲線仍保留作比較基準；
目前資料不能把它或任一加密結果當成已收斂真值。
seed 在 2 mm 節點存在斜率轉折，漸變網格會重新近似這些特徵。
新增 `--restart-knot-window` 保留指定端點區間的原節點，
`--restart-knot-subdivisions` 可再細分該區間。
細分 patch 與中段採一致的漸變銜接，避免把兩套網格疊加成過短的面板。

另固定 P1 trace、只把角落附近 P0 density 加密 2／4 倍：
原 hypersingular 弱式限制回原 trace 後的相對差低於 4.5e-12，
壓力／剪力初始分布誤差由約 12.1%／15.0% 降到 9.85%／11.75%，
但屈服預估由 24.25 ms 移到 22.82 ms，未改善原 FEM 時間對照。
因此目前保留為誤差分離實驗，未升為動態預設值。

![靜態屈服的網格與 seed 特徵敏感度](figures/galerkin_static_onset.png)

下一組採 contact 5 mm、corner 0.5 mm、端點 8 mm 區間的 0.5 mm patch，
250+254 面板、137 接觸節點、Gauss 6。記憶體陣列預估約 18333 MiB；
靜態 lift 與 1／5／10 kHz 的取樣被動性預檢通過，現以 4 workers 建立完整權重。
其 Gauss 6／16 靜態屈服預估差約 0.08 us；完整動態精度尚未驗證。
執行設定保存在 `outputs/galerkin-patch5-dt8-q6/run_manifest.json`，
在產生完整 summary 前不列為已完成的事件。

## 重跑

從 repository 根目錄執行，所有輸出留在 `bem/outputs`：

```sh
FDBEM_CORES=4 bem/run.sh galerkin-wave
FDBEM_CORES=4 bem/run.sh galerkin-benchmark --graded
FDBEM_CORES=4 bem/run.sh galerkin-audit --refine-corners --static-only
FDBEM_CORES=4 bem/run.sh galerkin --h 80 --steps 32 --dt 8e-6 \
  --diagnostic-actuator-increment 2 --sampling dense --name galerkin-sliding-32
FDBEM_CORES=4 bem/run.sh adjoint bem/outputs/galerkin-sliding-32 \
  --finite-difference --fd-parameters mu_s
```

`galerkin` 未給 diagnostic 選項時採原 20 mm/s 載入與 coverage hold。
完整 45 ms 的預設步長仍需驗證；可明確指定 `--duration .045`、接觸／角落尺度、
積分階數和資源上限，並保留 manifest。失敗或資源不足會回報，不能算完成。

AD 重用固定材料／幾何的 CQ 權重，高效計算摩擦參數的條件式梯度。
SciPy bulk、靜態 lift 和停止載入事件時間的參數導數仍未串接；
這些限制對新 backend 同樣成立。原 `/Volumes/Gauss-T7/tatva` 保持只讀。

FEM 靜態 study 的小型設定已包入 `resources/data/fem_static_onset_input.json`，
原 solver revision 保存在 repository 的 `codex/fem-event16-reference` 分支。
需要 Tatva/FEM 依賴；一般完整 clone 已有此 Git object，若是單分支 clone 可先 fetch：

```sh
git fetch origin codex/fem-event16-reference
python -B bem/tools/fem_static_onset.py
FDBEM_CORES=1 bem/run.sh galerkin-density
FDBEM_CORES=1 bem/run.sh galerkin-audit --refine-corners --static-only \
  --restart-window 8 --restart-subdivisions 4 --end-values .5
```

此 FEM study 不需要外部原 HDF5；以包入設定重跑 2 mm 的預估時間與先前結果相同。
完整動態較細案例的目前命令為：

```sh
FDBEM_CORES=4 bem/run.sh galerkin --h 20 --contact-h 5 \
  --contact-end-h .5 --corner-h .5 --restart-knot-window 8 \
  --restart-knot-subdivisions 4 --quadrature 6 --dt 8e-6 --duration .045 \
  --sampling adaptive --sampling-rtol 1e-8 --max-memory-mb 20480 \
  --name galerkin-patch5-dt8-q6
```
