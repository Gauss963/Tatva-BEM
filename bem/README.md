# finiteblock-bem

二維平面應變、有限區域的 **fully elastodynamic BEM** 研究實作，目標是與 Tatva 的雙 PMMA block 摩擦模型交叉驗證。

此版本已實作彈性波核、完整時間歷史、有限區域邊界、單向接觸、線性滑移弱化與剛性 platen／彈簧耦合。
**已完成原載入歷程的 45 ms 粗網格推進，尚未完成原解析度的收斂重現。**
目前驗證包括解析波傳／反射、雙 block 非線性滑動、完整時間範圍的原載入對照，以及 JAX 接觸參數梯度。
數值與執行條件見 [事件比較](docs/event16_comparison.md) 和 [AD 設計](docs/autodiff.md)。
新增 [接觸座標縮減](docs/condensation.md) 可降低完整歷史儲存，並提供端點與角落加密。
最新 [空間加密與完整歷史 adjoint 結果](docs/refinement.md) 說明改善與仍未收斂的部分。
新增 [2024–2026 文獻與實作決策](docs/literature.md)，以及可選的頻率插值與 bulk 被動性檢查。
新的 [P1/P0 對稱 Galerkin backend](docs/galerkin.md) 已接回完整歷史與 JAX，
提供一致的 nodal forces、端點接觸與批次組裝。較細網格已完成 45 ms，
[最新 2 us／Q12 完整結果](docs/fine2_event.md) 的最大滑移為0.5111 mm，
相對 FEM 的偏差約20.7%；前緣仍未吻合。上一組4 us使用Q6，這次不能作純時間收斂結論。
完整歷史 FFT 改為只快取可重用的區塊；同一較細軌跡及 adjoint 逐位相同，程序峰值 RSS 約減半。
Galerkin CLI 可用 `--spectrum-directory` 把完整半頻譜存到磁碟，降低建立權重時的 RAM 需求；
時間權重和全部歷史仍保留，原本的取樣及 IFFT 公式不變。
新增 [Radau 時間積分研究](docs/radau_experiment.md) 與
[彈性 resolvent 分解原型](docs/resolvent.md)，均保留獨立驗證與尚未通過的條件。
resolvent 現可直接建立新時間網格，並核對參數、載入及初始狀態的條件式 adjoint；
小型解與梯度驗證已通過，完整案例的誤差與成本仍需進一步比較。
另有 [徑向 traction／Horner 組裝實驗](docs/radial_assembly.md)，保留相同積分與物理模型；
目前只證實抽樣單頻率改善及短程解／梯度等價，尚未切換預設 backend。
新增 [逐點驗證的頻譜儲存壓縮](docs/polynomial_spectrum.md)：完整粗網格頻譜約縮減 4.27 倍，
並重新核對 45 ms 解與五參數 adjoint。
另有 [直接保存 CQ 插值與磁碟時間權重](docs/native_spectrum.md)，可避免首次配置完整稠密頻譜；
實際 BEM 的 512 步診斷已核對解與各種 cotangent；與近遠場積分組合的完整粗 45 ms
亦通過解與原方程核對，但有一筆載入 cotangent guard 尚未通過。
新增 [指定容量的 FFT 歷史快取](docs/bounded_history.md)：完整粗網格 45 ms 的解與各種 cotangent
在 0／8 MiB 設定下逐位元相同。較細網格 adjoint 亦與原快取規則逐位元相同，
但既有 L／原 Z 的單一載入梯度 guard 仍未通過，保留原容差與失敗紀錄。
新增 [FEM 四核心參考與獨立界面輸出](docs/fem_reference.md)：完整粗 FEM 事件的 9000 筆
shear 資料與最後狀態逐位元相同，bulk 輸出可大幅減少；原 h2 參考已排在現行 BEM 工作之後，尚未完成。
新增 [固定幾何近遠場積分](docs/mixed_quadrature.md)：細網格近 Q12／遠 Q8
減少約 53% 積分點，九個頻率點的額外算子差約 1e-12；短程解與條件式 AD 已核對。
完整粗 45 ms 的解／分支核對通過，載入梯度有一筆失敗；尚未切換預設 backend 或宣稱整體加速倍數。
另有 [避免反轉剛體模態的 Schur 消去](docs/schur_condensation.md)：
細網格實際 CQ 端點的消去矩陣條件數約由 320 萬降至 304，算子仍相差約 1e-12。
這改善低頻線性代數，尚未證明完整摩擦軌跡或整體效能優勢。
新增[局部接觸子步](docs/substepped_contact.md)，並在完整條件式adjoint傳遞前一步位移依賴；
小型原始Z-history／JAX scan作獨立核對，完整事件子步對照仍待完成。
另有[受限解析域的頻譜重用原型](docs/cauchy_reuse.md)，旨在減少相同Q下時間網格比較的重複建核；
尚未宣稱大型案例的誤差或加速結果。

程式與 Tatva、Bem2d.jl 獨立，不需要讀取或修改外接磁碟，也不需要上層 repository 的 Julia 環境。
可以把本資料夾獨立放進新的 GitHub repository。

## 快速執行

此電腦可直接使用現有 Python，所有輸出存到本資料夾的 `outputs/`：

```sh
./run.sh test -q
./run.sh benchmark --name bar-reference
./run.sh smoke --name smoke-reference
./run.sh smoke --dt 2e-6 --steps 256 --name smoke-refined
./run.sh event16 --name event16-short
./run.sh plot --name bar-reference
./run.sh plot --name smoke-reference
./run.sh plot --name event16-short
```

`run.sh` 優先使用本資料夾的 `.venv/bin/python`，否則使用此電腦現有的 conda Python。
可用 `FDBEM_PYTHON` 指定其他 Python。

新電腦安裝方式（Python 3.11+）：

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test,plot,jax]'
.venv/bin/python -m pytest
.venv/bin/finiteblock-bem benchmark
```

NumPy、SciPy、threadpoolctl 是核心依賴；JAX 為可選的 AD 後端，Matplotlib 供產圖。
套件包含預設 case 與小型重啟界面資料，wheel 不依賴外部 Tatva 路徑。

## 4 核心完整時間範圍對照

以下從 repository 根目錄執行，完整歷史 FFT 是預設值：

```sh
FDBEM_CORES=4 bem/run.sh event16 --h 40 --dt 8e-6 --duration .045 \
  --workers 4 --backend jax --max-memory-mb 4096 \
  --cache bem/outputs/cq-cache --name event16-4c-h40-dt8-fixed
FDBEM_CORES=4 bem/run.sh event16 --h 40 --dt 4e-6 --duration .045 \
  --workers 4 --backend jax --max-memory-mb 8192 \
  --cache bem/outputs/cq-cache --name event16-4c-h40-dt4
FDBEM_CORES=4 bem/run.sh ad-benchmark
FDBEM_CORES=4 bem/run.sh static-refinement
bem/run.sh compare bem/outputs/event16-4c-h40-dt8-fixed bem/outputs/event16-4c-h40-dt4
```

頻率核心以 4 worker 組裝，各 worker 的 BLAS 限為 1 thread；時間推進則以
`FDBEM_CORES` 設定 XLA thread，BLAS 在匯入前固定為 1 thread。macOS 不提供此處可用的硬 CPU affinity，
這是計算 worker 與 thread 設定，並非宣稱鎖住四個實體核心。JSON 會記錄設定、CPU 與 wall time。
CQ 快取含輸入和核心程式的 SHA256；只在同一離散問題重用，不會沿用不相符的權重。

## 已實作的方法

1. **Laplace 域二維 Navier 彈性波 Green tensor**，包含材料密度、P/S 波速與 source traction kernel。
2. **直接 interior BEM**：`(1/2 I + K(s)) u = V(s) t`，未知量是物理邊界位移與 traction。
3. **BDF2 convolution quadrature**：對 `V(s)^-1 H(s)` 取反變換，保留所有邊界位移歷史，計算當步 traction。
4. **兩個有限 block 的獨立邊界 operator**，耦合接觸；外邊界與支承造成的波反射由動態方程處理。
5. **單向 penalty contact、LSW、剛性 platen 與彈簧**，以解析或局部 AD 切線、Newton 和 L2 line search 求解。
6. **靜態 lift 初始化**：由原 FEM 的塑性滑移建立 BEM 自身的平衡場，再對零初始擾動解完整動態。
7. **線上 dyadic FFT 卷積**，逐步包含全部已接受位移；以直接歷史加總獨立核對。
8. **JAX 隱式根微分與短程可微分 rollout**，檢查摩擦參數、狀態與歷史依賴的梯度。

沒有把靜態 Bem2d.jl 矩陣加上人工質量，也沒有用局部 radiation damping 代替波傳。

詳見 [數學公式與離散方式](docs/formulation.md)、[驗證與限制](docs/validation.md)、[後續工作](docs/roadmap.md)。

## 三類執行要分清楚

### 解析波傳與反射

`benchmark`：對 ν=0 的有限長桿施加平滑壓力，右端固定、側面自由。
解析解由延遲入射／反射波組成，測試波到達、固定端反射及速度反向。
預設 h=0.1、dt=0.02，時間 3.6L/cp，會涵蓋第一個往返反射。

初步測得 loading-face 位移相對 L2 誤差約 1.21%、probe 約 1.38%；波到達前訊號約為峰值的 0.0064%。
測試也檢查 h、dt 同時加密後誤差降低。這是獨立解析驗證，不以 FEM 曲線當正確答案。

### 雙 block 強載入診斷

`smoke`：使用原雙 block 幾何、材料、摩擦參數，但從靜止開始，採加速載入以在短時間內觸發滑動。
**載入歷程與原事件不同**。預設 h=40 mm、dt=4 µs、128 steps，共 0.512 ms。
2 µs／256 steps 為同一物理載入的時間加密；未宣稱 h=40 mm 已解析約 13.9 mm 的原模型 process-zone 尺度。

目前兩個時間步都完成，最大累積滑移分別約 1.740、1.768 mm。
這證明求解器已能推進非線性接觸滑動，但尚不是原事件的驗證結果。

### 原事件重啟後的短程推進

`event16`：保留原塑性滑移、μs=0.8、μk=0.6、Dc=0.0125959 mm、16 MPa 壓力、25 MPa/mm 彈簧及 20 mm/s actuator 速度。
先建立 BEM 的靜態 lift，再以零擾動速度開始；時間原點是剪切階段起點，不包含原先的 2 ms 法向階段。
預設只推進 0.512 ms，尚未到原模型約 25.7 ms 的首次滑移，因此不應看到完整破裂。

目前 h=40、20、10 mm 的初始化都可保持原塑性滑移，滿足摩擦界限，不需要人工新增滑移。
這個 lift 沒有匯入 FEM 的整個體積位移／速度場，也不是重建第 15 次事件的波歷史。
它假設剪切起點可視為靜態平衡且擾動速度為零。

## 摩擦更新

預設 `weakening_evaluation="previous"`，與來源 Tatva 的單步 LSW 更新一致：

```text
p = kn * max(umx - usx, 0)
mu_applied = mus - (mus-muk) * min(cumulative_old / Dc, 1)
trial = kt * (umy - usy - plastic_old)
tau = clip(trial, -mu_applied*p, mu_applied*p)
plastic_new = plastic_old + (trial - tau) / kt    # 僅閉合接觸
cumulative_new = cumulative_old + abs(plastic_new - plastic_old)
```

Newton trial 不會改寫舊狀態；整步收斂後才 commit。
`trajectory.npz` 另存當步實際使用的 `friction_coefficient`。不可拿更新後 c 的強度，誤判當步依舊 c 計算的 τ 超限。
整體時間積分仍是 BDF2-CQ，與 Tatva explicit leapfrog 不同，需要做時間收斂比較。

可選 `weakening_evaluation="updated"` 使用步末弱化；高法向壓力時可能有多個局部返回解，程式會回報失敗，不會靜默切換分支。

## 資源限制與原模型解析度

這一版是可以檢查公式的 **dense reference implementation**：

- CQ 權重儲存約 O(Nt Nb²)。
- 預設 dyadic FFT 的時間工作量上界約 O(Nt log²(Nt) Nb²)。`--history direct` 保留 O(Nt² Nb²) 參考算法。
- 頻率核心可平行組裝與快取，但仍儲存 dense 空間／時間權重；尚未實作 oblivious CQ 或階層矩陣。
- `preflight` 在分配大型權重前檢查記憶體預估，預設 512 MiB 上限。它是工作陣列預估，不是作業系統 RSS 的硬限制。

先檢查成本：

```sh
./run.sh preflight
./run.sh preflight --h 2 --dt 1.2492928633545953e-8 --duration .045 --name original-resolution
```

第二個指令會拒絕不合理的 dense 配置並回傳 exit code 2，不會開始大量配置。
不能以縮短歷史或粗化到未解析的時間步，宣稱完成原事件。

## 產物與失敗處理

每個 run 都有獨立子資料夾：

```text
outputs/<name>/summary.json       方法、狀態、參數、版本及程式 SHA256
outputs/<name>/history.csv        接觸／載入／Newton 殘差
outputs/<name>/trajectory.npz     邊界場、接觸狀態與完整有效時間序列
outputs/<name>/verification.png   產圖後的驗證圖
```

時間步不收斂時，exit code 3，保存截至最後收斂步的有效前綴，並記錄失敗時間；不能視為完整 run。
輸入、資源或重啟轉移失敗時，exit code 2，說明存在 `summary.json`。
NPZ 是診斷資料，目前尚未提供通用跨 run resume API。重啟動態 CQ 必須保留歷史，不能只拿最後一個位移／滑移狀態繼續。

## 程式結構

```text
src/finiteblock_bem/
  material.py       plane-strain 常數與 P/S 波速
  mesh.py           有方向、標籤的 P0 邊界元素
  kernels.py        Laplace Green tensor、空間導數與 traction
  operators.py      奇異積分、直接 interior BEM 與 impedance
  cq.py             BDF2 symbol、反變換、完整歷史與記憶體檢查
  friction.py       接觸狀態、LSW、局部返回映射及切線
  constraints.py    支承、界面配對與 work-conjugate platen
  nonlinear.py      單步 Newton 與 line search
  condensation.py   頻率域消去線性邊界、接觸座標 impedance
  statics.py        不建立頻率幾何快取的串流靜態初始化
  adjoint.py        完整 reverse-CQ、JAX VJP、固定載入歷程的條件式梯度
  jax_backend.py    局部 AD 切線、JIT Newton、隱式解梯度、可微分短程 scan
  dynamics.py       靜態 lift、雙區域完整動態時間推進
  case.py           獨立 case adapter
  benchmarks.py     有解析解的波傳／反射
  io.py / cli.py    結果、來源指紋與命令列
  comparison.py     FEM 節點場投影到 P0 面板及誤差
  resources/        可隨套件攜帶的小型 case／參考資料
tests/              物理公式、歷史、接觸、耦合與波反射檢查
tools/              封存 FEM 執行、AD 計時、初始狀態加密、比較與科學圖
docs/               公式、驗證紀錄、後續工作
../.github/workflows/ Python 3.11/3.12 測試及 Ruff 檢查
```

核心是獨立推導的數值實作；來源模型資料與來源指紋保留在 resources 中。
尚未修改或整合進 `/Volumes/Gauss-T7/tatva`；所有工作都在本資料夾。
