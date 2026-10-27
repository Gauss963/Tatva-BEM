# 在消去之前插值原始邊界算子

目標是降低下一輪 bulk／time refinement 的冷建核成本，而不刪除有限體反射歷史。
目前 fine2/Q12 resolvent 的 adaptive sampler 記錄 9665 次 transfer evaluation，
相對 32769 個半頻譜點；原建核計時含競爭，不能當成公平 speed benchmark。

原路徑插值已消去的 resolvent，包含有限體模態極點；原始 V/H/D 的頻率依賴
來自 Green kernels，在右半平面沒有相同的全域消去分母。新的實作推論是：
先以較少的原始算子取樣建立局部頻率插值，再在**每一個**原 CQ grid point
執行原線性消去。這可能節省昂貴的積分組裝，是否省時仍須實際量測。

`raw_layer_sampling.py` 使用 piecewise Chebyshev 插值，各 body 的 V、H、D
分別作相對 norm、獨立點與係數尾端檢查，避免不同單位或大矩陣族掩蓋小矩陣族。
另對獨立點的消去後 work output 作誤差核對，防止矩陣求解放大原始誤差。
未解析的區段退回逐頻率原始組裝。所有 FFT grid points 都輸出，沒有改變 dt、
輪廓、time lag、接觸律或空間秩。`max_workspace_mb` 限制插值陣列估計，
不包含原 provider 的 geometry、solver workspace 或整個程序的 RSS。

`ConstrainedSchurTransfer` 新增 supplied-operator 入口，重用原支承與 jump-map
消去，不需要再配置一份積分 geometry。複數頻率仍使用 bilinear transpose；
沒有對稱化、裁剪 eigenvalues 或人工 regularization。

這受到 [2026 elastodynamic 3D-ACA 研究](https://arxiv.org/html/2607.15292v1)
區分 raw kernel 與消去後 transfer 成本的啟發，但**不是**其 3D-ACA 演算法的重現。
目前只做 frequency Chebyshev，沒有空間低秩或論文中的 fast multipole method。

新增控制包括：

- 有共振的消去算子：逐點核對完整 FFT 半頻譜及所有需要的 CQ lags。
- 大原始矩陣族中的弱方向：必須由消去後獨立檢查發現放大誤差。
- 真實有限雙 block：每一頻率重新組裝 BEM 作對照，不重用其矩陣作插值資料。
- 固定 mixed mass、記憶體 preflight 與 supplied-operator reconstruction 的等價性。

這些測試交由 CI 執行；大型 cold build、完整非線性事件及 AD 尚未由此路徑執行。
目前 full event 的壓力／剪力差仍存在。即使本工具節省積分次數，也不構成同精度
FEM 速度優勢；後續須計入原始組裝、所有逐頻率線性消去、IFFT、IO 及線上求解。

下一個驗證會用已保存的原完整頻譜作**建核後**逐頻率核對，並保留 fresh frequency
檢查。原頻譜不得用於選樣或產生新插值資料，驗證成本也須與新建核成本分開記錄。

來源 `a0b0c36` 已通過 Python 3.11／3.12 的
[完整 CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40365637516)，包括真實有限雙 block
的每頻率 fresh-BEM 與完整 CQ lags 對照。三個小矩陣／guard 控制亦於本機通過，
0.46 s 是這三個測試的時間，不是大型效能數據。

`prepare_raw_layer_resolvent.py` 新增完整產物入口：依來源 metadata 建立新頻譜、
保留全部時間權重及 causal elastic startup，並檢查 lag-zero inverse composition。
來源的 spectrum／weights 不作為新計算資料；來源只供物理 case 和離散預設值。
獨立測試會以無效的來源 numerical files 確認此點，再逐頻率重新組裝 BEM 核對產物。
這個新入口已通過 [Python 3.11／3.12 CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40435777232)，
包含重新組裝真實雙 block 每個頻率、所有權重及 causal startup 的對照；預設正式建核入口維持不變。

先執行 preflight，不啟動另一個同時競爭的 heavy job：

```sh
PYTHONPATH=bem/src python -B bem/tools/prepare_raw_layer_resolvent.py SOURCE_KERNEL \
  --dt 2e-6 --duration .045 --workers 4 --fft-rows 4 \
  --max-workspace-mb 3072 --max-memory-mb 8192 --preflight-only --output NEW_PREFLIGHT
```

fine2/Q12、2 us、45 ms 的實際 preflight 已完成：完整產物需約
53,314,127,752 bytes 磁碟，插值陣列估計 2348.99 MiB，整體陣列估計
6168.79 MiB。這仍只是 preflight，不是完成的核心或 RSS 實測。

`compare_raw_kernel_spectrum.py` 用既有完整 work resolvent 與已獨立稽核的原算子，
對每個新頻率檢查相對矩陣差及 `A_old * L_new - I`，後者避免弱方向誤差被
Frobenius norm 掩蓋。三個 NPY 檔在同一讀取 pass 核對完整 SHA-256，
驗證成本與新建核時間分開。弱方向注入誤差的測試已通過。

`compare_refined_primal.py` 再核對相同 bulk/time/contact grid 的所有 saved fields，
包含 retained displacement、初始 trace／force 與完整 45 ms histories；沿用既有
壓力／剪力 1e-3 MPa、位移／slip 1e-6 mm 容差。只對末端看起來相近不算通過。
這些是新 solver 的等價性 gate，沒有把未達標的 BEM/FEM 差異改名成成功。

## 時間連續的權重儲存

`--weight-layout fortran` 在完成原 C-order IFFT、physical scaling 及 lag-zero
composition 後，將這次新建的權重轉成標準 Fortran-order NPY。
邏輯 shape 仍為 `(lag, row, column)`；每個矩陣元素的完整時間序列連續儲存，
供重複的歷史 FFT 使用。所有 lag、matrix entry 與 float64 bits 都保留。
預設 `c` 不變，原來源 kernel 始終只讀。

轉換按明確 workspace 分塊，逐塊重新讀取並核對 uint64 bits，全部通過後才取代
**本次新產物**的 C-order 檔。preflight 計入兩份權重同時存在的峰值磁碟需求，
總計時亦包括轉換、驗證及輸出 hashing。這不是低秩或記憶截斷。

本機已核對 C／Fortran 與其轉置的全部非對稱 causal pairs 和 forward/reverse
inner product；signed zero 與 subnormal 的檔案 bitwise 控制也通過。
新 producer 的兩種 layout 都有 fresh-BEM 全頻譜、全權重及 startup 測試。
完整事件的 runtime、primal 及 gradient equivalence 仍須另測，不能由 layout 推定加速。

下一個物理對照將直接使用原 FEM h2 的 241 個接觸節點 `y=0,2,...,480 mm`，
以相同 nodal contact rule、原 float64 restart、2 us bulk step 與 N64 局部子步推進。
bulk 接觸 trace 也增加到這些節點；僅增加接觸狀態積分點不會增加 bulk 空間模式。
外邊界與時間收斂仍要分別檢驗，此設定不是宣稱 FEM 或 BEM 已收斂。
