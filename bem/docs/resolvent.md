# 彈性 resolvent 與完整摩擦歷史

2026-10-24。這是實驗性求解路徑；目前與 FEM 的最大滑移仍差約 25%。
尚未證明同等精度下的速度優勢或 slip-weakening 全域穩定性。
[數值紀錄與來源 SHA-256](resolvent_results.json) 保留已通過及未通過的條件。

## 等價變數與因果性

令 q 為靜態 lift 周圍的接觸 jumps／platen 擾動，Z 為有限體動態 impedance，
C 為 kn／kt 乘以接觸線權重的彈性接觸矩陣，K 為 platen 彈簧：

```text
A(s) = Z(s) + C + K, L(s) = inverse(A(s))
chi(q,state) = Fc(q,state) - Fc0 - C*q
q_el = L * f_ramp
q = q_el + L * (f_hold - chi).
```

`f_ramp` 是持續原速率的載入；`f_hold` 是原 coverage hold 相對它的偏差。
`q_el` 包含啟動、波傳與有限體反射，並非逐步重算的靜態平衡。
先將 L 的最後一欄反變換成因果 CQ 係數，再以零補齊的線性卷積作用於載入。
直接把有限載入的 FFT 乘上頻譜會有未來載入的週期回繞；新增延遲載入測試防止此問題。

閉合／黏著、狀態未變且尚未 hold 時，修正源為零。第一次非零後保留每一項歷史；
給定的修正 horizon 不足時拒絕繼續，不裁掉舊資料。
每步以 `past_virtual = f_ramp - A0*(q_el+h)` 接回原 JAX Newton，
求解相同的 previous-state 摩擦平衡根，維持原 penalties、摩擦係數及停止載入規則。

## 已完成的解核對

|檢查|結果與限制|
|---|---|
|非交換矩陣 inverse/composition、延遲載入、零源 rebase|測試通過；意外非零源與超出修正 horizon 均拒絕繼續。|
|直接建立 L：h=80 mm、8 us、32 步|滑移最大差 3.69e-11 mm，壓力／剪力差小於 4e-9 MPa。這是強載入診斷。|
|獨立新時間網格：來源 8 us，直接 L 改為 4 us／64 步|與另外建立的完整 Z 解相比，滑移最大差 2.62e-11 mm，壓力最大差 4.06e-9 MPa，載入相同，所有精確接觸分支相同。尚非原 45 ms event-16。|
|粗網格直接建立 L：4 us／45 ms、11250 步|滑移最大差 1.96e-8 mm，壓力／剪力最大差 7.27e-7／4.47e-6 MPa，載入與 28.896 ms hold 相同。嚴格欄位 guard 通過。|
|完整細網格 4 us／45 ms，以既有 Z 頻譜建立 L|滑移最大差 1.48e-7 mm，剪力最大差 3.38e-4 MPa；同在 29.216 ms hold。原本「所有物理欄位絕對差 ≤1e-5」的 guard **失敗，退出碼 1**，沒有放寬門檻。|
|上項解代回原始完整 Z 歷史|獨立零補齊 FFT 與直接求和核對後，原始平衡 residual 加初始靜態誤差界為 2.73e-8 MPa，小於原 1e-7 MPa tolerance；滑動、開閉及 weakening clip 分支變更皆為 0。這支持原平衡容差內的等價，並非總能量或 FEM 一致性證明。|

完整細網格結果仍約 0.48145 mm，FEM 約 0.64412 mm。
換變數本身不應改變這個差距；它讓後續時間加密所需的矩陣歷史較短。
完整粗網格的直接 L 核已以四個 frequency workers 建立，5326 次實際 transfer
evaluation 對應 16385 個 contour 點；這是自適應取樣的數值檢查，非 certified bound。
解代回原完整 Z 方程，residual 加初始誤差界為 1.16e-8 MPa；
滑動、開閉、weakening clip 與 clip 等值分支變更皆為 0。
最大滑移仍為約 0.442915 mm，與原粗網格結果相同，不代表與 FEM 已吻合。

對下一個細網格 2 us／45 ms 的九個實際 contour 角度，另比較 Gauss 6／12／18。
相對 Gauss 18 的 traction-scaled 矩陣差最大分別為 1.792e-4 與 8.482e-7。
因此後續 2 us 計算採 Gauss 12；不能把不同積分階數的結果當成只改 dt 的收斂序列。
這是有限點檢查，不是全時間歷程的積分誤差上界。

## 保留 JAX 自動微分

`tools/adjoint_resolvent.py` 逐步對局部 return map 作 VJP，再以隱式根的轉置
Jacobian 傳播，沒有反向儲存全部 Newton 迭代。啟動期即使基準修正源為零，
kn／kt 導數仍不為零；程式另以頻率分塊的向量轉置作用取得這段 cotangent。
只返回尚未啟動的前綴，避免將有週期回繞的後段值誤當因果 adjoint。

- 非對稱矩陣歷史：physical／work coordinates 均與直接因果轉置和一致。
- 獨立耦合 toy：五參數、每步 actuator 與兩個初始狀態 cotangent，
  同時核對原完整 CQ reverse 與 JAX scan AD；保留非零 kn／kt 前綴貢獻。
- 32 步實際幾何：參數梯度最大相對差 3.74e-9；初始狀態差小於 5.3e-10。
- 新 4 us／64 步：參數梯度最大相對差 2.45e-9；actuator 梯度最大絕對差
  3.16e-12、初始狀態差小於 6.3e-10。與獨立原 Z adjoint 比較，所有分支相同。
- 完整粗網格 4 us／45 ms：五參數梯度對原 Z reverse 的最大相對差為 1.46e-6，
  精確分支變更皆為 0；原 Z reverse 已另與完整 forward JVP 核對。
  本項只比較五參數，尚未核對該完整窗的 actuator／初始狀態 cotangent。

這些是固定 bulk、lift、已記錄載入與 active branches 的離散導數。
沒有 E／rho／幾何或 coverage hold 事件時間導數；新路徑細網格 45 ms 的 AD 尚待驗證。
參考梯度載入時核對來源、目標值、參數次序及 JSON／NPZ 一致性。
prepared candidate 的比較現使用該前向 solver 的瞬時 impedance，核對檔案指紋，
並另列全部七個前向欄位的原容差 guard；standalone 不填入獨立比較通過標記。
此檢查亦用於[固定近遠場積分](mixed_quadrature.md)的實驗。
2026-10-24 補上原粗 Q8 直接 L 的全部 cotangent 比較：參數與初始狀態通過，
但 actuator index 1724 的絕對差 2.56e-10，略超過原 `rtol=1e-5, atol=1e-12`。
近 Q8／遠 Q6 native 組合亦在同一 index 失敗，絕對差 8.40e-10。
兩個控制都保留退出碼 1；原 Z 的該點已另以 forward-CQ JVP 核對至約 1.6e-14 絕對差。
此項更新補足早期只核對五參數的驗證範圍，不把微小殘差視為所有載入導數已吻合。

另修正原 `run_adjoint.py` 的有限差分診斷：以接受的 jumps 和**前一步**狀態判斷
精確滑動／開閉，另外記錄 weakening clip 及 `c_old == Dc` 的等值分支。
細網格完整 4 us 的 mu_s 差分
`epsilon=1e-9` 仍差 3.86%，但正側已有一個 clip 分支改變；
舊的塑性增量閾值未發現它。此結果不能作為固定分支 AD 的失敗證據，
也尚未證明全部差異都來自這一次切換。獨立原方程 forward JVP 與 reverse VJP
五參數已在完整細網格相對吻合至約 1.2e-13，見 [Galerkin 紀錄](galerkin.md)。

## 重現入口與成本範圍

```sh
# 直接建立 L；全時間窗決定頻率 contour，修正窗只限制矩陣係數數量。
FDBEM_CORES=4 PYTHONPATH=bem/src python -B bem/tools/build_resolvent_history.py \
  SOURCE_RUN --correction-duration .022 --sampling adaptive \
  --sampling-rtol 1e-8 --sampling-atol 1e-14 --workers 4 \
  --max-memory-mb 4096 --output KERNEL_DIR

# 同網格對原 Z 軌跡作 replay 核對。
FDBEM_CORES=1 bem/run.sh resolvent-replay SOURCE_RUN \
  --prepared-kernel KERNEL_DIR --output REPLAY_DIR
PYTHONPATH=bem/src python -B bem/tools/audit_replay_residual.py SOURCE_RUN REPLAY_DIR
PYTHONPATH=bem/src python -B bem/tools/adjoint_resolvent.py SOURCE_RUN REPLAY_DIR

# kernel 可用 --dt 建立新時間網格；此入口直接模擬，無需該網格的原 Z 核。
PYTHONPATH=bem/src python -B bem/tools/simulate_resolvent.py KERNEL_DIR --output RUN_DIR
PYTHONPATH=bem/src python -B bem/tools/adjoint_resolvent.py RUN_DIR
```

執行前依 `bem/run.sh` 設 BLAS thread limits；本機四個 workers 並不等於 macOS 硬 affinity。
頻譜以 work coordinates `D*L*D` 儲存（D 為線權重平方根），時間係數轉回 physical L。
兩者不能混用。所有離線 L 點仍包含完整有限體 BEM 求解；沒有 rational fit。
磁碟頻譜只讀重用，核對 SHA-256；舊 Z 轉出的 L 另核對每一頻率的逆矩陣殘差。

目前冷建立仍昂貴，且部份診斷有工作重疊／分頁，未據此作受控速度比較。
32／64 步結果只建立等價性證據。正式比較仍須完成更細時間步長的 45 ms 解、
同精度 FEM 對照與無其他重工作競爭的四核心計時。
