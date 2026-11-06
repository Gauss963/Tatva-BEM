# 將已驗證的歷史接回原始 Radau 推進

**2026-11-04 最新結果：**原細網格第一組已藉此介面完成 45 ms；
[完整 FEM 接觸場比較](recovered_full_event.md)未通過原門檻。
第二組較細接觸積分的局部根、自己的歷史前綴與全部未來項均已通過驗證。
下方啟動快照與 coarse 控制仍保留各自範圍，不能視為完整細網格物理驗收。

2026-11-04：`tools/simulate_recovered_radau.py` 整合原 initializer cache、
完整歷史恢復、局部 AD 切線組裝與原平衡方程的求根備援。
[控制結果與來源](recovered_radau_resume_results.json)已保存。
完整細網格 45 ms／FEM 一致性、初始化 AD、時間／空間收斂與同精度速度／穩定性優勢仍待完成。

2026-11-04 11:47（台北）執行快照：數值來源 `26e08ed` 的
[CI 已通過](https://github.com/Gauss963/Tatva-BEM/actions/runs/41765449855)，
原細網格 `(64,128)` 案例已啟動，從驗證過的前綴接回完整 45 ms。
後續 `(128,256)` 子步比較會從相同初始狀態重新計算。
當時尚未產生完成軌跡；排程啟動不代表通過全事件驗收。
四核心總預算中，數值子程序配額為平均 3.8 核心，其餘保留給管理程序。

## 如何恢復

`radau_prefix_resume.py` 對原 `simulate_stages` 作三個 private AST 適配：
在迴圈前載入已驗證狀態、從下一步開始，以及選用恢復根作初始猜測。
迴圈內的物理方程、原 root 接受門檻、stage／endpoint 狀態提交、載入停止與完整 histories 不變。
工具會核對原函式的迴圈及 solve call；不符合預期結構時拒絕適配。

載入前檢查來源與 SHA，恢復 q、七個接觸欄位、actuator、殘差、完整已接受 inputs、
兩個 family 的未來 pending 及 delayed F1。邊界 forcing 需再次通過 `1e-9 MPa` 差異門檻。
原缺失 Newton counts 使用 -1；先前 prefix／初始化／建核成本另有繼承記錄，不能省略。
恢復根只是一個 guess，仍由原完整方程檢查後才提交。

後續原求解器若失敗，`recovered_radau_equilibrium.py` 可保留全部未弱化點與法向耦合，
以固定 seed 的 multistart 求根，再由**原完整 residual／custom_root**驗證候選。
這是 primal 備援；參數導數不能把縮減矩陣當作常數。
完整初始化 AD 必須傳遞所有 prefix 狀態、初始 residual 與完整轉置歷史。

## 通過的控制

- 四項原迴圈控制：含非零初始化 residual、弱化、載入 hold 前／後的恢復，
  以及 forcing 變動與原函式結構變動的拒絕。不中斷與恢復後全部 q／接觸欄位／actuator 逐位相同。
- 兩項備援控制：原細網格實際失敗方程能通過原 `1e-7 MPa`；old state 不被 trial 更新。
  一般非 identity jump maps 仍保留原失敗結果，不套用錯誤的消去。
- 真正雙 block 的完整 128 步 CLI 控制：原版、新工具直接執行、在第 65 步前載入 checkpoint
  後續跑，三者 `initial_lift.npz`、`trajectory.npz`、`stages.npz` 的每個 array member 都逐位相同。
  其原 kernel、float64 seed、h=160、dt=2 µs、`(8,16)` 子步與診斷載入皆保持一致。

第一次 CLI 控制在數值推進前被原 resource planner 拒絕：估算 1025.8 MiB 高於可用 1023 MiB。
保留該紀錄，控制總預算改為 2048 MiB；沒有改變物理或接受門檻，已完成的 affine 準備也直接重用。
所有本機控制平均 CPU 配額為 3.8；這些秒數不是隔離的全事件 benchmark。

## 接續後的初始化 AD

[獨立核對結果](recovered_radau_initialized_ad_results.json)：同一個完整 128 步雙 block 控制，
直接計算與 checkpoint 接續的五參數梯度、七個 JVP 及對應 VJP 內積全部逐位相同。
兩者均遍歷全部保存 stage，傳遞初始靜態場、初始 residual 與每筆完整歷史，
沒有把 checkpoint 前的參數依賴視為常數。
正反向最大絕對差 `9.2843e-15`，相對非零差 `1.0808e-12`，
維持 `rtol=1e-7, atol=1e-18`。
獨立原始因果 RK 方程核對也涵蓋全部 stage，最大 residual `1.6202e-8 MPa`。

範圍是固定 bulk、五個接觸參數、selected healed preload 與已記錄的載入歷程。
目前不包含 material／geometry 或 endogenous loading stop-time 導數，
也不是完整細網格 event-16 的梯度收斂證明。

## Recovery 配額選項

`--recovery-random-starts` 的[候選 patch](diagnostics/recovery_random_budget_candidate.patch)
已在隔離 namespace 驗證，[結果與來源](recovery_random_budget_candidate_results.json)另存。
預設維持 32，僅增加可選的候選起始猜測上限，不更動原接受方程或容差。
較細版本的捕捉失敗在原版／候選預設 32 下皆保持失敗（35 個候選）；
設定 512 後，第 75 個候選通過完整 residual，並與獨立 NumPy 接觸律核對。
這些 trial 不提交物理狀態；CLI help 與負配額拒絕亦已核對。

隔離工具目錄以 512 配額跑完 128 步案例。直接執行、checkpoint 續跑與原版的
位移、接觸狀態與其餘保存陣列逐位相同。11 月 6 日已安裝同一份通過驗證的原始碼。
[實際 fine128 前綴的資源預檢](fine128_resume_resource_plan_results.json)通過：
32-lag prefix 加 2048 MiB FFT cache 在 16384 MiB array budget 下餘約 452 MiB。
預測 kernel FFT 呼叫為 8252，原 8 MiB 設定為 66766；這是邏輯存取計數，
不是實際磁碟流量或速度倍數。大型診斷釋出記憶體後才啟動續跑。

這兩次 AD 數值程式都已寫出完成結果，但舊管理程序在收尾時遇到程序 CPU 計數回跳，
因此其資源管理驗收失敗；舊低配額等待方式亦未確實遵守宣告的 0.1 核心額度。
AD 期間完整事件仍在等待 CI，沒有與其數值推進重疊。上述時間不能作效能 benchmark。
新完整事件的管理程序已改用單調 CPU 高水位、`wait4` 最終 CPU 計數及完整償還 CPU 等待額度。
人工注入計數回跳的控制中，`0.260614 CPU s / 2.613259 wall s = 0.099728` 核心，
通過 0.1 配額；這僅驗證記帳，不是物理解的穩定性結果。

## 原細網格命令

下列三份輸入分別是[完整核](full_original_radau_progress.md)、
[完整 affine／前綴驗證](streamed_initialization.md)與[原方程的恢復根](captured_radau_failure.md)：

```bash
PYTHONPATH=bem/src python -B bem/tools/simulate_recovered_radau.py KERNEL \
  --affine-cache AFFINE_CACHE --initial-lift ORIGINAL_RUN/initial_lift.npz \
  --checkpoint VALIDATED_CHECKPOINT \
  --recovered-root bem/docs/diagnostics/radau_step14451/candidate_roots.npz \
  --output NEW_RUN --segment-substeps 64 128 \
  --contact-root-recovery --kinetic-root-seed --fft-cache-mb 8 --fft-rows 4 \
  --max-memory-mb 14336 --history-prefix-cache-mb 144
```

必須使用與原核相同的 runtime／x64 設定及外部四核心 CPU 管理。
CLI 先從總陣列預算保留 prefix 與讀入 checkpoint 的容量，再執行原 planner。
不指定 checkpoint 時可從相同已驗證 lift 開始新軌跡；子步 refinement 必須從初始狀態重算，
不能把 `(64,128)` 的前綴當作 `(128,256)` 的結果。
