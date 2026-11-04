# 將已驗證的歷史接回原始 Radau 推進

2026-11-04：`tools/simulate_recovered_radau.py` 整合原 initializer cache、
完整歷史恢復、局部 AD 切線組裝與原平衡方程的求根備援。
[控制結果與來源](recovered_radau_resume_results.json)已保存。
完整細網格 45 ms／FEM 一致性、初始化 AD、時間／空間收斂與同精度速度／穩定性優勢仍待完成。

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
