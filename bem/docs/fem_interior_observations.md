# FEM float64 內部 probes 與原始積分一致性

2026-10-29 新增 [`run_fem_interior_reference.py`](../tools/run_fem_interior_reference.py)
與 [`fem_interior_observations.py`](../tools/fem_interior_observations.py)。
它們在原 high-rate output helper 組合 contact／interior observers，
不改積分、接觸或原始 Tatva 檔案。正在執行的完整 FEM 沒有被替換或重啟；
此工具供後續獨立波場觀測控制使用，目前未另外啟動第二份完整 FEM。

## 觀測值與時鐘

每個物體儲存固定實體座標的 Q1 位移、原始 `v_half`、位移梯度與
plane-strain 應力 `(xx,yy,xy,zz)`，全部為 true float64。
應力是在該實體點計算，不是原 bulk output 的 Gauss 平均，也沒有節點平均。
probe 在元素邊／外部或存在多重映射時會拒絕，避免替不連續的元素應力挑一側。

Tatva 的 leapfrog 更新是 `u_new = u + dt*v_half`，再更新 `v_half_new`。
因此 `time_s = absolute_step*dt` 用於位移與應力，
`velocity_time_s = (absolute_step + 1/2)*dt` 是原始半步速度的 nominal clock。
速度保留 source 的 relaxation／constraints；沒有額外力評估、時間中心化或重新積分。
normal handoff 前的值也不改標成 shear `t=0`。
後續和 BEM 速度比較時必須使用各自的真實時鐘，不能把 v_half 當 endpoint velocity。

輸出 group 為 `interior_observations`。它保留每個 probe 的 element ID、node IDs、
reference coordinates、shape values／gradients、mesh／plan SHA256 與實際 integer clocks。
[預設計畫](../examples/event16_interior_probes.json)共有 12 點：兩個物體各有近界面與
較內部的三個高度，使用帶 `.25 mm` 的座標避免普通網格的元素邊。
另已在目前實際 graded h2／end .015625 mm 的保存網格定位全部 12 點，
最大實體座標重建差為 `5.69e-14 mm`。
這只是位置映射檢查，沒有補造過去未記錄的內部波形。

resume 要求 checkpoint 前已有完整 probe prefix，並核對 plan、mesh、材料與時鐘；
不允許在原本沒有 probes 的歷程後補零值。接續前還會實際讀取原始壓縮 history、
float64 contact fields 與 probes，拒絕已損毀的 prefix。
observer 只回捲並覆寫尚未由 source checkpoint 確認的 output rows。
原 `reference_manifest.json`、`summary.json` 保留，額外 provenance 與計時寫入
`interior_manifest.json`、`interior_wrapper_status.json`、`interior_invocations.jsonl`。
原 runner 的時間包含在 wrapper interval 內，兩者不能相加。

## 實際 pinned FEM 控制

[完整紀錄與指紋](fem_interior_observations_results.json)。
使用 archived `933ae1f`，mesh 40 mm、end 10 mm、要求 normal 50 ns／shear .5 ms，
實際時鐘仍由原模型的整數步數與 dt 決定；
一核心依序執行；這是小型正確性控制，不是完整 event-16 或速度基準。

- 加入 probes 後，全部 **65 個原有 datasets 逐位元相同**，只新增 25 個 datasets。
- 對比已驗證的 pinned original control，全部 **12 個 carry arrays 逐位元相同**。
- 在 HDF5 建立後送 SIGTERM，由原 solver 完成 checkpoint，再用新 wrapper resume。
  完成後全部 **90 個 datasets** 與不中斷的觀測版逐位元相同，12 個 carry arrays 亦相同。
- 最後狀態的 probe 位移、速度與梯度另與 archived `Quad4.interpolate/gradient` 核對。
  最大差分別為 `1.78e-15 mm`、`3.56e-15 mm/s`、`1.34e-16`。
- 13 個新測試與 4 個既有 contact-output 測試通過，涵蓋扭曲 Q1 映射、half-step clock、
  affine patch、float32 拒絕、plan／clock 損毀、resume prefix、source array 不變。

原 solver 完成後會刪除 checkpoint。第一次控制的數值執行與輸出比較已完成，
但測試腳本最後誤讀被刪除的 NPZ，因此核對程序失敗；其成本與紀錄保留。
修正後的控制只在測試中包裝原 checkpoint writer，另存 NPZ 再讓原流程繼續；
原模組 bindings 與積分方程保持不變。
這個 [checkpoint 保留 fixture](../tests/fixtures/fem_interior_checkpoint_capture.py)
供重現 12-state 核對，正式 observer 不會額外保留完整 state。
兩輪通過控制、第一次核對失敗與網格映射檢查的時間均已計入原 FEM 的等待 wall interval，
沒有丟棄失敗嘗試以宣稱效能。

## 執行方式與尚未完成的工作

從 repository root 執行，例如重現小型觀測控制：

```sh
PYTHONPATH=bem/src python -B bem/tools/run_fem_interior_reference.py \
  --output bem/outputs/new-interior-control -- \
  --cores 1 --mesh-size 40 --contact-end-size 10 --time-step 0 \
  --normal-duration .00000005 --shear-duration .0005 \
  --normal-frames 2 --shear-frames 4 \
  --interface-normal-frames 2 --interface-shear-frames 32 \
  --checkpoint-interval .05 --float64-contact-observations
```

接續時，在 `--output ...` 後、分隔用的 `--` 前加入 `--resume`，其餘參數必須相同。
完整觀測控制仍需沿用現有平均 CPU 配額；`--cores` 是原 runner 的 thread 設定，
不是硬體 affinity。原 `/Volumes/Gauss-T7/tatva` 維持只讀。

後續仍需完成：完整 true float64 probe reference、BEM 有界記憶體的內部歷程重建、
反射波／能量／空間與時間收斂，以及同精度穩定性或總時間優勢。
這些新觀測與小型 identity checks 不代表上述目標已達成。

同日另保留了[實際 graded FEM 的前段快照](graded_fem_prefix_results.json)：
shear 已觀測到 `28.623180 ms`，最大 c 為 `.000643190 mm`，尚未達 `.1 Dc`，
也尚未停止載入。`c>1e-7 mm` 的原生節點最早區間為 `21.717414–21.722415 ms`，
可能最早位置為 `y=2.001823 mm`。這與舊未 graded FEM 的前段不同，
不能將舊完整曲線當作已收斂的真值；完整新 reference 仍在執行。
