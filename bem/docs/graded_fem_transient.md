# 保留原 source kernels 的 graded FEM 動態參考

`run_fem_reference.py --contact-end-size ...` 現將已驗證的 chamfer mesh adapter
接入原 `run_simulation_dumped`。只在複製函式的 globals 中選用新 mesh builder，
不改原 module bindings、source files、材料、static initializer 或 explicit dynamics。
新的模型從相同原始 float64 P1 restart 場取樣，並驗證原 archive／input 指紋。
只用來投影 restart 的第一份 model 會在正式 transient 前釋放。

小模型的真實 transport／resume 控制已完成：h40、end10、原自動 dt
4.83849e-8 s、0.5 ms shear。完整一次執行，另一次以 SIGUSR1 觸發原 source 的
checkpoint，然後從它接續。兩次 HDF5 的 **49 個 datasets 全部逐位元相同**，
涵蓋模型、保存的歷史、界面場及最後狀態。
這驗證 mesh adapter 與 checkpoint 接口可接回真實 source；不是細網格完整事件的證據。
[控制結果與來源指紋](graded_fem_resume_results.json)保存全部 dataset 名稱與計時。

`--checkpoint-interval SECONDS` 使用原 source 的周期 checkpoint；收到 SIGUSR1／SIGTERM
時也會安全儲存，再以原 `SimulationCheckpointed` 例外結束。
使用完全相同的命令加上 `--resume` 接續。runner 在動到保存的 simulation 之前，
核對 source、全部數值／輸出 arguments、原始輸入、投影 restart 及 effective config。
不覆寫原 manifest 或 projected restart，已完成的 run 會拒絕 resume。

`invocations.jsonl` 記錄各次執行。最後的 `wall_seconds`、CPU time 與
`timing_seconds` 累加已中斷及接續的工作，包括重複準備成本，避免只量最後一段。
若有 start 沒有對應 end，累計時間會設為 null，並標示 accounting 不完整。
單次保存資料結束後的報告格式化時間不在這些 timer 內；並行負載仍須另外記錄。

本次 32.46 s 的小模型驗證期間，暫停同一個已確認身分的 BEM 建核程序，
完成後已繼續。這段 wall time 保留在該次 BEM 建核計時內，沒有重啟或扣除。
不據此小模型宣稱 FEM／BEM 速度比。

候選完整參考的命令如下；`--time-step 0` 讓原模型為加密網格重新計算 dt，
不能沿用原 h2 的時間步而忽略最細元素與原 contact stability bound。

```sh
PYTHONDONTWRITEBYTECODE=1 python -B bem/tools/run_fem_reference.py \
  --output bem/outputs/fem-event16-graded-h2-end015625-4c-interface \
  --cores 4 --mesh-size 2 --contact-end-size .015625 --time-step 0 \
  --shear-duration .045 --normal-duration .002 \
  --normal-frames 4 --shear-frames 64 \
  --interface-normal-frames 4 --interface-shear-frames 9000 \
  --float64-contact-observations --checkpoint-interval 600
```

先加 `--prepare-only` 可建立配置與投影 restart，不能當成已執行時間積分。
完整 graded FEM transient 尚未完成；目前 h2／h1 的 complete events 仍是原未端點加密的參考。
原 `/Volumes/Gauss-T7/tatva` 保持只讀，所有新輸出寫入本 checkout。

新增的[雙精度接觸觀測與完整時空比較](float64_contact_comparison.md)已通過
真實 FEM 的原輸出逐位元相等及中斷／接續控制。
它直接保存帶正負號的 endpoint shear，供新的完整 graded 參考使用。
