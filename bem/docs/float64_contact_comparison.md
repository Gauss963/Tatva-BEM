# 雙精度接觸歷史與帶正負號的剪力

完整 graded FEM 參考新增可選 `--float64-contact-observations`。
原 source 的高頻輸出保存 float32 的摩擦強度、μ 與狀態，沒有節點的 signed shear。
摩擦強度是上限，不能當成實際剪力。新增輸出直接觀察 float64 endpoint state：
`pressure = kn*max(normal_jump,0)`，
`elastic_shear = kt*(tangential_jump-plastic_slip)`，接觸開啟時剪力為零。
同時保存位移跳躍、塑性 offset、弱化記憶、μ、強度與整數步數。

`fem_contact_observations.py` 僅對已核對 SHA 的 archived `933ae1f` 函式建立副本，
在既有 `save_high_rate_interface` 末端加入一次 observer call。
移除該 call 後，AST 必須與原函式完全相同；原檔案與 module bindings 都不修改。
這是明確記錄的輸出 adapter，不會對 integrator 多做一次接觸力更新。
新 group 亦遵守 source checkpoint 的 frame prefix，覆寫 checkpoint 之後被放棄的列。

真實 FEM 小模型驗證使用 h40、end10、原自動 dt=4.838490394517019e-8 s、
0.5 ms shear、四核心 thread 設定。三次獨立執行分別為：原輸出、加入 observer、
以及加入 observer 後以 SIGUSR1 checkpoint 再接續。

|驗證|實際結果|
|---|---|
|加入 observer 前後原有 49 個 HDF5 datasets|全部逐位元相同|
|完整一次執行與中斷／接續的全部 69 個 datasets|全部逐位元相同|
|35 frames × 21 nodes 的八個新增接觸場|全部真正 float64|
|直接壓力與 carry 中的 strength/μ 最大差|7.11e-15 MPa|
|新增最後塑性 offset／弱化記憶與 float64 final_state 最大差|0／0 mm|

[結果與來源指紋](float64_contact_observation_results.json)保留細節。
這是輸出與接續驗證；小模型沒有經歷完整破裂，不能據此宣稱完整事件準確或更快。
驗證期間暫停已確認身分的自有 BEM 建核程序 43.73 s，完成後恢復。
暫停時間仍計入該建核實際 wall time，不從總成本扣除。

`compare_float64_contact_history.py` 對完整 45 ms 事件逐項比較壓力、signed shear、
塑性 offset、累積滑移、μ、摩擦強度。它沿用已驗證的 P1 時空積分：
保留兩側所有保存的空間與時間節點，對共同時間區間、整個接觸面積分，
不擬合時間偏移、不外插、不裁掉尖端。SHA 以 streaming 方式核對兩個完整資料檔。

FEM 剪切時間由 `step_id * actual_dt` 重建，允許原 solver 自動計算 dt。
不能以 float32 timestamp 相減，也不能以要求的 normal duration 取代實際整數步數。
原 source 在最後一筆法向輸出之後重設弱化記憶，所以比較從第一筆真正剪切觀測開始，
省略的初始區間會明確記錄。保存幀之間採 P1；這不界定未保存的振盪，亦不是非線性穩定性證明。

完整參考需加上該旗標，再使用：

```sh
PYTHONPATH=bem/src python -B bem/tools/compare_float64_contact_history.py \
  BEM_RUN FEM_RUN --output FRESH_COMPARISON_FOLDER
```

工具要求完整 source-frozen float64 事件、相同來源版本、材料、有限體尺寸、chamfer、
接觸律與載入參數。輸出誤差尚須和獨立空間／時間／摩擦子步收斂、
原始因果方程稽核及全部付費計算成本一起判讀。
