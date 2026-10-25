# FEM 參考精度的修正與子步失敗紀錄

2026-10-25。前一次79.5分鐘的FEM重跑錯用了float32模型，不能列為原案例的正式參考或
同精度效能基準。原始`input/case.toml`與`input/resolved.json`都指定float64；
舊wrapper卻把`stats/summary.json`的`dtype`當作RunConfig輸入。
查閱archived source後確認，這個統計欄位實際來自`history.dtype`，描述儲存的scalar history。
因此它與float64計算模型不同，不能拿來重建計算設定。

|量|原FEM存檔|錯誤float32重跑|
|---|---:|---:|
|最大累積滑移 [mm]|0.644123|0.398881|
|首次保存滑移 [ms]，剪切時間原點|25.733|16.832|
|包含輸出的模擬耗時 [s]|舊環境，不能公平比較|4767.901|

舊結果仍保留在`outputs/fem-event16-original-h2-4c-interface`，
以`reference_fidelity.json`標明排除原因，沒有覆寫其數值或來源紀錄。
這個不同精度的結果也不能用來讓BEM的比較看起來更好。
原Tatva資料夾維持只讀。

後續檢查亦確認`fem-static-h1`、`fem-static-h05`的有效RunConfig及舊
`fem_static_onset_input.json`都指定float32。原數值與指紋保留，
但不再以這些控制與原float64存檔混用，推論純網格收斂。
靜態onset工具現預設從repository內保存的原`input/case.toml`／`resolved.json`
重建float64設定，並使用與BEM相同的固定packaged seed。
這個seed仍是float32匯出的分段線性函數，與正式FEM重跑所用的原float64 HDF5 seed分開標記。
新h2／h1／h0.5靜態比較會在獨立FEM計時後執行；目前沒有新的靜態收斂結論。
此處的float32指model geometry／operator輸入；archived static solver會在
`jax.enable_x64(True)`下建立稀疏剛度，因此不能誤稱所有靜態線性代數也是float32。
新紀錄分別保存model、座標與實際剛度矩陣dtype。

第一次新float64靜態研究已完成h2與h1，h0.5則被model builder的動態dt檢查拒絕：
原h2的配置dt=1.2492929e-8s，大於h0.5的自動穩定上限1.0684239e-8s。
未完成的資料保留在原輸出目錄，沒有把兩個已完成網格當成完整研究。
工具已改為純靜態研究使用各網格自動dt，只用來建立合法模型，不執行任何時間推進；
原authored dt、每個effective config及此政策皆記錄在新summary。
這沒有修改原完整動態FEM的dt。重跑須使用新輸出資料夾；
報告可用`report_galerkin.py --fem-static-f64 NEW_STATIC_DIRECTORY`明確選取完整研究。

```sh
PYTHONPATH=bem/src python -B bem/tools/fem_static_onset.py --cores 4 \
  --meshes 2 1 .5 --output bem/outputs/fem-static-onset-authored-f64
```

修正後，wrapper從`input/resolved.json`的完整case／solver重建模型，
核對authored TOML的SHA256、source revision及兩份輸入的精度一致性；
不再從statistics反推solver設定。另保存實際使用的restart塑性滑移／actuator指紋，
輸出`computational_dtype`、`stored_history_dtype`與`jax_enable_x64`，分開呈現計算與儲存精度。

修正後的短程原h2案例已完成。與原HDF5相比，雙block網格、第一筆保存位移、
初始塑性／累積滑移及摩擦係數皆逐位相同；static-normal的屬性也逐項保存比較。
這項短程結果支持初始化與輸入重建的正確性。
數據及指紋見[fem_precision_audit_results.json](fem_precision_audit_results.json)。

後續完整h2／float64重跑亦已通過：9000筆剪切資料的17個歷史量、
四種接觸狀態、所檢查的初始場及最後u／v_half／滑移全部與原存檔精確相同。
含輸出的模擬8032.347s，準備加模擬8032.866s；四核心設定，
監督程序沒有偵測到其他已知求解工作重疊，仍不宣稱OS隔離。
另已匯出實際使用的原始float64塑性seed／actuator，原HDF5保持未變更。
完整數據與驗證範圍見[fem_verified_reference_results.json](fem_verified_reference_results.json)。

接觸子步方面，一子步的完整45ms軌跡與原BEM七個場逐位相同，
但四子步在step14572（29.144ms）未收斂，殘差約17.39MPa，整體流程保留exit1。
之後加入有殘差檢查的信賴域／延續備援，四與十六子步皆完成45ms，
最大滑移分別0.546689與0.559055mm；距原FEM仍約15.1%／13.2%。
這恢復了這兩個離散問題的收斂，尚未證明相對FEM的穩定性或同精度速度優勢。
新增失敗平衡根／已接受前綴的診斷輸出，以便查清局部Newton與接觸分支；
它不是可以丟掉CQ記憶的resume checkpoint。

原一步BEM最大滑移差約20.7%，最新十六子步仍約13.2%，目標未達成。
正確float64 FEM參考已完成，接下來分開檢查原始seed、子步、相同Q12時間步與接觸空間誤差。
