# 固定原時間步的 FEM 空間控制

2026-10-26。完整 h1 控制已完成，並已通過相同步數／保存時間、原始 float64 seed
及末端資料檢查。h2／h1 最大累積滑移分別為0.644122821／0.619884079 mm；
兩網格末端累積滑移 P1 L2 差3.52%，壓力場差17.62%。
原 h2 已逐欄核對原存檔；本控制只縮小 FEM 網格，保留原來源版本、float64、
`dt=1.2492928633545953e-8 s`、2 ms normal、45 ms shear、摩擦參數與停止載入政策。
restart 的241個原始 float64 塑性值作 P1 插值至481個節點；
每個網格仍重新求自己的 static normal initialization。
原 `/Volumes/Gauss-T7/tatva` 只讀，運算與比較都在本 checkout。

## 為何需要這個控制

目前原始 float64 seed 的完整 Radau BEM 最大累積滑移為0.565404111 mm，
原 h2 FEM 為0.644122821 mm，差12.22%。尚未達成相同精度的時間或穩定性優勢。
[連續時間閉鎖接觸的線性比較](linear_bulk_response_results.json)顯示，
h1 FEM 相對同一 BEM 的壓力／剪應力差比 h2 小；
這支持檢查 FEM 空間誤差，不能推論非線性破裂已收斂。

Linder 等（2026）的[動態 FE–BE 接觸研究](https://arxiv.org/html/2603.26446v1)，
§2.1–2.3，討論有限邊界節點質量造成的人工高頻振盪，以及 massless boundary 方法。
其系統包含半空間接觸與 reduced FE，並非本案兩有限 block 的獨立 BEM；
因此不採用論文的速度比，也不據此宣稱 event-16 的誤差來自邊界質量。
本控制保持原 FEM 方程與 penalty，先量測網格改變帶來的差異。

## 已執行的驗收

`tools/compare_fem_mesh.py` / `bem/run.sh fem-mesh-compare` 要求：

- h2 基準已通過 archived-field verification，檔案大小、mtime、summary hash仍相符。
- 兩個 run 均為已完成的原始時段、float64、相同 source archive 與 protected case inputs。
- resolved config 只允許 mesh size、已檢查的 restart path 不同；時間步與其他參數必須相同。
- 重新讀取 source／projected restart，逐值確認原始 float64 P1 投影與 actuator；比較前後再次核對。
- 全部 shear step IDs 與保存時間精確相同，不能用時間插值掩蓋不同軌跡。
- 最後保存的 z／c 必須等於 float64 final_state 按儲存 dtype 捨入後的值；
  最後 bulk u／v_half 也必須有限。

每個 scalar history 與五個界面量都保留最大絕對差、末端差及完整時段 L2 差。
界面以兩套節點的聯集積分 P1 場，時間使用相同保存時間的 P1 插值；
空間與時間積分對這些插值場精確，**沒有約束未保存步之間的振盪**。
零參考的 relative norm 記為 `null`，仍保留 absolute norm，不以小常數掩蓋。

高頻輸出未直接保存壓力，因此用 float32 `friction_strength / friction_coefficient` 重建，
並要求 coefficient 嚴格為正。末端另從真正 float64 `final_state/u`、原始座標／node pairs、
`z`、`c` 重建壓力與 `kt*(tangential jump-z)`。
後者明確標為 `elastic_trial_shear`，並非另呼叫完整摩擦 return map 的結果。
已完成 h2 的末端壓力與高頻資料重建差最大為1.054e-5 MPa，符合兩者儲存精度不同的範圍；
這個觀察不是一般浮點誤差上界。

正確的 seed 投影不代表 static initializer 不會再改變塑性值；
報告另外列出初始 frame、第一個 shear frame 與輸入 seed 的差。
兩者也是 float32 儲存，不能直接把微小差異認定為初始化誤差。
目前 h2 初始 frame 相對原 seed 的最大差為4.766e-7 mm。

兩張網格不足以推出 Richardson order 或連續解誤差界。
結果會保留 `fem_mesh_convergence_verified=false`、`same_accuracy_fem_advantage_verified=false`；
需要依量測結果繼續縮小相關離散誤差，不能把通過比較程式當成達成物理驗收。

```sh
FDBEM_CORES=1 bem/run.sh fem-mesh-compare \
  bem/outputs/fem-event16-authored-f64-h2-4c-interface \
  bem/outputs/fem-event16-authored-f64-h1-original-dt-4c-interface \
  --field-verification bem/outputs/fem-event16-authored-f64-field-comparison/summary.json \
  --bem-runs bem/outputs/galerkin-fine2-q12-contact64-original-seed-kinetic \
  bem/outputs/compact-radau6-q12-event16-original-seed-contact192 \
  --output bem/outputs/fem-h2-h1-original-dt-comparison-schemafix-20261026
```

BEM 末端另對 h2、h1 分別比較相同 nominal 45 ms shear 終點，保留各 run 的 dt 與 schedule。
FEM 最後一步可略超過45 ms；保存時間與 summary 中的實際終點需一併檢視。
沒有用不同停止載入政策的 diagnostic 取代原 case，也沒有以 endpoint 差冒充整段軌跡誤差。

## 執行與時間範圍

完整 h1 使用四核心 thread 設定，沒有 macOS hard affinity。
監督器保存 process samples、已知求解工作、來源與輸入指紋。
先前兩次AD控制，加上[非線性根的必要序列化除錯](radiation_contact.md)，
四筆已記錄的h1排程暫停共92.553770543 s，均已恢復；
這組 h1 raw wall time **不能當成未中斷的速度基準**。
比較程式保留原始時間，暫停與排程記錄另見
[完整反射控制紀錄](radiation_bar_results.json)中的 `h1_scheduling_annotations`。
該紀錄保留先前兩筆，另兩筆及最新總數見非線性根的驗證紀錄。
既有 h2 的8032.347 s模擬時間屬另外已驗證的四核心控制。
h1 含輸出的原始模擬時間為30716.085 s，準備加模擬為30723.748 s；
上述已記錄暫停保留在原始數值中，不將扣除暫停後的時間包裝成新速度基準。

## 完整結果的含義

|末端量／事件|FEM h2|FEM h1|
|---|---:|---:|
|最大累積滑移 [mm]|0.644122821|0.619884079|
|P1 平均累積滑移 [mm]|0.632589659|0.610415084|
|最大壓力 [MPa]|345.203168|491.647490|
|首次保存滑移超過1e-7 mm，shear時間 [ms]|25.732864|28.143138|
|首次保存停止載入，shear時間 [ms]|29.788319|29.448275|

首次滑移依保存間隔判定，並非解析成核時間。端點壓力峰值隨網格增大，
不能從兩個峰值或兩張網格推出局部角落已收斂。完整最大滑移歷史 L2 差9.22%，
動能歷史 L2 差51.31%；末端滑移接近也不代表整段波傳一致。

既有原始 seed／N64 BDF2 相對 h1 的末端累積滑移 P1 L2 差8.28%、
壓力差10.53%；Radau 相應差7.99%、10.04%。
Radau 的最大累積滑移仍比 h1 低8.79%，相對原 h2 的差仍為12.22%。
這些量測顯示 FEM 空間誤差確實影響比較，剩餘 BEM 差異仍須由時間／接觸控制處理。

單元測試以獨立 tensor-product Gauss 積分檢查時空 norm，
另以小型完整 HDF5 fixture 檢查來源／投影、初始化差、float64 endpoint、
不相同時間步與被修改的 seed 會被拒絕。
真實 h1 工作已完成，以上比較使用完整9000筆 shear 資料與真正 float64 末端。

[機讀稽核紀錄](fem_mesh_control_results.json)保存 h2 儲存／末端核對的來源指紋，
以及 h1 原程序與已啟動後處理 observer 的派送狀態。
observer 等原 solver 與 supervisor 都結束，確認正式完成、source SHA 與比較程式 CI 後，
才執行網格／BEM末端比較及歷史曲線繪圖；它不重啟或傳送 signal 給 FEM。
該舊紀錄的 `pending_full_h1_spatial_result` 是執行中的歷史快照，仍保留。
最新[完整結果紀錄](fem_mesh_completed_results.json)另外保存真正完成與比較的指紋。
原後處理先遇到舊 h2 manifest 沒有 `restart_projection` 欄位；
修正後以 actual/source seed 完全相同確認它未經投影，再接受缺少的 null 欄位。
來源、物理量與 seed 檢查均保留，沒有更改既有模擬資料。
