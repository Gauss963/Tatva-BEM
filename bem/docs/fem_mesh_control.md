# 固定原時間步的 FEM 空間控制

2026-10-26。完整 h1 控制仍在執行，尚無 h1 最終解。
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

## 完成後的驗收

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
  --output bem/outputs/fem-h2-h1-original-dt-comparison-20261026
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

單元測試以獨立 tensor-product Gauss 積分檢查時空 norm，
另以小型完整 HDF5 fixture 檢查來源／投影、初始化差、float64 endpoint、
不相同時間步與被修改的 seed 會被拒絕。
正式 h1 結果只有在真實工作完成並通過上述檢查後才產生。

[機讀稽核紀錄](fem_mesh_control_results.json)保存 h2 儲存／末端核對的來源指紋，
以及 h1 原程序與已啟動後處理 observer 的派送狀態。
observer 等原 solver 與 supervisor 都結束，確認正式完成、source SHA 與比較程式 CI 後，
才執行網格／BEM末端比較及歷史曲線繪圖；它不重啟或傳送 signal 給 FEM。
該紀錄的 `pending_full_h1_spatial_result` 明確表示尚無完整 h1 結果。
