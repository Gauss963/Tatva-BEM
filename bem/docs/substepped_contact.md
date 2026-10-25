# 接觸子步與完整條件式 adjoint

2026-10-24；更新2026-10-25。這是實驗選項，精度與收斂驗收仍未完成。
2026-10-25：完整一子步的七個場已與原BEM逐位相同。
原Newton四子步在29.144ms失敗；加入殘差檢查的信賴域／延續備援後，
四與十六子步均完成45ms。最大滑移差降至約15.1%／13.2%，仍未達成FEM驗收。
失敗與[FEM參考精度修正](reference_precision.md)分開保留。
動機是[2 us完整事件](fine2_event.md)仍可在一步內跨過84%的弱化區間。
這項改進保留 bulk CQ 時間網格、參數及原 `contact_point` 更新公式，
在前後兩個已接受 jump 間取線性路徑，以較小局部步更新塑性／累積滑移。
微步內的中間波動仍由此線性路徑近似，不能當作細化 bulk 時間步的替代證明。

`tools/substepped_contact_experiment.py` 以 `jax.lax.scan` 更新局部狀態，
`jacfwd` 組裝端點切線；外層平衡根沿用原 Newton／line search 與 implicit root AD。
子步數為1時，直接委派原求解器；單點回傳值與完整小型CQ軌跡的七個場皆逐位核對。
定壓正向滑移的測試另與連續弱化解析解比較，子步加密使誤差下降。

子步使當步接觸除了依賴舊塑性／累積滑移，也依賴舊位移。
反向計算因此增加一個位移 cotangent carry，並將 reverse 起點往前一個平衡根，
再接上原 closed/sticking prefix 的完整轉置歷史。
程式會檢查該前綴確實不再收到額外位移 cotangent；不足的 correction horizon 直接報錯。
`tools/adjoint_resolvent.py` 已加入這條路徑，沒有把 Newton 迭代保存到反向圖中。

小型20步、8子步案例以獨立的原始 Z 完整歷史 JAX scan 核對 forward，
並逐項比較五參數、初始 z/c、全部 actuator 載入 cotangent。
此驗證與本機完整事件分開看待；完整事件梯度尚須另有獨立參考，
單獨算出 adjoint 不會將 `all_compared_cotangents_passed` 設成true。
材料、幾何、靜態lift和停止載入事件時間仍固定。

建立完成的 kernel 可直接重用，例如：

```sh
FDBEM_CORES=4 bem/run.sh resolvent-simulate bem/outputs/direct-resolvent-fine2-q12 \
  --constitutive-substeps 8 --fft-cache-mb 8 --max-memory-mb 8192 \
  --contact-root-recovery \
  --output bem/outputs/fine2-q12-contact8
FDBEM_CORES=4 bem/run.sh resolvent-adjoint bem/outputs/fine2-q12-contact8 \
  --fft-cache-mb 8 --max-memory-mb 8192
```

輸出保留子步原始碼指紋及微步摩擦功；固定各微步端點壓力的積分功差只作診斷，
不當作總能量或滑移誤差上界。比較器會使用真實微步功，並另外保存宏步端點估計。
目前只允許子步案例的 standalone adjoint；跨run完整微步分支比較仍未實作。
接觸子步將原一步約20.7%的最大滑移差降到13.2%，頻率插值本身並不保證精度改善。

## 完整事件的備援求解結果

`robust_contact_experiment.py`只在原Newton失敗時啟用Powell hybrid信賴域法；
仍不通過時，從previous-state contact root以接觸方程延續到完整子步方程。
所有trial都不commit狀態；只有最後原始子步residual低於1e-7MPa時，
重新進入既有implicit-root API接受q/z/c。SciPy的success旗標不能取代殘差檢查。
沒有加入阻尼或改變最後的接觸平衡式。

|量|原一步|4子步＋備援|16子步＋備援|原FEM存檔|
|---|---:|---:|---:|---:|
|最大累積滑移 [mm]|0.511082|0.546689|0.559055|0.644123|
|最大滑移相對差|20.65%|15.13%|13.21%|—|
|末滑移分布相對L2差|20.03%|14.20%|12.13%|—|
|停止載入 [ms]|29.200|29.154|29.138|29.788|
|推進 [s]|73.81|92.30|91.74|不同演算法與設定，尚不作同精度比|
|成功恢復的step數|—|2|3|—|
|微步固定壓力積分功差 [MPa mm²]|6.31|1.95|0.516|未量測|

保存場皆有限、壓力為正、累積滑移非遞減，彈簧在已保存路徑中維持壓縮。
兩組最大殘差約7.80e-8與7.46e-8MPa，均低於原門檻，並未放寬容差。
首次保存滑移仍為26.390ms，與原FEM25.733ms的差距沒有消失；
壓力／剪力末分布L2差仍約20%。因此這不是已收斂或與FEM一致的解。

真正失敗step的275座標矩陣／狀態已保存為小型測試fixture，
核對備援恢復、trial不commit，以及恢復根的mu_k隱式梯度與獨立差分。
這些測試也不代表完整45ms梯度已有獨立驗證。

完整數據與來源指紋見[substepped_contact_results.json](substepped_contact_results.json)。

![完整事件與接觸子步](figures/substepped_contact_event.png)

![最後界面分布](figures/substepped_contact_profiles.png)
