# 接觸子步與完整條件式 adjoint

2026-10-24。這是實驗選項，尚未完成原事件的完整子步對照。
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
  --output bem/outputs/fine2-q12-contact8
FDBEM_CORES=4 bem/run.sh resolvent-adjoint bem/outputs/fine2-q12-contact8 \
  --fft-cache-mb 8 --max-memory-mb 8192
```

輸出保留子步原始碼指紋及微步摩擦功；固定各微步端點壓力的積分功差只作診斷，
不當作總能量或滑移誤差上界。比較器會使用真實微步功，並另外保存宏步端點估計。
目前只允許子步案例的 standalone adjoint；跨run完整微步分支比較仍未實作。
接觸子步與頻率插值都不會自動消除目前約20.7%的FEM差距。
