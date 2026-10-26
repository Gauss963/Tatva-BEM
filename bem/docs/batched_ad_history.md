# 共用 kernel FFT 的多右端可微分歷史

現有 conditional adjoint 使用收斂根的隱式 VJP，並完整反傳 CQ 歷史；五個
forward JVP 是獨立的核對。原先五個 JVP stream 只共用較小的 cached FFT，
容量外的大歷史區塊會對同一 kernel 重算五次 FFT。

新增實驗工具 `batched_history.py` 以 `(time, bulk_dof, direction)` 儲存來源，
每個 dyadic kernel FFT 同時作用於五個方向。cached 與 streamed 路徑均共用，
小區塊仍直接計算。不刪除 lag、不擬合 kernel、不改變摩擦律。
`batched_refined_adjoint.py:forward_gradient_batched` 使用既有 physical residual、
局部 JAX JVP、五右端隱式切線求解與此歷史。

非對稱完整因果矩陣、轉置、零 cache 及正 cache 的測試皆與直接卷積核對。
五 scalar streams 對照中，streamed kernel FFT 建立次數精確減為五分之一；
這是運算次數，並非已量測的整體速度比。另加入含 sliding 的完整短歷程
梯度對照 scalar JVP／reverse VJP，交由 CI 驗證。

這些新工具尚未接入正在執行的 frozen 完整事件控制；後續需在相同 saved
完整 primal 上分別量測 cold/JIT、cached runtime、RSS 和梯度誤差。
目前仍固定 bulk、lift、初態、actuator 及離散分支，不含材料／幾何或事件時間 AD。
