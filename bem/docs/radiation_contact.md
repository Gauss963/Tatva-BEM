# 二維瞬時接觸根與狀態導數

2026-10-26。`tools/radiation_contact.py` 將[接觸線快速矩陣解](structured_radiation.md)
接到原始 previous-state 摩擦律的非線性平衡與 JAX 隱式根微分。
這是未來完整動態方法的一個元件；**尚未供應或近似有限 block 的 retarded forcing**，
沒有改變 event-16 的12.22%滑移差，也沒有完整事件的加速成績。

## 方程與狀態

座標仍為 `(normal jumps, tangential jumps, platen)`，`q` 是相對 static lift 的擾動。
一個給定 incoming force 的瞬時步滿足

```text
F(q) = a*Cinf*(q-q_previous)
     + [weights*p; weights*tau; 0]
     + spring*q_platen*e_platen + forcing = 0.
```

接觸跳量為 `base_jump + q_contact`。使用擾動差計算輻射項，
避免把約9 mm的 restart 塑性值放入微小時間步的位移差。
`forcing` 必須由真正的歷史積分器提供，包括完整反射、外力、lift與actuator項。
只有上述方程並不是兩有限 block 的完整模擬；不能以它取代 `Z(s)` 的歷史。
這個元件不選擇 macro/micro 步長，也未使用已被有限 bar 反例否定的平均速度歷史。

Newton 使用各節點的原 pure `contact_point` 及局部 AD 切線，
結合 Armijo backtracking 與兩次方向的 lookahead。
每個 Newton trial 都使用**同一份舊 z／c**，不在迭代內提交狀態。
力方程按真實 contact weights、platen width轉成應力判定殘差；預設要求最大值小於1e-7 MPa。
只有通過後，呼叫端才可使用回傳位移計算及提交新塑性／累積滑移。
失敗根回傳 NaN位移，`response`亦整體回傳 NaN，
避免 unilateral law 的 opening 分支把 NaN遮成貌似有效的舊狀態。

## 反向與正向微分

使用官方 [`lax.custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
在已收斂根上求導，避開 Newton／line search tape。
若 `g(v)` 是應力殘差在根上的線性化，只需兩個固定探針：

```text
v_n = [ones(n); zeros(n); 0]
v_t = [zeros(n); ones(n); 0]
weights*g(v_n) - a*Cinf*v_n -> local normal columns
weights*g(v_t) - a*Cinf*v_t -> local tangential columns.
```

以 O(n) work還原各點的獨立切線，再用完整非對稱 Jacobian 的正向／轉置解。
沒有建立275×275的自動微分 Jacobian。
零 `dp/ds` 與非負對角的極小浮點殘差，只在保守的256 epsilon列和尺度內處理；
超出這個範圍的結構不符合回傳無效值，不能截掉真實耦合。
完整原始密集矩陣加上全域 `jacfwd` 是另一條獨立 reference 路徑。

本次找出一個 AD 組合問題：線性解本身可微，不表示可直接當作 `custom_root` 的切線回呼。
依賴 tangent RHS 的 `isfinite` 判斷，會讓本應線性的回呼帶入非線性 predicate，
在反向轉置時觸發 JAX assertion。
目前保留原始輸入／Newton方向的檢查，只在根的線性回呼使用
`chain.jax_solver(check_rhs_finite=False)`，並新增直接 `linear_transpose` 回歸。
這是修正回呼的數學契約；不是放寬根殘差或接觸分支檢查。

導數涵蓋固定 `Cinf` 下的五個摩擦參數、舊 z、舊 c、base jump、incoming force、
前一步位移、輻射時間係數與spring。
`base_jump`／forcing的偏導數不等於已完成物理 static lift 或完整歷史的導數；
這些來源仍需由外層方法正確連接。

## 已執行的驗證

原事件的250+254面板、137接觸節點，使用獨立 NumPy 摩擦律與完整密集 `Cinf`
製造已知平衡根，同時包含 sticking、正／反向 sliding、opening。
這些是具有指定 nodal forces 的元件測試，不是原事件的外力／波歷史。
快速與密集解均須達到1e-7 MPa，位移對已知根的絕對差小於2e-11 mm；
並用獨立完整矩陣與 NumPy return map重新驗算。

小型兩 block 的檢查包含：

- 五個摩擦參數與全部八類輸入的 reverse AD、同時 forward JVP及dense AD。
- 摩擦參數與混合狀態／forcing方向的中央差分。
- 三步中逐次提交 q／z／c的完整導數傳遞，核對五個 JVP與dense AD，
  最後狀態亦與獨立 NumPy逐步製造的目標核對。
- 不收斂根、非法設定與無效狀態的拒絕。

本機必要的序列化除錯重測為23 passed，當時沒有與h1 FEM同時進行數值工作。
另外增加的完整狀態目標及混合方向差分，以最後來源版本的CI結果驗收。
沒有新的本機速度 benchmark；同精度的完整事件耗時／穩定性目標仍未達成。

最後數值版本已通過[Python 3.11／3.12完整CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40298207424)。
[機讀紀錄](radiation_contact_results.json)保留來源、修正前失敗、修正後測試與各筆h1排程註記；
數字門檻不是未量測的 raw error，也不是連續解敏感度的誤差界。

```sh
bem/run.sh test -q -k 'radiation_contact or structured_radiation'
```

完整h1 FEM執行中應避免與數值測試重疊。
本輪兩次必要的序列化除錯暫停為7.892721 s與22.247724 s，均已恢復原程序；
加上先前AD控制的兩筆，已記錄的h1排程暫停共92.553771 s。
h1 raw wall time保留這些暫停，不作未中斷的效能基準。
