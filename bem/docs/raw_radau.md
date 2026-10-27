# 不依賴 BDF2 頻譜的 Radau raw 取樣核心

2026-10-28。[實際細網格時間響應](bulk_time_symbol.md)支持進一步測試 2 µs Radau，
但既有 BDF2 circle 無法安全重用到這個 RK contour。
`tools/radau_raw_sampling.py` 因此新增直接取樣核心；完整新 kernel producer、
fine 45 ms 事件與 AD 尚未完成，不把本核心當成正式 backend 或速度成果。

Radau symbol 的兩個無因次 eigenvalues 滿足

```text
w² - (4+2*z)*w + 6*(1-z) = 0.
```

在下半圓選擇連續的 square-root 分支，從兩個實際 `s=w/dt` 各自重新組裝
兩個 body 的 V／H／D，再做原 trace Schur elimination 與 work-coordinate compliance。
靠近 `z=1` 時，較慢 mode 使用 eigenvalue product 形式求值，避免相近數相減。
測試另核對其 causal limit，不以一般 eigensolver 的小 eigenvalue 當高精度 reference。

兩 branch 依序取樣，共用唯讀 geometry，維持單一 branch 的 interpolation workspace。
原 sampler 的 raw-family tolerance、Chebyshev tail、獨立 output checks、
細分及 direct fallback 全部保留。原 BDF2 symbol 在 adapter 內只作可反解的角度座標，
沒有使用任何 BDF2 frequency matrices 或時間權重。

各 branch 在半圓 endpoint 可以是複數，不能個別使用 Hermitian mirroring／irfft。
必須先得到兩個完整 branch，再以 Cayley–Hamilton 形成

```text
F0 = (w1*L0-w0*L1)/(w1-w0)
F1 = (L1-L0)/(w1-w0)
L_RK = I*F0 + Delta(z)*F1.
```

這裡的 w 為無因次 eigenvalue，與既有 compact history 的 API 一致。
組合後才可檢查 real endpoints、保存完整兩族時間權重與 finite-DFT shift wrap。
沒有刪除快 mode、非零 lag 或接觸狀態。

已完成的控制包括：連續 branch 與完整 stage symbol、所有 indices 恰好 emit 一次、
原 sampler binding 不變、dimensionless derivative scaling，以及近 causal limit 的精度。
另對真實小型兩有限體重新組裝全部 frequency matrices，將兩族完整 CQ 權重展開後，
與另一套原 compliance-condensation 路徑逐點建立的 dense stage weights 核對。
相關 25 項測試通過；這不是細網格資源或完整事件精度／效能驗證。

下一步 producer 仍需獨立資源與 disk preflight、完整因果 startup、
原始 RK operator audit、與現有 simulator 的 static lift／contact 初始化核對，
以及與 FEM 的完整空間／時間比較。
