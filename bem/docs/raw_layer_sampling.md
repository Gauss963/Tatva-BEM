# 在消去之前插值原始邊界算子

目標是降低下一輪 bulk／time refinement 的冷建核成本，而不刪除有限體反射歷史。
目前 fine2/Q12 resolvent 的 adaptive sampler 記錄 9665 次 transfer evaluation，
相對 32769 個半頻譜點；原建核計時含競爭，不能當成公平 speed benchmark。

原路徑插值已消去的 resolvent，包含有限體模態極點；原始 V/H/D 的頻率依賴
來自 Green kernels，在右半平面沒有相同的全域消去分母。新的實作推論是：
先以較少的原始算子取樣建立局部頻率插值，再在**每一個**原 CQ grid point
執行原線性消去。這可能節省昂貴的積分組裝，是否省時仍須實際量測。

`raw_layer_sampling.py` 使用 piecewise Chebyshev 插值，各 body 的 V、H、D
分別作相對 norm、獨立點與係數尾端檢查，避免不同單位或大矩陣族掩蓋小矩陣族。
另對獨立點的消去後 work output 作誤差核對，防止矩陣求解放大原始誤差。
未解析的區段退回逐頻率原始組裝。所有 FFT grid points 都輸出，沒有改變 dt、
輪廓、time lag、接觸律或空間秩。`max_workspace_mb` 限制插值陣列估計，
不包含原 provider 的 geometry、solver workspace 或整個程序的 RSS。

`ConstrainedSchurTransfer` 新增 supplied-operator 入口，重用原支承與 jump-map
消去，不需要再配置一份積分 geometry。複數頻率仍使用 bilinear transpose；
沒有對稱化、裁剪 eigenvalues 或人工 regularization。

這受到 [2026 elastodynamic 3D-ACA 研究](https://arxiv.org/html/2607.15292v1)
區分 raw kernel 與消去後 transfer 成本的啟發，但**不是**其 3D-ACA 演算法的重現。
目前只做 frequency Chebyshev，沒有空間低秩或論文中的 fast multipole method。

新增控制包括：

- 有共振的消去算子：逐點核對完整 FFT 半頻譜及所有需要的 CQ lags。
- 大原始矩陣族中的弱方向：必須由消去後獨立檢查發現放大誤差。
- 真實有限雙 block：每一頻率重新組裝 BEM 作對照，不重用其矩陣作插值資料。
- 固定 mixed mass、記憶體 preflight 與 supplied-operator reconstruction 的等價性。

這些測試交由 CI 執行；大型 cold build、完整非線性事件及 AD 尚未由此路徑執行。
目前 full event 的壓力／剪力差仍存在。即使本工具節省積分次數，也不構成同精度
FEM 速度優勢；後續須計入原始組裝、所有逐頻率線性消去、IFFT、IO 及線上求解。

下一個驗證會用已保存的原完整頻譜作**建核後**逐頻率核對，並保留 fresh frequency
檢查。原頻譜不得用於選樣或產生新插值資料，驗證成本也須與新建核成本分開記錄。
