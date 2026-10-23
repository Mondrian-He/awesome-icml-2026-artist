# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 53 of 197

## 目录 (Table of Contents)

1. [OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](#paper-1717)
2. [Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation](#paper-1718)
3. [Precision-Induced Miscalibration: Understanding and Correcting Confidence Distortion in Quantized Neural Networks](#paper-1719)
4. [Anytime-Valid Inference for Online Ranking of Large Language Models](#paper-1720)
5. [R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](#paper-1721)
6. [Parameter-Masked Decoupled Optimization for Cross-Domain Class-Incremental Learning](#paper-1722)
7. [From Prompts to Responses: Dual-Sided Data Leakage and Defense in Split Large Language Models](#paper-1723)
8. [Power-Boosted Granger-Causal Discovery for Large Heterogeneous Panel Data](#paper-1724)
9. [Ekka: Automated Diagnosis of Silent Errors in LLM Inference](#paper-1725)
10. [What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions](#paper-1726)
11. [VENOMREC: Cross-Modal Interactive Poisoning for Targeted Promotion in Multimodal LLM Recommender Systems](#paper-1727)
12. [From Content to Knowledge: Lightning Fast Long-Video Understanding with Neural Knowledge Representations](#paper-1728)
13. [TritonGym: A Benchmark for Agentic LLM Workflows in Triton GPU Code Generation](#paper-1729)
14. [TimeOmni-VL: Unified Models for Time Series Understanding and Generation](#paper-1730)
15. [LIMMT: Less Is More for Motion Tracking](#paper-1731)
16. [Bridging Tokens and Geometry: Token-wise 3D Supervision for CAD Generation](#paper-1732)
17. [MOC: Multi-Order Communication in LLM-based Multi-Agent Systems](#paper-1733)
18. [Federated Multi-view Clustering for Remote Sensing Data](#paper-1734)
19. [On the Coordination of Value-Maximizing Bidders](#paper-1735)
20. [ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models](#paper-1736)
21. [Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization](#paper-1737)
22. [Finite-Width Neural Tangent Kernels from Feynman Diagrams](#paper-1738)
23. [MASH: Modeling Abstention via Selective Help-Seeking](#paper-1739)
24. [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](#paper-1740)
25. [SIGMA-PPG: Statistical-prior Informed Generative Masking Architecture for PPG Foundation Model](#paper-1741)
26. [Dynamic Thinking-Token Selection for Efficient Reasoning in Large Reasoning Models](#paper-1742)
27. [MutAtlas: A PDB-Wide Energy-Guided Atlas of Protein Mutation Effects](#paper-1743)
28. [Delving into Non-Exchangeability for Conformal Prediction in Graph-Structured Multivariate Time Series](#paper-1744)
29. [Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](#paper-1745)
30. [PolicyGuard: Towards Test-time and Step-level Adversary Defense for Reinforcement Learning Agent](#paper-1746)
31. [Constrained Bayesian Experimental Design via Online Planning](#paper-1747)
32. [Probabilistic Salient Object Ranking](#paper-1748)
33. [VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model](#paper-1749)

---


<a id="paper-1717"></a>

## OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

### Images

![0e7148cba1b9fb2c2162cccf4aa0f73f93bf483542a377166f7761c8afbb48f9.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/0e7148cba1b9fb2c2162cccf4aa0f73f93bf483542a377166f7761c8afbb48f9.jpg)

![138bcb699deac7896f0b81632e4a69cc311f4f3e5becce9d40ca107a72501f07.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/138bcb699deac7896f0b81632e4a69cc311f4f3e5becce9d40ca107a72501f07.jpg)

![21cf5e13fa4cd466182c8fca9cecb1af793d4f80b7aa10b5bd6805dcfdf23d48.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/21cf5e13fa4cd466182c8fca9cecb1af793d4f80b7aa10b5bd6805dcfdf23d48.jpg)

![27b9803c7420eaafe7529ce9edcf31693dcf8dfb6037302c7a080066eff27491.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/27b9803c7420eaafe7529ce9edcf31693dcf8dfb6037302c7a080066eff27491.jpg)

![2873525bd7bd2269a8e2c494952ed20b9bc33a178b0a052d12e4278e39c4e3d9.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/2873525bd7bd2269a8e2c494952ed20b9bc33a178b0a052d12e4278e39c4e3d9.jpg)

![3b693cb7c8c428c840b4fd4cadc9e527cfe0cb5937d44a3986a56db301c7f635.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/3b693cb7c8c428c840b4fd4cadc9e527cfe0cb5937d44a3986a56db301c7f635.jpg)

![3bda2920eb1b057d2ecd9290d3d00cbb0fa9af4cbce699df94277eafa8fbecfd.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/3bda2920eb1b057d2ecd9290d3d00cbb0fa9af4cbce699df94277eafa8fbecfd.jpg)

![3f97e3cd77075d5cc98f7c67be7afdccfff98cbea3c551af27f650fd1a584ff4.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/3f97e3cd77075d5cc98f7c67be7afdccfff98cbea3c551af27f650fd1a584ff4.jpg)

![4561c3256f84092370873ef9ca890cd2b2ad439462d8f9e7cabc36c2a014f86f.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/4561c3256f84092370873ef9ca890cd2b2ad439462d8f9e7cabc36c2a014f86f.jpg)

![46103f2b18cc03ec3cfb5399116dc44cc27fd188524e7e19a98f573c7e6a1463.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/46103f2b18cc03ec3cfb5399116dc44cc27fd188524e7e19a98f573c7e6a1463.jpg)

![46b382c0382ad9092f8365274e97ba481eaee17b3ff4fd95edb5b48cb1c65f43.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/46b382c0382ad9092f8365274e97ba481eaee17b3ff4fd95edb5b48cb1c65f43.jpg)

![4a26e8ba41ce2b8a94422d74c64b2e733c1082d8442ebd2851bed3fbc850698a.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/4a26e8ba41ce2b8a94422d74c64b2e733c1082d8442ebd2851bed3fbc850698a.jpg)

![5ee11afe7f8ea56cc532c01c4e89b3b2edd004745c59b10770d48654e639dfe6.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/5ee11afe7f8ea56cc532c01c4e89b3b2edd004745c59b10770d48654e639dfe6.jpg)

![6aa7f49d23d36f39d7b5238c2b8c6621cc85a64473a47e89d951f67d99c5965c.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/6aa7f49d23d36f39d7b5238c2b8c6621cc85a64473a47e89d951f67d99c5965c.jpg)

![6d4f1065b924cc4cbd2a2a8409d308c790fa5f7eaef3bb5c9aff59e2300a5ad3.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/6d4f1065b924cc4cbd2a2a8409d308c790fa5f7eaef3bb5c9aff59e2300a5ad3.jpg)

![7033f3e5a33549ded085222de0ee69c51c77dcea640b324e29e57895aceb509f.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/7033f3e5a33549ded085222de0ee69c51c77dcea640b324e29e57895aceb509f.jpg)

![74329c67abcbe51b40bbd28519f43a645b1a153050312806ef290d33a906b925.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/74329c67abcbe51b40bbd28519f43a645b1a153050312806ef290d33a906b925.jpg)

![86166e645d317f1af27338aedb13691d0e3481598e3eaff474a02094b3874798.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/86166e645d317f1af27338aedb13691d0e3481598e3eaff474a02094b3874798.jpg)

![8de738c3c9c4548fdffa7a8ccd1389c49b26f8794cae6d9a5a6b178f3e594903.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/8de738c3c9c4548fdffa7a8ccd1389c49b26f8794cae6d9a5a6b178f3e594903.jpg)

![9ac2297cd9404cc8bc65d0a583fb132246cfd829887e9fe899f222eb694725b2.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/9ac2297cd9404cc8bc65d0a583fb132246cfd829887e9fe899f222eb694725b2.jpg)

![d99bc5178d867a9a61f4872f87ec9fb8e81b2ced2cfa74330cd9c28f582ad619.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/d99bc5178d867a9a61f4872f87ec9fb8e81b2ced2cfa74330cd9c28f582ad619.jpg)

![da47a2df3188bafa0f088b8e5e748f27daecfeff36cb546aa57a0305d3942578.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/da47a2df3188bafa0f088b8e5e748f27daecfeff36cb546aa57a0305d3942578.jpg)

![e825c89627f28a065e521efe4e4d89822fb90d445dcf29f31c6b7ac31a0d727c.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/e825c89627f28a065e521efe4e4d89822fb90d445dcf29f31c6b7ac31a0d727c.jpg)

![e9d3a1dd31c8dc304508ae5b9fa02723d066e7c482b11a6bc478b2be830b9535.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/e9d3a1dd31c8dc304508ae5b9fa02723d066e7c482b11a6bc478b2be830b9535.jpg)

![ecd0bda487e7bacab67a619252418bb0282ff169c29fa1ded0fe45c7c1380870.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/ecd0bda487e7bacab67a619252418bb0282ff169c29fa1ded0fe45c7c1380870.jpg)

![f138ce2c54f45a6f1025630648324c56d95057b47324072e22a669ba4743a2e5.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/f138ce2c54f45a6f1025630648324c56d95057b47324072e22a669ba4743a2e5.jpg)

![f9fc6e9f685d3cc36ea2b7217d44d8f91c56c70bc0396020770398c0818cc9a8.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/f9fc6e9f685d3cc36ea2b7217d44d8f91c56c70bc0396020770398c0818cc9a8.jpg)

### Tables

![11c9958d7b860e8175e1e72d8fb17dad7f77c80df5a2fe508c584e8304a6faf8.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/11c9958d7b860e8175e1e72d8fb17dad7f77c80df5a2fe508c584e8304a6faf8.jpg)

![223e6dbaac59547bd47688420a115d176f3723eefa5631f49ff4d7340c279240.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/223e6dbaac59547bd47688420a115d176f3723eefa5631f49ff4d7340c279240.jpg)

![2c95c72f2688f99a4363b730169fc5578535663aa60543e02abcb2a7770df73f.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/2c95c72f2688f99a4363b730169fc5578535663aa60543e02abcb2a7770df73f.jpg)

![6f9eb91ead2c6b55cabf27677fb126c5473628665a51491e5efb39bf05118ce5.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/6f9eb91ead2c6b55cabf27677fb126c5473628665a51491e5efb39bf05118ce5.jpg)

![7dd4501cc0a172db1ed23f18a5714426a5b3a30aee11601a34f9a6a9fae62e70.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/7dd4501cc0a172db1ed23f18a5714426a5b3a30aee11601a34f9a6a9fae62e70.jpg)

![8b0e17e50d186264328389e9d33c19fa02d3e7cf26e706b1eef1af65461156fe.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/8b0e17e50d186264328389e9d33c19fa02d3e7cf26e706b1eef1af65461156fe.jpg)

![a7c12ff14136830d4ab182fa1de75b805543a3d89fda00171a7155c8bcad0071.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/a7c12ff14136830d4ab182fa1de75b805543a3d89fda00171a7155c8bcad0071.jpg)

![c4fd1468606ae21e9e6630d408064f8e2a72a371f608a327f6ea4b7c977964c5.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/c4fd1468606ae21e9e6630d408064f8e2a72a371f608a327f6ea4b7c977964c5.jpg)

![df66909decbe2162881246950edbf22cf43c05888843294b1aec769e820a00c0.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/df66909decbe2162881246950edbf22cf43c05888843294b1aec769e820a00c0.jpg)

![e907786c802a0ad929d28354a2a608ea6d9313f7057e27fb343999b66be8aee6.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/e907786c802a0ad929d28354a2a608ea6d9313f7057e27fb343999b66be8aee6.jpg)

![eae5ec9b74107f9aebe21afaf543a81567d4c0277ff0182d757f0b73c214f5f1.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/eae5ec9b74107f9aebe21afaf543a81567d4c0277ff0182d757f0b73c214f5f1.jpg)

## OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference


### Images

![3d983b4f2486009bcc68ad84de2df61608346b0db4c2de1812add06fffa2e05f.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/3d983b4f2486009bcc68ad84de2df61608346b0db4c2de1812add06fffa2e05f.jpg)

![6c2deb043e8da4bf9ffb3fe7e891310b42d6b242b504b83bfe9d3b2f9ebf7d74.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/6c2deb043e8da4bf9ffb3fe7e891310b42d6b242b504b83bfe9d3b2f9ebf7d74.jpg)

![a85831988a1e8ac9dec3772714e1fd65fa0790593d007261a191a722e4f87e72.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/a85831988a1e8ac9dec3772714e1fd65fa0790593d007261a191a722e4f87e72.jpg)

![d80c372dbf2c27232098d32b318b62bd84ca1e6018e62168c33834c484be962f.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/d80c372dbf2c27232098d32b318b62bd84ca1e6018e62168c33834c484be962f.jpg)

![d8be3d1ba6bca510919bf6e205042e3ddfda6154a6fef44f320006eb76213679.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/d8be3d1ba6bca510919bf6e205042e3ddfda6154a6fef44f320006eb76213679.jpg)

![e096a6b51c6dfa697da7bd0f868c21715e8cc801599dc198337cdccc9097ee85.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/e096a6b51c6dfa697da7bd0f868c21715e8cc801599dc198337cdccc9097ee85.jpg)

![eea3136f389b6b4b8fc6305b6d27d26317a8f8adfd5ab967c6c029e50b9516f3.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/eea3136f389b6b4b8fc6305b6d27d26317a8f8adfd5ab967c6c029e50b9516f3.jpg)

![f7a779b00e1291f5a59c76c8703ed873bcd21b2d0c13e3ed1e7ae5d44ad7a8f7.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/images/f7a779b00e1291f5a59c76c8703ed873bcd21b2d0c13e3ed1e7ae5d44ad7a8f7.jpg)

### Tables

![0a722d02431568c96b825d4064b4964cefa0e4021687a65ad675a140dfbd9ad0.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/0a722d02431568c96b825d4064b4964cefa0e4021687a65ad675a140dfbd9ad0.jpg)

![130a34c7146301b0d70a23b1d39380bdd58503de87ac4732880200610587bd9e.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/130a34c7146301b0d70a23b1d39380bdd58503de87ac4732880200610587bd9e.jpg)

![1448bae0b7549bbfa52bd1569f8354d10cd0a4dfeac2ae077025e85636685cac.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/1448bae0b7549bbfa52bd1569f8354d10cd0a4dfeac2ae077025e85636685cac.jpg)

![18ac02701218de0508ad7e7b5ae89f0a3d0a94a42196ed83d9a64de1c2985ca1.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/18ac02701218de0508ad7e7b5ae89f0a3d0a94a42196ed83d9a64de1c2985ca1.jpg)

![4bf6e988ea0bf970e37be9317fa1e6873fe8ab620ff035167560fc08d4f7efe1.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/4bf6e988ea0bf970e37be9317fa1e6873fe8ab620ff035167560fc08d4f7efe1.jpg)

![5d48b6278f5ccb556b4c8d17a1acf5a5fd0dbfaee1365175052b0b37e62ca591.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/5d48b6278f5ccb556b4c8d17a1acf5a5fd0dbfaee1365175052b0b37e62ca591.jpg)

![72bab5384903333d9b39884568d4e8135c6bc12518316a299f5e0849957455d1.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/72bab5384903333d9b39884568d4e8135c6bc12518316a299f5e0849957455d1.jpg)

![d5728637c8359ae68fd04ec6db15a7903d4edee71a6b45c00a638c5ade4a54d0.jpg](../icml_results/1740_OBCache_%20Optimal%20Brain%20KV%20Cache%20Pruning%20for%20Efficient%20Long-Context%20LLM%20Inference/tables/d5728637c8359ae68fd04ec6db15a7903d4edee71a6b45c00a638c5ade4a54d0.jpg)

<a id="paper-1718"></a>

## Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation


### Images

![1af6fd465ac7864ac7c837a9e433be672d774c8373e6c6d98a80af58f3c28e3e.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/images/1af6fd465ac7864ac7c837a9e433be672d774c8373e6c6d98a80af58f3c28e3e.jpg)

![4c38a0b124a0268f540642bca830d07bb0a1137e3f31c69af6cbc2585ebe36d2.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/images/4c38a0b124a0268f540642bca830d07bb0a1137e3f31c69af6cbc2585ebe36d2.jpg)

![52f8f99479ff7dc75babe915a9f90098ab0a43b16dc99266d32aee393a5f1532.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/images/52f8f99479ff7dc75babe915a9f90098ab0a43b16dc99266d32aee393a5f1532.jpg)

![56d6524f26d59df63b8176f8958a6ec90aa2358d3147147b8dc5ec9d70f711e4.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/images/56d6524f26d59df63b8176f8958a6ec90aa2358d3147147b8dc5ec9d70f711e4.jpg)

![5aa254abd9b7b393027a2cdbe43c26a3f21a9571a31dad53c408ba9baa573495.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/images/5aa254abd9b7b393027a2cdbe43c26a3f21a9571a31dad53c408ba9baa573495.jpg)

![e0935851efc8be786a7f80442627826ab1474669b37d54b79258004247496d34.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/images/e0935851efc8be786a7f80442627826ab1474669b37d54b79258004247496d34.jpg)

### Tables

![07fe5e9f04a18d711b88f7e83e10212294be51e6649297979f8c7c4b51cbec10.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/07fe5e9f04a18d711b88f7e83e10212294be51e6649297979f8c7c4b51cbec10.jpg)

![18e223bfbed0f7ecdcc6d6d7e625487888f9e592806219529a3952607203214f.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/18e223bfbed0f7ecdcc6d6d7e625487888f9e592806219529a3952607203214f.jpg)

![4fc470c439915c5d28c0dd8f340ab070219a688078450c6294bd7e3ff41af0ee.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/4fc470c439915c5d28c0dd8f340ab070219a688078450c6294bd7e3ff41af0ee.jpg)

![5929202b91230c3c849ea11d68ae6878416a9df0e027212c800b6c0fb5ab8a93.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/5929202b91230c3c849ea11d68ae6878416a9df0e027212c800b6c0fb5ab8a93.jpg)

![61277d933f3c88e1ed40a089de13bd31fa23d3723513607d6eb1a5eee4f5cdf9.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/61277d933f3c88e1ed40a089de13bd31fa23d3723513607d6eb1a5eee4f5cdf9.jpg)

![6207be355e62d2c73fc438c3b5ba24705d99d2d1f8f91947f476fac488f72103.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/6207be355e62d2c73fc438c3b5ba24705d99d2d1f8f91947f476fac488f72103.jpg)

![65f78c8da0ad0b6fb516c33c7a4297414bd970fcc194b50e95b363cc599bc1e5.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/65f78c8da0ad0b6fb516c33c7a4297414bd970fcc194b50e95b363cc599bc1e5.jpg)

![6e8fc70c2a3e704be1ead67dca1fd2f04c11716a61cd393dff6157f23e8d7209.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/6e8fc70c2a3e704be1ead67dca1fd2f04c11716a61cd393dff6157f23e8d7209.jpg)

![725806badfc199c3e1568bb369eab989da62d694be174a908a94216d500cf7a9.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/725806badfc199c3e1568bb369eab989da62d694be174a908a94216d500cf7a9.jpg)

![a87273362cf6b84429caff7258e531b0ffade60971a85214b9e1f50bff821c29.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/a87273362cf6b84429caff7258e531b0ffade60971a85214b9e1f50bff821c29.jpg)

![ac3d01e2cda4f2ecc343e4409e05e4c8c6b3c69e42bb4bd315aab876e104d485.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/ac3d01e2cda4f2ecc343e4409e05e4c8c6b3c69e42bb4bd315aab876e104d485.jpg)

![f5ed33aa71c4fa32ddf90f6a7444278d8c61c5a574c9fc1168917c02e1ce0d41.jpg](../icml_results/1741_Spectral%20Imbalance%20Causes%20Forgetting%20in%20Low-Rank%20Continual%20Adaptation/tables/f5ed33aa71c4fa32ddf90f6a7444278d8c61c5a574c9fc1168917c02e1ce0d41.jpg)

<a id="paper-1719"></a>

## Precision-Induced Miscalibration: Understanding and Correcting Confidence Distortion in Quantized Neural Networks


### Images

![0d30b3027a973dd84d3ea2d6ebb3423ca5d04a00bebd4c1a916f99a46e24356c.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/0d30b3027a973dd84d3ea2d6ebb3423ca5d04a00bebd4c1a916f99a46e24356c.jpg)

![29f0189e8917d05f765d80152d4fb02014844127152363ca78f7a7e5f30b063c.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/29f0189e8917d05f765d80152d4fb02014844127152363ca78f7a7e5f30b063c.jpg)

![30daa1671975b0f545e095a7559364f1de324ddc9519f85caa2ed6d530c70199.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/30daa1671975b0f545e095a7559364f1de324ddc9519f85caa2ed6d530c70199.jpg)

![4af8654e4f2fcdc5ed1f010d83b00ff0a3145c673b9a1fcd8aaf519b04705cd3.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/4af8654e4f2fcdc5ed1f010d83b00ff0a3145c673b9a1fcd8aaf519b04705cd3.jpg)

![4ccdc2b0c80fc8488ad6268cf77978203b4e70117317ccf2b500db3c8a247145.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/4ccdc2b0c80fc8488ad6268cf77978203b4e70117317ccf2b500db3c8a247145.jpg)

![516640022e2fb3290218b7ed1196bc2bbc24ea677f4c99d70aef0391ad8c0616.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/516640022e2fb3290218b7ed1196bc2bbc24ea677f4c99d70aef0391ad8c0616.jpg)

![547fd6434aff905dbefe550f36b99c093b833c924509b677c3c2e495a6e273fc.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/547fd6434aff905dbefe550f36b99c093b833c924509b677c3c2e495a6e273fc.jpg)

![6fa225030f9e0916e220b4c54724d1c1e59e7dd6be233b3177424f7ec0ef7298.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/6fa225030f9e0916e220b4c54724d1c1e59e7dd6be233b3177424f7ec0ef7298.jpg)

![907c6b226b0be6241a8902984175779272dfd67845f3656cb09e51bc6455b826.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/907c6b226b0be6241a8902984175779272dfd67845f3656cb09e51bc6455b826.jpg)

![98ac96aba1cfef54a6103d6231e4f54a6dd95b52f452aa0beb0b191e355c8807.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/98ac96aba1cfef54a6103d6231e4f54a6dd95b52f452aa0beb0b191e355c8807.jpg)

![a992835c6d1d6e183f24dde2bfb22cfd7f43ee422ae618aabd96f564c371b835.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/a992835c6d1d6e183f24dde2bfb22cfd7f43ee422ae618aabd96f564c371b835.jpg)

![bf615c9957cc045dfd8918db6e9ae5c6189c1c9e8ac2a900c9b4efd709f5781a.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/bf615c9957cc045dfd8918db6e9ae5c6189c1c9e8ac2a900c9b4efd709f5781a.jpg)

![d7e9b0b657c0860ef724a46bb87d413059ca2b9f520881ae158a7f814ee0f516.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/d7e9b0b657c0860ef724a46bb87d413059ca2b9f520881ae158a7f814ee0f516.jpg)

![d929fa5c4e83984f10b5170bb7485d005e0f626a9083ae1c9ac122ea8b71b634.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/d929fa5c4e83984f10b5170bb7485d005e0f626a9083ae1c9ac122ea8b71b634.jpg)

![e2c1fca54ce9588acba61fdc43f5bf2e226c6eb41cbfc0a87eed65042ebd2fbe.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/e2c1fca54ce9588acba61fdc43f5bf2e226c6eb41cbfc0a87eed65042ebd2fbe.jpg)

![ec029ff7530aaf16459b29b0f593cc28db434b930bbcf6c41e757934f586ed67.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/images/ec029ff7530aaf16459b29b0f593cc28db434b930bbcf6c41e757934f586ed67.jpg)

### Tables

![0c94423d58804d65152de36c7c7d60bef9b0672256b68823f89b976be87a63de.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/0c94423d58804d65152de36c7c7d60bef9b0672256b68823f89b976be87a63de.jpg)

![0d486366403ccb746dc27a53823eafb2b1111ac45d75775d894b8437e49cee62.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/0d486366403ccb746dc27a53823eafb2b1111ac45d75775d894b8437e49cee62.jpg)

![12768b176e9daf469717ccdfbd8a0a6dfb12344cefc18d8637f1c1c12202262d.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/12768b176e9daf469717ccdfbd8a0a6dfb12344cefc18d8637f1c1c12202262d.jpg)

![18967e15cc7f169d27475096fa7e0d59c94890df77c44ef1f057d2b7b41e37cc.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/18967e15cc7f169d27475096fa7e0d59c94890df77c44ef1f057d2b7b41e37cc.jpg)

![20c4568a8b39e7f218389d7b19d053fa802e8fddb1195a15ef840f7c8210d250.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/20c4568a8b39e7f218389d7b19d053fa802e8fddb1195a15ef840f7c8210d250.jpg)

![41446c75533dac6b4d4ef22496dcbc5ddd39cf16ae017ee7124ceff083b9483a.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/41446c75533dac6b4d4ef22496dcbc5ddd39cf16ae017ee7124ceff083b9483a.jpg)

![4c155094ba8261266ded9271cb0105be0bda892e4c002ca67c43975a33adee2e.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/4c155094ba8261266ded9271cb0105be0bda892e4c002ca67c43975a33adee2e.jpg)

![5209a0a3a66795cdedb9bdd7f402a194b8d85db27ba14ec583ee2521f4c9cd6f.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/5209a0a3a66795cdedb9bdd7f402a194b8d85db27ba14ec583ee2521f4c9cd6f.jpg)

![52aab38502f82bce5c1d4a9ca59cf26ad6937e399b888413891c7804f0844741.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/52aab38502f82bce5c1d4a9ca59cf26ad6937e399b888413891c7804f0844741.jpg)

![7a165aa8d61d89fdd5dad4da2aab052166fea2a547e1809c5f378d6c1eb8ca87.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/7a165aa8d61d89fdd5dad4da2aab052166fea2a547e1809c5f378d6c1eb8ca87.jpg)

![88133a4045bb8845bc72bad341e2000a2907dfd83d866e9c4f014b4ca4497462.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/88133a4045bb8845bc72bad341e2000a2907dfd83d866e9c4f014b4ca4497462.jpg)

![89ae7a03c2e4a607df3813d9cc6d70ee1b52bc5aa00757c065ca18cde48fe035.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/89ae7a03c2e4a607df3813d9cc6d70ee1b52bc5aa00757c065ca18cde48fe035.jpg)

![99579d4715ee3a388079cea1f5f1aa2cd020d8021ab3e6ef7c747d369deada07.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/99579d4715ee3a388079cea1f5f1aa2cd020d8021ab3e6ef7c747d369deada07.jpg)

![9c980116ac19e7355216017d2ace818295ffa5b63a0c825cde9f54e38eccdbdf.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/9c980116ac19e7355216017d2ace818295ffa5b63a0c825cde9f54e38eccdbdf.jpg)

![b11c67de436bb95e0460d81bc3e0c92db406fca01ce7fc3c40c4e83499fd6a07.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/b11c67de436bb95e0460d81bc3e0c92db406fca01ce7fc3c40c4e83499fd6a07.jpg)

![c857849ea242671f5ab6d666579ba54f4393f683aa9583adfc3ee34743b7c20a.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/c857849ea242671f5ab6d666579ba54f4393f683aa9583adfc3ee34743b7c20a.jpg)

![d0350b50a3c55eadf08b2dbb453b830f17ad7659316fcad1e3a6fe0189bca079.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/d0350b50a3c55eadf08b2dbb453b830f17ad7659316fcad1e3a6fe0189bca079.jpg)

![d65bf6f81d3ac6869c847b8d3b475be6ce9ab48fd97852150b43b3b41896b9eb.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/d65bf6f81d3ac6869c847b8d3b475be6ce9ab48fd97852150b43b3b41896b9eb.jpg)

![fab23284335621313c5aabc043d9b1b8f4766c8792cc589d5103c914a4bdd154.jpg](../icml_results/1742_Precision-Induced%20Miscalibration_%20Understanding%20and%20Correcting%20Confidence%20Distortion%20in%20Quantized%20Ne/tables/fab23284335621313c5aabc043d9b1b8f4766c8792cc589d5103c914a4bdd154.jpg)

<a id="paper-1720"></a>

## Anytime-Valid Inference for Online Ranking of Large Language Models


### Images

![0360f48335b0d29b1b7fb3201ced488b3226ce7fe3bfa9c7f8d68769d838c53e.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/0360f48335b0d29b1b7fb3201ced488b3226ce7fe3bfa9c7f8d68769d838c53e.jpg)

![26e2da356b3a13168f0d0d682deee84b4895764e9e3336d5e7f6816025a3635c.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/26e2da356b3a13168f0d0d682deee84b4895764e9e3336d5e7f6816025a3635c.jpg)

![36efc8b18b0eb19de04fb21bd5655b7a87b3cea8163d3e46f534d87079d4ff33.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/36efc8b18b0eb19de04fb21bd5655b7a87b3cea8163d3e46f534d87079d4ff33.jpg)

![3d10cf7cb118a6cfefb78b68304cf3e9ff153a9852bf0623bfb67b39c90daa23.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/3d10cf7cb118a6cfefb78b68304cf3e9ff153a9852bf0623bfb67b39c90daa23.jpg)

![4cacee716e30bf6771acbd65841a2a37d816f3c87583bef4c11667560f7e0fc8.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/4cacee716e30bf6771acbd65841a2a37d816f3c87583bef4c11667560f7e0fc8.jpg)

![5524c0724056c1b6128f686494a59ed344df04d2bbe4c6928254396772bbbd27.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/5524c0724056c1b6128f686494a59ed344df04d2bbe4c6928254396772bbbd27.jpg)

![5b63bf2c3a6d33053cf7aac306d61dc25522f6ad353ddfcfc1de37df4f07f333.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/5b63bf2c3a6d33053cf7aac306d61dc25522f6ad353ddfcfc1de37df4f07f333.jpg)

![61f8518f80c596904608d5d6509e1e4e0cb1d44fc744f72f2d5e18cfaa922e32.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/61f8518f80c596904608d5d6509e1e4e0cb1d44fc744f72f2d5e18cfaa922e32.jpg)

![75b831e28a330c06dcd9239092d14ed8099626f22670b9ca3fb95f2b5018bd58.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/75b831e28a330c06dcd9239092d14ed8099626f22670b9ca3fb95f2b5018bd58.jpg)

![8fac54a2a98132249ca4014cf9345d8a5564aca698d7f5232fb7f7ebbaaa6c8d.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/8fac54a2a98132249ca4014cf9345d8a5564aca698d7f5232fb7f7ebbaaa6c8d.jpg)

![9dff4cb349859eff64f1045c5e15e4ce7873ec3531c29bc79e838d92fcd65537.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/9dff4cb349859eff64f1045c5e15e4ce7873ec3531c29bc79e838d92fcd65537.jpg)

![a38e409542934adb62432654142f7dd4408721181a990309387d2801250539b6.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/a38e409542934adb62432654142f7dd4408721181a990309387d2801250539b6.jpg)

![b630c96c4e9f01b489a4c0866b458047c19f299c294762a9372868f08c70ee82.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/b630c96c4e9f01b489a4c0866b458047c19f299c294762a9372868f08c70ee82.jpg)

![cc7a95f49505de37a4c233f70725cace3cc1f17bd46199a9b5828c5f5f9ab0ba.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/cc7a95f49505de37a4c233f70725cace3cc1f17bd46199a9b5828c5f5f9ab0ba.jpg)

![f3e86144753ca20d1c0476b5d1ede3550c5d5315f9c7aa8adc3b2c75cee07611.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/images/f3e86144753ca20d1c0476b5d1ede3550c5d5315f9c7aa8adc3b2c75cee07611.jpg)

### Tables

![353dbd376ec07f04f4a2f443d4ed62f49383da6b2ed3a335b93105a7ec6da693.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/353dbd376ec07f04f4a2f443d4ed62f49383da6b2ed3a335b93105a7ec6da693.jpg)

![50f807d6e4aba982dda86f61f6e37aecd4a8f52e6fbb5c7d9e5085718eed74c9.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/50f807d6e4aba982dda86f61f6e37aecd4a8f52e6fbb5c7d9e5085718eed74c9.jpg)

![5b62d9bd81d65f7606e64d0e03ff20f6db6ffd7c56f62dfc9e95422e127eb5c3.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/5b62d9bd81d65f7606e64d0e03ff20f6db6ffd7c56f62dfc9e95422e127eb5c3.jpg)

![5fb6732547fb752074d661514afe7ac11d7f9d82a289a02e6a79aaa61fa25bc3.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/5fb6732547fb752074d661514afe7ac11d7f9d82a289a02e6a79aaa61fa25bc3.jpg)

![646691140d439cdd3f81e1f66f53686209cd1173593139f1fd2f8dc829eae706.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/646691140d439cdd3f81e1f66f53686209cd1173593139f1fd2f8dc829eae706.jpg)

![867b3dd60e9f207aceba2871fb84ae40610c40fe1c6ad0010e5bfa7c6268c21e.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/867b3dd60e9f207aceba2871fb84ae40610c40fe1c6ad0010e5bfa7c6268c21e.jpg)

![a150c492b67dacbed41c1e1b03d8d4db12574f1bdc72c9d032dc461259064e16.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/a150c492b67dacbed41c1e1b03d8d4db12574f1bdc72c9d032dc461259064e16.jpg)

![b23387b4e67ed294d5823f15d8380663b3396849fad1e8ad0202d81416bd30c5.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/b23387b4e67ed294d5823f15d8380663b3396849fad1e8ad0202d81416bd30c5.jpg)

![d2fc699ea65fac20898e9507cb36dfdb6258d2300ee742692adfaadb59dac648.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/d2fc699ea65fac20898e9507cb36dfdb6258d2300ee742692adfaadb59dac648.jpg)

![d307d82ce564adcf13cba5bc39a3916b19df98e187215382c524d84e08b76dda.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/d307d82ce564adcf13cba5bc39a3916b19df98e187215382c524d84e08b76dda.jpg)

![e169bf0235e12688eec76de8896af8b29337e8aa4c64e414f40261636afc4b34.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/e169bf0235e12688eec76de8896af8b29337e8aa4c64e414f40261636afc4b34.jpg)

![f046ddac9431bb45da55b66e5ff74efab01440e007f7c3d0b4b47cd7568d31a4.jpg](../icml_results/1743_Anytime-Valid%20Inference%20for%20Online%20Ranking%20of%20Large%20Language%20Models/tables/f046ddac9431bb45da55b66e5ff74efab01440e007f7c3d0b4b47cd7568d31a4.jpg)

<a id="paper-1721"></a>

## R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations


### Images

![0c1c619ad402e7b8b85c6d26660a1380c01e3dff369cab0f5adda1cff84bb893.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/0c1c619ad402e7b8b85c6d26660a1380c01e3dff369cab0f5adda1cff84bb893.jpg)

![1903263826ff004699292aa1bbc39f0286968037f6657095c37718f135152f97.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/1903263826ff004699292aa1bbc39f0286968037f6657095c37718f135152f97.jpg)

![2a55fa29c87affe5576e10aa872e97dcced8be80dca936fa10e2572baadb71f9.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/2a55fa29c87affe5576e10aa872e97dcced8be80dca936fa10e2572baadb71f9.jpg)

![2bc9fbbdc3e891add1a65060e031b35102b0d15a15700ec550e01e4182f4769a.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/2bc9fbbdc3e891add1a65060e031b35102b0d15a15700ec550e01e4182f4769a.jpg)

![2d72afed5414f0e125ec886bbcab1d216776bdd31f2c1f2c372a038148e14d2c.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/2d72afed5414f0e125ec886bbcab1d216776bdd31f2c1f2c372a038148e14d2c.jpg)

![357425bee04115f452280fb0d72740291e68284b62e97744025a0b656ce209ba.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/357425bee04115f452280fb0d72740291e68284b62e97744025a0b656ce209ba.jpg)

![51f5de05664b41c431fc5c77d83b6b29f1c1233dc2be1480517e7a4093db817b.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/51f5de05664b41c431fc5c77d83b6b29f1c1233dc2be1480517e7a4093db817b.jpg)

![56400310c7a7303f98a0d0c94c8647e119bb9c275a7ae00d0d0b235479af9423.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/56400310c7a7303f98a0d0c94c8647e119bb9c275a7ae00d0d0b235479af9423.jpg)

![7bc054ba9f546b1c98ffdc16ab11826fa6be458cc366b97b142a1ef9c4a34493.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/7bc054ba9f546b1c98ffdc16ab11826fa6be458cc366b97b142a1ef9c4a34493.jpg)

![839774ee0eb8dbe9e11499aecc5bd6bdbfdfc406db787bf4d446506a850ee6a8.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/839774ee0eb8dbe9e11499aecc5bd6bdbfdfc406db787bf4d446506a850ee6a8.jpg)

![96d6bcd89c2742374322d8e1552f1edb662165cad6435f90ab89e0b1e5dbf5b4.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/96d6bcd89c2742374322d8e1552f1edb662165cad6435f90ab89e0b1e5dbf5b4.jpg)

![d4fed76e0a9a375def385b8ccf644a9a48e5a9dee1db469b0a7b965658d8f940.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/d4fed76e0a9a375def385b8ccf644a9a48e5a9dee1db469b0a7b965658d8f940.jpg)

![de4e51cef7cbcf919983626ee3dd5dfc52b57ccb23571cb8c2245b6f45e300c2.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/de4e51cef7cbcf919983626ee3dd5dfc52b57ccb23571cb8c2245b6f45e300c2.jpg)

![dfa5a61248f22722179a2d7b0b142a259c391f3376954784f840953d9ea5ec95.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/dfa5a61248f22722179a2d7b0b142a259c391f3376954784f840953d9ea5ec95.jpg)

![e430beeff73e5c6fe196983b09fccf40a3d8b8f8e86938f2e0ab24d8c151d21f.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/e430beeff73e5c6fe196983b09fccf40a3d8b8f8e86938f2e0ab24d8c151d21f.jpg)

![e4c9ca8fde554e2addc3b6c9e665ab3726fdb5d49ba42157be986db11bf74ba1.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/e4c9ca8fde554e2addc3b6c9e665ab3726fdb5d49ba42157be986db11bf74ba1.jpg)

![fd05bb5696655fcac6d90093e73f17d57bfcb6f139a3d35c0a7b7e9ebc76fab6.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/fd05bb5696655fcac6d90093e73f17d57bfcb6f139a3d35c0a7b7e9ebc76fab6.jpg)

![fe058ef829d0456eb6e4b8455a06605756c7edebdcd7aed9194996b601ec7b22.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/images/fe058ef829d0456eb6e4b8455a06605756c7edebdcd7aed9194996b601ec7b22.jpg)

### Tables

![05b0e162fd01907d6e6e3e50e60a429c1d29af3b2b2108c664bfc88c8237ecd4.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/05b0e162fd01907d6e6e3e50e60a429c1d29af3b2b2108c664bfc88c8237ecd4.jpg)

![1dc4179f923443956adef9e8fcb36ef8bb0996e22770cd43655dbd2b385d2502.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/1dc4179f923443956adef9e8fcb36ef8bb0996e22770cd43655dbd2b385d2502.jpg)

![2bfcd45d356fecb94e0994dbd542abdab5c22b23e740a6619df00c7566c5a8b8.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/2bfcd45d356fecb94e0994dbd542abdab5c22b23e740a6619df00c7566c5a8b8.jpg)

![4975d8ccd908b6685137f8fe1f57949aa0f307e2bd625dc38a40fff2539e3ba2.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/4975d8ccd908b6685137f8fe1f57949aa0f307e2bd625dc38a40fff2539e3ba2.jpg)

![7b986028bf63941700b9ef950eb7396a4d8b22f255fac1798dbb3e0b382d4118.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/7b986028bf63941700b9ef950eb7396a4d8b22f255fac1798dbb3e0b382d4118.jpg)

![884d6e0a4c80a6a9d3ab7213c808db674af1b0fc86b0e476548653fd4eca6380.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/884d6e0a4c80a6a9d3ab7213c808db674af1b0fc86b0e476548653fd4eca6380.jpg)

![90075a4684c35152b1aca7b940001d3ba921d53482990c149ff39d9e7b731ccf.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/90075a4684c35152b1aca7b940001d3ba921d53482990c149ff39d9e7b731ccf.jpg)

![91f35aef8f5eabd2f165cc2305836cd653c24213f70b57945b4e8f6ebf6c3193.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/91f35aef8f5eabd2f165cc2305836cd653c24213f70b57945b4e8f6ebf6c3193.jpg)

![a8d4df0e3ddde3ecfcdb9c6fa0acdc4cf1c454f3e48ed5d084eb667f9532b60c.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/a8d4df0e3ddde3ecfcdb9c6fa0acdc4cf1c454f3e48ed5d084eb667f9532b60c.jpg)

![aaaaa9617e3ed52ef5fb87a0ecbd92df74f26229605626b7fdd1a978dc1f16ea.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/aaaaa9617e3ed52ef5fb87a0ecbd92df74f26229605626b7fdd1a978dc1f16ea.jpg)

![b270a00d46d622cedc32d8f8bd565537f84fca57d080b38cb628b16eb3d633b6.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/b270a00d46d622cedc32d8f8bd565537f84fca57d080b38cb628b16eb3d633b6.jpg)

![b77bcbbce5a7088c6b5dfa5b2c2eeb0101303d5e2843180b03a5ba138aad4686.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/b77bcbbce5a7088c6b5dfa5b2c2eeb0101303d5e2843180b03a5ba138aad4686.jpg)

![deab92433d13301ef9e02b7175a1083da67f5837a489f0813dddf57ff0bc4334.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/deab92433d13301ef9e02b7175a1083da67f5837a489f0813dddf57ff0bc4334.jpg)

![dfdfe4517d64d0b135c0c65759075f4902b358b9b63a820cafdb23acaf8b19bd.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/dfdfe4517d64d0b135c0c65759075f4902b358b9b63a820cafdb23acaf8b19bd.jpg)

![fb1981d8e2afbc1223dbb222d95384cbf914b8a943761fb1210a79d55ae4aa85.jpg](../icml_results/1744_R%24%5E3%24L_%20Reasoning%203D%20Layouts%20from%20Relative%20Spatial%20Relations/tables/fb1981d8e2afbc1223dbb222d95384cbf914b8a943761fb1210a79d55ae4aa85.jpg)

<a id="paper-1722"></a>

## Parameter-Masked Decoupled Optimization for Cross-Domain Class-Incremental Learning


### Images

![20893b2790276c286ffcc1f87950f4b5cb1a6c8bf5b092ff9567b9e5e6910029.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/images/20893b2790276c286ffcc1f87950f4b5cb1a6c8bf5b092ff9567b9e5e6910029.jpg)

### Tables

![1d58d23c6b931a80f2cba4be57040620f3d7722561f7634f0f9750ebdc0e2e90.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/1d58d23c6b931a80f2cba4be57040620f3d7722561f7634f0f9750ebdc0e2e90.jpg)

![38b89339bdbfb8a590c54669f95cb07ccc84840f3791de1f4045e59a30f7b69c.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/38b89339bdbfb8a590c54669f95cb07ccc84840f3791de1f4045e59a30f7b69c.jpg)

![3ff1c369b2fdddf9670324bd050a54c89b770b70a9ae2d34a91efe258dbc72ab.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/3ff1c369b2fdddf9670324bd050a54c89b770b70a9ae2d34a91efe258dbc72ab.jpg)

![4793c1c92ce94277a58b8a4579c73fca9dfa94f5fbea917d791b7d70e1cc1109.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/4793c1c92ce94277a58b8a4579c73fca9dfa94f5fbea917d791b7d70e1cc1109.jpg)

![5d85f58b3f700df10933e5bc5834a63ef18a36eed460014ce60e6179a220585f.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/5d85f58b3f700df10933e5bc5834a63ef18a36eed460014ce60e6179a220585f.jpg)

![62f7051f12c05a45817cf8ce6f83e5b9fe8806a5c27e91b0beb47dc7204b9830.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/62f7051f12c05a45817cf8ce6f83e5b9fe8806a5c27e91b0beb47dc7204b9830.jpg)

![9312012ded3b3b725b4a640823b2528700f3b58b895246af66ffe9799cd1e799.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/9312012ded3b3b725b4a640823b2528700f3b58b895246af66ffe9799cd1e799.jpg)

![be5dce2b55539fce2be1a5b0fb8e0ffb892d8b9624f674f9497e29a49e1ed24e.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/be5dce2b55539fce2be1a5b0fb8e0ffb892d8b9624f674f9497e29a49e1ed24e.jpg)

![e9962bb7396dbffe35552c95326ee10a1ca08365c8996f387f6c10df5c4c646a.jpg](../icml_results/1745_Parameter-Masked%20Decoupled%20Optimization%20for%20Cross-Domain%20Class-Incremental%20Learning/tables/e9962bb7396dbffe35552c95326ee10a1ca08365c8996f387f6c10df5c4c646a.jpg)

<a id="paper-1723"></a>

## From Prompts to Responses: Dual-Sided Data Leakage and Defense in Split Large Language Models


### Images

![04f78671dd70cdf0b72c50ced758427d37bdc85bd97f4bdb7847305b765a7b0e.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/04f78671dd70cdf0b72c50ced758427d37bdc85bd97f4bdb7847305b765a7b0e.jpg)

![34c5cbf61d234f9d90af1ac03926cafb0bd6b1be19db4bfdc3f9b6a441f07fec.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/34c5cbf61d234f9d90af1ac03926cafb0bd6b1be19db4bfdc3f9b6a441f07fec.jpg)

![422ee04d26c2c1d7ea0aa290f66cd7f350261dab10d69e90b9313ef998a291c1.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/422ee04d26c2c1d7ea0aa290f66cd7f350261dab10d69e90b9313ef998a291c1.jpg)

![866a3c1ce1afb2542e63b58324ee4d63e723691b1542882d86c281fdb9bab549.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/866a3c1ce1afb2542e63b58324ee4d63e723691b1542882d86c281fdb9bab549.jpg)

![a6c17423df06ca24d4f6ac70cbeae470125fe8c95423af417bed3cc5053596bf.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/a6c17423df06ca24d4f6ac70cbeae470125fe8c95423af417bed3cc5053596bf.jpg)

![c4a076856fa5ce55cf570bf3c23684adb96cfa6c2de35bd1c1ac14d7a6bd5e8c.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/c4a076856fa5ce55cf570bf3c23684adb96cfa6c2de35bd1c1ac14d7a6bd5e8c.jpg)

![c9ae735b9d6c42f7cd977e75aa5db4a79db6effffeffe79632618f80e47384ec.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/c9ae735b9d6c42f7cd977e75aa5db4a79db6effffeffe79632618f80e47384ec.jpg)

![e83864b36670f83b2b8e8cc3e8b5c80122f63004ad3c361a78fda6fcb0286edb.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/e83864b36670f83b2b8e8cc3e8b5c80122f63004ad3c361a78fda6fcb0286edb.jpg)

![f2901e367aaa618700b026d32994b847c9a8e0d95ec7044ab9f959597d143d30.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/f2901e367aaa618700b026d32994b847c9a8e0d95ec7044ab9f959597d143d30.jpg)

![fbefdf40d55891774814c9aa922f6f99dda65c66e65f29f26755600a0fd21fdc.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/images/fbefdf40d55891774814c9aa922f6f99dda65c66e65f29f26755600a0fd21fdc.jpg)

### Tables

![15d1f2bb94838e4767f47707e965cd20c7242ddca5c00c1c68bd0babe5615fdc.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/15d1f2bb94838e4767f47707e965cd20c7242ddca5c00c1c68bd0babe5615fdc.jpg)

![275b118e58951321147114a5f5363a2b82315a9abf4e5392981de83f695b2f9a.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/275b118e58951321147114a5f5363a2b82315a9abf4e5392981de83f695b2f9a.jpg)

![322c9966273ccf170db8e88bde757df7b183d27e946cf4ef13236754b3513cc2.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/322c9966273ccf170db8e88bde757df7b183d27e946cf4ef13236754b3513cc2.jpg)

![444a7fdbff8981dea73eb614d333cb94190cdc4ce34e3c083dd59bc8bed38158.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/444a7fdbff8981dea73eb614d333cb94190cdc4ce34e3c083dd59bc8bed38158.jpg)

![5c197fc57ab34d874a62610178b3f718742a4d8d81bb41d6cbe897a9f61e9501.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/5c197fc57ab34d874a62610178b3f718742a4d8d81bb41d6cbe897a9f61e9501.jpg)

![7173fb3d1e7f6af38d5133e19d824e6f39675b3668d53ddd151c1b999e99b0c6.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/7173fb3d1e7f6af38d5133e19d824e6f39675b3668d53ddd151c1b999e99b0c6.jpg)

![735efbb5f01f558ab89f332714f11b346c4efab479f1ce931033ddc8eade29c1.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/735efbb5f01f558ab89f332714f11b346c4efab479f1ce931033ddc8eade29c1.jpg)

![89246d02a30ae4f35f2a5544ffd9a01ef19e45973d26cf058d3e555626c60fa0.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/89246d02a30ae4f35f2a5544ffd9a01ef19e45973d26cf058d3e555626c60fa0.jpg)

![8a0fab286b9d8eec54a810e2f0d07736dfcd5da81ac28e505b6d1fb40d5fc363.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/8a0fab286b9d8eec54a810e2f0d07736dfcd5da81ac28e505b6d1fb40d5fc363.jpg)

![8aa6206ab9c18dcce727f59d7171e79c92a31f76ac9c3c1abda99b456fcec5d3.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/8aa6206ab9c18dcce727f59d7171e79c92a31f76ac9c3c1abda99b456fcec5d3.jpg)

![9d7844e05471398a4346fe6e41d57cb1ba0c4c707161a3c4836420a67a074449.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/9d7844e05471398a4346fe6e41d57cb1ba0c4c707161a3c4836420a67a074449.jpg)

![a6c82593a5ab6b219297aeac9145997cceca2bddae549ddf887c56de0912ff58.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/a6c82593a5ab6b219297aeac9145997cceca2bddae549ddf887c56de0912ff58.jpg)

![ae2eedc4e5cbb1ceb73b87d809cab20ed4f0c28d9212c4bee41f8a7d2cf52c58.jpg](../icml_results/1746_From%20Prompts%20to%20Responses_%20Dual-Sided%20Data%20Leakage%20and%20Defense%20in%20Split%20Large%20Language%20Models/tables/ae2eedc4e5cbb1ceb73b87d809cab20ed4f0c28d9212c4bee41f8a7d2cf52c58.jpg)

<a id="paper-1724"></a>

## Power-Boosted Granger-Causal Discovery for Large Heterogeneous Panel Data


### Images

![0446ec49dfd8a3a9bced2c61c439423e31685f86950d0b9eb54002f659e514b1.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/images/0446ec49dfd8a3a9bced2c61c439423e31685f86950d0b9eb54002f659e514b1.jpg)

![0b68171aeca874864f1b9fd736604ee8b486e22e6a50302eff90d4d4a7b820f0.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/images/0b68171aeca874864f1b9fd736604ee8b486e22e6a50302eff90d4d4a7b820f0.jpg)

![3f228625c66112108f50f00a1bdb84f3565e69887dc80493f3103deb0feff372.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/images/3f228625c66112108f50f00a1bdb84f3565e69887dc80493f3103deb0feff372.jpg)

![e434b2c7a66a595aefc569e273604e89c73ef5ef25fbbe2eb80227bdeb78a8e7.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/images/e434b2c7a66a595aefc569e273604e89c73ef5ef25fbbe2eb80227bdeb78a8e7.jpg)

![f68499d1b4cde6e01b0fd1f055ad9248716125c465e613530f715ab0c6462f67.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/images/f68499d1b4cde6e01b0fd1f055ad9248716125c465e613530f715ab0c6462f67.jpg)

### Tables

![1138a4a4b9c54e6648b583aaa1a9dd6f1539b81e0663b65e7fad358e874f2150.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/1138a4a4b9c54e6648b583aaa1a9dd6f1539b81e0663b65e7fad358e874f2150.jpg)

![17d8c4a13f398cbf6afdbdd756cf5ebda6722ea65b5598d52867c6c691881edc.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/17d8c4a13f398cbf6afdbdd756cf5ebda6722ea65b5598d52867c6c691881edc.jpg)

![2d1bcad83da95f4bfbdeaa4f992fa5977cb051e70a0bf89a38271e285dbd2b98.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/2d1bcad83da95f4bfbdeaa4f992fa5977cb051e70a0bf89a38271e285dbd2b98.jpg)

![4ad47485ed5bdccc6d6e42af9c7eea616f3becfdc77d2b8e268de38e2b1f56c9.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/4ad47485ed5bdccc6d6e42af9c7eea616f3becfdc77d2b8e268de38e2b1f56c9.jpg)

![6259b159af536b7c3626ed9dc79ac45b8f6360b16d4e9cc4956f9fb33551fcad.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/6259b159af536b7c3626ed9dc79ac45b8f6360b16d4e9cc4956f9fb33551fcad.jpg)

![a8af76238078d59bf8d9146f8b1cd22ae4ea1747b9abdb107e5846ace5b68147.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/a8af76238078d59bf8d9146f8b1cd22ae4ea1747b9abdb107e5846ace5b68147.jpg)

![b475f181839209fc3739e48c4515fdba62605b38a2881c99fcacffd337edd961.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/b475f181839209fc3739e48c4515fdba62605b38a2881c99fcacffd337edd961.jpg)

![d1bed05400573194dea339db213d0c7adefe3634e4e1862fc92229fc358d3490.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/d1bed05400573194dea339db213d0c7adefe3634e4e1862fc92229fc358d3490.jpg)

![d7a65a95f0adf1770c2a21c2846d47e027ba332bcc3eee0bc9981f8ce908199e.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/d7a65a95f0adf1770c2a21c2846d47e027ba332bcc3eee0bc9981f8ce908199e.jpg)

![e19b98cac1eaafb3fb0109345ddbc32e4f9b6dce882b5eb1f1200c2ef5d69616.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/e19b98cac1eaafb3fb0109345ddbc32e4f9b6dce882b5eb1f1200c2ef5d69616.jpg)

![eac6dd28ee23465d5188d288504333601cfc747c8499db5a54c518db8823abd0.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/eac6dd28ee23465d5188d288504333601cfc747c8499db5a54c518db8823abd0.jpg)

![f380876a3ea93d92b06066112605fb915dcd2d8171c4acc1ab792a6effb96443.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/f380876a3ea93d92b06066112605fb915dcd2d8171c4acc1ab792a6effb96443.jpg)

![f8a3639a723d7e005b9460d17aca7ea0bfbf46b33941b35b0a86a250e006b0a0.jpg](../icml_results/1747_Power-Boosted%20Granger-Causal%20Discovery%20for%20Large%20Heterogeneous%20Panel%20Data/tables/f8a3639a723d7e005b9460d17aca7ea0bfbf46b33941b35b0a86a250e006b0a0.jpg)

<a id="paper-1725"></a>

## Ekka: Automated Diagnosis of Silent Errors in LLM Inference


### Images

![1473108311c961fdbc62991132bb3d841b7037e187d1c2dc927a4fd95a55ffd1.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/1473108311c961fdbc62991132bb3d841b7037e187d1c2dc927a4fd95a55ffd1.jpg)

![200485933874b919b1c4946d838156908ef7ea6c71a99006078271e6c63c75ab.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/200485933874b919b1c4946d838156908ef7ea6c71a99006078271e6c63c75ab.jpg)

![77873ebe8b775d4a1e497d65b16ef11a4a2989ce5269194799f1bd706d3a3072.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/77873ebe8b775d4a1e497d65b16ef11a4a2989ce5269194799f1bd706d3a3072.jpg)

![82c19cb942b1bd13e76995193a1d3dd5fac75ee95b4ceec9503d71b86e2e1930.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/82c19cb942b1bd13e76995193a1d3dd5fac75ee95b4ceec9503d71b86e2e1930.jpg)

![85edee926d733d809549634d92c85eaa962dab5facb5f67c6807883b253327e8.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/85edee926d733d809549634d92c85eaa962dab5facb5f67c6807883b253327e8.jpg)

![9d0ccb80b0068acfbe64d493f1fd026086dc508cb926a204780bdbddd3e05df0.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/9d0ccb80b0068acfbe64d493f1fd026086dc508cb926a204780bdbddd3e05df0.jpg)

![ac7fb9dea51d0756c9bc6f265ed70df4a762018cefd67377473c1ff64c3be784.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/ac7fb9dea51d0756c9bc6f265ed70df4a762018cefd67377473c1ff64c3be784.jpg)

![b24dcdb398d9f1b26eb48995294e50bf5637ac0325dd52b28482557368f06c6a.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/b24dcdb398d9f1b26eb48995294e50bf5637ac0325dd52b28482557368f06c6a.jpg)

![c61bf9d110603ca91a9251ae19bc4525d015b68ae521610a2e6a6e3c844f19ae.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/c61bf9d110603ca91a9251ae19bc4525d015b68ae521610a2e6a6e3c844f19ae.jpg)

![cbdd70816e6404e9aae4eed1a525fe71305d5eaf90e86a47b71871f88203c996.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/cbdd70816e6404e9aae4eed1a525fe71305d5eaf90e86a47b71871f88203c996.jpg)

![f0764197d6c047a0dbf0715e5e3aaf4a8817a66d1b5f945901d83345daf133de.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/images/f0764197d6c047a0dbf0715e5e3aaf4a8817a66d1b5f945901d83345daf133de.jpg)

### Tables

![03be5b8a7ea602bd437f717dde8529078d6ce0c731cdfb09e282bd66d6eb4938.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/03be5b8a7ea602bd437f717dde8529078d6ce0c731cdfb09e282bd66d6eb4938.jpg)

![31c6ad3fb92526af3e93030bde8c596af4bdd261ca43bac06d6abae87975c161.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/31c6ad3fb92526af3e93030bde8c596af4bdd261ca43bac06d6abae87975c161.jpg)

![4508605d5673b1ab3c59badedd6c639f9b67aa7fee4c7886ac5e029bf604a400.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/4508605d5673b1ab3c59badedd6c639f9b67aa7fee4c7886ac5e029bf604a400.jpg)

![5a6c9b7ab065a50fc9af8b17d8ad5d1ae4cdeb34ed6cd0c8fa05eb8b186b5029.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/5a6c9b7ab065a50fc9af8b17d8ad5d1ae4cdeb34ed6cd0c8fa05eb8b186b5029.jpg)

![612ae9b33f836e1c0c104d40adf7d995190a582d71ed89926ff6a54babcafbdb.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/612ae9b33f836e1c0c104d40adf7d995190a582d71ed89926ff6a54babcafbdb.jpg)

![b662a0352766acdc328d507a166da5df487f7d9592f93a26211507e1955be4ea.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/b662a0352766acdc328d507a166da5df487f7d9592f93a26211507e1955be4ea.jpg)

![f2db22e7f274853aa1727a1f0966e6547d9d88668f35acc48ab4f445a0aa1e33.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/f2db22e7f274853aa1727a1f0966e6547d9d88668f35acc48ab4f445a0aa1e33.jpg)

![f6c2c3bf7760e4cb80ac78639dc0891aad5e2d6d8fd368e079b6e7d94395fbc5.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/f6c2c3bf7760e4cb80ac78639dc0891aad5e2d6d8fd368e079b6e7d94395fbc5.jpg)

![fc88460c3914333fc78887cb4348aa81bec833f11771196d16625a3960fb1e3e.jpg](../icml_results/1748_Ekka_%20Automated%20Diagnosis%20of%20Silent%20Errors%20in%20LLM%20Inference/tables/fc88460c3914333fc78887cb4348aa81bec833f11771196d16625a3960fb1e3e.jpg)

<a id="paper-1726"></a>

## What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions


### Images

![065c260e181638b84081b0a9a993255b0c69f98c1f5415c1ce090e75c134906d.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/065c260e181638b84081b0a9a993255b0c69f98c1f5415c1ce090e75c134906d.jpg)

![0c2ffb8483694d2dbbacc143e518b74ad98f7651bd73a6c19c3b6c560021d300.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/0c2ffb8483694d2dbbacc143e518b74ad98f7651bd73a6c19c3b6c560021d300.jpg)

![22b6cf427fa83945d6659963a87d5ef09a58e3a88aecda301da46044167a4eb8.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/22b6cf427fa83945d6659963a87d5ef09a58e3a88aecda301da46044167a4eb8.jpg)

![6fbac63d936f15a0bc8663dd57dd94ff6ceb054f04b68990c38e90a1ddeac357.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/6fbac63d936f15a0bc8663dd57dd94ff6ceb054f04b68990c38e90a1ddeac357.jpg)

![859a1ef4ab77ee5453e6064d18b70f7e534b130f9cafb8619671e1441bb36c73.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/859a1ef4ab77ee5453e6064d18b70f7e534b130f9cafb8619671e1441bb36c73.jpg)

![8648f11694950a81fa066bca1e40faa41bef3c3efdf1d5aeeea85ddc490f8cab.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/8648f11694950a81fa066bca1e40faa41bef3c3efdf1d5aeeea85ddc490f8cab.jpg)

![89d0707d5791a2100b3a348c8d0073d52513a5d825e82389847fb7f945dc0b09.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/89d0707d5791a2100b3a348c8d0073d52513a5d825e82389847fb7f945dc0b09.jpg)

![9a578f2183da543c03342f14d8edd6e653bc43725f87e4ff983c111480a9d9de.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/9a578f2183da543c03342f14d8edd6e653bc43725f87e4ff983c111480a9d9de.jpg)

![b29606ce5188e0bb88c9a2134eff7596603d79c565a62af7cf1e9c9ebece8eae.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/b29606ce5188e0bb88c9a2134eff7596603d79c565a62af7cf1e9c9ebece8eae.jpg)

![cad94273005a789e444ed9f7b12e3881d8548e77b47a64295276ff435c51f83f.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/images/cad94273005a789e444ed9f7b12e3881d8548e77b47a64295276ff435c51f83f.jpg)

### Tables

![3f1c63c999cc5af62969790597e18a66d9286453eed3d20e11d8b1e1731b7c8d.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/3f1c63c999cc5af62969790597e18a66d9286453eed3d20e11d8b1e1731b7c8d.jpg)

![697e84f5b854a0493c1a48d7963cdc5ff3706b32e32227b0498bcf6798c92cf2.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/697e84f5b854a0493c1a48d7963cdc5ff3706b32e32227b0498bcf6798c92cf2.jpg)

![729ed54028d5bcbfc216b6f255517c78c271f28fd10bfb52e311cdadedfb2a7b.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/729ed54028d5bcbfc216b6f255517c78c271f28fd10bfb52e311cdadedfb2a7b.jpg)

![782881fa2e4f1121769049638e3f9651c3f9aa694373a22713b02285b43313fc.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/782881fa2e4f1121769049638e3f9651c3f9aa694373a22713b02285b43313fc.jpg)

![a95d5b0f950bdeec3999810034ea13c98ae8ca79a334f70320b40278f875810b.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/a95d5b0f950bdeec3999810034ea13c98ae8ca79a334f70320b40278f875810b.jpg)

![beb3dfdaf1118e0e24c1327313f2ee9eec91f33dd2fc0a54b554235e0ef21d7f.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/beb3dfdaf1118e0e24c1327313f2ee9eec91f33dd2fc0a54b554235e0ef21d7f.jpg)

![e45907c639d3958136b3d19144a00e4344ad64df412dc666eb231729e0f685cf.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/e45907c639d3958136b3d19144a00e4344ad64df412dc666eb231729e0f685cf.jpg)

![e46e5e5e0c3b60182892aa6f0d7dbdc615cef60ecc7e5b95f3d5750c77ca140f.jpg](../icml_results/1749_What%20if%20Tomorrow%20is%20the%20World%20Cup%20Final_%20Counterfactual%20Time%20Series%20Forecasting%20with%20Textual%20Conditi/tables/e46e5e5e0c3b60182892aa6f0d7dbdc615cef60ecc7e5b95f3d5750c77ca140f.jpg)

<a id="paper-1727"></a>

## VENOMREC: Cross-Modal Interactive Poisoning for Targeted Promotion in Multimodal LLM Recommender Systems


### Images

![225636509e28ad10d8a66a0a3393259935ec673e1f217d22feede01e1c4ea48a.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/images/225636509e28ad10d8a66a0a3393259935ec673e1f217d22feede01e1c4ea48a.jpg)

![8433efca9a50b00438d1c20b39c8b91843e823dac1dcc466a98d3763b268c0fa.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/images/8433efca9a50b00438d1c20b39c8b91843e823dac1dcc466a98d3763b268c0fa.jpg)

![d13f74a9a2185cfba62ba894c7833a0ebaf9199131c8cb83c0dbe34ebc1dfdd5.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/images/d13f74a9a2185cfba62ba894c7833a0ebaf9199131c8cb83c0dbe34ebc1dfdd5.jpg)

### Tables

![2c35576366abb84a8be702d079fac625460291b1cc000e825ba5d63b67181065.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/2c35576366abb84a8be702d079fac625460291b1cc000e825ba5d63b67181065.jpg)

![39bd86503fd522ecb2f73399bc03e078ac186049cb630a1f61b55856fa40cd49.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/39bd86503fd522ecb2f73399bc03e078ac186049cb630a1f61b55856fa40cd49.jpg)

![3a119d2d4f162ca5576128432f0f55145fcec68e4177e079d299c3e92bed6977.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/3a119d2d4f162ca5576128432f0f55145fcec68e4177e079d299c3e92bed6977.jpg)

![5631e4c188c4fcd6a8afba4e3b2b0943e7c9762319e7771b1cdd7b7ad889003a.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/5631e4c188c4fcd6a8afba4e3b2b0943e7c9762319e7771b1cdd7b7ad889003a.jpg)

![7472027f441d381e21cfc8fea5dd7fabe8347de22a37ba1fd0677d5a87207f04.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/7472027f441d381e21cfc8fea5dd7fabe8347de22a37ba1fd0677d5a87207f04.jpg)

![8c321eb00bbe949272c6de6cd352c2f0c0513ab03cabb16257a874d08b064d00.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/8c321eb00bbe949272c6de6cd352c2f0c0513ab03cabb16257a874d08b064d00.jpg)

![a0fb588a808002131ea4bda26dd147e4e5b887f2ace792e1b62e6b70440620f1.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/a0fb588a808002131ea4bda26dd147e4e5b887f2ace792e1b62e6b70440620f1.jpg)

![d4e10d092cc8b2097cec1a90239afaa604f2d305b3fd8a1ade6788b72fbbc70d.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/d4e10d092cc8b2097cec1a90239afaa604f2d305b3fd8a1ade6788b72fbbc70d.jpg)

![dd692b7cab86685bd40b8b1001152229fad3e228bc85c70207488a420b065963.jpg](../icml_results/1750_VENOMREC_%20Cross-Modal%20Interactive%20Poisoning%20for%20Targeted%20Promotion%20in%20Multimodal%20LLM%20Recommender%20Sys/tables/dd692b7cab86685bd40b8b1001152229fad3e228bc85c70207488a420b065963.jpg)

<a id="paper-1728"></a>

## From Content to Knowledge: Lightning Fast Long-Video Understanding with Neural Knowledge Representations


### Images

![1409bbfe1731e06ca1b93e687441455e43293aa83e728b030bb7010075379809.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/images/1409bbfe1731e06ca1b93e687441455e43293aa83e728b030bb7010075379809.jpg)

![141a37d8563bf47bc446a52b68b7aa87e8e02417d32fb0e6f79b96088289e6a9.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/images/141a37d8563bf47bc446a52b68b7aa87e8e02417d32fb0e6f79b96088289e6a9.jpg)

![296f11b338673970e66e2aac6232bb4c9f513f56ee6469821aab07bce7c11768.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/images/296f11b338673970e66e2aac6232bb4c9f513f56ee6469821aab07bce7c11768.jpg)

![a763d4cd6bfdcc8d5a3e1680b2b3200f40384c4dd3838202a9772ecd4ff7d966.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/images/a763d4cd6bfdcc8d5a3e1680b2b3200f40384c4dd3838202a9772ecd4ff7d966.jpg)

![e68083f07a3f71a3e19ab619e674dbb39dfabc77bb7b700172b31a4f45f5a537.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/images/e68083f07a3f71a3e19ab619e674dbb39dfabc77bb7b700172b31a4f45f5a537.jpg)

![fcedd6f3121fd0fb466fe90eeebbe1d829616276663f14221f48926548a01970.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/images/fcedd6f3121fd0fb466fe90eeebbe1d829616276663f14221f48926548a01970.jpg)

### Tables

![401ab74e9595881ffaef5d3f782e08c8a50831a889c420c5ad9a91097a92e416.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/401ab74e9595881ffaef5d3f782e08c8a50831a889c420c5ad9a91097a92e416.jpg)

![6866fb530ca7a1975af2a8288a2de0d1b2f4e1c495a935d6341d9e2b41a4869f.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/6866fb530ca7a1975af2a8288a2de0d1b2f4e1c495a935d6341d9e2b41a4869f.jpg)

![7053edf36eefe07ade90a9255fb0c4705aa67e694d325a2ce832fdf96e382073.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/7053edf36eefe07ade90a9255fb0c4705aa67e694d325a2ce832fdf96e382073.jpg)

![7a315125c18c97b96213c40de5004de055ee17e587413c6fba81ab497477b6d1.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/7a315125c18c97b96213c40de5004de055ee17e587413c6fba81ab497477b6d1.jpg)

![837381e8fd63d5a88518cfa31a300e33b4bc9ce96c4c324ecbb755f9816a9a00.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/837381e8fd63d5a88518cfa31a300e33b4bc9ce96c4c324ecbb755f9816a9a00.jpg)

![8f247fb7ba405d888854f56beeee4389418348c1fb5918fe7e1fdf9ac5fe8f22.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/8f247fb7ba405d888854f56beeee4389418348c1fb5918fe7e1fdf9ac5fe8f22.jpg)

![aab37cbdea103a0c8cf2db322970d0a4d557784aa636750d82fe7fb3a559fd41.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/aab37cbdea103a0c8cf2db322970d0a4d557784aa636750d82fe7fb3a559fd41.jpg)

![c00ddb8a8aaab39fc1edeac44193c2dfa7b4fcc7c227915a672fea10d1c9c864.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/c00ddb8a8aaab39fc1edeac44193c2dfa7b4fcc7c227915a672fea10d1c9c864.jpg)

![c22f801487076062c3f30b33560fb18860b5b88504b914163fa342e4165d17dd.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/c22f801487076062c3f30b33560fb18860b5b88504b914163fa342e4165d17dd.jpg)

![cf30ab75508e912f09e0342863d17ec42ad5908db0165ed9fcfebde8078ad241.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/cf30ab75508e912f09e0342863d17ec42ad5908db0165ed9fcfebde8078ad241.jpg)

![dfbc047aba57b08c6f468238e7cbdc967715d72dacade1241996b43072c21724.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/dfbc047aba57b08c6f468238e7cbdc967715d72dacade1241996b43072c21724.jpg)

![e56601b0a08d2693fcd76e8e311d6142b4a44eed544c653d6387ccbb4398086f.jpg](../icml_results/1751_From%20Content%20to%20Knowledge_%20Lightning%20Fast%20Long-Video%20Understanding%20with%20Neural%20Knowledge%20Representat/tables/e56601b0a08d2693fcd76e8e311d6142b4a44eed544c653d6387ccbb4398086f.jpg)

<a id="paper-1729"></a>

## TritonGym: A Benchmark for Agentic LLM Workflows in Triton GPU Code Generation


### Images

![0bb5c486764b7f1f0917aa0c51d07d6e7b86c29fcbc14be6a0417e8c9ef1241e.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/0bb5c486764b7f1f0917aa0c51d07d6e7b86c29fcbc14be6a0417e8c9ef1241e.jpg)

![399952f16544e86309f9c33ec4d3924d331303c191582eb311d23c027bb61e97.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/399952f16544e86309f9c33ec4d3924d331303c191582eb311d23c027bb61e97.jpg)

![3bdc40c5f5283275347058b3672e7a460cfbfb391f0e88257255b3fe5ac7b53d.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/3bdc40c5f5283275347058b3672e7a460cfbfb391f0e88257255b3fe5ac7b53d.jpg)

![58b484785f5cb4e20695d60f63a04a7cc8731bc942b307713fe55e64ecb769da.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/58b484785f5cb4e20695d60f63a04a7cc8731bc942b307713fe55e64ecb769da.jpg)

![85732a0435bd757cf8f39e33e158076d67b2f439d8a064b4251cfb1fe24b7168.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/85732a0435bd757cf8f39e33e158076d67b2f439d8a064b4251cfb1fe24b7168.jpg)

![c0d5948e3389b144c20144f6e0a64c2e7cd229e5ef004f7c5509336911e80754.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/c0d5948e3389b144c20144f6e0a64c2e7cd229e5ef004f7c5509336911e80754.jpg)

![dfb5a224b5b21492c4e65c314803b501b5d63f6256d2d17f825fdd08acaf28a9.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/images/dfb5a224b5b21492c4e65c314803b501b5d63f6256d2d17f825fdd08acaf28a9.jpg)

### Tables

![04f55c627f1f73c4ae3d3b1cfc9db5519909c699e9f4ca6460e25ba8796dd141.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/tables/04f55c627f1f73c4ae3d3b1cfc9db5519909c699e9f4ca6460e25ba8796dd141.jpg)

![280cd8d72eff51d04364a24ad82e518d778aab71b75216ee4e7f9a7dba9ccd33.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/tables/280cd8d72eff51d04364a24ad82e518d778aab71b75216ee4e7f9a7dba9ccd33.jpg)

![39ad0f53212dd230fd955ac08ed11924953c4a68d1680885e6d3ae6ac55d5e0f.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/tables/39ad0f53212dd230fd955ac08ed11924953c4a68d1680885e6d3ae6ac55d5e0f.jpg)

![a7b6d228db4a886d07c7a8bb5e95cd20f67db9b381771b521695d23c9ff78e40.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/tables/a7b6d228db4a886d07c7a8bb5e95cd20f67db9b381771b521695d23c9ff78e40.jpg)

![c43d06f983cbbad079651ea110a13e0484a0dbf1d2aeeb3a0c677b3453923f59.jpg](../icml_results/1752_TritonGym_%20A%20Benchmark%20for%20Agentic%20LLM%20Workflows%20in%20Triton%20GPU%20Code%20Generation/tables/c43d06f983cbbad079651ea110a13e0484a0dbf1d2aeeb3a0c677b3453923f59.jpg)

<a id="paper-1730"></a>

## TimeOmni-VL: Unified Models for Time Series Understanding and Generation


### Images

![001e784a1a4c916074ab87eb122632fa1ee57b0d297d56fdb1a7e727c167ac8d.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/001e784a1a4c916074ab87eb122632fa1ee57b0d297d56fdb1a7e727c167ac8d.jpg)

![0b6fc3af58e831e0886eabecd71f605c8feeab4f04049f51011cae2cc0270446.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/0b6fc3af58e831e0886eabecd71f605c8feeab4f04049f51011cae2cc0270446.jpg)

![23427245ed05bd217b389881cb8070a1b7c9eef274f5ab48fde7b6c06e400dec.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/23427245ed05bd217b389881cb8070a1b7c9eef274f5ab48fde7b6c06e400dec.jpg)

![35990ff2974ff84aededb4149eb728c4dbcf7e869c666f03a39bc85ddb193106.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/35990ff2974ff84aededb4149eb728c4dbcf7e869c666f03a39bc85ddb193106.jpg)

![50e565c85cf5d9b719e2d3b5c677d7ba0d9709f107a3f702899884c09277303e.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/50e565c85cf5d9b719e2d3b5c677d7ba0d9709f107a3f702899884c09277303e.jpg)

![516a832eeac80d34e4cc613bdeb2da20016d940d9f84760c4581fe4940c37dee.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/516a832eeac80d34e4cc613bdeb2da20016d940d9f84760c4581fe4940c37dee.jpg)

![531c6d49202f2318c77972ac6e0962b86054bdad055a9c079d271897c7d72ba1.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/531c6d49202f2318c77972ac6e0962b86054bdad055a9c079d271897c7d72ba1.jpg)

![55b88ffc811d0e5099d87d66302e1e5ad93a32b4337f8880c8d72dba0b968068.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/55b88ffc811d0e5099d87d66302e1e5ad93a32b4337f8880c8d72dba0b968068.jpg)

![5be661016ce2edd880ac464678c5b87c11119a5c9b4d38e43a9a201b00940a87.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/5be661016ce2edd880ac464678c5b87c11119a5c9b4d38e43a9a201b00940a87.jpg)

![5c01620e0c6661d49dc1005fc52b1761f706738519b08d2aba40c70148236830.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/5c01620e0c6661d49dc1005fc52b1761f706738519b08d2aba40c70148236830.jpg)

![6c1af99b7ea34cea5561477cde1342cfb9d0f8ff6f9f59eaee2fe642055273e1.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/6c1af99b7ea34cea5561477cde1342cfb9d0f8ff6f9f59eaee2fe642055273e1.jpg)

![7499f8c922e0ffc659beb30f8cc3a6b5ae0ab20fd654dd4c1205c2c3aa96064b.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/7499f8c922e0ffc659beb30f8cc3a6b5ae0ab20fd654dd4c1205c2c3aa96064b.jpg)

![75676d7233e4ce7e1c485fc436bc088e965955967ec0506bb3a319e14e9ee5cd.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/75676d7233e4ce7e1c485fc436bc088e965955967ec0506bb3a319e14e9ee5cd.jpg)

![79b886994a1bad4bae4f0837340bf447166c49b1a83deeb748c2591fa3b57c2e.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/79b886994a1bad4bae4f0837340bf447166c49b1a83deeb748c2591fa3b57c2e.jpg)

![87dfdf1e647b8537f814d965f9d9a030980bf926774591ae0b3e80f2d364a499.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/87dfdf1e647b8537f814d965f9d9a030980bf926774591ae0b3e80f2d364a499.jpg)

![87e41fa2f8eca1f3dfb4a7c79b12b10819a8b145d3f46db0cd4c34497b39c11f.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/87e41fa2f8eca1f3dfb4a7c79b12b10819a8b145d3f46db0cd4c34497b39c11f.jpg)

![9e377f06fcdf8b9ff8bf2c67d544af745d699294cfd3a98c9961b729dd586c3a.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/9e377f06fcdf8b9ff8bf2c67d544af745d699294cfd3a98c9961b729dd586c3a.jpg)

![b71bc0977dd74b4d10f9d6cd4a77a0754a2f5425158c358153d70248c6aceedd.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/b71bc0977dd74b4d10f9d6cd4a77a0754a2f5425158c358153d70248c6aceedd.jpg)

![c9402f6c9d969239469cdbd95eb00a963f5014bb7ad8cae1f3d471f1b433c9a0.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/c9402f6c9d969239469cdbd95eb00a963f5014bb7ad8cae1f3d471f1b433c9a0.jpg)

![d9db406af065df75f4aa6cfe1859d4d8aa2ad5e08064c4c3dac89cafa0f3c8c4.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/d9db406af065df75f4aa6cfe1859d4d8aa2ad5e08064c4c3dac89cafa0f3c8c4.jpg)

![da9c56bbc7cadab6e55c3bb727bdabeb795778d87580ca68ebbc04202b4fc964.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/da9c56bbc7cadab6e55c3bb727bdabeb795778d87580ca68ebbc04202b4fc964.jpg)

![e128846dbbf6bcd428515885cba4308a245525a38113845005107d4116bf6057.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/e128846dbbf6bcd428515885cba4308a245525a38113845005107d4116bf6057.jpg)

![e2202c8f4ad6c9ebd83b4abcffc1b151d287ee5da560b9e63bed8203bbc55c69.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/e2202c8f4ad6c9ebd83b4abcffc1b151d287ee5da560b9e63bed8203bbc55c69.jpg)

![e2e69ebe83893ba5a624a1dc63c9981846bcecaaf2af3712f1bbfab4000a5dbd.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/e2e69ebe83893ba5a624a1dc63c9981846bcecaaf2af3712f1bbfab4000a5dbd.jpg)

![e63346db8f519fde7c423accf8688ae89e0aed2a3be0f77d42597b7608e659a6.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/e63346db8f519fde7c423accf8688ae89e0aed2a3be0f77d42597b7608e659a6.jpg)

![e76b8570e829774ab156707fe9a4929bdc442ff04b547a09d15412fcea5ae151.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/e76b8570e829774ab156707fe9a4929bdc442ff04b547a09d15412fcea5ae151.jpg)

![eda09ca4b775ff79f08d2ec15f68fdf8ea2fbbd86098a1fe6cf22149003091cc.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/eda09ca4b775ff79f08d2ec15f68fdf8ea2fbbd86098a1fe6cf22149003091cc.jpg)

![f0c7721d42a417787ccfa940601fbc748217e52e68bb6b0a390130a164156b15.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/f0c7721d42a417787ccfa940601fbc748217e52e68bb6b0a390130a164156b15.jpg)

![f516b43761df9ffa89d67f5a27063cf25fe55c8ee16fbf474a908c2a2441de12.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/images/f516b43761df9ffa89d67f5a27063cf25fe55c8ee16fbf474a908c2a2441de12.jpg)

### Tables

![19f72a5f71a632e71b155cda5caa2ae32c6014430d5eaf925e4bdac674c4183e.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/19f72a5f71a632e71b155cda5caa2ae32c6014430d5eaf925e4bdac674c4183e.jpg)

![24fa96c0edbeb8539b674a73304ca08c3b0862fa7406aa5c08736a887eaebecb.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/24fa96c0edbeb8539b674a73304ca08c3b0862fa7406aa5c08736a887eaebecb.jpg)

![43eeea50fa26eb6f613a1012f70fafd6354dcb83cd48477eb487f3100094b683.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/43eeea50fa26eb6f613a1012f70fafd6354dcb83cd48477eb487f3100094b683.jpg)

![6c4026c5459da863526cf1b5b14743d402992a4cf9ca0594bbfa1755a525954b.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/6c4026c5459da863526cf1b5b14743d402992a4cf9ca0594bbfa1755a525954b.jpg)

![6cf77046fbb7a96388830c91019895373f7bd8201ce5dd8b9ce67ec4c193cde5.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/6cf77046fbb7a96388830c91019895373f7bd8201ce5dd8b9ce67ec4c193cde5.jpg)

![745a219f678bec4632d88885bc2f1eb083492a92b5f1a4d0e34d19687d0f960e.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/745a219f678bec4632d88885bc2f1eb083492a92b5f1a4d0e34d19687d0f960e.jpg)

![8b7887384a600811f2c244e5d9da6a4d152f8fa25a453abab541c705bf47ef25.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/8b7887384a600811f2c244e5d9da6a4d152f8fa25a453abab541c705bf47ef25.jpg)

![a7adbd38ac6837fbb12f627a892e443bff8e76730199493cd2b881ecb950ce2b.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/a7adbd38ac6837fbb12f627a892e443bff8e76730199493cd2b881ecb950ce2b.jpg)

![be4ee5589fbed3f51c9ece12b7feaf564766a572fc2aa5461c0b0a633ce22c94.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/be4ee5589fbed3f51c9ece12b7feaf564766a572fc2aa5461c0b0a633ce22c94.jpg)

![c06e5f6557b8856b89cecca4426f2e1a072da44114a0f5b7e321d1a872fce88d.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/c06e5f6557b8856b89cecca4426f2e1a072da44114a0f5b7e321d1a872fce88d.jpg)

![c4ed5575bf264e682d7dbfccdde7533676785d868c38fb472880557ca2ba9cc7.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/c4ed5575bf264e682d7dbfccdde7533676785d868c38fb472880557ca2ba9cc7.jpg)

![de3d39e782db45b960bd65f8851add67c444aebaccd3190e5099d89524d03db5.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/de3d39e782db45b960bd65f8851add67c444aebaccd3190e5099d89524d03db5.jpg)

![e60c7f2e8903001e39f677979e7f118abf568f4053c7395f917f5428b34834e7.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/e60c7f2e8903001e39f677979e7f118abf568f4053c7395f917f5428b34834e7.jpg)

![f9e69b3e7d752fd6caf1d89e294f6e63f9bc3b590d71d9a9a13f7038acfeaa66.jpg](../icml_results/1753_TimeOmni-VL_%20Unified%20Models%20for%20Time%20Series%20Understanding%20and%20Generation/tables/f9e69b3e7d752fd6caf1d89e294f6e63f9bc3b590d71d9a9a13f7038acfeaa66.jpg)

<a id="paper-1731"></a>

## LIMMT: Less Is More for Motion Tracking


### Images

![0ad7488e5c4d448223f3cf3bd59f997c7f9e15eeaa3d238672cb1d6c2ccfe127.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/0ad7488e5c4d448223f3cf3bd59f997c7f9e15eeaa3d238672cb1d6c2ccfe127.jpg)

![26e488cde77b4631b19d9709c0f0c467ad36a56582a7d6af6648aa37453f71bd.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/26e488cde77b4631b19d9709c0f0c467ad36a56582a7d6af6648aa37453f71bd.jpg)

![4f54f461bb37df63880b0f761d1e1abb681a08cac9ef52c258e6f0df76f9dc54.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/4f54f461bb37df63880b0f761d1e1abb681a08cac9ef52c258e6f0df76f9dc54.jpg)

![57fbd87486a4fa30e91afe1820a79b4b8c63d2742f946ba97e7d0a57c7363d9d.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/57fbd87486a4fa30e91afe1820a79b4b8c63d2742f946ba97e7d0a57c7363d9d.jpg)

![a4d38d0e18ebfbc2d008d399a4b00c35bb318eaff73a77e781f8212adbf2e6a4.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/a4d38d0e18ebfbc2d008d399a4b00c35bb318eaff73a77e781f8212adbf2e6a4.jpg)

![b19a01a164a299a089e320650fd75b3366f76aaa28c7e3abcc97481cba0a42ca.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/b19a01a164a299a089e320650fd75b3366f76aaa28c7e3abcc97481cba0a42ca.jpg)

![b21ccd374cbcadf60015bab705fd278c1f30baa7ce1343587d994b001a43d53f.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/b21ccd374cbcadf60015bab705fd278c1f30baa7ce1343587d994b001a43d53f.jpg)

![c5937f651942b095eab09180e6f3f9a0e4656a24a7cb7a22e9fd96c997bfef86.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/images/c5937f651942b095eab09180e6f3f9a0e4656a24a7cb7a22e9fd96c997bfef86.jpg)

### Tables

![10f91a70519671d007402dc3432383306288e3ea0a1282c18efee0d75148035b.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/10f91a70519671d007402dc3432383306288e3ea0a1282c18efee0d75148035b.jpg)

![1872e1fee4d78076eabdcfca3e4289762997bbf99be7882f4aa4a9d3304e6c79.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/1872e1fee4d78076eabdcfca3e4289762997bbf99be7882f4aa4a9d3304e6c79.jpg)

![3bce25a50e2ccedacfd6682c15a636ee2932710b79968ba6f6151c5f2c4664d7.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/3bce25a50e2ccedacfd6682c15a636ee2932710b79968ba6f6151c5f2c4664d7.jpg)

![69e792f84d0fbd7f19043d533d8679f0d0a4abb6c9df2957c75b98e7725b5853.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/69e792f84d0fbd7f19043d533d8679f0d0a4abb6c9df2957c75b98e7725b5853.jpg)

![758876af005352a40a36f97c5dca24d4f7dcacfeaf01b2b05cfb7f8ac0b43c6c.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/758876af005352a40a36f97c5dca24d4f7dcacfeaf01b2b05cfb7f8ac0b43c6c.jpg)

![8347ed87a2751dd49a591a3609c76596765ed78d6f5950bff549c09a3792728d.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/8347ed87a2751dd49a591a3609c76596765ed78d6f5950bff549c09a3792728d.jpg)

![8bd41a4b4cc87b090ef87d3a8158f38cdedfa944276133de161c7d6b7ffb9825.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/8bd41a4b4cc87b090ef87d3a8158f38cdedfa944276133de161c7d6b7ffb9825.jpg)

![b7e07be7730bd436ffaaf414aca04c4726ed0349e23d3c4e7487f1d751c59e63.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/b7e07be7730bd436ffaaf414aca04c4726ed0349e23d3c4e7487f1d751c59e63.jpg)

![ba2f0dfe75edfab8eccbf66dfcf7c341eba47f72aa1dfb859c48777f0daea7a2.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/ba2f0dfe75edfab8eccbf66dfcf7c341eba47f72aa1dfb859c48777f0daea7a2.jpg)

![d33948fa052eed94ee7071789339061a0160537bfb5ced7280f6b26186b3685c.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/d33948fa052eed94ee7071789339061a0160537bfb5ced7280f6b26186b3685c.jpg)

![d72f34ba04dc3198134169bf7ecea4b7daac8792bcf7cecadaafb7fe659a67fd.jpg](../icml_results/1754_LIMMT_%20Less%20Is%20More%20for%20Motion%20Tracking/tables/d72f34ba04dc3198134169bf7ecea4b7daac8792bcf7cecadaafb7fe659a67fd.jpg)

<a id="paper-1732"></a>

## Bridging Tokens and Geometry: Token-wise 3D Supervision for CAD Generation


### Images

![0615786ec478687a9e51764d63035d621fe89a6ea244fefb8d3f1e79bd70ba0b.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/0615786ec478687a9e51764d63035d621fe89a6ea244fefb8d3f1e79bd70ba0b.jpg)

![3e2263ea43d754c9c23835765f96aa003d220da8083861b68af69b01587f0971.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/3e2263ea43d754c9c23835765f96aa003d220da8083861b68af69b01587f0971.jpg)

![6098600a875f9a8cf576a8235ac13ebb42df0900b8986f48c058e95df261e413.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/6098600a875f9a8cf576a8235ac13ebb42df0900b8986f48c058e95df261e413.jpg)

![619a58d6905e304dfb92e6fece12a03de03b6feb98568c722a569892ae01793f.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/619a58d6905e304dfb92e6fece12a03de03b6feb98568c722a569892ae01793f.jpg)

![792183096dc4da82b65966ec123c4943727239aad3c8b308244024823e198acc.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/792183096dc4da82b65966ec123c4943727239aad3c8b308244024823e198acc.jpg)

![9934d4168799da148defcdfd9148439129816379680c88b98b8da2e1418f0645.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/9934d4168799da148defcdfd9148439129816379680c88b98b8da2e1418f0645.jpg)

![bc3e6ccc6286fa48a8e034731be9e87cd5850f74d0d22306e3081b9c14250fa8.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/bc3e6ccc6286fa48a8e034731be9e87cd5850f74d0d22306e3081b9c14250fa8.jpg)

![c5f190397f54b3dfd6e35f75d53e58170b9b795a44074272f23bf71496301e44.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/images/c5f190397f54b3dfd6e35f75d53e58170b9b795a44074272f23bf71496301e44.jpg)

### Tables

![13a5921332391aa7d29d087842b800667a7b45c222340db970f12accad14f412.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/tables/13a5921332391aa7d29d087842b800667a7b45c222340db970f12accad14f412.jpg)

![39ba48f73dc1eecd2b5f774f19dbcef8c8f4e796b787c404d3a566660ba888ac.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/tables/39ba48f73dc1eecd2b5f774f19dbcef8c8f4e796b787c404d3a566660ba888ac.jpg)

![41a28b3761469db15d3f41b9b7d944a37ba94e8086ae068fbbc50dfbfa1c73f0.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/tables/41a28b3761469db15d3f41b9b7d944a37ba94e8086ae068fbbc50dfbfa1c73f0.jpg)

![7d41077439bf58c7dce920d0bf35ae3120785360c557f655830a1a14bda37e82.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/tables/7d41077439bf58c7dce920d0bf35ae3120785360c557f655830a1a14bda37e82.jpg)

![c4ab067f9b6df198535334387825ec661b7df90656071a2519a2865d175b9800.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/tables/c4ab067f9b6df198535334387825ec661b7df90656071a2519a2865d175b9800.jpg)

![f56bb3092ca4a7f63473ced8643790e85e5c5811723ff1eeda601f50e21a4b17.jpg](../icml_results/1755_Bridging%20Tokens%20and%20Geometry_%20Token-wise%203D%20Supervision%20for%20CAD%20Generation/tables/f56bb3092ca4a7f63473ced8643790e85e5c5811723ff1eeda601f50e21a4b17.jpg)

<a id="paper-1733"></a>

## MOC: Multi-Order Communication in LLM-based Multi-Agent Systems


### Images

![26180399c080f81803e1c3d17cc21579a06a7c67caf68bcb19d9a401accd4b90.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/images/26180399c080f81803e1c3d17cc21579a06a7c67caf68bcb19d9a401accd4b90.jpg)

![38b2109bb5b45b0474f03dad0047bd090476dce861e553f333121562c4bc9b34.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/images/38b2109bb5b45b0474f03dad0047bd090476dce861e553f333121562c4bc9b34.jpg)

![accf0d5d664c5c81c0e2897d16b5fb2205fb32e406af66c346fd34ab3b063d77.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/images/accf0d5d664c5c81c0e2897d16b5fb2205fb32e406af66c346fd34ab3b063d77.jpg)

![b2856b6fef131dc3f65d15d3142f272ed766523c1ba7aa8ad77732c2282d2465.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/images/b2856b6fef131dc3f65d15d3142f272ed766523c1ba7aa8ad77732c2282d2465.jpg)

![d6a8818887f9cb18004f474fae3af0595e5fad0f297f345edda6317949563ff3.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/images/d6a8818887f9cb18004f474fae3af0595e5fad0f297f345edda6317949563ff3.jpg)

### Tables

![2e883c36318aab3067aeda94f475b8690e410d94a26e13c715a94d31f2a2838f.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/tables/2e883c36318aab3067aeda94f475b8690e410d94a26e13c715a94d31f2a2838f.jpg)

![4d96bf7f8cb2b978f37de4f9cc2290310d818c81ad478db2f54f82e20ed4f796.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/tables/4d96bf7f8cb2b978f37de4f9cc2290310d818c81ad478db2f54f82e20ed4f796.jpg)

![85db739d8d9970cf2e43354a6152252f12d258cf16053f5798f9eb83eac0d3eb.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/tables/85db739d8d9970cf2e43354a6152252f12d258cf16053f5798f9eb83eac0d3eb.jpg)

![aec1d28ad916b1458ed46527e2304043535b1d190fd5452fd8ac288f4e78ce9c.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/tables/aec1d28ad916b1458ed46527e2304043535b1d190fd5452fd8ac288f4e78ce9c.jpg)

![af9d2461da00b8b9a43a364df9a8e074f41bbe37b8d76e0af2693528bcc56f46.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/tables/af9d2461da00b8b9a43a364df9a8e074f41bbe37b8d76e0af2693528bcc56f46.jpg)

![d30e87e4b37d174c3fc78ce349137bc2539b5b5c12e6b15c90daf5137703a810.jpg](../icml_results/1756_MOC_%20Multi-Order%20Communication%20in%20LLM-based%20Multi-Agent%20Systems/tables/d30e87e4b37d174c3fc78ce349137bc2539b5b5c12e6b15c90daf5137703a810.jpg)

<a id="paper-1734"></a>

## Federated Multi-view Clustering for Remote Sensing Data


### Images

![24b8a91dd0fbbb75c9b274122636aefba59f720d3cf9547fb0dfda75ab76276b.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/images/24b8a91dd0fbbb75c9b274122636aefba59f720d3cf9547fb0dfda75ab76276b.jpg)

![6991fe9077764edc903e724430d4eba58cc3ae86d466d88b781e497e58272329.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/images/6991fe9077764edc903e724430d4eba58cc3ae86d466d88b781e497e58272329.jpg)

![97aaf59f7abeb02f4df5d048937e9aced5ea70d5e0f9665edb83b4588bdfec09.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/images/97aaf59f7abeb02f4df5d048937e9aced5ea70d5e0f9665edb83b4588bdfec09.jpg)

![d4129411084446156060f3c65b52e4b7b1e81c2346a4ee5fe4937bc902cb2535.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/images/d4129411084446156060f3c65b52e4b7b1e81c2346a4ee5fe4937bc902cb2535.jpg)

![d9ad158557701ff778e0a1cb8cf1b0dec4860a691e86a7538fe39aa776a8ee7f.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/images/d9ad158557701ff778e0a1cb8cf1b0dec4860a691e86a7538fe39aa776a8ee7f.jpg)

### Tables

![5ccd69def2708c1b765f205fbe89f4312f2c7337db521a63db3e340a6682f941.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/tables/5ccd69def2708c1b765f205fbe89f4312f2c7337db521a63db3e340a6682f941.jpg)

![8facdc5be4c1454bea1f86362ddb7aab9f13e3bc546bddb7ebf9941eed34e1ee.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/tables/8facdc5be4c1454bea1f86362ddb7aab9f13e3bc546bddb7ebf9941eed34e1ee.jpg)

![9652673f61728d58f0e4afc84e5e541c04e83ea91bef451d0dd5b44007d3b737.jpg](../icml_results/1757_Federated%20Multi-view%20Clustering%20for%20Remote%20Sensing%20Data/tables/9652673f61728d58f0e4afc84e5e541c04e83ea91bef451d0dd5b44007d3b737.jpg)

<a id="paper-1735"></a>

## On the Coordination of Value-Maximizing Bidders


### Images

![7c502032a771cb88b6875dd08d24a6c48be01792de930fb0bd313eaee383d601.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/images/7c502032a771cb88b6875dd08d24a6c48be01792de930fb0bd313eaee383d601.jpg)

![7fe6375683059433b23edbcc7e78a1d9da27c7fb38ba477b87308db3fd63e9e2.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/images/7fe6375683059433b23edbcc7e78a1d9da27c7fb38ba477b87308db3fd63e9e2.jpg)

![9407c8b266e0852c828187e8ae96c77d867f5b84daa55dd311e31107b8530e76.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/images/9407c8b266e0852c828187e8ae96c77d867f5b84daa55dd311e31107b8530e76.jpg)

![afa007db9898caeea5d7448fc3ca2f91d9a77893b0df3ccf99fab5952c44f24f.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/images/afa007db9898caeea5d7448fc3ca2f91d9a77893b0df3ccf99fab5952c44f24f.jpg)

![dd53364410a25b7b533534ad3c821f0502c3707ffbf97602fd707a9f9ae6e056.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/images/dd53364410a25b7b533534ad3c821f0502c3707ffbf97602fd707a9f9ae6e056.jpg)

![e04940a454db3c3d92ce3486b0abe7f8d09646998dce7a9325fbc4c55a2f48a5.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/images/e04940a454db3c3d92ce3486b0abe7f8d09646998dce7a9325fbc4c55a2f48a5.jpg)

### Tables

![536a5a7b3b5c14c0036cb1859da22b2ca54b932afef33d2bcf1b603049f9ae4d.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/tables/536a5a7b3b5c14c0036cb1859da22b2ca54b932afef33d2bcf1b603049f9ae4d.jpg)

![a69dd540a2ca07bdabd33b0429e0a3c94a322633c72ea55d970dbf7fefe57c7c.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/tables/a69dd540a2ca07bdabd33b0429e0a3c94a322633c72ea55d970dbf7fefe57c7c.jpg)

![eb21c2eb4580d8c3e80005f1c67095e90e16faa2c5b854fcb97fec82fb85a849.jpg](../icml_results/1758_On%20the%20Coordination%20of%20Value-Maximizing%20Bidders/tables/eb21c2eb4580d8c3e80005f1c67095e90e16faa2c5b854fcb97fec82fb85a849.jpg)

<a id="paper-1736"></a>

## ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models


### Images

![59270b9e70ad270a4f65fd1f99ea65c9713b251efc9065c7ebbd9a7fe92cc77e.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/59270b9e70ad270a4f65fd1f99ea65c9713b251efc9065c7ebbd9a7fe92cc77e.jpg)

![5cbf77f96af29f18cdf62803a3478077989153bcf942895c82ed23e0daf64037.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/5cbf77f96af29f18cdf62803a3478077989153bcf942895c82ed23e0daf64037.jpg)

![9c0833a57c06f56b91ad6d9d3c649c9fffdb7d034a21bbbe10d061055c7e81fb.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/9c0833a57c06f56b91ad6d9d3c649c9fffdb7d034a21bbbe10d061055c7e81fb.jpg)

![9ee80488dd8aedf4ccd42be177247406bcc2f94f2be579cb928f4a2491e01cda.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/9ee80488dd8aedf4ccd42be177247406bcc2f94f2be579cb928f4a2491e01cda.jpg)

![c57902637cee6c45abf6203d508dd5e388edd95da3c0b033db56201f2f009bfe.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/c57902637cee6c45abf6203d508dd5e388edd95da3c0b033db56201f2f009bfe.jpg)

![cd9cd983485881c2c1d386258bd419b6f6564b6a7fd959016d1a0c1561ba0bc7.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/cd9cd983485881c2c1d386258bd419b6f6564b6a7fd959016d1a0c1561ba0bc7.jpg)

![e49a93e36bc68ed8346aeeb17bf4e42e84b506e6f800f4522d2f5e456c6e5e27.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/e49a93e36bc68ed8346aeeb17bf4e42e84b506e6f800f4522d2f5e456c6e5e27.jpg)

![e6d5617ca9edabd85fe38b544cb8cf1abf0501963760ff01f04d84be3963cf24.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/images/e6d5617ca9edabd85fe38b544cb8cf1abf0501963760ff01f04d84be3963cf24.jpg)

### Tables

![000b89edd67f154ca94430b5a5cae56b789811c0291e16568e4928694d93e086.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/000b89edd67f154ca94430b5a5cae56b789811c0291e16568e4928694d93e086.jpg)

![0b9185ce739371428aa077de61c91f680abaae5f97f37ff4f15942518244a944.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/0b9185ce739371428aa077de61c91f680abaae5f97f37ff4f15942518244a944.jpg)

![20ec20e5afcc5c3798c363b4b705da2adf5e5a09402b86192d59e07133f76835.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/20ec20e5afcc5c3798c363b4b705da2adf5e5a09402b86192d59e07133f76835.jpg)

![2fa9a22a8da72219b13d1208cfa6a979607f55e2d9ac339ba61281e01dfbc4db.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/2fa9a22a8da72219b13d1208cfa6a979607f55e2d9ac339ba61281e01dfbc4db.jpg)

![3d1911bd88cbe926481d437cc1e4bbaa43c1c2bdf5c7019f54dcb7b1158035e6.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/3d1911bd88cbe926481d437cc1e4bbaa43c1c2bdf5c7019f54dcb7b1158035e6.jpg)

![4e0bf8e0c9968bebd59f1465b81e5051f7b8f3610c0dd6dce959555534e46baf.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/4e0bf8e0c9968bebd59f1465b81e5051f7b8f3610c0dd6dce959555534e46baf.jpg)

![68b17817ff507b23bec7033fb9917e90880c24c4b5f0951ef2b7f6d388c8fff4.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/68b17817ff507b23bec7033fb9917e90880c24c4b5f0951ef2b7f6d388c8fff4.jpg)

![85bd55f4d39f9a6d3b3b9cee18fbbaf118ef04c2b974e8088d3da54ed41168d7.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/85bd55f4d39f9a6d3b3b9cee18fbbaf118ef04c2b974e8088d3da54ed41168d7.jpg)

![93381db542e1191d397f20ea36d0f5d4bbb8dadf9157b27be0eed324bbf5d7d4.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/93381db542e1191d397f20ea36d0f5d4bbb8dadf9157b27be0eed324bbf5d7d4.jpg)

![95b267a4ae2ce474c40ed52d19af2254643a10dbd908fa98314301f40213eca8.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/95b267a4ae2ce474c40ed52d19af2254643a10dbd908fa98314301f40213eca8.jpg)

![9b74a052fbf8fbea4fe5ac2e9a9a505b3efea51ef701256b500ca560d7788689.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/9b74a052fbf8fbea4fe5ac2e9a9a505b3efea51ef701256b500ca560d7788689.jpg)

![a1f1a20d48dea87a7c5796717a6be773d1616fcaf6aaadfe6c9c704a5cb3e9a9.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/a1f1a20d48dea87a7c5796717a6be773d1616fcaf6aaadfe6c9c704a5cb3e9a9.jpg)

![a200d7d8ba3c338ce2ec069be3fb4f729cd65dbe504f978a778cc979c7a38846.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/a200d7d8ba3c338ce2ec069be3fb4f729cd65dbe504f978a778cc979c7a38846.jpg)

![a24695fda87096be973c896a8e1e5cfe9b9a94744b128611c1c1b66abdc8aaec.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/a24695fda87096be973c896a8e1e5cfe9b9a94744b128611c1c1b66abdc8aaec.jpg)

![c342f936fb35e5b5f7fab0131a9445a5dd2b9d01074cb302e5120b15a8ac56fc.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/c342f936fb35e5b5f7fab0131a9445a5dd2b9d01074cb302e5120b15a8ac56fc.jpg)

![d91e06b1a82626288297b00e3075354d19f2dc090a25d339cd45a43354dfdead.jpg](../icml_results/1759_ASRU_%20Activation%20Steering%20Meets%20Reinforcement%20Unlearning%20for%20Multimodal%20Large%20Language%20Models/tables/d91e06b1a82626288297b00e3075354d19f2dc090a25d339cd45a43354dfdead.jpg)

<a id="paper-1737"></a>

## Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization


### Images

![089dedbcd51e031c0ec32c34791a2c90edd29917da02cb21faa45a4bbfa0b203.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/images/089dedbcd51e031c0ec32c34791a2c90edd29917da02cb21faa45a4bbfa0b203.jpg)

![4bee674634515fbd16eff2b77eb16f4b2ea8ada2f71c1d5fd9b4bfe1e5ad2b77.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/images/4bee674634515fbd16eff2b77eb16f4b2ea8ada2f71c1d5fd9b4bfe1e5ad2b77.jpg)

![588dc746c41fbbe380db1cad93b13a0775e2d542b1a0d61d9356f0457479b275.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/images/588dc746c41fbbe380db1cad93b13a0775e2d542b1a0d61d9356f0457479b275.jpg)

![5d28e912876373e04e5aa3ba4fdef60b02df71ccad3b67aec77c4ce4f9d1614c.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/images/5d28e912876373e04e5aa3ba4fdef60b02df71ccad3b67aec77c4ce4f9d1614c.jpg)

![8fb139cb48568f8915c8fcf2ea094f9c48f30aebc2e5d86290c0d67c189a705d.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/images/8fb139cb48568f8915c8fcf2ea094f9c48f30aebc2e5d86290c0d67c189a705d.jpg)

![b23f6e1a6d37b784d4f0e95bd524d90c0093a67deec23142ebb2064251e8a9c8.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/images/b23f6e1a6d37b784d4f0e95bd524d90c0093a67deec23142ebb2064251e8a9c8.jpg)

### Tables

![7163cada8ad9d4a80f817bcf9e3c3bfc83e82d9ff1e98810997b44d446f84697.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/tables/7163cada8ad9d4a80f817bcf9e3c3bfc83e82d9ff1e98810997b44d446f84697.jpg)

![78ebbf79f647264a287eb5611f68478c44e004cabb2942c6e0f804cf6f714c50.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/tables/78ebbf79f647264a287eb5611f68478c44e004cabb2942c6e0f804cf6f714c50.jpg)

![8d7bee8c8c5180b188fff19591aa341255f3cfd153b73410ac2c74b16915d463.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/tables/8d7bee8c8c5180b188fff19591aa341255f3cfd153b73410ac2c74b16915d463.jpg)

![cef36f59745575726cab0b10b092a0ce942ecef6cbd0aaf18ce14b89a1f1ee92.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/tables/cef36f59745575726cab0b10b092a0ce942ecef6cbd0aaf18ce14b89a1f1ee92.jpg)

![e02480e3b446195b38085750e4a9a7cff0783150132406486c7fa409de186ec8.jpg](../icml_results/1760_Short%20Chains%2C%20Deep%20Thoughts_%20Balancing%20Reasoning%20Efficiency%20and%20Intra-Segment%20Capability%20via%20Split-M/tables/e02480e3b446195b38085750e4a9a7cff0783150132406486c7fa409de186ec8.jpg)

<a id="paper-1738"></a>

## Finite-Width Neural Tangent Kernels from Feynman Diagrams


### Images

![0203b98d5a8a35a649969dbe808b264cdb7ad9f66c494e889ec08db44d640b1f.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/0203b98d5a8a35a649969dbe808b264cdb7ad9f66c494e889ec08db44d640b1f.jpg)

![027f9761ae4cee76e0a68a674f02e72abe4e5872ff44fde75efda76d8a54754b.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/027f9761ae4cee76e0a68a674f02e72abe4e5872ff44fde75efda76d8a54754b.jpg)

![0aca04b562723e876256d0c0ed609d18cebe20ff7c5b74409076199fc1f9b10b.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/0aca04b562723e876256d0c0ed609d18cebe20ff7c5b74409076199fc1f9b10b.jpg)

![0b16747cc974488311291cb536b91baffca46e5d1e8e458dc89a7ab488b5ba6e.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/0b16747cc974488311291cb536b91baffca46e5d1e8e458dc89a7ab488b5ba6e.jpg)

![12fdb977e02f08f2b5d1ba7caa5dfc24eedde3feddc58bf8589db0e24ea48494.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/12fdb977e02f08f2b5d1ba7caa5dfc24eedde3feddc58bf8589db0e24ea48494.jpg)

![136375a301d71e1018cd967793092e68fe22eead413552a8b2167ce26124e162.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/136375a301d71e1018cd967793092e68fe22eead413552a8b2167ce26124e162.jpg)

![1b424d176ade715dc09e8fbbac5465656d2f6306448f9a16fee781ec84ab806f.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/1b424d176ade715dc09e8fbbac5465656d2f6306448f9a16fee781ec84ab806f.jpg)

![1ecbc22381b82072d5fac71ac31d2b77d86105135edd0d138c6a3fd99f5a16e3.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/1ecbc22381b82072d5fac71ac31d2b77d86105135edd0d138c6a3fd99f5a16e3.jpg)

![1eee2fe3e5a01ccb64949d174926ea628b37c4bd3c02f952132acfc2029e9171.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/1eee2fe3e5a01ccb64949d174926ea628b37c4bd3c02f952132acfc2029e9171.jpg)

![204f2e08b647b27517f03893d0e49a1855966bea4c056d43eaab16e0fa502104.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/204f2e08b647b27517f03893d0e49a1855966bea4c056d43eaab16e0fa502104.jpg)

![214f5715d083e20d1df5ea5b0636c27e3ca926cfc524eb26de4315e21bb06206.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/214f5715d083e20d1df5ea5b0636c27e3ca926cfc524eb26de4315e21bb06206.jpg)

![26b467f3f8243787901bb6c1a795ae1f4cc1ab3ac8cfeb61b50d7797267c715d.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/26b467f3f8243787901bb6c1a795ae1f4cc1ab3ac8cfeb61b50d7797267c715d.jpg)

![27f0b5689c6ef7d3d296561df3e338c6af25963835d5bb646bbc1b861a463702.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/27f0b5689c6ef7d3d296561df3e338c6af25963835d5bb646bbc1b861a463702.jpg)

![34f915bb252272375b9a0af6b7e3e3385647f57fba69fffa5a1597a0fd21c692.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/34f915bb252272375b9a0af6b7e3e3385647f57fba69fffa5a1597a0fd21c692.jpg)

![3aa5f02eb3a2e592d5613681c3ebef4c7457a1311cc129e048752ce8d17d5f89.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/3aa5f02eb3a2e592d5613681c3ebef4c7457a1311cc129e048752ce8d17d5f89.jpg)

![3d4034503c8e704a3ee9ac40385e6a1a78efc04591d20707be359cbd6c74db9c.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/3d4034503c8e704a3ee9ac40385e6a1a78efc04591d20707be359cbd6c74db9c.jpg)

![42ee956f4ea00b8866cf4a78abade1c5816939ba37764983889b864b64570a97.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/42ee956f4ea00b8866cf4a78abade1c5816939ba37764983889b864b64570a97.jpg)

![535bb6e5c4af9b16e2c94dc85a2b1d6e5c993e0b50cb77202b91900a585aa739.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/535bb6e5c4af9b16e2c94dc85a2b1d6e5c993e0b50cb77202b91900a585aa739.jpg)

![56e46a25c3c1c653d5cf1726be712e7b7ba1aeebcf9592f8d69eb6faa7c42bf0.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/56e46a25c3c1c653d5cf1726be712e7b7ba1aeebcf9592f8d69eb6faa7c42bf0.jpg)

![5877cb1835300a65e5d286d98879812609746c0c96e185d2f80061bed281debc.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/5877cb1835300a65e5d286d98879812609746c0c96e185d2f80061bed281debc.jpg)

![62184aca83942351450fd39c65f5fbd1d8ec6dc371d09ea35145f94738a1a39f.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/62184aca83942351450fd39c65f5fbd1d8ec6dc371d09ea35145f94738a1a39f.jpg)

![66c17d2d8a5fbd6d39b5eace1559cb8fc52ffea87b29b4f9fcb98f7be30ea302.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/66c17d2d8a5fbd6d39b5eace1559cb8fc52ffea87b29b4f9fcb98f7be30ea302.jpg)

![70b50e11b193da836cf59294a8ec7b5e6c0a5ea8457ab0873ea09fb32025e19b.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/70b50e11b193da836cf59294a8ec7b5e6c0a5ea8457ab0873ea09fb32025e19b.jpg)

![7366cb1990902de356d50554f859edd1cf3bc9cb0ce2fa832300025d73fdf03f.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/7366cb1990902de356d50554f859edd1cf3bc9cb0ce2fa832300025d73fdf03f.jpg)

![7ae983ca63a6c9f41cb2e70e823deb56b7c683e0891928ab1ca26aa4196f276e.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/7ae983ca63a6c9f41cb2e70e823deb56b7c683e0891928ab1ca26aa4196f276e.jpg)

![96f4ca0f7805efc9ff57c00f8a27d7ad3b9c811c9905027e9a6d29a316a05760.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/96f4ca0f7805efc9ff57c00f8a27d7ad3b9c811c9905027e9a6d29a316a05760.jpg)

![a8e0ea3adbb62c6021ce1d522407d09476c80e058f716b204036cccc3e799764.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/a8e0ea3adbb62c6021ce1d522407d09476c80e058f716b204036cccc3e799764.jpg)

![a972894e6a545e0ef72415c9fbe5ac30ff5229627876560346859ae4823c3f8e.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/a972894e6a545e0ef72415c9fbe5ac30ff5229627876560346859ae4823c3f8e.jpg)

![c3b5611f742994db996a853b11f67a9ed016143d902ffd2e6879ff1347b28b4c.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/c3b5611f742994db996a853b11f67a9ed016143d902ffd2e6879ff1347b28b4c.jpg)

![cdd35770afd3dfe7aecc844ca1255c220061aad0da9039b8b44065087fe5a9f2.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/cdd35770afd3dfe7aecc844ca1255c220061aad0da9039b8b44065087fe5a9f2.jpg)

![cf390ecc1ab781c7319b6451134324c79566dc4df4f060795d5d6640bd6e1c5d.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/cf390ecc1ab781c7319b6451134324c79566dc4df4f060795d5d6640bd6e1c5d.jpg)

![cf951403e85e30d18400ed437828effc022781e6bcf78d3efac686127ab962de.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/cf951403e85e30d18400ed437828effc022781e6bcf78d3efac686127ab962de.jpg)

![d6bc5e69bd283916059c430ea68caa0867c5fe45e8878b23e848625793ba6a7a.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/d6bc5e69bd283916059c430ea68caa0867c5fe45e8878b23e848625793ba6a7a.jpg)

![dbc3a8db2949b84b8cd3121c791658c51ce49f0f83e6f1bc5bf3812e9359a16a.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/dbc3a8db2949b84b8cd3121c791658c51ce49f0f83e6f1bc5bf3812e9359a16a.jpg)

![f0187b46a481b01c702d0fa29c73dcdd201f16d0613ea01e48dfdd309614beb1.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/f0187b46a481b01c702d0fa29c73dcdd201f16d0613ea01e48dfdd309614beb1.jpg)

![fd1ae914d33c2cc4d3102d76c20fc49ac8e6ad4b7990384b6bbb6eb602d07866.jpg](../icml_results/1761_Finite-Width%20Neural%20Tangent%20Kernels%20from%20Feynman%20Diagrams/images/fd1ae914d33c2cc4d3102d76c20fc49ac8e6ad4b7990384b6bbb6eb602d07866.jpg)

<a id="paper-1739"></a>

## MASH: Modeling Abstention via Selective Help-Seeking


### Images

![32cb1ebc39a91ae9d63c9e73f949c1d347e2d70a24656e2302f767e13b4f7888.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/images/32cb1ebc39a91ae9d63c9e73f949c1d347e2d70a24656e2302f767e13b4f7888.jpg)

![65eb7779e894506871088bdedf1004ad1864329033e1db71b5ed56c09f45ff61.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/images/65eb7779e894506871088bdedf1004ad1864329033e1db71b5ed56c09f45ff61.jpg)

![fddd30616ffe0721fb26e09fd9f1e21e60c09011f1d40f0a1e112354de7ddedc.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/images/fddd30616ffe0721fb26e09fd9f1e21e60c09011f1d40f0a1e112354de7ddedc.jpg)

### Tables

![00931546d32d4e5f28dd5a28b22382aea66d9b592591d048bb66284d7d19b35d.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/00931546d32d4e5f28dd5a28b22382aea66d9b592591d048bb66284d7d19b35d.jpg)

![0660a4127182a22c64a914226afdfbdd6767a2f247b095b4f5d020241addf4ef.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/0660a4127182a22c64a914226afdfbdd6767a2f247b095b4f5d020241addf4ef.jpg)

![29fe1d904005b55a0be8fc2eff0180efb6165a975e06ae94841bea15f1619617.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/29fe1d904005b55a0be8fc2eff0180efb6165a975e06ae94841bea15f1619617.jpg)

![39bd9c243f9cf2afc6337ea5c2bf8c69579d36f804154c7b1c6503bbb83095bd.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/39bd9c243f9cf2afc6337ea5c2bf8c69579d36f804154c7b1c6503bbb83095bd.jpg)

![3ca3be17571fa5e09e0af90a64217c392f7f4f8b96e30e722eefb5d1607ba583.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/3ca3be17571fa5e09e0af90a64217c392f7f4f8b96e30e722eefb5d1607ba583.jpg)

![41b89446463f6462e2c771548f5d96745c6dc3bb38bea9f8dcad10097c396be3.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/41b89446463f6462e2c771548f5d96745c6dc3bb38bea9f8dcad10097c396be3.jpg)

![42326dbde3e35fe3bf033e496db9b2ab35927f005443dfc964b62d6b8f769890.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/42326dbde3e35fe3bf033e496db9b2ab35927f005443dfc964b62d6b8f769890.jpg)

![45b7c328a53e3144d5e695138d891b5a4ee9fa7e8dbf6f535989337592fafaad.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/45b7c328a53e3144d5e695138d891b5a4ee9fa7e8dbf6f535989337592fafaad.jpg)

![520d9023ed4c2e78a38637545804901721f2dc44b6bbb7ab84eff614cd98cbd9.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/520d9023ed4c2e78a38637545804901721f2dc44b6bbb7ab84eff614cd98cbd9.jpg)

![654a9fbbb4f8755335b11962bddfdd1c99a0b49fe8ae305cd253f467f5cd7902.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/654a9fbbb4f8755335b11962bddfdd1c99a0b49fe8ae305cd253f467f5cd7902.jpg)

![67f496fc63cc9f1907581b2b941a1e4cfdd6c24701ea0c631990cb0ff0480375.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/67f496fc63cc9f1907581b2b941a1e4cfdd6c24701ea0c631990cb0ff0480375.jpg)

![7c5d070d31225ad818483384636f64ba1b0e9d1526532442c97435a656e67ccf.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/7c5d070d31225ad818483384636f64ba1b0e9d1526532442c97435a656e67ccf.jpg)

![a495c63a041c7e8a0e03b1001a330cf41b22ed6d13f9f46b14f17c7209b35845.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/a495c63a041c7e8a0e03b1001a330cf41b22ed6d13f9f46b14f17c7209b35845.jpg)

![b1be796b931c0dd9fb8423e36d42e31fa718a54e81520db60c8a0d9868a64f5a.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/b1be796b931c0dd9fb8423e36d42e31fa718a54e81520db60c8a0d9868a64f5a.jpg)

![bcb8d11b29f29245b2f6e1bd0bbe677702e635fc909ee06d064b7193ef79b2c6.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/bcb8d11b29f29245b2f6e1bd0bbe677702e635fc909ee06d064b7193ef79b2c6.jpg)

![cde0fa5ad3fb296660cc678cb1e90ccb1034acdb3ebf73fb7c5585ad19d0437b.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/cde0fa5ad3fb296660cc678cb1e90ccb1034acdb3ebf73fb7c5585ad19d0437b.jpg)

![d9319e705f3209a9168b378d2cc38d8db53d4b53e428e029ff3b49d594fbfdd7.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/d9319e705f3209a9168b378d2cc38d8db53d4b53e428e029ff3b49d594fbfdd7.jpg)

![d9b428ecc66d8d8ea0f9f830bfc6f9e02612ef02efe4253fb332bcc904d01c0a.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/d9b428ecc66d8d8ea0f9f830bfc6f9e02612ef02efe4253fb332bcc904d01c0a.jpg)

![dea03d79960780524a72d85ad63c5da10342245350565abf8e623f3e165a0132.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/dea03d79960780524a72d85ad63c5da10342245350565abf8e623f3e165a0132.jpg)

![f4532d03cb37721054f0055db01f3451646028f3158ef79365b985eba41efc03.jpg](../icml_results/1762_MASH_%20Modeling%20Abstention%20via%20Selective%20Help-Seeking/tables/f4532d03cb37721054f0055db01f3451646028f3158ef79365b985eba41efc03.jpg)

<a id="paper-1740"></a>

## PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs


### Images

![6976912573c8954e2e439b24efb62d33dbdf7c024d96af98816460bfabaf3cdc.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/images/6976912573c8954e2e439b24efb62d33dbdf7c024d96af98816460bfabaf3cdc.jpg)

![6bcc90b0d1b67e2425e8f707b5bf3e1d38d816162a5ee23862ee84d6ebb7f96b.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/images/6bcc90b0d1b67e2425e8f707b5bf3e1d38d816162a5ee23862ee84d6ebb7f96b.jpg)

![717bb75b06e8ace425db16ca3ba872ec78b2f4f5bb6cc1c156f1f061695ce16e.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/images/717bb75b06e8ace425db16ca3ba872ec78b2f4f5bb6cc1c156f1f061695ce16e.jpg)

![a7a9b989c52ccfeb11d8bf8ba087c1e6408fe3d5e868edec3a468e0c00dc0936.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/images/a7a9b989c52ccfeb11d8bf8ba087c1e6408fe3d5e868edec3a468e0c00dc0936.jpg)

![c7c66a3329fd042fbd7a36e2e7790d3a714550eed174581d8e2c5feb0fd783c5.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/images/c7c66a3329fd042fbd7a36e2e7790d3a714550eed174581d8e2c5feb0fd783c5.jpg)

![d88e180a88f4b844e1098b5d98f6530ecd604a3648d7a63a6c55250b4212b24e.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/images/d88e180a88f4b844e1098b5d98f6530ecd604a3648d7a63a6c55250b4212b24e.jpg)

### Tables

![0e3fdbf1c190a981b1c2def3536d6b82eb1b3df99c4fe43651ce6397c5043962.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/0e3fdbf1c190a981b1c2def3536d6b82eb1b3df99c4fe43651ce6397c5043962.jpg)

![314bb2d0fd34132766048bb1ac8f002d82dc0f2ad2c2260e0cf8b11329d56713.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/314bb2d0fd34132766048bb1ac8f002d82dc0f2ad2c2260e0cf8b11329d56713.jpg)

![31dc22bd8b68c6d22a9ac32d8ef243a72a990ba4f50115ad2c6e862ddcd2a964.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/31dc22bd8b68c6d22a9ac32d8ef243a72a990ba4f50115ad2c6e862ddcd2a964.jpg)

![3e866e907f91e89ae1031cc478d87d593608f22ef9e3f0f2b0d09ae7f55bfc99.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/3e866e907f91e89ae1031cc478d87d593608f22ef9e3f0f2b0d09ae7f55bfc99.jpg)

![465e30a6574ae2bcc6e57c0882f01dd4202b3be9a6a5dc9d0da11a60382ba581.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/465e30a6574ae2bcc6e57c0882f01dd4202b3be9a6a5dc9d0da11a60382ba581.jpg)

![5bd7a59b44c566620f78e115296ed055267d3dfa28f389376b1f9463d2ef3749.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/5bd7a59b44c566620f78e115296ed055267d3dfa28f389376b1f9463d2ef3749.jpg)

![667fcd993df77132b1fa01b95ec4a84a077cc4d6449018bb344d341cb724e6f3.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/667fcd993df77132b1fa01b95ec4a84a077cc4d6449018bb344d341cb724e6f3.jpg)

![73592b40ca0c9648a681913815c411d9a59c4a30eb90004cfd37532f90ebacf2.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/73592b40ca0c9648a681913815c411d9a59c4a30eb90004cfd37532f90ebacf2.jpg)

![7633acd5a554dc716f56794a9b242fadaf905e81887beb437156aa2ba6949dcb.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/7633acd5a554dc716f56794a9b242fadaf905e81887beb437156aa2ba6949dcb.jpg)

![803fa9bae709ac6a8459b54e178929961f69cfc894702f8f1e009f937c037705.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/803fa9bae709ac6a8459b54e178929961f69cfc894702f8f1e009f937c037705.jpg)

![8298c2e7745cd35e803c252b500d4340c13b006dd6e6411e4eb95086f51222d5.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/8298c2e7745cd35e803c252b500d4340c13b006dd6e6411e4eb95086f51222d5.jpg)

![880aa6086380d0e43c25e7dff2670e7728965c6d3a9d0f370fc2302fa13eb84b.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/880aa6086380d0e43c25e7dff2670e7728965c6d3a9d0f370fc2302fa13eb84b.jpg)

![90e10861193dd64808c89dfd32dd2f0ed2b51cd4b962690977769e4a6b096c7e.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/90e10861193dd64808c89dfd32dd2f0ed2b51cd4b962690977769e4a6b096c7e.jpg)

![a769c7c45af87eb0c0596150c9a9759ab354d9e13563a71ac154f169ef685bcd.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/a769c7c45af87eb0c0596150c9a9759ab354d9e13563a71ac154f169ef685bcd.jpg)

![b48d07dcf77f1880d81dab102d9702b5b7d5bb5040d5470efe343871e40f6ccb.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/b48d07dcf77f1880d81dab102d9702b5b7d5bb5040d5470efe343871e40f6ccb.jpg)

![b491755735ca3e8b4d2297221dca7674dbb8588994e11f1e065d3aff08f26a4d.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/b491755735ca3e8b4d2297221dca7674dbb8588994e11f1e065d3aff08f26a4d.jpg)

![b5c3af81aefa3661254d76f4e2d4a18e3967255c4d5e72ac035915e28e50530a.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/b5c3af81aefa3661254d76f4e2d4a18e3967255c4d5e72ac035915e28e50530a.jpg)

![d7331e16359797034bc212df1b7d46ad760789ea1b9417be4996f0e246239891.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/d7331e16359797034bc212df1b7d46ad760789ea1b9417be4996f0e246239891.jpg)

![d84616a262835137a131ef52e9eb5e075afe48261d774253b8868af76cf6ae91.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/d84616a262835137a131ef52e9eb5e075afe48261d774253b8868af76cf6ae91.jpg)

![e3285ea6037cb30fb51bbf13f48bd53136defb09163d8a5929cdffb096a74458.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/e3285ea6037cb30fb51bbf13f48bd53136defb09163d8a5929cdffb096a74458.jpg)

![fd79e8498b1d5f5bc433d52377aa2b05fdea6c59ed73d4563e1f987c50b9eb2a.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/fd79e8498b1d5f5bc433d52377aa2b05fdea6c59ed73d4563e1f987c50b9eb2a.jpg)

![fe1a16c97d6ad2937f10780584374d31c8a6563175a5d9caa36442d49314eaae.jpg](../icml_results/1763_PathWise_%20Planning%20through%20World%20Model%20for%20Automated%20Heuristic%20Design%20via%20Self-Evolving%20LLMs/tables/fe1a16c97d6ad2937f10780584374d31c8a6563175a5d9caa36442d49314eaae.jpg)

<a id="paper-1741"></a>

## SIGMA-PPG: Statistical-prior Informed Generative Masking Architecture for PPG Foundation Model


### Images

![03b3e94c4d403c4c7c42a429185683ac4ec90603083200fd26476031eb345575.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/03b3e94c4d403c4c7c42a429185683ac4ec90603083200fd26476031eb345575.jpg)

![20f659d9f0e20e3189175f885ee4631dc33a1484bbacacbec0e808fda152722a.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/20f659d9f0e20e3189175f885ee4631dc33a1484bbacacbec0e808fda152722a.jpg)

![21888aa03c4d437a58e44cd5a626edf2d306fd589010a265052bf373426d0674.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/21888aa03c4d437a58e44cd5a626edf2d306fd589010a265052bf373426d0674.jpg)

![4e5349fbd9224ae194b62b4e733cbfa645538e2eafaf7636a5843ce9a196f21d.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/4e5349fbd9224ae194b62b4e733cbfa645538e2eafaf7636a5843ce9a196f21d.jpg)

![626ab5174537388953dd656ba26342512b1da94a25b65c57bc2cdb4c5c7c7cf3.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/626ab5174537388953dd656ba26342512b1da94a25b65c57bc2cdb4c5c7c7cf3.jpg)

![76338037cc6d250f957d2b17060dcce832748fd835ecea39af2df0a58cd85d5b.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/76338037cc6d250f957d2b17060dcce832748fd835ecea39af2df0a58cd85d5b.jpg)

![82a65ca1d5b2b20de84b833d22ba077ae0b7f1ad6fcedd4da9f417f79731b9b3.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/82a65ca1d5b2b20de84b833d22ba077ae0b7f1ad6fcedd4da9f417f79731b9b3.jpg)

![891f04b787d13007fd1c303294a45295b9c3709b19c133ba907784f64014a9b6.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/891f04b787d13007fd1c303294a45295b9c3709b19c133ba907784f64014a9b6.jpg)

![a54b1c0126919c42595a77871c3222e575b8f2b520546681f4f37515c65bf514.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/a54b1c0126919c42595a77871c3222e575b8f2b520546681f4f37515c65bf514.jpg)

![a9ac19ca73f3931ccb4463fd6e3e5ab3d12a423f1c05bcbd17351e049a2d90c2.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/a9ac19ca73f3931ccb4463fd6e3e5ab3d12a423f1c05bcbd17351e049a2d90c2.jpg)

![df561cd7ddeabbc98228525d61168bb97197e5ecddbe5d899a0e31475d46e1e9.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/images/df561cd7ddeabbc98228525d61168bb97197e5ecddbe5d899a0e31475d46e1e9.jpg)

### Tables

![19910b09a35d81d32003e5fb1caf6eacd0f79938236c2a3cc002074e587feb1a.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/19910b09a35d81d32003e5fb1caf6eacd0f79938236c2a3cc002074e587feb1a.jpg)

![51bdfe8e80e5622118f25108e35fd596218dd127b36ea6d301ec367f1a3bc5f1.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/51bdfe8e80e5622118f25108e35fd596218dd127b36ea6d301ec367f1a3bc5f1.jpg)

![5e442193ce585acbc5e2a37371a1b1b303ce9ea5f92a4245a6b5f0d0d3d06655.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/5e442193ce585acbc5e2a37371a1b1b303ce9ea5f92a4245a6b5f0d0d3d06655.jpg)

![6379a823ecd8bffbe523948e55ba0fff6027690b0c341fee9d5d446e9a098f96.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/6379a823ecd8bffbe523948e55ba0fff6027690b0c341fee9d5d446e9a098f96.jpg)

![6df50028f7b35be2c2f12908f3286c2efdc74ff02fbd96c8a2333b8174eaf41a.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/6df50028f7b35be2c2f12908f3286c2efdc74ff02fbd96c8a2333b8174eaf41a.jpg)

![72224c89d19fb7d739ca34d18fd494071bceb7076fb9c5c40a0a2dd860f9a3b3.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/72224c89d19fb7d739ca34d18fd494071bceb7076fb9c5c40a0a2dd860f9a3b3.jpg)

![7e617667aecf1e54c6c54a9b7e6e8c8297e83ef7ac279f4b4259b7d6810e4e21.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/7e617667aecf1e54c6c54a9b7e6e8c8297e83ef7ac279f4b4259b7d6810e4e21.jpg)

![8b6dac7f95bbd9a616879b68338b3e97a70ca866c91355932f4790d981191cc5.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/8b6dac7f95bbd9a616879b68338b3e97a70ca866c91355932f4790d981191cc5.jpg)

![8bd7e5ebeb760022ef8d5dbf9cad11449b35620fff1d5b114fbf65224be4c920.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/8bd7e5ebeb760022ef8d5dbf9cad11449b35620fff1d5b114fbf65224be4c920.jpg)

![9101d4980c352831ce78b791c29051d82428916ed610cea9058c5e86a886d1e4.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/9101d4980c352831ce78b791c29051d82428916ed610cea9058c5e86a886d1e4.jpg)

![9c0b596b5bcd504b6cb06aacbcedb0258929c34ca01e58ce13d3d53f92a2a557.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/9c0b596b5bcd504b6cb06aacbcedb0258929c34ca01e58ce13d3d53f92a2a557.jpg)

![a1a670c0d94e2a085868b1941f08d2d28431de74a57f95044f69f915a989dc02.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/a1a670c0d94e2a085868b1941f08d2d28431de74a57f95044f69f915a989dc02.jpg)

![a9665710443b4ae3da1520b0b500656f1506f1748c4a64ecbcd238c173d6e064.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/a9665710443b4ae3da1520b0b500656f1506f1748c4a64ecbcd238c173d6e064.jpg)

![af04619bfcdbd47fcfe2c29527ff79ad30624d07c99fabbea6f9cc845304b881.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/af04619bfcdbd47fcfe2c29527ff79ad30624d07c99fabbea6f9cc845304b881.jpg)

![af5e3c7bb907810ad6af673e092dd85ce18ad6a42a791f5126c4993640277e22.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/af5e3c7bb907810ad6af673e092dd85ce18ad6a42a791f5126c4993640277e22.jpg)

![bdf2b6872028d5ab68300d1e94113e4f80e74ca1a7d4959e02fae4efd2f6165e.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/bdf2b6872028d5ab68300d1e94113e4f80e74ca1a7d4959e02fae4efd2f6165e.jpg)

![c01582d0df2ec6225e7a4c70e6fde53062b6581f9652efb0e716963ceb6c7177.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/c01582d0df2ec6225e7a4c70e6fde53062b6581f9652efb0e716963ceb6c7177.jpg)

![e24e830b0630b231f7e23a6897ddcbff07f754688e615600e235d574a4b31421.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/e24e830b0630b231f7e23a6897ddcbff07f754688e615600e235d574a4b31421.jpg)

![e90c675c114ba1a57bb8d8ed25c1fb885e8740b9423ad8c3852f7bfe60616e7b.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/e90c675c114ba1a57bb8d8ed25c1fb885e8740b9423ad8c3852f7bfe60616e7b.jpg)

![fcbca1862d89dfd434745cd83578328ab7ec8cdd5bf415a0e15803b71716ef90.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/fcbca1862d89dfd434745cd83578328ab7ec8cdd5bf415a0e15803b71716ef90.jpg)

![fe2d029299db036dfba0cd45d0f0737a5fabd133821601b38c00291f5f745485.jpg](../icml_results/1764_SIGMA-PPG_%20Statistical-prior%20Informed%20Generative%20Masking%20Architecture%20for%20PPG%20Foundation%20Model/tables/fe2d029299db036dfba0cd45d0f0737a5fabd133821601b38c00291f5f745485.jpg)

<a id="paper-1742"></a>

## Dynamic Thinking-Token Selection for Efficient Reasoning in Large Reasoning Models


### Images

![34bd1206656ac3ec383cdad02293848af32cca814c2a7748c80aa20d6998aba9.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/34bd1206656ac3ec383cdad02293848af32cca814c2a7748c80aa20d6998aba9.jpg)

![39d5ab7c6fe457d2c013f2356b95269b45500e08daf4070c32c4e60b04a4527a.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/39d5ab7c6fe457d2c013f2356b95269b45500e08daf4070c32c4e60b04a4527a.jpg)

![3d6ed8a3dc71fd37e3509d849a75c67fb47ef01936992a6e3bd1a893e38f49f6.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/3d6ed8a3dc71fd37e3509d849a75c67fb47ef01936992a6e3bd1a893e38f49f6.jpg)

![79a4050973dbc3ecf78fb3aa0fd4bc7c4e38ca3b14a0e364f81cd63664b56111.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/79a4050973dbc3ecf78fb3aa0fd4bc7c4e38ca3b14a0e364f81cd63664b56111.jpg)

![82a07cb07a04bcfe71697e5fd6bee76b47315cd0899b82ca5a317ba956d2531a.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/82a07cb07a04bcfe71697e5fd6bee76b47315cd0899b82ca5a317ba956d2531a.jpg)

![8432076194b64d08150ae7adee7beda530c3d33601d9bb8c22627cbb9d9f41a4.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/8432076194b64d08150ae7adee7beda530c3d33601d9bb8c22627cbb9d9f41a4.jpg)

![91d95a823ee2866e8e87066d963018b710a865095c5f0878c813de2d574de1bd.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/91d95a823ee2866e8e87066d963018b710a865095c5f0878c813de2d574de1bd.jpg)

![93cef5a5e6f0e40859769ebda9814aa505e80a04898ee3d20aee43480e4def77.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/93cef5a5e6f0e40859769ebda9814aa505e80a04898ee3d20aee43480e4def77.jpg)

![9514ce87cc59440ea5d592182291d3fae087e62ab6b43e81b31f3ba639eec692.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/9514ce87cc59440ea5d592182291d3fae087e62ab6b43e81b31f3ba639eec692.jpg)

![960ac132004b01006e40d50d9ec4363eb45a330fe6e8ec5d1402f072fc3ff2f5.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/960ac132004b01006e40d50d9ec4363eb45a330fe6e8ec5d1402f072fc3ff2f5.jpg)

![a4c46e9f6be9f9c4a48d8da3a121728bc30148cd50ed0bc4ef3f98d7b80f6f87.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/a4c46e9f6be9f9c4a48d8da3a121728bc30148cd50ed0bc4ef3f98d7b80f6f87.jpg)

![ace5f401abdf1cfd02e9336ac16025cbcbc7e49f7fdb5697734dc8ec36df911e.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/ace5f401abdf1cfd02e9336ac16025cbcbc7e49f7fdb5697734dc8ec36df911e.jpg)

![bfd974848e5b6c978af75b4a3f81ca7bdb1217fd2756ba849c054481b65ec403.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/bfd974848e5b6c978af75b4a3f81ca7bdb1217fd2756ba849c054481b65ec403.jpg)

![cde9d658f9325bd88eb33bcc9457de05099b1dbae1d99602541cb984774a555c.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/cde9d658f9325bd88eb33bcc9457de05099b1dbae1d99602541cb984774a555c.jpg)

![d4e73cde08761d09855e74ca4e34899094afe7edaf9594b78cbed2bdc807aee5.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/d4e73cde08761d09855e74ca4e34899094afe7edaf9594b78cbed2bdc807aee5.jpg)

![e587cc2a9056ca6d8459a8139c18290073f8c1c2b421c49878867fa6618a9096.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/e587cc2a9056ca6d8459a8139c18290073f8c1c2b421c49878867fa6618a9096.jpg)

![fd877fdbe964337be8e46f0b822ae7ed27e38965558fa50766d44c8a9409d4df.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/images/fd877fdbe964337be8e46f0b822ae7ed27e38965558fa50766d44c8a9409d4df.jpg)

### Tables

![4e02f6468c6901a7e02ed2c7e03937cb9fdc42c74831ff159a3ecc1efae471e9.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/tables/4e02f6468c6901a7e02ed2c7e03937cb9fdc42c74831ff159a3ecc1efae471e9.jpg)

![73663635079e87a89196796051d30f7bfc50a61f220c1b44dfaa41f050dbfbb6.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/tables/73663635079e87a89196796051d30f7bfc50a61f220c1b44dfaa41f050dbfbb6.jpg)

![9baf674440cf07d7eb4fdd2a0a8d527581bc8c572bab9b3718dd98cafd81efff.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/tables/9baf674440cf07d7eb4fdd2a0a8d527581bc8c572bab9b3718dd98cafd81efff.jpg)

![daaed8de5f8ca2347b274bc65a8e785d2e83d98af0af22c86488b14216f112e9.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/tables/daaed8de5f8ca2347b274bc65a8e785d2e83d98af0af22c86488b14216f112e9.jpg)

![dd69845e94c0043716bd48bbb643c003eb2e64e2ebac14b0e6b2df21f335f682.jpg](../icml_results/1765_Dynamic%20Thinking-Token%20Selection%20for%20Efficient%20Reasoning%20in%20Large%20Reasoning%20Models/tables/dd69845e94c0043716bd48bbb643c003eb2e64e2ebac14b0e6b2df21f335f682.jpg)

<a id="paper-1743"></a>

## MutAtlas: A PDB-Wide Energy-Guided Atlas of Protein Mutation Effects


### Images

![19cd2b17851476c2b14c7b20b90d977f224d5c647d1a60e327cd00a53ffa3970.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/19cd2b17851476c2b14c7b20b90d977f224d5c647d1a60e327cd00a53ffa3970.jpg)

![2f071eab43f200f1314db603ddc7c0cafaabaa73c25dab03bd0726947e11c991.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/2f071eab43f200f1314db603ddc7c0cafaabaa73c25dab03bd0726947e11c991.jpg)

![451290e76095a76796ae31d113a11a6899079f0e0d3af6c6073414a40aa38056.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/451290e76095a76796ae31d113a11a6899079f0e0d3af6c6073414a40aa38056.jpg)

![604ad3fa055b2fa07760bb320a91ac47c9746aee9cb0a4206ac53795bf4ddcaf.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/604ad3fa055b2fa07760bb320a91ac47c9746aee9cb0a4206ac53795bf4ddcaf.jpg)

![9db162fa3d0384081e0595c7b8bcc0bfde583b0fa696c722dde531e578378eeb.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/9db162fa3d0384081e0595c7b8bcc0bfde583b0fa696c722dde531e578378eeb.jpg)

![9f7f6147d0e12150154ab12c5690586cc1ee87675533d8540e1924310903373e.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/9f7f6147d0e12150154ab12c5690586cc1ee87675533d8540e1924310903373e.jpg)

![b831172f46ca88195d41e3fc560dd2c20889e0d80d22d3d4afc83c671eade2e6.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/b831172f46ca88195d41e3fc560dd2c20889e0d80d22d3d4afc83c671eade2e6.jpg)

![c7e95c5da93e966c8b65e955bb41d0c1316b20d2446e2eee163b3e7e51df90f6.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/c7e95c5da93e966c8b65e955bb41d0c1316b20d2446e2eee163b3e7e51df90f6.jpg)

![ca2ab2f421ca790484831a291b8aaad5f1e23a5a5fd784d74cd6d4256aff67be.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/ca2ab2f421ca790484831a291b8aaad5f1e23a5a5fd784d74cd6d4256aff67be.jpg)

![d06220d3f2735f4ed44d462e96c5d1213325ade0cee9b49c5e601ad3ea171279.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/d06220d3f2735f4ed44d462e96c5d1213325ade0cee9b49c5e601ad3ea171279.jpg)

![f93b91c9e40eadef1bad0805b8d3fea09034d271a4044eab46dea58f897d86e5.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/images/f93b91c9e40eadef1bad0805b8d3fea09034d271a4044eab46dea58f897d86e5.jpg)

### Tables

![87b2ce7ec6abc88f9d50af9d09ea6e7d38f4831fd41b9ff04804e66283d48f62.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/tables/87b2ce7ec6abc88f9d50af9d09ea6e7d38f4831fd41b9ff04804e66283d48f62.jpg)

![950b62ee0ebd2313d354fa8804a37b534a26dfe24ec5d7093c68139d8ce1c791.jpg](../icml_results/1766_MutAtlas_%20A%20PDB-Wide%20Energy-Guided%20Atlas%20of%20Protein%20Mutation%20Effects/tables/950b62ee0ebd2313d354fa8804a37b534a26dfe24ec5d7093c68139d8ce1c791.jpg)

<a id="paper-1744"></a>

## Delving into Non-Exchangeability for Conformal Prediction in Graph-Structured Multivariate Time Series


### Images

![10339d0a155dba24048e27734e35d0811e8f44780c107caad4498624307dae80.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/10339d0a155dba24048e27734e35d0811e8f44780c107caad4498624307dae80.jpg)

![4c4864a685f050ac676d1be5290066774bae59b3705c3354fb04e5792a875279.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/4c4864a685f050ac676d1be5290066774bae59b3705c3354fb04e5792a875279.jpg)

![6568401452ae29d377a94df6cc37d96555e2c35c955c9634ad97c71c0e7871ff.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/6568401452ae29d377a94df6cc37d96555e2c35c955c9634ad97c71c0e7871ff.jpg)

![6c55110d0dfdc407722e8c1582151f324dde291111023c363e6eb07a4007e01f.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/6c55110d0dfdc407722e8c1582151f324dde291111023c363e6eb07a4007e01f.jpg)

![8cbbcd0fcf20751564589bbb5147baa137c4167792b734458c22f6a0bfe07630.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/8cbbcd0fcf20751564589bbb5147baa137c4167792b734458c22f6a0bfe07630.jpg)

![922e1392580631ce76f945207b9c88c6bfcff0c79710b0dad60d6028b4d27947.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/922e1392580631ce76f945207b9c88c6bfcff0c79710b0dad60d6028b4d27947.jpg)

![f0b310a74141821b2d26bf478dc45e0c2c5e90a67fc60610f1fb1eb287785597.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/images/f0b310a74141821b2d26bf478dc45e0c2c5e90a67fc60610f1fb1eb287785597.jpg)

### Tables

![0f39230a60631fe18a78fd4d2f5e55c819cdfdb36cbc82a2df2d8f8048d36dee.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/0f39230a60631fe18a78fd4d2f5e55c819cdfdb36cbc82a2df2d8f8048d36dee.jpg)

![1d93d375d8b0474d9a5bce42a1325560bbe4cf5cc3044ba09ae7fa5eab2ee2a6.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/1d93d375d8b0474d9a5bce42a1325560bbe4cf5cc3044ba09ae7fa5eab2ee2a6.jpg)

![57fc785b7b1bf7cbbda1a2108423326924389170d752fc63a4587b35e205f4e5.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/57fc785b7b1bf7cbbda1a2108423326924389170d752fc63a4587b35e205f4e5.jpg)

![5b2596d6b4dc677dd8235299a3b4200d845d1a54d1e318b3befabe17309db3a1.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/5b2596d6b4dc677dd8235299a3b4200d845d1a54d1e318b3befabe17309db3a1.jpg)

![a3a57b2ff61ad3c3689d305ea7682c92a938260a6b1817dcb50ee8303b303e8b.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/a3a57b2ff61ad3c3689d305ea7682c92a938260a6b1817dcb50ee8303b303e8b.jpg)

![af97b0f92ba0eb419e6c494d8d69242f1fb62d3b61239b6aadbbdc3702abf5cd.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/af97b0f92ba0eb419e6c494d8d69242f1fb62d3b61239b6aadbbdc3702abf5cd.jpg)

![d09ff37fd453ecb46a1727d853e0f722f7cfcdb62a66c79d3de68e924057c23b.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/d09ff37fd453ecb46a1727d853e0f722f7cfcdb62a66c79d3de68e924057c23b.jpg)

![d609f8c2dba1b366f6f12342b469677103e47693dc53c8932e2ea19db4ab8eb6.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/d609f8c2dba1b366f6f12342b469677103e47693dc53c8932e2ea19db4ab8eb6.jpg)

![e2bfebaed4afe4e6385db3d789fe73c160018a25f683642cb20d66eb98ef0aa0.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/e2bfebaed4afe4e6385db3d789fe73c160018a25f683642cb20d66eb98ef0aa0.jpg)

![fce6f65fec6ee483c7f153b2ad1e58855ec3c0f861bf7b2815a4c2dc341aa89e.jpg](../icml_results/1767_Delving%20into%20Non-Exchangeability%20for%20Conformal%20Prediction%20in%20Graph-Structured%20Multivariate%20Time%20Seri/tables/fce6f65fec6ee483c7f153b2ad1e58855ec3c0f861bf7b2815a4c2dc341aa89e.jpg)

<a id="paper-1745"></a>

## Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models


### Images

![123ad8fdb79f7d8821c33b6b2d2bda4707a3ca89d90f5b249f9734fb5611f686.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/123ad8fdb79f7d8821c33b6b2d2bda4707a3ca89d90f5b249f9734fb5611f686.jpg)

![1bd3b778b133e4f4f7f0a29cf7eaac6118f2ad1a8b4241527155a7dfd7a12c1e.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/1bd3b778b133e4f4f7f0a29cf7eaac6118f2ad1a8b4241527155a7dfd7a12c1e.jpg)

![2dbf8d50e67c481f6da8895f023f6a316f9e1dad658e6bac11aab1cee3ca3457.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/2dbf8d50e67c481f6da8895f023f6a316f9e1dad658e6bac11aab1cee3ca3457.jpg)

![2fca2741b3cc4298c3262517d8aec984b1dac54d4ca6fe896a8f0242c23c687b.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/2fca2741b3cc4298c3262517d8aec984b1dac54d4ca6fe896a8f0242c23c687b.jpg)

![322aa1360ef3233ffe89ead39dee6002a05c692aea7a216cd9dcc742d9cfef93.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/322aa1360ef3233ffe89ead39dee6002a05c692aea7a216cd9dcc742d9cfef93.jpg)

![33b9a094d6efcc7666c63f5a4d2a25a3fcf0616d6a13721e8d964560d5ef5b3e.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/33b9a094d6efcc7666c63f5a4d2a25a3fcf0616d6a13721e8d964560d5ef5b3e.jpg)

![36ad2ebc28a7c72234012454107df2af1d111825de50e818c19854d13dad2b04.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/36ad2ebc28a7c72234012454107df2af1d111825de50e818c19854d13dad2b04.jpg)

![44184f406514a55905979c0845a0bfb8396261bca2e75e813c1fc27604b1f006.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/44184f406514a55905979c0845a0bfb8396261bca2e75e813c1fc27604b1f006.jpg)

![496ce49571a80a78ee182ad49178365ffd36bf326fa759e0b9322a0ddc24cfcd.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/496ce49571a80a78ee182ad49178365ffd36bf326fa759e0b9322a0ddc24cfcd.jpg)

![62f6729277f8ce7d31fd5abc601b02f8bfb75bab3896ed62c48733c86ead5a30.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/62f6729277f8ce7d31fd5abc601b02f8bfb75bab3896ed62c48733c86ead5a30.jpg)

![699d58ebfe71c05b745ea2816ba815144cc9a6ee0064a8e68849995429879daf.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/699d58ebfe71c05b745ea2816ba815144cc9a6ee0064a8e68849995429879daf.jpg)

![6a156370672e3b3a6c7e1d21676ab31852ad1e71aad8bb5cfb0d7b06e96ce4f2.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/6a156370672e3b3a6c7e1d21676ab31852ad1e71aad8bb5cfb0d7b06e96ce4f2.jpg)

![701f62d3eb3a7fc5b543f53c079845b7acb392a4dc5aeed5a7c9539f557311cc.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/701f62d3eb3a7fc5b543f53c079845b7acb392a4dc5aeed5a7c9539f557311cc.jpg)

![7046eaeb75737af2d58ccc208c246fc955c6a801438e8d6b347866c0aa684fda.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/7046eaeb75737af2d58ccc208c246fc955c6a801438e8d6b347866c0aa684fda.jpg)

![7e8b2ecd8287ffade57f2066ffbbebe28e7523163f37689cb5d3354da6f12ce4.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/7e8b2ecd8287ffade57f2066ffbbebe28e7523163f37689cb5d3354da6f12ce4.jpg)

![849f563cfdccf837845664e6f98bded0e00d2ca4936ef4328ec2dadab81e80ea.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/849f563cfdccf837845664e6f98bded0e00d2ca4936ef4328ec2dadab81e80ea.jpg)

![8b72affe692ce2487ace3585b0b01d65d146953cea81d72bbe3929d1ef6b93df.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/8b72affe692ce2487ace3585b0b01d65d146953cea81d72bbe3929d1ef6b93df.jpg)

![94157e930372206e39344e09e51a95b95167b9da3dcec4a1e780a9d1b7b9ff74.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/94157e930372206e39344e09e51a95b95167b9da3dcec4a1e780a9d1b7b9ff74.jpg)

![97c5cded0ebbdcf222a036303b686f7b17ac6829f4ccaaba62efa4a2ba02db4d.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/97c5cded0ebbdcf222a036303b686f7b17ac6829f4ccaaba62efa4a2ba02db4d.jpg)

![9fae76ad61e88e00449ef078a3edee8be91c3284f482981a12988ac457764a3a.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/9fae76ad61e88e00449ef078a3edee8be91c3284f482981a12988ac457764a3a.jpg)

![a4d538dc00946abcbbebc234e7f28a7cf1a1efdcfc03b4a7a48bdf37a4421561.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/a4d538dc00946abcbbebc234e7f28a7cf1a1efdcfc03b4a7a48bdf37a4421561.jpg)

![b85b349bc36261d25f4b2cea93fececac8ead3d9ade45954b1a49d1a95b74392.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/b85b349bc36261d25f4b2cea93fececac8ead3d9ade45954b1a49d1a95b74392.jpg)

![c086b9a3d4bb418f396b6c3dc29e1ca0fdfebc56c4e5e18f53bccb459a619eb4.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/c086b9a3d4bb418f396b6c3dc29e1ca0fdfebc56c4e5e18f53bccb459a619eb4.jpg)

![c923293d8479f83d0b193a21e97f94cc245aaa3d9398ff134061eee638e155bd.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/c923293d8479f83d0b193a21e97f94cc245aaa3d9398ff134061eee638e155bd.jpg)

![dadbf2101a3aaa765af343a051722a314f9d57d0d18cb08dd0d13ecabf030ab5.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/dadbf2101a3aaa765af343a051722a314f9d57d0d18cb08dd0d13ecabf030ab5.jpg)

![db1e9c95d3b2f665b78107d64e41f709fdc01c736116034144d0fef2f3608fa3.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/db1e9c95d3b2f665b78107d64e41f709fdc01c736116034144d0fef2f3608fa3.jpg)

![e4690a923ef17d771ec1213118bbbf502aabc0b523f199d4d07b27b4c71ae866.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/e4690a923ef17d771ec1213118bbbf502aabc0b523f199d4d07b27b4c71ae866.jpg)

![e6689a562a1bc9a161bb0d4d5de5cd15530f46a3d924ccef38ad5f28d8917dc3.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/e6689a562a1bc9a161bb0d4d5de5cd15530f46a3d924ccef38ad5f28d8917dc3.jpg)

![e741138bbd7ac93273192482c87fc632650f85755971416405e5357c4a6f8b42.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/e741138bbd7ac93273192482c87fc632650f85755971416405e5357c4a6f8b42.jpg)

![eaf9f3dff9200a002d3b5e57c674720493fde49e1d04ca3c334d51135057d129.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/eaf9f3dff9200a002d3b5e57c674720493fde49e1d04ca3c334d51135057d129.jpg)

![ebfb61ed7d19f8af494be1a53822caa33cb69296ff4ee4621102a0710dd2a2c1.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/ebfb61ed7d19f8af494be1a53822caa33cb69296ff4ee4621102a0710dd2a2c1.jpg)

![f3c0f260dd375cddfdf30af13a087b9fbdbe7fe71660fcc0b624ddf6d05d38bd.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/f3c0f260dd375cddfdf30af13a087b9fbdbe7fe71660fcc0b624ddf6d05d38bd.jpg)

![fae24f087e121ab20a5a2c044e21a8d1c0c1c1dd268c84d800c7b7e11a7da6b9.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/fae24f087e121ab20a5a2c044e21a8d1c0c1c1dd268c84d800c7b7e11a7da6b9.jpg)

![fd2db13db0463d916a2691f04588931c47a864ce5fa9deb8966a41d7514404c0.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/fd2db13db0463d916a2691f04588931c47a864ce5fa9deb8966a41d7514404c0.jpg)

![fdfac909c60b568596df06a72b7ad070be3c5a877e8f73337c86f2e687d529d2.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/images/fdfac909c60b568596df06a72b7ad070be3c5a877e8f73337c86f2e687d529d2.jpg)

### Tables

![2b0a2d8ba71e23a1787a3930947f57e644865e25b351d8a4aa8dd7f1f5e023b6.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/tables/2b0a2d8ba71e23a1787a3930947f57e644865e25b351d8a4aa8dd7f1f5e023b6.jpg)

![7602c6e1e72ead3ea33a8119759847f8c21855f6477b1c328091e47873a60a1d.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/tables/7602c6e1e72ead3ea33a8119759847f8c21855f6477b1c328091e47873a60a1d.jpg)

![7a5379730e60ffd528b326e481242a51f15ea5af8d5cb9f904480c57ced4c5c7.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/tables/7a5379730e60ffd528b326e481242a51f15ea5af8d5cb9f904480c57ced4c5c7.jpg)

![b8da23c5f68348c6b89478d2eb0e02b4c1334a8c12e6bb83413b67f1de04171f.jpg](../icml_results/1768_Compression%20as%20Adaptation_%20Implicit%20Visual%20Representation%20with%20Diffusion%20Foundation%20Models/tables/b8da23c5f68348c6b89478d2eb0e02b4c1334a8c12e6bb83413b67f1de04171f.jpg)

<a id="paper-1746"></a>

## PolicyGuard: Towards Test-time and Step-level Adversary Defense for Reinforcement Learning Agent


### Images

![072d7040b0591138c3cdf6bce36426be1f8c0fe1bbeab9d496f77071f3396d57.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/072d7040b0591138c3cdf6bce36426be1f8c0fe1bbeab9d496f77071f3396d57.jpg)

![2d9818059d811cda20205fb35a79c7b0d6e2eabe5b6af2ad1a37e27648c1d3c1.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/2d9818059d811cda20205fb35a79c7b0d6e2eabe5b6af2ad1a37e27648c1d3c1.jpg)

![31fbfd6568caf1116b7cabb7a9df1dcc6f73d6713bac281fce387227199348b4.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/31fbfd6568caf1116b7cabb7a9df1dcc6f73d6713bac281fce387227199348b4.jpg)

![36d152447af816800f408f3f9ac52bdec3382ec770711a1b502a2753d38fbb8c.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/36d152447af816800f408f3f9ac52bdec3382ec770711a1b502a2753d38fbb8c.jpg)

![4ffc5a60f5eb45e236914c5d823a71aa9639ae789edb0523d63b6bc67180eb00.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/4ffc5a60f5eb45e236914c5d823a71aa9639ae789edb0523d63b6bc67180eb00.jpg)

![6a9e00b6bdbd5169bbe85638c2f6d55436145d35cc34938c368fa00b2ad5cbdf.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/6a9e00b6bdbd5169bbe85638c2f6d55436145d35cc34938c368fa00b2ad5cbdf.jpg)

![729b5118ec0c4bcaa5fd8491bcb9ef63c39947e5b9b65ce17a7b1bab71a82b98.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/729b5118ec0c4bcaa5fd8491bcb9ef63c39947e5b9b65ce17a7b1bab71a82b98.jpg)

![7b4720983fa8e0895b78342068d573d0bd73b90984fce3e29406f2ece16ce66e.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/7b4720983fa8e0895b78342068d573d0bd73b90984fce3e29406f2ece16ce66e.jpg)

![7b85a34dfea42feba5646338549ee56dfd6041d2f08c7bc2ae2f438731b118c9.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/7b85a34dfea42feba5646338549ee56dfd6041d2f08c7bc2ae2f438731b118c9.jpg)

![aa794de2aaca91f31124b5cee82ce5ba9b1c594c305549206750f0bb23efaba1.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/aa794de2aaca91f31124b5cee82ce5ba9b1c594c305549206750f0bb23efaba1.jpg)

![cc6d8b882b42bad988a889635d4cf945d90919e7a2570cdb717c5711c316b751.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/cc6d8b882b42bad988a889635d4cf945d90919e7a2570cdb717c5711c316b751.jpg)

![d66a11d2f8f02e1025cddc86354e5c2cd4ee5530a67770baf16f8c4c98eb6a16.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/images/d66a11d2f8f02e1025cddc86354e5c2cd4ee5530a67770baf16f8c4c98eb6a16.jpg)

### Tables

![82a9d49fefb7fe69f5971a4f4c866bf24307bb7f1b1c7a475bb4b7ea09c2f9b0.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/tables/82a9d49fefb7fe69f5971a4f4c866bf24307bb7f1b1c7a475bb4b7ea09c2f9b0.jpg)

![ae5ea87135c6556cb912889e9e6c7de468da5d278fb1944e9c6b8eaa562506e1.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/tables/ae5ea87135c6556cb912889e9e6c7de468da5d278fb1944e9c6b8eaa562506e1.jpg)

![ded6b31ba30560c866dc45c5c4b979d2d0518b2a63781f57d87ac2e32a4c0b8f.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/tables/ded6b31ba30560c866dc45c5c4b979d2d0518b2a63781f57d87ac2e32a4c0b8f.jpg)

![f453435a3f3870f8b2385b2dbe20f5f504a1cb74e88eee44f134ae31af77cf86.jpg](../icml_results/1769_PolicyGuard_%20Towards%20Test-time%20and%20Step-level%20Adversary%20Defense%20for%20Reinforcement%20Learning%20Agent/tables/f453435a3f3870f8b2385b2dbe20f5f504a1cb74e88eee44f134ae31af77cf86.jpg)

<a id="paper-1747"></a>

## Constrained Bayesian Experimental Design via Online Planning


### Images

![1ca61bbfd022e6937cb0f595a7ce9c68b3089e8a5189ea3774c01a6570a3f6d5.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/1ca61bbfd022e6937cb0f595a7ce9c68b3089e8a5189ea3774c01a6570a3f6d5.jpg)

![51ab22693a4265d0613e0f11d5cb2bc1d9de307a33297393c12e6452c8f63724.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/51ab22693a4265d0613e0f11d5cb2bc1d9de307a33297393c12e6452c8f63724.jpg)

![5a3723bc8a9acb7c92ec5d58d7f8827a7d919510eb9491d843dd02ea95068370.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/5a3723bc8a9acb7c92ec5d58d7f8827a7d919510eb9491d843dd02ea95068370.jpg)

![8b304e72d4fa1ca5fa74ab5ce3aaf9391c6fdf1c1c5f7afb9c15dafbb49d5e32.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/8b304e72d4fa1ca5fa74ab5ce3aaf9391c6fdf1c1c5f7afb9c15dafbb49d5e32.jpg)

![a31650d101f256e318abc15430d9fe4affbe90d9abf06361f9652fe26f37ecf0.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/a31650d101f256e318abc15430d9fe4affbe90d9abf06361f9652fe26f37ecf0.jpg)

![a35d20af4a2a41e722d4c52ba7394b5d75e7c769e13251270937d5c390843e55.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/a35d20af4a2a41e722d4c52ba7394b5d75e7c769e13251270937d5c390843e55.jpg)

![a6e0716c80a3fd3271a5ce41008b391b8eefd8d61ea2412899bcc43f00c6f350.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/a6e0716c80a3fd3271a5ce41008b391b8eefd8d61ea2412899bcc43f00c6f350.jpg)

![ef920cf4a67ae9fa3e824aaa0e7f46b2b4491de25a3dcc38b23c2a2703952026.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/ef920cf4a67ae9fa3e824aaa0e7f46b2b4491de25a3dcc38b23c2a2703952026.jpg)

![fdb72f85a3cf9dd43be0fc37dd3317108e347cde751a586d1b8c9f55b63a1abb.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/images/fdb72f85a3cf9dd43be0fc37dd3317108e347cde751a586d1b8c9f55b63a1abb.jpg)

### Tables

![072531886fb1dd6880cf1b36c2f1a1f19994c647fe215e5c9f52a944cb004a20.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/072531886fb1dd6880cf1b36c2f1a1f19994c647fe215e5c9f52a944cb004a20.jpg)

![338444486b733b1a56cdf360145b9555c3d8711a8c4e363e7b8d04e0eeed9904.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/338444486b733b1a56cdf360145b9555c3d8711a8c4e363e7b8d04e0eeed9904.jpg)

![63d37ccce52efeeb3dbbdeb0597a22d07eb362289a8f3a1d94e07c414111ccac.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/63d37ccce52efeeb3dbbdeb0597a22d07eb362289a8f3a1d94e07c414111ccac.jpg)

![70ce944d55b27b61ed675e2d622ebb6517c6cf69164f1f9118d89f836a4726d1.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/70ce944d55b27b61ed675e2d622ebb6517c6cf69164f1f9118d89f836a4726d1.jpg)

![82edd445941e7b0ea2a0525578ef9b9381cbe0984202dd4f06cbb3bcda633ca8.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/82edd445941e7b0ea2a0525578ef9b9381cbe0984202dd4f06cbb3bcda633ca8.jpg)

![fa3213d06e6186f7c643cac5b2a266c11c4f3fdcbbc6ea705dd24a433ffeda8a.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/fa3213d06e6186f7c643cac5b2a266c11c4f3fdcbbc6ea705dd24a433ffeda8a.jpg)

![ff31c141b3e257df254c81b05690f375d62b91c7219d500cc137a5c5adb8f6d0.jpg](../icml_results/1770_Constrained%20Bayesian%20Experimental%20Design%20via%20Online%20Planning/tables/ff31c141b3e257df254c81b05690f375d62b91c7219d500cc137a5c5adb8f6d0.jpg)

<a id="paper-1748"></a>

## Probabilistic Salient Object Ranking


### Images

![0b6038e2e5846db5bc03bfb49969a4a9cf2efa4edcdd80f81b7a384824f94672.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/images/0b6038e2e5846db5bc03bfb49969a4a9cf2efa4edcdd80f81b7a384824f94672.jpg)

![0d0ec451569b47b819963d6293a4a3993bdf644f3fa7cb5b8b368561000db821.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/images/0d0ec451569b47b819963d6293a4a3993bdf644f3fa7cb5b8b368561000db821.jpg)

![48e9ddf4530a2024078114adbe41a05fb8df60abd4b1a53dba09913d2979156d.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/images/48e9ddf4530a2024078114adbe41a05fb8df60abd4b1a53dba09913d2979156d.jpg)

![547f639f252077175b693098c7f7832fd60e70a00eebde03ca05594ac7045b7d.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/images/547f639f252077175b693098c7f7832fd60e70a00eebde03ca05594ac7045b7d.jpg)

![64c22808a8f264b79272e6810d35c7de3ea9d1a329305b885144908711ae412b.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/images/64c22808a8f264b79272e6810d35c7de3ea9d1a329305b885144908711ae412b.jpg)

![c934eefebfcb54c79dae29c7ae51325a09679150293a2bba1321a71e8013e3e3.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/images/c934eefebfcb54c79dae29c7ae51325a09679150293a2bba1321a71e8013e3e3.jpg)

### Tables

![0cb8bd09fc61bbfa476857e2158c5d1adcd7cccf471bd54b142baba22f387983.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/tables/0cb8bd09fc61bbfa476857e2158c5d1adcd7cccf471bd54b142baba22f387983.jpg)

![265ec3313fa0a234a7d3b09a6ac204073d72ef88a226ed020e03ae4c8a377644.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/tables/265ec3313fa0a234a7d3b09a6ac204073d72ef88a226ed020e03ae4c8a377644.jpg)

![6ffa56ddf4b0cccb3a6d36e0d80a33eb8c0bec114d4fd3b127ba742bc73b6786.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/tables/6ffa56ddf4b0cccb3a6d36e0d80a33eb8c0bec114d4fd3b127ba742bc73b6786.jpg)

![c048469c44015c3175a6fa6c1deffc0c6118cbf11501114595d9c480a50d638b.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/tables/c048469c44015c3175a6fa6c1deffc0c6118cbf11501114595d9c480a50d638b.jpg)

![c9ded3545dcbdb7dc835cdccf0e543c5ed312f15cf89be34d34c2afff594fbfb.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/tables/c9ded3545dcbdb7dc835cdccf0e543c5ed312f15cf89be34d34c2afff594fbfb.jpg)

![e66f755700d7ba20448755e190f1f3ddad6ed3d13f213c4f0bcb0f9d3e90d804.jpg](../icml_results/1771_Probabilistic%20Salient%20Object%20Ranking/tables/e66f755700d7ba20448755e190f1f3ddad6ed3d13f213c4f0bcb0f9d3e90d804.jpg)

<a id="paper-1749"></a>

## VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

### Images

![0460f04a8b75e13389e3664aa5a349aa0b260f511d255a7033a80a17133dc7ab.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/0460f04a8b75e13389e3664aa5a349aa0b260f511d255a7033a80a17133dc7ab.jpg)

![13b10f6bb662d3cc60d00fda76e4bc629b0de973c1463e895923e75c83d877b7.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/13b10f6bb662d3cc60d00fda76e4bc629b0de973c1463e895923e75c83d877b7.jpg)

![16fc5c642efa94a5f2a20e5cc8deee1f703c7f3b5947f50922fe77b22f282fc9.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/16fc5c642efa94a5f2a20e5cc8deee1f703c7f3b5947f50922fe77b22f282fc9.jpg)

![3a09e9f9ddbb44c7461f2bf05448ea8dfd1899c4a2600b52eb548bb1e8338834.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/3a09e9f9ddbb44c7461f2bf05448ea8dfd1899c4a2600b52eb548bb1e8338834.jpg)

![57056b261444e7606dc759296a9aecf5a076542d5c963204adeb82e2bd0d797b.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/57056b261444e7606dc759296a9aecf5a076542d5c963204adeb82e2bd0d797b.jpg)

![c6f56435e392cc8835f4228fcce8600668c8ba18a7d54869a24d2400a85b6b6c.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/c6f56435e392cc8835f4228fcce8600668c8ba18a7d54869a24d2400a85b6b6c.jpg)

![e4bf16c1cb4e82e4f61b0c7c50bfa4d8e145762e20d59b8a566e03ea10e76465.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/e4bf16c1cb4e82e4f61b0c7c50bfa4d8e145762e20d59b8a566e03ea10e76465.jpg)

![ebd8d7b70b8a1d683e2648318fe266672eba54a437a537479c838e60c25e7a5f.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/ebd8d7b70b8a1d683e2648318fe266672eba54a437a537479c838e60c25e7a5f.jpg)

![f0318c4aec23a5892b47f5377dcc96a938f3d968cbbab80f5b58c3f0b9239009.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/images/f0318c4aec23a5892b47f5377dcc96a938f3d968cbbab80f5b58c3f0b9239009.jpg)

### Tables

![56916df7e088da33676b625c6c4cb6b2561a3fb17e1e9bb8f248a196450a60e4.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/tables/56916df7e088da33676b625c6c4cb6b2561a3fb17e1e9bb8f248a196450a60e4.jpg)

![68ac491ce46f8b0d5fee2995906ad48e2e484f3ba0e0b85d3dda3316c179ee3d.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/tables/68ac491ce46f8b0d5fee2995906ad48e2e484f3ba0e0b85d3dda3316c179ee3d.jpg)

![768ebc9e6498cddb2fa48d04e0552c99cffa307d38d9a663ca0dd2cdffdaf61c.jpg](../icml_results/1772_VLAW_%20Iterative%20Co-Improvement%20of%20Vision-Language-Action%20Policy%20and%20World%20Model/tables/768ebc9e6498cddb2fa48d04e0552c99cffa307d38d9a663ca0dd2cdffdaf61c.jpg)
