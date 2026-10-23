# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 69 of 197

## 目录 (Table of Contents)

1. [CRAG: Can 3D Generative Models Help 3D Assembly?](#paper-2245)
2. [USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning](#paper-2246)
3. [What Makes a Representation Good for Single-Cell Perturbation Prediction?](#paper-2247)
4. [LithoDreamer: A Physics-Informed World Model for Multi-Stage Computational Lithography](#paper-2248)
5. [Beyond Scalars: Evaluating and Understanding LLM Reasoning via Geometric Progress and Stability](#paper-2249)
6. [OvisOCR: End-to-End Document Parsing via Aligning Specialized Perception with General Reasoning](#paper-2250)
7. [Visual Implicit Autoregressive Modeling](#paper-2251)
8. [Boosting World Models Learning via Latent-Space Value Alignment](#paper-2252)
9. [Quantum Algorithms for Triangle Cut Sparsification](#paper-2253)
10. [Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](#paper-2254)
11. [GFMate: Empowering Graph Foundation Models with Test-time Prompt Tuning](#paper-2255)
12. [Enhancing LLM Training via Spectral Clipping](#paper-2256)
13. [Efficient Transformer Attention for SNNs via Hadamard Simplification](#paper-2257)
14. [Transformed Latent Variable Multi-Output Gaussian Processes](#paper-2258)
15. [XSkill: Continual Learning from Experience and Skills in Multimodal Agents](#paper-2259)
16. [Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit](#paper-2260)
17. [AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks](#paper-2261)
18. [Spatial Conformal Inference through Localized Quantile Regression](#paper-2262)
19. [KineFlow: Kinematic Second-Order Flow Matching for Time-Series Forecasting](#paper-2263)
20. [DecoVer: A Decompose-and-Verify Neuro-Symbolic Framework for Embodied Task Planning with BC+](#paper-2264)
21. [BRIDGE: Triangular Fixed-Point Refinement for Long-Horizon Persona Consistency](#paper-2265)
22. [Convergence Analysis of the Lion Optimizer in Centralized and Distributed Settings](#paper-2266)
23. [SCoA: Revisiting Domain Generalized Object Detection with Style-Conditioned Adaptation](#paper-2267)
24. [EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](#paper-2268)
25. [Budget-Efficient Attacks and Robustness Training for Cooperative MARL](#paper-2269)
26. [StructMamPose: From Sequential Perception to Structural Reasoning for 3D Human Pose Estimation](#paper-2270)
27. [Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](#paper-2271)
28. [Ophiuchus: Incentivizing Tool-augmented ”Think with Images” for Joint Medical Segmentation, Understanding and Reasoning](#paper-2272)
29. [NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration](#paper-2273)
30. [Seeing Symbols, Missing Structure: A Real-World Handwritten Mathematical Expression Recognition Benchmark for Large Models](#paper-2274)
31. [PICACO: Pluralistic In-Context Value Alignment via Total Correlation Optimization](#paper-2275)
32. [$\textttPRISM$:A 3D Probabilistic Neural Representation for Interpretable Shape Modeling](#paper-2276)
33. [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](#paper-2277)

---


<a id="paper-2245"></a>

## CRAG: Can 3D Generative Models Help 3D Assembly?

### Images

![331236db8bc453500a90fb5127847c5de58b7ed038d3bea718bdb6a1c355ce62.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/images/331236db8bc453500a90fb5127847c5de58b7ed038d3bea718bdb6a1c355ce62.jpg)

![5c81145cb2693b0ca70c01cfe0465fa7ca16935a5f74f1b7316fc52b32a1d671.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/images/5c81145cb2693b0ca70c01cfe0465fa7ca16935a5f74f1b7316fc52b32a1d671.jpg)

![8c6579f328f124cc0cbf4fbbc476f4e56ebabe9923e3efbb583d379feba8535b.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/images/8c6579f328f124cc0cbf4fbbc476f4e56ebabe9923e3efbb583d379feba8535b.jpg)

![9f9ac6f65d052cc81c38e9bc8c59dab1d0256d71e2057bdd837a5ecbb1e43d88.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/images/9f9ac6f65d052cc81c38e9bc8c59dab1d0256d71e2057bdd837a5ecbb1e43d88.jpg)

![a3bbf21c874220741aa3dd7f388ac07dd45a8bdad6f4d6796d57b1222d5bfc0b.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/images/a3bbf21c874220741aa3dd7f388ac07dd45a8bdad6f4d6796d57b1222d5bfc0b.jpg)

![e05fa2ea74b456061142113bc671160c85c813a3dfc08027fe1599a5f309abfc.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/images/e05fa2ea74b456061142113bc671160c85c813a3dfc08027fe1599a5f309abfc.jpg)

### Tables

![2c951069e9e73647cc419b0e40c92f284359605bcb3617163a5809e2ce3047fd.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/tables/2c951069e9e73647cc419b0e40c92f284359605bcb3617163a5809e2ce3047fd.jpg)

![381daa6ea4e8dbc0cfd7bd0773970b7911155df4402b0688505cda67fec23095.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/tables/381daa6ea4e8dbc0cfd7bd0773970b7911155df4402b0688505cda67fec23095.jpg)

![ab7ce8023e4d11d9588c58cbbd10bfd1999bf168741067099654370bbed7785e.jpg](../icml_results/2273_Efficient%2C%20Property-Aligned%20Fan-Out%20Retrieval%20via%20RL-Compiled%20Diffusion/tables/ab7ce8023e4d11d9588c58cbbd10bfd1999bf168741067099654370bbed7785e.jpg)

## CRAG: Can 3D Generative Models Help 3D Assembly?


### Images

![5d9d86307d88217f0979506081afae91bf7a390db3d150c179c4d470635a874f.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/5d9d86307d88217f0979506081afae91bf7a390db3d150c179c4d470635a874f.jpg)

![6cb62f42e5bc1b5ac16ab18b8658f4f888ee22cc968f2bf3dbd7093ed30776b4.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/6cb62f42e5bc1b5ac16ab18b8658f4f888ee22cc968f2bf3dbd7093ed30776b4.jpg)

![7d906b5cebc2cc9b0008b20cbfa0a9ff4123471ac31e93d74984334e0c673dc0.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/7d906b5cebc2cc9b0008b20cbfa0a9ff4123471ac31e93d74984334e0c673dc0.jpg)

![7fa8330ddf3f35f1be616c0abd789ba52646d842d9f92becdb14f755da8bb298.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/7fa8330ddf3f35f1be616c0abd789ba52646d842d9f92becdb14f755da8bb298.jpg)

![8415fcce3b036f6d0a94e858051850fc47773bcaa6494f78fc28e1eaf21ceb05.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/8415fcce3b036f6d0a94e858051850fc47773bcaa6494f78fc28e1eaf21ceb05.jpg)

![bedbdc625aec3a6f5fd8f187bcbd0d9b49f795810042ce652befe4d2a414ea02.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/bedbdc625aec3a6f5fd8f187bcbd0d9b49f795810042ce652befe4d2a414ea02.jpg)

![dc670f243539add8f679c01ed3dd3a4868201f9bb10b9bb27a7e1ea20d3d68dc.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/dc670f243539add8f679c01ed3dd3a4868201f9bb10b9bb27a7e1ea20d3d68dc.jpg)

![eefdabd76a3984c2d695860302e4ac15a04cd0299f54acce2c0dfbcb93dc4515.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/images/eefdabd76a3984c2d695860302e4ac15a04cd0299f54acce2c0dfbcb93dc4515.jpg)

### Tables

![0580b82ce5add4d3f645c72b992514a8c2b9f3cdb6c418e1aff9c558ae9e4648.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/0580b82ce5add4d3f645c72b992514a8c2b9f3cdb6c418e1aff9c558ae9e4648.jpg)

![0a4dc5dfcd07e9a704cbe56e7f146f86007de56f4164eae7c63481d0dbc2d309.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/0a4dc5dfcd07e9a704cbe56e7f146f86007de56f4164eae7c63481d0dbc2d309.jpg)

![232fe865aa03d7580c566a565e965ab81de904cb2633ee57cb62829019d575f8.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/232fe865aa03d7580c566a565e965ab81de904cb2633ee57cb62829019d575f8.jpg)

![5947a6cdada679d8824fa36eeaee67ad2b3ba0b660cf50ce88d09913dadd81bf.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/5947a6cdada679d8824fa36eeaee67ad2b3ba0b660cf50ce88d09913dadd81bf.jpg)

![8b76b6225b43a84a002f34a1ddb037ae1fddd9f0084aaf65197e1ae84dc651c1.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/8b76b6225b43a84a002f34a1ddb037ae1fddd9f0084aaf65197e1ae84dc651c1.jpg)

![eb82460418440ee5f2b25ca4edfdd51cde88bea6fb9b9556c06d2f818d52c47b.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/eb82460418440ee5f2b25ca4edfdd51cde88bea6fb9b9556c06d2f818d52c47b.jpg)

![f95368af8d09f8f64ffbe4ae97fc88085b02cfc675996f3a183177213ee08bf6.jpg](../icml_results/2274_CRAG_%20Can%203D%20Generative%20Models%20Help%203D%20Assembly_/tables/f95368af8d09f8f64ffbe4ae97fc88085b02cfc675996f3a183177213ee08bf6.jpg)

<a id="paper-2246"></a>

## USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning


### Images

![507002de8787159cc0bdd8a902e8e25c7c7b2a73a19af8b4ac5ca3075d5bb3be.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/507002de8787159cc0bdd8a902e8e25c7c7b2a73a19af8b4ac5ca3075d5bb3be.jpg)

![5bc4fd35eb332a2a30a583d9785a409f0cba923ffa812c276ee0fc6273833915.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/5bc4fd35eb332a2a30a583d9785a409f0cba923ffa812c276ee0fc6273833915.jpg)

![684c1598e20b1a5e70fe03bf02c241f6dc9a08766db8857cd56ad5f9e3024914.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/684c1598e20b1a5e70fe03bf02c241f6dc9a08766db8857cd56ad5f9e3024914.jpg)

![74785ffcc14bc3f14597ed1b85dfd5f24344decd6bfdc9a503e7fce67a22b78a.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/74785ffcc14bc3f14597ed1b85dfd5f24344decd6bfdc9a503e7fce67a22b78a.jpg)

![ec612ce746e08352f6ee2cef476fe96fe7c292399dc6415aff7260e138f36957.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/ec612ce746e08352f6ee2cef476fe96fe7c292399dc6415aff7260e138f36957.jpg)

![f1e4e3da736d692eb1972b270b4de7074b56af5c604c5e110b30296552974e4d.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/f1e4e3da736d692eb1972b270b4de7074b56af5c604c5e110b30296552974e4d.jpg)

![f96e9b5143cc83a0b42cfea2ae965691236909832879e0eb0e2b560eb4099fa6.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/images/f96e9b5143cc83a0b42cfea2ae965691236909832879e0eb0e2b560eb4099fa6.jpg)

### Tables

![0ac4d50334523a8fa1d6151cbd29bce069e6ba8abe7c35b1d32c642a470ce153.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/0ac4d50334523a8fa1d6151cbd29bce069e6ba8abe7c35b1d32c642a470ce153.jpg)

![1704e61cd07a5ce0e3923989204b4e387831bc5b5997f4a85990d6f8ce0dec75.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/1704e61cd07a5ce0e3923989204b4e387831bc5b5997f4a85990d6f8ce0dec75.jpg)

![25c355006a31076c660de0b925d4aa8b1ea86431e8061bc10d0f26b8f7f25a80.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/25c355006a31076c660de0b925d4aa8b1ea86431e8061bc10d0f26b8f7f25a80.jpg)

![2a853652bbf670a63b33feb1b585f43f563a8e8cf320d0000db0af484945cf97.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/2a853652bbf670a63b33feb1b585f43f563a8e8cf320d0000db0af484945cf97.jpg)

![39206f867709ef4841a0ac11d940a2bd5d5928da02f4b740f24dcb998a90dc61.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/39206f867709ef4841a0ac11d940a2bd5d5928da02f4b740f24dcb998a90dc61.jpg)

![3cacea2aa9f2ec6963f36be4661d4117e0b5d853c461c4902648e425c5d1dc32.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/3cacea2aa9f2ec6963f36be4661d4117e0b5d853c461c4902648e425c5d1dc32.jpg)

![4628e793d7dbd65122481553bee872c9040f171be2169d2d78255fc17433f0fd.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/4628e793d7dbd65122481553bee872c9040f171be2169d2d78255fc17433f0fd.jpg)

![4ffe9e76da2a6568ec43ebb51abfa6261b7b5d6e9b986e3de1257a2646a6999a.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/4ffe9e76da2a6568ec43ebb51abfa6261b7b5d6e9b986e3de1257a2646a6999a.jpg)

![501c4e8efefd1d29620faf81a7e7ea5dd0bc2079df937e26d787f928143c993f.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/501c4e8efefd1d29620faf81a7e7ea5dd0bc2079df937e26d787f928143c993f.jpg)

![5055cdd8de19b6edf8f5691c7c526679eb92f006b7e527f89ec3ef3dd561503e.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/5055cdd8de19b6edf8f5691c7c526679eb92f006b7e527f89ec3ef3dd561503e.jpg)

![50a56748e49d0476d2017b6e5851c97b95cc25dd3556b4d3b703a7e584cf4121.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/50a56748e49d0476d2017b6e5851c97b95cc25dd3556b4d3b703a7e584cf4121.jpg)

![5871d1da051ec5a46a2c75b8116eb84ac47ef0ca033daa48af8ec510d0bed167.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/5871d1da051ec5a46a2c75b8116eb84ac47ef0ca033daa48af8ec510d0bed167.jpg)

![93ef1074a03a892bf82a0aab0604cf64476c42a033ae4ee2e3c04f0c128940a2.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/93ef1074a03a892bf82a0aab0604cf64476c42a033ae4ee2e3c04f0c128940a2.jpg)

![9cb8949b9edc390101b4accd9d1bcb192b03e1473f9be16add76d0eb89e91ecc.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/9cb8949b9edc390101b4accd9d1bcb192b03e1473f9be16add76d0eb89e91ecc.jpg)

![a2c530a329dcd8a078f06f4f6c8d08714c86027cda61d1d7bc83418b599c4656.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/a2c530a329dcd8a078f06f4f6c8d08714c86027cda61d1d7bc83418b599c4656.jpg)

![a343d433b15d6bbe9c1a9ec1dacfd4055f22e6cf915904f70be0a722b53ce4f1.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/a343d433b15d6bbe9c1a9ec1dacfd4055f22e6cf915904f70be0a722b53ce4f1.jpg)

![acaa774670ae8672ba164e8acda7800f09b1204c33b75a86d1c3f8569ac33cef.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/acaa774670ae8672ba164e8acda7800f09b1204c33b75a86d1c3f8569ac33cef.jpg)

![ba71c9d3ddfa302683128fc7eb937b57023bc600f09a06b76fc3dad72c717136.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/ba71c9d3ddfa302683128fc7eb937b57023bc600f09a06b76fc3dad72c717136.jpg)

![c0fc75e87fabff94409e671b35c6dc448058b9f3a36885d6a15e6f57524bf6c4.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/c0fc75e87fabff94409e671b35c6dc448058b9f3a36885d6a15e6f57524bf6c4.jpg)

![c9978c5c8897ee837ce8ec852df82fc7f07613975e63146005ff840ddd2dad73.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/c9978c5c8897ee837ce8ec852df82fc7f07613975e63146005ff840ddd2dad73.jpg)

![deaf46bebffef6dce5b5945aa0aab7e198d1634b18b76358fdb442a764591bca.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/deaf46bebffef6dce5b5945aa0aab7e198d1634b18b76358fdb442a764591bca.jpg)

![ecdb90d1328ddac5bfe5888c07bdf744e416c275ba21cbeae6e634520f1dbb80.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/ecdb90d1328ddac5bfe5888c07bdf744e416c275ba21cbeae6e634520f1dbb80.jpg)

![f6263fd1c15ec1d62d1faa7b0292a8cd844c7daeae5fae1dfab614064f16ee38.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/f6263fd1c15ec1d62d1faa7b0292a8cd844c7daeae5fae1dfab614064f16ee38.jpg)

![f8e845e1b23eba8b3a462f82baa282d80eadb30b622bbdc930b6534e1b7c54c0.jpg](../icml_results/2275_USE_%20A%20Unified%20Self-Ensembling%20Framework%20for%20Test-Time%20Prompt%20Tuning/tables/f8e845e1b23eba8b3a462f82baa282d80eadb30b622bbdc930b6534e1b7c54c0.jpg)

<a id="paper-2247"></a>

## What Makes a Representation Good for Single-Cell Perturbation Prediction?


### Images

![163778802cd9f8c6f0a13043ad7768494f62befc31add50fa37d75b26615b198.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/163778802cd9f8c6f0a13043ad7768494f62befc31add50fa37d75b26615b198.jpg)

![1ae01668ef70ab8a9ea87b850b8847a2fe423deca6a33d56db77ab2d55f73851.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/1ae01668ef70ab8a9ea87b850b8847a2fe423deca6a33d56db77ab2d55f73851.jpg)

![277d61e59c55568fc93307bbda12f0b57090a9b7593a56d028a78b6dc9a44d82.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/277d61e59c55568fc93307bbda12f0b57090a9b7593a56d028a78b6dc9a44d82.jpg)

![2ac84f2f8d7ca958d57356d769efa7ba93cb8a715f4c73a9a0e977985f411eaf.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/2ac84f2f8d7ca958d57356d769efa7ba93cb8a715f4c73a9a0e977985f411eaf.jpg)

![345f14032aa4eaff79d41acca0aaac2022085d6be1d427587375217ca0c07b1f.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/345f14032aa4eaff79d41acca0aaac2022085d6be1d427587375217ca0c07b1f.jpg)

![642fb5507e8b942ba912b0ee5edce500335b0c0bc8f828ed0460f9a2f30b0fa5.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/642fb5507e8b942ba912b0ee5edce500335b0c0bc8f828ed0460f9a2f30b0fa5.jpg)

![742415f9902cbb304059f40b3a9cd62044971b42958f644aa333c99de67b8791.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/742415f9902cbb304059f40b3a9cd62044971b42958f644aa333c99de67b8791.jpg)

![b17ad576bb1a1a97da2054b9265d20d4228d273491e431b145f6102ce1e2942a.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/b17ad576bb1a1a97da2054b9265d20d4228d273491e431b145f6102ce1e2942a.jpg)

![ce280a3264f5344d4c443511d90ff5676b6d0752ce011ea69c1bcff5309b1002.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/ce280a3264f5344d4c443511d90ff5676b6d0752ce011ea69c1bcff5309b1002.jpg)

![d2d38b82779011948b488441013b495ce47584b37d7a83c1d74faad5ff371d7a.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/d2d38b82779011948b488441013b495ce47584b37d7a83c1d74faad5ff371d7a.jpg)

![de02db4daa0aed65706bf5dac7a3878291ec5d00c304f6c5764c3b9dad6e804a.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/de02db4daa0aed65706bf5dac7a3878291ec5d00c304f6c5764c3b9dad6e804a.jpg)

![eac8b4e48d0708ecb5f37ba889c1e7bfbd624cb5921e1531fd47f509db0222b4.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/eac8b4e48d0708ecb5f37ba889c1e7bfbd624cb5921e1531fd47f509db0222b4.jpg)

![f9d43873a352b94233777a31b2991a3d664e9efaf16c26f33684c32ec284d933.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/images/f9d43873a352b94233777a31b2991a3d664e9efaf16c26f33684c32ec284d933.jpg)

### Tables

![14f13177eaa680bb07e7750162aab770057b8fa986cd71dc2048476abd8ac7ec.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/14f13177eaa680bb07e7750162aab770057b8fa986cd71dc2048476abd8ac7ec.jpg)

![2805041f095fe0628c7cc1cae1ee5a8bb8a61565940a657661d8d6594d3b8d76.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/2805041f095fe0628c7cc1cae1ee5a8bb8a61565940a657661d8d6594d3b8d76.jpg)

![3ef9a097952711b8316582194b5d4af0163159cf9b1b5c1244b7e843760a785c.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/3ef9a097952711b8316582194b5d4af0163159cf9b1b5c1244b7e843760a785c.jpg)

![4295da087ba0624dc155f247929fb892db2ec01eb579504a6b2fb8d2a2379462.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/4295da087ba0624dc155f247929fb892db2ec01eb579504a6b2fb8d2a2379462.jpg)

![55682b7d81e9fc44f147cd2a9c0752947b6fd510024ee47df80c821424618149.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/55682b7d81e9fc44f147cd2a9c0752947b6fd510024ee47df80c821424618149.jpg)

![62c42330d4dc86998babd3288cda311a98ac97738d243a23ce7ba30e768ac796.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/62c42330d4dc86998babd3288cda311a98ac97738d243a23ce7ba30e768ac796.jpg)

![7c8e76477a26e01216604cc7e22d018dd19fbfd4e6d5d3296096c62d3c75857d.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/7c8e76477a26e01216604cc7e22d018dd19fbfd4e6d5d3296096c62d3c75857d.jpg)

![7dcff40589182fd68336c330ddeee0dfdadcbe42c63a39d34cd37a2590b487ff.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/7dcff40589182fd68336c330ddeee0dfdadcbe42c63a39d34cd37a2590b487ff.jpg)

![88cdef93613ef3c89f58f6c33d0eae98bb3e76b1633dd0e2d65a532d9bec18a4.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/88cdef93613ef3c89f58f6c33d0eae98bb3e76b1633dd0e2d65a532d9bec18a4.jpg)

![a477ab37d89c048c9d35d4d4304b5e56bcc80cea6ac206ae671469510a9334a2.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/a477ab37d89c048c9d35d4d4304b5e56bcc80cea6ac206ae671469510a9334a2.jpg)

![c161c14851682bb748fd26448177e94834c475e0233e9d8cc84002a1fa28066b.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/c161c14851682bb748fd26448177e94834c475e0233e9d8cc84002a1fa28066b.jpg)

![c42a28a3c6509e7725d97a950f8328db6dff5a1eafa02f4dc504a6703805778e.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/c42a28a3c6509e7725d97a950f8328db6dff5a1eafa02f4dc504a6703805778e.jpg)

![cc4cdee1537acd068c0800a8571e6e6a1ec7be7dba12cb584dade6c3689a66ee.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/cc4cdee1537acd068c0800a8571e6e6a1ec7be7dba12cb584dade6c3689a66ee.jpg)

![cf60d671f7817581e5b74db7ce352517eb913a88f11b5703f41f3451b55a4a92.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/cf60d671f7817581e5b74db7ce352517eb913a88f11b5703f41f3451b55a4a92.jpg)

![dc136f945adbddc776c8f8d78b1b96410c3ca2dded9d0eeab360550bd4710cae.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/dc136f945adbddc776c8f8d78b1b96410c3ca2dded9d0eeab360550bd4710cae.jpg)

![ead4b4b9e54caedb5cb49fd7c55c2c871330da53d45f1c221a807ec128308f48.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/ead4b4b9e54caedb5cb49fd7c55c2c871330da53d45f1c221a807ec128308f48.jpg)

![fa874edbcf62964911b00183a2366c587dae2e84b712b7ebaf778d6ff5d6c773.jpg](../icml_results/2276_What%20Makes%20a%20Representation%20Good%20for%20Single-Cell%20Perturbation%20Prediction_/tables/fa874edbcf62964911b00183a2366c587dae2e84b712b7ebaf778d6ff5d6c773.jpg)

<a id="paper-2248"></a>

## LithoDreamer: A Physics-Informed World Model for Multi-Stage Computational Lithography


### Images

![25459fc4200f0592e6a305b75cc7639c043bb88392b70bc8ad65316918871b18.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/25459fc4200f0592e6a305b75cc7639c043bb88392b70bc8ad65316918871b18.jpg)

![50a136c34bf387effe16c3f1d03f442db19728aaf19536a25f03c824c58393f1.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/50a136c34bf387effe16c3f1d03f442db19728aaf19536a25f03c824c58393f1.jpg)

![62c69d89fe5772c2155c8c98d0dbc715ac55bd45e2734688b07bf9b2ce6e63e3.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/62c69d89fe5772c2155c8c98d0dbc715ac55bd45e2734688b07bf9b2ce6e63e3.jpg)

![7a1454f5c088edae791e0207e54863d8d531970f3746ef95fb7b27a50b7c54c7.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/7a1454f5c088edae791e0207e54863d8d531970f3746ef95fb7b27a50b7c54c7.jpg)

![8bf2ac47dc75680678152222c7d263f502b05c9e855b7a710fe953f985848c25.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/8bf2ac47dc75680678152222c7d263f502b05c9e855b7a710fe953f985848c25.jpg)

![8d7af00fe22fb052ad45fbafe4d3e72832e8f105d1158d441b9b41ec588de06f.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/8d7af00fe22fb052ad45fbafe4d3e72832e8f105d1158d441b9b41ec588de06f.jpg)

![991fb9083e03b94dd5dd982d1f0bcf3aff29cbe9c386cb6acad0f75d8733d0d3.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/991fb9083e03b94dd5dd982d1f0bcf3aff29cbe9c386cb6acad0f75d8733d0d3.jpg)

![bf47f78d86dc7fc0352f8f7639095da60e7ff9c5c61accae9839adfe3654066c.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/bf47f78d86dc7fc0352f8f7639095da60e7ff9c5c61accae9839adfe3654066c.jpg)

![e657b403ea8db93a47e9af6d49fcb01695aee40bbe0c9f74a90261cfb458df9c.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/images/e657b403ea8db93a47e9af6d49fcb01695aee40bbe0c9f74a90261cfb458df9c.jpg)

### Tables

![5b08e8cd9012c09d2f89afd0e6fe7a77c20f9b05a90c6c825bcd3b67261ab05b.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/5b08e8cd9012c09d2f89afd0e6fe7a77c20f9b05a90c6c825bcd3b67261ab05b.jpg)

![5fba34a47d627b984a4553680375fb0cb07323fdec59956356dbd1bfd5b99b0f.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/5fba34a47d627b984a4553680375fb0cb07323fdec59956356dbd1bfd5b99b0f.jpg)

![6897cd1d2c16897c6566a601e53deb49e41fb5ff2993a3551a47d53514a3d7ce.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/6897cd1d2c16897c6566a601e53deb49e41fb5ff2993a3551a47d53514a3d7ce.jpg)

![81105c36e9ef3961d2e07054dde69c1ab84d030e086effbcf25c958872647a8b.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/81105c36e9ef3961d2e07054dde69c1ab84d030e086effbcf25c958872647a8b.jpg)

![9b789f1b1193f1e738b196ab3917c8e6e66c4716d13814068f9df1429d5f40ec.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/9b789f1b1193f1e738b196ab3917c8e6e66c4716d13814068f9df1429d5f40ec.jpg)

![9b9e3d97f31d7dfa85a64ecdcc9f8ea95932226603173d38f2c0809254b7bf18.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/9b9e3d97f31d7dfa85a64ecdcc9f8ea95932226603173d38f2c0809254b7bf18.jpg)

![b59b17050b5650eb16d57e2e7f5f049dbcdbc7f2912a6920e11594e5c2fc63f5.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/b59b17050b5650eb16d57e2e7f5f049dbcdbc7f2912a6920e11594e5c2fc63f5.jpg)

![bfe9cb537d055fcd0c35e4b1abc062abc2f41637a346e1dc103bd77ee5e5461a.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/bfe9cb537d055fcd0c35e4b1abc062abc2f41637a346e1dc103bd77ee5e5461a.jpg)

![e83d5af803b6f424a9673878b849a63f6fd5918e15ca33137ce8e6987b98390b.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/e83d5af803b6f424a9673878b849a63f6fd5918e15ca33137ce8e6987b98390b.jpg)

![f744dc4328903ababf3668e2283e19807db24a5a660031fd867d20beb8eeef48.jpg](../icml_results/2277_LithoDreamer_%20A%20Physics-Informed%20World%20Model%20for%20Multi-Stage%20Computational%20Lithography/tables/f744dc4328903ababf3668e2283e19807db24a5a660031fd867d20beb8eeef48.jpg)

<a id="paper-2249"></a>

## Beyond Scalars: Evaluating and Understanding LLM Reasoning via Geometric Progress and Stability


### Images

![05615bef0c72f122e240da8cb1bbb173440c4ff677cdc67ea1cf3003daecb5bb.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/05615bef0c72f122e240da8cb1bbb173440c4ff677cdc67ea1cf3003daecb5bb.jpg)

![1a294e4c0e322e997f53dbd970b62241990e6346f8bd0e2839ecdad0e2612cdf.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/1a294e4c0e322e997f53dbd970b62241990e6346f8bd0e2839ecdad0e2612cdf.jpg)

![55b8708913a106763fcc110937de880fc1588696ebecadcd52d283a88fd4f555.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/55b8708913a106763fcc110937de880fc1588696ebecadcd52d283a88fd4f555.jpg)

![5d3231b0459b3ba5b9afec6cb4a367cb8cbff82cc1416643354f9cdbc64505ee.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/5d3231b0459b3ba5b9afec6cb4a367cb8cbff82cc1416643354f9cdbc64505ee.jpg)

![613465b0cbcdaa03596a7d3661e4ed7747f3c3c3379ffaf4da817afbf947c87f.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/613465b0cbcdaa03596a7d3661e4ed7747f3c3c3379ffaf4da817afbf947c87f.jpg)

![79f28b5b8cdcf918d917975fcd494405b6efbef8fd5ce1c7c5510735fe8c2324.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/79f28b5b8cdcf918d917975fcd494405b6efbef8fd5ce1c7c5510735fe8c2324.jpg)

![7bd854d29aa0b3b88d536c15fe0119caf2a757162b143b5393921113d5f2cf65.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/7bd854d29aa0b3b88d536c15fe0119caf2a757162b143b5393921113d5f2cf65.jpg)

![9e46c3bb98f8fd766ae6aa43085d38c3ccdad4be807a46e9c3ccee9cea02e520.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/9e46c3bb98f8fd766ae6aa43085d38c3ccdad4be807a46e9c3ccee9cea02e520.jpg)

![a3433498cea27e0880ac8f2a5aa09b68807fa5ef5d535d10aad11fc5250f657e.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/a3433498cea27e0880ac8f2a5aa09b68807fa5ef5d535d10aad11fc5250f657e.jpg)

![b44c673ceef4655d5c69ffa469b2e53d4006fbca305b1e93b3e9458d087c1bd2.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/b44c673ceef4655d5c69ffa469b2e53d4006fbca305b1e93b3e9458d087c1bd2.jpg)

![d90074d6195b4b9e821a2f37618e6c6eba9ae51716abd01d0235d6200cea419f.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/d90074d6195b4b9e821a2f37618e6c6eba9ae51716abd01d0235d6200cea419f.jpg)

![decf72c21eb38dd37679b4473a8e9c1a2c9805440526af36afa39d18212a6f47.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/decf72c21eb38dd37679b4473a8e9c1a2c9805440526af36afa39d18212a6f47.jpg)

![e41d931c28eef2ab46b23fefd5e94149923b796e4f24a4ac5803855e1086d1cf.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/e41d931c28eef2ab46b23fefd5e94149923b796e4f24a4ac5803855e1086d1cf.jpg)

![fee2461082d7ec97505575a218af2d48a9c9117fda5153a2c5457ae390b09ec8.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/images/fee2461082d7ec97505575a218af2d48a9c9117fda5153a2c5457ae390b09ec8.jpg)

### Tables

![1b35542daf997faeb512633b3dcbe4d622b662b1759262f14249fe2587db8e03.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/1b35542daf997faeb512633b3dcbe4d622b662b1759262f14249fe2587db8e03.jpg)

![3dd7fd0b516f14283c5b4ed962c7460b5eccacc9f3fcbdd9296b61e2a1807b20.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/3dd7fd0b516f14283c5b4ed962c7460b5eccacc9f3fcbdd9296b61e2a1807b20.jpg)

![3e4eef1ca4af7ebbddc49f2d940a8d4b7dfa9a8148edf278984b6c67f3e6102d.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/3e4eef1ca4af7ebbddc49f2d940a8d4b7dfa9a8148edf278984b6c67f3e6102d.jpg)

![62be8e535a072653369ec6667cd36a4759bc3eb63316f279b684e8d85a39088f.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/62be8e535a072653369ec6667cd36a4759bc3eb63316f279b684e8d85a39088f.jpg)

![70b9bfde905cc1b1d8ffbbd1fe0fb15b3f44aa07847ca5e4bdc77ee8b45e0346.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/70b9bfde905cc1b1d8ffbbd1fe0fb15b3f44aa07847ca5e4bdc77ee8b45e0346.jpg)

![892dc0b322a9a7e2c56716821011a888fb9a0614b485224aeeb55eaeb0d1a2cc.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/892dc0b322a9a7e2c56716821011a888fb9a0614b485224aeeb55eaeb0d1a2cc.jpg)

![8a8b5508ae1ab7aa7e1ce0a9ad2dca7e52954904f4a6f834af7108b34cd2a3b0.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/8a8b5508ae1ab7aa7e1ce0a9ad2dca7e52954904f4a6f834af7108b34cd2a3b0.jpg)

![b86d0121655f696c373a825ffe312482f60b750a278804d658823c602e650cf1.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/b86d0121655f696c373a825ffe312482f60b750a278804d658823c602e650cf1.jpg)

![d96add6c256d3ae1bfac8b74182902493e649f444fb0c9f94fda0470f7dd4b18.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/d96add6c256d3ae1bfac8b74182902493e649f444fb0c9f94fda0470f7dd4b18.jpg)

![ed453e42277bf5380d76d6e6627e2d2963879e3cf06a54aba4ba456ed9e3e9cf.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/ed453e42277bf5380d76d6e6627e2d2963879e3cf06a54aba4ba456ed9e3e9cf.jpg)

![f9518b63ff92b9611ba01b38f388542a8e0b882a81c7b1819ca9f7d98629aa60.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/f9518b63ff92b9611ba01b38f388542a8e0b882a81c7b1819ca9f7d98629aa60.jpg)

![fb8708ddcea910d836478aceb1b04378c9e200da642b960fbdc5c0b0aa429f6f.jpg](../icml_results/2278_Beyond%20Scalars_%20Evaluating%20and%20Understanding%20LLM%20Reasoning%20via%20Geometric%20Progress%20and%20Stability/tables/fb8708ddcea910d836478aceb1b04378c9e200da642b960fbdc5c0b0aa429f6f.jpg)

<a id="paper-2250"></a>

## OvisOCR: End-to-End Document Parsing via Aligning Specialized Perception with General Reasoning


### Images

![0b44dcd38623828cfa5cd29e63c00279b993022de9df56102b459d75d42be451.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/images/0b44dcd38623828cfa5cd29e63c00279b993022de9df56102b459d75d42be451.jpg)

![366eb7b62df90e34a864f85ac756757909086d380612dc8c53363998947b6a00.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/images/366eb7b62df90e34a864f85ac756757909086d380612dc8c53363998947b6a00.jpg)

![8409e36fbf7d3553cc62150dd89c02b601232fd625fb85c8c2975c8197b91ae4.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/images/8409e36fbf7d3553cc62150dd89c02b601232fd625fb85c8c2975c8197b91ae4.jpg)

![d0267926cc56db10d5d4e05c600e5fc76e0762bc01308f14cebe59a8054e55ae.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/images/d0267926cc56db10d5d4e05c600e5fc76e0762bc01308f14cebe59a8054e55ae.jpg)

![df2c96cf8bec709e91a4b47f7c152d1719a52c395d1526dd5f4902f74b8cddfa.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/images/df2c96cf8bec709e91a4b47f7c152d1719a52c395d1526dd5f4902f74b8cddfa.jpg)

### Tables

![28c03e913c415a3bf7eeb5253c35005a75d66e5448e8d1bf747b6806aa3a493a.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/tables/28c03e913c415a3bf7eeb5253c35005a75d66e5448e8d1bf747b6806aa3a493a.jpg)

![3f353e725e1b9bb04f678b03e314d11264ad843e028e82ad062274ab7e1d9102.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/tables/3f353e725e1b9bb04f678b03e314d11264ad843e028e82ad062274ab7e1d9102.jpg)

![5829cc576d982485b4cbfe191a50cdabf0df8873534847370606a587d7b7764d.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/tables/5829cc576d982485b4cbfe191a50cdabf0df8873534847370606a587d7b7764d.jpg)

![f366f6920a9c0c227b60fdede96231a6de25083b18c553ec41cd772085bc9db8.jpg](../icml_results/2279_OvisOCR_%20End-to-End%20Document%20Parsing%20via%20Aligning%20Specialized%20Perception%20with%20General%20Reasoning/tables/f366f6920a9c0c227b60fdede96231a6de25083b18c553ec41cd772085bc9db8.jpg)

<a id="paper-2251"></a>

## Visual Implicit Autoregressive Modeling


### Images

![01fb7efdd3b9235f7c1dabd9e9411021e8cd125cb839e7228be55f96505412bb.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/01fb7efdd3b9235f7c1dabd9e9411021e8cd125cb839e7228be55f96505412bb.jpg)

![61dde84e6621555de2972654dab9ff5361a23bf22d143b0f34e239d017ed25e0.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/61dde84e6621555de2972654dab9ff5361a23bf22d143b0f34e239d017ed25e0.jpg)

![706118d7b3a290effbb04e3396c2251ba2529243a7fcc66d7fb6a6e31eca5ab0.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/706118d7b3a290effbb04e3396c2251ba2529243a7fcc66d7fb6a6e31eca5ab0.jpg)

![7454745636b4937e583f7d24827a57fc468cd96921e5ac7b3418b643c46feac6.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/7454745636b4937e583f7d24827a57fc468cd96921e5ac7b3418b643c46feac6.jpg)

![8ad8b03a84ddb6098f297d662d41c5410925e1d7b19cdb3d8b0607482c951a08.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/8ad8b03a84ddb6098f297d662d41c5410925e1d7b19cdb3d8b0607482c951a08.jpg)

![987dae4d41c1d6e97a27994d6124cf441cca5c69d7424a1f45330929c29ae92d.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/987dae4d41c1d6e97a27994d6124cf441cca5c69d7424a1f45330929c29ae92d.jpg)

![b546c52719452412eea144978b0e4bd2ad992c5718a6a567197bc4306c2b5b09.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/b546c52719452412eea144978b0e4bd2ad992c5718a6a567197bc4306c2b5b09.jpg)

![bf4add19ee5fd24cfc47521f9fe7ffebf3bdc5bfb781e7439937886f81de8428.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/bf4add19ee5fd24cfc47521f9fe7ffebf3bdc5bfb781e7439937886f81de8428.jpg)

![e7df7c702dce562f3710ede63913fd3c570ec7f3a8d3fe75b45369970ac2e414.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/images/e7df7c702dce562f3710ede63913fd3c570ec7f3a8d3fe75b45369970ac2e414.jpg)

### Tables

![0704d7ec0546fc6d768a3f9c98f340a7eea0808550d1d323c708a87fe749027b.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/tables/0704d7ec0546fc6d768a3f9c98f340a7eea0808550d1d323c708a87fe749027b.jpg)

![16a95fb4cbc9d82bdbd8f32ee31ad873de403f8ef5f4bdbbc5e3371196c1620c.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/tables/16a95fb4cbc9d82bdbd8f32ee31ad873de403f8ef5f4bdbbc5e3371196c1620c.jpg)

![4c79fc0ea07b40f32e81615c806e444fefe3947049e7612388abbbf7e10b578a.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/tables/4c79fc0ea07b40f32e81615c806e444fefe3947049e7612388abbbf7e10b578a.jpg)

![921bc38af1879fe89e3b2798a106548da40a9394b9c1529b92d44b5cc9cdbdcd.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/tables/921bc38af1879fe89e3b2798a106548da40a9394b9c1529b92d44b5cc9cdbdcd.jpg)

![b04300118d67da9abdefc8ce6842c45f5b5f894eb042938decdcd8d3b9ae0af5.jpg](../icml_results/2280_Visual%20Implicit%20Autoregressive%20Modeling/tables/b04300118d67da9abdefc8ce6842c45f5b5f894eb042938decdcd8d3b9ae0af5.jpg)

<a id="paper-2252"></a>

## Boosting World Models Learning via Latent-Space Value Alignment


### Images

![234c38fb7b3fce5a0524c31c6afef94356ee7b7953d1831288e5acf43b55a446.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/234c38fb7b3fce5a0524c31c6afef94356ee7b7953d1831288e5acf43b55a446.jpg)

![34b2eb5ea6cf0498f9c18903306f40e78cdea372fc2741f0773a4f7f25dddc87.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/34b2eb5ea6cf0498f9c18903306f40e78cdea372fc2741f0773a4f7f25dddc87.jpg)

![3f71d543ec71e8c7fa2f75d4ad5450cf7f229129eca513f0c792cb87e4640053.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/3f71d543ec71e8c7fa2f75d4ad5450cf7f229129eca513f0c792cb87e4640053.jpg)

![3feca768a769d2bc8915c3fea81b073ae8f8b20427f79ea194b994ebdd0c1a51.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/3feca768a769d2bc8915c3fea81b073ae8f8b20427f79ea194b994ebdd0c1a51.jpg)

![4710413b6b4d0bf43407ceb039964df0201de20981e7203f00d27c16d1ecd30d.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/4710413b6b4d0bf43407ceb039964df0201de20981e7203f00d27c16d1ecd30d.jpg)

![4f99c69f53165cf222cdcde8aa5c402abc254811e7e3072dcc01f9c19645f0d8.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/4f99c69f53165cf222cdcde8aa5c402abc254811e7e3072dcc01f9c19645f0d8.jpg)

![5d0652a07c0cbd8424d6d1b825718f17d9185b23020c46362bc2c4cb87b2814a.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/5d0652a07c0cbd8424d6d1b825718f17d9185b23020c46362bc2c4cb87b2814a.jpg)

![63fbaaf12e7bc5476cf8aa854e098f660b744c0ad6a6e46d699a114dd846a533.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/63fbaaf12e7bc5476cf8aa854e098f660b744c0ad6a6e46d699a114dd846a533.jpg)

![88bd49afe7cac0d16b88faefb9d6fd4d35c31bcb744528f07c8e6a5d1b5e972c.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/88bd49afe7cac0d16b88faefb9d6fd4d35c31bcb744528f07c8e6a5d1b5e972c.jpg)

![cd162cdd8fea45774837728c5444d6b0baa09b92fcf63bdb013cdb3dd103d545.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/cd162cdd8fea45774837728c5444d6b0baa09b92fcf63bdb013cdb3dd103d545.jpg)

![eff32677dbaf71c12205d8d478a7deb2ad1c5763379de0b0087ed5968acaba9d.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/eff32677dbaf71c12205d8d478a7deb2ad1c5763379de0b0087ed5968acaba9d.jpg)

![f0bc24e797497437ce8f3bb7afe2a4951fbe88808deff54155d7a9751ccdb3c8.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/f0bc24e797497437ce8f3bb7afe2a4951fbe88808deff54155d7a9751ccdb3c8.jpg)

![f9477a08dec61ea3c63466eceb31d91091993539383d4c836af12482249178dc.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/images/f9477a08dec61ea3c63466eceb31d91091993539383d4c836af12482249178dc.jpg)

### Tables

![0ccc23bfa5fac3e5a1190cfe8ce8756fe7e2d0b8eaf9b95bc88fe637e933cd39.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/0ccc23bfa5fac3e5a1190cfe8ce8756fe7e2d0b8eaf9b95bc88fe637e933cd39.jpg)

![10245b76f7d9d03511c702f718d30f69b8eb3885bab840d615dbe6708f2bbb7c.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/10245b76f7d9d03511c702f718d30f69b8eb3885bab840d615dbe6708f2bbb7c.jpg)

![115042474de43a2ee57f4a4ae167274626b8241a5ddf949e25d00259ee1995b3.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/115042474de43a2ee57f4a4ae167274626b8241a5ddf949e25d00259ee1995b3.jpg)

![19e1e95cbea3a315665cda29991b35d4e1e9bfb07a8e5af7daa2a2c644000a7e.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/19e1e95cbea3a315665cda29991b35d4e1e9bfb07a8e5af7daa2a2c644000a7e.jpg)

![1c76ebea66c6018f41b37e0c090e62162dd9640095117c23d1d34f0f9b898a36.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/1c76ebea66c6018f41b37e0c090e62162dd9640095117c23d1d34f0f9b898a36.jpg)

![2b05dbef6c3deeea65f7e183c7f137ecc81660e1be1f8982172671753190e7c2.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/2b05dbef6c3deeea65f7e183c7f137ecc81660e1be1f8982172671753190e7c2.jpg)

![4bd1503aa3c37a37c2ed99794240cc6ccd026c279971b689588030ce10cbf686.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/4bd1503aa3c37a37c2ed99794240cc6ccd026c279971b689588030ce10cbf686.jpg)

![4c9e9370ed4b5f0cd05e7625715435a71fe045f38eb6166031087bee7f1a5fa3.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/4c9e9370ed4b5f0cd05e7625715435a71fe045f38eb6166031087bee7f1a5fa3.jpg)

![5f41cc98b52ee481645a11d1be840a2fafeff4dc5a75acf4b6b6a4d2850516e2.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/5f41cc98b52ee481645a11d1be840a2fafeff4dc5a75acf4b6b6a4d2850516e2.jpg)

![627e1f4a5d67fdf35898cc962501134a356e0ddab229f51875594cdc0806bdd8.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/627e1f4a5d67fdf35898cc962501134a356e0ddab229f51875594cdc0806bdd8.jpg)

![63c4a8d5a18aa201380fbc9118f5a7db80c0a80296c662611eaa9bf564ac9acc.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/63c4a8d5a18aa201380fbc9118f5a7db80c0a80296c662611eaa9bf564ac9acc.jpg)

![659a817bc9a28a9c8ddf04d3e5e3b7f94b4724c2cd06879c72ddb05b424129a7.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/659a817bc9a28a9c8ddf04d3e5e3b7f94b4724c2cd06879c72ddb05b424129a7.jpg)

![86e9bb596f726d72cf231dfc4f6ae17a979b35412efd698243e37a9f3480a576.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/86e9bb596f726d72cf231dfc4f6ae17a979b35412efd698243e37a9f3480a576.jpg)

![9a7046a76a1b845d8076b83552abf9136960fc01e3d3833cea7cbb888c0834e0.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/9a7046a76a1b845d8076b83552abf9136960fc01e3d3833cea7cbb888c0834e0.jpg)

![a40ea6b9d589f4a49f2fe67f05a2c8b46ca968fd3ed8236fc5345125bca5241e.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/a40ea6b9d589f4a49f2fe67f05a2c8b46ca968fd3ed8236fc5345125bca5241e.jpg)

![ab76003fcb01252ea2fcdcde80f6f5f7cf0813a1f3a359edb94f8460f1ef68b9.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/ab76003fcb01252ea2fcdcde80f6f5f7cf0813a1f3a359edb94f8460f1ef68b9.jpg)

![afcd786841bb70011b67f383aa4419182e71073cb6875bdad10c25ff0577031a.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/afcd786841bb70011b67f383aa4419182e71073cb6875bdad10c25ff0577031a.jpg)

![b4146b9d31864970103103e18cfcd225ee1f53a901b64ad0be18ef3f31f9e961.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/b4146b9d31864970103103e18cfcd225ee1f53a901b64ad0be18ef3f31f9e961.jpg)

![df36fcd768814dbc499a060f3f83bdd72842f3123e481582245b48178e98026b.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/df36fcd768814dbc499a060f3f83bdd72842f3123e481582245b48178e98026b.jpg)

![df9fed55f62cf82442bdd5043726019d27e5fa600b7dfc04b7f715dfd0001a3c.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/df9fed55f62cf82442bdd5043726019d27e5fa600b7dfc04b7f715dfd0001a3c.jpg)

![feca9f72b827ceecb4d2f361f8f983e4e33b7a6104660e4c6b5cf33cb0add884.jpg](../icml_results/2281_Boosting%20World%20Models%20Learning%20via%20Latent-Space%20Value%20Alignment/tables/feca9f72b827ceecb4d2f361f8f983e4e33b7a6104660e4c6b5cf33cb0add884.jpg)

<a id="paper-2253"></a>

## Quantum Algorithms for Triangle Cut Sparsification


### Tables

![e96fd1e558bbec596046ce6613943681b65400b89ccfc25fa77f7eaf1de542d1.jpg](../icml_results/2282_Quantum%20Algorithms%20for%20Triangle%20Cut%20Sparsification/tables/e96fd1e558bbec596046ce6613943681b65400b89ccfc25fa77f7eaf1de542d1.jpg)

<a id="paper-2254"></a>

## Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning


### Images

![38367c0d4f789f43f5a2b77ffcafe30170361f62483bc518a6d42c65c2752b0c.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/38367c0d4f789f43f5a2b77ffcafe30170361f62483bc518a6d42c65c2752b0c.jpg)

![404fbfab3460cc1c0bc86b55420bc4340b0f790d108efa0129d90e71776e0d8b.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/404fbfab3460cc1c0bc86b55420bc4340b0f790d108efa0129d90e71776e0d8b.jpg)

![5a5994a414a68efa2233d85a65396ba96dd2f645e2b2733a3d9143118f2d3cd8.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/5a5994a414a68efa2233d85a65396ba96dd2f645e2b2733a3d9143118f2d3cd8.jpg)

![6247885f66cfc66ee280097d168413742b2a69890c34c10c5a425147d6d88b3b.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/6247885f66cfc66ee280097d168413742b2a69890c34c10c5a425147d6d88b3b.jpg)

![6c77262fa3eb3a893250a732af96ce3767ce3e2997f076153163460a475a0dfc.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/6c77262fa3eb3a893250a732af96ce3767ce3e2997f076153163460a475a0dfc.jpg)

![6f801465f7624913a84f32484f837fe93b1a3a5d2e37ee582d51682bdd2f28da.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/6f801465f7624913a84f32484f837fe93b1a3a5d2e37ee582d51682bdd2f28da.jpg)

![6fc0f352ae40242571a871679abf1931954d165da33a8482221f293e5add749c.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/6fc0f352ae40242571a871679abf1931954d165da33a8482221f293e5add749c.jpg)

![88cbc944f00774138c526373ceeb2d7bc82009c099d9ff584800bf2d18bec6bc.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/88cbc944f00774138c526373ceeb2d7bc82009c099d9ff584800bf2d18bec6bc.jpg)

![980ec8822659d116573658351e81454803d30515874977dd3cc1c31b84497231.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/980ec8822659d116573658351e81454803d30515874977dd3cc1c31b84497231.jpg)

![cab693e9539ab48806ea06d14cb3ad93c5f11c33bf49a8b94ab89a73afc5068f.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/cab693e9539ab48806ea06d14cb3ad93c5f11c33bf49a8b94ab89a73afc5068f.jpg)

![fd3cf27fdf53cf2da4b14b4e8263099fd8f949fb308c37e9d81307a6a708b3d7.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/images/fd3cf27fdf53cf2da4b14b4e8263099fd8f949fb308c37e9d81307a6a708b3d7.jpg)

### Tables

![2ba356466ea421b9fab752c295cee3ff0f07b36a52077ef23994de4ec53c892f.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/2ba356466ea421b9fab752c295cee3ff0f07b36a52077ef23994de4ec53c892f.jpg)

![7504a8ca598c7394dc84bd53229c4946b8d659fdc00288c3dfd92ee9c2909608.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/7504a8ca598c7394dc84bd53229c4946b8d659fdc00288c3dfd92ee9c2909608.jpg)

![7c0259d2bad0546aa9d4fc4602dcb614f41b3c34c01cda72f490b6497178de18.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/7c0259d2bad0546aa9d4fc4602dcb614f41b3c34c01cda72f490b6497178de18.jpg)

![9c5bd327b9d6101250bd82a122e127a430cb6bdc61389f0afbfb5d813068a61a.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/9c5bd327b9d6101250bd82a122e127a430cb6bdc61389f0afbfb5d813068a61a.jpg)

![c4c3bf81478f74e8adc69a0f1193b42730efdc6745a5f6a0713a794dcd29d614.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/c4c3bf81478f74e8adc69a0f1193b42730efdc6745a5f6a0713a794dcd29d614.jpg)

![c8454633b84b0ea8780cbe6a4670d9999a57c7e7d5ac02279e80540a01952b85.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/c8454633b84b0ea8780cbe6a4670d9999a57c7e7d5ac02279e80540a01952b85.jpg)

![d9d7f99e73c7a666096923be21f023950364cd4fb0f9006da43c067ab6146b3c.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/d9d7f99e73c7a666096923be21f023950364cd4fb0f9006da43c067ab6146b3c.jpg)

![e81913181e581d777489961cdf416978299cef804f89fa3be45dd7b43abf3e87.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/e81913181e581d777489961cdf416978299cef804f89fa3be45dd7b43abf3e87.jpg)

![f7aab9548a1407dcff0da9c5111fc6353c10f8f87bcd7ad8ce27a79b8082b8f9.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/f7aab9548a1407dcff0da9c5111fc6353c10f8f87bcd7ad8ce27a79b8082b8f9.jpg)

![fbaa1686580bc3bdf55b79c50c3e14434a4b2db0e4c49d6a9392f053cdb0b0c5.jpg](../icml_results/2283_Break%20the%20Block_%20Dynamic-size%20Reasoning%20Blocks%20for%20Diffusion%20Large%20Language%20Models%20via%20Monotonic%20Ent/tables/fbaa1686580bc3bdf55b79c50c3e14434a4b2db0e4c49d6a9392f053cdb0b0c5.jpg)

<a id="paper-2255"></a>

## GFMate: Empowering Graph Foundation Models with Test-time Prompt Tuning


### Images

![011fa76b529a9eea077e0f60b29a04c5ef34df861de3038e925606c308f2b655.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/011fa76b529a9eea077e0f60b29a04c5ef34df861de3038e925606c308f2b655.jpg)

![0ac5bdf3941622122da3eb78f2c5b549dd3039e5377d879365f42d153f1933a4.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/0ac5bdf3941622122da3eb78f2c5b549dd3039e5377d879365f42d153f1933a4.jpg)

![12b60a2c832bef59adf4ab915b5cc06849dd49da3cf62bd6d69ffe795b17b9ca.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/12b60a2c832bef59adf4ab915b5cc06849dd49da3cf62bd6d69ffe795b17b9ca.jpg)

![2816a92042ff0491d04c4da59e63a3117ebc909074e21250eba268919666ff21.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/2816a92042ff0491d04c4da59e63a3117ebc909074e21250eba268919666ff21.jpg)

![29974789444609c7795688d6ea190566ca8c487b87ce65a345840bb775df162b.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/29974789444609c7795688d6ea190566ca8c487b87ce65a345840bb775df162b.jpg)

![2ddbcee84eb403adff258308f1bbd28f9fa879a942965574d8fdb88599407521.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/2ddbcee84eb403adff258308f1bbd28f9fa879a942965574d8fdb88599407521.jpg)

![30795b0299100fca3baa5af231c319c50d630a3ac6b3d0a026a65a355e64b154.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/30795b0299100fca3baa5af231c319c50d630a3ac6b3d0a026a65a355e64b154.jpg)

![8064c9049d9a2a44ca21eef9ce3cd3a541e9ffc5d59542aba8635c527d80be9c.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/8064c9049d9a2a44ca21eef9ce3cd3a541e9ffc5d59542aba8635c527d80be9c.jpg)

![84b8cd990f4807085a6baa2c205cf7e0c2486923e589a451ec17d7dd97089f00.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/84b8cd990f4807085a6baa2c205cf7e0c2486923e589a451ec17d7dd97089f00.jpg)

![a412d6a122c34262763146870189d2a163e4eb2604e6fdab720ebcbdb6ba8c69.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/images/a412d6a122c34262763146870189d2a163e4eb2604e6fdab720ebcbdb6ba8c69.jpg)

### Tables

![0074bc1c976a102379859c7e2f7bf074c34949aa6fd214021dd97b1da0b6b006.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/0074bc1c976a102379859c7e2f7bf074c34949aa6fd214021dd97b1da0b6b006.jpg)

![0a37285f955812594692c5d01563b723db8adc6681e1177ffad7e42c356b510f.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/0a37285f955812594692c5d01563b723db8adc6681e1177ffad7e42c356b510f.jpg)

![15615e527d4539ff1c5757aa65b124a9474bae02b5fe5df5f7e1a88a5c9d36e9.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/15615e527d4539ff1c5757aa65b124a9474bae02b5fe5df5f7e1a88a5c9d36e9.jpg)

![1c2d28e05cb9893ecc94621bbeb5611d3a0b1195f93a99e9f689384e32d1a46c.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/1c2d28e05cb9893ecc94621bbeb5611d3a0b1195f93a99e9f689384e32d1a46c.jpg)

![21b12b2b0263de7cb2ff2dfb878461ada765bdb55bcaf208c376f8e925f7e40a.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/21b12b2b0263de7cb2ff2dfb878461ada765bdb55bcaf208c376f8e925f7e40a.jpg)

![365a8559d5232af04b5c5a6c9b1aede90a8ed7c530b78f0f92c4447d120223c6.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/365a8559d5232af04b5c5a6c9b1aede90a8ed7c530b78f0f92c4447d120223c6.jpg)

![3813ca789ba1e7168025b67408993f243a3e243577630326edfbf217428ffa7c.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/3813ca789ba1e7168025b67408993f243a3e243577630326edfbf217428ffa7c.jpg)

![38a2c8f1c6711fe7371918a801e1dc66e93baf7ea0261816c033d0b47b9116e1.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/38a2c8f1c6711fe7371918a801e1dc66e93baf7ea0261816c033d0b47b9116e1.jpg)

![431dc4ac9c43dd6148342d3002e3e08043109ae3ea8635a8d14fb865784b6b88.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/431dc4ac9c43dd6148342d3002e3e08043109ae3ea8635a8d14fb865784b6b88.jpg)

![4826b0994b4acc855bf1ad1c37a94d2df0c318a51a69a4dd08eb600a8968167a.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/4826b0994b4acc855bf1ad1c37a94d2df0c318a51a69a4dd08eb600a8968167a.jpg)

![4f5178970ffb0faa89637e1c7074aa1034e33c38ea5228414a7e715f4791d703.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/4f5178970ffb0faa89637e1c7074aa1034e33c38ea5228414a7e715f4791d703.jpg)

![5c27d1aa560f973f565452fd5c4ad1821074b533836c7fb363a65ad1f5c6e2b2.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/5c27d1aa560f973f565452fd5c4ad1821074b533836c7fb363a65ad1f5c6e2b2.jpg)

![7f11da9770d412212d8a0f76b84a2f99e2d1987fe23d3d48f20edd7b5d84dfe6.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/7f11da9770d412212d8a0f76b84a2f99e2d1987fe23d3d48f20edd7b5d84dfe6.jpg)

![87b6e2a3def9082d6a6fbc7213695d5e3cfe4264a2133edb4aebed614e7fa176.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/87b6e2a3def9082d6a6fbc7213695d5e3cfe4264a2133edb4aebed614e7fa176.jpg)

![8c30c79b4801f8689da3b75dd8259505a54ed9f03cb7d5f0017d07d62583a5f6.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/8c30c79b4801f8689da3b75dd8259505a54ed9f03cb7d5f0017d07d62583a5f6.jpg)

![af039b69f76fbcffa19f34e9a4a74debf7180d858dd16de285f426216c934745.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/af039b69f76fbcffa19f34e9a4a74debf7180d858dd16de285f426216c934745.jpg)

![bc1988dd675a9d411502d16ea97e9f9a7a3289fd3cba1136be049c6d601de05e.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/bc1988dd675a9d411502d16ea97e9f9a7a3289fd3cba1136be049c6d601de05e.jpg)

![e0b6755ad2cdbc71c828563d318af374f0ef078a23e0fdaab15fa97a7245606d.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/e0b6755ad2cdbc71c828563d318af374f0ef078a23e0fdaab15fa97a7245606d.jpg)

![e3dba723eba06e10d7691ae4737738db894ce8e1f669e778a6cb179d5552a127.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/e3dba723eba06e10d7691ae4737738db894ce8e1f669e778a6cb179d5552a127.jpg)

![e4b1fe91a885e4000b5f201befdd5fd1e99eab554e1fb6d38dd6c77bcc7d5b7c.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/e4b1fe91a885e4000b5f201befdd5fd1e99eab554e1fb6d38dd6c77bcc7d5b7c.jpg)

![fcb613d3ceb0981b4c51102f85187bd1dbc6c2da5bd18a6741e210607302911a.jpg](../icml_results/2284_GFMate_%20Empowering%20Graph%20Foundation%20Models%20with%20Test-time%20Prompt%20Tuning/tables/fcb613d3ceb0981b4c51102f85187bd1dbc6c2da5bd18a6741e210607302911a.jpg)

<a id="paper-2256"></a>

## Enhancing LLM Training via Spectral Clipping


### Images

![0016bec8d68e6515e91e2da76f2f905797e41c7c8782791b39501a80966e8a76.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/0016bec8d68e6515e91e2da76f2f905797e41c7c8782791b39501a80966e8a76.jpg)

![0aaccd82c6a47ad2d5547335ae54858bc6659f621b66ff7f28988d700bb0d79a.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/0aaccd82c6a47ad2d5547335ae54858bc6659f621b66ff7f28988d700bb0d79a.jpg)

![18fad7a05dce246a4324e8ac158723257d3d50230ca32d79aa3e4f532bbd2c12.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/18fad7a05dce246a4324e8ac158723257d3d50230ca32d79aa3e4f532bbd2c12.jpg)

![20884c12177409ad2e241c4df3b6129f88e4dda2181bdfe36cf69b58e079a8d5.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/20884c12177409ad2e241c4df3b6129f88e4dda2181bdfe36cf69b58e079a8d5.jpg)

![20cf1700c36db1d05d0b874db0cd85aea4ba610fc787e4a8c6fa0345a84f718f.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/20cf1700c36db1d05d0b874db0cd85aea4ba610fc787e4a8c6fa0345a84f718f.jpg)

![371c8432a39b9bb7e060487f9ec19b8da01129bb5c3d5f22a01c20d88d6c0152.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/371c8432a39b9bb7e060487f9ec19b8da01129bb5c3d5f22a01c20d88d6c0152.jpg)

![453c916753552deea3be14fa477e30f8d58bcbf76a4bfd4a4e8d43dff3fbb146.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/453c916753552deea3be14fa477e30f8d58bcbf76a4bfd4a4e8d43dff3fbb146.jpg)

![4c4d41dbf8118ebae224a72473c1c7187f25eea7b171a040dc18eb0694733b50.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/4c4d41dbf8118ebae224a72473c1c7187f25eea7b171a040dc18eb0694733b50.jpg)

![619b3ce6763940807e644d069d62393ca89bb22bc84654ae4e624191037cba54.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/619b3ce6763940807e644d069d62393ca89bb22bc84654ae4e624191037cba54.jpg)

![656b62d8085cc85ea4b8a001381ed7232ef01100e075b8d3b551fc5b1d840f4f.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/656b62d8085cc85ea4b8a001381ed7232ef01100e075b8d3b551fc5b1d840f4f.jpg)

![7cf9068dc7e21c0a875e1d0079fbfb0d42e95a0f436f07ad7255b5769592de01.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/7cf9068dc7e21c0a875e1d0079fbfb0d42e95a0f436f07ad7255b5769592de01.jpg)

![7dea8fcb859d296377c5263d83484d01cb8c39a26c679862fad68a10e36bc81e.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/7dea8fcb859d296377c5263d83484d01cb8c39a26c679862fad68a10e36bc81e.jpg)

![a8302d794d0efbff47657cebd98e64e7a638854ad44b7b6905e2b1d4f94511fc.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/a8302d794d0efbff47657cebd98e64e7a638854ad44b7b6905e2b1d4f94511fc.jpg)

![ac204afcd1b045052207e91380eb6a541e93b476404ebe4a6ffe848b791fee08.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/ac204afcd1b045052207e91380eb6a541e93b476404ebe4a6ffe848b791fee08.jpg)

![b50eb6c5670a9a4e3dbd5e6582f166fd2f4a8e8174e99fdb4f5f3f71d52c2f04.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/b50eb6c5670a9a4e3dbd5e6582f166fd2f4a8e8174e99fdb4f5f3f71d52c2f04.jpg)

![be102c2929565a5c9201bdd3a86b6ce002914435b6984e9c91bf13c65a2fd459.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/be102c2929565a5c9201bdd3a86b6ce002914435b6984e9c91bf13c65a2fd459.jpg)

![c31f97e7093930c881876ce95fb2794791437b874b4e1bb20e2be3ba0f24a8a5.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/c31f97e7093930c881876ce95fb2794791437b874b4e1bb20e2be3ba0f24a8a5.jpg)

![cf5914b9254943a3ee9852b5a1b0447c46a96a5e172b49da2b2b77719397545e.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/cf5914b9254943a3ee9852b5a1b0447c46a96a5e172b49da2b2b77719397545e.jpg)

![e276f2412058a281d2e6822d5f51f88bbc49b12ba686ad729b264607f97e8973.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/e276f2412058a281d2e6822d5f51f88bbc49b12ba686ad729b264607f97e8973.jpg)

![e5dfaddf8e4be66ed9054e8074290a1347997160c8900dc73e19e4b0452edc28.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/e5dfaddf8e4be66ed9054e8074290a1347997160c8900dc73e19e4b0452edc28.jpg)

![e9441f8ff8508321c6ce799119fa6f16e3ed152c95a7d7b18bd5e19818c803ee.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/e9441f8ff8508321c6ce799119fa6f16e3ed152c95a7d7b18bd5e19818c803ee.jpg)

![eb5ec05f0fada150fba57699e1105be74c9f49ef266263e196c48ce9a441ce07.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/eb5ec05f0fada150fba57699e1105be74c9f49ef266263e196c48ce9a441ce07.jpg)

![f1dc00bcece3471cb40970e4e38a9b83e00f4be3e011123d1021a6a67e5ec919.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/images/f1dc00bcece3471cb40970e4e38a9b83e00f4be3e011123d1021a6a67e5ec919.jpg)

### Tables

![1ec0d94ba08a778c7d592d294a206be2e7cafbe3f1355ae25adbd44ed2fd78dd.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/1ec0d94ba08a778c7d592d294a206be2e7cafbe3f1355ae25adbd44ed2fd78dd.jpg)

![2aeff54eb65b0348f50143abe782e0fb5ad8e0a0400d8a743e31cf636fbe488b.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/2aeff54eb65b0348f50143abe782e0fb5ad8e0a0400d8a743e31cf636fbe488b.jpg)

![2e54f4e3f3087557811d4d21097e327349f0de6606e7abbbfe6218445bf9f1d2.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/2e54f4e3f3087557811d4d21097e327349f0de6606e7abbbfe6218445bf9f1d2.jpg)

![4b39dc10610ffcb84072413bc589ce83ac35baa5f6a59d7dcce5b0c0a88af32c.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/4b39dc10610ffcb84072413bc589ce83ac35baa5f6a59d7dcce5b0c0a88af32c.jpg)

![4d709aa9452ddecd43107f144cbfe9fb8ba3d73763a1a6db79331dc49529268f.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/4d709aa9452ddecd43107f144cbfe9fb8ba3d73763a1a6db79331dc49529268f.jpg)

![524896573d5a8ff8e8dccacc2751a04348a245556041ba75480b9850e0042c4e.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/524896573d5a8ff8e8dccacc2751a04348a245556041ba75480b9850e0042c4e.jpg)

![52a87d2f236cec00ba35975bf7c7f9ec3b6e002847e12580b7490382f8954993.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/52a87d2f236cec00ba35975bf7c7f9ec3b6e002847e12580b7490382f8954993.jpg)

![57729b29c7b188a5d5d18be037f63ed6b1ed8598dbbf761a64480d8f468e1aab.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/57729b29c7b188a5d5d18be037f63ed6b1ed8598dbbf761a64480d8f468e1aab.jpg)

![5acb1254aa946d029c68f48506549a263b370ba7c10048d2358008fb6ed4da7f.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/5acb1254aa946d029c68f48506549a263b370ba7c10048d2358008fb6ed4da7f.jpg)

![779ff965c03314630e43aadcbd68b903c74797f05208da48e6c507a60fe49ef3.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/779ff965c03314630e43aadcbd68b903c74797f05208da48e6c507a60fe49ef3.jpg)

![7da7a7e5f0c90fc065fd2cf55474bf4bb4db140c96ba41af5c04940ee4b1901c.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/7da7a7e5f0c90fc065fd2cf55474bf4bb4db140c96ba41af5c04940ee4b1901c.jpg)

![ac96f1f8d29e1a21fdcf84ad9f949abf6f95376cf0aaf7c76fe399ed55c91ebf.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/ac96f1f8d29e1a21fdcf84ad9f949abf6f95376cf0aaf7c76fe399ed55c91ebf.jpg)

![ae34c54a3c7ba001251c6b21bafd032e51ffdb85ed32927ec58d3bc00fe43ed8.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/ae34c54a3c7ba001251c6b21bafd032e51ffdb85ed32927ec58d3bc00fe43ed8.jpg)

![bb4ead38239357b3d560671417b68518b2f605305b09e38f9a08e9eac2411dfc.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/bb4ead38239357b3d560671417b68518b2f605305b09e38f9a08e9eac2411dfc.jpg)

![bea29b8537edbe98b324c4e25afd63dcf11885a40ac6fd63f94f1d5f295ec3eb.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/bea29b8537edbe98b324c4e25afd63dcf11885a40ac6fd63f94f1d5f295ec3eb.jpg)

![bf7c864cb0af149e46c43ae1e09b31e102c4b1b8a21a5b318b30a70e74d30e1e.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/bf7c864cb0af149e46c43ae1e09b31e102c4b1b8a21a5b318b30a70e74d30e1e.jpg)

![c7c85fb317a51d7cef3b741367608bb8f6a6b01fe5386ba94207e50a5a58a2e0.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/c7c85fb317a51d7cef3b741367608bb8f6a6b01fe5386ba94207e50a5a58a2e0.jpg)

![e340f9889582eaef5ab44cb1b69fa52cf6b6646c393d1b690786b06953145e0c.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/e340f9889582eaef5ab44cb1b69fa52cf6b6646c393d1b690786b06953145e0c.jpg)

![e37fa2adcb5c6b8b02347b89bddaf3113064bdd728ba5274552c8c07a0b40061.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/e37fa2adcb5c6b8b02347b89bddaf3113064bdd728ba5274552c8c07a0b40061.jpg)

![e873f2d108807a567d967ae5731a967535a4aa289d81344c39e7fb446c0bf4cc.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/e873f2d108807a567d967ae5731a967535a4aa289d81344c39e7fb446c0bf4cc.jpg)

![fea1f542654ff4f779a07ddb0a392afda9c965edd23b0b12f9ab1b43192ff054.jpg](../icml_results/2285_Enhancing%20LLM%20Training%20via%20Spectral%20Clipping/tables/fea1f542654ff4f779a07ddb0a392afda9c965edd23b0b12f9ab1b43192ff054.jpg)

<a id="paper-2257"></a>

## Efficient Transformer Attention for SNNs via Hadamard Simplification


### Images

![30f2e3bd4d374cdb5587088b679ded17a7d99e9d2f7ef0a197d0c4db0faad59d.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/images/30f2e3bd4d374cdb5587088b679ded17a7d99e9d2f7ef0a197d0c4db0faad59d.jpg)

![d822348d546507fceb1784a4c4f5c968b5139751cb7b9f7ea6c7570ecdc7a635.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/images/d822348d546507fceb1784a4c4f5c968b5139751cb7b9f7ea6c7570ecdc7a635.jpg)

![f6a4c568d685fd80eab345d2871f6a0e9a11272d8479739df4a5eeaf1e903923.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/images/f6a4c568d685fd80eab345d2871f6a0e9a11272d8479739df4a5eeaf1e903923.jpg)

### Tables

![15ebfd4b712043b6aada5a8a8d101f9d82b404305f2b22afae960370e384547b.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/15ebfd4b712043b6aada5a8a8d101f9d82b404305f2b22afae960370e384547b.jpg)

![3b44696e3afacfe1ed859f469a7f579124a025e3721848c2e1d3069f1a010344.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/3b44696e3afacfe1ed859f469a7f579124a025e3721848c2e1d3069f1a010344.jpg)

![6f19c4d3c471ac93fceb2a63c70f0617e9bbc5427f57557be0c36f8c54f40829.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/6f19c4d3c471ac93fceb2a63c70f0617e9bbc5427f57557be0c36f8c54f40829.jpg)

![d5da7ae6a91346b57fcdd40b534152e143ea470493e921c29009de794c2cea34.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/d5da7ae6a91346b57fcdd40b534152e143ea470493e921c29009de794c2cea34.jpg)

![d9161325e9016a943caec23250d41e4cc8b48ea8a19f2a289335ac210dd441a4.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/d9161325e9016a943caec23250d41e4cc8b48ea8a19f2a289335ac210dd441a4.jpg)

![e2ceee5df5a3c2a8a4cf86627a5c221181296a253ca165e8fdae8f4441ffef90.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/e2ceee5df5a3c2a8a4cf86627a5c221181296a253ca165e8fdae8f4441ffef90.jpg)

![fef256c4b2034128f757f69ce3bbac27d55283f775d5f5a019191b8d623ba322.jpg](../icml_results/2286_Efficient%20Transformer%20Attention%20for%20SNNs%20via%20Hadamard%20Simplification/tables/fef256c4b2034128f757f69ce3bbac27d55283f775d5f5a019191b8d623ba322.jpg)

<a id="paper-2258"></a>

## Transformed Latent Variable Multi-Output Gaussian Processes


### Images

![22499418275a56b9e0fb5f95673fa34cd3037f1929d25fa078d686a49bf9040b.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/22499418275a56b9e0fb5f95673fa34cd3037f1929d25fa078d686a49bf9040b.jpg)

![2ffee980a88a8fca45fceb3558239f958027563755fa975ad10b1d0b9fdfab1f.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/2ffee980a88a8fca45fceb3558239f958027563755fa975ad10b1d0b9fdfab1f.jpg)

![32484861329a95c593f999dd63854e3beb8e84dae7754bd81aea4acf3704eac0.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/32484861329a95c593f999dd63854e3beb8e84dae7754bd81aea4acf3704eac0.jpg)

![3ffaa841ac618f1fbf94d6396123941f14759610ce30f710dc1d8ac78e430fbb.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/3ffaa841ac618f1fbf94d6396123941f14759610ce30f710dc1d8ac78e430fbb.jpg)

![8d32eaa12ad071442894fe0a2f8547e9cde595ec8cec5db2cd9b99bd9ce32f6e.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/8d32eaa12ad071442894fe0a2f8547e9cde595ec8cec5db2cd9b99bd9ce32f6e.jpg)

![9b1047ad4b9e49639d40f506b0b9c80f7a84a9cb9eb528d69a740ece17b826a5.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/9b1047ad4b9e49639d40f506b0b9c80f7a84a9cb9eb528d69a740ece17b826a5.jpg)

![9bd9b5ac79b0ae792051f8f7a7d5a34be80a54e52f6b92efd0ae17e4bf108336.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/9bd9b5ac79b0ae792051f8f7a7d5a34be80a54e52f6b92efd0ae17e4bf108336.jpg)

![a1ff907840b68b9f36aa3340372f89f29107cd6d0044e8b09150cdd93b73f63e.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/a1ff907840b68b9f36aa3340372f89f29107cd6d0044e8b09150cdd93b73f63e.jpg)

![a4d6dddc8486b04a630e50085818f59d6ab87ff1603649629cc99d1c97d5f264.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/a4d6dddc8486b04a630e50085818f59d6ab87ff1603649629cc99d1c97d5f264.jpg)

![d37f09d15ebd1a47e93bd4dd83bb3cef1d608ee5814dc131618f9590ced96d03.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/images/d37f09d15ebd1a47e93bd4dd83bb3cef1d608ee5814dc131618f9590ced96d03.jpg)

### Tables

![0a49726b4183382e23cb29389f254d913aba11183f792f29c05d548da47be877.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/0a49726b4183382e23cb29389f254d913aba11183f792f29c05d548da47be877.jpg)

![0ba4239ce31b1083331ae9a8608fab0bc52f750880adbd1715e3e10a87a5b42a.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/0ba4239ce31b1083331ae9a8608fab0bc52f750880adbd1715e3e10a87a5b42a.jpg)

![157e32b8570f3a278737610ef1dfe0bf026670a4f795fa47889561e8e0b51fcf.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/157e32b8570f3a278737610ef1dfe0bf026670a4f795fa47889561e8e0b51fcf.jpg)

![297abce3161c3da7e65239a42301f3d1e6d0241e06cbfb4c04accd25ac004acc.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/297abce3161c3da7e65239a42301f3d1e6d0241e06cbfb4c04accd25ac004acc.jpg)

![2b5dbabc64b9ba91850a9d3bd349b3d0212521e4ed6a06a576cc4e2c8f3135eb.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/2b5dbabc64b9ba91850a9d3bd349b3d0212521e4ed6a06a576cc4e2c8f3135eb.jpg)

![308b781412f3f9fa8d637e5761afc044a407c05316b9cb20a2af181cedebed1d.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/308b781412f3f9fa8d637e5761afc044a407c05316b9cb20a2af181cedebed1d.jpg)

![3f0bc0e3a97851786e70c83b5df723609d6c55967245a0fcf72b0063f464e6bc.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/3f0bc0e3a97851786e70c83b5df723609d6c55967245a0fcf72b0063f464e6bc.jpg)

![4ffc6461e74020e927a33d1bf7f00bdae1afeff9627320a730f256c0d4e50562.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/4ffc6461e74020e927a33d1bf7f00bdae1afeff9627320a730f256c0d4e50562.jpg)

![50f565dbde27918cf78c688579a9f6a27c3935b2fdb8161772e2f7c6e3a8c07f.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/50f565dbde27918cf78c688579a9f6a27c3935b2fdb8161772e2f7c6e3a8c07f.jpg)

![55cfb691fa6cc984f5b9aeed891820f839824a4c58400ef9a34b68732b707525.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/55cfb691fa6cc984f5b9aeed891820f839824a4c58400ef9a34b68732b707525.jpg)

![56cdbec4d9a7a74b979ccc138f0a88277bcaea22a2bc456d5c66a06714f9af86.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/56cdbec4d9a7a74b979ccc138f0a88277bcaea22a2bc456d5c66a06714f9af86.jpg)

![5833fcda0d0b89ffee576bed092b2e1167fd34c796cec2537e799f29252b8594.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/5833fcda0d0b89ffee576bed092b2e1167fd34c796cec2537e799f29252b8594.jpg)

![5882d9c9212ede7f562e8543339383423a113f0b2244714413d997bf1b666390.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/5882d9c9212ede7f562e8543339383423a113f0b2244714413d997bf1b666390.jpg)

![5b22775dd95201be10879c8ce3708aa43b096aaef08fc9383788892b23b32f70.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/5b22775dd95201be10879c8ce3708aa43b096aaef08fc9383788892b23b32f70.jpg)

![5c643633cb7b94a09ee06966423c2b4ce8900a34b8ac65fa07270820b9f5a8a6.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/5c643633cb7b94a09ee06966423c2b4ce8900a34b8ac65fa07270820b9f5a8a6.jpg)

![6561e2c392184b8fc1e099496335e5a80fc25218ed125d2003ab840a4fd0b1c8.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/6561e2c392184b8fc1e099496335e5a80fc25218ed125d2003ab840a4fd0b1c8.jpg)

![677ec8290b4b649d955c454ffbcd070918465d9b95634c8d3d17fdbdbacbb7f5.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/677ec8290b4b649d955c454ffbcd070918465d9b95634c8d3d17fdbdbacbb7f5.jpg)

![9a4bea93aad0e02755214f9bfe4806d530d90a875b2ff6718b5815640847f1d9.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/9a4bea93aad0e02755214f9bfe4806d530d90a875b2ff6718b5815640847f1d9.jpg)

![9e0578cc142837bb54125da1ecb71e26f1e44350a4fef10874d59ece28de3544.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/9e0578cc142837bb54125da1ecb71e26f1e44350a4fef10874d59ece28de3544.jpg)

![aa5597cf5a30785b1c53b65ddf321ddead7c0442d90c315e2a9c6b96c256d5cb.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/aa5597cf5a30785b1c53b65ddf321ddead7c0442d90c315e2a9c6b96c256d5cb.jpg)

![c065ab7fb4714967557c3344989160929911ef70643d7c5fa8ba3d78929fcd9c.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/c065ab7fb4714967557c3344989160929911ef70643d7c5fa8ba3d78929fcd9c.jpg)

![c7baecc1079b6c13e50282afd09258abaa3b4a5d13b1fb8b650051b9efe379fc.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/c7baecc1079b6c13e50282afd09258abaa3b4a5d13b1fb8b650051b9efe379fc.jpg)

![c813ea16e70a994670a1b775cb898db2b86f7a074d118944d4a6831cb462237f.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/c813ea16e70a994670a1b775cb898db2b86f7a074d118944d4a6831cb462237f.jpg)

![c8b8405fd949b86fbbf9fab0d55eb1a45345e31d04dc2b8d0414c6f9b44612ec.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/c8b8405fd949b86fbbf9fab0d55eb1a45345e31d04dc2b8d0414c6f9b44612ec.jpg)

![e8068b4f463cb91ae94691f1d52d4884088a196a002c4513f336836daefcced0.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/e8068b4f463cb91ae94691f1d52d4884088a196a002c4513f336836daefcced0.jpg)

![f745d51d1a77d3a5fa6cf023197e182f575328ff134ac119edc44fdf2843aab2.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/f745d51d1a77d3a5fa6cf023197e182f575328ff134ac119edc44fdf2843aab2.jpg)

![feabe2fae9d34819bcf3c1e48d3f4fd0026c77a9472ff27c58c820112ef04c2b.jpg](../icml_results/2287_Transformed%20Latent%20Variable%20Multi-Output%20Gaussian%20Processes/tables/feabe2fae9d34819bcf3c1e48d3f4fd0026c77a9472ff27c58c820112ef04c2b.jpg)

<a id="paper-2259"></a>

## XSkill: Continual Learning from Experience and Skills in Multimodal Agents


### Images

![0476da778e4bf36e3040e7ae0316740abbac125ddaa4a75af9c134a6310bce95.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/0476da778e4bf36e3040e7ae0316740abbac125ddaa4a75af9c134a6310bce95.jpg)

![2a90a9e091dbba95f885c4d677b40b213ff54a825b3ae1d8142a404cf2a43371.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/2a90a9e091dbba95f885c4d677b40b213ff54a825b3ae1d8142a404cf2a43371.jpg)

![386fcba5b3d859b578960f7e60c158e9fa87ea1669600e183f191d26919a5b81.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/386fcba5b3d859b578960f7e60c158e9fa87ea1669600e183f191d26919a5b81.jpg)

![46fa6ac6c4c12a4f2e506981e32550469ccb793bb49f6acbcce6a7ad49bd58f7.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/46fa6ac6c4c12a4f2e506981e32550469ccb793bb49f6acbcce6a7ad49bd58f7.jpg)

![4dba3bcc0b6431af74db96c79aa665a7b2af5e9826962c3806bcc6fdad9b1c82.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/4dba3bcc0b6431af74db96c79aa665a7b2af5e9826962c3806bcc6fdad9b1c82.jpg)

![7d346a8a4777bf30da94c1de63b6aa937aa344371d782edd8841b7296fbae562.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/7d346a8a4777bf30da94c1de63b6aa937aa344371d782edd8841b7296fbae562.jpg)

![e58ec32d1f00e1e72efd372b1b6e20863e57f6b1fc591a62c104a800452ed171.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/e58ec32d1f00e1e72efd372b1b6e20863e57f6b1fc591a62c104a800452ed171.jpg)

![f765bc7393081e1e2ace98f9f0d34acb3bd5a917a7b3bc6b0c7933c0b86bf738.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/images/f765bc7393081e1e2ace98f9f0d34acb3bd5a917a7b3bc6b0c7933c0b86bf738.jpg)

### Tables

![063030379685ebb3171c9594ec7486c59a4f71bd2a93aaa1b9b3e21d515187f7.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/063030379685ebb3171c9594ec7486c59a4f71bd2a93aaa1b9b3e21d515187f7.jpg)

![06491d6a78659cde483991a66d9c467bb75d8adbb9d8ee3943ca91b53c6418e6.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/06491d6a78659cde483991a66d9c467bb75d8adbb9d8ee3943ca91b53c6418e6.jpg)

![085988b9fd3e028d01594e65d9abae21a25c6c614deeeab593abc2f625656ed8.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/085988b9fd3e028d01594e65d9abae21a25c6c614deeeab593abc2f625656ed8.jpg)

![27de96bf5dd3555d80ce076054c0ba3f3911f6858753452c20d1777a94fc5592.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/27de96bf5dd3555d80ce076054c0ba3f3911f6858753452c20d1777a94fc5592.jpg)

![5446a1a2aca237f38ddcf10e7ab981d6102918e60484b08f1d7994d6a1363020.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/5446a1a2aca237f38ddcf10e7ab981d6102918e60484b08f1d7994d6a1363020.jpg)

![5bebd20f7ec1e26656148c08db8d7c89c3384d9000194acbc4d7d6af8676034d.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/5bebd20f7ec1e26656148c08db8d7c89c3384d9000194acbc4d7d6af8676034d.jpg)

![89edb7b0e488d86855721e07443ba516fb73cc014c53890d04e30ba8cb2c2655.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/89edb7b0e488d86855721e07443ba516fb73cc014c53890d04e30ba8cb2c2655.jpg)

![98c04dfa1de88c74400e85e1212969fd273aeb268305de09cc6d5e3a74f845d4.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/98c04dfa1de88c74400e85e1212969fd273aeb268305de09cc6d5e3a74f845d4.jpg)

![9cc5e178b202e0968328dedd46fc7ed04a1f4eb5804e403f5a19520c667be19f.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/9cc5e178b202e0968328dedd46fc7ed04a1f4eb5804e403f5a19520c667be19f.jpg)

![bd6cafe0a00c986321c106374facf64649c9f895c06ed1be7c209293c6268474.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/bd6cafe0a00c986321c106374facf64649c9f895c06ed1be7c209293c6268474.jpg)

![d4fe114865589f0a8ae9f429cfca1f2c9a3f8fe5528c94e267de36074daaa51e.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/d4fe114865589f0a8ae9f429cfca1f2c9a3f8fe5528c94e267de36074daaa51e.jpg)

![e6679d2fa8afe91e5011663a812ec5ade5cce402549369dd4f2131da3bb650b0.jpg](../icml_results/2288_XSkill_%20Continual%20Learning%20from%20Experience%20and%20Skills%20in%20Multimodal%20Agents/tables/e6679d2fa8afe91e5011663a812ec5ade5cce402549369dd4f2131da3bb650b0.jpg)

<a id="paper-2260"></a>

## Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit


### Images

![0c0b136c251f222ba38864c6833fcb211efa30989978a0fafd0f299e666f01e9.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/0c0b136c251f222ba38864c6833fcb211efa30989978a0fafd0f299e666f01e9.jpg)

![1753603c2f2fb62f64556e9e3b7143dab9fa058a3417ae4fb5a45c0dd3debdc2.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/1753603c2f2fb62f64556e9e3b7143dab9fa058a3417ae4fb5a45c0dd3debdc2.jpg)

![214a82868e7be2535b415848668bd1e30675199852c199a665b2a0b877a1a7f6.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/214a82868e7be2535b415848668bd1e30675199852c199a665b2a0b877a1a7f6.jpg)

![2f0a3900b17318cf7cc97c1635572545ba8c33a7eb78f9a6c843ab9691f29dc1.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/2f0a3900b17318cf7cc97c1635572545ba8c33a7eb78f9a6c843ab9691f29dc1.jpg)

![36d39929868a2d0ab626ad30ce469ed0f3c06322680e675af49dbe1890652838.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/36d39929868a2d0ab626ad30ce469ed0f3c06322680e675af49dbe1890652838.jpg)

![3dbdeb95ca75822f706fa42b7451d4469bf82d3addb4a3de3f0c445fca500bb7.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/3dbdeb95ca75822f706fa42b7451d4469bf82d3addb4a3de3f0c445fca500bb7.jpg)

![3ef0b75308c1d66313cefc5fccb0cdbe49cc37c3c8e2a72c9ad67df5f47e3b38.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/3ef0b75308c1d66313cefc5fccb0cdbe49cc37c3c8e2a72c9ad67df5f47e3b38.jpg)

![4605dbd7ffed81b85b8963b973d56395acecf6aca59a890203d389194fb9a9dc.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/4605dbd7ffed81b85b8963b973d56395acecf6aca59a890203d389194fb9a9dc.jpg)

![47510bb00dc9846a8fec4adb164ddad22abf865d112e3f9dbe83330c6f54c98e.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/47510bb00dc9846a8fec4adb164ddad22abf865d112e3f9dbe83330c6f54c98e.jpg)

![4c91cd929223fffd174b6ac5078f6e39c60d58b1ab6b4f4f561df93776d708be.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/4c91cd929223fffd174b6ac5078f6e39c60d58b1ab6b4f4f561df93776d708be.jpg)

![5d8cc6d607b67f973489c7225d8755163f94c4bb041e0c2adc469703fd3df736.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/5d8cc6d607b67f973489c7225d8755163f94c4bb041e0c2adc469703fd3df736.jpg)

![637762b8845ed71f6bea9536092619856def8c9f425e4ceedecc3a1d5412c7cf.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/637762b8845ed71f6bea9536092619856def8c9f425e4ceedecc3a1d5412c7cf.jpg)

![72bde441c2c5e6ac9ff1dad6fd990f793657f766328fa78a7882a26dcc992341.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/72bde441c2c5e6ac9ff1dad6fd990f793657f766328fa78a7882a26dcc992341.jpg)

![7ccf52620ce3feab8e87ad32a058036e38f4df4a2df91c308fdbad37c3aad434.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/7ccf52620ce3feab8e87ad32a058036e38f4df4a2df91c308fdbad37c3aad434.jpg)

![8579dfbfa9a9ee421541a2ec6807c15dd2b30097348bd0b1aeafde902cb5e3f8.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/8579dfbfa9a9ee421541a2ec6807c15dd2b30097348bd0b1aeafde902cb5e3f8.jpg)

![90754d267c14a011b1227f4dcf5dc62a34f32918615c07dd2ba60c0b281b9bf3.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/90754d267c14a011b1227f4dcf5dc62a34f32918615c07dd2ba60c0b281b9bf3.jpg)

![9661bcb9959087f7c2541b88a7bf7db12573bdba5c53202bd39ac0cbdb322552.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/9661bcb9959087f7c2541b88a7bf7db12573bdba5c53202bd39ac0cbdb322552.jpg)

![98ebd07bfe80a573d1df897ec77921830199d53dbd9743ec33ed57e8362c5442.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/98ebd07bfe80a573d1df897ec77921830199d53dbd9743ec33ed57e8362c5442.jpg)

![9b242b20a523dde898737c49e2f14d0ea38cae0365c8d55f7cb77ac15a559046.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/9b242b20a523dde898737c49e2f14d0ea38cae0365c8d55f7cb77ac15a559046.jpg)

![a501b3e1520a446608392d446ac4cd3819c562d99ffde2c01b1888750f93fcc5.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/a501b3e1520a446608392d446ac4cd3819c562d99ffde2c01b1888750f93fcc5.jpg)

![acb898ed6276b5ecc0aa64f225281559508b970877b7c8dc4267b14ea9996e8c.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/acb898ed6276b5ecc0aa64f225281559508b970877b7c8dc4267b14ea9996e8c.jpg)

![ae7cc64ce9e7d4dbc18473c29072caf3aa387aec735b3e3f506898a4b32f76f2.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/ae7cc64ce9e7d4dbc18473c29072caf3aa387aec735b3e3f506898a4b32f76f2.jpg)

![afc0b0d605ee04329706c0a4c1d075d5d9037826cf6f10dfaaa0bda4468e070f.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/afc0b0d605ee04329706c0a4c1d075d5d9037826cf6f10dfaaa0bda4468e070f.jpg)

![b4fd8d00409ed82e0f9b7ea46c0638a2d41f3fe8c4fa5ce787c8559e6d4fd007.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/b4fd8d00409ed82e0f9b7ea46c0638a2d41f3fe8c4fa5ce787c8559e6d4fd007.jpg)

![ba65532ce6ce64c7d26ec14117c0cf8584e34ec7b9794afa8dab882130d98eed.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/ba65532ce6ce64c7d26ec14117c0cf8584e34ec7b9794afa8dab882130d98eed.jpg)

![bf4fa9f96053d1d0e4cfe3223ce4f4360d13e8b01cc8feb63e9edb74f175558d.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/bf4fa9f96053d1d0e4cfe3223ce4f4360d13e8b01cc8feb63e9edb74f175558d.jpg)

![c139f1e2d2f026a5177f80e5b42553b3fb068f794bd40d99f27567f6939ab62d.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/c139f1e2d2f026a5177f80e5b42553b3fb068f794bd40d99f27567f6939ab62d.jpg)

![d03eaa48c6c5581d181e71281212556a155dba13ef29f643a48333f02220caf6.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/d03eaa48c6c5581d181e71281212556a155dba13ef29f643a48333f02220caf6.jpg)

![d0b06c27a6474373acbdf6f368c6bae55cc63cad7083d13775ccb2a8e14c8e07.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/d0b06c27a6474373acbdf6f368c6bae55cc63cad7083d13775ccb2a8e14c8e07.jpg)

![d4cd77a44629245807bb3210b190e3a60366a45f259dd18fe69cb4779f07c119.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/d4cd77a44629245807bb3210b190e3a60366a45f259dd18fe69cb4779f07c119.jpg)

![d697ea25a570b942639c56534f8fc9180d75e9aba3d7a4abd633c8329c89808f.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/d697ea25a570b942639c56534f8fc9180d75e9aba3d7a4abd633c8329c89808f.jpg)

![e61a1d92fde0c8723032d15e772625199a4457e4ad5c4a915614bfeebf524c21.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/e61a1d92fde0c8723032d15e772625199a4457e4ad5c4a915614bfeebf524c21.jpg)

![ef5a5cfa08bb41004355a0735f8ce28e2938a1839d36c9b708d14d4a78be21d7.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/ef5a5cfa08bb41004355a0735f8ce28e2938a1839d36c9b708d14d4a78be21d7.jpg)

![f10da2abf35915c94e3d616bef380690d040b78be8c03595c7fc711608c03c92.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/f10da2abf35915c94e3d616bef380690d040b78be8c03595c7fc711608c03c92.jpg)

![f1b5bfb8323ef4b58e9e8af6fa8f5752c056881d5fafafa8a1c3bfa72077acb0.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/f1b5bfb8323ef4b58e9e8af6fa8f5752c056881d5fafafa8a1c3bfa72077acb0.jpg)

![f2bc04c7a77532be82f6309fac5f8a1bd249768ed188ea67676151a71c6b7867.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/f2bc04c7a77532be82f6309fac5f8a1bd249768ed188ea67676151a71c6b7867.jpg)

![f79d886a8da7eca8db46919d25324bed92a074d73aad09afd7c7f4bdccb40898.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/images/f79d886a8da7eca8db46919d25324bed92a074d73aad09afd7c7f4bdccb40898.jpg)

### Tables

![05cd2e13fdabd5746ab8a846c2f0a1e79efa390f8472d315ac682fdb1ec1d54e.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/05cd2e13fdabd5746ab8a846c2f0a1e79efa390f8472d315ac682fdb1ec1d54e.jpg)

![0be3917221a2365aff1a8872444023db11387415a4d6f3f889e90804121f1244.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/0be3917221a2365aff1a8872444023db11387415a4d6f3f889e90804121f1244.jpg)

![0fe0b8ee9d76c10201c533466711f4a8a4a49a360d49b12c19455da030dd3582.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/0fe0b8ee9d76c10201c533466711f4a8a4a49a360d49b12c19455da030dd3582.jpg)

![147e3189538ac5f612fad8928736a6846885406b30f144d7151b2512bfd0a394.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/147e3189538ac5f612fad8928736a6846885406b30f144d7151b2512bfd0a394.jpg)

![18b5b6e0424f2678e5b16c88e5a318400843725ba35bccef0905de4cc9837c2d.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/18b5b6e0424f2678e5b16c88e5a318400843725ba35bccef0905de4cc9837c2d.jpg)

![1a77b536f3ed4851aef39c91d4ae3911837693a4364a5bd1d977b161a8cddfcc.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/1a77b536f3ed4851aef39c91d4ae3911837693a4364a5bd1d977b161a8cddfcc.jpg)

![2c11791ad1b7fbfd46b4f2c8e863e9c69b7950843cc8d69ec80c562cf2a7bdaa.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/2c11791ad1b7fbfd46b4f2c8e863e9c69b7950843cc8d69ec80c562cf2a7bdaa.jpg)

![3546ffd8f7466172ee559b15a308bd42afc6b65996ab65b7ac55d869bf0fc59b.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/3546ffd8f7466172ee559b15a308bd42afc6b65996ab65b7ac55d869bf0fc59b.jpg)

![3f51dd1a230ef78746e56317d2279fb773a37c5fa4a8ff8f030264bd5af649dc.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/3f51dd1a230ef78746e56317d2279fb773a37c5fa4a8ff8f030264bd5af649dc.jpg)

![408bcdb09e9f0e2ef34cb9cc87bb30c2598a3b9b5a17d1bbf39df151364397b7.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/408bcdb09e9f0e2ef34cb9cc87bb30c2598a3b9b5a17d1bbf39df151364397b7.jpg)

![43a3a3ab81313026bc27ad159e82e53313cb98ae15cbf04f0d773b8ab266315b.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/43a3a3ab81313026bc27ad159e82e53313cb98ae15cbf04f0d773b8ab266315b.jpg)

![52097fe5d49fe61fead0c6eb6ff93d1344f3260d55240614e583a3922835949c.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/52097fe5d49fe61fead0c6eb6ff93d1344f3260d55240614e583a3922835949c.jpg)

![66e64b0f9b276b7f845fff8a8fa9a62be8325996b1f7fbed5411148485071712.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/66e64b0f9b276b7f845fff8a8fa9a62be8325996b1f7fbed5411148485071712.jpg)

![6b775cf0b357c37afac0507e84780a97c722b5d63998045a9d7a27178a5cd854.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/6b775cf0b357c37afac0507e84780a97c722b5d63998045a9d7a27178a5cd854.jpg)

![6bd3d0bda10cd523500fc6d3123a8fd65f21a2b59c07df3da964ae731e051eac.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/6bd3d0bda10cd523500fc6d3123a8fd65f21a2b59c07df3da964ae731e051eac.jpg)

![71dd97f38f89e5aff3b38520d07f2dfbc25992e3fd601bab752419b5a527df4d.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/71dd97f38f89e5aff3b38520d07f2dfbc25992e3fd601bab752419b5a527df4d.jpg)

![74a47393b1cb344d64018c12482998bb87b17890ac03ca9fd723e36e82dd3653.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/74a47393b1cb344d64018c12482998bb87b17890ac03ca9fd723e36e82dd3653.jpg)

![815a1047c1f9eaa5234213231632ad54c5c02f58aa8ae76ab9f9c98201ab3a2c.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/815a1047c1f9eaa5234213231632ad54c5c02f58aa8ae76ab9f9c98201ab3a2c.jpg)

![86e8be93fa57618dcc76b9c8f864c8911648d2d62cf1ae7ab2e4d9011abf64fb.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/86e8be93fa57618dcc76b9c8f864c8911648d2d62cf1ae7ab2e4d9011abf64fb.jpg)

![8a7725981f908ae9a1b7a896e311b9a59043de16fb89a300ad3dddca923cf5cd.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/8a7725981f908ae9a1b7a896e311b9a59043de16fb89a300ad3dddca923cf5cd.jpg)

![98a0fc542074474b9cd660f55d1fb6714a0b6c83b3118899af11073fe37564a5.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/98a0fc542074474b9cd660f55d1fb6714a0b6c83b3118899af11073fe37564a5.jpg)

![9b98ea3fec88935c212fcf59d5450f70de64d213631019a364c31dac8f968008.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/9b98ea3fec88935c212fcf59d5450f70de64d213631019a364c31dac8f968008.jpg)

![9cd58b52062d0387b26b69982193a8d2c9534e5a4febe8506d69f724b7aa3c51.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/9cd58b52062d0387b26b69982193a8d2c9534e5a4febe8506d69f724b7aa3c51.jpg)

![a209bb75dd34a72e7941504df91e2e1cd9c640d12067eeeeff84df48e076a280.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/a209bb75dd34a72e7941504df91e2e1cd9c640d12067eeeeff84df48e076a280.jpg)

![b86ad107414a5667ea96b9a956143a484476147337cbe33c4f40e36e73419d34.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/b86ad107414a5667ea96b9a956143a484476147337cbe33c4f40e36e73419d34.jpg)

![bb08be3c27f1bd946b81bc6c6c6cd8404ead710c8a890cc358fb210fa2656526.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/bb08be3c27f1bd946b81bc6c6c6cd8404ead710c8a890cc358fb210fa2656526.jpg)

![ca3c8b57a9a1efe1cdae0689c467bf35a43156d0b665d0c73e604132ba465cc9.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/ca3c8b57a9a1efe1cdae0689c467bf35a43156d0b665d0c73e604132ba465cc9.jpg)

![d0f5bfbbfe56d50426737f75432382cd3f5abbfd835b64e2873cee05e9e8208c.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/d0f5bfbbfe56d50426737f75432382cd3f5abbfd835b64e2873cee05e9e8208c.jpg)

![d101c717cd8dc1aeba06399b4c11f85ab90516750847985b0eb5f4ae329a5a29.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/d101c717cd8dc1aeba06399b4c11f85ab90516750847985b0eb5f4ae329a5a29.jpg)

![d98302ecbd7aec69b584edc087ad0b9dd9baea65e4ab54453752424ccd801145.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/d98302ecbd7aec69b584edc087ad0b9dd9baea65e4ab54453752424ccd801145.jpg)

![e0a28394b88f22e4559c3269d68be459b7f74dba472f4303eb34c8e9f6527837.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/e0a28394b88f22e4559c3269d68be459b7f74dba472f4303eb34c8e9f6527837.jpg)

![e27003229db43ddf7468032a5d64cee3b25e8dfa63687ec6bd693324fec6dc0b.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/e27003229db43ddf7468032a5d64cee3b25e8dfa63687ec6bd693324fec6dc0b.jpg)

![e513e738dfb86b54b048a0fdd549827161571f5c081153eb2a71f2f8098aba76.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/e513e738dfb86b54b048a0fdd549827161571f5c081153eb2a71f2f8098aba76.jpg)

![efbb61c1276151cfe1b42c46655a4e7f9f60d7f7574070ca09f7f3e96561c35e.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/efbb61c1276151cfe1b42c46655a4e7f9f60d7f7574070ca09f7f3e96561c35e.jpg)

![f458a8c25c65e294a46c271f076ea7fd5c15c93e84b83baacde0bd57e50fe5bc.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/f458a8c25c65e294a46c271f076ea7fd5c15c93e84b83baacde0bd57e50fe5bc.jpg)

![f463d9691a0259a181e331ade213ca35d1eb0dd3ef4eb5551deb51a303e8dc2c.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/f463d9691a0259a181e331ade213ca35d1eb0dd3ef4eb5551deb51a303e8dc2c.jpg)

![f6cc54c0af0be5384e54bc2116e141e1fcc2765c298bbb73114a6373b516839a.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/f6cc54c0af0be5384e54bc2116e141e1fcc2765c298bbb73114a6373b516839a.jpg)

![fe5be8e33474d65b83792afefc027aac5439d94574992270a3429380f57faf66.jpg](../icml_results/2289_Interpretable%20Embeddings%20with%20Sparse%20Autoencoders_%20A%20Data%20Analysis%20Toolkit/tables/fe5be8e33474d65b83792afefc027aac5439d94574992270a3429380f57faf66.jpg)

<a id="paper-2261"></a>

## AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks


### Images

![0b3f28d1f22125f3e6f1d67f5512f3c3b6c7f38141b5ce552f6ab2be2794c5b6.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/images/0b3f28d1f22125f3e6f1d67f5512f3c3b6c7f38141b5ce552f6ab2be2794c5b6.jpg)

![5a8abeeb7c3c668012a7d42f0ded01c1e0aeb2f81283b6bbf5e349daa7fe1e9e.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/images/5a8abeeb7c3c668012a7d42f0ded01c1e0aeb2f81283b6bbf5e349daa7fe1e9e.jpg)

![62e091ad4d931a5d3cbe5dfa6af095f345e594fe9582cb292f25366c5aa6c509.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/images/62e091ad4d931a5d3cbe5dfa6af095f345e594fe9582cb292f25366c5aa6c509.jpg)

![6fe7352f63ae54d4ae0e486f9d05df82f4761eba9ebda22f073684c6d173d85e.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/images/6fe7352f63ae54d4ae0e486f9d05df82f4761eba9ebda22f073684c6d173d85e.jpg)

![94f4d830ea67e946f51236518df0d5220ac1d3224e1e8f58be259a05be61566a.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/images/94f4d830ea67e946f51236518df0d5220ac1d3224e1e8f58be259a05be61566a.jpg)

![e07cf9e21f61adb5cb6bffd8bd1f498276c924d009e81995e20ffd9492b90fe9.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/images/e07cf9e21f61adb5cb6bffd8bd1f498276c924d009e81995e20ffd9492b90fe9.jpg)

### Tables

![14404ca9c4089329eb9e111b0ff5d720ee1c9b5468c1ba74f63f07674c7a64f7.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/tables/14404ca9c4089329eb9e111b0ff5d720ee1c9b5468c1ba74f63f07674c7a64f7.jpg)

![7116633ccf8a703a19735b160b0c1685bdf1787c7e3f0b9321c5881ee536e49d.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/tables/7116633ccf8a703a19735b160b0c1685bdf1787c7e3f0b9321c5881ee536e49d.jpg)

![7cbb8267dfb26f3f6e79e8a215d711778aa8740905e299c2f40d47a625bb4ecc.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/tables/7cbb8267dfb26f3f6e79e8a215d711778aa8740905e299c2f40d47a625bb4ecc.jpg)

![92ae690166150e719e164c3aa657dac38161bf0d930aade9b0ff0b7294d33f48.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/tables/92ae690166150e719e164c3aa657dac38161bf0d930aade9b0ff0b7294d33f48.jpg)

![eb42db073f64fdc371da1c83249551f29d89096892fd36c74935ae67b26977d7.jpg](../icml_results/2290_AgentLAB_%20Benchmarking%20LLM%20Agents%20against%20Long-Horizon%20Attacks/tables/eb42db073f64fdc371da1c83249551f29d89096892fd36c74935ae67b26977d7.jpg)

<a id="paper-2262"></a>

## Spatial Conformal Inference through Localized Quantile Regression


### Images

![436388f438d5f57bbdafa2c0c885e0b5d12fca72e3c18cfed1549d327f3fc5a2.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/436388f438d5f57bbdafa2c0c885e0b5d12fca72e3c18cfed1549d327f3fc5a2.jpg)

![522ae19ecafad9ab9da7c8fd18d9a2c6bc594aed7f3383a3c9620ac876fef28a.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/522ae19ecafad9ab9da7c8fd18d9a2c6bc594aed7f3383a3c9620ac876fef28a.jpg)

![59d7dde5e34904b56d4f93447ae95f7c72d797a2de6898fb56d9a2807f67b246.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/59d7dde5e34904b56d4f93447ae95f7c72d797a2de6898fb56d9a2807f67b246.jpg)

![6094ffb85fd3a8841da94e65b1a442c374fc2cf2ab61d6c5be0514f9d63998b9.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/6094ffb85fd3a8841da94e65b1a442c374fc2cf2ab61d6c5be0514f9d63998b9.jpg)

![68d9a605a5d02a4215154d80271e5b9e73727ef6bb53fed73d563b6442b7cd94.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/68d9a605a5d02a4215154d80271e5b9e73727ef6bb53fed73d563b6442b7cd94.jpg)

![72b5d0fb762f0a560f1ff7128a7ceb6a40f10bcbbb04a0a1a915bf6f8d0ecd81.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/72b5d0fb762f0a560f1ff7128a7ceb6a40f10bcbbb04a0a1a915bf6f8d0ecd81.jpg)

![c2c07c61375b89ac806237ed1c8c0a32da0332fdaefbb9bbf1c4e4008075b204.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/images/c2c07c61375b89ac806237ed1c8c0a32da0332fdaefbb9bbf1c4e4008075b204.jpg)

### Tables

![0ffb529af12a8fa687e95c29194539aad3e174fed945d8c4fe6dffdccd12acb7.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/tables/0ffb529af12a8fa687e95c29194539aad3e174fed945d8c4fe6dffdccd12acb7.jpg)

![3ed1a9327270c15df12c330fb1da0161e804e925f71f340b8d14d23101ea4f11.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/tables/3ed1a9327270c15df12c330fb1da0161e804e925f71f340b8d14d23101ea4f11.jpg)

![41b547404c2cc66db144eda8d42531623088b4f1cb159e5ad7efe20a233f9430.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/tables/41b547404c2cc66db144eda8d42531623088b4f1cb159e5ad7efe20a233f9430.jpg)

![a6b6e08c551674b1c0bce7ef88d5ab95abeb33bd022bde3dcf5cd25912be0ca2.jpg](../icml_results/2291_Spatial%20Conformal%20Inference%20through%20Localized%20Quantile%20Regression/tables/a6b6e08c551674b1c0bce7ef88d5ab95abeb33bd022bde3dcf5cd25912be0ca2.jpg)

<a id="paper-2263"></a>

## KineFlow: Kinematic Second-Order Flow Matching for Time-Series Forecasting


### Images

![3030fc00a560708676293d3d9192eec075f414aefacd008ea9c1c230ce80c48f.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/images/3030fc00a560708676293d3d9192eec075f414aefacd008ea9c1c230ce80c48f.jpg)

![36b60d3b0eda6e5d0eb7d4dd9fc201b1555afb0d90895fcccfdd3b90c5b07475.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/images/36b60d3b0eda6e5d0eb7d4dd9fc201b1555afb0d90895fcccfdd3b90c5b07475.jpg)

![3d554ef84be4586501d456f406685f3c9bee5166f20a5fcad071fb12ad4a1b46.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/images/3d554ef84be4586501d456f406685f3c9bee5166f20a5fcad071fb12ad4a1b46.jpg)

![654b5be663c0afd3ee003d199bfb47cc611673994399f6da0f69fd4f5695343d.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/images/654b5be663c0afd3ee003d199bfb47cc611673994399f6da0f69fd4f5695343d.jpg)

![ab0b09d39b0b67449b2929843c806054522060b4b9db72db08d05f2e6f2c62aa.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/images/ab0b09d39b0b67449b2929843c806054522060b4b9db72db08d05f2e6f2c62aa.jpg)

![df0f3b5d8997ed15bc3e68c3561e70f02cb6c771c731b7ce8439bc7c6458dfca.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/images/df0f3b5d8997ed15bc3e68c3561e70f02cb6c771c731b7ce8439bc7c6458dfca.jpg)

### Tables

![29e296fcb2292b022c84f5458dcbd951bc2bdd338f95d2f9852b6adcceff6ad3.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/29e296fcb2292b022c84f5458dcbd951bc2bdd338f95d2f9852b6adcceff6ad3.jpg)

![3304ce8008981004f37d37f7250cc6817ee777dfca852ecea017acc64a06576a.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/3304ce8008981004f37d37f7250cc6817ee777dfca852ecea017acc64a06576a.jpg)

![33e7ea6a899ddd91dbf33bd8955851ba74d4dacd54fb37ba8782fa06dd4f3197.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/33e7ea6a899ddd91dbf33bd8955851ba74d4dacd54fb37ba8782fa06dd4f3197.jpg)

![46cf3ecd38b1ce36c756a22461349f668e91b716f694b796f8448cda64f4852c.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/46cf3ecd38b1ce36c756a22461349f668e91b716f694b796f8448cda64f4852c.jpg)

![4bd0571b7789ca73622ab6c479e52a482408b8376dab1659385c30675a7f8a70.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/4bd0571b7789ca73622ab6c479e52a482408b8376dab1659385c30675a7f8a70.jpg)

![5b61a66001d55e75258ffe7fece22623a7e78c26ce49628e480cbd5a72012425.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/5b61a66001d55e75258ffe7fece22623a7e78c26ce49628e480cbd5a72012425.jpg)

![5bed3894df97389d791f4edf10149659c1be13da539e04456be890032ccbde3e.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/5bed3894df97389d791f4edf10149659c1be13da539e04456be890032ccbde3e.jpg)

![60367baaaaaa97e5872f9d491db40921f19a0f1b374479230be6bef15a4f4987.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/60367baaaaaa97e5872f9d491db40921f19a0f1b374479230be6bef15a4f4987.jpg)

![86be530fd8b9e9e61b1611c76431130a80793768d4b28777e3348e8424e84db8.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/86be530fd8b9e9e61b1611c76431130a80793768d4b28777e3348e8424e84db8.jpg)

![b68fc8cf185802de0d47798cad3482acf4c8bf023a86e1c594f87dcf7ad32c57.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/b68fc8cf185802de0d47798cad3482acf4c8bf023a86e1c594f87dcf7ad32c57.jpg)

![cd1a3977e4341ff0aa175c85fbd503d30857f05f5fe677e96c46592c088c722e.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/cd1a3977e4341ff0aa175c85fbd503d30857f05f5fe677e96c46592c088c722e.jpg)

![d041a0be0b2aa92a513266b9d7e730b3ec2a8b0d0d392cc07d9f2c891aadf693.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/d041a0be0b2aa92a513266b9d7e730b3ec2a8b0d0d392cc07d9f2c891aadf693.jpg)

![d3d77c918802a5b8306b4576107461bd80369f4c162972aece32fa356e2b2a39.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/d3d77c918802a5b8306b4576107461bd80369f4c162972aece32fa356e2b2a39.jpg)

![d7ac2b0aee0622df7fe8f650996474376ef321e282c1bbd9ae315c6e0f72509e.jpg](../icml_results/2292_KineFlow_%20Kinematic%20Second-Order%20Flow%20Matching%20for%20Time-Series%20Forecasting/tables/d7ac2b0aee0622df7fe8f650996474376ef321e282c1bbd9ae315c6e0f72509e.jpg)

<a id="paper-2264"></a>

## DecoVer: A Decompose-and-Verify Neuro-Symbolic Framework for Embodied Task Planning with BC+


### Images

![0491dafec456ebb01b6140cfa49b2b75730d04e6f919514ba830b58c787be8a6.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/0491dafec456ebb01b6140cfa49b2b75730d04e6f919514ba830b58c787be8a6.jpg)

![1135c6119f4f46632d9692909984c70e018a996439a412a53f8ca9b98cf7b2e9.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/1135c6119f4f46632d9692909984c70e018a996439a412a53f8ca9b98cf7b2e9.jpg)

![193a0e9a1436a165082a7aa6d9a4507b0a44625d9b9ca6f969aebb98ddfc840e.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/193a0e9a1436a165082a7aa6d9a4507b0a44625d9b9ca6f969aebb98ddfc840e.jpg)

![20a9388bbafae2f338e5bc6186170207b6bea8f4706bf14fdf0c5bcc3d4b020f.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/20a9388bbafae2f338e5bc6186170207b6bea8f4706bf14fdf0c5bcc3d4b020f.jpg)

![21ac01c7878df00550c7cf712b4ccbd38887b940d6cbca2c59e31e0d9a9c2b10.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/21ac01c7878df00550c7cf712b4ccbd38887b940d6cbca2c59e31e0d9a9c2b10.jpg)

![24c2b60797313056eaceac4c69544d795e22a2045a7c6ef9f890e9b7c9065ebd.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/24c2b60797313056eaceac4c69544d795e22a2045a7c6ef9f890e9b7c9065ebd.jpg)

![2622f028c09d119fed222ba5a917e06e3cd8fadb2d01bc128edcfcbbc44ba9ac.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/2622f028c09d119fed222ba5a917e06e3cd8fadb2d01bc128edcfcbbc44ba9ac.jpg)

![2be54cf9ecbe12e58e550d8bcb44fc692277bf92f22fd7290779a922a3a13f96.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/2be54cf9ecbe12e58e550d8bcb44fc692277bf92f22fd7290779a922a3a13f96.jpg)

![30726c1fb386a8b2a9d5c7b8ce69c7d2f57758c16b40e476a35879303088e583.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/30726c1fb386a8b2a9d5c7b8ce69c7d2f57758c16b40e476a35879303088e583.jpg)

![608a6f258cda2b5a107f5d28e52200ec173f659f3c4d9839645ee5662524fee0.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/608a6f258cda2b5a107f5d28e52200ec173f659f3c4d9839645ee5662524fee0.jpg)

![62f3b7c7a8127474142b131b47325f5cb06119c5d57de673c0b5939861781709.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/62f3b7c7a8127474142b131b47325f5cb06119c5d57de673c0b5939861781709.jpg)

![65411b3d95e4ef1c301414dc875ba8d46850051330e88e583fb43dbca8fb482a.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/65411b3d95e4ef1c301414dc875ba8d46850051330e88e583fb43dbca8fb482a.jpg)

![6923fdc9e974c25deb8c0e15f0090c56e2d18e168270c64a3d31046f2130929e.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/6923fdc9e974c25deb8c0e15f0090c56e2d18e168270c64a3d31046f2130929e.jpg)

![8199aa3014ef6ca70f287580e0c9c32cd34bfc3fdf4025b6b8ae4fe79b57750a.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/8199aa3014ef6ca70f287580e0c9c32cd34bfc3fdf4025b6b8ae4fe79b57750a.jpg)

![90fbe60074cc45d4c7bba796eaf72e87fc386490953f0f88aaea846282be0d1c.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/90fbe60074cc45d4c7bba796eaf72e87fc386490953f0f88aaea846282be0d1c.jpg)

![aa8ce06e95d3d74249206ea1e9649d80f23cf3d79835f2b9fb5106c7c82612b9.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/aa8ce06e95d3d74249206ea1e9649d80f23cf3d79835f2b9fb5106c7c82612b9.jpg)

![aadc7446c9641c24622077967e24593190b2807fff02447253bc6e88df4442d9.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/aadc7446c9641c24622077967e24593190b2807fff02447253bc6e88df4442d9.jpg)

![d208ae0eccfd355da206520ba6f5feca70e33a40bb8821f6b7def44d8e1832ff.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/d208ae0eccfd355da206520ba6f5feca70e33a40bb8821f6b7def44d8e1832ff.jpg)

![d3453447e84c62487f427e5b527ca13f9203f15f2593b92101ddffae05a90151.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/d3453447e84c62487f427e5b527ca13f9203f15f2593b92101ddffae05a90151.jpg)

![dfb2d12b536dbaf4ed9d94274e9ec2ddeb337bb3c6dc4a9b04074bd490170a4e.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/dfb2d12b536dbaf4ed9d94274e9ec2ddeb337bb3c6dc4a9b04074bd490170a4e.jpg)

![eb8466b64af67a62c92c96c4b19fef89fe218d6eb8de118c09c238bb158f9312.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/eb8466b64af67a62c92c96c4b19fef89fe218d6eb8de118c09c238bb158f9312.jpg)

![f6651eb208e82d92bbf48681fb0ff7a7d4658910ae4fe03aa68a70036904effe.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/images/f6651eb208e82d92bbf48681fb0ff7a7d4658910ae4fe03aa68a70036904effe.jpg)

### Tables

![0416220a34c0a7f0eec433861326bd45e8bf7aed060f1a03a77527388ac6884d.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/0416220a34c0a7f0eec433861326bd45e8bf7aed060f1a03a77527388ac6884d.jpg)

![088d5f2b949d0b136051e64d2a95388a3ebece54b37aebd01bbfcc59b588f981.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/088d5f2b949d0b136051e64d2a95388a3ebece54b37aebd01bbfcc59b588f981.jpg)

![1fa1d2cdeb8d65b1a9d85dc97c400a42d857be0aed3d305ab3b0fde6adec74ac.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/1fa1d2cdeb8d65b1a9d85dc97c400a42d857be0aed3d305ab3b0fde6adec74ac.jpg)

![290043c9a289a91f90b1406ae0baa87e75fa304d6c7130224e3593a3b7add45c.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/290043c9a289a91f90b1406ae0baa87e75fa304d6c7130224e3593a3b7add45c.jpg)

![3bf5a5f771b64b3ef5bddeb381ffa44edf5ffe0a00e7501ce0976a14f84b2a3a.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/3bf5a5f771b64b3ef5bddeb381ffa44edf5ffe0a00e7501ce0976a14f84b2a3a.jpg)

![45af97fbaffa87915f08d56338c11149cf4f10c99edf891dc35d5a2db82e4810.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/45af97fbaffa87915f08d56338c11149cf4f10c99edf891dc35d5a2db82e4810.jpg)

![4be02283ebf1d79c80e73bac76dd5c15454d9f66219db900fd1173a5d40ff146.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/4be02283ebf1d79c80e73bac76dd5c15454d9f66219db900fd1173a5d40ff146.jpg)

![5a824d3f6e1facf2995714f17ced6e3dd7c2b69c8b6eddc3be98f342fe83efc1.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/5a824d3f6e1facf2995714f17ced6e3dd7c2b69c8b6eddc3be98f342fe83efc1.jpg)

![5f75c7f7530efd74c06b40df5b0b91961853dc06c14e9293cae28ea23c48ffc2.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/5f75c7f7530efd74c06b40df5b0b91961853dc06c14e9293cae28ea23c48ffc2.jpg)

![64330ee7de1dd50f73b4b1c43a60c07398f6147c3442fa212a64dfdb3abd68d2.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/64330ee7de1dd50f73b4b1c43a60c07398f6147c3442fa212a64dfdb3abd68d2.jpg)

![686d3f574381a04c3210b32594a89a365d12648cd921102e133838b6283c1ad0.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/686d3f574381a04c3210b32594a89a365d12648cd921102e133838b6283c1ad0.jpg)

![6b4287c1a3718e08bc833242b2bd44194f3bbb8a8759c4fa61462dd9c53c2f1f.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/6b4287c1a3718e08bc833242b2bd44194f3bbb8a8759c4fa61462dd9c53c2f1f.jpg)

![6c955930dee7aaf3446881dbc6a28cc071a38c7cdf2419dbe6e76bd32b9c2a31.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/6c955930dee7aaf3446881dbc6a28cc071a38c7cdf2419dbe6e76bd32b9c2a31.jpg)

![6d753775534f1a40d671296649edc6f2356ea36317828e4bf9b351532b8ffbfa.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/6d753775534f1a40d671296649edc6f2356ea36317828e4bf9b351532b8ffbfa.jpg)

![948d56f81a4a3bd5a964349f0bf14aad318b0524eb648b5316dd9a0c6995a0e2.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/948d56f81a4a3bd5a964349f0bf14aad318b0524eb648b5316dd9a0c6995a0e2.jpg)

![98d91f616455199059e8cc15ea1d5fecbcfe0ae1172fddc495276ad1df7f80e7.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/98d91f616455199059e8cc15ea1d5fecbcfe0ae1172fddc495276ad1df7f80e7.jpg)

![adc12e326b300548949aa550021900c2c3c9ba600cc09c498133305cb79022bb.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/adc12e326b300548949aa550021900c2c3c9ba600cc09c498133305cb79022bb.jpg)

![cab0351e9b9ac9030b082d67c8d4b8fec52202d1cc5779d38e07f18c97dfdf35.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/cab0351e9b9ac9030b082d67c8d4b8fec52202d1cc5779d38e07f18c97dfdf35.jpg)

![d68f5d5bb50b44b6ff44b92f618ea96572b21f56154626414321cbe886314e47.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/d68f5d5bb50b44b6ff44b92f618ea96572b21f56154626414321cbe886314e47.jpg)

![ddbb026569f5d0362e28c87f94303aeba4b178859ad401e61007490575dded60.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/ddbb026569f5d0362e28c87f94303aeba4b178859ad401e61007490575dded60.jpg)

![f607dece707c928e062ac08891ddd1f669ce63f435078994cbe41626f177cc48.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/f607dece707c928e062ac08891ddd1f669ce63f435078994cbe41626f177cc48.jpg)

![f88c34a710601280db52e67abdd2cc410d0de97f39b1ae6af8fe441b6fcd5dde.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/f88c34a710601280db52e67abdd2cc410d0de97f39b1ae6af8fe441b6fcd5dde.jpg)

![ff4febb4e65ae56c395f07a77c85ba7a7637af7641b9f62d84a1b628f793babd.jpg](../icml_results/2293_DecoVer_%20A%20Decompose-and-Verify%20Neuro-Symbolic%20Framework%20for%20Embodied%20Task%20Planning%20with%20BC%2B/tables/ff4febb4e65ae56c395f07a77c85ba7a7637af7641b9f62d84a1b628f793babd.jpg)

<a id="paper-2265"></a>

## BRIDGE: Triangular Fixed-Point Refinement for Long-Horizon Persona Consistency


### Images

![35abfe3b5ba60f218fa598ccb8c18cf7c897d6f6a49cfe5ea3dceec3c1707a71.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/35abfe3b5ba60f218fa598ccb8c18cf7c897d6f6a49cfe5ea3dceec3c1707a71.jpg)

![49740a9290955b8be48a610c1aa31defe23bb764e57146cbc0dc6f86fc4dfacb.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/49740a9290955b8be48a610c1aa31defe23bb764e57146cbc0dc6f86fc4dfacb.jpg)

![5143177617ec43a8d5b588385ca0e25107656da5c5947b3efc2f629fc0908794.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/5143177617ec43a8d5b588385ca0e25107656da5c5947b3efc2f629fc0908794.jpg)

![551369ecaf59590187c745fc05680d544fc81e0514cd3bc68bdea3050b5e0010.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/551369ecaf59590187c745fc05680d544fc81e0514cd3bc68bdea3050b5e0010.jpg)

![64fb8a4d5b2556227a429926dc676572d5cd7c9645ad99c72fdc9af5a13f8020.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/64fb8a4d5b2556227a429926dc676572d5cd7c9645ad99c72fdc9af5a13f8020.jpg)

![6854936a3ad037bba335d31099ebe8b822273e7251d7654dff15c732776fbb52.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/6854936a3ad037bba335d31099ebe8b822273e7251d7654dff15c732776fbb52.jpg)

![815a39e554cdee9d0438b6017e4efd5b659e4e4042bcf879a1272c96c992391f.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/815a39e554cdee9d0438b6017e4efd5b659e4e4042bcf879a1272c96c992391f.jpg)

![9f580a8fc3664c2c1bf5048a75cc81474a0c10f037451126b34ca77611896ad2.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/9f580a8fc3664c2c1bf5048a75cc81474a0c10f037451126b34ca77611896ad2.jpg)

![b6def4bfedca1c67ebff93b38e6baa783b87fbe3c550dbd68d2165f37192c362.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/images/b6def4bfedca1c67ebff93b38e6baa783b87fbe3c550dbd68d2165f37192c362.jpg)

### Tables

![027a5361669201278a76a4fb78de428841c31b848148cbd3dfec2418491dc91d.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/027a5361669201278a76a4fb78de428841c31b848148cbd3dfec2418491dc91d.jpg)

![0a03b029e05d3d9b5eb29d7791d4eb9e06f3a789291939824e0fa9f8b1796eba.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/0a03b029e05d3d9b5eb29d7791d4eb9e06f3a789291939824e0fa9f8b1796eba.jpg)

![15f96b225608acbba6ed52401a9581c23dd07f21a2544a15789397f48604f7e7.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/15f96b225608acbba6ed52401a9581c23dd07f21a2544a15789397f48604f7e7.jpg)

![231301eb90599d93eb53f24604a56a96dda9e52097bbf9149da9a7db7e3a43c8.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/231301eb90599d93eb53f24604a56a96dda9e52097bbf9149da9a7db7e3a43c8.jpg)

![2939a5d04b8d420d7a5c843e65f8d7d24d48080a2c799957505d6531fe099000.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/2939a5d04b8d420d7a5c843e65f8d7d24d48080a2c799957505d6531fe099000.jpg)

![2fc5ef13171ca1ca80bfdfb97fd51e46aaba98830633cb3c6c32588e3b28943b.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/2fc5ef13171ca1ca80bfdfb97fd51e46aaba98830633cb3c6c32588e3b28943b.jpg)

![4350fb4707865d3fb12329b37fb903e127d03454d71d3465a2048dc4fb4327c0.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/4350fb4707865d3fb12329b37fb903e127d03454d71d3465a2048dc4fb4327c0.jpg)

![4a697467e7edfffe496d3f5d8e2d661afb6fab3d8242c9166276aa6b3ed3c346.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/4a697467e7edfffe496d3f5d8e2d661afb6fab3d8242c9166276aa6b3ed3c346.jpg)

![4f7ca40256f500f88743c0bf1e0977e7035db5820d8fa62461bff960cf54c686.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/4f7ca40256f500f88743c0bf1e0977e7035db5820d8fa62461bff960cf54c686.jpg)

![56a2acbe85f1e1ecbffcd9c9ebf18dd10db43448f1086301ff1d071f773d4237.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/56a2acbe85f1e1ecbffcd9c9ebf18dd10db43448f1086301ff1d071f773d4237.jpg)

![5f3fc9dff13c33e3dce0ba7f5cc17485a5dc26666006be4ace4da45af2ced412.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/5f3fc9dff13c33e3dce0ba7f5cc17485a5dc26666006be4ace4da45af2ced412.jpg)

![6780d9141519ad03e653276e20a803f2eb1219359e0aa44a8eab25719dcb1110.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/6780d9141519ad03e653276e20a803f2eb1219359e0aa44a8eab25719dcb1110.jpg)

![a475111b37fb11f86ce76482b42104ba6f1614b29720bce79ffe564bbb0363df.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/a475111b37fb11f86ce76482b42104ba6f1614b29720bce79ffe564bbb0363df.jpg)

![b5e6a5ae32a7157405229506da85a4f957869a36cdbf88f66de298862c9c11ce.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/b5e6a5ae32a7157405229506da85a4f957869a36cdbf88f66de298862c9c11ce.jpg)

![c286f433d0d19bf16c6a246e39065929941a6ee9b7340c333699af0f217848ab.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/c286f433d0d19bf16c6a246e39065929941a6ee9b7340c333699af0f217848ab.jpg)

![c3b5fa82ba6a88dabf16a38eb0cef7af478cf16aaabf0aeba3cc9ac1b9b8fe55.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/c3b5fa82ba6a88dabf16a38eb0cef7af478cf16aaabf0aeba3cc9ac1b9b8fe55.jpg)

![edc542d6cf4d1b683109c624863cd1d7e96e368506e4a43ec37467c20fb08e0f.jpg](../icml_results/2294_BRIDGE_%20Triangular%20Fixed-Point%20Refinement%20for%20Long-Horizon%20Persona%20Consistency/tables/edc542d6cf4d1b683109c624863cd1d7e96e368506e4a43ec37467c20fb08e0f.jpg)

<a id="paper-2266"></a>

## Convergence Analysis of the Lion Optimizer in Centralized and Distributed Settings


### Images

![049be40a66546f9964feb458b4f9cfdd1bb09b4c547b3a1e326986ce9447a07a.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/images/049be40a66546f9964feb458b4f9cfdd1bb09b4c547b3a1e326986ce9447a07a.jpg)

![15f3029c85b6efc2f72d253f3e77753429d83f2fde65a670e83bf0504888107a.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/images/15f3029c85b6efc2f72d253f3e77753429d83f2fde65a670e83bf0504888107a.jpg)

![355b889c4909954bf29cd37dd74e35046d4474bf25a66cd041a724564efe6348.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/images/355b889c4909954bf29cd37dd74e35046d4474bf25a66cd041a724564efe6348.jpg)

![7397148d7bfadf0fdf5a7500f8161c032acf415ce8d57789339fa66c0a8334ae.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/images/7397148d7bfadf0fdf5a7500f8161c032acf415ce8d57789339fa66c0a8334ae.jpg)

### Tables

![05533a2f33e61ac1deaf11e041c96e5dbc06b4bb67e7792832ae81a4e94529a3.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/tables/05533a2f33e61ac1deaf11e041c96e5dbc06b4bb67e7792832ae81a4e94529a3.jpg)

![3177f3c303e297e9736c7f18c6db7ad6167fdf2ffecdb3fdc43eb669afde8a77.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/tables/3177f3c303e297e9736c7f18c6db7ad6167fdf2ffecdb3fdc43eb669afde8a77.jpg)

![636dbcec1179c975ccf485af7ed00246c841aab835ae068948ca9767cbfc20a9.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/tables/636dbcec1179c975ccf485af7ed00246c841aab835ae068948ca9767cbfc20a9.jpg)

![a81d5254a68b0861bae866f6569a32c969ed81c21d1ed4e36496b530513bce84.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/tables/a81d5254a68b0861bae866f6569a32c969ed81c21d1ed4e36496b530513bce84.jpg)

![c781a06c643924f45d373ebab3d9753a148d8a2680051487272cae1b77008c96.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/tables/c781a06c643924f45d373ebab3d9753a148d8a2680051487272cae1b77008c96.jpg)

![ef358025d23ac6971e1932e0933b40da2de0dc66f75594b72b95babea3bc56e0.jpg](../icml_results/2295_Convergence%20Analysis%20of%20the%20Lion%20Optimizer%20in%20Centralized%20and%20Distributed%20Settings/tables/ef358025d23ac6971e1932e0933b40da2de0dc66f75594b72b95babea3bc56e0.jpg)

<a id="paper-2267"></a>

## SCoA: Revisiting Domain Generalized Object Detection with Style-Conditioned Adaptation


### Images

![00ea484043a0e3cfd780ddf3886f78c7a617aa53ee118b6022868c75d2ac8023.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/00ea484043a0e3cfd780ddf3886f78c7a617aa53ee118b6022868c75d2ac8023.jpg)

![012fbbbb058dec4d89ad06412081cfd4932e19bf6ad890c22785b055b20b9faa.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/012fbbbb058dec4d89ad06412081cfd4932e19bf6ad890c22785b055b20b9faa.jpg)

![04c2eb7c1f3a78a5b6ad8ece1ad2e557480280f1dc8c78f17234d515e011d010.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/04c2eb7c1f3a78a5b6ad8ece1ad2e557480280f1dc8c78f17234d515e011d010.jpg)

![0633a2e868a8622d3f229030758cc82fd8a9d1fe663ba9f1ab9d93fe0a99b161.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/0633a2e868a8622d3f229030758cc82fd8a9d1fe663ba9f1ab9d93fe0a99b161.jpg)

![081c96aa0a25d4f210c37c3db7dc89c2e589b8dd28391b124b1da381fc6b2d5c.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/081c96aa0a25d4f210c37c3db7dc89c2e589b8dd28391b124b1da381fc6b2d5c.jpg)

![3467305a79a27932c6484d0ea2690dae972adc426b6f08a1c61cc5f5c65a6cb3.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/3467305a79a27932c6484d0ea2690dae972adc426b6f08a1c61cc5f5c65a6cb3.jpg)

![4b3d165fe9c4c8ad0e0375f5c6bbb7c139435deb2480059089119c6cdbde008c.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/4b3d165fe9c4c8ad0e0375f5c6bbb7c139435deb2480059089119c6cdbde008c.jpg)

![5100cc2704ca14dc828fb434aab1a92c5d385b9abe660a0c9c4de35dc1197a03.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/5100cc2704ca14dc828fb434aab1a92c5d385b9abe660a0c9c4de35dc1197a03.jpg)

![6193903dcd8840602ab11da740d25bbfa79abe9af83baccf0b2be84fceea9246.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/6193903dcd8840602ab11da740d25bbfa79abe9af83baccf0b2be84fceea9246.jpg)

![6c96148b3e05edf4b4ef4c3493d77fd59ce1511a3d465c154903e2b9dee452ae.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/6c96148b3e05edf4b4ef4c3493d77fd59ce1511a3d465c154903e2b9dee452ae.jpg)

![823e6140d0455bffdce2f7ca96aa51dd8ff1dfe808271d46fae795439eac9aba.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/823e6140d0455bffdce2f7ca96aa51dd8ff1dfe808271d46fae795439eac9aba.jpg)

![84a52a867b8386ced7cdb3abb2f0b18ae97bc10e145753cdd4a6219b632aae12.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/84a52a867b8386ced7cdb3abb2f0b18ae97bc10e145753cdd4a6219b632aae12.jpg)

![9027c6241b0c0d6f70cc75e8ba9faff338a24b6b00c66fecc4c6e52f134dc0e8.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/9027c6241b0c0d6f70cc75e8ba9faff338a24b6b00c66fecc4c6e52f134dc0e8.jpg)

![9ada0bd29fe537dfcfba8d4a4c3c1e01dd5acfd4720bc3e97a13c5e9dffabb24.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/9ada0bd29fe537dfcfba8d4a4c3c1e01dd5acfd4720bc3e97a13c5e9dffabb24.jpg)

![a51a0fad8d7adf6b07b71ff37edbd6dda792cb33009b53cdc2daa839b1111996.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/a51a0fad8d7adf6b07b71ff37edbd6dda792cb33009b53cdc2daa839b1111996.jpg)

![c8657b08c2aa3ae91953d5d1bfd2e8a52a3e42b51c4ace4c71ce128a6cb36c40.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/c8657b08c2aa3ae91953d5d1bfd2e8a52a3e42b51c4ace4c71ce128a6cb36c40.jpg)

![e798b080fab5ce57beab715e833ccb3b30f7352cb229e94956ede7646b1fe17f.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/e798b080fab5ce57beab715e833ccb3b30f7352cb229e94956ede7646b1fe17f.jpg)

![e94af7238b1c9b321166405319c9c5f3b96dff7d20cf484c02319c804190a596.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/images/e94af7238b1c9b321166405319c9c5f3b96dff7d20cf484c02319c804190a596.jpg)

### Tables

![0075b4cd7a448010f14c3b0c3fbb317fbdeb8943982e67829e6d810b03e70877.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/0075b4cd7a448010f14c3b0c3fbb317fbdeb8943982e67829e6d810b03e70877.jpg)

![074347441284df8f31c89d4cea27534981adb2a09051dfcf308e3c67a3587ee0.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/074347441284df8f31c89d4cea27534981adb2a09051dfcf308e3c67a3587ee0.jpg)

![0892ef814f3b639ef2eb8e3c3417f483eabd7a2161f48eee59401a0c7a5c596e.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/0892ef814f3b639ef2eb8e3c3417f483eabd7a2161f48eee59401a0c7a5c596e.jpg)

![3a698a6c8c2daf29fc324f6d16187dfacf1bb270f30c38935544759f2b517485.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/3a698a6c8c2daf29fc324f6d16187dfacf1bb270f30c38935544759f2b517485.jpg)

![404069dc04fc55efa87dc8e1ae018a46bb5f6e0341f2af954169fda9a18c50b6.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/404069dc04fc55efa87dc8e1ae018a46bb5f6e0341f2af954169fda9a18c50b6.jpg)

![4221f20a46d7e5092ecb3d14f4731e2bcc757bed569299815ecc7c482656163e.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/4221f20a46d7e5092ecb3d14f4731e2bcc757bed569299815ecc7c482656163e.jpg)

![4775018185efeb1285dc0a2af4d7e911633578eb9b377b477cacf8c685436546.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/4775018185efeb1285dc0a2af4d7e911633578eb9b377b477cacf8c685436546.jpg)

![840d22cc2d9fa40e29acb764d21dd0f4a9d47acec1c47ff6869e6534959807af.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/840d22cc2d9fa40e29acb764d21dd0f4a9d47acec1c47ff6869e6534959807af.jpg)

![8ce5d8cea3c0c4ac1c3ec5cd4062ed89181832540c654db3bf9199d029b9a2d7.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/8ce5d8cea3c0c4ac1c3ec5cd4062ed89181832540c654db3bf9199d029b9a2d7.jpg)

![9fe2d684895f928a5d34f2d2317ebef8df0d8542b26cf9771de94066693a467a.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/9fe2d684895f928a5d34f2d2317ebef8df0d8542b26cf9771de94066693a467a.jpg)

![e530743d25f431f41c6ba70f60929fc474bed2bb63e4d77da6c8fd892e410160.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/e530743d25f431f41c6ba70f60929fc474bed2bb63e4d77da6c8fd892e410160.jpg)

![eaa93f92af74d6b92fe9d1e8ba24f554a39475b10d6acc1ae0f4c2d94e34fab9.jpg](../icml_results/2296_SCoA_%20Revisiting%20Domain%20Generalized%20Object%20Detection%20with%20Style-Conditioned%20Adaptation/tables/eaa93f92af74d6b92fe9d1e8ba24f554a39475b10d6acc1ae0f4c2d94e34fab9.jpg)

<a id="paper-2268"></a>

## EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization


### Images

![013c75159ade6b10f7284861a9ca016e41db976337a58e56ea4a991d00546432.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/013c75159ade6b10f7284861a9ca016e41db976337a58e56ea4a991d00546432.jpg)

![0983d9aad77ebe4d29c7351a938165b030a47842cf3a65a5d6a1a859917bd88c.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/0983d9aad77ebe4d29c7351a938165b030a47842cf3a65a5d6a1a859917bd88c.jpg)

![50343b691a59a9baeacd1b5b4eeb9af7ce2ca7e935cf4d1695b65aa893c7c2a7.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/50343b691a59a9baeacd1b5b4eeb9af7ce2ca7e935cf4d1695b65aa893c7c2a7.jpg)

![750847f94b4f8b327dff81e16fe4e82c4ad05554b13800ddf8d3eeb07f729cd2.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/750847f94b4f8b327dff81e16fe4e82c4ad05554b13800ddf8d3eeb07f729cd2.jpg)

![9a59a6c273ae17ba772a582374e84198208833a18436f7ce7a2a8c3610f0b3e5.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/9a59a6c273ae17ba772a582374e84198208833a18436f7ce7a2a8c3610f0b3e5.jpg)

![b30da46dc410f81eb57595fdfa908d95278ef1fc175ae7d51a0fc8c75f585f0c.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/b30da46dc410f81eb57595fdfa908d95278ef1fc175ae7d51a0fc8c75f585f0c.jpg)

![e92473c5fe1c9d58488569059aee22c7e72782e9899796c3dd5ba2b7fb9af744.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/e92473c5fe1c9d58488569059aee22c7e72782e9899796c3dd5ba2b7fb9af744.jpg)

![f6796b73f3091b3818687d78e8ccafd48a3046b63444f433fa54e708bfea0977.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/images/f6796b73f3091b3818687d78e8ccafd48a3046b63444f433fa54e708bfea0977.jpg)

### Tables

![2eda8ed20a538c288ff7cc2580b2c90d301aa894de32f7de3ffa6e95c69e50ac.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/2eda8ed20a538c288ff7cc2580b2c90d301aa894de32f7de3ffa6e95c69e50ac.jpg)

![3267727cbb1ba9059179acbd4d32e10ca8790e2d77e6b76b119a91716a3565f0.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/3267727cbb1ba9059179acbd4d32e10ca8790e2d77e6b76b119a91716a3565f0.jpg)

![381063a48baf7d2e9f0548a50d228a92d8ddffb63c23832a644c2e135bc0fa55.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/381063a48baf7d2e9f0548a50d228a92d8ddffb63c23832a644c2e135bc0fa55.jpg)

![417df82fe66260737fcf32d4aea9ee85718e339bc906cab8752700740320156f.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/417df82fe66260737fcf32d4aea9ee85718e339bc906cab8752700740320156f.jpg)

![49af8b4618d639febf939bd3eedf4de81c5dc603876301468f253a4eee63a808.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/49af8b4618d639febf939bd3eedf4de81c5dc603876301468f253a4eee63a808.jpg)

![50dd0e73410067abbbd57aad01c00ae5a746c0e6412dd63b1322bee68020a1fe.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/50dd0e73410067abbbd57aad01c00ae5a746c0e6412dd63b1322bee68020a1fe.jpg)

![57d0a58d7a75d14218f613ef3482a1f73c357d988822ece1de5897cbeeca22e4.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/57d0a58d7a75d14218f613ef3482a1f73c357d988822ece1de5897cbeeca22e4.jpg)

![589f33e3ccffcb0204fa63da9ab3b5393a555fcecb71bbed2bd454199cb7f4e6.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/589f33e3ccffcb0204fa63da9ab3b5393a555fcecb71bbed2bd454199cb7f4e6.jpg)

![60ae51ea5032e0472b6b0f15cd8beb7d505ce838ea550b274775964eb55b17b5.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/60ae51ea5032e0472b6b0f15cd8beb7d505ce838ea550b274775964eb55b17b5.jpg)

![8f8dd2513333298beb00edc981bcfc74ec22bbac48fd75e8448b1a631e42fdd8.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/8f8dd2513333298beb00edc981bcfc74ec22bbac48fd75e8448b1a631e42fdd8.jpg)

![961e3ae7d23f639e655103075fd459bc2fcc34572afc3c7409e7295f6fee6793.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/961e3ae7d23f639e655103075fd459bc2fcc34572afc3c7409e7295f6fee6793.jpg)

![b2b890995c064ff256ec9904811448aeef37b71908bd56d41ceafb6bf9ad6c3c.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/b2b890995c064ff256ec9904811448aeef37b71908bd56d41ceafb6bf9ad6c3c.jpg)

![d67668729cdab7995b43c5ca978ce7e82f137d705bdea9afd3f33f5334d6ce59.jpg](../icml_results/2297_EvoGM_%20Learning%20to%20Merge%20LLMs%20via%20Evolutionary%20Generative%20Optimization/tables/d67668729cdab7995b43c5ca978ce7e82f137d705bdea9afd3f33f5334d6ce59.jpg)

<a id="paper-2269"></a>

## Budget-Efficient Attacks and Robustness Training for Cooperative MARL


### Images

![0d98972291cb024f1a123e707ea5a9313a8dc5ebd648602fe78427df111f9249.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/images/0d98972291cb024f1a123e707ea5a9313a8dc5ebd648602fe78427df111f9249.jpg)

![1e3f738c12bcbe926a2c846d0668492fa5027c4a46c14385aa5659e576e0f19b.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/images/1e3f738c12bcbe926a2c846d0668492fa5027c4a46c14385aa5659e576e0f19b.jpg)

![794c249d03a7483db8a2af989807e70630e62ba98f5ab5e3da5a53e1a4a7a6ca.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/images/794c249d03a7483db8a2af989807e70630e62ba98f5ab5e3da5a53e1a4a7a6ca.jpg)

![87155c6522317d75b0e6430c1b092e77875f079435a3d8528249887d5212684c.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/images/87155c6522317d75b0e6430c1b092e77875f079435a3d8528249887d5212684c.jpg)

![a98520020d99aa8c18c3178202543c52d9e460b1c433fa41ded05c1419b1d60d.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/images/a98520020d99aa8c18c3178202543c52d9e460b1c433fa41ded05c1419b1d60d.jpg)

### Tables

![086332491e7e533a3d280c966eaabb9cdcc133968a798d9b5e2e080aedb06f21.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/086332491e7e533a3d280c966eaabb9cdcc133968a798d9b5e2e080aedb06f21.jpg)

![0b13edc16b391addc527fd49011f39230dfa4dc6abeb157e321895d3fa513f8b.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/0b13edc16b391addc527fd49011f39230dfa4dc6abeb157e321895d3fa513f8b.jpg)

![0d2826c40213e112ed79f9a76609d5be5694e4159d25ec273f199e54147f6265.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/0d2826c40213e112ed79f9a76609d5be5694e4159d25ec273f199e54147f6265.jpg)

![3a5a6cc7aed2763acbf1efe8df7e7fe38f0bb64fa363516215534dbb4768ece1.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/3a5a6cc7aed2763acbf1efe8df7e7fe38f0bb64fa363516215534dbb4768ece1.jpg)

![4f5c3cda3d810c43162b76d7e49ad4823927a60ab73298cbf8d087a49a633743.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/4f5c3cda3d810c43162b76d7e49ad4823927a60ab73298cbf8d087a49a633743.jpg)

![5dfbae3d3623e7a40adf407857d4e652687a7432ef0b4c0fc755b7fbb7429a9e.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/5dfbae3d3623e7a40adf407857d4e652687a7432ef0b4c0fc755b7fbb7429a9e.jpg)

![78785484e073755221565b4fd8bfa5fada1b2820dce209b7249e18a9c57c8ab7.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/78785484e073755221565b4fd8bfa5fada1b2820dce209b7249e18a9c57c8ab7.jpg)

![80ee7408bdc4daafedee5e03c3d5f4d990d8107d04478621f69ec9e6e7bc8e14.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/80ee7408bdc4daafedee5e03c3d5f4d990d8107d04478621f69ec9e6e7bc8e14.jpg)

![b079b1f8dc39bbe63a305fd8ea1a0a3b22d4b5c32061c5128b5901b9a778a328.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/b079b1f8dc39bbe63a305fd8ea1a0a3b22d4b5c32061c5128b5901b9a778a328.jpg)

![b4b21264e279a63bf73bccab78290c716258b13d29df5f64c608a94e33453b79.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/b4b21264e279a63bf73bccab78290c716258b13d29df5f64c608a94e33453b79.jpg)

![d267bbffccb0f56b6f890f54956ca9f2667fd299eb6c2a459289e5b97a82d350.jpg](../icml_results/2298_Budget-Efficient%20Attacks%20and%20Robustness%20Training%20for%20Cooperative%20MARL/tables/d267bbffccb0f56b6f890f54956ca9f2667fd299eb6c2a459289e5b97a82d350.jpg)

<a id="paper-2270"></a>

## StructMamPose: From Sequential Perception to Structural Reasoning for 3D Human Pose Estimation


### Images

![29800ab42d5b339b5f957237b554f9f8c1fe7f4a9b25be729b281c8e432465a3.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/images/29800ab42d5b339b5f957237b554f9f8c1fe7f4a9b25be729b281c8e432465a3.jpg)

![9b6b7ca1a23e4b965448e99418f9ff579e163de1dc28895089e752c765f82e98.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/images/9b6b7ca1a23e4b965448e99418f9ff579e163de1dc28895089e752c765f82e98.jpg)

![a17d014b3093be4971b18b5e5ce9d57bbb492eb46153f965342226cdaa31ad3d.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/images/a17d014b3093be4971b18b5e5ce9d57bbb492eb46153f965342226cdaa31ad3d.jpg)

![e1cc0d472722aa4eeac9cc50930cf56c21cdd62dcf2bea342bc8c39f1fb104da.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/images/e1cc0d472722aa4eeac9cc50930cf56c21cdd62dcf2bea342bc8c39f1fb104da.jpg)

![f86d2bb7e4b20677bb6cdfd63ee0e8f55970c8ecfe0b245bad01a7b51c364569.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/images/f86d2bb7e4b20677bb6cdfd63ee0e8f55970c8ecfe0b245bad01a7b51c364569.jpg)

![fbee37fddc7c5424d5da3a2501e959915fc874f3876010750eca5373d4581a6e.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/images/fbee37fddc7c5424d5da3a2501e959915fc874f3876010750eca5373d4581a6e.jpg)

### Tables

![121f2c3730b8618b49a8a8e2a1e5a397e1e68a0476b49d198d3ae3db4ed93fc4.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/121f2c3730b8618b49a8a8e2a1e5a397e1e68a0476b49d198d3ae3db4ed93fc4.jpg)

![18f4ad93670c5e6947af11dcf0065c76c9ab841fadf9d9641dfd05b7c4f35777.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/18f4ad93670c5e6947af11dcf0065c76c9ab841fadf9d9641dfd05b7c4f35777.jpg)

![329928920c15881c32176a695453db3cc64896505e1d38a86d30c6912d8cf1ed.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/329928920c15881c32176a695453db3cc64896505e1d38a86d30c6912d8cf1ed.jpg)

![4b6e3a4eac151d46d3115557120764ec592c4cbc4e4e06340d9a561b008d07fb.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/4b6e3a4eac151d46d3115557120764ec592c4cbc4e4e06340d9a561b008d07fb.jpg)

![50197a4d7359c317c0f2012b359989487e189e416531faf34f23fbe9d24c5d1d.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/50197a4d7359c317c0f2012b359989487e189e416531faf34f23fbe9d24c5d1d.jpg)

![9d39ead346d50f528fe35abcdc10a042c99fd665adc4d8c73f427aa70ebf1d78.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/9d39ead346d50f528fe35abcdc10a042c99fd665adc4d8c73f427aa70ebf1d78.jpg)

![b970e3dd5d0fae06831df9b3191d545054f5656fa5e90b2d497605d3f54d235e.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/b970e3dd5d0fae06831df9b3191d545054f5656fa5e90b2d497605d3f54d235e.jpg)

![d5c0a02d4dd6c7b9c75e802f24eb48d6ca0c09f3945bf73042f7acac6aa89809.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/d5c0a02d4dd6c7b9c75e802f24eb48d6ca0c09f3945bf73042f7acac6aa89809.jpg)

![ef1b362423aa9722c87f47b5a74a107721076b98fa7084431c2848d21d6dd8f0.jpg](../icml_results/2299_StructMamPose_%20From%20Sequential%20Perception%20to%20Structural%20Reasoning%20for%203D%20Human%20Pose%20Estimation/tables/ef1b362423aa9722c87f47b5a74a107721076b98fa7084431c2848d21d6dd8f0.jpg)

<a id="paper-2271"></a>

## Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction


### Images

![1c32d5fced3ff1a9f24816142c636911f2d3fe179597a3159d13d71f303c0290.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/1c32d5fced3ff1a9f24816142c636911f2d3fe179597a3159d13d71f303c0290.jpg)

![27babab35e4ef174b7f97723e7759213f59d943b229f7e0b297ba8dd3cba62be.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/27babab35e4ef174b7f97723e7759213f59d943b229f7e0b297ba8dd3cba62be.jpg)

![288c8a474a00c9b46fb8985ca4602874ed8fa30bf07e35be1e0ff099a4d2b97f.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/288c8a474a00c9b46fb8985ca4602874ed8fa30bf07e35be1e0ff099a4d2b97f.jpg)

![29d33eb2d374afa1a0cfccc834bf9de54226414bd727fd4f34db5cdbbb8c6dd2.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/29d33eb2d374afa1a0cfccc834bf9de54226414bd727fd4f34db5cdbbb8c6dd2.jpg)

![38648e8a31aa20600e8c5c12ba3c05278c49212a0d11d54eaeedeb7b6c9afdcc.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/38648e8a31aa20600e8c5c12ba3c05278c49212a0d11d54eaeedeb7b6c9afdcc.jpg)

![4bae485089ddd882a580b797fe8373a23de1cff62a4d664ffbb997ac53150138.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/4bae485089ddd882a580b797fe8373a23de1cff62a4d664ffbb997ac53150138.jpg)

![542143de98686d4c5d956561b8bf6a6b7683c012afed8af8a1164ab7f8d99954.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/542143de98686d4c5d956561b8bf6a6b7683c012afed8af8a1164ab7f8d99954.jpg)

![667338a71ce456f68a34244203eab784e00cd3b8a7a359a0d74d499e0e261d10.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/667338a71ce456f68a34244203eab784e00cd3b8a7a359a0d74d499e0e261d10.jpg)

![6bfe8fb2d4c795d6f47d700b89901ca9e69f292f9c98a2b94200b1ec54bd17b4.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/6bfe8fb2d4c795d6f47d700b89901ca9e69f292f9c98a2b94200b1ec54bd17b4.jpg)

![99898a47273e80e5f6ce61712cb024080dbbd05021f667fccf72f7af656140bc.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/99898a47273e80e5f6ce61712cb024080dbbd05021f667fccf72f7af656140bc.jpg)

![ab072831d8521d7977d43bb278c70170bd5a7b4dcdbf6695b460b43c062a181a.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/ab072831d8521d7977d43bb278c70170bd5a7b4dcdbf6695b460b43c062a181a.jpg)

![b4731798a7386a60e82a11757390333b0ebb31862b475c9ffd261eb1513c46c1.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/b4731798a7386a60e82a11757390333b0ebb31862b475c9ffd261eb1513c46c1.jpg)

![d005112f2ab4bfed44ea4200eb0e204f178f217bbd7e3e2d532bfabaa5ce01c5.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/d005112f2ab4bfed44ea4200eb0e204f178f217bbd7e3e2d532bfabaa5ce01c5.jpg)

![ff4d5723c69996a1e7b8adc1818711534ee62f1b3c15b5844a29b96ed91c126f.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/images/ff4d5723c69996a1e7b8adc1818711534ee62f1b3c15b5844a29b96ed91c126f.jpg)

### Tables

![02bbf2ecdc288f903dea72494038a02651bc02877b74998b22828b7804bc4281.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/02bbf2ecdc288f903dea72494038a02651bc02877b74998b22828b7804bc4281.jpg)

![36b72b4c0b9e4e8aaba3dc442aefc602a0e63cb750cda32b15e157d6c1bd44d0.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/36b72b4c0b9e4e8aaba3dc442aefc602a0e63cb750cda32b15e157d6c1bd44d0.jpg)

![650ef3bea928a5efdfdad2d1ce4a357dcb292bb322ca99162c0a73bdf6151cfd.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/650ef3bea928a5efdfdad2d1ce4a357dcb292bb322ca99162c0a73bdf6151cfd.jpg)

![6a5d50879e627b16a49c209485cc8c748c998e4cb6768e1684aa74329421539b.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/6a5d50879e627b16a49c209485cc8c748c998e4cb6768e1684aa74329421539b.jpg)

![6e4ea6d4fc88a1ac5b302937a0e2dfb51757dbca70c25886b8031ed5f4b8c9dd.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/6e4ea6d4fc88a1ac5b302937a0e2dfb51757dbca70c25886b8031ed5f4b8c9dd.jpg)

![76ac6cad06b4b4d0f62c3a86bfe61e29c5f02171883e08769417db6cb3323406.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/76ac6cad06b4b4d0f62c3a86bfe61e29c5f02171883e08769417db6cb3323406.jpg)

![90d18875d210937c0214804044b938878c537a9907fe3d6d4d5b3f433c78f127.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/90d18875d210937c0214804044b938878c537a9907fe3d6d4d5b3f433c78f127.jpg)

![99e9d4f83bd24ff6540459d94685ad9d846b6bfffee921504a6dcb5815252096.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/99e9d4f83bd24ff6540459d94685ad9d846b6bfffee921504a6dcb5815252096.jpg)

![9cca830c5fc8ba4fe2915c7c1063d00946d874701abb03d049d8e908ac1c1bf7.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/9cca830c5fc8ba4fe2915c7c1063d00946d874701abb03d049d8e908ac1c1bf7.jpg)

![c169a7e2534d2f065eedf34f39ee4963752a327ff373468f983d7194af7c5e4b.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/c169a7e2534d2f065eedf34f39ee4963752a327ff373468f983d7194af7c5e4b.jpg)

![e7a2662c5bec551b3dd3bf1302ccef264097695e01052cfbe161e8a814f5917b.jpg](../icml_results/2300_Think%20Twice%20Before%20You%20Act_%20Enhancing%20Agent%20Behavioral%20Safety%20with%20Thought%20Correction/tables/e7a2662c5bec551b3dd3bf1302ccef264097695e01052cfbe161e8a814f5917b.jpg)

<a id="paper-2272"></a>

## Ophiuchus: Incentivizing Tool-augmented ”Think with Images” for Joint Medical Segmentation, Understanding and Reasoning


### Images

![04a1889581a2b734897170006c32ad1a8f18124ff64b1d7997f5e8d9e19a94e5.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/04a1889581a2b734897170006c32ad1a8f18124ff64b1d7997f5e8d9e19a94e5.jpg)

![06df7432a4738de1f7fa289087295cac0e6263a280fe0f9ce5c0724848a867f2.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/06df7432a4738de1f7fa289087295cac0e6263a280fe0f9ce5c0724848a867f2.jpg)

![0de73499cab39f1eaf940a6581960438eeb8aab340dc1dc3d79357beed039a7c.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/0de73499cab39f1eaf940a6581960438eeb8aab340dc1dc3d79357beed039a7c.jpg)

![0ea839e8a47c71e44945953850d3692914710474b5c147a595a1ded9007d769b.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/0ea839e8a47c71e44945953850d3692914710474b5c147a595a1ded9007d769b.jpg)

![0efc2f209282d7ac0fd164d2d46a4fdd70cd901ffb48b4d104586c6a1d0df58f.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/0efc2f209282d7ac0fd164d2d46a4fdd70cd901ffb48b4d104586c6a1d0df58f.jpg)

![1e0676d95ba42ba49017f262708c7bc67fdc1423d6cf20879023c5f3b1511117.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/1e0676d95ba42ba49017f262708c7bc67fdc1423d6cf20879023c5f3b1511117.jpg)

![204f67747b9b90aa6bc306b4b10b5ee8c259f5828f4db0c97325b37cb9234c62.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/204f67747b9b90aa6bc306b4b10b5ee8c259f5828f4db0c97325b37cb9234c62.jpg)

![223732b43d259c528e951881ac8653342f69756449a9ba79624f8552880d748a.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/223732b43d259c528e951881ac8653342f69756449a9ba79624f8552880d748a.jpg)

![29dc06164d5017464f20c4878ef1aaa203ba4ae1c753580c9e8ccd38b5dfd58a.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/29dc06164d5017464f20c4878ef1aaa203ba4ae1c753580c9e8ccd38b5dfd58a.jpg)

![38758dc93f43f8e3c7dd2112a4723c9cd8550bf7c30051e98823201b83e839ea.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/38758dc93f43f8e3c7dd2112a4723c9cd8550bf7c30051e98823201b83e839ea.jpg)

![421c98ca7db47ce3e71be94e3d52fc2edead8960ed221fb2756bf7f8656fc724.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/421c98ca7db47ce3e71be94e3d52fc2edead8960ed221fb2756bf7f8656fc724.jpg)

![449847009b1c58a6abb35712cfa521565003e06e3ca3d97935cd0a4ece9928b9.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/449847009b1c58a6abb35712cfa521565003e06e3ca3d97935cd0a4ece9928b9.jpg)

![5401d55f069fc004fa53f9acdceeae64f6634c8402a419c4425e3b4aa0dc1c3b.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/5401d55f069fc004fa53f9acdceeae64f6634c8402a419c4425e3b4aa0dc1c3b.jpg)

![5c329ae8d0a575311dc598cffc9f3e7f161cd761ffd8d04d6fb36cb76777d91d.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/5c329ae8d0a575311dc598cffc9f3e7f161cd761ffd8d04d6fb36cb76777d91d.jpg)

![68356db132ff9c86591913b47f654bdcf415441e38b63c9485105b699c6218df.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/68356db132ff9c86591913b47f654bdcf415441e38b63c9485105b699c6218df.jpg)

![728d6ae24ebd2024f60f24e70eadd70f348b91597c539d48fb692d3127e3c192.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/728d6ae24ebd2024f60f24e70eadd70f348b91597c539d48fb692d3127e3c192.jpg)

![77c53f9de2b6e57b725f4636cb46f1b5157c85ddb6e5872430a80bc8bf45e8c8.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/77c53f9de2b6e57b725f4636cb46f1b5157c85ddb6e5872430a80bc8bf45e8c8.jpg)

![79d1f0e48ce227c6f6e17f8fe4418d6f5370e6737fcda4b8470a69d5e56a37ed.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/79d1f0e48ce227c6f6e17f8fe4418d6f5370e6737fcda4b8470a69d5e56a37ed.jpg)

![8b3ebe046ea848787a1c78e4149e655c2bee2912f1ab0fa03122ab61f4a1749e.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/8b3ebe046ea848787a1c78e4149e655c2bee2912f1ab0fa03122ab61f4a1749e.jpg)

![912ca941777fa32e2507133432dc9ae536b20bb98c385f82b4a540e5ac745370.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/912ca941777fa32e2507133432dc9ae536b20bb98c385f82b4a540e5ac745370.jpg)

![9e198c54e5f030f86f329c44f0ebf2e6fc505a8ac2d3bc041f463d15299ec53b.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/9e198c54e5f030f86f329c44f0ebf2e6fc505a8ac2d3bc041f463d15299ec53b.jpg)

![a00385ec3a24fe3b25bf93c2c4c811f66ac090539a4bc55bfee359fdaf160af7.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/a00385ec3a24fe3b25bf93c2c4c811f66ac090539a4bc55bfee359fdaf160af7.jpg)

![a254fa28c98e7cf3ad2a8e344f1c20bf78f84592194fed6758f33719eb9b1983.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/a254fa28c98e7cf3ad2a8e344f1c20bf78f84592194fed6758f33719eb9b1983.jpg)

![b43d60414d15dca7753bb5a33c28a7073e2ba9c4dba96e378ce4f9efaa57cecc.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/b43d60414d15dca7753bb5a33c28a7073e2ba9c4dba96e378ce4f9efaa57cecc.jpg)

![c71869dc33defec212d738a985bdbd9264cf472d5e71ef0ef7d8ddc49b4e67ae.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/c71869dc33defec212d738a985bdbd9264cf472d5e71ef0ef7d8ddc49b4e67ae.jpg)

![ce162964d15b90c09b9f77b7cfcac1dd77c24a088a5ec1f104e31701e4890bf3.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/ce162964d15b90c09b9f77b7cfcac1dd77c24a088a5ec1f104e31701e4890bf3.jpg)

![d35038ca9a4106f59921a67ab8473a47047270384755d39ceff54eaa8571fa50.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/d35038ca9a4106f59921a67ab8473a47047270384755d39ceff54eaa8571fa50.jpg)

![dc77b140d8c3c5d37db3041d597eb23bd9d60fefebcef6143659a157c283669c.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/dc77b140d8c3c5d37db3041d597eb23bd9d60fefebcef6143659a157c283669c.jpg)

![dc8ceeb21b484216ebc77a6e7a0bb55b55005b1dd00d83f8ef51f1a466a90d1f.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/dc8ceeb21b484216ebc77a6e7a0bb55b55005b1dd00d83f8ef51f1a466a90d1f.jpg)

![e51666e3ee931c6e673ecae9a2c8819c90bbd31fbd4ff1a5a1d974a8c3bd5687.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/e51666e3ee931c6e673ecae9a2c8819c90bbd31fbd4ff1a5a1d974a8c3bd5687.jpg)

![e5e939b21849e5c4d126c644684995da56fe7b9803330f956ffbcda6121df3ce.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/e5e939b21849e5c4d126c644684995da56fe7b9803330f956ffbcda6121df3ce.jpg)

![fd9e052a0758dc28426cfd9d390a0348ae2bf7184d052569330b5cf0ac000d02.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/images/fd9e052a0758dc28426cfd9d390a0348ae2bf7184d052569330b5cf0ac000d02.jpg)

### Tables

![3d389543b8f1c036d0f876c87f3bb33b8ce4239e6c293088bec16cc5f6ac1b3c.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/3d389543b8f1c036d0f876c87f3bb33b8ce4239e6c293088bec16cc5f6ac1b3c.jpg)

![40c9b596a2d72c5b19a0afd6921588cc719ed25396418bcb5e543caf4b122748.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/40c9b596a2d72c5b19a0afd6921588cc719ed25396418bcb5e543caf4b122748.jpg)

![4acc949992fa3ce0996a1264f5d30c7104b4ff975008499c644fdf8e5971eded.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/4acc949992fa3ce0996a1264f5d30c7104b4ff975008499c644fdf8e5971eded.jpg)

![4da4ee5ad94ec8f705e0f8f91b9063aaecc0677aff162d5fc5d9d61b122fec7a.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/4da4ee5ad94ec8f705e0f8f91b9063aaecc0677aff162d5fc5d9d61b122fec7a.jpg)

![63b8dbd03b2d61cd46ce2073a14778c7f03e30584049c52760571c5b2f02c39b.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/63b8dbd03b2d61cd46ce2073a14778c7f03e30584049c52760571c5b2f02c39b.jpg)

![7594de1259f6372f44f307c9cc1c4abc0a5dc4ab94b58737123b66e54c9ec958.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/7594de1259f6372f44f307c9cc1c4abc0a5dc4ab94b58737123b66e54c9ec958.jpg)

![80bc5b74bc5715400f7423f62eb3c6e3db26993d7c94f55931402bb44cc5b455.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/80bc5b74bc5715400f7423f62eb3c6e3db26993d7c94f55931402bb44cc5b455.jpg)

![8a88828135ea4478f6e6909c110c0bca51eaa5834df85ce514976a1e3fe033ab.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/8a88828135ea4478f6e6909c110c0bca51eaa5834df85ce514976a1e3fe033ab.jpg)

![9b679538362d22e41979374f035556d17b13d77d19287c7f21089379157d88dd.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/9b679538362d22e41979374f035556d17b13d77d19287c7f21089379157d88dd.jpg)

![a1c6ce48468622e274932dc3fa775a44cb3904631a3b79ada18752bd503b0f3a.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/a1c6ce48468622e274932dc3fa775a44cb3904631a3b79ada18752bd503b0f3a.jpg)

![a3959d6b60c486d904067fa626ad239d9b6a6e6a4d40b38b9b78ee72de8e0dd4.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/a3959d6b60c486d904067fa626ad239d9b6a6e6a4d40b38b9b78ee72de8e0dd4.jpg)

![c0fa14455b36c9d002fa4047dc4a5179c7a04a4dd214de12caca6b0c5c325054.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/c0fa14455b36c9d002fa4047dc4a5179c7a04a4dd214de12caca6b0c5c325054.jpg)

![c1021e4db64ccf2e6c573a6e12325079e1e293669029f8086e67b37ae1172fb6.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/c1021e4db64ccf2e6c573a6e12325079e1e293669029f8086e67b37ae1172fb6.jpg)

![d15544399a1e2e7cc7c4286c315b714beea46f19e47fa60b814a52518180eb63.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/d15544399a1e2e7cc7c4286c315b714beea46f19e47fa60b814a52518180eb63.jpg)

![e33564762914c6923a5dbe470b43edb4f9087eebb589118100268a37f0d82b33.jpg](../icml_results/2301_Ophiuchus_%20Incentivizing%20Tool-augmented%20%E2%80%9DThink%20with%20Images%E2%80%9D%20for%20Joint%20Medical%20Segmentation%2C%20Understa/tables/e33564762914c6923a5dbe470b43edb4f9087eebb589118100268a37f0d82b33.jpg)

<a id="paper-2273"></a>

## NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration


### Images

![0117a06c2e2545620dd79406de6a2427139d65a51e36979828a05e52e5cb5997.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/0117a06c2e2545620dd79406de6a2427139d65a51e36979828a05e52e5cb5997.jpg)

![1c0706a9a2cfd57888daf17aa0d21cb1a3d832179122561bda3c13a39d3e7d0e.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/1c0706a9a2cfd57888daf17aa0d21cb1a3d832179122561bda3c13a39d3e7d0e.jpg)

![43e7403f29937282bfe4520a5bf882789072508c77df623f8923016aa2e163ee.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/43e7403f29937282bfe4520a5bf882789072508c77df623f8923016aa2e163ee.jpg)

![51a3258cced278b5b203e1dcedda7001881084f4bc6a9fde42ec57166700843a.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/51a3258cced278b5b203e1dcedda7001881084f4bc6a9fde42ec57166700843a.jpg)

![75c483ef18c4d9cf7740774319a0750f9280278108acdfafa25a1d53cbaec3cc.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/75c483ef18c4d9cf7740774319a0750f9280278108acdfafa25a1d53cbaec3cc.jpg)

![88f4cb44c8e92f804dcb69e01b5a62aeab0abdf93ff99387b7567d54e7fe3539.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/88f4cb44c8e92f804dcb69e01b5a62aeab0abdf93ff99387b7567d54e7fe3539.jpg)

![d510d11e87fe3e8f4543965b54efc69218f59e12b6df25d7482db8d88888ff12.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/d510d11e87fe3e8f4543965b54efc69218f59e12b6df25d7482db8d88888ff12.jpg)

![e1f07ca6cb5bd23ed04476c0cd14ad44508cb2e2eb04ccda021c48b081d0d75b.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/e1f07ca6cb5bd23ed04476c0cd14ad44508cb2e2eb04ccda021c48b081d0d75b.jpg)

![ed44a9d8a43e1a069c02640fce4b451741e8c758555dcb9265a5137b0136a203.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/images/ed44a9d8a43e1a069c02640fce4b451741e8c758555dcb9265a5137b0136a203.jpg)

### Tables

![467ce4bd2096ca04a35015597d331bbb6c9b1d4019de535ff9ba3bbe01c57e1d.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/467ce4bd2096ca04a35015597d331bbb6c9b1d4019de535ff9ba3bbe01c57e1d.jpg)

![482080fe1514f2f65f2609b847a9fe89960af9eec48fd336df4eb6210c7fee20.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/482080fe1514f2f65f2609b847a9fe89960af9eec48fd336df4eb6210c7fee20.jpg)

![4c24ba0282a0cc00e6e8182d329c39b530efb150a13a7afede8cb485ba540f5c.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/4c24ba0282a0cc00e6e8182d329c39b530efb150a13a7afede8cb485ba540f5c.jpg)

![55c5378beea85155bc0e56fc66bb2af49a117aca270406bcc2aa7544897cdc1f.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/55c5378beea85155bc0e56fc66bb2af49a117aca270406bcc2aa7544897cdc1f.jpg)

![6cff75fb605f00f978ffc7172590f087027452a61583a73343215f6c58636c7c.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/6cff75fb605f00f978ffc7172590f087027452a61583a73343215f6c58636c7c.jpg)

![6f6bc1aad576a6bdc81fc9d47ba49a79d2b3b0fe21a5b57a4afe2b77998e43d1.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/6f6bc1aad576a6bdc81fc9d47ba49a79d2b3b0fe21a5b57a4afe2b77998e43d1.jpg)

![b0fe0b70744382d0e9f47a7a37e2a21c14a7b1abc584aaf543646d229d6c9c5b.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/b0fe0b70744382d0e9f47a7a37e2a21c14a7b1abc584aaf543646d229d6c9c5b.jpg)

![c50c4ddf6eb9d4531c46e8453df654add14e5344770011d9ae276dfd5935cac9.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/c50c4ddf6eb9d4531c46e8453df654add14e5344770011d9ae276dfd5935cac9.jpg)

![d163a94b8af1c81d038a2bf63afa668e22591b2725495772e0b343fcd64f6fbd.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/d163a94b8af1c81d038a2bf63afa668e22591b2725495772e0b343fcd64f6fbd.jpg)

![dc8439ea32ae3df54ea66e87c3538787c323bad7807b095f7d96ec34873ecb9f.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/dc8439ea32ae3df54ea66e87c3538787c323bad7807b095f7d96ec34873ecb9f.jpg)

![f218bb3fdd5d0741857a3e983e07eb1629bd836b123398ba6688571e4d69f836.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/f218bb3fdd5d0741857a3e983e07eb1629bd836b123398ba6688571e4d69f836.jpg)

![fcef01026922b23df232f4f536ff9ec05d80c3f0d88d9009d1236aed70918e9c.jpg](../icml_results/2302_NaviAgent_%20Graph-Driven%20Bilevel%20Planning%20for%20Scalable%20Tool%20Orchestration/tables/fcef01026922b23df232f4f536ff9ec05d80c3f0d88d9009d1236aed70918e9c.jpg)

<a id="paper-2274"></a>

## Seeing Symbols, Missing Structure: A Real-World Handwritten Mathematical Expression Recognition Benchmark for Large Models


### Images

![01c7f3a57a0faaa2a5b9f0fd810b2d139a3f71773f3f041685df91f6a258627a.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/01c7f3a57a0faaa2a5b9f0fd810b2d139a3f71773f3f041685df91f6a258627a.jpg)

![0a9ca46b42b6988483a72c8928a2e490de02fd575fc0bbbc0ab5d1168b2833f4.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/0a9ca46b42b6988483a72c8928a2e490de02fd575fc0bbbc0ab5d1168b2833f4.jpg)

![1d0ba4de6824fb51ba27c07d348a1f1ccdcd0cf560a266fdbfa70da930d44050.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/1d0ba4de6824fb51ba27c07d348a1f1ccdcd0cf560a266fdbfa70da930d44050.jpg)

![2b08edf8abe22a70b5c976956338ecb9c06c44085736dbebc377f71c1813e435.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/2b08edf8abe22a70b5c976956338ecb9c06c44085736dbebc377f71c1813e435.jpg)

![438164aa651e9cc8376eb611bae0cccf5ab5a37048891edc58619c3e8001e546.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/438164aa651e9cc8376eb611bae0cccf5ab5a37048891edc58619c3e8001e546.jpg)

![445789b0fef9134f0ab11bbb13608080fb57939e8e87cfe0952fabf512572b5c.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/445789b0fef9134f0ab11bbb13608080fb57939e8e87cfe0952fabf512572b5c.jpg)

![5f65117de96123921b4bafdcd0f1e5d9a0d8731d8c83b5fb6e956be8c0332ca5.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/5f65117de96123921b4bafdcd0f1e5d9a0d8731d8c83b5fb6e956be8c0332ca5.jpg)

![6998edc5813ddff138f26e5ce0656096d6d50439988723fd605c8fc3f292949e.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/6998edc5813ddff138f26e5ce0656096d6d50439988723fd605c8fc3f292949e.jpg)

![72c05ba72081ea2b84c22aaa2cd01eab384ffe8e30b16420a8f3bed15be2e78f.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/72c05ba72081ea2b84c22aaa2cd01eab384ffe8e30b16420a8f3bed15be2e78f.jpg)

![787f90b4dca60c6295778493c0b4bc5ea8227e4cda214d08355378c59e82b0ba.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/787f90b4dca60c6295778493c0b4bc5ea8227e4cda214d08355378c59e82b0ba.jpg)

![88e41b67258dbb732b168391159b7f1b731d7f701225a5753d019a9e5bb143e0.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/88e41b67258dbb732b168391159b7f1b731d7f701225a5753d019a9e5bb143e0.jpg)

![e8f196489b8819052c6deba433d6c1f6a637c13dac6cc4c151053c338b3ab913.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/e8f196489b8819052c6deba433d6c1f6a637c13dac6cc4c151053c338b3ab913.jpg)

![f57aaa6e2f14262c4eaf2a35bd319d2abed7bcdc728601d732c636a7bf8ba8e7.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/images/f57aaa6e2f14262c4eaf2a35bd319d2abed7bcdc728601d732c636a7bf8ba8e7.jpg)

### Tables

![005194a4be283e9261dd7f905f30219c9b937f33a043e4e50102bd632955ecd8.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/005194a4be283e9261dd7f905f30219c9b937f33a043e4e50102bd632955ecd8.jpg)

![27bb3e5b472361886702b512b19d04bb1cc2233c42b390e22e7ecaf782a54db9.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/27bb3e5b472361886702b512b19d04bb1cc2233c42b390e22e7ecaf782a54db9.jpg)

![4eb02f841c767115ecc7843464be82913e3c003fdee3ece40173a24880f26fa2.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/4eb02f841c767115ecc7843464be82913e3c003fdee3ece40173a24880f26fa2.jpg)

![5659c54c181a8dd467b1ff9c54c43721d5a51b279b1ee9cdb0b0d4591f6c9f97.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/5659c54c181a8dd467b1ff9c54c43721d5a51b279b1ee9cdb0b0d4591f6c9f97.jpg)

![5b0d02326a5f5c4c8842424c8b1457e7b5fd091aab54057b94674215667e7e04.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/5b0d02326a5f5c4c8842424c8b1457e7b5fd091aab54057b94674215667e7e04.jpg)

![8999a0fa61ac65904197fa066a0c0fa2721077b33acd05c898fe7cd1bde4ecc3.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/8999a0fa61ac65904197fa066a0c0fa2721077b33acd05c898fe7cd1bde4ecc3.jpg)

![a252e84fadddb3fd63f8301c70a3b4867ba8d46c452ded2cb97902a1c7834680.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/a252e84fadddb3fd63f8301c70a3b4867ba8d46c452ded2cb97902a1c7834680.jpg)

![a430c219f68381a87b97e24c1c3ab51f91461e68638b3df5be355892a9852b93.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/a430c219f68381a87b97e24c1c3ab51f91461e68638b3df5be355892a9852b93.jpg)

![b5d2b8d88e5d3ecfc32db0e25c9be684a07c798cdccbfca2c97fc07ae092d2e4.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/b5d2b8d88e5d3ecfc32db0e25c9be684a07c798cdccbfca2c97fc07ae092d2e4.jpg)

![c418143fd9782c9da78801886b2aa551b1ff949edae49b2798ee77c4a3177808.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/c418143fd9782c9da78801886b2aa551b1ff949edae49b2798ee77c4a3177808.jpg)

![dd98a9fddc591779509104546401d3459b0f8c33395ee2cda60f37ca31a3b0d3.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/dd98a9fddc591779509104546401d3459b0f8c33395ee2cda60f37ca31a3b0d3.jpg)

![e78833f34957bdb1189aebe76c28cf03bdf1ac182dd5ffccedfabea9c7aab891.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/e78833f34957bdb1189aebe76c28cf03bdf1ac182dd5ffccedfabea9c7aab891.jpg)

![ffc15b9abc5753171fab770a1f53bcf8eb6e6b05bb986f1156a07626ff300895.jpg](../icml_results/2303_Seeing%20Symbols%2C%20Missing%20Structure_%20A%20Real-World%20Handwritten%20Mathematical%20Expression%20Recognition%20Benc/tables/ffc15b9abc5753171fab770a1f53bcf8eb6e6b05bb986f1156a07626ff300895.jpg)

<a id="paper-2275"></a>

## PICACO: Pluralistic In-Context Value Alignment via Total Correlation Optimization


### Images

![384429758ed978c9d0c06b9858554706c3347534565de83ecf7a65e7a8e89e4e.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/384429758ed978c9d0c06b9858554706c3347534565de83ecf7a65e7a8e89e4e.jpg)

![38bc11f032144fa93c4657b73185833772dda5897608bbc209964c24331feb76.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/38bc11f032144fa93c4657b73185833772dda5897608bbc209964c24331feb76.jpg)

![90d48f6c831d041bfa306c0111c180b4f6a3b53a95c6955a52af5e88621bdc19.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/90d48f6c831d041bfa306c0111c180b4f6a3b53a95c6955a52af5e88621bdc19.jpg)

![9c688cbe9b46eb2167d772df4df32681677dbb5d587ace825495ce5e6159d0a9.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/9c688cbe9b46eb2167d772df4df32681677dbb5d587ace825495ce5e6159d0a9.jpg)

![a846138134c6c4cca2a89a950fd086239547d11a24f88be61ac21f61f644bcac.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/a846138134c6c4cca2a89a950fd086239547d11a24f88be61ac21f61f644bcac.jpg)

![b5546dd589602faf2f76b19e17ca017d2dd21b7618fc3f26c8a2e5f2b8b181a1.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/b5546dd589602faf2f76b19e17ca017d2dd21b7618fc3f26c8a2e5f2b8b181a1.jpg)

![d274c80323ecc854479594024197ed699ce19310032f89346a4ae41118405cb5.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/images/d274c80323ecc854479594024197ed699ce19310032f89346a4ae41118405cb5.jpg)

### Tables

![121cd776fec0fafe5b7125b58c52353c971665a613a66cf25f1672db66c8b1e9.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/121cd776fec0fafe5b7125b58c52353c971665a613a66cf25f1672db66c8b1e9.jpg)

![215e25e19e4bc2bd4449d7db5b346d32d243e1271e6f3c367a28f00f10a755f4.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/215e25e19e4bc2bd4449d7db5b346d32d243e1271e6f3c367a28f00f10a755f4.jpg)

![377a438125ecbb44dc343e2bbf8c517e454ff279930b93c7ddab7f14483d8bd8.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/377a438125ecbb44dc343e2bbf8c517e454ff279930b93c7ddab7f14483d8bd8.jpg)

![3bfa31ee782b23cf45c462d52fa85bee68095d9aa7aad299c36284f6e0560ade.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/3bfa31ee782b23cf45c462d52fa85bee68095d9aa7aad299c36284f6e0560ade.jpg)

![50c399434b64b99366c442e0256d1518165a78a96324616e548aa2062ad7c779.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/50c399434b64b99366c442e0256d1518165a78a96324616e548aa2062ad7c779.jpg)

![50eafaba3107e4d026dc89dbfa49614c948b049481a765438782b105c1d52c0a.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/50eafaba3107e4d026dc89dbfa49614c948b049481a765438782b105c1d52c0a.jpg)

![52f9b25146b4956a65fab82c1d31a4ff5a4f72881534b67d627eeb73622cc822.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/52f9b25146b4956a65fab82c1d31a4ff5a4f72881534b67d627eeb73622cc822.jpg)

![78ed803eb923fd1028c76f7d5501dcfcd8413af72c79ee1b66e42d4fe7339958.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/78ed803eb923fd1028c76f7d5501dcfcd8413af72c79ee1b66e42d4fe7339958.jpg)

![946b3dca7f72af76e58d2a74fb3dd369be42d08a5e5b9471302d2226e12882a3.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/946b3dca7f72af76e58d2a74fb3dd369be42d08a5e5b9471302d2226e12882a3.jpg)

![9836215e0427286601919b159ec081b5ed2ac1884d65e90c21c1c0cb4e9f3046.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/9836215e0427286601919b159ec081b5ed2ac1884d65e90c21c1c0cb4e9f3046.jpg)

![9e6fea4e018371a05add6bb1d74c7ca60f40fe6cf3058c2db506716f6ef42f0b.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/9e6fea4e018371a05add6bb1d74c7ca60f40fe6cf3058c2db506716f6ef42f0b.jpg)

![adae8281db4e0f4cb4f4fab917fd6fa73078b422210fb291d4bda7da0653b923.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/adae8281db4e0f4cb4f4fab917fd6fa73078b422210fb291d4bda7da0653b923.jpg)

![c70951d436e754c12ad9400756ee6f05f4a31d1e078d0d4785b585edb7d0e1cf.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/c70951d436e754c12ad9400756ee6f05f4a31d1e078d0d4785b585edb7d0e1cf.jpg)

![c7a78bd5e61b22a5f47761e24c9bde45d336eb0dbff9bfd71501930c98e906d4.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/c7a78bd5e61b22a5f47761e24c9bde45d336eb0dbff9bfd71501930c98e906d4.jpg)

![d127a48d73809e9afcba926df5b029f8982d87961c0c499fe8e75ddfda77685b.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/d127a48d73809e9afcba926df5b029f8982d87961c0c499fe8e75ddfda77685b.jpg)

![e7ca6bea645ed24c74de04f9cec42501d433768de4ad0aef3d946eb1cdda7be5.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/e7ca6bea645ed24c74de04f9cec42501d433768de4ad0aef3d946eb1cdda7be5.jpg)

![e824ecc29837e420d6ff7db4ceca3ffac5f55f30979bff3471d5e3af5caad632.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/e824ecc29837e420d6ff7db4ceca3ffac5f55f30979bff3471d5e3af5caad632.jpg)

![ec5ab2b16f74326ee919ad5f1cda13aaf22b6a6a0ca1672825e3a2c412d0a44a.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/ec5ab2b16f74326ee919ad5f1cda13aaf22b6a6a0ca1672825e3a2c412d0a44a.jpg)

![f3a923e8830bc53498f0e62603f2b326070b4a656e353ab9a1888febc05b8140.jpg](../icml_results/2304_PICACO_%20Pluralistic%20In-Context%20Value%20Alignment%20via%20Total%20Correlation%20Optimization/tables/f3a923e8830bc53498f0e62603f2b326070b4a656e353ab9a1888febc05b8140.jpg)

<a id="paper-2276"></a>

## $\textttPRISM$:A 3D Probabilistic Neural Representation for Interpretable Shape Modeling


### Images

![2b64a50c0cb55dbb74d517c66eb464389e50c9568e06b89f672875913f60f7a8.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/2b64a50c0cb55dbb74d517c66eb464389e50c9568e06b89f672875913f60f7a8.jpg)

![5790ba902fb3ef44b8114ff435084288d37e1ab3840843724992f9c2efaa67ba.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/5790ba902fb3ef44b8114ff435084288d37e1ab3840843724992f9c2efaa67ba.jpg)

![76a095f37ddda0b359c977ce3995438a261b8ac2c659582ce5a01b69ecf5541c.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/76a095f37ddda0b359c977ce3995438a261b8ac2c659582ce5a01b69ecf5541c.jpg)

![8913d5b9ce6c70faa09671322bab5a87fec71cad81a8b736ed2eff94c18411c3.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/8913d5b9ce6c70faa09671322bab5a87fec71cad81a8b736ed2eff94c18411c3.jpg)

![8bc1e02f0ea930f4552062d2190b9eb067ff3d11a28066d62fd2a72e72148527.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/8bc1e02f0ea930f4552062d2190b9eb067ff3d11a28066d62fd2a72e72148527.jpg)

![aa352c2ee01c85918ff85ad5de5e19df6cc4ab5d6ff0276590b09a72c0b00cbc.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/aa352c2ee01c85918ff85ad5de5e19df6cc4ab5d6ff0276590b09a72c0b00cbc.jpg)

![c679b4ad2fde7dd084fb876db6b94b1923f77123103d96952ff7ae224629abfd.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/images/c679b4ad2fde7dd084fb876db6b94b1923f77123103d96952ff7ae224629abfd.jpg)

### Tables

![361ab7423b75957d20a4b9da11d9c47d9763ba1573c6196863e11ff2a04c64af.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/361ab7423b75957d20a4b9da11d9c47d9763ba1573c6196863e11ff2a04c64af.jpg)

![37dcb8975eb86106d18307e61f865e9a74820cf3486a8a392061c437d5f793a7.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/37dcb8975eb86106d18307e61f865e9a74820cf3486a8a392061c437d5f793a7.jpg)

![6a81e81c6e96a9b1e72956f165d4edddc9ce7b30fbe25661e50fe58cdf68ed5e.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/6a81e81c6e96a9b1e72956f165d4edddc9ce7b30fbe25661e50fe58cdf68ed5e.jpg)

![87233fc8262a483d96626e8bb0547291da9efd712bcde19322a29053a6b21c7e.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/87233fc8262a483d96626e8bb0547291da9efd712bcde19322a29053a6b21c7e.jpg)

![9ebae95881000ef06403130206e910c6e6bd3e689eca2f772ac4a45ed4ad1fa3.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/9ebae95881000ef06403130206e910c6e6bd3e689eca2f772ac4a45ed4ad1fa3.jpg)

![a4276850b47aa024f47c5e89154f105940ad2768a69c84a4b3aded309535234a.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/a4276850b47aa024f47c5e89154f105940ad2768a69c84a4b3aded309535234a.jpg)

![c84e48fbef0d22f25fb083e3fe1a524bb672661d82aaa82622e512c7c399a988.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/c84e48fbef0d22f25fb083e3fe1a524bb672661d82aaa82622e512c7c399a988.jpg)

![df32b04f6bb31362700162947416067cf256961e3790d688db539df4121d7ce1.jpg](../icml_results/2305_%24_textttPRISM%24_A%203D%20Probabilistic%20Neural%20Representation%20for%20Interpretable%20Shape%20Modeling/tables/df32b04f6bb31362700162947416067cf256961e3790d688db539df4121d7ce1.jpg)

<a id="paper-2277"></a>

## DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?

### Images

![24a3b7abb035589a2c3460f03305e6c359e56b884a9c38853213f1baaa8ffb74.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/24a3b7abb035589a2c3460f03305e6c359e56b884a9c38853213f1baaa8ffb74.jpg)

![59ae8e6bdfc09b2b6dd65957d123aedaba2e10086cc4b515de17f861fb1c910a.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/59ae8e6bdfc09b2b6dd65957d123aedaba2e10086cc4b515de17f861fb1c910a.jpg)

![6c3a1e5fbb56478f2ec3c47e630be092c86be6d7e0279fb781082c2f885d5cb3.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/6c3a1e5fbb56478f2ec3c47e630be092c86be6d7e0279fb781082c2f885d5cb3.jpg)

![7252fbab7b846f3bc480c84f75003060c6a431a2c2baebc9d3d9937af6c4363b.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/7252fbab7b846f3bc480c84f75003060c6a431a2c2baebc9d3d9937af6c4363b.jpg)

![7c325c88ce28f4d6127879f9f8d461ca7fd099aeab982653eabd09e21507d45d.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/7c325c88ce28f4d6127879f9f8d461ca7fd099aeab982653eabd09e21507d45d.jpg)

![7d84486c56dc94557b62ff2ec0c611328b31cab631c67b401890660a9882a0bb.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/7d84486c56dc94557b62ff2ec0c611328b31cab631c67b401890660a9882a0bb.jpg)

![a19269b6769d0b116eb087a977f38bf5357af2a4f64e46e06b945df040bc9f65.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/a19269b6769d0b116eb087a977f38bf5357af2a4f64e46e06b945df040bc9f65.jpg)

![ba5cb39720768927c7c00490e769c09bfacebff9a01a64e84afe530cff604a9f.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/ba5cb39720768927c7c00490e769c09bfacebff9a01a64e84afe530cff604a9f.jpg)

![c4a0d2e446e28eceedb4a37500bca7d12e27fd57c692bd2203f4bb67968207e6.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/c4a0d2e446e28eceedb4a37500bca7d12e27fd57c692bd2203f4bb67968207e6.jpg)

![eb25ff8e89407da4f8d248a09c47e493e5b77373409065072dc3f271b81e9499.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/images/eb25ff8e89407da4f8d248a09c47e493e5b77373409065072dc3f271b81e9499.jpg)

### Tables

![05f38c8467c30772337e7b1c8fd9c62a9f4c6539bcd7472020e2dd9f1e747e23.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/05f38c8467c30772337e7b1c8fd9c62a9f4c6539bcd7472020e2dd9f1e747e23.jpg)

![29e70a6d1a83f147fb2e4eb2c1ad2bd8819b6efea90e82ffbf7d85e6bed859e6.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/29e70a6d1a83f147fb2e4eb2c1ad2bd8819b6efea90e82ffbf7d85e6bed859e6.jpg)

![319d31c9666ca3dd5ec3522ed05910eddd12ebfa5d0544a85833c82b6867ac42.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/319d31c9666ca3dd5ec3522ed05910eddd12ebfa5d0544a85833c82b6867ac42.jpg)

![3976c5c66a6b74fcfe6395917785754c8009b44230d563df396ba0e34eb30423.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/3976c5c66a6b74fcfe6395917785754c8009b44230d563df396ba0e34eb30423.jpg)

![53c033ecd35c528c30bb0e6c30765922f81a9f2b07625f66fb3b4a11e1355843.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/53c033ecd35c528c30bb0e6c30765922f81a9f2b07625f66fb3b4a11e1355843.jpg)

![55403c8d8ff1c53f7931539ed74ea63a4e67beb0117a99a6b9c363153b4891f1.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/55403c8d8ff1c53f7931539ed74ea63a4e67beb0117a99a6b9c363153b4891f1.jpg)

![5681441a329fc2f1c70f6aab28d1b686b7c9298453fa72e37f0d382275a877c0.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/5681441a329fc2f1c70f6aab28d1b686b7c9298453fa72e37f0d382275a877c0.jpg)

![5d1043acf173d975da69360f29b7627e958288e39f9c6dea33ac8aca2ad660a3.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/5d1043acf173d975da69360f29b7627e958288e39f9c6dea33ac8aca2ad660a3.jpg)

![662b9ff21a23899ba7418bcf729b64a9bc8e272c046de5cc19c4b98d4c98e949.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/662b9ff21a23899ba7418bcf729b64a9bc8e272c046de5cc19c4b98d4c98e949.jpg)

![6bddc38affd3c398c4ba0a83ba362a8af68a62414308a61b0227063f7533611d.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/6bddc38affd3c398c4ba0a83ba362a8af68a62414308a61b0227063f7533611d.jpg)

![6f0d423a1663e5ef932eaceb45acf3b28f4026fda605e914316e59138119a0af.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/6f0d423a1663e5ef932eaceb45acf3b28f4026fda605e914316e59138119a0af.jpg)

![7ea8e3b05156e92965116fde03bf72275423787b14147472d10c43f7d762b71f.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/7ea8e3b05156e92965116fde03bf72275423787b14147472d10c43f7d762b71f.jpg)

![8444bcf73727d24d7ca59a1fa130fc89e1033f2977af3c6df2cc2d6dfec73cbf.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/8444bcf73727d24d7ca59a1fa130fc89e1033f2977af3c6df2cc2d6dfec73cbf.jpg)

![a15424d363ee96a36b86a95354376488cf3d295d757e8c98a320e040e1c6a409.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/a15424d363ee96a36b86a95354376488cf3d295d757e8c98a320e040e1c6a409.jpg)

![ccad2fd8164409bf0f2d55585097f5a54baac0677772484a6dad470a1b045a23.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/ccad2fd8164409bf0f2d55585097f5a54baac0677772484a6dad470a1b045a23.jpg)

![d60641ee6e1c67faaefcf06862692d4282ae41a19185b839b1989b1b6c30d111.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/d60641ee6e1c67faaefcf06862692d4282ae41a19185b839b1989b1b6c30d111.jpg)

![de51a8417e7a811de1e83dacef81fbf07cb7ad8c73094dd1285fb6a82051acb0.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/de51a8417e7a811de1e83dacef81fbf07cb7ad8c73094dd1285fb6a82051acb0.jpg)

![ec21bdcfe919452338cfcb025dcfe3eddd050556c31d0875ef86d662c6a5ca94.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/ec21bdcfe919452338cfcb025dcfe3eddd050556c31d0875ef86d662c6a5ca94.jpg)

![eda74c7be1111e50b855040922c66ac22896875691e1598815c6d760da1cdf8a.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/eda74c7be1111e50b855040922c66ac22896875691e1598815c6d760da1cdf8a.jpg)

![ee013307a34a37b8a26d97ab95d26f0d4b780914dbf98532eb854e08ba745d7f.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/ee013307a34a37b8a26d97ab95d26f0d4b780914dbf98532eb854e08ba745d7f.jpg)

![f84f0c83bb281c8ecfdc6c38cf0b5ad4a0bff12f9465b610b53ab191d940bfa7.jpg](../icml_results/2306_DetailMaster_%20Can%20Your%20Text-to-Image%20Model%20Handle%20Long%20Prompts_/tables/f84f0c83bb281c8ecfdc6c38cf0b5ad4a0bff12f9465b610b53ab191d940bfa7.jpg)
