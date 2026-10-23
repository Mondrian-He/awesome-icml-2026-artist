# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 138 of 197

## 目录 (Table of Contents)

1. [MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo](#paper-4522)
2. [Coupled Cluster con MoLe: Molecular Orbital Learning for Neural Wavefunctions](#paper-4523)
3. [Effects of Structural Reward Shaping on Biophysical Properties in RL-Trained Plasmid Generators](#paper-4524)
4. [Calibrated Preference Learning: The Case of Label Ranking](#paper-4525)
5. [CodeTaste: Can LLMs Generate Human-Level Code Refactorings?](#paper-4526)
6. [LLMs Lean on Priors, Not Programming Language Semantics](#paper-4527)
7. [Advancing SVD-based LLM Compression via Layer-Wise Error Model Search](#paper-4528)
8. [Deep Incentive Design with Differentiable Equilibrium Blocks](#paper-4529)
9. [Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations](#paper-4530)
10. [A Tight Theory of Error Feedback Algorithms in Distributed Optimization](#paper-4531)
11. [Connecting Independently Trained Modes via Layer-Wise Connectivity](#paper-4532)
12. [Bridging the Gap Between Average and Discounted TD Learning](#paper-4533)
13. [Beyond Benchmarks: Toward Causally Faithful Evaluation of Large Language Models](#paper-4534)
14. [AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](#paper-4535)
15. [QPKO: Differentiable QP-Embedded Deep Koopman Framework for Modeling Nonlinear Systems](#paper-4536)
16. [RGMem: Renormalization Group–inspired Memory Evolution for Language Agents](#paper-4537)
17. [Learning Discriminative and Generalizable Anomaly Detector for Dynamic Graph with Limited Supervision](#paper-4538)
18. [A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification](#paper-4539)
19. [Neutral-Reference Prompting for Vision–Language Models](#paper-4540)
20. [From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers](#paper-4541)
21. [Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](#paper-4542)
22. [LogicSAGE: Neuro-Symbolic Reasoning with Socratic-Guided Enhancement](#paper-4543)
23. [LOTTERY: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry](#paper-4544)
24. [Routing and Reasoned Evaluation with Large Language Models](#paper-4545)
25. [Joint Model and Data Sparsification via the Marginal Likelihood](#paper-4546)
26. [Content-Style Identification via Differential Independence](#paper-4547)
27. [SVL: Goal-Conditioned Reinforcement Learning as Survival Learning](#paper-4548)
28. [Semi-Supervised Learning for Molecular Graphs via Ensemble Consensus](#paper-4549)
29. [Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning](#paper-4550)
30. [Reward Under Attack: Analyzing the Robustness and Hackability of Process Reward Models](#paper-4551)
31. [MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning](#paper-4552)
32. [Reasoning about Reasoning: BAPO Bounds on Chain-of-Thought Token Complexity in LLMs](#paper-4553)
33. [Task-Awareness Improves LLM Generations and Uncertainty](#paper-4554)

---


<a id="paper-4522"></a>

## MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo

### Images

![40e033bbccaa6f64f2f24ce839c0103696483c165bf73c4f2f2dc00b8f9e6d4e.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/40e033bbccaa6f64f2f24ce839c0103696483c165bf73c4f2f2dc00b8f9e6d4e.jpg)

![50d6cdccb170a5cd5d0e5370bc6d6a734d3e3234b81cb4402138671f0eae9dcb.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/50d6cdccb170a5cd5d0e5370bc6d6a734d3e3234b81cb4402138671f0eae9dcb.jpg)

![8802c1e0584606cad64d0292003a34fece19f8711761ee4cf539b8f27136037d.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/8802c1e0584606cad64d0292003a34fece19f8711761ee4cf539b8f27136037d.jpg)

![88558a37194675e3df9eac47a646bd51b97a7c8a96fc1b6d9e1b9504046abb24.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/88558a37194675e3df9eac47a646bd51b97a7c8a96fc1b6d9e1b9504046abb24.jpg)

![8be590543eab2ddea3b826397c1d3bd77e424cae0614987d3a8c8c5e03c61695.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/8be590543eab2ddea3b826397c1d3bd77e424cae0614987d3a8c8c5e03c61695.jpg)

![c3688829ae96fd69b0f0aa51d82548f893d6f2fc56072950e8227083f4bf35fd.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/c3688829ae96fd69b0f0aa51d82548f893d6f2fc56072950e8227083f4bf35fd.jpg)

![f3767b4b4af2ca65529d2733677c193673b1296d3c7de2d9a1420e20016e44e8.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/f3767b4b4af2ca65529d2733677c193673b1296d3c7de2d9a1420e20016e44e8.jpg)

![f383cb1f5da0e273a1c4eaeb8bf1ce116c0a7f38c17369b0317c6cf5590a2b4f.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/images/f383cb1f5da0e273a1c4eaeb8bf1ce116c0a7f38c17369b0317c6cf5590a2b4f.jpg)

### Tables

![25f8015a773aa71e790db7c815ed6faa116842c4a7d8dd038565b055ec641229.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/25f8015a773aa71e790db7c815ed6faa116842c4a7d8dd038565b055ec641229.jpg)

![2a4adf84a2b0ff422c72c8fab436db5c476af167f0fd6e44da7e480ea2e0f79f.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/2a4adf84a2b0ff422c72c8fab436db5c476af167f0fd6e44da7e480ea2e0f79f.jpg)

![5d6033157919564143ce8bb3655b9b5d0543f7478e714f74ba40ff1076810cf7.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/5d6033157919564143ce8bb3655b9b5d0543f7478e714f74ba40ff1076810cf7.jpg)

![9663a0c40e80a8f4b044f7565e992bb2ca94b743962fd19a24bbe84783f5acad.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/9663a0c40e80a8f4b044f7565e992bb2ca94b743962fd19a24bbe84783f5acad.jpg)

![a331629f770b80378b8a3c3f9ca8236abb1bc2fadbd27dfce833cad291d0f9b4.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/a331629f770b80378b8a3c3f9ca8236abb1bc2fadbd27dfce833cad291d0f9b4.jpg)

![d1eb296622018c41bd0579bc1ca6e6c15fff32be88c0c59d54f6c14646f5e27a.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/d1eb296622018c41bd0579bc1ca6e6c15fff32be88c0c59d54f6c14646f5e27a.jpg)

![e419728b838e1e3e8b4a35f1e339a72f13a05442e4b5e235024045cc325d2b44.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/e419728b838e1e3e8b4a35f1e339a72f13a05442e4b5e235024045cc325d2b44.jpg)

![fb7dc1f237ac5f756e7741044e42b4f8ae3a85163390773cdda730bdf57320f1.jpg](../icml_results/4568_Generalized%20Schr%C3%B6dinger%20Bridge%20on%20Graphs/tables/fb7dc1f237ac5f756e7741044e42b4f8ae3a85163390773cdda730bdf57320f1.jpg)

## MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo


### Images

![0bc3c0ae1bdd7789409c516c5ef62d10730be55a1ddb9adb838e6babe1bf1731.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/0bc3c0ae1bdd7789409c516c5ef62d10730be55a1ddb9adb838e6babe1bf1731.jpg)

![12141dc83f6ffd50cc539f693204db39615239fcf0632668b5752dd716174d28.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/12141dc83f6ffd50cc539f693204db39615239fcf0632668b5752dd716174d28.jpg)

![14d776431e157bd92e385f5025e6f9fef2dfed473bda49c51742a977142fcada.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/14d776431e157bd92e385f5025e6f9fef2dfed473bda49c51742a977142fcada.jpg)

![27a4db1a510d107699e4f322939149937cfa197394de1da1516911f551ea9523.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/27a4db1a510d107699e4f322939149937cfa197394de1da1516911f551ea9523.jpg)

![35a42b853b575c1fb2f3e306132c44492931a3218be0d217771ae2ea5b3b606f.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/35a42b853b575c1fb2f3e306132c44492931a3218be0d217771ae2ea5b3b606f.jpg)

![36b1e3432c7a98f1bc234861466036720e4f1dfa179179fdf261f6ee59c84585.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/36b1e3432c7a98f1bc234861466036720e4f1dfa179179fdf261f6ee59c84585.jpg)

![3bcfd0161c412c23a6d85e3e29e54ebce5496398ba6734b66c151cd3f1671c97.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/3bcfd0161c412c23a6d85e3e29e54ebce5496398ba6734b66c151cd3f1671c97.jpg)

![3d053fdb11d268a5c74f3636c71c0a9198d870b61c893709ceedd71839b3186d.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/3d053fdb11d268a5c74f3636c71c0a9198d870b61c893709ceedd71839b3186d.jpg)

![3f9dc314a55b93f097372c7d34b857a6c556a1fe4811dc2fb836de18f3db3d1f.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/3f9dc314a55b93f097372c7d34b857a6c556a1fe4811dc2fb836de18f3db3d1f.jpg)

![487b4fda625168e95f4a7e0c161a6d0299474d800e659a424cacaa881227cc06.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/487b4fda625168e95f4a7e0c161a6d0299474d800e659a424cacaa881227cc06.jpg)

![50349886221e382fdb68e506c1e16905ad582758cb60344a97757c4b2b966647.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/50349886221e382fdb68e506c1e16905ad582758cb60344a97757c4b2b966647.jpg)

![52f70ece25e4b7d092f9abbd8de42c0e440d77cc9b8f53dc5c368ca4be3ba8d5.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/52f70ece25e4b7d092f9abbd8de42c0e440d77cc9b8f53dc5c368ca4be3ba8d5.jpg)

![70c48d8d7fe19d003ba4783add29e1499e629c029b237326a53a3dbd9c6922f2.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/70c48d8d7fe19d003ba4783add29e1499e629c029b237326a53a3dbd9c6922f2.jpg)

![74e273bc257ce04e602fc2e21213e37d74fdbc8366d9958db94713652e99c377.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/74e273bc257ce04e602fc2e21213e37d74fdbc8366d9958db94713652e99c377.jpg)

![754a06d3ac40f22ec59e4c16b83ca612bed81490d7f82d64490a55b84c93f071.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/754a06d3ac40f22ec59e4c16b83ca612bed81490d7f82d64490a55b84c93f071.jpg)

![7a46c45e132df8e44b7ae79bf532c92719503520aec58429f44ef742a43a2277.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/7a46c45e132df8e44b7ae79bf532c92719503520aec58429f44ef742a43a2277.jpg)

![7cada7076fd9ca5b7e510ba29a36f1b7a448d4b9821ae7b97e041dd0d2695da1.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/7cada7076fd9ca5b7e510ba29a36f1b7a448d4b9821ae7b97e041dd0d2695da1.jpg)

![8d822ebff16e9618260c8dcb74362397be6bda93849ce2b2725ca9facdeca53a.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/8d822ebff16e9618260c8dcb74362397be6bda93849ce2b2725ca9facdeca53a.jpg)

![9796d278f084c561a5ca8ffa321ca0f860dbef219ff51222dc02e3dd9db85434.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/9796d278f084c561a5ca8ffa321ca0f860dbef219ff51222dc02e3dd9db85434.jpg)

![a22c62a567cd04f9a7047333e37930d848c71034c9615719a31d9e3e474dc201.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/a22c62a567cd04f9a7047333e37930d848c71034c9615719a31d9e3e474dc201.jpg)

![a578f8b6717f61fdaecb54aaf9292ac5a6e25b08617558115e3cc43549099e57.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/a578f8b6717f61fdaecb54aaf9292ac5a6e25b08617558115e3cc43549099e57.jpg)

![b884f48348a5725d8675f8571cf6ab0f32a1951983a7d575e9fcef8d93a5c759.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/b884f48348a5725d8675f8571cf6ab0f32a1951983a7d575e9fcef8d93a5c759.jpg)

![bb58079b0ea5cc4d0d95058cafef918b9a9777334f4af9b4bda6807a621560b3.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/bb58079b0ea5cc4d0d95058cafef918b9a9777334f4af9b4bda6807a621560b3.jpg)

![c55e770a082ab50c94096aa6ad644c1e38b5bd7ef09a3db954c317962af4bdb3.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/c55e770a082ab50c94096aa6ad644c1e38b5bd7ef09a3db954c317962af4bdb3.jpg)

![cad9294558cd8cfcf359cedef7928ea3b1b45dbc8d7e8db36c86c3f17a79b48a.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/cad9294558cd8cfcf359cedef7928ea3b1b45dbc8d7e8db36c86c3f17a79b48a.jpg)

![ce0729de0f23bdf0ae2a3bebacdf16de75faa3c61830473c90a7fe4b34ff0e5f.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/ce0729de0f23bdf0ae2a3bebacdf16de75faa3c61830473c90a7fe4b34ff0e5f.jpg)

![e137972fa5c45d1d8556b924dcbbc70a15ab0e6be46546ee6755fc39efea528f.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/e137972fa5c45d1d8556b924dcbbc70a15ab0e6be46546ee6755fc39efea528f.jpg)

![e98143b4c0142a11c4707aa8c80a38a726759ccc4c4f027cfc74c8ab69df71a6.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/images/e98143b4c0142a11c4707aa8c80a38a726759ccc4c4f027cfc74c8ab69df71a6.jpg)

### Tables

![05316960203cfacffc3ded4cad9590265863b98d5558e6a26515f25fc9e1ba8b.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/05316960203cfacffc3ded4cad9590265863b98d5558e6a26515f25fc9e1ba8b.jpg)

![1532b5cb92478dec964ac69017e01b5170055d91bd4741ff89c94132a1ecfb59.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/1532b5cb92478dec964ac69017e01b5170055d91bd4741ff89c94132a1ecfb59.jpg)

![21a92d6d15694874d5cda7f10c3e0ae85fe111103f690e7dcd442e7f46ff92be.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/21a92d6d15694874d5cda7f10c3e0ae85fe111103f690e7dcd442e7f46ff92be.jpg)

![321555e9d7a76143c3dd6b50282863c628a455f42b8ebe0a90aa740a5649e042.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/321555e9d7a76143c3dd6b50282863c628a455f42b8ebe0a90aa740a5649e042.jpg)

![3511ec8ec9839b7fbb45b9aee672d50c2410f04ea0dad74a3131cccaa2307811.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/3511ec8ec9839b7fbb45b9aee672d50c2410f04ea0dad74a3131cccaa2307811.jpg)

![4b48d1c9e02e2162e034c6870eb36e761dbe4d401f043e81c4cb145c381efe80.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/4b48d1c9e02e2162e034c6870eb36e761dbe4d401f043e81c4cb145c381efe80.jpg)

![4efbb67058e748dfd8b38f9f9c01e2b60f61c0620a35a18901279582f0c4cac8.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/4efbb67058e748dfd8b38f9f9c01e2b60f61c0620a35a18901279582f0c4cac8.jpg)

![64fdd1d72d3e5ef1d416f3ed17650b60c359bbdecfd2b3918fa2f991291de65c.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/64fdd1d72d3e5ef1d416f3ed17650b60c359bbdecfd2b3918fa2f991291de65c.jpg)

![71a203ed2adf613cf5dbd4baf8e023509b952eea1a3daa678412113d840c3a36.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/71a203ed2adf613cf5dbd4baf8e023509b952eea1a3daa678412113d840c3a36.jpg)

![857131a26343b4ff839dac9ac6cacecdd89ba8bfa69ee9befe9e57f4f8d8afa5.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/857131a26343b4ff839dac9ac6cacecdd89ba8bfa69ee9befe9e57f4f8d8afa5.jpg)

![8c8d9dde1adc62c377f9abbd0f487d836523ee05d57a0fba830b0ca4c817728e.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/8c8d9dde1adc62c377f9abbd0f487d836523ee05d57a0fba830b0ca4c817728e.jpg)

![ceb866e9e6f57a43e9401564058846f88cb408ae5d74fa07a655ec324a55b401.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/ceb866e9e6f57a43e9401564058846f88cb408ae5d74fa07a655ec324a55b401.jpg)

![df5cd5cf26766f3264e776d83fc1c9f6ab2fc2746b8051f7cf63bfae7a7e9f05.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/df5cd5cf26766f3264e776d83fc1c9f6ab2fc2746b8051f7cf63bfae7a7e9f05.jpg)

![e0d14968a11abdd20abad42a5012efd1ece6cdbe16a7035bfc7fac16f3b806b8.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/e0d14968a11abdd20abad42a5012efd1ece6cdbe16a7035bfc7fac16f3b806b8.jpg)

![e528d380c97da634ab7a353c9ccc89f8ee8e4c1139f3f3707897ef7c68d1dfd7.jpg](../icml_results/4569_MuLoCo_%20Muon%20is%20a%20Practical%20Inner%20Optimizer%20for%20DiLoCo/tables/e528d380c97da634ab7a353c9ccc89f8ee8e4c1139f3f3707897ef7c68d1dfd7.jpg)

<a id="paper-4523"></a>

## Coupled Cluster con MoLe: Molecular Orbital Learning for Neural Wavefunctions


### Images

![3f0144870f2e17e165279cf10648423c17c2d90a282bfb1689fc0c56e8bbeb5b.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/3f0144870f2e17e165279cf10648423c17c2d90a282bfb1689fc0c56e8bbeb5b.jpg)

![4cd022d49eb3fe15079619725f0a7283c4e0ce9ee64c828944fae77d1fb93bff.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/4cd022d49eb3fe15079619725f0a7283c4e0ce9ee64c828944fae77d1fb93bff.jpg)

![bae5745c1fb8b4d59e41d07da344e5a12d18a1dceeb137dc9d6f378c5e3193b2.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/bae5745c1fb8b4d59e41d07da344e5a12d18a1dceeb137dc9d6f378c5e3193b2.jpg)

![ce1e280d4dfb8adb44a1a9b346a79caa39dbe64683887d096e49e7027450df22.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/ce1e280d4dfb8adb44a1a9b346a79caa39dbe64683887d096e49e7027450df22.jpg)

![d153839314fe2fdfc40f2d6f85929e48bb743ea4e391b134ca4fdb8cca84591e.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/d153839314fe2fdfc40f2d6f85929e48bb743ea4e391b134ca4fdb8cca84591e.jpg)

![d6d41a27b6f05629e711edd435cf9e5a44d2f80ed31f71e2058e91f927659212.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/d6d41a27b6f05629e711edd435cf9e5a44d2f80ed31f71e2058e91f927659212.jpg)

![da7f5d979ec7cc389fc8a84c7e92b78ddfafd81648a5c1f8c52fcfcd126a1a21.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/da7f5d979ec7cc389fc8a84c7e92b78ddfafd81648a5c1f8c52fcfcd126a1a21.jpg)

![ea5d2411bc312b3603d164ae4103b448ef3fa732e1945ae9612c458458c6bfa9.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/ea5d2411bc312b3603d164ae4103b448ef3fa732e1945ae9612c458458c6bfa9.jpg)

![eeb6daea9afff99fe808faa8da0acb9d478c111d8dbee4eb1dcb801875eeb681.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/images/eeb6daea9afff99fe808faa8da0acb9d478c111d8dbee4eb1dcb801875eeb681.jpg)

### Tables

![10e81d7c490d1a0548bb9815d4afda5df265b76b1aae336aebd7100c6eb1b900.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/10e81d7c490d1a0548bb9815d4afda5df265b76b1aae336aebd7100c6eb1b900.jpg)

![6478a3e6ca74308136822cf403919e2fd13353089537b9fd3f2b6944ae0a70bb.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/6478a3e6ca74308136822cf403919e2fd13353089537b9fd3f2b6944ae0a70bb.jpg)

![7f4afa609b12c5f3fcfa0d2b9efedb5f3dd2b3f568f9232ee6156bb0e0c27185.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/7f4afa609b12c5f3fcfa0d2b9efedb5f3dd2b3f568f9232ee6156bb0e0c27185.jpg)

![8f44e734a7e2d9fd369a6a3406a3ed22e4b1dbd9d033eb7c4989e9533b6e3ad5.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/8f44e734a7e2d9fd369a6a3406a3ed22e4b1dbd9d033eb7c4989e9533b6e3ad5.jpg)

![b15166f04e891f6b953ed4dc6d70a232b394f3bc3de52c4e73e497b96b3f36b1.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/b15166f04e891f6b953ed4dc6d70a232b394f3bc3de52c4e73e497b96b3f36b1.jpg)

![bd6792cce31f19922e43ffb6a78827c955c283b139d35de926ff81ad7fe48ed2.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/bd6792cce31f19922e43ffb6a78827c955c283b139d35de926ff81ad7fe48ed2.jpg)

![c96c928cc382b8e8654d287696f261150fef824e35da944e51952cd23198047a.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/c96c928cc382b8e8654d287696f261150fef824e35da944e51952cd23198047a.jpg)

![e534ba0c661f004440bd5e5ccb6ad3cdd9b0cc89e5520a64f2db0c1483cf86fe.jpg](../icml_results/4570_Coupled%20Cluster%20con%20MoLe_%20Molecular%20Orbital%20Learning%20for%20Neural%20Wavefunctions/tables/e534ba0c661f004440bd5e5ccb6ad3cdd9b0cc89e5520a64f2db0c1483cf86fe.jpg)

<a id="paper-4524"></a>

## Effects of Structural Reward Shaping on Biophysical Properties in RL-Trained Plasmid Generators


### Images

![12ba3681a8244b6367bb872a058568e37324ca18aaed9d2661c80847b1711ba1.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/images/12ba3681a8244b6367bb872a058568e37324ca18aaed9d2661c80847b1711ba1.jpg)

![30a458e5bf8d2ed28e81898d064e663f983c2e8ae9d717a959e2bfbe68831b47.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/images/30a458e5bf8d2ed28e81898d064e663f983c2e8ae9d717a959e2bfbe68831b47.jpg)

![bea0cb06b88456eaf8c117b553f491ce9e6c01944f08d7680276c0ad2935eac9.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/images/bea0cb06b88456eaf8c117b553f491ce9e6c01944f08d7680276c0ad2935eac9.jpg)

![ea8b2fea5522e807c73aaf39aa425a7e46503e889a0071802c36108bce2ce360.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/images/ea8b2fea5522e807c73aaf39aa425a7e46503e889a0071802c36108bce2ce360.jpg)

![efb2c2723f1f0ea69e14248bd4f0b954d7b3c52bfa31ec2415befa1469937d86.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/images/efb2c2723f1f0ea69e14248bd4f0b954d7b3c52bfa31ec2415befa1469937d86.jpg)

### Tables

![085345f21fb3073ae9661b8d4905af255a7e8f08e021803257e2feff5e8b93df.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/085345f21fb3073ae9661b8d4905af255a7e8f08e021803257e2feff5e8b93df.jpg)

![15520659b2edad9b30a9d4250f22d8e0ec210f9d25b73b292c14764b06758fdb.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/15520659b2edad9b30a9d4250f22d8e0ec210f9d25b73b292c14764b06758fdb.jpg)

![5bfc59b5fe88538d48ce83faadaf3f8579d23b8ee437f4e952f00e44ce0b0279.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/5bfc59b5fe88538d48ce83faadaf3f8579d23b8ee437f4e952f00e44ce0b0279.jpg)

![5f4bf633907fae8ef2b1050c903ec53a222b8d7365ff5fb81b9f25a9ee65ca97.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/5f4bf633907fae8ef2b1050c903ec53a222b8d7365ff5fb81b9f25a9ee65ca97.jpg)

![70c1d61774107b38f56ca4727597df27511f2d4677fa126da66a0c6afaa24dd1.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/70c1d61774107b38f56ca4727597df27511f2d4677fa126da66a0c6afaa24dd1.jpg)

![7348928dccdc70d648cb5c4ddf1f7c8a6744d9d713e4edff0cddee4ba90306c9.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/7348928dccdc70d648cb5c4ddf1f7c8a6744d9d713e4edff0cddee4ba90306c9.jpg)

![7f4eb4235c206157f6b3b810de410d3eaae9406e0d26f62b2aecba51f291b240.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/7f4eb4235c206157f6b3b810de410d3eaae9406e0d26f62b2aecba51f291b240.jpg)

![b790a03745502a3a47dc2eba167362b7ad1efae21f2caf4439242cd3052131da.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/b790a03745502a3a47dc2eba167362b7ad1efae21f2caf4439242cd3052131da.jpg)

![fb63850b88c3ad0f6166c3d84fcf095aee637307efd168c8a118e5990d14d2cd.jpg](../icml_results/4571_Effects%20of%20Structural%20Reward%20Shaping%20on%20Biophysical%20Properties%20in%20RL-Trained%20Plasmid%20Generators/tables/fb63850b88c3ad0f6166c3d84fcf095aee637307efd168c8a118e5990d14d2cd.jpg)

<a id="paper-4525"></a>

## Calibrated Preference Learning: The Case of Label Ranking


### Images

![0190568a73816dab0a16159da2689f66ff5ae964d4e5be66fb55e8774c1089fa.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/0190568a73816dab0a16159da2689f66ff5ae964d4e5be66fb55e8774c1089fa.jpg)

![0ced62eb5fa99f27c4d7fbfffaf741a15ba32b1b0a14ac494ac21e300f332b2b.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/0ced62eb5fa99f27c4d7fbfffaf741a15ba32b1b0a14ac494ac21e300f332b2b.jpg)

![0eba0f0c4fcdc4f2f72554091df414e93849e392c54c276621c3b189ffe3b0d1.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/0eba0f0c4fcdc4f2f72554091df414e93849e392c54c276621c3b189ffe3b0d1.jpg)

![1a02b84596e744d07c5bc59c34ae76d6222002e1b0aca561913f2cc9bcfdb37e.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/1a02b84596e744d07c5bc59c34ae76d6222002e1b0aca561913f2cc9bcfdb37e.jpg)

![29c03226df08b9b9e46509fb4940ec134ce22630234e17ed543b86bba124b9bc.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/29c03226df08b9b9e46509fb4940ec134ce22630234e17ed543b86bba124b9bc.jpg)

![33fe32616868d7cffae98526abfc0a8db6da96e5974d5b4d27550ff75baa264b.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/33fe32616868d7cffae98526abfc0a8db6da96e5974d5b4d27550ff75baa264b.jpg)

![535d8326282c6e0638cf0b905535ed8cc0986a3e01dee16bd9dc3c955276ce01.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/535d8326282c6e0638cf0b905535ed8cc0986a3e01dee16bd9dc3c955276ce01.jpg)

![558d5f26e40f50387ed083028c19ebb62e4c7153e5b75c572906894025b0c031.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/558d5f26e40f50387ed083028c19ebb62e4c7153e5b75c572906894025b0c031.jpg)

![5bf916a55bd83258a0ce4da967e1239ee910e35ef13084a8f9f1f6dadf246da6.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/5bf916a55bd83258a0ce4da967e1239ee910e35ef13084a8f9f1f6dadf246da6.jpg)

![64acd77ff77f9c7eaba4dd57e9882999930f1e694e0ad02e12bc566b05acf139.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/64acd77ff77f9c7eaba4dd57e9882999930f1e694e0ad02e12bc566b05acf139.jpg)

![65c57a27d1716bef300a488dd1fab745f2010b2081e3cf89cc113f89e4dcf466.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/65c57a27d1716bef300a488dd1fab745f2010b2081e3cf89cc113f89e4dcf466.jpg)

![6fd0c6d96a99e1c0f85f2aab35300f1954057df077c875d7303d41dbbfd1ba74.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/6fd0c6d96a99e1c0f85f2aab35300f1954057df077c875d7303d41dbbfd1ba74.jpg)

![a61bf1b3ee5e44e60dd2ad4e2cf13163003733c9636b28a6735b078de2455ba7.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/a61bf1b3ee5e44e60dd2ad4e2cf13163003733c9636b28a6735b078de2455ba7.jpg)

![a7a8a14aac0d1c472b9971b639ef90d5142fd7fa7dc6f286957e947fa761ae44.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/a7a8a14aac0d1c472b9971b639ef90d5142fd7fa7dc6f286957e947fa761ae44.jpg)

![d3feb70b2388b632c432d0bff850ee5dff45af61defabb7bf4be46a784445d38.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/d3feb70b2388b632c432d0bff850ee5dff45af61defabb7bf4be46a784445d38.jpg)

![e1b32fd971d5d7ddc101ed126a25f1908cb97bcb70ce79295659c69af0231481.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/images/e1b32fd971d5d7ddc101ed126a25f1908cb97bcb70ce79295659c69af0231481.jpg)

### Tables

![14fba34dca664e7cdef7cc19163a5cef5350b76ad5a0911c32d787f5279da6d1.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/tables/14fba34dca664e7cdef7cc19163a5cef5350b76ad5a0911c32d787f5279da6d1.jpg)

![180d5b03fa35ccd84b7ed0f07ffa4b8f158e291cc143f992dbe2263095f0304a.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/tables/180d5b03fa35ccd84b7ed0f07ffa4b8f158e291cc143f992dbe2263095f0304a.jpg)

![6084f50c9187700db50ef6dedae67f8184b4078679fa73d3ec0a15d451560194.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/tables/6084f50c9187700db50ef6dedae67f8184b4078679fa73d3ec0a15d451560194.jpg)

![fdf24f65405f28a566fb99c89e27433d7576f888c42dc41a0f0a28f06fd11ac5.jpg](../icml_results/4572_Calibrated%20Preference%20Learning_%20The%20Case%20of%20Label%20Ranking/tables/fdf24f65405f28a566fb99c89e27433d7576f888c42dc41a0f0a28f06fd11ac5.jpg)

<a id="paper-4526"></a>

## CodeTaste: Can LLMs Generate Human-Level Code Refactorings?


### Images

![14b66d7cdc0851b8952333ee1bc5ad42e89b4b99f015d8a1f83a1844bd672558.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/14b66d7cdc0851b8952333ee1bc5ad42e89b4b99f015d8a1f83a1844bd672558.jpg)

![5244569be1c266109aa144806b59674e0949286b0c0d553ab1348d5935cb5058.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/5244569be1c266109aa144806b59674e0949286b0c0d553ab1348d5935cb5058.jpg)

![70c663b3f5c82ec1dfed0c5f7527c755f4d377761d74edd8b99b5a4e93334119.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/70c663b3f5c82ec1dfed0c5f7527c755f4d377761d74edd8b99b5a4e93334119.jpg)

![79123c9f4ec885644e0752c2e7afdced2af832ee5213b5309505673e80e3fc9b.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/79123c9f4ec885644e0752c2e7afdced2af832ee5213b5309505673e80e3fc9b.jpg)

![991d5f7614b4cb7d3715bbcd303f1c064d86a8f3479838234c3584ccf01bcded.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/991d5f7614b4cb7d3715bbcd303f1c064d86a8f3479838234c3584ccf01bcded.jpg)

![a4ca725e2d73b516b9f701915707a761c4c090320d86a735a095084eca90cf3e.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/a4ca725e2d73b516b9f701915707a761c4c090320d86a735a095084eca90cf3e.jpg)

![a4d901c7cd0faff1cbcd090849c0801455cb170c86746edb3819dddd20e524e8.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/a4d901c7cd0faff1cbcd090849c0801455cb170c86746edb3819dddd20e524e8.jpg)

![ceb4bf20afa3c760b33b819ef1c210c064b52fcad71f33e4e87035c1c6ba98f2.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/ceb4bf20afa3c760b33b819ef1c210c064b52fcad71f33e4e87035c1c6ba98f2.jpg)

![e22b983879192e301e48dc6f7a3323d1a0e0d60deff842795d75a6bfbe5f72e4.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/images/e22b983879192e301e48dc6f7a3323d1a0e0d60deff842795d75a6bfbe5f72e4.jpg)

### Tables

![0576b042d8c58fa511d715e4a6d06ad17686d634b423e990fee0bee5eb08ea15.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/0576b042d8c58fa511d715e4a6d06ad17686d634b423e990fee0bee5eb08ea15.jpg)

![150d7df5ec1a682b4610af8d69ef27b51eac71a836fedec0f52175926819eee6.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/150d7df5ec1a682b4610af8d69ef27b51eac71a836fedec0f52175926819eee6.jpg)

![4e08dec5252ae9b79000ac34aabb5f688b8c75030c22636e7e78b1dc6885a3d0.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/4e08dec5252ae9b79000ac34aabb5f688b8c75030c22636e7e78b1dc6885a3d0.jpg)

![5df32fd015c8704823a435e8d2b3d4a4786be28426b57124ba4ca29992f9c98d.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/5df32fd015c8704823a435e8d2b3d4a4786be28426b57124ba4ca29992f9c98d.jpg)

![814289a8bf8027c0980a2a4c834edcc550f041869b2388014d1002fdc08ef07f.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/814289a8bf8027c0980a2a4c834edcc550f041869b2388014d1002fdc08ef07f.jpg)

![abf6f7a7b9e1567a166e0075d88660770aa0eb6702bc60b9965802af32e67f8f.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/abf6f7a7b9e1567a166e0075d88660770aa0eb6702bc60b9965802af32e67f8f.jpg)

![c46b3fbccec8d21ca74e64e83b0590cc622843b28d6eb95bff3efec455350578.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/c46b3fbccec8d21ca74e64e83b0590cc622843b28d6eb95bff3efec455350578.jpg)

![e1814221370e081cabd12a3bec7c50ca490b738941daca4b3a94d7022c61ac83.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/e1814221370e081cabd12a3bec7c50ca490b738941daca4b3a94d7022c61ac83.jpg)

![e61f3cc6abd640753e1fcac1a8e4fb38072a209905fa6a7da072c8a68dc886fa.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/e61f3cc6abd640753e1fcac1a8e4fb38072a209905fa6a7da072c8a68dc886fa.jpg)

![fc500e40140b21ba7bcf5a5dc5bce0ab884c5cb6249b283c9be2ed2f5cac5d17.jpg](../icml_results/4573_CodeTaste_%20Can%20LLMs%20Generate%20Human-Level%20Code%20Refactorings_/tables/fc500e40140b21ba7bcf5a5dc5bce0ab884c5cb6249b283c9be2ed2f5cac5d17.jpg)

<a id="paper-4527"></a>

## LLMs Lean on Priors, Not Programming Language Semantics


### Images

![124038c21490433ba2554479ef291978af3461eb82b6cecb488541b580ccf37a.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/124038c21490433ba2554479ef291978af3461eb82b6cecb488541b580ccf37a.jpg)

![1b56ccd460ddeabe9f2b901bbc22046df74ff29c8188c98783f549b06702e3fa.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/1b56ccd460ddeabe9f2b901bbc22046df74ff29c8188c98783f549b06702e3fa.jpg)

![2cd76910ecf918fec9f9a56e0f287d664bd73fdfe05a9a1e78b612d58d83349c.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/2cd76910ecf918fec9f9a56e0f287d664bd73fdfe05a9a1e78b612d58d83349c.jpg)

![44e33c63586c06757dcbc66b63b6adb383838424df744c6f3ac2211b9829e69f.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/44e33c63586c06757dcbc66b63b6adb383838424df744c6f3ac2211b9829e69f.jpg)

![7e3f9f24baae1c05da2289d71e131fd3c54174db1f315c9bb18f2b2b904d2879.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/7e3f9f24baae1c05da2289d71e131fd3c54174db1f315c9bb18f2b2b904d2879.jpg)

![9faaa07a9b32856d1036945d7c14f99d42396da04160b0f2fab357cf520473c1.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/9faaa07a9b32856d1036945d7c14f99d42396da04160b0f2fab357cf520473c1.jpg)

![b3c3f706ebabac84878b927082092f327815450588139b631174ecc8a4f1f2cf.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/b3c3f706ebabac84878b927082092f327815450588139b631174ecc8a4f1f2cf.jpg)

![f5e9220efc2320bb70a6056cc80d9512ef4eaaab76db64e755d163bb1bf0244e.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/f5e9220efc2320bb70a6056cc80d9512ef4eaaab76db64e755d163bb1bf0244e.jpg)

![f7c5afcd7fd28e7a063a5b31a76968beecc83473515bbf3162961cb7245137de.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/f7c5afcd7fd28e7a063a5b31a76968beecc83473515bbf3162961cb7245137de.jpg)

![fe3fafcc029a94481f9c724e452317d1e78e5b9d87dba169745f99e5955fffcd.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/images/fe3fafcc029a94481f9c724e452317d1e78e5b9d87dba169745f99e5955fffcd.jpg)

### Tables

![0dbfd9ec907765b5207d65887f18f5d1fdc32bfab6fbf87ff975dc944a20b296.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/0dbfd9ec907765b5207d65887f18f5d1fdc32bfab6fbf87ff975dc944a20b296.jpg)

![29079a5f4f946d1cfbe92059d7c2b3f1806ca6c2c8c693532cd44a01ca02d411.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/29079a5f4f946d1cfbe92059d7c2b3f1806ca6c2c8c693532cd44a01ca02d411.jpg)

![2beb7d522dcf3cb1573a919c8eb3ea1974b7c6379ca37d6afa10612853071534.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/2beb7d522dcf3cb1573a919c8eb3ea1974b7c6379ca37d6afa10612853071534.jpg)

![2cbd675cfa39100393f96ba3b123cf24c084e09e39a015b508402aeed2c15f14.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/2cbd675cfa39100393f96ba3b123cf24c084e09e39a015b508402aeed2c15f14.jpg)

![36e385e641add7b086cce4e726eb4138d82d839ac86d85746c873dbf03286f42.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/36e385e641add7b086cce4e726eb4138d82d839ac86d85746c873dbf03286f42.jpg)

![46d7fd81b4c7a705a6b1d0805336a54871ebd60553457dee4a365927cc3678e5.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/46d7fd81b4c7a705a6b1d0805336a54871ebd60553457dee4a365927cc3678e5.jpg)

![4a286e6c7fff59f4294268322b4bec30fda09eac37d6971cec130f6995672e3b.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/4a286e6c7fff59f4294268322b4bec30fda09eac37d6971cec130f6995672e3b.jpg)

![57f84cb36f0333c5341f85e7fe7912f06b5cdc7b3c867fc87c94f95774225b4d.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/57f84cb36f0333c5341f85e7fe7912f06b5cdc7b3c867fc87c94f95774225b4d.jpg)

![5d5b1fe4e55badcc96554b2034246d7b1357cb2d2becaf1b5853835f3aca80cd.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/5d5b1fe4e55badcc96554b2034246d7b1357cb2d2becaf1b5853835f3aca80cd.jpg)

![5e1e13482df886e534fabd226c724750f83092757bffc599b04951621c044ed6.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/5e1e13482df886e534fabd226c724750f83092757bffc599b04951621c044ed6.jpg)

![93f4b6845dd1c5d8b1cf949feb2cdc1f3f0b408a6aa1a4889cdf7d6d63fb9ea5.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/93f4b6845dd1c5d8b1cf949feb2cdc1f3f0b408a6aa1a4889cdf7d6d63fb9ea5.jpg)

![a1902d28fd4dd730e4286d51ca1d9849f890f87f6ea324b17fc1e4066844f8b2.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/a1902d28fd4dd730e4286d51ca1d9849f890f87f6ea324b17fc1e4066844f8b2.jpg)

![a3383bcd98b2aac83b5ff9a8af3f4b5d0504b017833c3a9b04fadd640a784d41.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/a3383bcd98b2aac83b5ff9a8af3f4b5d0504b017833c3a9b04fadd640a784d41.jpg)

![a4321546daa4863d9d8c7ee35d7652862e4967eef8d8f0c85fb68ad17bc77908.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/a4321546daa4863d9d8c7ee35d7652862e4967eef8d8f0c85fb68ad17bc77908.jpg)

![c3ee734c9b5dd4c0237a949ab4536c04b77b402d67012025f55558841ee4f6e8.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/c3ee734c9b5dd4c0237a949ab4536c04b77b402d67012025f55558841ee4f6e8.jpg)

![d94d9757a02ac550ec1376329dee32a4d0f97a48948e8e98e4cb589d1cd87367.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/d94d9757a02ac550ec1376329dee32a4d0f97a48948e8e98e4cb589d1cd87367.jpg)

![de9e49f7ff7521a5d2f8430a581b9401987ecda48853fc974a346d4f0907e7bc.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/de9e49f7ff7521a5d2f8430a581b9401987ecda48853fc974a346d4f0907e7bc.jpg)

![e392bd3e7a4f4c2afd85e3e582870661888f806ce4f60a516c56825889fee881.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/e392bd3e7a4f4c2afd85e3e582870661888f806ce4f60a516c56825889fee881.jpg)

![f83a564121d3f2d3e69e12a5eaab44a9d8e432c3b16c966fcca076a46c6d75ad.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/f83a564121d3f2d3e69e12a5eaab44a9d8e432c3b16c966fcca076a46c6d75ad.jpg)

![fd23d638b1eacdc6cf2779b487cad240053f64c34351527271f9fa0477f2c2fb.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/fd23d638b1eacdc6cf2779b487cad240053f64c34351527271f9fa0477f2c2fb.jpg)

![fd5aef197280995017fd5d8f1dff52c6fdc873ab9098208d2f53595ad000a753.jpg](../icml_results/4574_LLMs%20Lean%20on%20Priors%2C%20Not%20Programming%20Language%20Semantics/tables/fd5aef197280995017fd5d8f1dff52c6fdc873ab9098208d2f53595ad000a753.jpg)

<a id="paper-4528"></a>

## Advancing SVD-based LLM Compression via Layer-Wise Error Model Search


### Images

![21c95df5439f69cf8a9d9d50f194c769aa2357286f7e99cb4e24c62cad41152c.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/21c95df5439f69cf8a9d9d50f194c769aa2357286f7e99cb4e24c62cad41152c.jpg)

![30a000d42a2ef0269216d3062e0b2e76576edafc8143a9f5ff1680b6231494f5.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/30a000d42a2ef0269216d3062e0b2e76576edafc8143a9f5ff1680b6231494f5.jpg)

![3165777df2aa84694e1572fa3580157f2086e2a713fec4634409629b3f2d110c.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/3165777df2aa84694e1572fa3580157f2086e2a713fec4634409629b3f2d110c.jpg)

![359c6206b38d51ff88f1955617810207e494f6c7fe9319cfbe05994c7613aefa.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/359c6206b38d51ff88f1955617810207e494f6c7fe9319cfbe05994c7613aefa.jpg)

![39343b8caba552d77e69e1d11539691f79360b531acf2e47312e0e2ccf441295.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/39343b8caba552d77e69e1d11539691f79360b531acf2e47312e0e2ccf441295.jpg)

![4b3bba7d41e7424b077ca31bcdc4e0069983ecb1a37a931738879e5e5e7cc182.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/4b3bba7d41e7424b077ca31bcdc4e0069983ecb1a37a931738879e5e5e7cc182.jpg)

![4e3bd9af6a3773f737b04af7eb59ba02070d9370a4c6d50622a6c4e5d656f389.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/4e3bd9af6a3773f737b04af7eb59ba02070d9370a4c6d50622a6c4e5d656f389.jpg)

![7d5b1cba4357cb5d485a6caf210e6cf963f4daa723cadce68faf89611ebaa0da.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/7d5b1cba4357cb5d485a6caf210e6cf963f4daa723cadce68faf89611ebaa0da.jpg)

![a9d960a0f24a7f633cea8d3085cc3f8c2e0779b2df92219e1f18210fa5e90831.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/a9d960a0f24a7f633cea8d3085cc3f8c2e0779b2df92219e1f18210fa5e90831.jpg)

![b4ef18eff4c4b8cdb835e7a8b62272502d19278b53e3679fcd0d3f1fa17ece38.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/b4ef18eff4c4b8cdb835e7a8b62272502d19278b53e3679fcd0d3f1fa17ece38.jpg)

![bcc1b68230de3d9ca8201799fa6de24b808b52350d89ac86ccb3667595a999da.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/bcc1b68230de3d9ca8201799fa6de24b808b52350d89ac86ccb3667595a999da.jpg)

![bd32723872e3d45e78a4e39bee5d255e01ee5befae69d0ee227beab9ca5db106.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/bd32723872e3d45e78a4e39bee5d255e01ee5befae69d0ee227beab9ca5db106.jpg)

![cce6fed0176c92c1ac99361dcdda8aba10cba644616f22e5dc50bbd3158cf5c2.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/cce6fed0176c92c1ac99361dcdda8aba10cba644616f22e5dc50bbd3158cf5c2.jpg)

![e990e3e67696287550315a6aa234e2af2e2e7f5f19c6c892b55776ba4a70053a.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/e990e3e67696287550315a6aa234e2af2e2e7f5f19c6c892b55776ba4a70053a.jpg)

![eab8a8dff60bdc7d10618ef8a4ab20c97d9efefdddf30b47d9bcb219783d576d.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/images/eab8a8dff60bdc7d10618ef8a4ab20c97d9efefdddf30b47d9bcb219783d576d.jpg)

### Tables

![168c15ac3f06499039a760ff298b641253f72a074cf2b85139d4b6c325fb6cc9.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/168c15ac3f06499039a760ff298b641253f72a074cf2b85139d4b6c325fb6cc9.jpg)

![1d840c79f3abebd5b9aa68595327f27693c7a2ac1f56ae8734dd5fd427bb72b4.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/1d840c79f3abebd5b9aa68595327f27693c7a2ac1f56ae8734dd5fd427bb72b4.jpg)

![3114348f300c8c59827d3e9c8842fc2170ae6d48e2837e23d2e01b683e30686f.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/3114348f300c8c59827d3e9c8842fc2170ae6d48e2837e23d2e01b683e30686f.jpg)

![334b56a9f4e1b92b7472e34a6bc955b2cd3908b296a19aea8ddfa2ce253e6c63.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/334b56a9f4e1b92b7472e34a6bc955b2cd3908b296a19aea8ddfa2ce253e6c63.jpg)

![47836d2534d32e284eb92d42e3bda63aec3d49da394732fe15d52902fb370643.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/47836d2534d32e284eb92d42e3bda63aec3d49da394732fe15d52902fb370643.jpg)

![4afc62debafb70f98c32edbac87d399937bea4bda05f4ffbbaf4f680a5c2e323.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/4afc62debafb70f98c32edbac87d399937bea4bda05f4ffbbaf4f680a5c2e323.jpg)

![5ca80f2f64192c6ca7b1899d45289eec681aca5ded7d4b99b0f04529da323c17.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/5ca80f2f64192c6ca7b1899d45289eec681aca5ded7d4b99b0f04529da323c17.jpg)

![6264b2709bbf048230909380e9e5c8ab8e3e5d6cc6d09874d5010455a0cdb99f.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/6264b2709bbf048230909380e9e5c8ab8e3e5d6cc6d09874d5010455a0cdb99f.jpg)

![71bdd16ae0d2e0d9b83d881e9956a08de1af3bbbf81b727e934f6e9891b43d28.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/71bdd16ae0d2e0d9b83d881e9956a08de1af3bbbf81b727e934f6e9891b43d28.jpg)

![71d574ee896b300359bba0bd53dd7d63af0b1fe9b8ae4ed1650731bf7a1910d4.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/71d574ee896b300359bba0bd53dd7d63af0b1fe9b8ae4ed1650731bf7a1910d4.jpg)

![7aa7a17a5686367bae9c9be9918f89807132a06e721e378b0d0c309864d6e359.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/7aa7a17a5686367bae9c9be9918f89807132a06e721e378b0d0c309864d6e359.jpg)

![7b21bd0ca5a2b32f0cb3ece364e882174199ab7a7f93b79f4ac2debb7f6a13b7.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/7b21bd0ca5a2b32f0cb3ece364e882174199ab7a7f93b79f4ac2debb7f6a13b7.jpg)

![a8cfd66c332e3859a9f629ecfee3214baa27eec10e9a085e2eea04d18d31db88.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/a8cfd66c332e3859a9f629ecfee3214baa27eec10e9a085e2eea04d18d31db88.jpg)

![b2d00c3df0e17c85247393ce8852c4548ad11d8d980b95befe086b32fa835d8c.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/b2d00c3df0e17c85247393ce8852c4548ad11d8d980b95befe086b32fa835d8c.jpg)

![b50e663f0724dfc9c94898379b59607e0dd3bd49913e03ccecb2c436fcb0ae49.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/b50e663f0724dfc9c94898379b59607e0dd3bd49913e03ccecb2c436fcb0ae49.jpg)

![b97ad2051bf060a89ac15f14f4ad4987fb5b81f972d3a10f26d03ce2071584c1.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/b97ad2051bf060a89ac15f14f4ad4987fb5b81f972d3a10f26d03ce2071584c1.jpg)

![c56b13db2e1ea722da11ef46be8d8f992e408535e7a7602585f7f6458c293a63.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/c56b13db2e1ea722da11ef46be8d8f992e408535e7a7602585f7f6458c293a63.jpg)

![f0d54c7fda1cf990157908e9fd7371457c8213bfa6f7d4aba1482cf2a8164fee.jpg](../icml_results/4575_Advancing%20SVD-based%20LLM%20Compression%20via%20Layer-Wise%20Error%20Model%20Search/tables/f0d54c7fda1cf990157908e9fd7371457c8213bfa6f7d4aba1482cf2a8164fee.jpg)

<a id="paper-4529"></a>

## Deep Incentive Design with Differentiable Equilibrium Blocks


### Images

![0acb4de07c6d36fda56425c8c995aa8827fb124dd3f80baa4bcc1f96b7a57de5.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/0acb4de07c6d36fda56425c8c995aa8827fb124dd3f80baa4bcc1f96b7a57de5.jpg)

![0b72a4275958d8ebed8656d831403b5496a14056c71965cccb1df29771d9cdbc.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/0b72a4275958d8ebed8656d831403b5496a14056c71965cccb1df29771d9cdbc.jpg)

![1c7a13591f4575b4dd0812f4f908df3287d627a9d3fe4402d2cc6119cb9ad0b4.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/1c7a13591f4575b4dd0812f4f908df3287d627a9d3fe4402d2cc6119cb9ad0b4.jpg)

![5b475cee22b463fc8eaf995e10965b0ec3ba2ea3931ca91fe9f8d9697ca8fd5a.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/5b475cee22b463fc8eaf995e10965b0ec3ba2ea3931ca91fe9f8d9697ca8fd5a.jpg)

![b4d736ca08cdbd87496a3f6061ce1f1827b076b31cc39ffd660e390b2a636b31.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/b4d736ca08cdbd87496a3f6061ce1f1827b076b31cc39ffd660e390b2a636b31.jpg)

![dcc91ac122b6086e0a34926eb7b1170c69f946a7cc4697b6629858d344938716.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/dcc91ac122b6086e0a34926eb7b1170c69f946a7cc4697b6629858d344938716.jpg)

![ff752ee2ffb5a820c5f4ce52ab53bb664963aa1af9459096339ed24e3e5eb56e.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/images/ff752ee2ffb5a820c5f4ce52ab53bb664963aa1af9459096339ed24e3e5eb56e.jpg)

### Tables

![8fd56093edcf0e122d0a3c86a5b1ae7a96e7bf7e073271d377e8508d594bd65d.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/tables/8fd56093edcf0e122d0a3c86a5b1ae7a96e7bf7e073271d377e8508d594bd65d.jpg)

![94be2070fbbba3d636c10c80cf2e991f7aec8ca408664dba476ee55a588d1bd7.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/tables/94be2070fbbba3d636c10c80cf2e991f7aec8ca408664dba476ee55a588d1bd7.jpg)

![f9ba9f7312eed23c13441e7582961d5d63985eb26f82be1420e278a05de63e46.jpg](../icml_results/4576_Deep%20Incentive%20Design%20with%20Differentiable%20Equilibrium%20Blocks/tables/f9ba9f7312eed23c13441e7582961d5d63985eb26f82be1420e278a05de63e46.jpg)

<a id="paper-4530"></a>

## Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations


### Images

![36db82bb5a27987187aee8ffb61e60e38050a8a0baf70056ee343c47a951121a.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/36db82bb5a27987187aee8ffb61e60e38050a8a0baf70056ee343c47a951121a.jpg)

![49fd9d8c1ca76fdf15ab7d229fe6e3e4e9ab15f2e3da5166e9316079595f6aaf.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/49fd9d8c1ca76fdf15ab7d229fe6e3e4e9ab15f2e3da5166e9316079595f6aaf.jpg)

![68a72a8be2b5993d4829bae3f556b9099dfbaa324a0b5c5dd69c91c6bd2d4ac2.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/68a72a8be2b5993d4829bae3f556b9099dfbaa324a0b5c5dd69c91c6bd2d4ac2.jpg)

![7068c3e9356b35194dc229fcad33fb9e34db808b12935772ef720f99dcdce55e.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/7068c3e9356b35194dc229fcad33fb9e34db808b12935772ef720f99dcdce55e.jpg)

![87c5444b2d1093b9d709763d0d1c124d99792917ae85bee8e616437e8354a876.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/87c5444b2d1093b9d709763d0d1c124d99792917ae85bee8e616437e8354a876.jpg)

![abf9903c6df5e682d583609fb13ed7562c944579cc526fa054aec7cbef08925e.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/abf9903c6df5e682d583609fb13ed7562c944579cc526fa054aec7cbef08925e.jpg)

![c62f05ac91df09e14b53d080ce231b608b600c6ddae41970a9bd43e713966620.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/c62f05ac91df09e14b53d080ce231b608b600c6ddae41970a9bd43e713966620.jpg)

![e4303f8481abddb4e3754de481aa8abd1299e1fe7d72f5175bfc6b04e0338ba1.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/images/e4303f8481abddb4e3754de481aa8abd1299e1fe7d72f5175bfc6b04e0338ba1.jpg)

### Tables

![057ee6169705838b9c834bad2f732f6467db3e28ca1062bdc07c2fbcf4cea93d.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/057ee6169705838b9c834bad2f732f6467db3e28ca1062bdc07c2fbcf4cea93d.jpg)

![14caed15b8285f98017d573260688d10f5889a3579b23ee7b864a7d31f4ed922.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/14caed15b8285f98017d573260688d10f5889a3579b23ee7b864a7d31f4ed922.jpg)

![1ff250d542345cbf5caee6a8b78288f53da9870af592e7061c72248be1e232d5.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/1ff250d542345cbf5caee6a8b78288f53da9870af592e7061c72248be1e232d5.jpg)

![5687080822e13b837e75dc84336e6fdc4cd45ae8e25f4cf08ff24f6d122fa545.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/5687080822e13b837e75dc84336e6fdc4cd45ae8e25f4cf08ff24f6d122fa545.jpg)

![b5f633d7f1585fb5b873b16a8c7cb4c48227d0cea8b34d31f32955f14400dd1a.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/b5f633d7f1585fb5b873b16a8c7cb4c48227d0cea8b34d31f32955f14400dd1a.jpg)

![bf58927ae6cbcf2c0c4fd6fc446c19b28cba7c93fa8a55a5e77060e79c650abf.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/bf58927ae6cbcf2c0c4fd6fc446c19b28cba7c93fa8a55a5e77060e79c650abf.jpg)

![f29124d7fc84cc6f45d76df368ccbacfea0e70fbb09e2ec6f28d725642aa0581.jpg](../icml_results/4577_Learning%20Manifold%20and%20It%C3%B4%20Dynamics%20with%20Branched%20Neural%20Rough%20Differential%20Equations/tables/f29124d7fc84cc6f45d76df368ccbacfea0e70fbb09e2ec6f28d725642aa0581.jpg)

<a id="paper-4531"></a>

## A Tight Theory of Error Feedback Algorithms in Distributed Optimization


### Images

![8e53bf8f70fc31ab476b298ae0244405783a5215f9241b0e556e6eb2a95c3924.jpg](../icml_results/4578_A%20Tight%20Theory%20of%20Error%20Feedback%20Algorithms%20in%20Distributed%20Optimization/images/8e53bf8f70fc31ab476b298ae0244405783a5215f9241b0e556e6eb2a95c3924.jpg)

![bce04e16725ed3e73b129fbb3afc737db4c6eb53b1298641bf469c03d7fdadf6.jpg](../icml_results/4578_A%20Tight%20Theory%20of%20Error%20Feedback%20Algorithms%20in%20Distributed%20Optimization/images/bce04e16725ed3e73b129fbb3afc737db4c6eb53b1298641bf469c03d7fdadf6.jpg)

![da236cfd5ef583bc74ed6423536c1ed1a58aa32dfec4a45c08d9b696e3f1f38e.jpg](../icml_results/4578_A%20Tight%20Theory%20of%20Error%20Feedback%20Algorithms%20in%20Distributed%20Optimization/images/da236cfd5ef583bc74ed6423536c1ed1a58aa32dfec4a45c08d9b696e3f1f38e.jpg)

### Tables

![c918a2d4f54e3283118bbe6af7a9e893df813ee542cace1b3b9523087a154f34.jpg](../icml_results/4578_A%20Tight%20Theory%20of%20Error%20Feedback%20Algorithms%20in%20Distributed%20Optimization/tables/c918a2d4f54e3283118bbe6af7a9e893df813ee542cace1b3b9523087a154f34.jpg)

<a id="paper-4532"></a>

## Connecting Independently Trained Modes via Layer-Wise Connectivity


### Images

![0e758cf57729ea966c34f5700dfcb708cfdb18a9d6b869172c956424f2aead6e.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/0e758cf57729ea966c34f5700dfcb708cfdb18a9d6b869172c956424f2aead6e.jpg)

![20693ea57ac47849bc7af3ea916ecef01f5135290700d5e366034593d4d8ac5b.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/20693ea57ac47849bc7af3ea916ecef01f5135290700d5e366034593d4d8ac5b.jpg)

![2c054897cbb83c61abdabb2d04e07c7d7c7c8d626afb8c665e7739b8327932a6.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/2c054897cbb83c61abdabb2d04e07c7d7c7c8d626afb8c665e7739b8327932a6.jpg)

![325f8a210204bdbdee10f58ae53b143089a232b6a180348eff76f6902ddc1b44.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/325f8a210204bdbdee10f58ae53b143089a232b6a180348eff76f6902ddc1b44.jpg)

![385bb7ca1e97c1ddeedc08daf8e98e52cf703a10ee8bbcc524808f0d86012def.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/385bb7ca1e97c1ddeedc08daf8e98e52cf703a10ee8bbcc524808f0d86012def.jpg)

![4d0652345f668614c1b82d15359bd95d071d5033ea82436c8937f0926d70d59c.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/4d0652345f668614c1b82d15359bd95d071d5033ea82436c8937f0926d70d59c.jpg)

![5b60ea4df0234625b5fb195cefa03e6681c2f0d9f65379fa8bf7325137e58ae8.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/5b60ea4df0234625b5fb195cefa03e6681c2f0d9f65379fa8bf7325137e58ae8.jpg)

![63a81980849745b844334fb843b618b3304a452e0b24ec34d87861776e0dcb90.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/63a81980849745b844334fb843b618b3304a452e0b24ec34d87861776e0dcb90.jpg)

![6e07517ec1d0739fa8dc74b0f7566f7feb979c129bc3513818f198c6c5bc3abc.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/6e07517ec1d0739fa8dc74b0f7566f7feb979c129bc3513818f198c6c5bc3abc.jpg)

![750f4a45e5ef300dc0be6c5e364009cfa5b98ca08b03936933bd722cd05ed098.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/750f4a45e5ef300dc0be6c5e364009cfa5b98ca08b03936933bd722cd05ed098.jpg)

![784d736abb1598c0f8ed7f4df244dad621490ab255084387f7033cabf53fb79c.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/784d736abb1598c0f8ed7f4df244dad621490ab255084387f7033cabf53fb79c.jpg)

![80e31dd8a1ef0dd30403a8dee4a1af75e483fdf2cc7049714da3e05d17d977db.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/80e31dd8a1ef0dd30403a8dee4a1af75e483fdf2cc7049714da3e05d17d977db.jpg)

![82d9ebbe1159b7a1f74734b540653e31eca2ceb5c32f3d3a0f3f2e15189248c4.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/82d9ebbe1159b7a1f74734b540653e31eca2ceb5c32f3d3a0f3f2e15189248c4.jpg)

![954ed0f4bd2f34465c7456aa7489a42839a8771c8fabf6937ceed7015c297cce.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/954ed0f4bd2f34465c7456aa7489a42839a8771c8fabf6937ceed7015c297cce.jpg)

![a5c537d3e86179e8ff55137c08107cf7a1ffb663f8d5907c39abd3b8371ca326.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/a5c537d3e86179e8ff55137c08107cf7a1ffb663f8d5907c39abd3b8371ca326.jpg)

![a6c564fdf8130706530e0c362e1e2642540b7463f01a0d4e47ce085a452ab5f0.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/a6c564fdf8130706530e0c362e1e2642540b7463f01a0d4e47ce085a452ab5f0.jpg)

![c70f56a160460db7db568cbb6441fd9fefa11c99c1b2a019d7cecdbea9103ee1.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/c70f56a160460db7db568cbb6441fd9fefa11c99c1b2a019d7cecdbea9103ee1.jpg)

![c82f3456db9439454a255d5e06a5614080106fafd2b676b72127e7fe6e1403f5.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/c82f3456db9439454a255d5e06a5614080106fafd2b676b72127e7fe6e1403f5.jpg)

![ddc6619a03fbded6c61ebbbe3bb59f1af84d31b194ab6a31f9426b0fce0dc68e.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/ddc6619a03fbded6c61ebbbe3bb59f1af84d31b194ab6a31f9426b0fce0dc68e.jpg)

![f244bb7dd208f17c3f73effea13622ca3d8e4cfcd07ca51b68b462cc9666575a.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/f244bb7dd208f17c3f73effea13622ca3d8e4cfcd07ca51b68b462cc9666575a.jpg)

![f2c8a31a4356da5ade526e8ffc5f97b8927694d1b883228e8da46e27c4a410ed.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/f2c8a31a4356da5ade526e8ffc5f97b8927694d1b883228e8da46e27c4a410ed.jpg)

![f6ff64f1e96b871273edacef721b8169653ea51852be73ff55ebfd194a3e3a1b.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/images/f6ff64f1e96b871273edacef721b8169653ea51852be73ff55ebfd194a3e3a1b.jpg)

### Tables

![2799f28c5c9468534bfe9c86432f60fa7038b11fce987d087d37f21b8f7d62dc.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/2799f28c5c9468534bfe9c86432f60fa7038b11fce987d087d37f21b8f7d62dc.jpg)

![27ee67fa45467aa449bc7e06664935540a91cf6c71e896f57632dff559d8af15.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/27ee67fa45467aa449bc7e06664935540a91cf6c71e896f57632dff559d8af15.jpg)

![2c0ffa861547b89eafd63da007c56a136f95cbb2fb046c9ec4b934516d2e3392.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/2c0ffa861547b89eafd63da007c56a136f95cbb2fb046c9ec4b934516d2e3392.jpg)

![30d9e6386e52c4aed9d208edd898cdad0303a067149ee1c0a9349d302c077130.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/30d9e6386e52c4aed9d208edd898cdad0303a067149ee1c0a9349d302c077130.jpg)

![5bbb452fbce19c8b997360aacaacaedc01f05b9abf1d24d5ac6a0984bf271e18.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/5bbb452fbce19c8b997360aacaacaedc01f05b9abf1d24d5ac6a0984bf271e18.jpg)

![78b0de1b45d764746e6b6350a021356092b8532751d81d9286aa1fd652c704f5.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/78b0de1b45d764746e6b6350a021356092b8532751d81d9286aa1fd652c704f5.jpg)

![7bdbf3d0ac7a4b83b334385adf5dd8286b2d7551c2d0fbadcc91dc42671e4884.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/7bdbf3d0ac7a4b83b334385adf5dd8286b2d7551c2d0fbadcc91dc42671e4884.jpg)

![c9f1038ffc46532cfb20e6457fe1cb415d496068392bcc1ef876adddc3fad9aa.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/c9f1038ffc46532cfb20e6457fe1cb415d496068392bcc1ef876adddc3fad9aa.jpg)

![f1aa906e382cd8295555ab9963f39695d8cbd476928372776caa7c0e7e51c5ef.jpg](../icml_results/4579_Connecting%20Independently%20Trained%20Modes%20via%20Layer-Wise%20Connectivity/tables/f1aa906e382cd8295555ab9963f39695d8cbd476928372776caa7c0e7e51c5ef.jpg)

<a id="paper-4533"></a>

## Bridging the Gap Between Average and Discounted TD Learning


### Images

![270b2a1d4d041b026cd956b37fc950497538f8ac27eee70ef1c428429fa2df77.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/images/270b2a1d4d041b026cd956b37fc950497538f8ac27eee70ef1c428429fa2df77.jpg)

![5dd554deb02d6eb943a6fd4e10fa40fdef4622311e523c0ef70fa6d46d970604.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/images/5dd554deb02d6eb943a6fd4e10fa40fdef4622311e523c0ef70fa6d46d970604.jpg)

![5e5df41f994bc5b863504ce97c067a1f6ad153a0b03fd7e1761837d09a1cdbd0.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/images/5e5df41f994bc5b863504ce97c067a1f6ad153a0b03fd7e1761837d09a1cdbd0.jpg)

### Tables

![200bb4ca1743982fe8b13a302f9dd1c9953fcd9811ebecd054fd92475096f120.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/tables/200bb4ca1743982fe8b13a302f9dd1c9953fcd9811ebecd054fd92475096f120.jpg)

![32dbc0ebd508ad919d07984782ea34785f0d4fab7f6c6dee5cb3993f098b17ec.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/tables/32dbc0ebd508ad919d07984782ea34785f0d4fab7f6c6dee5cb3993f098b17ec.jpg)

![34f75176700cdd9e992ed2d5bf31bbf72d2bddb12d13ee98a79f327d8ad101c1.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/tables/34f75176700cdd9e992ed2d5bf31bbf72d2bddb12d13ee98a79f327d8ad101c1.jpg)

![6fe720aab1f8c89913f77a8bb337d1aead763866f742bd22ab42b4d14d54f4ae.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/tables/6fe720aab1f8c89913f77a8bb337d1aead763866f742bd22ab42b4d14d54f4ae.jpg)

![fa5a6145b6a993f07acea06aebd1ec5d6942964c24fcc05ad7eccd39eb74a863.jpg](../icml_results/4580_Bridging%20the%20Gap%20Between%20Average%20and%20Discounted%20TD%20Learning/tables/fa5a6145b6a993f07acea06aebd1ec5d6942964c24fcc05ad7eccd39eb74a863.jpg)

<a id="paper-4534"></a>

## Beyond Benchmarks: Toward Causally Faithful Evaluation of Large Language Models


### Images

![0198a785e0676dad755e98de805e2e5e0996404e49a02e315ca851fa1abe7465.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/0198a785e0676dad755e98de805e2e5e0996404e49a02e315ca851fa1abe7465.jpg)

![0e37bb30e2d4d4d65e2e145ac33037d25bc633e4dc04af4a02de60ae4c3d4878.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/0e37bb30e2d4d4d65e2e145ac33037d25bc633e4dc04af4a02de60ae4c3d4878.jpg)

![0e5be0c6611d89962697c8aaa54b3d8c6af2b62ba1787ee21e52ef83d4c16edc.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/0e5be0c6611d89962697c8aaa54b3d8c6af2b62ba1787ee21e52ef83d4c16edc.jpg)

![13b9048155c7de4d646adb0f55071fb493eb55c4d56b7310e6e37c1f7002de33.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/13b9048155c7de4d646adb0f55071fb493eb55c4d56b7310e6e37c1f7002de33.jpg)

![1a74f6c09ddecdfac9fd5404a7b1f9e6485ea792ae01a928e64049059d9179e8.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/1a74f6c09ddecdfac9fd5404a7b1f9e6485ea792ae01a928e64049059d9179e8.jpg)

![2014cfe1d166e2ec755aca714e13d43870ce4d25a70dc1b572ab7e2e900abb9d.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/2014cfe1d166e2ec755aca714e13d43870ce4d25a70dc1b572ab7e2e900abb9d.jpg)

![24fb935ccacfd0fd3495ab12193d86debea0fa797c0e828f86a01224cd0368ab.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/24fb935ccacfd0fd3495ab12193d86debea0fa797c0e828f86a01224cd0368ab.jpg)

![395c379e7df46f076760a2f0543c6cad68e30e161b39ef57a8498a3db5996419.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/395c379e7df46f076760a2f0543c6cad68e30e161b39ef57a8498a3db5996419.jpg)

![50d2badb649f54b38bc368b5072ca830d30f9bac16d24670f5dd3c1566ebc3a5.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/50d2badb649f54b38bc368b5072ca830d30f9bac16d24670f5dd3c1566ebc3a5.jpg)

![5f7a7435d795f66a2942a7ae05ef0a4dfe2dd956874bb0c99d9992d2ee1b4558.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/5f7a7435d795f66a2942a7ae05ef0a4dfe2dd956874bb0c99d9992d2ee1b4558.jpg)

![61418ad016ef286eb2a759851d98685050984cb058b1c64408dba2f42a1717e4.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/61418ad016ef286eb2a759851d98685050984cb058b1c64408dba2f42a1717e4.jpg)

![660d6938f5a5dc81f388e617dc17aee02de080801910977253a8ed57ea11727c.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/660d6938f5a5dc81f388e617dc17aee02de080801910977253a8ed57ea11727c.jpg)

![ab931622f636510dff39a05dd3eee07fb626095649bc37b9e465de8cf40d3166.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/ab931622f636510dff39a05dd3eee07fb626095649bc37b9e465de8cf40d3166.jpg)

![aefa7a8cfc480f5f50f05d57ea2c25fb0df3f406a9a915232fccc6c096066ae9.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/aefa7a8cfc480f5f50f05d57ea2c25fb0df3f406a9a915232fccc6c096066ae9.jpg)

![af9bdf058b4fd7af42eac216b87c79a277b165866b71e0a82149f87412a1b17a.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/af9bdf058b4fd7af42eac216b87c79a277b165866b71e0a82149f87412a1b17a.jpg)

![c11fc4987f813add9b2cd77ef877ff1a209b2063b0512999bb8283a09f74c650.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/c11fc4987f813add9b2cd77ef877ff1a209b2063b0512999bb8283a09f74c650.jpg)

![c27e65076a9d33482d4b9a8e52a3d7b723e4dff003f71ff6235fdc568781117a.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/c27e65076a9d33482d4b9a8e52a3d7b723e4dff003f71ff6235fdc568781117a.jpg)

![cd545d5cedd648c1b531ec6a64c4ff5b742d051ac13825d52cf4320e6de3d7ba.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/cd545d5cedd648c1b531ec6a64c4ff5b742d051ac13825d52cf4320e6de3d7ba.jpg)

![f26860b5da2adeac08bfe33b0ed17a7ce9d7eb0ab43cd4aa872c40129e3a1528.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/images/f26860b5da2adeac08bfe33b0ed17a7ce9d7eb0ab43cd4aa872c40129e3a1528.jpg)

### Tables

![04d17fe7e699eb4268e9c8d71fc5f65ef07593904d4b425a0c2c14b9e4e12001.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/04d17fe7e699eb4268e9c8d71fc5f65ef07593904d4b425a0c2c14b9e4e12001.jpg)

![09ff7e1add0e0580946ef1dc8d8a231f5c4a0a9977dbdd26b9d6e453139df8a1.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/09ff7e1add0e0580946ef1dc8d8a231f5c4a0a9977dbdd26b9d6e453139df8a1.jpg)

![0be314ab5aa48c31b4d9311e8575bc0371a0ee5d40dab7a1b469c857e6d3306a.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/0be314ab5aa48c31b4d9311e8575bc0371a0ee5d40dab7a1b469c857e6d3306a.jpg)

![107de325343a0c5dc13b0c311aee0a84911e3f0f0b2410c19d2d0a65d19765df.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/107de325343a0c5dc13b0c311aee0a84911e3f0f0b2410c19d2d0a65d19765df.jpg)

![2f09f50029b3359eedd2c1e1af9c0bd43bab4b9d4cd9e5836eda97e4d9ef4077.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/2f09f50029b3359eedd2c1e1af9c0bd43bab4b9d4cd9e5836eda97e4d9ef4077.jpg)

![336aa0ae98c5f284acc5892c2d56138de9b2a6556c580657f2fc601d525baea4.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/336aa0ae98c5f284acc5892c2d56138de9b2a6556c580657f2fc601d525baea4.jpg)

![354f691529524bc70444b9828a5088975c0f99d8ccbaa49af98edeedc945d843.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/354f691529524bc70444b9828a5088975c0f99d8ccbaa49af98edeedc945d843.jpg)

![3609fa051c6055d5b6c1d10918d72b535595178e994d11b9df76de3d27cb2435.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/3609fa051c6055d5b6c1d10918d72b535595178e994d11b9df76de3d27cb2435.jpg)

![45bb9034e84b468ee1579f69f9ddeedced82015d3dd849d459812d9fca0f055c.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/45bb9034e84b468ee1579f69f9ddeedced82015d3dd849d459812d9fca0f055c.jpg)

![50d5d115357f42c16296f686aff9c01ceecfd60875f3e79ec39bf7915599fa9a.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/50d5d115357f42c16296f686aff9c01ceecfd60875f3e79ec39bf7915599fa9a.jpg)

![5ab2763270ad5fbd966616f84902524fb73c9642d5b275347e65c0857f20e8e5.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/5ab2763270ad5fbd966616f84902524fb73c9642d5b275347e65c0857f20e8e5.jpg)

![6321dbc1c5c9303536b346080594da6d8bafd713ddc5b8560fb8191343edcd44.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/6321dbc1c5c9303536b346080594da6d8bafd713ddc5b8560fb8191343edcd44.jpg)

![701bca5e306da1d4697b919313da5b2012a71743dfb1ddbe7c7cdf24094ef86b.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/701bca5e306da1d4697b919313da5b2012a71743dfb1ddbe7c7cdf24094ef86b.jpg)

![71cecce267fb5e0b1af47b9e94f3806c303ebc3f1357da7818ae95fb7e88d9bf.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/71cecce267fb5e0b1af47b9e94f3806c303ebc3f1357da7818ae95fb7e88d9bf.jpg)

![79fe913bfef23121d58fe39c540388d7970216d9bf1d01e075b887671eb92c5d.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/79fe913bfef23121d58fe39c540388d7970216d9bf1d01e075b887671eb92c5d.jpg)

![877a09dcfc6ab8fb031ca64e95b60e3d794eba35ed33282c81e0685cb6f484a3.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/877a09dcfc6ab8fb031ca64e95b60e3d794eba35ed33282c81e0685cb6f484a3.jpg)

![8a5d8ce38576152019e1844b5534c258f704501d6179c4411a8e37e751014093.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/8a5d8ce38576152019e1844b5534c258f704501d6179c4411a8e37e751014093.jpg)

![a15929ea05271c3ac4a637aa822665bf4a947947d48fd448ef737261e702b5f6.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/a15929ea05271c3ac4a637aa822665bf4a947947d48fd448ef737261e702b5f6.jpg)

![a516f8c64e9ab9b95cf3eac273ead92d280437b6c69ef3b1a085355d46f2d693.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/a516f8c64e9ab9b95cf3eac273ead92d280437b6c69ef3b1a085355d46f2d693.jpg)

![a9ae01462a1e6d07ea1c9fc973853812e70bc0c85f6dc6b05642bb6515c72b76.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/a9ae01462a1e6d07ea1c9fc973853812e70bc0c85f6dc6b05642bb6515c72b76.jpg)

![ace96983074c26f14cf9ebf3779d565745e84bc6c52803f15799d18d29052dde.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/ace96983074c26f14cf9ebf3779d565745e84bc6c52803f15799d18d29052dde.jpg)

![b02cc648791494ab55b2b54d06d472d29e284a1a2deecb93f4cefc19bd86143b.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/b02cc648791494ab55b2b54d06d472d29e284a1a2deecb93f4cefc19bd86143b.jpg)

![b59293a3d3a2634d9cf3316a65ad22dbc0cdacb7ec6e3f65c0362e15e5fff9a7.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/b59293a3d3a2634d9cf3316a65ad22dbc0cdacb7ec6e3f65c0362e15e5fff9a7.jpg)

![bca4c2304857e6a2b4088728c5faf18bc41664b2ae6ed95c39dc04db6da50816.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/bca4c2304857e6a2b4088728c5faf18bc41664b2ae6ed95c39dc04db6da50816.jpg)

![be604f632ddeb6071d8a273aa4e8957f76a441b5398906a7178cfe5638984dd2.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/be604f632ddeb6071d8a273aa4e8957f76a441b5398906a7178cfe5638984dd2.jpg)

![bfc97c1ce63c8dda54d4bd85dd5678df609510c571d2c74d44e8bc34431e0410.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/bfc97c1ce63c8dda54d4bd85dd5678df609510c571d2c74d44e8bc34431e0410.jpg)

![d10d5a9768ca8aad92c60d9b3d228bf14e225fa2229b2a1b224f41f093b328f6.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/d10d5a9768ca8aad92c60d9b3d228bf14e225fa2229b2a1b224f41f093b328f6.jpg)

![dffcb5cd2a48cd112b9c1b9d2445baf8f57e3933712f9274416b20d17805cf79.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/dffcb5cd2a48cd112b9c1b9d2445baf8f57e3933712f9274416b20d17805cf79.jpg)

![f586bd712bdad5f90bcfa462b17b8cd62924b20dedeefbc2ec4c21b28379a1bb.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/f586bd712bdad5f90bcfa462b17b8cd62924b20dedeefbc2ec4c21b28379a1bb.jpg)

![fdb126856329938fa63cc3500c95918124cf73135673ccb2a08e5299413efb94.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/fdb126856329938fa63cc3500c95918124cf73135673ccb2a08e5299413efb94.jpg)

![fed55ad0443add347cb1861c52a667cc330fb401f09ea4195bf4216b871d9d86.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/fed55ad0443add347cb1861c52a667cc330fb401f09ea4195bf4216b871d9d86.jpg)

![fee0c93bf0ea07246e3c7ec47649675f4d33e0a925927182ba147b3e48e38cd8.jpg](../icml_results/4581_Beyond%20Benchmarks_%20Toward%20Causally%20Faithful%20Evaluation%20of%20Large%20Language%20Models/tables/fee0c93bf0ea07246e3c7ec47649675f4d33e0a925927182ba147b3e48e38cd8.jpg)

<a id="paper-4535"></a>

## AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise


### Images

![148931196754536883743eef22f68a5d2d60a9abbdfeff5fb122464b245cfc1d.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/148931196754536883743eef22f68a5d2d60a9abbdfeff5fb122464b245cfc1d.jpg)

![2db4189ba08e0f097602259f6fc6dc0f4c85e01a3d9f733e1aa6e2cd2b6f2054.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/2db4189ba08e0f097602259f6fc6dc0f4c85e01a3d9f733e1aa6e2cd2b6f2054.jpg)

![5ce7f3ec8451eee506e519cde668b404448977da2365a1af4778534efb3ab215.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/5ce7f3ec8451eee506e519cde668b404448977da2365a1af4778534efb3ab215.jpg)

![b8df15aa37cfe2d0673967e2424c134e748a656abfb22a4eb1c1297c1b54d854.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/b8df15aa37cfe2d0673967e2424c134e748a656abfb22a4eb1c1297c1b54d854.jpg)

![d510433bd482513c5bc469145097a2dd512989fd0979cc65ba9ffbbaa2cca585.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/d510433bd482513c5bc469145097a2dd512989fd0979cc65ba9ffbbaa2cca585.jpg)

![db5304b0711f3e60c74cd5a55c728b6827b284460205daf09db6b753740c16bb.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/db5304b0711f3e60c74cd5a55c728b6827b284460205daf09db6b753740c16bb.jpg)

![e9e339ff979bb61956121673e470b062bca8d54f9b3018972afd569be23cd2a8.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/images/e9e339ff979bb61956121673e470b062bca8d54f9b3018972afd569be23cd2a8.jpg)

### Tables

![14c5e09e365ebc2cb0e4ee39af0b6682d6f6e738c57415b039a049e8ea2ff699.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/tables/14c5e09e365ebc2cb0e4ee39af0b6682d6f6e738c57415b039a049e8ea2ff699.jpg)

![4f853ad275972307852593efa716e0d824674e128b1432a275f8fa11cb42b360.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/tables/4f853ad275972307852593efa716e0d824674e128b1432a275f8fa11cb42b360.jpg)

![831fce36728a94460651198a4df58cfd8ed2aff5b8038393d000ad4dec5d066c.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/tables/831fce36728a94460651198a4df58cfd8ed2aff5b8038393d000ad4dec5d066c.jpg)

![91495f0c874921bca7410a19cf7ef7e1202495f5b315bb4bf237ac443b8d30ae.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/tables/91495f0c874921bca7410a19cf7ef7e1202495f5b315bb4bf237ac443b8d30ae.jpg)

![b517d7a62226d435739b211839728342db95d7e5c48821fe251bf26f8bfbd390.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/tables/b517d7a62226d435739b211839728342db95d7e5c48821fe251bf26f8bfbd390.jpg)

![e3f0a792fbee0d28d65340e9d2bc7a0434a9995a797c9d08dba3680f0dd0e1da.jpg](../icml_results/4582_AnyEdit%2B%2B_%20Adaptive%20Long-Form%20Knowledge%20Editing%20via%20Bayesian%20Surprise/tables/e3f0a792fbee0d28d65340e9d2bc7a0434a9995a797c9d08dba3680f0dd0e1da.jpg)

<a id="paper-4536"></a>

## QPKO: Differentiable QP-Embedded Deep Koopman Framework for Modeling Nonlinear Systems


### Images

![5a924561625a68b05bd5925e55984575459fe6ac3aace146769dfc1a8ddc701c.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/images/5a924561625a68b05bd5925e55984575459fe6ac3aace146769dfc1a8ddc701c.jpg)

![c72461e08d31074b69bbf4d9422c6eb714605f16eec0aabe42367c1ae65901b7.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/images/c72461e08d31074b69bbf4d9422c6eb714605f16eec0aabe42367c1ae65901b7.jpg)

![d6d5c9325f8c8f2a7f6070979fd57f72aad0731588372677c5743eedac0ca4a1.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/images/d6d5c9325f8c8f2a7f6070979fd57f72aad0731588372677c5743eedac0ca4a1.jpg)

![ea40fc9a6476ae709ebcd10ba87ffd30a4c52da190acc547dfa362f1342d3801.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/images/ea40fc9a6476ae709ebcd10ba87ffd30a4c52da190acc547dfa362f1342d3801.jpg)

### Tables

![05ed75ff2c7d8c696b20b25961843ab876d4ff5ca03fde6e20252a40f9ec4f96.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/05ed75ff2c7d8c696b20b25961843ab876d4ff5ca03fde6e20252a40f9ec4f96.jpg)

![1a4a96df7ee6afefc2ae11c0f387716077a11e45d5fdeb9034e96909ed4bd8f6.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/1a4a96df7ee6afefc2ae11c0f387716077a11e45d5fdeb9034e96909ed4bd8f6.jpg)

![2d64d82b4c71a8f156e75cc056ac3bcfb699e497c4deb43c119dd397056e4dd2.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/2d64d82b4c71a8f156e75cc056ac3bcfb699e497c4deb43c119dd397056e4dd2.jpg)

![327fe98ec8b6e258fff39fe831456b913070624e57f00f2a78a3dcf2247031cd.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/327fe98ec8b6e258fff39fe831456b913070624e57f00f2a78a3dcf2247031cd.jpg)

![3da80aa95f77a6a661e2d94c0670a1402464129f6ba023111a3ab65a247ce38b.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/3da80aa95f77a6a661e2d94c0670a1402464129f6ba023111a3ab65a247ce38b.jpg)

![43889570357cd24aed1b2590413c1bc98bc6cbe3f758d4fc5c8cfa98da128b7e.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/43889570357cd24aed1b2590413c1bc98bc6cbe3f758d4fc5c8cfa98da128b7e.jpg)

![772cda1614098e8c3b447840a10cc80dd4f28bf95286d380232407b9a3ac74aa.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/772cda1614098e8c3b447840a10cc80dd4f28bf95286d380232407b9a3ac74aa.jpg)

![92b4018aa0927ada5c8bb7dd5b2684b262479dd1c51ff62223e97c1b861a7bc3.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/92b4018aa0927ada5c8bb7dd5b2684b262479dd1c51ff62223e97c1b861a7bc3.jpg)

![b51eda8b7cae1bdc58726aa0a7c12fa0c8905d62c27bc8219280ebc6e7c78456.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/b51eda8b7cae1bdc58726aa0a7c12fa0c8905d62c27bc8219280ebc6e7c78456.jpg)

![bff6dcf0468ec338dc5297e13ff0b13e9454469fb5f8c58813f3960e8ca759e5.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/bff6dcf0468ec338dc5297e13ff0b13e9454469fb5f8c58813f3960e8ca759e5.jpg)

![cb48d326ab35e21932b38975d1d0349c71091e583a971c9d1c4011b67068b263.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/cb48d326ab35e21932b38975d1d0349c71091e583a971c9d1c4011b67068b263.jpg)

![e104bb48895eb8d17f583e9d0ba98f55fe10dad7dc9655cb788605086bd2ccac.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/e104bb48895eb8d17f583e9d0ba98f55fe10dad7dc9655cb788605086bd2ccac.jpg)

![f1c1d45d2120a19053f09a0580b16e3de92592c15397e3c2d5d5a065df0847cd.jpg](../icml_results/4583_QPKO_%20Differentiable%20QP-Embedded%20Deep%20Koopman%20Framework%20for%20Modeling%20Nonlinear%20Systems/tables/f1c1d45d2120a19053f09a0580b16e3de92592c15397e3c2d5d5a065df0847cd.jpg)

<a id="paper-4537"></a>

## RGMem: Renormalization Group–inspired Memory Evolution for Language Agents


### Images

![027084da63a1323fe442a73d8e6335e653ddeca22012dad9b7444af65ec48c69.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/027084da63a1323fe442a73d8e6335e653ddeca22012dad9b7444af65ec48c69.jpg)

![030562035d53d6a5603a911994740b38887033efa8b187e1966f2c76ca1cc1a4.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/030562035d53d6a5603a911994740b38887033efa8b187e1966f2c76ca1cc1a4.jpg)

![0dfe8b349ba79c64c3fe2b197cb1edbf5d9ce1bb5941dc5850a70b571ad6283c.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/0dfe8b349ba79c64c3fe2b197cb1edbf5d9ce1bb5941dc5850a70b571ad6283c.jpg)

![12b980a8b0a34ab9481a02e3d783d4540fa47e3fc8d6ebe5ddeeb54471b5aa83.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/12b980a8b0a34ab9481a02e3d783d4540fa47e3fc8d6ebe5ddeeb54471b5aa83.jpg)

![44718b1299df0bb35badf1af8f19068ec12939aa8130ad9f53e17043a4ba0593.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/44718b1299df0bb35badf1af8f19068ec12939aa8130ad9f53e17043a4ba0593.jpg)

![730915f85dab958198f3aafab34e6389988f07a80400098699da7c19a6392023.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/730915f85dab958198f3aafab34e6389988f07a80400098699da7c19a6392023.jpg)

![a852d9b838731fa4dbedafb4a85585fa557f2f080517e137f88418b7ab50d741.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/a852d9b838731fa4dbedafb4a85585fa557f2f080517e137f88418b7ab50d741.jpg)

![cc4abc2d09c77e3547328bf8ad78febc914e0914000fce84c980c88a19e43992.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/cc4abc2d09c77e3547328bf8ad78febc914e0914000fce84c980c88a19e43992.jpg)

![dbd653eef1718d1fbaedcb8ea23acdd706f0eaaffff3b46a29f59d95be7fa2d7.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/dbd653eef1718d1fbaedcb8ea23acdd706f0eaaffff3b46a29f59d95be7fa2d7.jpg)

![f7d8acb7a544f1b24c027ba66589a9a990934e95bc6d1f5f4fa4be099fd7ff84.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/f7d8acb7a544f1b24c027ba66589a9a990934e95bc6d1f5f4fa4be099fd7ff84.jpg)

![fa33da2363c314568dea85c3552f13ffc6253dda66ecdbe6970ac67e25e58f2f.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/images/fa33da2363c314568dea85c3552f13ffc6253dda66ecdbe6970ac67e25e58f2f.jpg)

### Tables

![02d91fedbdd6362dea389bb08308a62685f0cf9f9cef75f1e02030bf492a67df.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/02d91fedbdd6362dea389bb08308a62685f0cf9f9cef75f1e02030bf492a67df.jpg)

![1193e11173af7ee154a211e169e7fae206acbfd6eb0e79f8b0d5319be15af87b.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/1193e11173af7ee154a211e169e7fae206acbfd6eb0e79f8b0d5319be15af87b.jpg)

![146ec9b405a6a5a9781313c13324de40749ad9ca1c330aa9757f077bc41fc91f.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/146ec9b405a6a5a9781313c13324de40749ad9ca1c330aa9757f077bc41fc91f.jpg)

![4e71b8398762125aa9a69c0d5b49d7386261c7abf390d66e3664e1eadfe44690.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/4e71b8398762125aa9a69c0d5b49d7386261c7abf390d66e3664e1eadfe44690.jpg)

![924cf90240c645aaedce1a6f26b9787fcdf30d3fd446d105face4a0baed1cc71.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/924cf90240c645aaedce1a6f26b9787fcdf30d3fd446d105face4a0baed1cc71.jpg)

![d97ca42870bc8e0eab6caedf614f61c85b83ea3d96e43ccbdf7810998d1b53e6.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/d97ca42870bc8e0eab6caedf614f61c85b83ea3d96e43ccbdf7810998d1b53e6.jpg)

![fa94971a7382a61a72bb9fd03545bae9de5e77ed5b5b1cb361240aa3d885af9b.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/fa94971a7382a61a72bb9fd03545bae9de5e77ed5b5b1cb361240aa3d885af9b.jpg)

![ffba7a5f2337c41b98ecd097edfa19531e59c697f4ba64b7c621ec83018f762d.jpg](../icml_results/4584_RGMem_%20Renormalization%20Group%E2%80%93inspired%20Memory%20Evolution%20for%20Language%20Agents/tables/ffba7a5f2337c41b98ecd097edfa19531e59c697f4ba64b7c621ec83018f762d.jpg)

<a id="paper-4538"></a>

## Learning Discriminative and Generalizable Anomaly Detector for Dynamic Graph with Limited Supervision


### Images

![124c194e12f55c8420dfb9c4b428ce36e353be2d8bc0942dd8a56a88bdeb2244.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/124c194e12f55c8420dfb9c4b428ce36e353be2d8bc0942dd8a56a88bdeb2244.jpg)

![1734ad576f3f95eb9903fc6a39034e21f7bdc5e02da38b770bceb9f271d18d8e.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/1734ad576f3f95eb9903fc6a39034e21f7bdc5e02da38b770bceb9f271d18d8e.jpg)

![1fb18c65ea8798ad7af581f1f8af768abec90ddff4461772d5a637686fc6ee51.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/1fb18c65ea8798ad7af581f1f8af768abec90ddff4461772d5a637686fc6ee51.jpg)

![4b10cc1bc61cf63870f5302c06b0acd0579bab6753dbf71a0dd7582d6756ff7b.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/4b10cc1bc61cf63870f5302c06b0acd0579bab6753dbf71a0dd7582d6756ff7b.jpg)

![6bdcc239533cc88ea8fe0e171096e9c8d494e2596c852a3b8c2f4d0149997eba.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/6bdcc239533cc88ea8fe0e171096e9c8d494e2596c852a3b8c2f4d0149997eba.jpg)

![b348e5a60f2da857721ffc704283a3f641f9e9e31892acada8b328d7e2430402.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/b348e5a60f2da857721ffc704283a3f641f9e9e31892acada8b328d7e2430402.jpg)

![f9f14ec3b625559490faa98886211a83fd472e19a41b4b173a5a48cf05b50eee.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/images/f9f14ec3b625559490faa98886211a83fd472e19a41b4b173a5a48cf05b50eee.jpg)

### Tables

![01f0754147ba9a09827ca96d11c13cfbebe6d899d5936a81acc023296c14221f.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/01f0754147ba9a09827ca96d11c13cfbebe6d899d5936a81acc023296c14221f.jpg)

![18e6e308411e73057170305ed4804116c829031673268af735354fd0ad9a2877.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/18e6e308411e73057170305ed4804116c829031673268af735354fd0ad9a2877.jpg)

![203a85e74805bca93dd5d9acbee745be94efde3e60f7597121da9cb922654758.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/203a85e74805bca93dd5d9acbee745be94efde3e60f7597121da9cb922654758.jpg)

![2deb1758bc67d1400fa3f40320b1e9dc445d60c7677c72422591ed9416f0e968.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/2deb1758bc67d1400fa3f40320b1e9dc445d60c7677c72422591ed9416f0e968.jpg)

![2eda44904e8aea0fa3e8e9ced4098154df5623a21178559909d592761d8e43e1.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/2eda44904e8aea0fa3e8e9ced4098154df5623a21178559909d592761d8e43e1.jpg)

![357254648d3a03c84255adbed30be5b55d24524b744fffa30a42df6263b77a11.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/357254648d3a03c84255adbed30be5b55d24524b744fffa30a42df6263b77a11.jpg)

![52720788b4aef9184a216eb7b6098c81b4197863131dd359f1fb056e53c2740f.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/52720788b4aef9184a216eb7b6098c81b4197863131dd359f1fb056e53c2740f.jpg)

![5be61a45a9679361ed6460e439e9cccce330d5d77a81af6c7ee12a63ded2bf56.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/5be61a45a9679361ed6460e439e9cccce330d5d77a81af6c7ee12a63ded2bf56.jpg)

![9f9ccc32cdce20cf7d821eebe092b8e9730e048c485a419e231cba0d7eeae8f0.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/9f9ccc32cdce20cf7d821eebe092b8e9730e048c485a419e231cba0d7eeae8f0.jpg)

![b965cfb968842c88f3411e1b79b50f945249207b18244eecacda38ff2bd9f6b2.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/b965cfb968842c88f3411e1b79b50f945249207b18244eecacda38ff2bd9f6b2.jpg)

![c560922e08383e0653418421701637bf876ac8a7ec3073e2bf3ed41713bd7743.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/c560922e08383e0653418421701637bf876ac8a7ec3073e2bf3ed41713bd7743.jpg)

![db0e1e28ca7040d0c61db9c73f8323ece8141e1ced49e85144814ae743153788.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/db0e1e28ca7040d0c61db9c73f8323ece8141e1ced49e85144814ae743153788.jpg)

![edb496d0d4472e08955af4e5f48b85f114c90d8e522525ac8a1ba5fd07c09d49.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/edb496d0d4472e08955af4e5f48b85f114c90d8e522525ac8a1ba5fd07c09d49.jpg)

![f100ba6ab67e0b7bdee343d6bb801aae535a8ffa4851c33db307d89df10293cb.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/f100ba6ab67e0b7bdee343d6bb801aae535a8ffa4851c33db307d89df10293cb.jpg)

![f71cb092bd7a972cc0171fd4f613abb6b1dfaf0ee8e1f12a32b16ad0cbc8fc3c.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/f71cb092bd7a972cc0171fd4f613abb6b1dfaf0ee8e1f12a32b16ad0cbc8fc3c.jpg)

![fbbd505de074ec8f5e179d74754e43918c23533cc07bebcf143997f2911ed1cd.jpg](../icml_results/4585_Learning%20Discriminative%20and%20Generalizable%20Anomaly%20Detector%20for%20Dynamic%20Graph%20with%20Limited%20Supervisio/tables/fbbd505de074ec8f5e179d74754e43918c23533cc07bebcf143997f2911ed1cd.jpg)

<a id="paper-4539"></a>

## A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification


### Images

![75a710c879f72267c4c2d06a9673b9e35fe8fd9cff86d7a387c1e80896e45659.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/75a710c879f72267c4c2d06a9673b9e35fe8fd9cff86d7a387c1e80896e45659.jpg)

![9b0c2c1a474a814cf7fe674cf8e17f3f43607ceaec83169ef082b9462291f698.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/9b0c2c1a474a814cf7fe674cf8e17f3f43607ceaec83169ef082b9462291f698.jpg)

![b30962228d2916e3607bb28cf4cb5ff28337ac3b6b1e3f80116888f4cb2b4228.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/b30962228d2916e3607bb28cf4cb5ff28337ac3b6b1e3f80116888f4cb2b4228.jpg)

![c4e802f224d92365f505d6944d39fe8d3afc39d5f011c82303f491a63d670090.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/c4e802f224d92365f505d6944d39fe8d3afc39d5f011c82303f491a63d670090.jpg)

![ce7768f60ff21f04d02a228c09e40a09a8917b57bed354e7e56ca999e5e73049.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/ce7768f60ff21f04d02a228c09e40a09a8917b57bed354e7e56ca999e5e73049.jpg)

![d10a2e291fdcaf99d32a5a4f16b156f0c43b006c945bd09e7db9f49131e74c18.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/d10a2e291fdcaf99d32a5a4f16b156f0c43b006c945bd09e7db9f49131e74c18.jpg)

![de4663e64736a04c51feb34246ee312f8d2ecc851f058c54d07ff5e180e7f3de.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/images/de4663e64736a04c51feb34246ee312f8d2ecc851f058c54d07ff5e180e7f3de.jpg)

### Tables

![158850cbfab0bde4708aeb0718533da9d81fe3ac2de2b8960ef6caa4ae38dfd5.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/158850cbfab0bde4708aeb0718533da9d81fe3ac2de2b8960ef6caa4ae38dfd5.jpg)

![55fd826541c44cb3a87639128830377dc47235e928566cada243663d91811dc1.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/55fd826541c44cb3a87639128830377dc47235e928566cada243663d91811dc1.jpg)

![5804e97f6480166380f8678b7a00bad1e3a2885d672d87e72a3a7d87150ea420.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/5804e97f6480166380f8678b7a00bad1e3a2885d672d87e72a3a7d87150ea420.jpg)

![96fb09c4907239bdb6d0c9a0d0655255bcd086e6f49dcbebb61e32ca705cbf3f.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/96fb09c4907239bdb6d0c9a0d0655255bcd086e6f49dcbebb61e32ca705cbf3f.jpg)

![b84d09895c1a1488d728b591c38ed670c8c7cc7630932580a972ad95cd13baf2.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/b84d09895c1a1488d728b591c38ed670c8c7cc7630932580a972ad95cd13baf2.jpg)

![c4983f97b6c3091bb05936107ebb9540941f28ca309e2760c21e083d8aa5dc13.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/c4983f97b6c3091bb05936107ebb9540941f28ca309e2760c21e083d8aa5dc13.jpg)

![ee7eddcb77bf1a1dd52931aaf00a508d964c0ef0ca995d89702ad24bda64c56d.jpg](../icml_results/4586_A%20Conflict-aware%20Evidential%20Framework%20for%20Reliable%20Sleep%20Stage%20Classification/tables/ee7eddcb77bf1a1dd52931aaf00a508d964c0ef0ca995d89702ad24bda64c56d.jpg)

<a id="paper-4540"></a>

## Neutral-Reference Prompting for Vision–Language Models


### Images

![0bc88e96c154a3dbda00f2fea0c9645f69530be06fcf57a3b84c3cf9958d1144.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/0bc88e96c154a3dbda00f2fea0c9645f69530be06fcf57a3b84c3cf9958d1144.jpg)

![0d97119f497b88a547be63af0b96072da17d61fd2b6aa895cd775e46c34595d6.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/0d97119f497b88a547be63af0b96072da17d61fd2b6aa895cd775e46c34595d6.jpg)

![75b234ff92c86ba64d9dbd12836555abd717b75d8288f1371cf95451e72349a8.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/75b234ff92c86ba64d9dbd12836555abd717b75d8288f1371cf95451e72349a8.jpg)

![80b88a125f89c19ee2d1ffd24353ed7a1130b0bde5cc99ca67fd39d25c45e777.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/80b88a125f89c19ee2d1ffd24353ed7a1130b0bde5cc99ca67fd39d25c45e777.jpg)

![a39026665d2a9850797a15ae83e7484e0c68efcd61f8438ce904ed0c3638e900.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/a39026665d2a9850797a15ae83e7484e0c68efcd61f8438ce904ed0c3638e900.jpg)

![bd2fc51e176cce1d9804d86105858aef9837116e33d1fdf19cf0424c30521137.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/bd2fc51e176cce1d9804d86105858aef9837116e33d1fdf19cf0424c30521137.jpg)

![cfd400b1b2b740c697aaa3016a6f260dc1869924996c63b4ca55f2600ad3a927.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/cfd400b1b2b740c697aaa3016a6f260dc1869924996c63b4ca55f2600ad3a927.jpg)

![d9769c68c6b90abb39881452939632cc0ce53938a6dbbc09793150b815d13791.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/d9769c68c6b90abb39881452939632cc0ce53938a6dbbc09793150b815d13791.jpg)

![da630c11d4eb886e0fb03deda6c80827147bb2df4031c5c09d46aa2a79f65059.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/images/da630c11d4eb886e0fb03deda6c80827147bb2df4031c5c09d46aa2a79f65059.jpg)

### Tables

![1c8c8a69b195a418389aa1430661e42c6e3a3576a4b33d282c84275cd27d5d37.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/1c8c8a69b195a418389aa1430661e42c6e3a3576a4b33d282c84275cd27d5d37.jpg)

![21569055d0650ca07bfb36ff0f0cd5616c92af6a284246ad752f354493e39f51.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/21569055d0650ca07bfb36ff0f0cd5616c92af6a284246ad752f354493e39f51.jpg)

![3255c596f1c5bf5756467583ed77e11fe81492327cf7c9f9f142bdd8733f8ebc.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/3255c596f1c5bf5756467583ed77e11fe81492327cf7c9f9f142bdd8733f8ebc.jpg)

![427286eec67390f43be06ca0c4c92ee2a5185fea0a383bbfdf338af611875639.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/427286eec67390f43be06ca0c4c92ee2a5185fea0a383bbfdf338af611875639.jpg)

![8126e9d1704759bf984ddf180a44a72277647fc1e9c17179a3c734f0bc41921c.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/8126e9d1704759bf984ddf180a44a72277647fc1e9c17179a3c734f0bc41921c.jpg)

![c577f78fb6b0dd3a25c39db72bcba76766129f6d7a48bca8182edec148e8019f.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/c577f78fb6b0dd3a25c39db72bcba76766129f6d7a48bca8182edec148e8019f.jpg)

![cd582ccad5a1f48bb729e04d14d666fe6adfc27caee6e08a862741245051df4a.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/cd582ccad5a1f48bb729e04d14d666fe6adfc27caee6e08a862741245051df4a.jpg)

![e4f5481788637338bcf2abb546568e1e7650afb67dddb9103c849770596ebbbc.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/e4f5481788637338bcf2abb546568e1e7650afb67dddb9103c849770596ebbbc.jpg)

![e69c65e568bd90aa6af3ff8ede7775dc8fca3599b35f04ae34b421f465aadbf3.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/e69c65e568bd90aa6af3ff8ede7775dc8fca3599b35f04ae34b421f465aadbf3.jpg)

![fa937997efe0bedb247a5bf0b15fe0b5e1b8cf406afc4dab3fe94f666383123a.jpg](../icml_results/4587_Neutral-Reference%20Prompting%20for%20Vision%E2%80%93Language%20Models/tables/fa937997efe0bedb247a5bf0b15fe0b5e1b8cf406afc4dab3fe94f666383123a.jpg)

<a id="paper-4541"></a>

## From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers


### Images

![0209d3dfd659ac08fa6f08064cd0ad06424dbb2c191213be191a6ff700d1b21e.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/0209d3dfd659ac08fa6f08064cd0ad06424dbb2c191213be191a6ff700d1b21e.jpg)

![0a1eae7d7f9e1d4677e5d68575f4abf999b384b2ad6c8668c5917ff0e3a34f8f.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/0a1eae7d7f9e1d4677e5d68575f4abf999b384b2ad6c8668c5917ff0e3a34f8f.jpg)

![3ff91d8afc52953f7addc634e9a0e607507f8fa0ff539487848cdb6ff1826114.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/3ff91d8afc52953f7addc634e9a0e607507f8fa0ff539487848cdb6ff1826114.jpg)

![43424a06a5e2de9a7da344fa9bb64c2728d28c76e72d0c6d19d61705e3a31ed8.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/43424a06a5e2de9a7da344fa9bb64c2728d28c76e72d0c6d19d61705e3a31ed8.jpg)

![4df3719ccacf3697a2838849bdad53ef02d48b43d3f5877603f7cd5ecb0b3065.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/4df3719ccacf3697a2838849bdad53ef02d48b43d3f5877603f7cd5ecb0b3065.jpg)

![611125d5ef63ceee22564c9132c500c89bad455bb848505a940a41933e49fc4c.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/611125d5ef63ceee22564c9132c500c89bad455bb848505a940a41933e49fc4c.jpg)

![68a5cbd0b578fbd9cb501a65ea68feb8fcec9476a5c2a72f3f0d8674939cda3b.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/68a5cbd0b578fbd9cb501a65ea68feb8fcec9476a5c2a72f3f0d8674939cda3b.jpg)

![7557de0153ae3487a9788d06b0c6df38e6e9d17b8a45b5303c59a07414c4bca9.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/7557de0153ae3487a9788d06b0c6df38e6e9d17b8a45b5303c59a07414c4bca9.jpg)

![796a5b54ec34c4b57dac666e8118b046496c229ff94b21d3fc1d5ff01b78fab4.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/796a5b54ec34c4b57dac666e8118b046496c229ff94b21d3fc1d5ff01b78fab4.jpg)

![7c412654b7e15cd7b17448b8651f5a25388e65230dd750893011945bdd45824f.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/7c412654b7e15cd7b17448b8651f5a25388e65230dd750893011945bdd45824f.jpg)

![863b168b21bc16d128e3f7d0a2af2b63d3061a9af07f1442b51eb44522a602ee.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/863b168b21bc16d128e3f7d0a2af2b63d3061a9af07f1442b51eb44522a602ee.jpg)

![89564aae7e7b391760b30b0ad713d5a52a797c96e162df71699cc69fbd4868dc.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/89564aae7e7b391760b30b0ad713d5a52a797c96e162df71699cc69fbd4868dc.jpg)

![937a710e6876216536d91e9042fe7633c2aa53f56a3733fb19de453b8ea6ea30.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/937a710e6876216536d91e9042fe7633c2aa53f56a3733fb19de453b8ea6ea30.jpg)

![99ee61a3e990ce0724a6b47feb21393db95ca1b13d9a7c2b1abe7f8857ed009d.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/99ee61a3e990ce0724a6b47feb21393db95ca1b13d9a7c2b1abe7f8857ed009d.jpg)

![a4a1ef9fb50a0a5b4a0f55e46bcbcc086339152db85031eb4cee81d13b52d37d.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/a4a1ef9fb50a0a5b4a0f55e46bcbcc086339152db85031eb4cee81d13b52d37d.jpg)

![a8b8a4ddf44f9e3b9ee78a378dbe0fbbf6f77f9a7c9fab75b2cfec8fb2ffb766.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/a8b8a4ddf44f9e3b9ee78a378dbe0fbbf6f77f9a7c9fab75b2cfec8fb2ffb766.jpg)

![ae91e462ff99176e00cda41139f748d352da1e7ece1379353d08a082eb5534de.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/ae91e462ff99176e00cda41139f748d352da1e7ece1379353d08a082eb5534de.jpg)

![d5999e2c5f2bd5e46c83aa8df4dc6c24cd11c942227c1560496c5aac3354c612.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/d5999e2c5f2bd5e46c83aa8df4dc6c24cd11c942227c1560496c5aac3354c612.jpg)

![d6283ed8a8c95c83d8d7325294eca8496fa598256ad429ba8f5a6d17735e90f3.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/d6283ed8a8c95c83d8d7325294eca8496fa598256ad429ba8f5a6d17735e90f3.jpg)

![dc2d024218c44f729a0675a1067d00200cbbd58f3e0eba5c645eb72d5879efa3.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/dc2d024218c44f729a0675a1067d00200cbbd58f3e0eba5c645eb72d5879efa3.jpg)

![dc89ac8f286e00549ff0ecb6da267d8f425897015e59db246c1bb3658271c67e.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/dc89ac8f286e00549ff0ecb6da267d8f425897015e59db246c1bb3658271c67e.jpg)

![ec40d062ba81ce421b2f66ffa2ca037987e04ba0ea3e0cb33fed94e6241d784e.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/images/ec40d062ba81ce421b2f66ffa2ca037987e04ba0ea3e0cb33fed94e6241d784e.jpg)

### Tables

![80ae9a8df1e952f02c24e7b593ca1afa9e30cc7d0d4a6a875cd5ae3d337cfbe0.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/80ae9a8df1e952f02c24e7b593ca1afa9e30cc7d0d4a6a875cd5ae3d337cfbe0.jpg)

![8accfa5f3a7c730ae725351edf9cfbafc0ec0421273b992adc3a98431000cba8.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/8accfa5f3a7c730ae725351edf9cfbafc0ec0421273b992adc3a98431000cba8.jpg)

![8d11016bbf040aa93c280264e94613660f10b235ce244e8584c561bff2a60008.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/8d11016bbf040aa93c280264e94613660f10b235ce244e8584c561bff2a60008.jpg)

![962fbadc150c29c464a91fbb6c6dc70df78d04e178bbca69eb90afebf8f7e2a2.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/962fbadc150c29c464a91fbb6c6dc70df78d04e178bbca69eb90afebf8f7e2a2.jpg)

![a0c71773c1e72f55fd4b7d87042af48cfbf4ca4f9f70caac96f7418777052551.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/a0c71773c1e72f55fd4b7d87042af48cfbf4ca4f9f70caac96f7418777052551.jpg)

![b655eab7ff36671590be01a3f1ab2ce604f4e8f2835cfe89e14255e742dbdeae.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/b655eab7ff36671590be01a3f1ab2ce604f4e8f2835cfe89e14255e742dbdeae.jpg)

![c1314572bed62a586787890747dae08eb93052d4f1ec0af680994d5e62687731.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/c1314572bed62a586787890747dae08eb93052d4f1ec0af680994d5e62687731.jpg)

![d395b140bdd1d7d78554e6d41ba7c95ca40d404f9405995e02b3f2fb56b1b500.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/d395b140bdd1d7d78554e6d41ba7c95ca40d404f9405995e02b3f2fb56b1b500.jpg)

![ed7db4a875184f17f54c09297cec685afcaf13ab775f6bd57de9d8c3f9e8415c.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/ed7db4a875184f17f54c09297cec685afcaf13ab775f6bd57de9d8c3f9e8415c.jpg)

![f7d5a1bf57bb681b61a9d65d5eea7501ed4a290996bef1df411d9792b0ffb0ad.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/f7d5a1bf57bb681b61a9d65d5eea7501ed4a290996bef1df411d9792b0ffb0ad.jpg)

![fe664ab2d51cf2261037d7e7bba4037db350b03b19ff1a00deb95295a6cd25b7.jpg](../icml_results/4588_From%20Per-Image%20Low-Rank%20to%20Encoding%20Mismatch_%20Rethinking%20Feature%20Distillation%20in%20Vision%20Transformers/tables/fe664ab2d51cf2261037d7e7bba4037db350b03b19ff1a00deb95295a6cd25b7.jpg)

<a id="paper-4542"></a>

## Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels


### Images

![03ae4ae06c4276323025c22397ae54475bee296b4eb45593a153458e293d4ae7.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/03ae4ae06c4276323025c22397ae54475bee296b4eb45593a153458e293d4ae7.jpg)

![08d37fbb6dba540f34f5c0e83e43173b51509659033ffc152faf533d6a86493a.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/08d37fbb6dba540f34f5c0e83e43173b51509659033ffc152faf533d6a86493a.jpg)

![1ee346c8b1bf238f6094e866b6902a5d92a8027d2d775c1cf9e43641ed448967.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/1ee346c8b1bf238f6094e866b6902a5d92a8027d2d775c1cf9e43641ed448967.jpg)

![22458e1cca9748f1b78715b78e8d12395538fc1f9fa8dd27b2ab59112a083ca2.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/22458e1cca9748f1b78715b78e8d12395538fc1f9fa8dd27b2ab59112a083ca2.jpg)

![3331d70dffe55b2c2edc34954958cc0c0adc044ed2b49dc9f4bb532852b44d1f.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/3331d70dffe55b2c2edc34954958cc0c0adc044ed2b49dc9f4bb532852b44d1f.jpg)

![524260e81761505ef9e6ce25f2dcc20ff108e1dcb5746934f80aa32006dcbeef.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/524260e81761505ef9e6ce25f2dcc20ff108e1dcb5746934f80aa32006dcbeef.jpg)

![5622e6df8efb7ca881695f888feb6021a20f0c8b9fa467ba5a1050e844cb4cc6.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/5622e6df8efb7ca881695f888feb6021a20f0c8b9fa467ba5a1050e844cb4cc6.jpg)

![6707b735206c212acda706df6e7a9694fbf3e7b630edb733ec8f49de6d70bed2.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/6707b735206c212acda706df6e7a9694fbf3e7b630edb733ec8f49de6d70bed2.jpg)

![6ebcfb163e8d6937689e32c8e9c7041fd430181adcbe91c1d928a6706628faa3.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/6ebcfb163e8d6937689e32c8e9c7041fd430181adcbe91c1d928a6706628faa3.jpg)

![704a9ca2e96483d74a39dada40bfb09ffa7c562a5c385a96243841c37eebdd19.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/704a9ca2e96483d74a39dada40bfb09ffa7c562a5c385a96243841c37eebdd19.jpg)

![8a1ca21b301b8612361e7354402f8f4cf665f397d48621028d8d3cb0d30d60c4.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/8a1ca21b301b8612361e7354402f8f4cf665f397d48621028d8d3cb0d30d60c4.jpg)

![8da25a72bd9bbde60a2cdedd699713c36020659e766e46d1cc8e56aef722ea51.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/8da25a72bd9bbde60a2cdedd699713c36020659e766e46d1cc8e56aef722ea51.jpg)

![9c8c6c2e5639cd4992660466256ce253ca8ae9ea3dbc64c53531e5fc2fdd65b9.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/9c8c6c2e5639cd4992660466256ce253ca8ae9ea3dbc64c53531e5fc2fdd65b9.jpg)

![9dd5be7ef1b98b65cc5d38d3988a8b0bd2527c1ba108612b89a7fa1fdf001f19.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/9dd5be7ef1b98b65cc5d38d3988a8b0bd2527c1ba108612b89a7fa1fdf001f19.jpg)

![cff83cd84d780bf1f5653552891afa2651bdfd848aef54476d2cb1ab0ac8c56e.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/cff83cd84d780bf1f5653552891afa2651bdfd848aef54476d2cb1ab0ac8c56e.jpg)

![d1daa1f66d3147a90856c6bcf962ba671c16df1c9a5ef7e467fb737f789a8eb1.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/d1daa1f66d3147a90856c6bcf962ba671c16df1c9a5ef7e467fb737f789a8eb1.jpg)

![dc48bcfd24f1b0a48ee895da0423fca0d0746478d663e1b4b71f4f287f9cef26.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/dc48bcfd24f1b0a48ee895da0423fca0d0746478d663e1b4b71f4f287f9cef26.jpg)

![f511539d3c6fb90f4b128bc58b270bc898195380e6879a56dde4748f3f8d87be.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/f511539d3c6fb90f4b128bc58b270bc898195380e6879a56dde4748f3f8d87be.jpg)

![f8447d4e6331617ea330fa0d8b7d4fb5418e2d576d7453aa792ff467f53a6434.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/f8447d4e6331617ea330fa0d8b7d4fb5418e2d576d7453aa792ff467f53a6434.jpg)

![fe70a0de035aee4413b03a6535878afb0d640060d8fa9a48e0a3526d864c5068.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/fe70a0de035aee4413b03a6535878afb0d640060d8fa9a48e0a3526d864c5068.jpg)

![ff953b9979ff0be86f21d78c3253e38974e1da5ed5e880d63d37782d61373887.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/ff953b9979ff0be86f21d78c3253e38974e1da5ed5e880d63d37782d61373887.jpg)

![ff9ea46681b385ee7966a0fdee3cd3548e694766f63497fc14861604e4f320de.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/images/ff9ea46681b385ee7966a0fdee3cd3548e694766f63497fc14861604e4f320de.jpg)

### Tables

![036ee1d0b3eec93e55663859fc18027953933fed908e7728920aa7729ccb6015.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/036ee1d0b3eec93e55663859fc18027953933fed908e7728920aa7729ccb6015.jpg)

![09637355e618e022951ba47e67ea4f65f4e5be21af6526885b9af3f69401f3e7.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/09637355e618e022951ba47e67ea4f65f4e5be21af6526885b9af3f69401f3e7.jpg)

![159380fe5922ce783b51673a7540a8d7a2f909b2478d1ecc5a8abc9f787e2151.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/159380fe5922ce783b51673a7540a8d7a2f909b2478d1ecc5a8abc9f787e2151.jpg)

![23146446ba2daa88d79c3b5c1e078cc665c6ff1931a7ea98decc5a3860815d7a.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/23146446ba2daa88d79c3b5c1e078cc665c6ff1931a7ea98decc5a3860815d7a.jpg)

![298700184639cf9c7ee006c21a39b157629e47c0f51c812122a59e5285b3fd1e.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/298700184639cf9c7ee006c21a39b157629e47c0f51c812122a59e5285b3fd1e.jpg)

![305e0fc0fd0f34100612881e7b7801075d9688969d9d178e8831cbec0b5d515c.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/305e0fc0fd0f34100612881e7b7801075d9688969d9d178e8831cbec0b5d515c.jpg)

![46281f8531f8d230b19b55b7d2e8f096e7fcd0fda8ba0afe2ed98b58b8ec6a38.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/46281f8531f8d230b19b55b7d2e8f096e7fcd0fda8ba0afe2ed98b58b8ec6a38.jpg)

![580b4cf7c3642d5eb7680046ccebb3664cca2b3d6d48a2c3861ba744ef124fb9.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/580b4cf7c3642d5eb7680046ccebb3664cca2b3d6d48a2c3861ba744ef124fb9.jpg)

![5b311be8985d9a9aac5eee06c57a695a137d6935ef6ec5cb00b54f90266e7d2e.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/5b311be8985d9a9aac5eee06c57a695a137d6935ef6ec5cb00b54f90266e7d2e.jpg)

![67d55903de7b4dd781c845c7e051656eb5cf7d289f83b41ca67abd2d252686dd.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/67d55903de7b4dd781c845c7e051656eb5cf7d289f83b41ca67abd2d252686dd.jpg)

![7e65741997684052295fd94451984a743252d0257bbd2137f53b64188976d539.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/7e65741997684052295fd94451984a743252d0257bbd2137f53b64188976d539.jpg)

![b04746dde6f11ff7347319572639be328ae70571f12687f7fd835f0b7dea07f2.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/b04746dde6f11ff7347319572639be328ae70571f12687f7fd835f0b7dea07f2.jpg)

![b3889d14a1f1301593fb770483574491f19871e3ca4fb181b456fcc977cc6c61.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/b3889d14a1f1301593fb770483574491f19871e3ca4fb181b456fcc977cc6c61.jpg)

![bc79a4c4858f414d357b737e33bb543c0372efa7398ee11ef4cc1b9f20418849.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/bc79a4c4858f414d357b737e33bb543c0372efa7398ee11ef4cc1b9f20418849.jpg)

![c5c8ae03ad1e2c965cfd72f8065df6513675bf4ee2ed191574d66a2f1508126d.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/c5c8ae03ad1e2c965cfd72f8065df6513675bf4ee2ed191574d66a2f1508126d.jpg)

![e082932d8a39525cb370123f0079f9c2c1fce4b0a22ddf9957ec706449f93953.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/e082932d8a39525cb370123f0079f9c2c1fce4b0a22ddf9957ec706449f93953.jpg)

![e85e1701a76bfe252c88182f45527bf968da2b0038b0a47c2649f6c5d9d6609d.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/e85e1701a76bfe252c88182f45527bf968da2b0038b0a47c2649f6c5d9d6609d.jpg)

![ec600c02f40302fae7f82538e6c7aa846ec7919b9233d42e66b1e5f02fea68e2.jpg](../icml_results/4589_Learning%20Locally%2C%20Revising%20Globally_%20Global%20Reviser%20for%20Federated%20Learning%20with%20Noisy%20Labels/tables/ec600c02f40302fae7f82538e6c7aa846ec7919b9233d42e66b1e5f02fea68e2.jpg)

<a id="paper-4543"></a>

## LogicSAGE: Neuro-Symbolic Reasoning with Socratic-Guided Enhancement


### Images

![19650b0783445129e53dd0709befc03ae98895ee5d359fc55fea3bdc10e5b938.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/images/19650b0783445129e53dd0709befc03ae98895ee5d359fc55fea3bdc10e5b938.jpg)

![82832aff94c396adf5aca22af63562071343e189485643d4f3e1496b35e0a137.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/images/82832aff94c396adf5aca22af63562071343e189485643d4f3e1496b35e0a137.jpg)

![ad7285fd7ad35a0664f09a17ded462d81adca7cf498689fbd85910385bdbff41.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/images/ad7285fd7ad35a0664f09a17ded462d81adca7cf498689fbd85910385bdbff41.jpg)

![bf6e7048ab4849bbdb32d4239f0b9b056238c22ace2c0a87bf31026ce314c5a2.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/images/bf6e7048ab4849bbdb32d4239f0b9b056238c22ace2c0a87bf31026ce314c5a2.jpg)

### Tables

![0239e34099aae5da0a9e58e00bc1f4c4fece52ded08b28a43ed8cd37a35ce8ac.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/tables/0239e34099aae5da0a9e58e00bc1f4c4fece52ded08b28a43ed8cd37a35ce8ac.jpg)

![67af566aa3a536066997d183f4cd9bc06f0360b5864716ff5e2aaae867b0b23f.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/tables/67af566aa3a536066997d183f4cd9bc06f0360b5864716ff5e2aaae867b0b23f.jpg)

![790263e5b053d7432899374d5c5ec4f73aa79ec6ebecddf541b701fd1ad285b5.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/tables/790263e5b053d7432899374d5c5ec4f73aa79ec6ebecddf541b701fd1ad285b5.jpg)

![afa3add3f041ebc8ece32df6f43976e088dccf89c9d555f7b317f53949c3db84.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/tables/afa3add3f041ebc8ece32df6f43976e088dccf89c9d555f7b317f53949c3db84.jpg)

![c413352d468428f6b848c15de521d3215dc1295f40ac16ab99c2749c74cbde6b.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/tables/c413352d468428f6b848c15de521d3215dc1295f40ac16ab99c2749c74cbde6b.jpg)

![f6b3db7366cb7d50ded7a7b1a208383caed331bf6639501ffcd7d74c1c573fd7.jpg](../icml_results/4590_LogicSAGE_%20Neuro-Symbolic%20Reasoning%20with%20Socratic-Guided%20Enhancement/tables/f6b3db7366cb7d50ded7a7b1a208383caed331bf6639501ffcd7d74c1c573fd7.jpg)

<a id="paper-4544"></a>

## LOTTERY: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry


### Images

![4199c8e0238ae67618c3894931e83d5805e86955bb0c1dae963e209da094add2.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/images/4199c8e0238ae67618c3894931e83d5805e86955bb0c1dae963e209da094add2.jpg)

### Tables

![02b18bc1b641e066d48ef9fee120508e773a043f8cfa4778f628ee4c6a313518.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/02b18bc1b641e066d48ef9fee120508e773a043f8cfa4778f628ee4c6a313518.jpg)

![083850e2521021bce90da9893e6055ac04b5a9ac0aa22c0f662fb6b38aba1e62.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/083850e2521021bce90da9893e6055ac04b5a9ac0aa22c0f662fb6b38aba1e62.jpg)

![0d4ce0ca2088bc869622d9b16314153049f88a978fc67e99c43278dd6d757b0d.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/0d4ce0ca2088bc869622d9b16314153049f88a978fc67e99c43278dd6d757b0d.jpg)

![39186a418774b4b874e74bb73d204e645695923c1573ff2bce22e450aef74abf.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/39186a418774b4b874e74bb73d204e645695923c1573ff2bce22e450aef74abf.jpg)

![a89905cbf59d300cf1a0bb04b98444503c4ae17e167135b5591581af6068ccfc.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/a89905cbf59d300cf1a0bb04b98444503c4ae17e167135b5591581af6068ccfc.jpg)

![bc42a1e758727bfa0e772e39d995cbef4a684ae284c2300d0c3aff6f3b6036df.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/bc42a1e758727bfa0e772e39d995cbef4a684ae284c2300d0c3aff6f3b6036df.jpg)

![ffeef08528b449ef5090d9e8dda50a601dd54390e2e15000540e68e90fee67df.jpg](../icml_results/4591_LOTTERY_%20Learning%20from%20Reference-Only%20Samples%20in%20Two-Sample%20Testing%20under%20Size%20Asymmetry/tables/ffeef08528b449ef5090d9e8dda50a601dd54390e2e15000540e68e90fee67df.jpg)

<a id="paper-4545"></a>

## Routing and Reasoned Evaluation with Large Language Models


### Images

![01a73a1006308820dd2a839f83bafc49d1b56ac2a735ee4e3732b8f35c12d306.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/01a73a1006308820dd2a839f83bafc49d1b56ac2a735ee4e3732b8f35c12d306.jpg)

![48964467950984deb9a0762be78b467cf3b427e3292425f0df70815f6b5c4760.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/48964467950984deb9a0762be78b467cf3b427e3292425f0df70815f6b5c4760.jpg)

![632460659b683c810fc674f89b0d2d0d62dba1534c89b0481f73031810315940.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/632460659b683c810fc674f89b0d2d0d62dba1534c89b0481f73031810315940.jpg)

![7ad52ecb93d01d820dfce76d7cd6c733fddcecf1892f6580dc672e9b3c460dba.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/7ad52ecb93d01d820dfce76d7cd6c733fddcecf1892f6580dc672e9b3c460dba.jpg)

![8759d266fafb56a76651a4b75b986ec26ca6cb92d3f5fdc9d5f307d74be7451f.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/8759d266fafb56a76651a4b75b986ec26ca6cb92d3f5fdc9d5f307d74be7451f.jpg)

![b829531eee7379c460eabbd7687302cbadbf18f9ebffaeff9da3eed54cc72f78.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/b829531eee7379c460eabbd7687302cbadbf18f9ebffaeff9da3eed54cc72f78.jpg)

![e4b451353592fd205df733c75f0479418f7192f1e7ed7a94e100cf3d89f4ccac.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/images/e4b451353592fd205df733c75f0479418f7192f1e7ed7a94e100cf3d89f4ccac.jpg)

### Tables

![0374a4abcff2bbbf3c3e0b32707da583fdecd7b439d6cffc9d741fcbdfd396a8.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/0374a4abcff2bbbf3c3e0b32707da583fdecd7b439d6cffc9d741fcbdfd396a8.jpg)

![04cf5ec1c5ee2d0ef259ce4c89c1fd623516e0fa5daf8b1cd4daf55b68ec7c43.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/04cf5ec1c5ee2d0ef259ce4c89c1fd623516e0fa5daf8b1cd4daf55b68ec7c43.jpg)

![15a68fef9e28ab4e6a810f86813c12f94ec692f36829b159947b0d88415b7f43.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/15a68fef9e28ab4e6a810f86813c12f94ec692f36829b159947b0d88415b7f43.jpg)

![15cde27af3d60448ecf9dfa844db6168ca0b173976df7fa05b56c6485c3db4b0.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/15cde27af3d60448ecf9dfa844db6168ca0b173976df7fa05b56c6485c3db4b0.jpg)

![199855f1e2f04cb424f311f4cecfa2b3536d96ea4cca1c2aa2340efda843c649.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/199855f1e2f04cb424f311f4cecfa2b3536d96ea4cca1c2aa2340efda843c649.jpg)

![20b0a9ceccc7ab930e21bd773ec5b383255361e231b5401c30c2e0e59d0718c5.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/20b0a9ceccc7ab930e21bd773ec5b383255361e231b5401c30c2e0e59d0718c5.jpg)

![22d03c23b3e116faed1e735dd45122e87061b8949a070750b191fadb4d2c29f8.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/22d03c23b3e116faed1e735dd45122e87061b8949a070750b191fadb4d2c29f8.jpg)

![245a6df6ecff8d4d357ca62ae89be9ed30a1703baa5cabafa0da8aaead6442ba.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/245a6df6ecff8d4d357ca62ae89be9ed30a1703baa5cabafa0da8aaead6442ba.jpg)

![489269205aad931c104de71c2f40ccf5d96a475961330ebb792427e04d9494ac.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/489269205aad931c104de71c2f40ccf5d96a475961330ebb792427e04d9494ac.jpg)

![5102e3cffa8caaa326e221a34a64d8959c5148262ae12ad74478e8d721ec244f.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/5102e3cffa8caaa326e221a34a64d8959c5148262ae12ad74478e8d721ec244f.jpg)

![54bc7ef4601fe144c5cdfc11c77d77f27ffe8caeab5262b278cfaf12b7ea5890.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/54bc7ef4601fe144c5cdfc11c77d77f27ffe8caeab5262b278cfaf12b7ea5890.jpg)

![7bf2acca4da1e8c50a01df98efc0d950515848dd15dd9acf27ee257b16dba4af.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/7bf2acca4da1e8c50a01df98efc0d950515848dd15dd9acf27ee257b16dba4af.jpg)

![b4d3ff9f1ffeb4ca8ede714adfb5dcae9c3040d1bbe327ea944ebc4527e77d95.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/b4d3ff9f1ffeb4ca8ede714adfb5dcae9c3040d1bbe327ea944ebc4527e77d95.jpg)

![b6f1c1c48346ec15e571088b411b5f13335de459c76f20960e106d63ffbf20cc.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/b6f1c1c48346ec15e571088b411b5f13335de459c76f20960e106d63ffbf20cc.jpg)

![d42d4fbe70f5a624ee7fd0927fbc585e3b6648ea63e434d4f36aee74132c2380.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/d42d4fbe70f5a624ee7fd0927fbc585e3b6648ea63e434d4f36aee74132c2380.jpg)

![e328566205d191afd32b50e0d244d2c74eb2ae58540f9ba19131e1c7848067af.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/e328566205d191afd32b50e0d244d2c74eb2ae58540f9ba19131e1c7848067af.jpg)

![eebcb94b1f3140324cb017b06ebc1aaf131fd09efbe0a2b6bfe41abb708ca1ae.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/eebcb94b1f3140324cb017b06ebc1aaf131fd09efbe0a2b6bfe41abb708ca1ae.jpg)

![fbeaca8d2fcb0e72135e9c52f244606f81269ea423a3b3eb3de37b4b4493fecf.jpg](../icml_results/4592_Routing%20and%20Reasoned%20Evaluation%20with%20Large%20Language%20Models/tables/fbeaca8d2fcb0e72135e9c52f244606f81269ea423a3b3eb3de37b4b4493fecf.jpg)

<a id="paper-4546"></a>

## Joint Model and Data Sparsification via the Marginal Likelihood


### Images

![1a425ed0ea5cb8925846f89ddf8ed7ac828a67cc11f2c5bb21b500d0b8c32d3a.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/1a425ed0ea5cb8925846f89ddf8ed7ac828a67cc11f2c5bb21b500d0b8c32d3a.jpg)

![2f043a5235e52e772f5e4911e56043adc6ddd59a92f8e94466ffe40b22ef4bf4.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/2f043a5235e52e772f5e4911e56043adc6ddd59a92f8e94466ffe40b22ef4bf4.jpg)

![3b004d3f8b01037b241d552d7ddcf92ad20eed49cf6cff29ee2f20adbbe343a0.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/3b004d3f8b01037b241d552d7ddcf92ad20eed49cf6cff29ee2f20adbbe343a0.jpg)

![4afb0636247b06c4495115143321a47078a591dba01b1df83f86e8e669bdca4e.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/4afb0636247b06c4495115143321a47078a591dba01b1df83f86e8e669bdca4e.jpg)

![b22a243aa14a8e09b6cfcc063cfedb52250c2b261a9103cd2a2547e2be4845b0.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/b22a243aa14a8e09b6cfcc063cfedb52250c2b261a9103cd2a2547e2be4845b0.jpg)

![b51966d2ecc558ccbd0941b951014431cab0f8ac2fd78bc840a4ff17ac02155c.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/b51966d2ecc558ccbd0941b951014431cab0f8ac2fd78bc840a4ff17ac02155c.jpg)

![ba2d9aa5a4babf74e33dfa3dea8546a9c2b459c5790372120c7fb633a8707700.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/ba2d9aa5a4babf74e33dfa3dea8546a9c2b459c5790372120c7fb633a8707700.jpg)

![cc01d5cba84ee4918628956c002bd4c6b125f5a774dac954176005ee0c351d35.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/images/cc01d5cba84ee4918628956c002bd4c6b125f5a774dac954176005ee0c351d35.jpg)

### Tables

![057cee546cee4db8711f00f877b43cd2efccb1f70dafa54dba866458484d9144.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/057cee546cee4db8711f00f877b43cd2efccb1f70dafa54dba866458484d9144.jpg)

![0d739662ad2e9bf7130e798f5dcb200e574fe431586ba1a0e0251903ba63f95b.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/0d739662ad2e9bf7130e798f5dcb200e574fe431586ba1a0e0251903ba63f95b.jpg)

![1bd2f70c621222a048d05afb8a4745b86a42be18b98802a18e4d531bf1805acb.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/1bd2f70c621222a048d05afb8a4745b86a42be18b98802a18e4d531bf1805acb.jpg)

![2c7832b9a54be9ec6f4e0f4ca05311615d36afdea81c63d2d519094d046f4480.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/2c7832b9a54be9ec6f4e0f4ca05311615d36afdea81c63d2d519094d046f4480.jpg)

![408c354ea1b48bc41687cfa382cff90253ca2adad97f57b15dd8bc0e608f60a4.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/408c354ea1b48bc41687cfa382cff90253ca2adad97f57b15dd8bc0e608f60a4.jpg)

![5ce9ddcce9fb1fd90399800d5796d2118caa9d0467972917176103dbc7f53a5f.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/5ce9ddcce9fb1fd90399800d5796d2118caa9d0467972917176103dbc7f53a5f.jpg)

![acb3b25dd1c4d278a4e94c14bbffac07b5e2c6a7fb2d5ecacfcdd33046d8e6c3.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/acb3b25dd1c4d278a4e94c14bbffac07b5e2c6a7fb2d5ecacfcdd33046d8e6c3.jpg)

![b67fd625bc3a3a596a0181d98c305dbbcba9426525b9a72b2f378e310eb10713.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/b67fd625bc3a3a596a0181d98c305dbbcba9426525b9a72b2f378e310eb10713.jpg)

![b85dfb717787d62fdc867594c92cff94f67d742edb15dd1fe9319c9f0dac582a.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/b85dfb717787d62fdc867594c92cff94f67d742edb15dd1fe9319c9f0dac582a.jpg)

![d95eb4a14a59dccc88818782d265d60d07dce75c44522b6db1e21406d4ae497c.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/d95eb4a14a59dccc88818782d265d60d07dce75c44522b6db1e21406d4ae497c.jpg)

![e8e6c4a43ccdeac723afa1961234b8a85918d2057383b36bfd3cb6d7b34878d5.jpg](../icml_results/4593_Joint%20Model%20and%20Data%20Sparsification%20via%20the%20Marginal%20Likelihood/tables/e8e6c4a43ccdeac723afa1961234b8a85918d2057383b36bfd3cb6d7b34878d5.jpg)

<a id="paper-4547"></a>

## Content-Style Identification via Differential Independence


### Images

![1cc8408d6da733569a4c40a5b64398dbfc816572d6ef1c2e953733bc87b064a0.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/1cc8408d6da733569a4c40a5b64398dbfc816572d6ef1c2e953733bc87b064a0.jpg)

![264a5c6178b9c5dfa704a9a24689ae00834e22b103d454b4645a6e545596bdf4.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/264a5c6178b9c5dfa704a9a24689ae00834e22b103d454b4645a6e545596bdf4.jpg)

![2e5dafd43091f8f9f392abd8c4121aaf11f1fc4f2146e460c06a7bf2ea2e9d2a.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/2e5dafd43091f8f9f392abd8c4121aaf11f1fc4f2146e460c06a7bf2ea2e9d2a.jpg)

![4cb5ab62cb2c9acabc189354c230fe89248e914fa1c62c5dcb43bbeca324f7f5.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/4cb5ab62cb2c9acabc189354c230fe89248e914fa1c62c5dcb43bbeca324f7f5.jpg)

![673c1b0ffe3e7b08008236a3e28c4cc0a9e60c9faf2272740d0de2cb493960bf.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/673c1b0ffe3e7b08008236a3e28c4cc0a9e60c9faf2272740d0de2cb493960bf.jpg)

![79442ee8744070f0cc4a4397d8c8e155de7d67f87fabf711a1c47e7cc35d5e4d.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/79442ee8744070f0cc4a4397d8c8e155de7d67f87fabf711a1c47e7cc35d5e4d.jpg)

![7c7c42319cabfbb97bb88c920055456bfaf973d1a3115c2587f7ebf04fcac0a0.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/7c7c42319cabfbb97bb88c920055456bfaf973d1a3115c2587f7ebf04fcac0a0.jpg)

![96e9a9ade19eb92960d46d4644ee572201ed8cef563aaa186035e482b9e2fa97.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/96e9a9ade19eb92960d46d4644ee572201ed8cef563aaa186035e482b9e2fa97.jpg)

![a67dd06cb974edc1bc715052ef67cf871df841eeea830c6f509acab21c480228.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/a67dd06cb974edc1bc715052ef67cf871df841eeea830c6f509acab21c480228.jpg)

![b62968a320270cab097732b999ff6a85370b23a88b1cdd9b7d01ec70e55ec282.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/b62968a320270cab097732b999ff6a85370b23a88b1cdd9b7d01ec70e55ec282.jpg)

![c3cf035e47ee75e19bb02a8443ad17d80870b6685c52906233fa8a6f63dcd33c.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/c3cf035e47ee75e19bb02a8443ad17d80870b6685c52906233fa8a6f63dcd33c.jpg)

![c623127abcb27bc8d026eed9fe0f7569dc92bd7fcd6b12d45007fbab574b4c0d.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/c623127abcb27bc8d026eed9fe0f7569dc92bd7fcd6b12d45007fbab574b4c0d.jpg)

![c8e2e92ab9e409351d1f909dd0cc2e542e34d0448e62700ef79e1dc66fd808f5.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/c8e2e92ab9e409351d1f909dd0cc2e542e34d0448e62700ef79e1dc66fd808f5.jpg)

![d8e89322f3316b2376c9ba88480ab1b25df44a151e5f0064c7efd4d048e10b0c.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/d8e89322f3316b2376c9ba88480ab1b25df44a151e5f0064c7efd4d048e10b0c.jpg)

![dd4e0b7db53b5c2c56f5bc3c7215cfde90e40ab7b182831d2f845f3c09291d5e.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/images/dd4e0b7db53b5c2c56f5bc3c7215cfde90e40ab7b182831d2f845f3c09291d5e.jpg)

### Tables

![15f813c2214235f3fa728bc30babf4f7400a998b5ed1b51f36eab2e71572be50.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/15f813c2214235f3fa728bc30babf4f7400a998b5ed1b51f36eab2e71572be50.jpg)

![3747575c8d8be75cdf03227b52b77819eee1a8c983bd69a3861b703cea8e244c.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/3747575c8d8be75cdf03227b52b77819eee1a8c983bd69a3861b703cea8e244c.jpg)

![50233fbee6f041e57fa103f8f87492066068fd210e98f4800960cb8d154d2e6a.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/50233fbee6f041e57fa103f8f87492066068fd210e98f4800960cb8d154d2e6a.jpg)

![7d5c5ba77a676b46584abe7d97186182438d363ca8cef547027765b96d9022a2.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/7d5c5ba77a676b46584abe7d97186182438d363ca8cef547027765b96d9022a2.jpg)

![9600898ca5c4b6ffbb48e894b9e0b62f2beea1ef3aed3c7d3a8c7b5d58a0eb84.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/9600898ca5c4b6ffbb48e894b9e0b62f2beea1ef3aed3c7d3a8c7b5d58a0eb84.jpg)

![b646fc66f2507312d7e5f97840a98ef57bafe55a00c41e8b61661a13aa0653c8.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/b646fc66f2507312d7e5f97840a98ef57bafe55a00c41e8b61661a13aa0653c8.jpg)

![b87b6765299d2a547af68826884bc2158d7d81497857788a693b751e95356bab.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/b87b6765299d2a547af68826884bc2158d7d81497857788a693b751e95356bab.jpg)

![e712bd26220c3ef5a2ee391355e90f5d0b6d3f6a2bf6e999381b5af37c2a6d91.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/e712bd26220c3ef5a2ee391355e90f5d0b6d3f6a2bf6e999381b5af37c2a6d91.jpg)

![f5a8a7d4ef07b23ca5cbf06ab492a36e4087071a6c16d3cd733737c1e39d3c7e.jpg](../icml_results/4594_Content-Style%20Identification%20via%20Differential%20Independence/tables/f5a8a7d4ef07b23ca5cbf06ab492a36e4087071a6c16d3cd733737c1e39d3c7e.jpg)

<a id="paper-4548"></a>

## SVL: Goal-Conditioned Reinforcement Learning as Survival Learning


### Images

![982446d78e80292b4fb08af91bb67e69f97d6f1d16523d8b9b821ef8394e9e6f.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/images/982446d78e80292b4fb08af91bb67e69f97d6f1d16523d8b9b821ef8394e9e6f.jpg)

![d18100e3b3e068d5c4a86aa8ee1a293d04e2993287e34c8f1e93ae7f380adaad.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/images/d18100e3b3e068d5c4a86aa8ee1a293d04e2993287e34c8f1e93ae7f380adaad.jpg)

![f4a308bb663b6479bd0865ab3964f989e6efc92e80e5df4fb41fbc136e550f6c.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/images/f4a308bb663b6479bd0865ab3964f989e6efc92e80e5df4fb41fbc136e550f6c.jpg)

![fe6ee076d4c1020ac756cc02fd97c00f43f410d2c3f51c9432fbc21b2507a6b5.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/images/fe6ee076d4c1020ac756cc02fd97c00f43f410d2c3f51c9432fbc21b2507a6b5.jpg)

### Tables

![61e3623ef7ba9f33c4cc2ae49e902ce30c3f1f67c1ae042264e1ebbc671b8654.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/61e3623ef7ba9f33c4cc2ae49e902ce30c3f1f67c1ae042264e1ebbc671b8654.jpg)

![7745d7e21eeca2215432d4f0a91067e98d2e73cdc39c3e35b7a467de0ca5a3ce.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/7745d7e21eeca2215432d4f0a91067e98d2e73cdc39c3e35b7a467de0ca5a3ce.jpg)

![8728c4e8d7f997d1c8e90cf5b65b0a4b53f027f64d1a21fa5c6c42e047ed5d4f.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/8728c4e8d7f997d1c8e90cf5b65b0a4b53f027f64d1a21fa5c6c42e047ed5d4f.jpg)

![886e0635cd0bcd088949b28462a03b329932c762e949629953bdfca15e1e2c51.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/886e0635cd0bcd088949b28462a03b329932c762e949629953bdfca15e1e2c51.jpg)

![8a83ed2bc5839b4d0a54390626c2c11a30bbb6e2145ad8e67d880546a01a46f0.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/8a83ed2bc5839b4d0a54390626c2c11a30bbb6e2145ad8e67d880546a01a46f0.jpg)

![b098e6c43bd1d2e1dac77980faddcafe3cf312f0ddd5406f20c28b620ee30875.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/b098e6c43bd1d2e1dac77980faddcafe3cf312f0ddd5406f20c28b620ee30875.jpg)

![b955f2b67890595442429955e71880b7dc80b057b506651e15b831581be00509.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/b955f2b67890595442429955e71880b7dc80b057b506651e15b831581be00509.jpg)

![c3613edf070dd9455720b4c72b6e4ac138facfad8aedb5e3f84040f627c70d33.jpg](../icml_results/4595_SVL_%20Goal-Conditioned%20Reinforcement%20Learning%20as%20Survival%20Learning/tables/c3613edf070dd9455720b4c72b6e4ac138facfad8aedb5e3f84040f627c70d33.jpg)

<a id="paper-4549"></a>

## Semi-Supervised Learning for Molecular Graphs via Ensemble Consensus


### Images

![34df7ef1db7f59820b43ae68627f040c4d95ebdc776d546dbb855a13be9e3b6b.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/34df7ef1db7f59820b43ae68627f040c4d95ebdc776d546dbb855a13be9e3b6b.jpg)

![506643939be6f5a8d23092676467e816943ec1d60229c1a7d4b9d2a339ae8be3.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/506643939be6f5a8d23092676467e816943ec1d60229c1a7d4b9d2a339ae8be3.jpg)

![5244358bf3f5bcb81e10fa0c942bc8a03a311bbd3b50b97d8e20e4f9476c2f35.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/5244358bf3f5bcb81e10fa0c942bc8a03a311bbd3b50b97d8e20e4f9476c2f35.jpg)

![6fd02f7dd372cef1d4f9b0625604b5f659d8566f3384a5f5c792bf4654a0fa61.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/6fd02f7dd372cef1d4f9b0625604b5f659d8566f3384a5f5c792bf4654a0fa61.jpg)

![72db6bedb44fa6339374ce9d606f1dc68a9bb392f07e587fa2540b1c6e15454d.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/72db6bedb44fa6339374ce9d606f1dc68a9bb392f07e587fa2540b1c6e15454d.jpg)

![74a56df2fab90562690ee9ed405e806788e3146382165c1201edebf8a05b1ce2.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/74a56df2fab90562690ee9ed405e806788e3146382165c1201edebf8a05b1ce2.jpg)

![786442516958dbeb6a603706c687596be02ecd523c981a7e9a1f07f075d7a2b8.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/786442516958dbeb6a603706c687596be02ecd523c981a7e9a1f07f075d7a2b8.jpg)

![800661f583e14c6250f3353fdec1687fdd03517cf94f5dd70e2ce826d6e6707d.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/800661f583e14c6250f3353fdec1687fdd03517cf94f5dd70e2ce826d6e6707d.jpg)

![91dae5ec0fbb29fd9840e19dbe75a47bef94deee30693442cb4befdc17eca036.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/91dae5ec0fbb29fd9840e19dbe75a47bef94deee30693442cb4befdc17eca036.jpg)

![a381cdcd1d0dac2d70bd6b7422a66994750d06694d1fc73bb19660125ae833e5.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/a381cdcd1d0dac2d70bd6b7422a66994750d06694d1fc73bb19660125ae833e5.jpg)

![b72bd008a79fcaceda05b31139efc14997c14ef90ec3a0e2f95ee7f8ae3cbadb.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/b72bd008a79fcaceda05b31139efc14997c14ef90ec3a0e2f95ee7f8ae3cbadb.jpg)

![bc09b8c1d6be11ab3599ee00ba901b15b231b35fd896f9ad78d325c5c5866541.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/bc09b8c1d6be11ab3599ee00ba901b15b231b35fd896f9ad78d325c5c5866541.jpg)

![bcfb743896afc3049115d350b334d01a7a00b7fad5d8a71aaa4b541721621ff5.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/bcfb743896afc3049115d350b334d01a7a00b7fad5d8a71aaa4b541721621ff5.jpg)

![fb690a3869fd94c4167c14e51d83e93ad5bec4634cf7f1ef4f403bf4f286a69c.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/images/fb690a3869fd94c4167c14e51d83e93ad5bec4634cf7f1ef4f403bf4f286a69c.jpg)

### Tables

![08015281628c23ea98169324258df52a675ba5cfde449472e356367ca7069c6d.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/08015281628c23ea98169324258df52a675ba5cfde449472e356367ca7069c6d.jpg)

![12fd618d1fa84ab1c0e3a573c820ae6b5a17d7a1856e679a12945f6c3291f0d6.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/12fd618d1fa84ab1c0e3a573c820ae6b5a17d7a1856e679a12945f6c3291f0d6.jpg)

![1420ad3d563c3249819869436ad0204e9d02e78a93a0190fd6d31e75735bcce4.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/1420ad3d563c3249819869436ad0204e9d02e78a93a0190fd6d31e75735bcce4.jpg)

![1d8a7345833789b2479a76f4c90e21a043c9737ca8b58afd4d06da8bb7800442.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/1d8a7345833789b2479a76f4c90e21a043c9737ca8b58afd4d06da8bb7800442.jpg)

![21a06d1ba26256fb97311e313c58f9d0c3f2679efbaa5edddd5f5e293a102969.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/21a06d1ba26256fb97311e313c58f9d0c3f2679efbaa5edddd5f5e293a102969.jpg)

![2d6e008bfa77980b55866fb052e698f5b31ddd4678da23700e783214442d6e2c.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/2d6e008bfa77980b55866fb052e698f5b31ddd4678da23700e783214442d6e2c.jpg)

![35e46033680ea7da8690a84452c68fb1cfd40c8f4c3e0dd8783d101c6fd25a39.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/35e46033680ea7da8690a84452c68fb1cfd40c8f4c3e0dd8783d101c6fd25a39.jpg)

![43c1be4bd0ffd44eb635c298381b99402403355ea0f915bd744546150da89d09.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/43c1be4bd0ffd44eb635c298381b99402403355ea0f915bd744546150da89d09.jpg)

![47a73d59adffcb77d5f0eeeaf19f679aae120d8f7791acf5fb2ec430020cee0a.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/47a73d59adffcb77d5f0eeeaf19f679aae120d8f7791acf5fb2ec430020cee0a.jpg)

![5d5b7a42846bb06ccd2700ec4458edc54fd69c33177d1230530ef8c430876fc4.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/5d5b7a42846bb06ccd2700ec4458edc54fd69c33177d1230530ef8c430876fc4.jpg)

![5df50cacee8715b71bf3e4acae30765cd30d1de1b56f196037be4a1117fddc2f.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/5df50cacee8715b71bf3e4acae30765cd30d1de1b56f196037be4a1117fddc2f.jpg)

![5f6e17764693c9f725a5a27dd49d8443b68de09e16494afb57a7db2a345a0b17.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/5f6e17764693c9f725a5a27dd49d8443b68de09e16494afb57a7db2a345a0b17.jpg)

![7b42f3b88890f6b11330de2bb8dd873063b793c1c721f23cf0c72e8fa85a7312.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/7b42f3b88890f6b11330de2bb8dd873063b793c1c721f23cf0c72e8fa85a7312.jpg)

![89d2a5420abff6aa07e9fd6c3235c4a06b03a291db92a67f54f177a949e9147c.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/89d2a5420abff6aa07e9fd6c3235c4a06b03a291db92a67f54f177a949e9147c.jpg)

![90360b101774c2439cc69030ee4b4b9ce8066b14102598a067efa75e597d309e.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/90360b101774c2439cc69030ee4b4b9ce8066b14102598a067efa75e597d309e.jpg)

![9c486c188aeb4122eefa10dec85e38c867f0c187b3d3ee6ac206c717435539df.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/9c486c188aeb4122eefa10dec85e38c867f0c187b3d3ee6ac206c717435539df.jpg)

![a1f284cbba9b036be4c437f63b46486316b14585b93c24aa17edce22ba2c3b10.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/a1f284cbba9b036be4c437f63b46486316b14585b93c24aa17edce22ba2c3b10.jpg)

![a8ce5882b4fb168046764b3f56ada3a1f3a8279162917041371152a6f60350ed.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/a8ce5882b4fb168046764b3f56ada3a1f3a8279162917041371152a6f60350ed.jpg)

![ad069695bd5135f27cae640b87693dea37efa9352c427cd2c836f776d827b50c.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/ad069695bd5135f27cae640b87693dea37efa9352c427cd2c836f776d827b50c.jpg)

![b1a2de7b291699b2082a43347f63a6a9e7211a5fe17d00e4e8ea8de8b2026d45.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/b1a2de7b291699b2082a43347f63a6a9e7211a5fe17d00e4e8ea8de8b2026d45.jpg)

![b7dda8a0ba805e17d9660f014cea81590abd2822a9cef8b5d22b9d8fad95a965.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/b7dda8a0ba805e17d9660f014cea81590abd2822a9cef8b5d22b9d8fad95a965.jpg)

![d1937a1c1489be96f6daeb72327e953a2db21f3a592dcffcdf1a50805833f138.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/d1937a1c1489be96f6daeb72327e953a2db21f3a592dcffcdf1a50805833f138.jpg)

![e3e07a0a37d1fa920c8b7b697c9b30c9375bb5948f9e7e48faff518d48ea5c7a.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/e3e07a0a37d1fa920c8b7b697c9b30c9375bb5948f9e7e48faff518d48ea5c7a.jpg)

![f0986c33877c1014c3f8c4f2d96d358ab34850652c6f7d5db5a892ddfffe9136.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/f0986c33877c1014c3f8c4f2d96d358ab34850652c6f7d5db5a892ddfffe9136.jpg)

![f09ffd16d758d0ce777d56ead2025e453a641f3c49c748c25f5c1e488f90f020.jpg](../icml_results/4596_Semi-Supervised%20Learning%20for%20Molecular%20Graphs%20via%20Ensemble%20Consensus/tables/f09ffd16d758d0ce777d56ead2025e453a641f3c49c748c25f5c1e488f90f020.jpg)

<a id="paper-4550"></a>

## Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning


### Images

![2b56e438c628e1a0185f842b7264a2e16815a4756f4ed5ab815deb76bebd070d.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/2b56e438c628e1a0185f842b7264a2e16815a4756f4ed5ab815deb76bebd070d.jpg)

![3774e5a652457917479e05d83782764f40bde7e5effda1e8c058e89a29c1f3c5.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/3774e5a652457917479e05d83782764f40bde7e5effda1e8c058e89a29c1f3c5.jpg)

![4c35c87455f00d547678ee286990f38baf3611fcd6187f6c7a5bff65c57d4eec.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/4c35c87455f00d547678ee286990f38baf3611fcd6187f6c7a5bff65c57d4eec.jpg)

![5727e530c871ba63e4bede6acf6a0e504f6ee7adf1d7566b068a692731818ba0.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/5727e530c871ba63e4bede6acf6a0e504f6ee7adf1d7566b068a692731818ba0.jpg)

![58aaeb2f01e448be2975f674d56f1d27e990d732eb39f7b42e3a38c68a7228e4.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/58aaeb2f01e448be2975f674d56f1d27e990d732eb39f7b42e3a38c68a7228e4.jpg)

![67f533ef1485e44d54c1ad7dd2dbcbeccd0e3f88c47bef4ebf3d40e16ba40576.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/67f533ef1485e44d54c1ad7dd2dbcbeccd0e3f88c47bef4ebf3d40e16ba40576.jpg)

![85f3005bf7b729ba0a26d5358617f2876dc4c9355f26b5c66475ad55e776d3ad.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/85f3005bf7b729ba0a26d5358617f2876dc4c9355f26b5c66475ad55e776d3ad.jpg)

![91ce1de3caf46b4a68c1b0c6d9991bf9dd932cbd34357dc72f8b4cc4a84cee63.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/91ce1de3caf46b4a68c1b0c6d9991bf9dd932cbd34357dc72f8b4cc4a84cee63.jpg)

![bfbb6effc9e91dd3d3e91e9f666c384e3c95f05e15ab32d4d66b5fad095f5cb5.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/bfbb6effc9e91dd3d3e91e9f666c384e3c95f05e15ab32d4d66b5fad095f5cb5.jpg)

![f3a548205dfb7428bd57132b80e14d01747894edad8cdf51c2e60582e0c04b89.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/images/f3a548205dfb7428bd57132b80e14d01747894edad8cdf51c2e60582e0c04b89.jpg)

### Tables

![011575bf5dcf0c858dc40cfd16e5a230d50f50056e0f6b486c2787b7a5fd28ec.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/011575bf5dcf0c858dc40cfd16e5a230d50f50056e0f6b486c2787b7a5fd28ec.jpg)

![0331de4da37d35174a25697be90d72c9810f959c6092e3802545382b3fa106da.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/0331de4da37d35174a25697be90d72c9810f959c6092e3802545382b3fa106da.jpg)

![0a71e7701d66f85edb7084f2526dc608dd590111c8ca33973d1d0ee2a0e094ec.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/0a71e7701d66f85edb7084f2526dc608dd590111c8ca33973d1d0ee2a0e094ec.jpg)

![16af730e5528dba4a3cd432360e28c2bffc8d4aed23b96fb931cd8ad375cebde.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/16af730e5528dba4a3cd432360e28c2bffc8d4aed23b96fb931cd8ad375cebde.jpg)

![216eeba7d033a6315e380b12148fca8cfdc98c068e869389385778c143acb284.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/216eeba7d033a6315e380b12148fca8cfdc98c068e869389385778c143acb284.jpg)

![298dd81e94ba0e7fbb6e31cdfe1c14a431e060a380c81f4264ebba3649ec9a9e.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/298dd81e94ba0e7fbb6e31cdfe1c14a431e060a380c81f4264ebba3649ec9a9e.jpg)

![3179b94b5f14114e978473b45d2f8ca06369dad33beee48120bb5a25d2b0cf8c.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/3179b94b5f14114e978473b45d2f8ca06369dad33beee48120bb5a25d2b0cf8c.jpg)

![49031ab6820b4e2a1e5fd8a29877a84e3738adee06f90c717bdffb35dc7e6d54.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/49031ab6820b4e2a1e5fd8a29877a84e3738adee06f90c717bdffb35dc7e6d54.jpg)

![59536f190bb69e47cabb0e323dd439890e594c00b128623c2a4063fc8df0c031.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/59536f190bb69e47cabb0e323dd439890e594c00b128623c2a4063fc8df0c031.jpg)

![7518be722f0d52cbacb846489e610302b688a2339a754fe632dc290a2bb8606f.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/7518be722f0d52cbacb846489e610302b688a2339a754fe632dc290a2bb8606f.jpg)

![752a4049b6084e75bd7de17a3266f8df3f001cc1ad8c4573d66e39b0e1e2580c.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/752a4049b6084e75bd7de17a3266f8df3f001cc1ad8c4573d66e39b0e1e2580c.jpg)

![90f36abe394cdd42f9bbbae1c58f4a2f6662a9710f813eb97154c527d1a29766.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/90f36abe394cdd42f9bbbae1c58f4a2f6662a9710f813eb97154c527d1a29766.jpg)

![a8ad37a9e216648a41e578ed9f8891aca7a3db3fd9c9dcc4f7477bdc5e10fd4f.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/a8ad37a9e216648a41e578ed9f8891aca7a3db3fd9c9dcc4f7477bdc5e10fd4f.jpg)

![c2fe08e484e48b1caf40ac722770d81821d0b9e1675cddea27d3939ca8f8caaf.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/c2fe08e484e48b1caf40ac722770d81821d0b9e1675cddea27d3939ca8f8caaf.jpg)

![e99c9cb800bb11391e493a0934ddfefdafd907156735693603eaeb6f6c68db43.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/e99c9cb800bb11391e493a0934ddfefdafd907156735693603eaeb6f6c68db43.jpg)

![f00277039bee2bbb83704d4461c7b7729048937a103e6a0b5ea9be87234f1bf7.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/f00277039bee2bbb83704d4461c7b7729048937a103e6a0b5ea9be87234f1bf7.jpg)

![fa08adc11ed507a4ec47c66f9d90d329424c55d427efa40dcd7d79eb6f0dccbf.jpg](../icml_results/4597_Turning%20Back%20Without%20Forgetting_%20Selective%20Backward%20Refinement%20for%20Parameter-Efficient%20Continual%20Lea/tables/fa08adc11ed507a4ec47c66f9d90d329424c55d427efa40dcd7d79eb6f0dccbf.jpg)

<a id="paper-4551"></a>

## Reward Under Attack: Analyzing the Robustness and Hackability of Process Reward Models


### Images

![02708766bda6b2a3d703d64188e89185a34d8ef61c55cf99c314c59f5c22e600.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/02708766bda6b2a3d703d64188e89185a34d8ef61c55cf99c314c59f5c22e600.jpg)

![255b79e92f2b67b45377ebfea5fa3a0cdecce7c96c802fb35e203af560f723bc.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/255b79e92f2b67b45377ebfea5fa3a0cdecce7c96c802fb35e203af560f723bc.jpg)

![27d4cc275b1cd6324ee4088c89544383ef184d2eab665ff567b76ca4ff8a60fe.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/27d4cc275b1cd6324ee4088c89544383ef184d2eab665ff567b76ca4ff8a60fe.jpg)

![2988b7bb9c89314091961ae8c630a6c1aca8563bf8a9f0286a96de5a5b50ae79.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/2988b7bb9c89314091961ae8c630a6c1aca8563bf8a9f0286a96de5a5b50ae79.jpg)

![62541400d0f897717bc49bba151f864da78f42f82c734cbec59768528c903fee.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/62541400d0f897717bc49bba151f864da78f42f82c734cbec59768528c903fee.jpg)

![7a73c907f8a44116e5ebf7f2959c846b76df6e1da501cd076e4553b89a967128.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/7a73c907f8a44116e5ebf7f2959c846b76df6e1da501cd076e4553b89a967128.jpg)

![7d935469ff33e14f07fd371c7f8040aa3f417733cb1691a7905a44dcd7679618.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/7d935469ff33e14f07fd371c7f8040aa3f417733cb1691a7905a44dcd7679618.jpg)

![b414defe507a2a2a5236d85d58c027d4ccef313214504d11587904584effdc30.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/b414defe507a2a2a5236d85d58c027d4ccef313214504d11587904584effdc30.jpg)

![b6b77ba7d41805b393494ce3ee6bbb500104182847c58191e0364db7532aecbe.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/b6b77ba7d41805b393494ce3ee6bbb500104182847c58191e0364db7532aecbe.jpg)

![e6e29b50dce01e7ee2a7ff54409b8cdaed42163073bb08a818434bde71d055ea.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/e6e29b50dce01e7ee2a7ff54409b8cdaed42163073bb08a818434bde71d055ea.jpg)

![e703fb103481b065f9be3a4051b6547caa7123ab480e01ae11eb291110436b98.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/e703fb103481b065f9be3a4051b6547caa7123ab480e01ae11eb291110436b98.jpg)

![edea950ccb9772ba9baaf31f41b80db290dc1a24f6e8107da8ba05019f0061d7.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/edea950ccb9772ba9baaf31f41b80db290dc1a24f6e8107da8ba05019f0061d7.jpg)

![eed8013f55e206e7abab078630eb735a64486b93f1e83cd0585374b87b815195.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/eed8013f55e206e7abab078630eb735a64486b93f1e83cd0585374b87b815195.jpg)

![f00c0e0ce19ca922d69da9f137ceac93f07ac8fbc7102422f9076e418e74c24d.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/images/f00c0e0ce19ca922d69da9f137ceac93f07ac8fbc7102422f9076e418e74c24d.jpg)

### Tables

![071a74a08a490ce2ce4ce8e670a43e26ab2c3ed9586a9265f597fabe331f76d8.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/071a74a08a490ce2ce4ce8e670a43e26ab2c3ed9586a9265f597fabe331f76d8.jpg)

![0dde70fbd990053e7ccbb049f8540de42a6362fb7bd0c026bd43a560f683bffd.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/0dde70fbd990053e7ccbb049f8540de42a6362fb7bd0c026bd43a560f683bffd.jpg)

![23c32af87d3a88a4b5edefe2ff420468a7b8852d21af918909cb626dbde9069f.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/23c32af87d3a88a4b5edefe2ff420468a7b8852d21af918909cb626dbde9069f.jpg)

![3605c6a79e1de31c9169393bc33cbbdbf6c19782b399af699c5144ffbe6f5d8a.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/3605c6a79e1de31c9169393bc33cbbdbf6c19782b399af699c5144ffbe6f5d8a.jpg)

![4dfa519f3b1038c43ba5f297af320173de61c7bd1c35f17263192d3101cdc15d.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/4dfa519f3b1038c43ba5f297af320173de61c7bd1c35f17263192d3101cdc15d.jpg)

![556cbaad0a95a16f9f3bdd27c798f1283ced9997a3bf861056232c92fad2ff6f.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/556cbaad0a95a16f9f3bdd27c798f1283ced9997a3bf861056232c92fad2ff6f.jpg)

![5784fb5831bd77c4038933aee5b5df2ecdcae905b669c128f28096faa2a776eb.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/5784fb5831bd77c4038933aee5b5df2ecdcae905b669c128f28096faa2a776eb.jpg)

![5ce436976287fd58713df1d59a595146dc1451e3175767f8033a8a64556e59e9.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/5ce436976287fd58713df1d59a595146dc1451e3175767f8033a8a64556e59e9.jpg)

![6e9bd74bd70a01846c508c2d2655b18069154a5d400191233fe786baa5e2e7fd.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/6e9bd74bd70a01846c508c2d2655b18069154a5d400191233fe786baa5e2e7fd.jpg)

![80e71e6f534ca2015a156e76567b6eb9707ed1d2a745f889fd63023e9fc4080e.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/80e71e6f534ca2015a156e76567b6eb9707ed1d2a745f889fd63023e9fc4080e.jpg)

![b8188d9dad135b2cc44d0edfca017bb99e08a07ec2e67e1cd80c3b4df34aaf9a.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/b8188d9dad135b2cc44d0edfca017bb99e08a07ec2e67e1cd80c3b4df34aaf9a.jpg)

![dd76a019eee73fae2c20e5869feeca5c985860c99a01c4a1e58ea1d34de74f9c.jpg](../icml_results/4598_Reward%20Under%20Attack_%20Analyzing%20the%20Robustness%20and%20Hackability%20of%20Process%20Reward%20Models/tables/dd76a019eee73fae2c20e5869feeca5c985860c99a01c4a1e58ea1d34de74f9c.jpg)

<a id="paper-4552"></a>

## MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning


### Images

![095840cc84f6c83796068d4ccae75323ec5456e8438906982738691028b66151.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/095840cc84f6c83796068d4ccae75323ec5456e8438906982738691028b66151.jpg)

![13424a26454c3f46458865c80a9f26b10afe8fd3aeca6539f542a3137c2a228b.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/13424a26454c3f46458865c80a9f26b10afe8fd3aeca6539f542a3137c2a228b.jpg)

![1fd962be6e31f66479078ed7a620c61875cc041a6c8149f9b5a3ef61fb54bb14.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/1fd962be6e31f66479078ed7a620c61875cc041a6c8149f9b5a3ef61fb54bb14.jpg)

![2406cb980e0f08769b1cd7255a16596d5081f798ed19f6e50b57101f13ddab72.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/2406cb980e0f08769b1cd7255a16596d5081f798ed19f6e50b57101f13ddab72.jpg)

![2bfd26377c172fb7f95ada3e2be1c9516a144aef34400b9e8ca2031cba3913b5.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/2bfd26377c172fb7f95ada3e2be1c9516a144aef34400b9e8ca2031cba3913b5.jpg)

![36537f9484a4332bfc30b704ade82e2b10201cdc2c94759627e0da66c2a5790f.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/36537f9484a4332bfc30b704ade82e2b10201cdc2c94759627e0da66c2a5790f.jpg)

![47630fb9709b7e87c5779faac40365ec3858c22e5158556e033e8dce88075ece.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/47630fb9709b7e87c5779faac40365ec3858c22e5158556e033e8dce88075ece.jpg)

![4fbde8dfba38681b6bed6649a08a96b4cb39c96237e27c7df80c9052a54860a9.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/4fbde8dfba38681b6bed6649a08a96b4cb39c96237e27c7df80c9052a54860a9.jpg)

![52e31201d1ccecae0e3da18a8a12c2ed9ccc97e73085e50cc0c143592f004a5e.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/52e31201d1ccecae0e3da18a8a12c2ed9ccc97e73085e50cc0c143592f004a5e.jpg)

![5f665013f0a44e9653d3c0fa151d04c7f4f4c7f737ba18a67abee3ddb971eaea.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/5f665013f0a44e9653d3c0fa151d04c7f4f4c7f737ba18a67abee3ddb971eaea.jpg)

![6295bcc0f990a16a357706caa114d746162f27bb7a1737ae7f14d335e9433b4d.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/6295bcc0f990a16a357706caa114d746162f27bb7a1737ae7f14d335e9433b4d.jpg)

![6a7072edb36e7e89fd012a53e711682d08ead57181c9a996131b275de5f7bf92.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/6a7072edb36e7e89fd012a53e711682d08ead57181c9a996131b275de5f7bf92.jpg)

![6bc78c441f9e74695817147180438ffc426871098f3efa12b080fedb57aabf33.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/6bc78c441f9e74695817147180438ffc426871098f3efa12b080fedb57aabf33.jpg)

![6f73a663d2de653c214bff20ffe12c4b83a792ab5510a302e49bbfa45496c8ce.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/6f73a663d2de653c214bff20ffe12c4b83a792ab5510a302e49bbfa45496c8ce.jpg)

![791f47550b6ef3748b1e6c84cef3e790a55d013bbcbb3393b18af3acd2444db3.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/791f47550b6ef3748b1e6c84cef3e790a55d013bbcbb3393b18af3acd2444db3.jpg)

![90faf57e43abe799b5ce820871d539486480c891e8d277d5628756eac49759b1.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/90faf57e43abe799b5ce820871d539486480c891e8d277d5628756eac49759b1.jpg)

![a011947ae9d63fdb5adc9208d78948dc530820a1cdc6a7a5fb8d54959bd7bb0a.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/a011947ae9d63fdb5adc9208d78948dc530820a1cdc6a7a5fb8d54959bd7bb0a.jpg)

![a5b0ada1e9d3a6aba3ee78cbc2d12702cb5481aeb406259ea1cb3e464d25c532.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/a5b0ada1e9d3a6aba3ee78cbc2d12702cb5481aeb406259ea1cb3e464d25c532.jpg)

![a655fe43c965aa521db9fb02d1852d055ba2bba1d861c45b877f928e837ea2fc.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/a655fe43c965aa521db9fb02d1852d055ba2bba1d861c45b877f928e837ea2fc.jpg)

![ac0c56c2223a9152f961849bfc8bafaa9eae985276fd686ecc79babbcf5fae53.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/ac0c56c2223a9152f961849bfc8bafaa9eae985276fd686ecc79babbcf5fae53.jpg)

![be9dd06a14da13dff47a87925cf6c429b3702001968a290c0eb6b0f15fbbba3a.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/be9dd06a14da13dff47a87925cf6c429b3702001968a290c0eb6b0f15fbbba3a.jpg)

![eea4a9f1bf923c518c499e494f6ea3040d81665be929540770867da7f15df150.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/eea4a9f1bf923c518c499e494f6ea3040d81665be929540770867da7f15df150.jpg)

![f06e3e8d690be5cb8897fc9f90c34e040c96bb910b78ac0dd1f963396ac5172a.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/f06e3e8d690be5cb8897fc9f90c34e040c96bb910b78ac0dd1f963396ac5172a.jpg)

![f12ca2dc9de5a9844f5689c3f555b2a3257945f895d5f53378ee01ab18afa1ef.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/f12ca2dc9de5a9844f5689c3f555b2a3257945f895d5f53378ee01ab18afa1ef.jpg)

![f13eca23faaf2a19a7f5e2d8dd0bdbff878a262fe16393f072b8e4275928c7f8.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/f13eca23faaf2a19a7f5e2d8dd0bdbff878a262fe16393f072b8e4275928c7f8.jpg)

![f3858b6528c350784531f0a5a4a4c94502c6ceb0845ce2e2cc684a32508a84db.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/f3858b6528c350784531f0a5a4a4c94502c6ceb0845ce2e2cc684a32508a84db.jpg)

![fab824f473d2f0b607e57245b24f09e33032530a2b27991b52effe27ab3869d9.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/images/fab824f473d2f0b607e57245b24f09e33032530a2b27991b52effe27ab3869d9.jpg)

### Tables

![0c26c3f4546b3969f5206651d121a9ae4a896cc235bbb6f100415ef8ef5b3d37.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/0c26c3f4546b3969f5206651d121a9ae4a896cc235bbb6f100415ef8ef5b3d37.jpg)

![108c798d05522654814abc038d54cb0838187d353db63491f22de7364bf853e8.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/108c798d05522654814abc038d54cb0838187d353db63491f22de7364bf853e8.jpg)

![354486720e112710287536cf9567523836d45db7ab8878d8a9d0745da9a5a1c4.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/354486720e112710287536cf9567523836d45db7ab8878d8a9d0745da9a5a1c4.jpg)

![539f53f8ec236ca28468627648688e0708b0d76934718bb4269e9dba6ff3ad0d.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/539f53f8ec236ca28468627648688e0708b0d76934718bb4269e9dba6ff3ad0d.jpg)

![572c2b5a85ee74bbec940f20bb2f4fad58e5cd4cf5e223ae3366c2818875ba30.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/572c2b5a85ee74bbec940f20bb2f4fad58e5cd4cf5e223ae3366c2818875ba30.jpg)

![6309674e17d71d6121454dd634425884ea94f4357bfe6855d9d52b0f5a359d33.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/6309674e17d71d6121454dd634425884ea94f4357bfe6855d9d52b0f5a359d33.jpg)

![70a981199b6a7cc9d1ac1c1ceac99daca98d300caee76529e2c67c1fdac40d9a.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/70a981199b6a7cc9d1ac1c1ceac99daca98d300caee76529e2c67c1fdac40d9a.jpg)

![778dac1b99def22407d8632aef15b36644793b6c33ce07e2bec72f830cdd45d9.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/778dac1b99def22407d8632aef15b36644793b6c33ce07e2bec72f830cdd45d9.jpg)

![895fd281952bde1413a23013a4e6a62fcdf8a1132f596ab014f89bcbd5ce99ad.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/895fd281952bde1413a23013a4e6a62fcdf8a1132f596ab014f89bcbd5ce99ad.jpg)

![a33959c7853295a0ad958a4faad335741a9334277f26023d66a15b858a253d7e.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/a33959c7853295a0ad958a4faad335741a9334277f26023d66a15b858a253d7e.jpg)

![c361ec6fd2a450da10ece1aeb2cdfbcf153f6d8d17ae3a0e474838565b0cdc3d.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/c361ec6fd2a450da10ece1aeb2cdfbcf153f6d8d17ae3a0e474838565b0cdc3d.jpg)

![c632872c9be92e6d605283e9cad1140e5089dfd05a81d4296b158e3db1c5f455.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/c632872c9be92e6d605283e9cad1140e5089dfd05a81d4296b158e3db1c5f455.jpg)

![e1968287cbe2e8d575b51bb385fdf9f250317fbd530d0cfed08dbe19fd19bc69.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/e1968287cbe2e8d575b51bb385fdf9f250317fbd530d0cfed08dbe19fd19bc69.jpg)

![e1ac29137db81336c3c3ee4f74964bdc104e7646589796cb70de909197893d7d.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/e1ac29137db81336c3c3ee4f74964bdc104e7646589796cb70de909197893d7d.jpg)

![f17653cb112f1094a65ce144c7a9937c99efbb1c7483b028c3a9095c99da36ed.jpg](../icml_results/4599_MEAL_%20A%20Benchmark%20for%20Continual%20Multi-Agent%20Reinforcement%20Learning/tables/f17653cb112f1094a65ce144c7a9937c99efbb1c7483b028c3a9095c99da36ed.jpg)

<a id="paper-4553"></a>

## Reasoning about Reasoning: BAPO Bounds on Chain-of-Thought Token Complexity in LLMs


### Images

![0ba04244c87425e85328c4c62d9da022f85445cc92ffe9bcfdb68611d03c7203.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/images/0ba04244c87425e85328c4c62d9da022f85445cc92ffe9bcfdb68611d03c7203.jpg)

![1b76e9c289b7f23546ad01b7b4e9e7bfe54f3bfd035623dd3dd54328ff6fc430.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/images/1b76e9c289b7f23546ad01b7b4e9e7bfe54f3bfd035623dd3dd54328ff6fc430.jpg)

![4ab7148b0ad626e37fccc74955dd67eefa246e90f75d5b72f5dce046747f27c8.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/images/4ab7148b0ad626e37fccc74955dd67eefa246e90f75d5b72f5dce046747f27c8.jpg)

![a3c0cec81d712767ecdc563f7b667ffda2dbcd4a502985fc2a67a2a7d635f9ed.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/images/a3c0cec81d712767ecdc563f7b667ffda2dbcd4a502985fc2a67a2a7d635f9ed.jpg)

![cf18f65e460f225a42679a46da62d5b0e9291d7ac645954ca3ea661ffcf23bb1.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/images/cf18f65e460f225a42679a46da62d5b0e9291d7ac645954ca3ea661ffcf23bb1.jpg)

![d95868214e4b94e65627d9cdd06caece6738d788bdd24ddbdd8cdc1ca484f4b8.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/images/d95868214e4b94e65627d9cdd06caece6738d788bdd24ddbdd8cdc1ca484f4b8.jpg)

### Tables

![3ba9df3f8d5288e1f98b051d328dfa64caf919717fc0cd897bdd256f21c077be.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/tables/3ba9df3f8d5288e1f98b051d328dfa64caf919717fc0cd897bdd256f21c077be.jpg)

![597ae4a615f31c4213f2ba7f35cc117c96a2523654ad9881abf756ecbb2c4d03.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/tables/597ae4a615f31c4213f2ba7f35cc117c96a2523654ad9881abf756ecbb2c4d03.jpg)

![6c900e6e164750bb875489a5f60ab0efabeb2fb30178c8602093dcb126c23d62.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/tables/6c900e6e164750bb875489a5f60ab0efabeb2fb30178c8602093dcb126c23d62.jpg)

![afe204c75e2882701e714fbd063bfaa892ca3421dc2f680ec0665e398e2e5a8e.jpg](../icml_results/4600_Reasoning%20about%20Reasoning_%20BAPO%20Bounds%20on%20Chain-of-Thought%20Token%20Complexity%20in%20LLMs/tables/afe204c75e2882701e714fbd063bfaa892ca3421dc2f680ec0665e398e2e5a8e.jpg)

<a id="paper-4554"></a>

## Task-Awareness Improves LLM Generations and Uncertainty

### Images

![1d5288be9716876ef78bdb1f45001ac133e39ccf25640c33b0cb305da681c9f0.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/1d5288be9716876ef78bdb1f45001ac133e39ccf25640c33b0cb305da681c9f0.jpg)

![2192d0e44fecbcee77592aa20791234ba6137c6f1ab855c4435660c020640b4f.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/2192d0e44fecbcee77592aa20791234ba6137c6f1ab855c4435660c020640b4f.jpg)

![3a7f04b9b594a0a5f147ac01396ccb13b13ab98e5a2662d5ef4d0c344d84a5a8.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/3a7f04b9b594a0a5f147ac01396ccb13b13ab98e5a2662d5ef4d0c344d84a5a8.jpg)

![42988d47cd7ad466a13f8fbb5d72fd458c28febbab628eeeb81972c4004cdd90.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/42988d47cd7ad466a13f8fbb5d72fd458c28febbab628eeeb81972c4004cdd90.jpg)

![4e384bcda06a4557ff2e4624eedeb6e1b2ed7791f9ead5628e5ce33e3c10fc8c.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/4e384bcda06a4557ff2e4624eedeb6e1b2ed7791f9ead5628e5ce33e3c10fc8c.jpg)

![6bf7db60e6106f98489d2a2ba4afd1edd62e3364341002e443fc14470377ec5f.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/6bf7db60e6106f98489d2a2ba4afd1edd62e3364341002e443fc14470377ec5f.jpg)

![7618acb89a3c6f3c0b38d5aa6e4ced1db14ea6730f3602d526d156f88021661b.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/7618acb89a3c6f3c0b38d5aa6e4ced1db14ea6730f3602d526d156f88021661b.jpg)

![a4cd017f9a3d2d39bed142d5244cbfb6d099ff43e9757c29c14a6ab350287e0f.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/a4cd017f9a3d2d39bed142d5244cbfb6d099ff43e9757c29c14a6ab350287e0f.jpg)

![a566aa7426b4bfda698a2e905bac3bb134c48e48bbc35e39f9601d8ee283f515.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/a566aa7426b4bfda698a2e905bac3bb134c48e48bbc35e39f9601d8ee283f515.jpg)

![c6576a0167e76bad2f36b0f8db3723693fa9571aacd4c2d7e52575d841561fcf.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/c6576a0167e76bad2f36b0f8db3723693fa9571aacd4c2d7e52575d841561fcf.jpg)

![f72717ac833c8b7aabfd83322d272dc035b9f9421aa28cbdbec50181f5276410.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/f72717ac833c8b7aabfd83322d272dc035b9f9421aa28cbdbec50181f5276410.jpg)

### Tables

![1ed22e483b83961456b5aea9a925c66fc6245c4f01e3f8c2c7da995126da4025.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/1ed22e483b83961456b5aea9a925c66fc6245c4f01e3f8c2c7da995126da4025.jpg)

![2b4c921a4c7fd757b18ed7d360b110ed3edbb17b05ebc815128cca8c6c5f3e80.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/2b4c921a4c7fd757b18ed7d360b110ed3edbb17b05ebc815128cca8c6c5f3e80.jpg)

![330d4aae7436b47386b0c8403459effc34902cbc21470d1114b43e91ecf7761c.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/330d4aae7436b47386b0c8403459effc34902cbc21470d1114b43e91ecf7761c.jpg)

![3b6588245d968dc6404be7313a12226e0164df3adaa60d5aa6fb7e9834f8495a.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/3b6588245d968dc6404be7313a12226e0164df3adaa60d5aa6fb7e9834f8495a.jpg)

![4b5ced5db5e1a21510179854fa91b75aa4f61fac64bd7be7084c1980dd212420.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/4b5ced5db5e1a21510179854fa91b75aa4f61fac64bd7be7084c1980dd212420.jpg)

![5818746e2f7cb94d6c10dbd364c37b65616cba8539e1948362f67dc5b06fa3a4.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/5818746e2f7cb94d6c10dbd364c37b65616cba8539e1948362f67dc5b06fa3a4.jpg)

![93e37471bc109b6b799d3a88a7bc853fe042a17c922372ad435531af02a1d077.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/93e37471bc109b6b799d3a88a7bc853fe042a17c922372ad435531af02a1d077.jpg)

![a4fc261512652fb07e30f0f8b427d5a9d35002ad2ef0db272defaf578919b9fa.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/a4fc261512652fb07e30f0f8b427d5a9d35002ad2ef0db272defaf578919b9fa.jpg)

![ad6a6bd054697c4ca1121baacaedd39057cce57d40e4218dd13713111ac02b0e.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/ad6a6bd054697c4ca1121baacaedd39057cce57d40e4218dd13713111ac02b0e.jpg)
