# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 1 of 197

## 目录 (Table of Contents)

1. [What Preferences Can—and Cannot—Predict in Multi-Agent Online Learning](#paper-0001)
2. [Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization](#paper-0002)
3. [Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](#paper-0003)
4. [BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](#paper-0004)
5. [MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems](#paper-0005)
6. [TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior](#paper-0006)
7. [Perceptrons and Localization of Attention’s Mean-Field Landscape](#paper-0007)
8. [(Doubly) Exponential Lower Bounds for Follow the Regularized Leader in Potential Games](#paper-0008)
9. [Estimating Tail Risks in Language Model Output Distributions](#paper-0009)
10. [Language Generation in the Limit: Complexity Barriers and Implications for Learning](#paper-0010)
11. [Language Model Circuits Are Sparse in the Neuron Basis](#paper-0011)
12. [Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](#paper-0012)
13. [VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization](#paper-0013)
14. [Falling Trees: A Model Class for Interpretable Risk Prioritization](#paper-0014)
15. [Learning to Theorize the World from Observation](#paper-0015)
16. [Sycophancy Towards Researchers Drives Performative Misalignment](#paper-0016)
17. [Towards Understanding Adam Convergence on Highly Degenerate Polynomials](#paper-0017)
18. [Learning in Structured Stackelberg Games](#paper-0018)
19. [Riemannian Metric Matching for Scalable Geometric Modeling of Distributions](#paper-0019)
20. [Universal Redundancies in Time Series Foundation Models](#paper-0020)
21. [Optimal Rates for Feasible Payoff Set Estimation in Games](#paper-0021)
22. [On the Origin of Neural Scaling Laws: from Random Graphs to Natural Language](#paper-0022)
23. [$τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment](#paper-0023)
24. [Uncovering the Latent Potential of Deep Intermediate Representations](#paper-0024)
25. [INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic](#paper-0025)
26. [The Relative Instability of Model Comparison with Cross-validation](#paper-0026)
27. [CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning](#paper-0027)
28. [Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes](#paper-0028)
29. [Skill Neologisms: Towards Skill-based Continual Learning](#paper-0029)
30. [STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control](#paper-0030)
31. [Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning](#paper-0031)
32. [Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers](#paper-0032)
33. [Optimal Transport under Group Fairness Constraints](#paper-0033)

---


<a id="paper-0001"></a>

## What Preferences Can—and Cannot—Predict in Multi-Agent Online Learning


### Images

![0e7e5cc3c141db20bf7436e52b0f669df0709fb1fc0b37d423f6cccb4a1d2d3f.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/0e7e5cc3c141db20bf7436e52b0f669df0709fb1fc0b37d423f6cccb4a1d2d3f.jpg)

![155e073f1bfe13ec3692fa065be968411a1eaea31ffe736e0a90febe160b09e5.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/155e073f1bfe13ec3692fa065be968411a1eaea31ffe736e0a90febe160b09e5.jpg)

![19872e3f01c7699f09ba8e9a9f116c713e80ad52606a53f31eeef4bee51a8c8e.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/19872e3f01c7699f09ba8e9a9f116c713e80ad52606a53f31eeef4bee51a8c8e.jpg)

![2346981a117901dc6958dd89aae42067e3996b8ad722be6b13ee9181c5dc248b.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/2346981a117901dc6958dd89aae42067e3996b8ad722be6b13ee9181c5dc248b.jpg)

![2db969bdc8e6557f2fa814f225ade2e4ce8063d1c54ab8cde0dd776764fc7824.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/2db969bdc8e6557f2fa814f225ade2e4ce8063d1c54ab8cde0dd776764fc7824.jpg)

![36d7f4fb9c90f3f267782779701ae84813073cc1df3b7ae589dcce70a22691aa.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/36d7f4fb9c90f3f267782779701ae84813073cc1df3b7ae589dcce70a22691aa.jpg)

![441655fe0f1bd6a47ee247bdd177827b35e4ae7d578b711de8769440d1e95594.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/441655fe0f1bd6a47ee247bdd177827b35e4ae7d578b711de8769440d1e95594.jpg)

![4a82cb9ecbd8b117b1162aa74f73403f0d07092d0f534dbcede9ad43394e538a.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/4a82cb9ecbd8b117b1162aa74f73403f0d07092d0f534dbcede9ad43394e538a.jpg)

![4b71fcf8d3db4d4bd2051c02d413ee720096c773cc5d6c4a27857f65613b1ee1.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/4b71fcf8d3db4d4bd2051c02d413ee720096c773cc5d6c4a27857f65613b1ee1.jpg)

![591d1c5447e82e12c5f769eeca5d6d23db21c3ebb1f252088648eba51bd7e756.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/591d1c5447e82e12c5f769eeca5d6d23db21c3ebb1f252088648eba51bd7e756.jpg)

![6328a87a17e1f735429de141d644a3fbc8cd190b4d70b92761b0413da5b79733.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/6328a87a17e1f735429de141d644a3fbc8cd190b4d70b92761b0413da5b79733.jpg)

![9e653626d77a48a0c0ea61307713972a0ec4e57e1dcff7890e111bb409fe0621.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/9e653626d77a48a0c0ea61307713972a0ec4e57e1dcff7890e111bb409fe0621.jpg)

![a09932b5a916c23a23010e8372ade148d1f01785528a6804d9f9c5396181dbd7.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/a09932b5a916c23a23010e8372ade148d1f01785528a6804d9f9c5396181dbd7.jpg)

![b2f5b28c6998c136408fa03e5d754c8fd19734f04bc1e8c6b598618629e06678.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/b2f5b28c6998c136408fa03e5d754c8fd19734f04bc1e8c6b598618629e06678.jpg)

![e67341a7c06a87f90eda55027542093004afd823c8050fe59a20d70f45efc3d5.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/e67341a7c06a87f90eda55027542093004afd823c8050fe59a20d70f45efc3d5.jpg)

![ea66e5c34d47f19584956930f0ac0496580f1225bb496d421d7b38bc762d09eb.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/images/ea66e5c34d47f19584956930f0ac0496580f1225bb496d421d7b38bc762d09eb.jpg)

### Tables

![1001497ef466a11ba3de05c0312f6ba2b34d5b7d64754b2f53d6491957e4ca54.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/1001497ef466a11ba3de05c0312f6ba2b34d5b7d64754b2f53d6491957e4ca54.jpg)

![6b0bc2a695f59c48b6900f9d3fdd2a4ae2630b87afd636535dc16edfced184ea.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/6b0bc2a695f59c48b6900f9d3fdd2a4ae2630b87afd636535dc16edfced184ea.jpg)

![78b20288a3a6c93aeaf39b4853cb886118c0d4ac6bfd25c53c94f20dd3f9c7da.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/78b20288a3a6c93aeaf39b4853cb886118c0d4ac6bfd25c53c94f20dd3f9c7da.jpg)

![8c1731aa6146ecaf6ad7c60dfc6307948a3e60731e92f1deab40bfc03c283a75.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/8c1731aa6146ecaf6ad7c60dfc6307948a3e60731e92f1deab40bfc03c283a75.jpg)

![91d4ae50346809cc3bce1b17d5546ffc268c1f23e8d15c1ddf2e115bc1053881.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/91d4ae50346809cc3bce1b17d5546ffc268c1f23e8d15c1ddf2e115bc1053881.jpg)

![9b8c9ef48bb779ee2930ea08d5be00ffc80564f0a8dcbef403c8f79eb61c3857.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/9b8c9ef48bb779ee2930ea08d5be00ffc80564f0a8dcbef403c8f79eb61c3857.jpg)

![9f723277eee15048c8e4eb97068e973f8c0a87ad2630be5be9b67ab1ae39c855.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/9f723277eee15048c8e4eb97068e973f8c0a87ad2630be5be9b67ab1ae39c855.jpg)

![b827022a2c03f3d140f8a2b711b40412e36206558af7c77bc31a00ff51b1f89a.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/b827022a2c03f3d140f8a2b711b40412e36206558af7c77bc31a00ff51b1f89a.jpg)

![d37d999873f1e74fa5a6856e2118a21d95dd3547c36cb00ee3fa834ca4c11a89.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/d37d999873f1e74fa5a6856e2118a21d95dd3547c36cb00ee3fa834ca4c11a89.jpg)

![dbfc6afbc030fc39ddf4af4bf26d7cbf5a8022c2dc2714bca6df213600a648ad.jpg](../icml_results/1_What%20Preferences%20Can%E2%80%94and%20Cannot%E2%80%94Predict%20in%20Multi-Agent%20Online%20Learning/tables/dbfc6afbc030fc39ddf4af4bf26d7cbf5a8022c2dc2714bca6df213600a648ad.jpg)

<a id="paper-0002"></a>

## Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization


### Images

![0875f965d86b089f63de8cccb1d4a5db2f13d4ab482b3a41245854f439228f4e.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/0875f965d86b089f63de8cccb1d4a5db2f13d4ab482b3a41245854f439228f4e.jpg)

![36d04a7f4a27069439d9ca8091141ffea74cbb19488f65011b372e9b28304103.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/36d04a7f4a27069439d9ca8091141ffea74cbb19488f65011b372e9b28304103.jpg)

![6ac7e9a4f0f40f63d0b493f64b9da064b706576933502f05aedfcbc7e1d61137.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/6ac7e9a4f0f40f63d0b493f64b9da064b706576933502f05aedfcbc7e1d61137.jpg)

![759ce403fb218709fd1dc91e1bf71b55750fdfbf2c38041d0e388d20d0b81181.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/759ce403fb218709fd1dc91e1bf71b55750fdfbf2c38041d0e388d20d0b81181.jpg)

![790b639aa15aac1024fd2b4480066b38b74d9ea331782a05053f949db61f14d8.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/790b639aa15aac1024fd2b4480066b38b74d9ea331782a05053f949db61f14d8.jpg)

![85be7adfd6ae7db3cd71c674edc316381b1d41d903624bbcabb4cc46d46d6759.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/85be7adfd6ae7db3cd71c674edc316381b1d41d903624bbcabb4cc46d46d6759.jpg)

![8f9eb0e9ea47bedc0c772962322b7b5e7b2f6766de7bd3669e8a2b8ec775f921.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/8f9eb0e9ea47bedc0c772962322b7b5e7b2f6766de7bd3669e8a2b8ec775f921.jpg)

![cc3bc557464d3134fd3855b7060b22f9ce81a3b2060495f7eee973d96f27c6c1.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/cc3bc557464d3134fd3855b7060b22f9ce81a3b2060495f7eee973d96f27c6c1.jpg)

![ffe500b75330b3329859771149cf4400ed2bf1c8809c5eea4b01382d18b2feaa.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/images/ffe500b75330b3329859771149cf4400ed2bf1c8809c5eea4b01382d18b2feaa.jpg)

### Tables

![4ad1398b62711cecf139bc97cf93cdae5b29a2ec47986908192af87359fd6b6f.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/tables/4ad1398b62711cecf139bc97cf93cdae5b29a2ec47986908192af87359fd6b6f.jpg)

![90e147f3729103df2b9b4c725dc1cb8be8cdbaf40b04447475ee1990af76c00e.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/tables/90e147f3729103df2b9b4c725dc1cb8be8cdbaf40b04447475ee1990af76c00e.jpg)

![f9722031fc0273849ac81df541afabe5ebfe0398580b7aada5515e3cfb5787cb.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/tables/f9722031fc0273849ac81df541afabe5ebfe0398580b7aada5515e3cfb5787cb.jpg)

![fca80b202b44e9b49c53d80d2156c58cb88bc3027dbcdda7f5290850549a44a4.jpg](../icml_results/2_Asymmetric%20Perturbation%20in%20Solving%20Bilinear%20Saddle-Point%20Optimization/tables/fca80b202b44e9b49c53d80d2156c58cb88bc3027dbcdda7f5290850549a44a4.jpg)

<a id="paper-0003"></a>

## Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering


### Images

![35e5917ccaf854e91b96a5321b96d382944e214f5b6251cda4b709f0135e0ccf.jpg](../icml_results/3_Simple%20Algorithms%20for%20Bad%20Triangle%20Transversals%20with%20Applications%20to%20Correlation%20Clustering/images/35e5917ccaf854e91b96a5321b96d382944e214f5b6251cda4b709f0135e0ccf.jpg)

![ed9a64c21d54cc3f5a5e0f73b6e203838fae9987f64670d5d152b52eacf90d25.jpg](../icml_results/3_Simple%20Algorithms%20for%20Bad%20Triangle%20Transversals%20with%20Applications%20to%20Correlation%20Clustering/images/ed9a64c21d54cc3f5a5e0f73b6e203838fae9987f64670d5d152b52eacf90d25.jpg)

![f6aa3b03e1dcc851e04c5aa36092af3189981d4c063613db7b0252983bbc7c02.jpg](../icml_results/3_Simple%20Algorithms%20for%20Bad%20Triangle%20Transversals%20with%20Applications%20to%20Correlation%20Clustering/images/f6aa3b03e1dcc851e04c5aa36092af3189981d4c063613db7b0252983bbc7c02.jpg)

### Tables

![244b6e4f8fde41e511b38a050db21a6ceec603f49efe790eb0b8a693a9a821bf.jpg](../icml_results/3_Simple%20Algorithms%20for%20Bad%20Triangle%20Transversals%20with%20Applications%20to%20Correlation%20Clustering/tables/244b6e4f8fde41e511b38a050db21a6ceec603f49efe790eb0b8a693a9a821bf.jpg)

![b7c1f0a3843551283f2cbf175a0e0690b2830b08ad3dc34cd9a490e0a2ace17e.jpg](../icml_results/3_Simple%20Algorithms%20for%20Bad%20Triangle%20Transversals%20with%20Applications%20to%20Correlation%20Clustering/tables/b7c1f0a3843551283f2cbf175a0e0690b2830b08ad3dc34cd9a490e0a2ace17e.jpg)

![bec248a12ea1c710cbe52d6c54a3422baf84e3d875feab314eda64e27ba71977.jpg](../icml_results/3_Simple%20Algorithms%20for%20Bad%20Triangle%20Transversals%20with%20Applications%20to%20Correlation%20Clustering/tables/bec248a12ea1c710cbe52d6c54a3422baf84e3d875feab314eda64e27ba71977.jpg)

<a id="paper-0004"></a>

## BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs


### Images

![2bb92c305b7f5a81655f8aa310bd9a12858eaa40de6601ff2c2d42854805d06a.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/2bb92c305b7f5a81655f8aa310bd9a12858eaa40de6601ff2c2d42854805d06a.jpg)

![5e2252228498b2f5baf6703033809f0a0ee87719c184f4cf09fb59379803c77d.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/5e2252228498b2f5baf6703033809f0a0ee87719c184f4cf09fb59379803c77d.jpg)

![5e73872c570c0635dce126055279e8738b3dcd0c3c3498b7c5a63be6df5743f4.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/5e73872c570c0635dce126055279e8738b3dcd0c3c3498b7c5a63be6df5743f4.jpg)

![8378accb1adcc5fab50232cf8651f0c1c2cde8a59f43fbd2641f1e606d90ba9c.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/8378accb1adcc5fab50232cf8651f0c1c2cde8a59f43fbd2641f1e606d90ba9c.jpg)

![99b86d5d7dbaf820a379f20131b17ea4a12891a343a2df2bcc32eb65f95926c9.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/99b86d5d7dbaf820a379f20131b17ea4a12891a343a2df2bcc32eb65f95926c9.jpg)

![ab46965dc71a8a8a2c4f6dadf7f23e6b6060f1c78263a5d4b24b11c1de323490.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/ab46965dc71a8a8a2c4f6dadf7f23e6b6060f1c78263a5d4b24b11c1de323490.jpg)

![c308e511c4b70f5d35d111083d6f2dc4ca4448b7cb5194380ff5286e5fa98463.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/c308e511c4b70f5d35d111083d6f2dc4ca4448b7cb5194380ff5286e5fa98463.jpg)

![c8323633c0cf63398a47971c0633992f77f33b39477b29a1ccdfe90467585a03.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/c8323633c0cf63398a47971c0633992f77f33b39477b29a1ccdfe90467585a03.jpg)

![d8afacf77db4370bc88c268b6ec0a29f899aca5722a26f05757a87e43b011cdf.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/d8afacf77db4370bc88c268b6ec0a29f899aca5722a26f05757a87e43b011cdf.jpg)

![f05e1218f02280fde2b1192265b1d9479aafe5b4679082bb38d8fb52aba77e83.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/images/f05e1218f02280fde2b1192265b1d9479aafe5b4679082bb38d8fb52aba77e83.jpg)

### Tables

![146dd77407c71ce54067e4b2469c004e5566c71d3947bcaea018f136cc30a3c7.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/146dd77407c71ce54067e4b2469c004e5566c71d3947bcaea018f136cc30a3c7.jpg)

![1b2b69ad39794ae34d4afa1bb537ea0b9673a811a1a73056bffc691ed2ce075b.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/1b2b69ad39794ae34d4afa1bb537ea0b9673a811a1a73056bffc691ed2ce075b.jpg)

![4f77431e03b898ea22ee3b9974df8587c6c0df19de31f533cc96e55d1d5273d1.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/4f77431e03b898ea22ee3b9974df8587c6c0df19de31f533cc96e55d1d5273d1.jpg)

![63611f0dfff27a0ddd86d8dc3c6b303de452584b5801b5dad581953b8ab046dd.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/63611f0dfff27a0ddd86d8dc3c6b303de452584b5801b5dad581953b8ab046dd.jpg)

![684938f5fda96daa0cd82ea54458fed4cd4fba55579a5a5bbc6db1f09d1546b7.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/684938f5fda96daa0cd82ea54458fed4cd4fba55579a5a5bbc6db1f09d1546b7.jpg)

![6cfeb79e293ba4c098b2eb77e098abbf7c2a34d43ce9daa262d74efb2efce8af.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/6cfeb79e293ba4c098b2eb77e098abbf7c2a34d43ce9daa262d74efb2efce8af.jpg)

![8069b62d18f8ca29823e78c86d2f85d39c3bee37c98600b43e9e1b795798cbba.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/8069b62d18f8ca29823e78c86d2f85d39c3bee37c98600b43e9e1b795798cbba.jpg)

![9be2143fab4c35b099678583d1fa5e985877608554c9b456f08fbbe11085ec08.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/9be2143fab4c35b099678583d1fa5e985877608554c9b456f08fbbe11085ec08.jpg)

![b565203d97e2590bd074fb314a0ec2851480aade88707ac28c5798d693a214a4.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/b565203d97e2590bd074fb314a0ec2851480aade88707ac28c5798d693a214a4.jpg)

![de70845538fcdf54f63af3be8daa5f1b020970084703b2c920ec127092658ab0.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/de70845538fcdf54f63af3be8daa5f1b020970084703b2c920ec127092658ab0.jpg)

![fa15c8deb2c40878483947dfcc21e48bfe71cf03ec231ac009afa21a0c4ea704.jpg](../icml_results/4_BlitzRank_%20Principled%20Zero-shot%20Ranking%20Agents%20with%20Tournament%20Graphs/tables/fa15c8deb2c40878483947dfcc21e48bfe71cf03ec231ac009afa21a0c4ea704.jpg)

<a id="paper-0005"></a>

## MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems


### Images

![06d26037cfdc0a55b341f4541bf15037ac04de97911d7b78289caeaab77c3c33.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/06d26037cfdc0a55b341f4541bf15037ac04de97911d7b78289caeaab77c3c33.jpg)

![07ff7950bdd0c6d3bf080255afc70036363051974a7042bd3d40d9f1256bd10b.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/07ff7950bdd0c6d3bf080255afc70036363051974a7042bd3d40d9f1256bd10b.jpg)

![0814a65b7aa2a257eebcffd0297d4bec8baee5cc07d708f7245a78aa2dc3dd18.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/0814a65b7aa2a257eebcffd0297d4bec8baee5cc07d708f7245a78aa2dc3dd18.jpg)

![16180299027137194fb3d10a2b8b30e2868df590a8749eeb67ee1f3c098a80ac.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/16180299027137194fb3d10a2b8b30e2868df590a8749eeb67ee1f3c098a80ac.jpg)

![1c73aea1f729687233324ff540212269ca14da4bfad49f0810e02702f9e6e24d.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/1c73aea1f729687233324ff540212269ca14da4bfad49f0810e02702f9e6e24d.jpg)

![2bc9f45d9007c85a3fcac73cd6790eb195e0b41d624e1f89e9bc987b31bca28f.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/2bc9f45d9007c85a3fcac73cd6790eb195e0b41d624e1f89e9bc987b31bca28f.jpg)

![2f97f124989d96f8d1c15b7867ca5f2ed70a612d185f05799a8383713f2afe51.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/2f97f124989d96f8d1c15b7867ca5f2ed70a612d185f05799a8383713f2afe51.jpg)

![34414a30f8c4008999e67c61c438b22ccb99aa964709651ff8e01c0e566b7e0a.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/34414a30f8c4008999e67c61c438b22ccb99aa964709651ff8e01c0e566b7e0a.jpg)

![4d946f043e51f82b09e7d15b011632334a06e30cd0d01e35db0f7830367b788e.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/4d946f043e51f82b09e7d15b011632334a06e30cd0d01e35db0f7830367b788e.jpg)

![56a99f7216a3b21c44d19ed6b1f928c3165b6e55478309cc7f1b4e1d8e02903a.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/56a99f7216a3b21c44d19ed6b1f928c3165b6e55478309cc7f1b4e1d8e02903a.jpg)

![5c4b6861482ddb8b99de0f665fd5b25cbc906cefc543083bb927e93fe3db0fc4.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/5c4b6861482ddb8b99de0f665fd5b25cbc906cefc543083bb927e93fe3db0fc4.jpg)

![760d4ef3349bb4d49a13fef536fe63b3680ed59d92478f573af5d18b64cac461.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/760d4ef3349bb4d49a13fef536fe63b3680ed59d92478f573af5d18b64cac461.jpg)

![767f97b0d25b003fbb368b94a72f808497347e90ca094c503ebfca46c5a1a5b0.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/767f97b0d25b003fbb368b94a72f808497347e90ca094c503ebfca46c5a1a5b0.jpg)

![85fcac3f9f971e9853d22df809892d5406457981276c58e94bfffa76f5121c0e.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/85fcac3f9f971e9853d22df809892d5406457981276c58e94bfffa76f5121c0e.jpg)

![b21e2e6f05904354d7558b109c11b453db5a3257b39552c99c705081104e1cb1.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/b21e2e6f05904354d7558b109c11b453db5a3257b39552c99c705081104e1cb1.jpg)

![b4abbd40bc8655ed6022b0e8430712b38981fe5aa33dad2a5017274887131167.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/b4abbd40bc8655ed6022b0e8430712b38981fe5aa33dad2a5017274887131167.jpg)

![bd754401ac543a263f4535bbaf4abbd23999f430459ce7c393d8cd5b50ae610e.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/bd754401ac543a263f4535bbaf4abbd23999f430459ce7c393d8cd5b50ae610e.jpg)

![d1774ed9d4e3f7503df7d0de187b6e7c6902e963a3b2b3dd5f5ab1716c750c19.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/d1774ed9d4e3f7503df7d0de187b6e7c6902e963a3b2b3dd5f5ab1716c750c19.jpg)

![ec47ba4ce56a7681c0cf4bfd1750f74a5d13f429bcb6fe34db51425e1c7758c3.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/ec47ba4ce56a7681c0cf4bfd1750f74a5d13f429bcb6fe34db51425e1c7758c3.jpg)

![fcf03618fae1111505d042d749854584618d60e1686e3ea8a02139e963ddb2a8.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/images/fcf03618fae1111505d042d749854584618d60e1686e3ea8a02139e963ddb2a8.jpg)

### Tables

![22bad1c2a7d56ae4232d74f6ca66ca51bcd0a06966e623903534d8494f6eae32.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/22bad1c2a7d56ae4232d74f6ca66ca51bcd0a06966e623903534d8494f6eae32.jpg)

![237643daf24e64f1295e456251348b9f51655c74556e998777d2c537449b7585.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/237643daf24e64f1295e456251348b9f51655c74556e998777d2c537449b7585.jpg)

![279211c049edde40c763047a84323279c088f69bb27014fb2140329fe6d933ec.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/279211c049edde40c763047a84323279c088f69bb27014fb2140329fe6d933ec.jpg)

![46ca3045df167472edf21c8f10cfdef8d47ae98943c218554b7525a1998bd852.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/46ca3045df167472edf21c8f10cfdef8d47ae98943c218554b7525a1998bd852.jpg)

![56d7f0c2c82fec0e783f99c429f495df317599b3eeb1ed8694c4d2fecbd4e496.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/56d7f0c2c82fec0e783f99c429f495df317599b3eeb1ed8694c4d2fecbd4e496.jpg)

![61e7bd6a3784be0aca963c3c4f0eb69d0179e25b67d49fe7e95fb2bd5b7152f0.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/61e7bd6a3784be0aca963c3c4f0eb69d0179e25b67d49fe7e95fb2bd5b7152f0.jpg)

![6605e0724f93669fe22cff35792b0fd407c69503f6b245b32763a07d55efa1a7.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/6605e0724f93669fe22cff35792b0fd407c69503f6b245b32763a07d55efa1a7.jpg)

![6a592e27dffe1b810dc2e7ba229fd3860e34ffbd6057d6bcf26e7b728fffc707.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/6a592e27dffe1b810dc2e7ba229fd3860e34ffbd6057d6bcf26e7b728fffc707.jpg)

![7ec150d09383fabb46d0a9e7e5d4f8a5a335df2024d6a9943e6b387401adc8cf.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/7ec150d09383fabb46d0a9e7e5d4f8a5a335df2024d6a9943e6b387401adc8cf.jpg)

![86ed68b9fc76b12b3f998e72ff9a86ccf89afe7c8e777bbe1e0b4c837972e1f3.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/86ed68b9fc76b12b3f998e72ff9a86ccf89afe7c8e777bbe1e0b4c837972e1f3.jpg)

![910bd82cf8fb843e03420d131e7a9320edf60defe515ae0ffd266ecd9e928403.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/910bd82cf8fb843e03420d131e7a9320edf60defe515ae0ffd266ecd9e928403.jpg)

![9374d73b48ffbdf339181cbf1a25a520263904486a2e7753dee2be550a91e110.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/9374d73b48ffbdf339181cbf1a25a520263904486a2e7753dee2be550a91e110.jpg)

![98314bfe4fdfdd52e245518e19a6e3bf24db07bf25dcd9ed42d7326d18039fb5.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/98314bfe4fdfdd52e245518e19a6e3bf24db07bf25dcd9ed42d7326d18039fb5.jpg)

![9c5adf786d7bec32cffb1ce23feac69b86e4ae39164c6d62db73c1c72b5203c2.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/9c5adf786d7bec32cffb1ce23feac69b86e4ae39164c6d62db73c1c72b5203c2.jpg)

![9f83f91b7acda8c6d0b20f9babc0442dfb1662884e6fead64a6cbf452491925d.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/9f83f91b7acda8c6d0b20f9babc0442dfb1662884e6fead64a6cbf452491925d.jpg)

![a9b14f50b8f2a1921e7c8330ac1df0ed76328d67aaf04dd97150ea40549e4c98.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/a9b14f50b8f2a1921e7c8330ac1df0ed76328d67aaf04dd97150ea40549e4c98.jpg)

![ac49ce2149f983d912fd6f8663c4e4a915434a2cebb9ed3288f09af423fa18ad.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/ac49ce2149f983d912fd6f8663c4e4a915434a2cebb9ed3288f09af423fa18ad.jpg)

![ad0a5cb10b3e4a5045c095182613458c9907d53266f96b9eb19094dd56439702.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/ad0a5cb10b3e4a5045c095182613458c9907d53266f96b9eb19094dd56439702.jpg)

![b6a2ce009b918a9c17b2804982833baca2a34be75f07a68ce4cef74c91d55e58.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/b6a2ce009b918a9c17b2804982833baca2a34be75f07a68ce4cef74c91d55e58.jpg)

![c172bc4543f36a4c633c0f86fe7a067cc43aa618731806a2d0d3ba349ce1a28e.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/c172bc4543f36a4c633c0f86fe7a067cc43aa618731806a2d0d3ba349ce1a28e.jpg)

![dfc749504cc689c238c9f5843b3c6630ccda87f7d4c43344e6bfda19a8e88f8f.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/dfc749504cc689c238c9f5843b3c6630ccda87f7d4c43344e6bfda19a8e88f8f.jpg)

![e3f0378f690829efbd917fef952e3f857c4c0ecbe0cfb5760ffff16641b89e05.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/e3f0378f690829efbd917fef952e3f857c4c0ecbe0cfb5760ffff16641b89e05.jpg)

![e85f80581a118730b1dd3a92fede1fd0492f1c34d4ab9e8db7ae56b07d2906ab.jpg](../icml_results/5_MemoryBench_%20A%20Benchmark%20for%20Memory%20and%20Continual%20Learning%20in%20LLM%20Systems/tables/e85f80581a118730b1dd3a92fede1fd0492f1c34d4ab9e8db7ae56b07d2906ab.jpg)

<a id="paper-0006"></a>

## TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior


### Images

![1bc68cd0edc3c99cd0f566c5722cf088cda445e1bc387cca68fe9e0600aabad1.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/1bc68cd0edc3c99cd0f566c5722cf088cda445e1bc387cca68fe9e0600aabad1.jpg)

![1fae6bde1837a6d40850c8b9840b9ba7df273d6f414c3baf178e6fdfee13f91a.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/1fae6bde1837a6d40850c8b9840b9ba7df273d6f414c3baf178e6fdfee13f91a.jpg)

![28d60f9447d49e46b9e73b1d89593ad3344a7ead1c2b0f7c3ae69f816aec1fe0.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/28d60f9447d49e46b9e73b1d89593ad3344a7ead1c2b0f7c3ae69f816aec1fe0.jpg)

![52e566b444ebf4e1857c39168c91d235afdf9f85ff97b5f0fef0e7f253d6f0d7.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/52e566b444ebf4e1857c39168c91d235afdf9f85ff97b5f0fef0e7f253d6f0d7.jpg)

![646cdc5f1675c2f3bbec8b53c641faea6f1989a009a31e055cd93a5a0704140c.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/646cdc5f1675c2f3bbec8b53c641faea6f1989a009a31e055cd93a5a0704140c.jpg)

![6f18daacbe5f80e65f64385e7d43bd2e8beaef60da22577f11b53e774527bdfe.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/6f18daacbe5f80e65f64385e7d43bd2e8beaef60da22577f11b53e774527bdfe.jpg)

![6ff7037f9faa059daf22e3363f99ed7995a34915998d21476bcc3bec24dd9b69.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/6ff7037f9faa059daf22e3363f99ed7995a34915998d21476bcc3bec24dd9b69.jpg)

![83f54af2dd46ad06c386cd7e4c94d3346f00d7a760019f342e0fb1485e00a0bf.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/83f54af2dd46ad06c386cd7e4c94d3346f00d7a760019f342e0fb1485e00a0bf.jpg)

![901e549d50f2ac6ff688bdb23fd2f00c6d63049f6e42ac19ac56b96ff51089af.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/901e549d50f2ac6ff688bdb23fd2f00c6d63049f6e42ac19ac56b96ff51089af.jpg)

![9d904cc92584db8a72be2372695542fe15b210d9e5da33d2dfc43d526c3e10e2.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/9d904cc92584db8a72be2372695542fe15b210d9e5da33d2dfc43d526c3e10e2.jpg)

![b49ebeb675abbd92c9c9f30940d4426808c7fe9de7761e7c1691a15cdf9141e3.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/b49ebeb675abbd92c9c9f30940d4426808c7fe9de7761e7c1691a15cdf9141e3.jpg)

![f3db8f324b0271909000a37189392e72f95a9b968e47d4dd5a0d62b92ce587f0.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/images/f3db8f324b0271909000a37189392e72f95a9b968e47d4dd5a0d62b92ce587f0.jpg)

### Tables

![0f20faa2d1a43b07c7d361a9c0134c6ca138c676875a3428f654f83eac434a39.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/0f20faa2d1a43b07c7d361a9c0134c6ca138c676875a3428f654f83eac434a39.jpg)

![16b11faa99b922b3556663bbde088abd4888475ecf02b4f0b2495292cadada93.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/16b11faa99b922b3556663bbde088abd4888475ecf02b4f0b2495292cadada93.jpg)

![2397ec15fade243480b54197bbfff61e95623a8d359db2960b8b337a2a3fb76a.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/2397ec15fade243480b54197bbfff61e95623a8d359db2960b8b337a2a3fb76a.jpg)

![23ead006d738f0e97020fc2fdc06a07f5ec07d660849f28270506c793659d6ce.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/23ead006d738f0e97020fc2fdc06a07f5ec07d660849f28270506c793659d6ce.jpg)

![27551ff3a7482d07a272b5106da88f03e659604f7af89a5cb14bf5cdf814f43b.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/27551ff3a7482d07a272b5106da88f03e659604f7af89a5cb14bf5cdf814f43b.jpg)

![2b1eae1b579389d6b2aecf2f36b63b4e7181bbe6761b7381a45a1fcb51bca2e2.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/2b1eae1b579389d6b2aecf2f36b63b4e7181bbe6761b7381a45a1fcb51bca2e2.jpg)

![2b6b4582949cf4bce9abed502b1ce692e440c6b0140d1631646744e6440627d8.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/2b6b4582949cf4bce9abed502b1ce692e440c6b0140d1631646744e6440627d8.jpg)

![306afbc34a8e0a1ba4406bdf889c1f8d47eb038c64e08fceda275841b08d7074.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/306afbc34a8e0a1ba4406bdf889c1f8d47eb038c64e08fceda275841b08d7074.jpg)

![33db4c590e50dc41f6e3c0ea305d17629275b85e87ac91c9593d62e58a725c51.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/33db4c590e50dc41f6e3c0ea305d17629275b85e87ac91c9593d62e58a725c51.jpg)

![40cdf6dfd1beaa83fcd64e3bc86a4b25cae14ffbc8b27a23c57debbe4ec9e188.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/40cdf6dfd1beaa83fcd64e3bc86a4b25cae14ffbc8b27a23c57debbe4ec9e188.jpg)

![5397d8f8d9ea715d4b08907b2d310f1bb06eb46c200cf82ec72166166049c017.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/5397d8f8d9ea715d4b08907b2d310f1bb06eb46c200cf82ec72166166049c017.jpg)

![53bf0fd67563ced2066c5e913a095e4776b3f0d1d5a46057289e499710b73a20.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/53bf0fd67563ced2066c5e913a095e4776b3f0d1d5a46057289e499710b73a20.jpg)

![5fc7da142b41b05f8c0b84260a64fba309dcc02f3d772116f29ef49747ad7087.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/5fc7da142b41b05f8c0b84260a64fba309dcc02f3d772116f29ef49747ad7087.jpg)

![705520cc8ed867e583b436a90a9008cc078ae90b08abbb0e27b856fb0b61e5e0.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/705520cc8ed867e583b436a90a9008cc078ae90b08abbb0e27b856fb0b61e5e0.jpg)

![761c5f8dc52b6268c60fc9430437040076fd8701955f7e361ce1e940d67004e0.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/761c5f8dc52b6268c60fc9430437040076fd8701955f7e361ce1e940d67004e0.jpg)

![77d01752c9948eba0f04ec7e479932075f407d02bcff67b2047eb063f911dc80.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/77d01752c9948eba0f04ec7e479932075f407d02bcff67b2047eb063f911dc80.jpg)

![7e0deac3df86334b2176e198daaacb12f82f074dc9710cc84f30f160e94692e9.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/7e0deac3df86334b2176e198daaacb12f82f074dc9710cc84f30f160e94692e9.jpg)

![7ffa720b255c971d83575f0a1be28f2c8a09b858a3007eae46b0166f0992a2a1.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/7ffa720b255c971d83575f0a1be28f2c8a09b858a3007eae46b0166f0992a2a1.jpg)

![80677dea4bad936f3cff1c7b8a2b99b30cbbfaf2e37466ac05cce97986ff698e.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/80677dea4bad936f3cff1c7b8a2b99b30cbbfaf2e37466ac05cce97986ff698e.jpg)

![814d0fc040fe87d90a779c4153c70d0bd74389278113c3996bc88ddde9dc2e13.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/814d0fc040fe87d90a779c4153c70d0bd74389278113c3996bc88ddde9dc2e13.jpg)

![8d596c8612847de68e066c232d94078134daeff9026f203e95e2520d644e7b86.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/8d596c8612847de68e066c232d94078134daeff9026f203e95e2520d644e7b86.jpg)

![9d884a1b8a1e1049d4a7f01d53c9eb9de070a9bda3b620024e366407857154f8.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/9d884a1b8a1e1049d4a7f01d53c9eb9de070a9bda3b620024e366407857154f8.jpg)

![afb3c708b568ba698fb1a485f68c5e8b30e33cb379e6b74517023cdeaf7c77cf.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/afb3c708b568ba698fb1a485f68c5e8b30e33cb379e6b74517023cdeaf7c77cf.jpg)

![b7acdb15fa16c9c8f898436ee6ddfdc2a8976d1df61f168da6fbf82cffd27dd1.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/b7acdb15fa16c9c8f898436ee6ddfdc2a8976d1df61f168da6fbf82cffd27dd1.jpg)

![bd8650589080a6ca2f09c5e3b21b29e7c1ef780e39d87c59bbfefd461ab78105.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/bd8650589080a6ca2f09c5e3b21b29e7c1ef780e39d87c59bbfefd461ab78105.jpg)

![c410bc137347ecaae87f13049f71237fb891494cc4c4bf9e441a5bf09515486b.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/c410bc137347ecaae87f13049f71237fb891494cc4c4bf9e441a5bf09515486b.jpg)

![c6b295e3f4e11e2d6e3231ba3ff0d23baa08c11f8779c50bb1c9ec16baa92a03.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/c6b295e3f4e11e2d6e3231ba3ff0d23baa08c11f8779c50bb1c9ec16baa92a03.jpg)

![da2d37b2049ba2dcc4fd478207b0d41dd988f14948f845bc99b31725fc7f75e5.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/da2d37b2049ba2dcc4fd478207b0d41dd988f14948f845bc99b31725fc7f75e5.jpg)

![ec10365c4f172267cc673d1714be46969f8547398b23c24f4df140fd70bac56e.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/ec10365c4f172267cc673d1714be46969f8547398b23c24f4df140fd70bac56e.jpg)

![ed4aadee8d3ad33650448c728de7a29e4bcca1802c5f82da31d52d804f61401d.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/ed4aadee8d3ad33650448c728de7a29e4bcca1802c5f82da31d52d804f61401d.jpg)

![ef55924970fa4662a540b1b2145dccd48c64cd49bc65489a2f9ddb2d04ecb82e.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/ef55924970fa4662a540b1b2145dccd48c64cd49bc65489a2f9ddb2d04ecb82e.jpg)

![f9e20ee13e5bfaca1b7ddfde929216203f82cfcee8581fb4f16709120ae3bd97.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/f9e20ee13e5bfaca1b7ddfde929216203f82cfcee8581fb4f16709120ae3bd97.jpg)

![fa4e7f5e34b619478c8de0712f3e5d084015329be5185bb0b05fb056dac81fdf.jpg](../icml_results/6_TokSuite_%20Measuring%20the%20Impact%20of%20Tokenizer%20Choice%20on%20Language%20Model%20Behavior/tables/fa4e7f5e34b619478c8de0712f3e5d084015329be5185bb0b05fb056dac81fdf.jpg)

<a id="paper-0007"></a>

## Perceptrons and Localization of Attention’s Mean-Field Landscape


### Images

![059ea2b1bee50e0b22147d6deb5539f876f67b8f946975864617698c2c3a0995.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/059ea2b1bee50e0b22147d6deb5539f876f67b8f946975864617698c2c3a0995.jpg)

![5e817740ad308a49949aff2063d503084541eb4a118a8a93ebac5852a720825c.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/5e817740ad308a49949aff2063d503084541eb4a118a8a93ebac5852a720825c.jpg)

![72975d17cfb866ae7dc060ee4aa82bb1742e2296dbd5197e4627ead644794b62.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/72975d17cfb866ae7dc060ee4aa82bb1742e2296dbd5197e4627ead644794b62.jpg)

![8ab07588dd651a3cb0dbc2b0921a0db81c3ddece00504a328202c8bd11a42239.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/8ab07588dd651a3cb0dbc2b0921a0db81c3ddece00504a328202c8bd11a42239.jpg)

![8c64565ef84bea3a8e0e24fec07d54d957275ae6b48cce46224ba5a69f349398.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/8c64565ef84bea3a8e0e24fec07d54d957275ae6b48cce46224ba5a69f349398.jpg)

![92988d5b37b6f4c1e667b6e1c50f3d81b31177e4ee357c3233d877fdb11c8f89.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/92988d5b37b6f4c1e667b6e1c50f3d81b31177e4ee357c3233d877fdb11c8f89.jpg)

![93a06b2f15606e8b99cdd497fcccd577f2f2113800679fd65cd259c590636454.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/93a06b2f15606e8b99cdd497fcccd577f2f2113800679fd65cd259c590636454.jpg)

![b4ac873a725821736a414b0b71227589bfdd02f8e84b730f98ac3b04f57b23ae.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/b4ac873a725821736a414b0b71227589bfdd02f8e84b730f98ac3b04f57b23ae.jpg)

![b4d28f2237e328a3bb11616234013c37230e401472517edff74739246040f6ae.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/b4d28f2237e328a3bb11616234013c37230e401472517edff74739246040f6ae.jpg)

![c859e67ebc315e41dce4493dca4950cfe053badc3e0848d59d5aa8da89fdbef1.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/c859e67ebc315e41dce4493dca4950cfe053badc3e0848d59d5aa8da89fdbef1.jpg)

![db40d4bb316671fe6f9e7109bdd09abdddd80a5d575363681f9f11ed6588b212.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/db40d4bb316671fe6f9e7109bdd09abdddd80a5d575363681f9f11ed6588b212.jpg)

![ea0ab0c84191e8d256aa8e9a76afa223cd044b60427fce04fe810c8396e6be0f.jpg](../icml_results/7_Perceptrons%20and%20Localization%20of%20Attention%E2%80%99s%20Mean-Field%20Landscape/images/ea0ab0c84191e8d256aa8e9a76afa223cd044b60427fce04fe810c8396e6be0f.jpg)

<a id="paper-0008"></a>

## (Doubly) Exponential Lower Bounds for Follow the Regularized Leader in Potential Games


### Images

![3a8ba7a6ec279dc47f6bdfe4d82e1714914d274650b9b4edfdee4b9a365e5fd8.jpg](../icml_results/8_%28Doubly%29%20Exponential%20Lower%20Bounds%20for%20Follow%20the%20Regularized%20Leader%20in%20Potential%20Games/images/3a8ba7a6ec279dc47f6bdfe4d82e1714914d274650b9b4edfdee4b9a365e5fd8.jpg)

![6916ee9b64f1a9035b9482001e89707631e7973a1a35bec7af6a9510d8bc1e79.jpg](../icml_results/8_%28Doubly%29%20Exponential%20Lower%20Bounds%20for%20Follow%20the%20Regularized%20Leader%20in%20Potential%20Games/images/6916ee9b64f1a9035b9482001e89707631e7973a1a35bec7af6a9510d8bc1e79.jpg)

![884d4f2d795f6c0d7e8951b089cecec2dc1c5d61e9e03fcf6d68d418f061376b.jpg](../icml_results/8_%28Doubly%29%20Exponential%20Lower%20Bounds%20for%20Follow%20the%20Regularized%20Leader%20in%20Potential%20Games/images/884d4f2d795f6c0d7e8951b089cecec2dc1c5d61e9e03fcf6d68d418f061376b.jpg)

![b55b56e2f5a4e5383a6a2159dc98a59e97a5b9538a3b55c25bf7ca322a5c785c.jpg](../icml_results/8_%28Doubly%29%20Exponential%20Lower%20Bounds%20for%20Follow%20the%20Regularized%20Leader%20in%20Potential%20Games/images/b55b56e2f5a4e5383a6a2159dc98a59e97a5b9538a3b55c25bf7ca322a5c785c.jpg)

![d7f69c84e8acc059567d69bba09be0eb511b6425b321819ac4b9ee8d4d10168e.jpg](../icml_results/8_%28Doubly%29%20Exponential%20Lower%20Bounds%20for%20Follow%20the%20Regularized%20Leader%20in%20Potential%20Games/images/d7f69c84e8acc059567d69bba09be0eb511b6425b321819ac4b9ee8d4d10168e.jpg)

![d923b01f9b4f3fb746f52454342f5875ab1d84291dfd23e2d4fe54f6bc814205.jpg](../icml_results/8_%28Doubly%29%20Exponential%20Lower%20Bounds%20for%20Follow%20the%20Regularized%20Leader%20in%20Potential%20Games/images/d923b01f9b4f3fb746f52454342f5875ab1d84291dfd23e2d4fe54f6bc814205.jpg)

<a id="paper-0009"></a>

## Estimating Tail Risks in Language Model Output Distributions


### Images

![1635733702e2f138638f5320484d6a3e67ca7ad41ee207e2f7a884a49fdd6cee.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/1635733702e2f138638f5320484d6a3e67ca7ad41ee207e2f7a884a49fdd6cee.jpg)

![17dfcdf0cb35cef80d50f8917617226467793966615aa158b39a2f02abe08761.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/17dfcdf0cb35cef80d50f8917617226467793966615aa158b39a2f02abe08761.jpg)

![1f7ce273a1de642d65e2f360443dcb8a393b580c0a67a59c98918a4423bfb26b.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/1f7ce273a1de642d65e2f360443dcb8a393b580c0a67a59c98918a4423bfb26b.jpg)

![69231c8bab9c34a505ed00521bcdde0787de51a92b1cfd6f16875048a76382f5.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/69231c8bab9c34a505ed00521bcdde0787de51a92b1cfd6f16875048a76382f5.jpg)

![7a91396fa9870a4e3bc573dc232594068501d6f7234b3dcdc65f43256b60de08.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/7a91396fa9870a4e3bc573dc232594068501d6f7234b3dcdc65f43256b60de08.jpg)

![7beefe69b9cf7b965bc9a3a79d992408bd83000a4355b0729e7ed1e28caa08fd.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/7beefe69b9cf7b965bc9a3a79d992408bd83000a4355b0729e7ed1e28caa08fd.jpg)

![aac340f8a4d2a2719441bf8f037d1e6a849a7eb2305ab5ebeb309413c32932c6.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/aac340f8a4d2a2719441bf8f037d1e6a849a7eb2305ab5ebeb309413c32932c6.jpg)

![d40e2f352a1450145b1e1399a48adc5b22c1bcb1b133a22578dc68672e9d120d.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/d40e2f352a1450145b1e1399a48adc5b22c1bcb1b133a22578dc68672e9d120d.jpg)

![dd2740d7642ce27d29c89531d907e9a80569f26a7926af1e788673e438b858d3.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/dd2740d7642ce27d29c89531d907e9a80569f26a7926af1e788673e438b858d3.jpg)

![f09f2c8edc17c1bd25df21934c9fef0a409eefcd949ebbe018e342dd038769e3.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/images/f09f2c8edc17c1bd25df21934c9fef0a409eefcd949ebbe018e342dd038769e3.jpg)

### Tables

![0c586c8312fcc57b889c1d704f6b24ee28ed7e0216155968e18a27b870fa3793.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/tables/0c586c8312fcc57b889c1d704f6b24ee28ed7e0216155968e18a27b870fa3793.jpg)

![0d8b41359ab003daf02fb45b7ca3e3ad13d8c24a457486e3995d566dfb6e4ac2.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/tables/0d8b41359ab003daf02fb45b7ca3e3ad13d8c24a457486e3995d566dfb6e4ac2.jpg)

![821ade8048b1d509a604ae56cfa9756344c147a91036c393149b54c88ad02c85.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/tables/821ade8048b1d509a604ae56cfa9756344c147a91036c393149b54c88ad02c85.jpg)

![c97c01bd331f2e3faba946b1ead78883c4a0227c55756e488540e247312a7737.jpg](../icml_results/9_Estimating%20Tail%20Risks%20in%20Language%20Model%20Output%20Distributions/tables/c97c01bd331f2e3faba946b1ead78883c4a0227c55756e488540e247312a7737.jpg)

<a id="paper-0010"></a>

## Language Generation in the Limit: Complexity Barriers and Implications for Learning


### Images

![06e108ded30ecbf66a9d82d3f913a4f56a2fb931dac7be663c05d29e24fed8a2.jpg](../icml_results/10_Language%20Generation%20in%20the%20Limit_%20Complexity%20Barriers%20and%20Implications%20for%20Learning/images/06e108ded30ecbf66a9d82d3f913a4f56a2fb931dac7be663c05d29e24fed8a2.jpg)

![135cd98b8a4c72da6655dfc356980a057ae04de3303aa3b023d8a38a5de1071c.jpg](../icml_results/10_Language%20Generation%20in%20the%20Limit_%20Complexity%20Barriers%20and%20Implications%20for%20Learning/images/135cd98b8a4c72da6655dfc356980a057ae04de3303aa3b023d8a38a5de1071c.jpg)

![a506848f7b49141045c5a989b21ee20e4cace78ca9425dfe40161e3f16719c80.jpg](../icml_results/10_Language%20Generation%20in%20the%20Limit_%20Complexity%20Barriers%20and%20Implications%20for%20Learning/images/a506848f7b49141045c5a989b21ee20e4cace78ca9425dfe40161e3f16719c80.jpg)

![c2913e791c9bc3a29c8af05013d2d80bfbf8defc1117615210975a39de3f6085.jpg](../icml_results/10_Language%20Generation%20in%20the%20Limit_%20Complexity%20Barriers%20and%20Implications%20for%20Learning/images/c2913e791c9bc3a29c8af05013d2d80bfbf8defc1117615210975a39de3f6085.jpg)

<a id="paper-0011"></a>

## Language Model Circuits Are Sparse in the Neuron Basis


### Images

![0fcf640fb721fb4f70a765c4f065283e1be11f6a005190dbd000f4729052ac28.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/0fcf640fb721fb4f70a765c4f065283e1be11f6a005190dbd000f4729052ac28.jpg)

![26c8f115a91c03172f049dded17fbf3fd2caaaf360b99b337c7b820e260b0d13.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/26c8f115a91c03172f049dded17fbf3fd2caaaf360b99b337c7b820e260b0d13.jpg)

![2b458e0e4e9b13cb535609e26d90be279b329b082a1c2e429ae8f7c355f1eed3.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/2b458e0e4e9b13cb535609e26d90be279b329b082a1c2e429ae8f7c355f1eed3.jpg)

![3c6b268366233a5e0ab63de3a2065091c3cd1bb6c13ba6487115bc2bc9a9ae7c.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/3c6b268366233a5e0ab63de3a2065091c3cd1bb6c13ba6487115bc2bc9a9ae7c.jpg)

![43a1f0077c26646f77a76af7c4371e968dcdb838ca9babd10aa2179751673d98.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/43a1f0077c26646f77a76af7c4371e968dcdb838ca9babd10aa2179751673d98.jpg)

![549b1c840a23bc2d1507289c90ad526dd4d465478b27c6e0079034342ac9d8a8.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/549b1c840a23bc2d1507289c90ad526dd4d465478b27c6e0079034342ac9d8a8.jpg)

![56cd11def7cbe04596b8287330156513e13cb81dbdb7ba3c06c8ddc91c134a50.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/56cd11def7cbe04596b8287330156513e13cb81dbdb7ba3c06c8ddc91c134a50.jpg)

![64d78d37c903cf007e1496f9dfe02134c2036de0b462268b657f03b9482885f3.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/64d78d37c903cf007e1496f9dfe02134c2036de0b462268b657f03b9482885f3.jpg)

![82c124d03174095fe07d76d65714d6ab213b1d661667dce5ad748d2e7c767942.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/82c124d03174095fe07d76d65714d6ab213b1d661667dce5ad748d2e7c767942.jpg)

![8ce55e7a62db329bb3cb581b4d3052a12cf67ab64fd6051620f9fdcfce9d8d09.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/8ce55e7a62db329bb3cb581b4d3052a12cf67ab64fd6051620f9fdcfce9d8d09.jpg)

![8d0f541e2e2304f1837c8311833d03178206c7d3df7fcf4a9b585d1177e76931.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/8d0f541e2e2304f1837c8311833d03178206c7d3df7fcf4a9b585d1177e76931.jpg)

![929c36b4f35931186515be174dd1a82c411010296c60789653e47b7e13441392.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/929c36b4f35931186515be174dd1a82c411010296c60789653e47b7e13441392.jpg)

![9e5957bf8d4bc0cf571c714d105cb17845ec328422728eed8b908a5e37baff44.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/9e5957bf8d4bc0cf571c714d105cb17845ec328422728eed8b908a5e37baff44.jpg)

![bebb480d25701a0d622f1c42d3a5e08c4dd2c117783dfa3cf7d77578ce51ae7d.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/bebb480d25701a0d622f1c42d3a5e08c4dd2c117783dfa3cf7d77578ce51ae7d.jpg)

![dfa0302eedaabc180aaab2873e046f9c68ad82b3c1873acee4b8bb8d41b8cd2d.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/dfa0302eedaabc180aaab2873e046f9c68ad82b3c1873acee4b8bb8d41b8cd2d.jpg)

![f093e47bffd08640d7b8baa49496db16d16833267557d81c7cd3c543a8c9c673.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/f093e47bffd08640d7b8baa49496db16d16833267557d81c7cd3c543a8c9c673.jpg)

![f64cf2db2eeaf541e9d6007c7d6204960bb8e3d35429a07e983722681726a31c.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/f64cf2db2eeaf541e9d6007c7d6204960bb8e3d35429a07e983722681726a31c.jpg)

![fa28a851d2c01d649123498c37dc8946cf8d09018b03c5ac54ccc8b30445c99d.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/images/fa28a851d2c01d649123498c37dc8946cf8d09018b03c5ac54ccc8b30445c99d.jpg)

### Tables

![064408ac353e8d8e385bb8f76ada751c09300118e5f8c5dd3931437ff0642cc8.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/064408ac353e8d8e385bb8f76ada751c09300118e5f8c5dd3931437ff0642cc8.jpg)

![0b8b615915c17e57ad4ad52d69922156590890674b8bc6f942033fdc0cd3edc8.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/0b8b615915c17e57ad4ad52d69922156590890674b8bc6f942033fdc0cd3edc8.jpg)

![0bb177abb9d058591a5959f331301d58f8ea666f13dd594f812ea191c3f3127c.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/0bb177abb9d058591a5959f331301d58f8ea666f13dd594f812ea191c3f3127c.jpg)

![15f646b714ae452c803e38637b674fff7d81e4300966966e720e0084de23f87b.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/15f646b714ae452c803e38637b674fff7d81e4300966966e720e0084de23f87b.jpg)

![1a29f6ac542ec62560f1baa41d20117f481f27fc18d45110f7e621a75102870e.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/1a29f6ac542ec62560f1baa41d20117f481f27fc18d45110f7e621a75102870e.jpg)

![210a17ef1c0939f55cacacf9336b8ada0ff5f72a463b972bc227cf730c118cd8.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/210a17ef1c0939f55cacacf9336b8ada0ff5f72a463b972bc227cf730c118cd8.jpg)

![248ecfb5523fa8e6ca79e4ea7ce2ca4ab8db5466162e731dc347ad4923d19994.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/248ecfb5523fa8e6ca79e4ea7ce2ca4ab8db5466162e731dc347ad4923d19994.jpg)

![26666243931671ef24a78de51e8c8d0bba69845c08cbb7da61a68d996e0b45aa.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/26666243931671ef24a78de51e8c8d0bba69845c08cbb7da61a68d996e0b45aa.jpg)

![334e6153f21fe9f4f4418ae8e15d778d5a61b55f2103b809053754f574d041d9.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/334e6153f21fe9f4f4418ae8e15d778d5a61b55f2103b809053754f574d041d9.jpg)

![3cf300c4e373c0ed7e1d6d1aab8dc5b6f426199484cfd7f7898e44fbac608b21.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/3cf300c4e373c0ed7e1d6d1aab8dc5b6f426199484cfd7f7898e44fbac608b21.jpg)

![3d65a7b55f110181d0a04f3e13fb0210f9cb6d2466f025f56b5cf399368a8466.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/3d65a7b55f110181d0a04f3e13fb0210f9cb6d2466f025f56b5cf399368a8466.jpg)

![3fca07d8c7ca08b9fcb12ef7a5df21cb08c1fff2a07542a19eaff073454a0505.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/3fca07d8c7ca08b9fcb12ef7a5df21cb08c1fff2a07542a19eaff073454a0505.jpg)

![40b76a54e6daab8e0b8ef290d6ce921bb21aae5cbb7484362c31796e05c27d3f.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/40b76a54e6daab8e0b8ef290d6ce921bb21aae5cbb7484362c31796e05c27d3f.jpg)

![44d262a8760dc4eabb394338bfd5f1bb989ee5fb4b665b835cf7a5c38c6fa2ae.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/44d262a8760dc4eabb394338bfd5f1bb989ee5fb4b665b835cf7a5c38c6fa2ae.jpg)

![4e041921536603bfde475cb04b12a05c6c8f6d19f618d72e708a79767d61c512.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/4e041921536603bfde475cb04b12a05c6c8f6d19f618d72e708a79767d61c512.jpg)

![55a01c2dfc1b9b04b6704cb63dd198863203eb85ed8b8c7826131fe720deb5f1.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/55a01c2dfc1b9b04b6704cb63dd198863203eb85ed8b8c7826131fe720deb5f1.jpg)

![58c9fa6a8a7e829846a776a47e65eccd355a17ba73bd1416522998607d4414ad.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/58c9fa6a8a7e829846a776a47e65eccd355a17ba73bd1416522998607d4414ad.jpg)

![5bb5e9d7a98ddaae2182cfa1e7a1489952278c71f20f1130ad20fee9dc70f60a.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/5bb5e9d7a98ddaae2182cfa1e7a1489952278c71f20f1130ad20fee9dc70f60a.jpg)

![5c40986fd1ab527c66383b8db86a327a4af1c592de060168fe8fb99157f60623.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/5c40986fd1ab527c66383b8db86a327a4af1c592de060168fe8fb99157f60623.jpg)

![5c8e804ceb9a86d573cb22e77adefdac7c411bfaef940d6930d953a8fe6c8a17.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/5c8e804ceb9a86d573cb22e77adefdac7c411bfaef940d6930d953a8fe6c8a17.jpg)

![60d062030537de025adfcd84b7e4bf32a78757e55463b2d133b4c509d2201f7b.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/60d062030537de025adfcd84b7e4bf32a78757e55463b2d133b4c509d2201f7b.jpg)

![67d1c5e5d57a6f43d7cd4793db2631b0cd9494b52869d30808607d4395e9062b.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/67d1c5e5d57a6f43d7cd4793db2631b0cd9494b52869d30808607d4395e9062b.jpg)

![6b85733230b73d1788f620ef866702fefc8cbffb274f75d4dc7f6cbad0338fdb.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/6b85733230b73d1788f620ef866702fefc8cbffb274f75d4dc7f6cbad0338fdb.jpg)

![74b91613e8c7c34229afcdeee8ffc75a91e367330e7d76c079302b81bdb1fe6e.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/74b91613e8c7c34229afcdeee8ffc75a91e367330e7d76c079302b81bdb1fe6e.jpg)

![78faa4c4a9c53c6593fa1a4eed7063d8216b840f783148bb4d7a28ff0b107300.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/78faa4c4a9c53c6593fa1a4eed7063d8216b840f783148bb4d7a28ff0b107300.jpg)

![810e34d2ce7d9969b7b29034e2789238bc74a7bc7e17f462cca7566d221baf6d.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/810e34d2ce7d9969b7b29034e2789238bc74a7bc7e17f462cca7566d221baf6d.jpg)

![8327638d1f06dc4692c9d05160519b534c9613ee066b6f05817ea307afb03f87.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/8327638d1f06dc4692c9d05160519b534c9613ee066b6f05817ea307afb03f87.jpg)

![84b0c9094ccb8947a2f3a6ac727541799fcc292d0c4288cdb0fec529c2c27682.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/84b0c9094ccb8947a2f3a6ac727541799fcc292d0c4288cdb0fec529c2c27682.jpg)

![86419c5b8c460ea42cf4d851db5bae742e0907ee074dc20cbdc2ddd314e3951e.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/86419c5b8c460ea42cf4d851db5bae742e0907ee074dc20cbdc2ddd314e3951e.jpg)

![88438a5fab34bfbe43edc497a8a74be2fcf33c00e9a7b6079247416bec3cedef.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/88438a5fab34bfbe43edc497a8a74be2fcf33c00e9a7b6079247416bec3cedef.jpg)

![896572b5540aa715c52b6143e3521fe93cbff5ec43d8f831bbaaad497ecbc580.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/896572b5540aa715c52b6143e3521fe93cbff5ec43d8f831bbaaad497ecbc580.jpg)

![992ff13586c04bd2b7802a6750099d4ed5554ce0193776fa1ee3d90b7560b111.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/992ff13586c04bd2b7802a6750099d4ed5554ce0193776fa1ee3d90b7560b111.jpg)

![9a1a9d36f60bc488c4ca5f0cf34fba74827bf8d54ce1ca2b59cedd82932be79b.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/9a1a9d36f60bc488c4ca5f0cf34fba74827bf8d54ce1ca2b59cedd82932be79b.jpg)

![9aa4318943801a565be23aa10a090ec4f05e886c83146a64313afecfb9edeac8.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/9aa4318943801a565be23aa10a090ec4f05e886c83146a64313afecfb9edeac8.jpg)

![9b497c6bec17f2e24b7d7188b50fc631817380c298b5e31236558adc3f51187f.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/9b497c6bec17f2e24b7d7188b50fc631817380c298b5e31236558adc3f51187f.jpg)

![9c554c7da5728c112574e9f6010b2aeb0cd7566f5a0f2a0f8731aab722811e5b.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/9c554c7da5728c112574e9f6010b2aeb0cd7566f5a0f2a0f8731aab722811e5b.jpg)

![a6168ae670660c82ce9cbef5ae3bc6d1f750df71b24c01983279547f732d36be.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/a6168ae670660c82ce9cbef5ae3bc6d1f750df71b24c01983279547f732d36be.jpg)

![b29950e7459cb5e87f1250ed5eaff4d53c0aadc77b5b005b890562d7481da7be.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/b29950e7459cb5e87f1250ed5eaff4d53c0aadc77b5b005b890562d7481da7be.jpg)

![b2e29cd7f720386e8f59484eb0bf539ef41f966fcfa59c108bb75f5aca989231.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/b2e29cd7f720386e8f59484eb0bf539ef41f966fcfa59c108bb75f5aca989231.jpg)

![bb502218454ac9ed6ac6e2130d903213a140f93b55106bcdbc9e0cb4e09e1c00.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/bb502218454ac9ed6ac6e2130d903213a140f93b55106bcdbc9e0cb4e09e1c00.jpg)

![bb5284f285ede9cd91838e1c44560682c3edbe581d5ee1c5c96fdc1d39295566.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/bb5284f285ede9cd91838e1c44560682c3edbe581d5ee1c5c96fdc1d39295566.jpg)

![c33687374c1e3fe8b5d87971245ff5e9a46390dcbd5340ce7b04db21d434e031.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/c33687374c1e3fe8b5d87971245ff5e9a46390dcbd5340ce7b04db21d434e031.jpg)

![c4576ed59dbdfa613f746c1d7052f24ddf830ffa90d07061dbce5d8316de3b6b.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/c4576ed59dbdfa613f746c1d7052f24ddf830ffa90d07061dbce5d8316de3b6b.jpg)

![ca0bfdbbc40bfce04661377dc18308f652cf893499e99323ef30521a53ea858e.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/ca0bfdbbc40bfce04661377dc18308f652cf893499e99323ef30521a53ea858e.jpg)

![d0eff46b397c977f69d35020a69ed1847e5271121085fc001245984ff6be8bf1.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/d0eff46b397c977f69d35020a69ed1847e5271121085fc001245984ff6be8bf1.jpg)

![d2384804eba0a25798c791a692751cebfaa0d1c8e98ff59e8ee9ab09025b19dc.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/d2384804eba0a25798c791a692751cebfaa0d1c8e98ff59e8ee9ab09025b19dc.jpg)

![daf4dfe7846eded15a17d2cc78980384e98839d2d298ab49332a640f255eb64e.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/daf4dfe7846eded15a17d2cc78980384e98839d2d298ab49332a640f255eb64e.jpg)

![ec1e1cb205f60bac9037335d5fede93afb1adf1ae3c2907aae4d1185df798e38.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/ec1e1cb205f60bac9037335d5fede93afb1adf1ae3c2907aae4d1185df798e38.jpg)

![eca44f05424ccb5632c6128ad74cd9da6f80cc43247170d95ba0e728f20e90a8.jpg](../icml_results/11_Language%20Model%20Circuits%20Are%20Sparse%20in%20the%20Neuron%20Basis/tables/eca44f05424ccb5632c6128ad74cd9da6f80cc43247170d95ba0e728f20e90a8.jpg)

<a id="paper-0012"></a>

## Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch


### Images

![0b86b39197e9ea2cf942f63ada50bd75c2c518234022ffd3076a4066251de573.jpg](../icml_results/12_Provable%20Accuracy%20Collapse%20in%20Embedding-Based%20Representations%20under%20Dimensionality%20Mismatch/images/0b86b39197e9ea2cf942f63ada50bd75c2c518234022ffd3076a4066251de573.jpg)

![1439cbd1f702ad0334adc43b18827c5469b6919264a0bb380868938d061ea844.jpg](../icml_results/12_Provable%20Accuracy%20Collapse%20in%20Embedding-Based%20Representations%20under%20Dimensionality%20Mismatch/images/1439cbd1f702ad0334adc43b18827c5469b6919264a0bb380868938d061ea844.jpg)

![3dfd1ee05ece808a5038416632649e0d35a895b945a7602a05efa6173f6ea874.jpg](../icml_results/12_Provable%20Accuracy%20Collapse%20in%20Embedding-Based%20Representations%20under%20Dimensionality%20Mismatch/images/3dfd1ee05ece808a5038416632649e0d35a895b945a7602a05efa6173f6ea874.jpg)

![719a24aa3bc5c0aec24535d9c4ede1f9460d1068fe190989095bb825da1b6074.jpg](../icml_results/12_Provable%20Accuracy%20Collapse%20in%20Embedding-Based%20Representations%20under%20Dimensionality%20Mismatch/images/719a24aa3bc5c0aec24535d9c4ede1f9460d1068fe190989095bb825da1b6074.jpg)

<a id="paper-0013"></a>

## VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization


### Images

![125126029e1db576d066b6536fca40e625792d925a9bf77fa4c313353ec029ed.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/125126029e1db576d066b6536fca40e625792d925a9bf77fa4c313353ec029ed.jpg)

![20005641557eea2a9254f4e526d3f76cb8da5cde0004e5c5351f7b0ea23016f1.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/20005641557eea2a9254f4e526d3f76cb8da5cde0004e5c5351f7b0ea23016f1.jpg)

![45f3e8d85956c6d43a71278318cf78cf511780f2442a29b5f937d7fe947b729f.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/45f3e8d85956c6d43a71278318cf78cf511780f2442a29b5f937d7fe947b729f.jpg)

![4b364706cc8cefe56416735b3c8eea48532f53959bb565132f0437e1f9d5b97c.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/4b364706cc8cefe56416735b3c8eea48532f53959bb565132f0437e1f9d5b97c.jpg)

![58b241c6c9f83deb6ede469a7d7687f15242611f862e3adbb8056f90a7c1b3e0.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/58b241c6c9f83deb6ede469a7d7687f15242611f862e3adbb8056f90a7c1b3e0.jpg)

![7097c842fe47bd3ff398f16343f67c37e0ad4d48205bf8aadf6b47617a5dbb69.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/7097c842fe47bd3ff398f16343f67c37e0ad4d48205bf8aadf6b47617a5dbb69.jpg)

![73ffd69091b91bb22fe9164611a6406e494ea672e74fbb47a44cc7c5f4d4d464.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/73ffd69091b91bb22fe9164611a6406e494ea672e74fbb47a44cc7c5f4d4d464.jpg)

![841a268d7a9b18dfcd6f9e676cf168e46c5abddec1f6297df3ae8090aaf5c231.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/841a268d7a9b18dfcd6f9e676cf168e46c5abddec1f6297df3ae8090aaf5c231.jpg)

![89f736907a5ebf40beb380ae64dc3f71e336cb5df43ab02cdf9e90552d8036c6.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/89f736907a5ebf40beb380ae64dc3f71e336cb5df43ab02cdf9e90552d8036c6.jpg)

![c616dac8c2fd1ec6974f09c9bb6a31a20f55537df6c1afaf49d5ef35588cd1da.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/c616dac8c2fd1ec6974f09c9bb6a31a20f55537df6c1afaf49d5ef35588cd1da.jpg)

![c98f6e05900276be1e23d4dfb3c9e2bec45dec667129d0549f7a3172c901cf2d.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/c98f6e05900276be1e23d4dfb3c9e2bec45dec667129d0549f7a3172c901cf2d.jpg)

![cc6bb19f71053beae8f6940d5bbafacc35b6d0c16acb10e8ac09cf35b11b5487.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/cc6bb19f71053beae8f6940d5bbafacc35b6d0c16acb10e8ac09cf35b11b5487.jpg)

![d17900992fa6d4bce5cdc0fd79a1c22fe5ebfea6ad4812b3bcead90e671608af.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/d17900992fa6d4bce5cdc0fd79a1c22fe5ebfea6ad4812b3bcead90e671608af.jpg)

![d2c8c7d0f9f5a8a9c48aeda04f4cb4775d6d9e4b3f4a1d88eb11136dc3e5a8fd.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/d2c8c7d0f9f5a8a9c48aeda04f4cb4775d6d9e4b3f4a1d88eb11136dc3e5a8fd.jpg)

![e231f53c3290d0451e2443cc381ffd1690a24519713f58ed90d697c4b09b5613.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/e231f53c3290d0451e2443cc381ffd1690a24519713f58ed90d697c4b09b5613.jpg)

![e2e49afbda8c32af5dd2e4ec25773407c020162599a0b2e08d52ad60a1703598.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/e2e49afbda8c32af5dd2e4ec25773407c020162599a0b2e08d52ad60a1703598.jpg)

![e93ef7423a5e488785a5a1791b080af712a883fdbf08e2d8f1c0c27acb573527.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/e93ef7423a5e488785a5a1791b080af712a883fdbf08e2d8f1c0c27acb573527.jpg)

![fed3d9807c66cef73955cee4f8572a424b4a3008a851c9ee8768a2bcf8a44adf.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/images/fed3d9807c66cef73955cee4f8572a424b4a3008a851c9ee8768a2bcf8a44adf.jpg)

### Tables

![4eac5f8f391c64ebf021220749ec70095d27560d58423fd53767f632ea3f2641.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/4eac5f8f391c64ebf021220749ec70095d27560d58423fd53767f632ea3f2641.jpg)

![7e3692f97976554a26647a1e34234aeb8a5560fc3490e766153f89ad3a21f2f5.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/7e3692f97976554a26647a1e34234aeb8a5560fc3490e766153f89ad3a21f2f5.jpg)

![84ce60e43e21d7d33474ef3f8b07a84477d52821dc7aed01d5a71049b66b8e0a.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/84ce60e43e21d7d33474ef3f8b07a84477d52821dc7aed01d5a71049b66b8e0a.jpg)

![85c0ce545d6183fde4696ff3a74ff2f694e9df6bc79d28f0492274b7bff75630.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/85c0ce545d6183fde4696ff3a74ff2f694e9df6bc79d28f0492274b7bff75630.jpg)

![85c3b32b95efb93568c89b724bee6592f831600fa3ad9047323218181e8c4d70.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/85c3b32b95efb93568c89b724bee6592f831600fa3ad9047323218181e8c4d70.jpg)

![a8ed1d551aae41401669fb1ccd9f61c1b650cda8de0647a9c0c1e805c8f78c88.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/a8ed1d551aae41401669fb1ccd9f61c1b650cda8de0647a9c0c1e805c8f78c88.jpg)

![aac5f8793ec94362539ed250a36b35ba2d49098653b3e138ae8805bc57fd7cf6.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/aac5f8793ec94362539ed250a36b35ba2d49098653b3e138ae8805bc57fd7cf6.jpg)

![ceabe2d86e84ea7d0795264b21c7fb5e5337bab8020ad9b1a84890f4e42db530.jpg](../icml_results/13_VideoFlexTok_%20Flexible-Length%20Coarse-to-Fine%20Video%20Tokenization/tables/ceabe2d86e84ea7d0795264b21c7fb5e5337bab8020ad9b1a84890f4e42db530.jpg)

<a id="paper-0014"></a>

## Falling Trees: A Model Class for Interpretable Risk Prioritization


### Images

![448c27d694f0f35350a1406b733eb4d12c8041417cad786e5bf88b452d09b60d.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/images/448c27d694f0f35350a1406b733eb4d12c8041417cad786e5bf88b452d09b60d.jpg)

![88b1a1de5a2fdee401d8063ff080b2458e40e14e60f14cec2d251b7ff1098a6a.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/images/88b1a1de5a2fdee401d8063ff080b2458e40e14e60f14cec2d251b7ff1098a6a.jpg)

![b4fff6cef91dd41d9faca311f3ea2dc96eab99d28bf9b49c57fc59f75d1fcd67.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/images/b4fff6cef91dd41d9faca311f3ea2dc96eab99d28bf9b49c57fc59f75d1fcd67.jpg)

![f3308d99bf92892dde32c0602beb962b9d8c2cf3bfd766a8353cb3220cb14fe4.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/images/f3308d99bf92892dde32c0602beb962b9d8c2cf3bfd766a8353cb3220cb14fe4.jpg)

![fc51bd72a7aef90dde2e8a2b1abc1dd54c290c8ffc20e71122f36d297ddcf13b.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/images/fc51bd72a7aef90dde2e8a2b1abc1dd54c290c8ffc20e71122f36d297ddcf13b.jpg)

### Tables

![007644505bc9c327e0447ac7a02288174c1aebd805d16ed13d57e5a5f91ba7a0.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/007644505bc9c327e0447ac7a02288174c1aebd805d16ed13d57e5a5f91ba7a0.jpg)

![0617ed949b509530b637e8a9588637443a654de61cc3cdee92ae5e230d826817.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/0617ed949b509530b637e8a9588637443a654de61cc3cdee92ae5e230d826817.jpg)

![10518678f798370672323d4b2e2419cb3d44b9d27229dbc8e1f46a164649dc77.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/10518678f798370672323d4b2e2419cb3d44b9d27229dbc8e1f46a164649dc77.jpg)

![16f7655d32b94f3b5f252891f2adaece6ddb3894a90bb1e65863cd609b382d10.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/16f7655d32b94f3b5f252891f2adaece6ddb3894a90bb1e65863cd609b382d10.jpg)

![22e637a5bd09a67f097756658fc4e99cf677ff8933a6f0badebc369d0b8839ac.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/22e637a5bd09a67f097756658fc4e99cf677ff8933a6f0badebc369d0b8839ac.jpg)

![28241dc0d5cc13ac84c2728d1d365e581d64c5ff7daa5e26c7522a595d9ff65d.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/28241dc0d5cc13ac84c2728d1d365e581d64c5ff7daa5e26c7522a595d9ff65d.jpg)

![2a0faa9911d38aaa3692f008db215e016966bfe79a84b1ad59eb4431945771d0.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/2a0faa9911d38aaa3692f008db215e016966bfe79a84b1ad59eb4431945771d0.jpg)

![33f0a9a04139eee4129d838245e60a2a7fd3e878c8289d5174e7ddb135e66d64.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/33f0a9a04139eee4129d838245e60a2a7fd3e878c8289d5174e7ddb135e66d64.jpg)

![46c49888db8f7f9c107f27d4b5d067c0c868f27f991bb86bcdf4e29e3abec906.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/46c49888db8f7f9c107f27d4b5d067c0c868f27f991bb86bcdf4e29e3abec906.jpg)

![4e8f0b68f8999b133c95f4f05ff3992928ee3c2208c2bc8b8f7310a49e2bee66.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/4e8f0b68f8999b133c95f4f05ff3992928ee3c2208c2bc8b8f7310a49e2bee66.jpg)

![52866fb6647de81fd90bcb9ac80258fd0820bd1f01edb5f85693c2bc21a140fa.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/52866fb6647de81fd90bcb9ac80258fd0820bd1f01edb5f85693c2bc21a140fa.jpg)

![5658ad5b2c7a57f9f3ad71ec5fb583985b9e007a4cf606d448ee5caea45c827d.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/5658ad5b2c7a57f9f3ad71ec5fb583985b9e007a4cf606d448ee5caea45c827d.jpg)

![5848bea8b849104384165f7daac8462875c31e916cdc9e4977fe6f59d3891add.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/5848bea8b849104384165f7daac8462875c31e916cdc9e4977fe6f59d3891add.jpg)

![5e4e81583702446cc83dffdb77e98f0fd4ad59395ecaaa490999144d2c3504ba.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/5e4e81583702446cc83dffdb77e98f0fd4ad59395ecaaa490999144d2c3504ba.jpg)

![5f722251397f66dbde10cd06ebe43cbe4e7c25e066ceab5674645747d57364e9.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/5f722251397f66dbde10cd06ebe43cbe4e7c25e066ceab5674645747d57364e9.jpg)

![5fd5a662b872fe12d9e823f09e163a9705483c17a762f888edea217387ccaa53.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/5fd5a662b872fe12d9e823f09e163a9705483c17a762f888edea217387ccaa53.jpg)

![86839def2afbcdbc05ab956ae7167b5fb701451e21f4f91e66800e04a6f9a086.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/86839def2afbcdbc05ab956ae7167b5fb701451e21f4f91e66800e04a6f9a086.jpg)

![8e2a3485c2fe94d185783d89c9238fc675a534308ef260818ed7d4c34e641e71.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/8e2a3485c2fe94d185783d89c9238fc675a534308ef260818ed7d4c34e641e71.jpg)

![bd4c5c325c058828e9e8e278021f3484fa51cc5a059a5a6b9f3201f743db9f61.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/bd4c5c325c058828e9e8e278021f3484fa51cc5a059a5a6b9f3201f743db9f61.jpg)

![c8f627ee65fa576202cd68d55eaf98090a626f6769bf052ddd8236db01c26173.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/c8f627ee65fa576202cd68d55eaf98090a626f6769bf052ddd8236db01c26173.jpg)

![d1abd795d9e21387bb9ff1ae77962db15f025138a9670b5c1c4a33f136c0d961.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/d1abd795d9e21387bb9ff1ae77962db15f025138a9670b5c1c4a33f136c0d961.jpg)

![ead933cf0ca981af7909f9943621e754809b2738dfe5a4278fa359ff5c42225c.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/ead933cf0ca981af7909f9943621e754809b2738dfe5a4278fa359ff5c42225c.jpg)

![f037ab124bf18456e44b69172e95c217160be7d0278e7ea2397bacc55b800580.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/f037ab124bf18456e44b69172e95c217160be7d0278e7ea2397bacc55b800580.jpg)

![f90c4c24d37ad51c00d3d7f799c72cfc9e7cb892434549cc63b7588c6bcef716.jpg](../icml_results/14_Falling%20Trees_%20A%20Model%20Class%20for%20Interpretable%20Risk%20Prioritization/tables/f90c4c24d37ad51c00d3d7f799c72cfc9e7cb892434549cc63b7588c6bcef716.jpg)

<a id="paper-0015"></a>

## Learning to Theorize the World from Observation


### Images

![0aff52fbd4f289d280ca7f9b9bc31bbd4c6ee618b7cdb8755a8079e56b058edc.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/0aff52fbd4f289d280ca7f9b9bc31bbd4c6ee618b7cdb8755a8079e56b058edc.jpg)

![1b83e1825c0d826264848087d329fa125a5e267316504240b67a1628e3ec192e.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/1b83e1825c0d826264848087d329fa125a5e267316504240b67a1628e3ec192e.jpg)

![1c09b2ed9300746f1a14e2fa9310dfa6dcb2051d041b0181749655437f2340cf.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/1c09b2ed9300746f1a14e2fa9310dfa6dcb2051d041b0181749655437f2340cf.jpg)

![42584022586f722039bedf05f5111a408bd6bc1686da3644c62a763fe4dac884.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/42584022586f722039bedf05f5111a408bd6bc1686da3644c62a763fe4dac884.jpg)

![636f64f0ecc260b0247b52cd68cdf967ca96b3f5fce5de7e378f0cd986e7f90b.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/636f64f0ecc260b0247b52cd68cdf967ca96b3f5fce5de7e378f0cd986e7f90b.jpg)

![649ed5a5a32e8c88d133ad6ddc5037aeb2699db025b1daa30628941f30e7d996.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/649ed5a5a32e8c88d133ad6ddc5037aeb2699db025b1daa30628941f30e7d996.jpg)

![66ff509208310a18946eca28d2f2f8b3056b84ea041a63fea4ac97ac45d459a7.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/66ff509208310a18946eca28d2f2f8b3056b84ea041a63fea4ac97ac45d459a7.jpg)

![6db722c73c21fe1c179f28008d5ae3b2bbfabc2bf209b6b79a19e503d45da7c7.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/6db722c73c21fe1c179f28008d5ae3b2bbfabc2bf209b6b79a19e503d45da7c7.jpg)

![70c7788cd8dc288cc2652467a826341322c4bd3b8a6312ff28cc02fcb98d4828.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/70c7788cd8dc288cc2652467a826341322c4bd3b8a6312ff28cc02fcb98d4828.jpg)

![7bc082ab803776209f20c129e360a2c2f913ded346b9db79904dc43461745e37.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/7bc082ab803776209f20c129e360a2c2f913ded346b9db79904dc43461745e37.jpg)

![800846754f9770f9608e9474c3493e10c901e8e9b6fc436f2b2094b257c0be5d.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/800846754f9770f9608e9474c3493e10c901e8e9b6fc436f2b2094b257c0be5d.jpg)

![854610bea8960e36dcdfac735f0d7668e30b1d5466762f280b84f3586bba198e.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/854610bea8960e36dcdfac735f0d7668e30b1d5466762f280b84f3586bba198e.jpg)

![90f5c9746b72c6ccae7251d9a28be66edee3535da911c8babea39536281db246.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/90f5c9746b72c6ccae7251d9a28be66edee3535da911c8babea39536281db246.jpg)

![955cb465f38e0855d082e9894425c73e99487e12cf6cd6a9b5e325c23405621f.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/955cb465f38e0855d082e9894425c73e99487e12cf6cd6a9b5e325c23405621f.jpg)

![a1764a14e908978daced3380636824a6883eec93d3fa51609c6ba33613d631cc.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/a1764a14e908978daced3380636824a6883eec93d3fa51609c6ba33613d631cc.jpg)

![aa01a6b1d03c8f9ce077e8a2866af32124af55d6d8135c490f0d246cbf91c4b1.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/aa01a6b1d03c8f9ce077e8a2866af32124af55d6d8135c490f0d246cbf91c4b1.jpg)

![b53bd7a208cbfafbb9cb8882561563ae254b250d949010528bfef73da405e683.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/b53bd7a208cbfafbb9cb8882561563ae254b250d949010528bfef73da405e683.jpg)

![bbafc8d6cabfd6964f19be48883882b4f88a2be84f53c1a785d8ebc36cde2acf.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/bbafc8d6cabfd6964f19be48883882b4f88a2be84f53c1a785d8ebc36cde2acf.jpg)

![d3daedef683ff2e82ba05cc59417de04c8c80bc04d05a6889e8657bbd0439412.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/d3daedef683ff2e82ba05cc59417de04c8c80bc04d05a6889e8657bbd0439412.jpg)

![ec1807f5a046f84adbc8fe102e79387f2091dc9ba3c8ecddf40457c6f627af0a.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/ec1807f5a046f84adbc8fe102e79387f2091dc9ba3c8ecddf40457c6f627af0a.jpg)

![f094dfc042bbd2e1a94d05c23002cbfefda21248711289f2ae52df896b1f2339.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/f094dfc042bbd2e1a94d05c23002cbfefda21248711289f2ae52df896b1f2339.jpg)

![fa9ef5c80a482f041f57a89825ca57f1b089acd2826ef9269a1001af37531492.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/images/fa9ef5c80a482f041f57a89825ca57f1b089acd2826ef9269a1001af37531492.jpg)

### Tables

![066157e6591438dd1fadffe9acba82b31fc913e24183275893ff97a0e73a910c.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/066157e6591438dd1fadffe9acba82b31fc913e24183275893ff97a0e73a910c.jpg)

![211cd2b0c81c28f586cfe452ef905b5921d60d492781707d582c7aa5296b0989.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/211cd2b0c81c28f586cfe452ef905b5921d60d492781707d582c7aa5296b0989.jpg)

![362233ab37dedca93d32c9792b3fb033265d2e41f9bb598a2b9c048d13341aaa.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/362233ab37dedca93d32c9792b3fb033265d2e41f9bb598a2b9c048d13341aaa.jpg)

![3f078e753c5a360b751034d5dfb7b16407eab430b3504be9502956780700b636.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/3f078e753c5a360b751034d5dfb7b16407eab430b3504be9502956780700b636.jpg)

![41534a4185173ea63613943239d60556ab48813d3a8bb490cb1d950dbe3a50c8.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/41534a4185173ea63613943239d60556ab48813d3a8bb490cb1d950dbe3a50c8.jpg)

![439f9e71ad6adce1806991d2114ff4ed58863e8c989ad97619873ae77ef10373.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/439f9e71ad6adce1806991d2114ff4ed58863e8c989ad97619873ae77ef10373.jpg)

![4eb42a014b1a1265332d356af675605736d26acdfe157460ea43214d3fdec3a1.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/4eb42a014b1a1265332d356af675605736d26acdfe157460ea43214d3fdec3a1.jpg)

![5a414311c57aee4ece233d38f21bc0cf38c208ed86d8e9e5f61eb4d46ad748ea.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/5a414311c57aee4ece233d38f21bc0cf38c208ed86d8e9e5f61eb4d46ad748ea.jpg)

![6461f65e148571b211b9090befaa8e10b73c1c5169e042928567b4a73490001c.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/6461f65e148571b211b9090befaa8e10b73c1c5169e042928567b4a73490001c.jpg)

![6ad405b77ecd06ce412dc50bade9bc8bd1f3ea690c524e9a69bc641d7fe32a09.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/6ad405b77ecd06ce412dc50bade9bc8bd1f3ea690c524e9a69bc641d7fe32a09.jpg)

![8407eb4ddefdf6b6574d88e1f1645e78013d63df896d68b78c3b9c90e6ab51ac.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/8407eb4ddefdf6b6574d88e1f1645e78013d63df896d68b78c3b9c90e6ab51ac.jpg)

![8774edf78fbe506942736389058062827d53f0a359909108b4d1eb2a5b0e04e2.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/8774edf78fbe506942736389058062827d53f0a359909108b4d1eb2a5b0e04e2.jpg)

![976b8d8ec75cb988f86790c22d65c35b6933786b601974f5313b0a20abd25d02.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/976b8d8ec75cb988f86790c22d65c35b6933786b601974f5313b0a20abd25d02.jpg)

![ad5095f47039505ab6c2cb40bf2957ac0c55c35a33b7cd3d6e6f597c626b4bad.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/ad5095f47039505ab6c2cb40bf2957ac0c55c35a33b7cd3d6e6f597c626b4bad.jpg)

![b2c6cadaa995570c82f9d6e6771593632ff78f50925d4c2b9bacecab39229581.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/b2c6cadaa995570c82f9d6e6771593632ff78f50925d4c2b9bacecab39229581.jpg)

![b38c265932eb44d4341df42b5b3cd5837dd5da18f50ae1fb8a3649a93a8b1614.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/b38c265932eb44d4341df42b5b3cd5837dd5da18f50ae1fb8a3649a93a8b1614.jpg)

![c12230fa778bd3ad73863ff2db2d36d416b2ebebeed06cf59a1c6bbb1e93e479.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/c12230fa778bd3ad73863ff2db2d36d416b2ebebeed06cf59a1c6bbb1e93e479.jpg)

![c341dd08816d59ece9334947afd3264abfe21ef2fccff001a2ef14f5ba1937a9.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/c341dd08816d59ece9334947afd3264abfe21ef2fccff001a2ef14f5ba1937a9.jpg)

![c94998b1e2a49c73ce277fbc2bb1d66a82c5747c61ea33b3a0dd4b7f80fb17ff.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/c94998b1e2a49c73ce277fbc2bb1d66a82c5747c61ea33b3a0dd4b7f80fb17ff.jpg)

![c97fe4ec1b6468b264ab8d4f8ee6af6ee904d91bbc96d47a6fb7b26c3dbcf379.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/c97fe4ec1b6468b264ab8d4f8ee6af6ee904d91bbc96d47a6fb7b26c3dbcf379.jpg)

![ce4d2c498db806f065a067c419a7f86aaa7cbd8b3dffff0f10ff985cab58b7d8.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/ce4d2c498db806f065a067c419a7f86aaa7cbd8b3dffff0f10ff985cab58b7d8.jpg)

![d0de90b3c5916371d1d77d8ca3dc47cb21da7faa2320d1e4c8e55266d44bb9ea.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/d0de90b3c5916371d1d77d8ca3dc47cb21da7faa2320d1e4c8e55266d44bb9ea.jpg)

![d61cfe16acba25a12675df916c5c37c26a1cdea7c3d9afe22d0a15bf6d0a024f.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/d61cfe16acba25a12675df916c5c37c26a1cdea7c3d9afe22d0a15bf6d0a024f.jpg)

![da7ff246e19a0c4759fd73cc92d2b10635be943febd61b25ddc2375aea6448d7.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/da7ff246e19a0c4759fd73cc92d2b10635be943febd61b25ddc2375aea6448d7.jpg)

![e00a532cecedcc45841284b7aaec98d6a6e5ad6f3ecb43932595e5cc6f4c122e.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/e00a532cecedcc45841284b7aaec98d6a6e5ad6f3ecb43932595e5cc6f4c122e.jpg)

![f51fed8a29f8860bde17836727c9675eac5fff4bde3a94ff5ca8413ccc57218d.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/f51fed8a29f8860bde17836727c9675eac5fff4bde3a94ff5ca8413ccc57218d.jpg)

![fa9ab8fe3ce484783ec81a660922a31c2dbee63e801c7ae6590a3ab89ebc9f88.jpg](../icml_results/15_Learning%20to%20Theorize%20the%20World%20from%20Observation/tables/fa9ab8fe3ce484783ec81a660922a31c2dbee63e801c7ae6590a3ab89ebc9f88.jpg)

<a id="paper-0016"></a>

## Sycophancy Towards Researchers Drives Performative Misalignment


### Images

![2ed72e4af750dbdd4194b3d93a87a58497407cd32ddac18855d4abb2d2e0432e.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/2ed72e4af750dbdd4194b3d93a87a58497407cd32ddac18855d4abb2d2e0432e.jpg)

![3451ecca367fd88aaaecd32079b4392df9d81b1fd85778235ea272f3922d1d5d.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/3451ecca367fd88aaaecd32079b4392df9d81b1fd85778235ea272f3922d1d5d.jpg)

![51b9aad6d381cd0a8fdfd57658c3782b3a54345711eb2ece3dd847b6d10bf8cc.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/51b9aad6d381cd0a8fdfd57658c3782b3a54345711eb2ece3dd847b6d10bf8cc.jpg)

![7121f610c548fa5af916f3c7e9f4c03a056f4ee52c6123f1f7885f3e055501d9.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/7121f610c548fa5af916f3c7e9f4c03a056f4ee52c6123f1f7885f3e055501d9.jpg)

![7fd099c2ffa32c6681bb987da32eed43c063ed5316d85f763aaf4155bd291f49.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/7fd099c2ffa32c6681bb987da32eed43c063ed5316d85f763aaf4155bd291f49.jpg)

![9b4908d2e44f9693ca5757e23029ae5f977057c33cb42dd4e051a89a652113b0.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/9b4908d2e44f9693ca5757e23029ae5f977057c33cb42dd4e051a89a652113b0.jpg)

![aacef14fd9a18ba71a0ce6a9c547df0b1331cabd0f95d525f003eb5981ccb502.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/aacef14fd9a18ba71a0ce6a9c547df0b1331cabd0f95d525f003eb5981ccb502.jpg)

![de9c91a5ebb067fe8ffbfe8484ca276808dcdd85f5e3cdfdaf528a9ab5de64ab.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/de9c91a5ebb067fe8ffbfe8484ca276808dcdd85f5e3cdfdaf528a9ab5de64ab.jpg)

![f62ef280d71871bfce447563ac7b2dc26fda8868bdaaaddde8bc9499d34ec158.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/f62ef280d71871bfce447563ac7b2dc26fda8868bdaaaddde8bc9499d34ec158.jpg)

![f69f7546585cb4cd1ff22dcc2a03dddd0cc5f34263b891fe49d04489930979e0.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/images/f69f7546585cb4cd1ff22dcc2a03dddd0cc5f34263b891fe49d04489930979e0.jpg)

### Tables

![11c1fff83bbb0e2fa455b695976ce3415d3ae2f58b55acf0ca73137a18477068.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/tables/11c1fff83bbb0e2fa455b695976ce3415d3ae2f58b55acf0ca73137a18477068.jpg)

![2385c6411090984220995872216188d2bf611018f0b4f02a0729573637363037.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/tables/2385c6411090984220995872216188d2bf611018f0b4f02a0729573637363037.jpg)

![47ad91ac2f32e6da400fc78f21fae0c3b24336ef78cd7fd89d3c407ee817320e.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/tables/47ad91ac2f32e6da400fc78f21fae0c3b24336ef78cd7fd89d3c407ee817320e.jpg)

![558aef1a0448aed16ab543173022e9ccd24ee5f347fd3bc206301c0659760fc1.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/tables/558aef1a0448aed16ab543173022e9ccd24ee5f347fd3bc206301c0659760fc1.jpg)

![754a0133eb38e096494997fab6230599ffed422cfe0f0a876fa18c9070263591.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/tables/754a0133eb38e096494997fab6230599ffed422cfe0f0a876fa18c9070263591.jpg)

![c80f9ec5e617b983683519242652cd4293f6dbe1499b822e08299a9d0c924527.jpg](../icml_results/16_Sycophancy%20Towards%20Researchers%20Drives%20Performative%20Misalignment/tables/c80f9ec5e617b983683519242652cd4293f6dbe1499b822e08299a9d0c924527.jpg)

<a id="paper-0017"></a>

## Towards Understanding Adam Convergence on Highly Degenerate Polynomials


### Images

![501cf90683b8542de5a846cc8fb0e84a26ab4d48a3200894bec45007dc0a86cb.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/501cf90683b8542de5a846cc8fb0e84a26ab4d48a3200894bec45007dc0a86cb.jpg)

![570f74a86ff1e96da17e649ebf4ff98bcfc018032a453103f27b7f898366bafa.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/570f74a86ff1e96da17e649ebf4ff98bcfc018032a453103f27b7f898366bafa.jpg)

![6716d54e76946653caa9010faed78e460516c1995937f75f115ba47bf70cfc3d.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/6716d54e76946653caa9010faed78e460516c1995937f75f115ba47bf70cfc3d.jpg)

![71005d3b7e6f10c907f4ff52a6ada19f11893faed3de7e4d64f155bd29116c82.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/71005d3b7e6f10c907f4ff52a6ada19f11893faed3de7e4d64f155bd29116c82.jpg)

![764110eebd2e70ca0f75c7d71fc677d5dadc3d379f2440f063c9ba92e950122a.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/764110eebd2e70ca0f75c7d71fc677d5dadc3d379f2440f063c9ba92e950122a.jpg)

![80348c215540e40d0c1bc71955b26f76e630eab39af963af45c09c2573f66de8.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/80348c215540e40d0c1bc71955b26f76e630eab39af963af45c09c2573f66de8.jpg)

![80b770d571cbffeb4c9ffa276dc3648a5067a00e02a65d48cabd4456a2a5f181.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/80b770d571cbffeb4c9ffa276dc3648a5067a00e02a65d48cabd4456a2a5f181.jpg)

![9bc62b9507bd168193abe01b6903b786c0d1b09fdf75ea4f09320252d2dd01be.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/9bc62b9507bd168193abe01b6903b786c0d1b09fdf75ea4f09320252d2dd01be.jpg)

![9e6b105c7653cb40bc10637639a0dbc5a71e2892a890791f0cd326d3067e6f82.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/9e6b105c7653cb40bc10637639a0dbc5a71e2892a890791f0cd326d3067e6f82.jpg)

![a5a914d8171d6235c6508f525f9aa7db16816b58994b171a91dce129aa293aa0.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/a5a914d8171d6235c6508f525f9aa7db16816b58994b171a91dce129aa293aa0.jpg)

![be2c392999faa31242ee9321db6a8aff8cac54663d5fe107a2b10b85470259ff.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/be2c392999faa31242ee9321db6a8aff8cac54663d5fe107a2b10b85470259ff.jpg)

![c8acec1b26d003bd6a6c8a2cd21ba7fde74646a5447c14b0327927b86bc3ed96.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/c8acec1b26d003bd6a6c8a2cd21ba7fde74646a5447c14b0327927b86bc3ed96.jpg)

![cd7ba6361b9e921ca63ab9945c278e23bd318746e606aed38f80aa776ce1d820.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/cd7ba6361b9e921ca63ab9945c278e23bd318746e606aed38f80aa776ce1d820.jpg)

![e402c3bd10e419b8d084337a2b6d9ac8a77b26a2712020ebb4b06c25cfb49480.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/e402c3bd10e419b8d084337a2b6d9ac8a77b26a2712020ebb4b06c25cfb49480.jpg)

![e87b5fccb4c6cfa2060d5e039bc457f1f10b58225b9debc4ac588d2db410b4e8.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/e87b5fccb4c6cfa2060d5e039bc457f1f10b58225b9debc4ac588d2db410b4e8.jpg)

![fa3a0a75f8aeda3b3a6f4822d3888e457e23424f26816f0a495b91db98fc543c.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/images/fa3a0a75f8aeda3b3a6f4822d3888e457e23424f26816f0a495b91db98fc543c.jpg)

### Tables

![2b3d48e69b58a91b2397e8c8fdc45c6e6fd7c30ab2d08780c7e08528f27d2fff.jpg](../icml_results/17_Towards%20Understanding%20Adam%20Convergence%20on%20Highly%20Degenerate%20Polynomials/tables/2b3d48e69b58a91b2397e8c8fdc45c6e6fd7c30ab2d08780c7e08528f27d2fff.jpg)

<a id="paper-0018"></a>

## Learning in Structured Stackelberg Games


### Images

![05bea43a5cc7c59ee063e4679f34dc0a5a855d12985d87652f31b63dacebfd50.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/images/05bea43a5cc7c59ee063e4679f34dc0a5a855d12985d87652f31b63dacebfd50.jpg)

![2fed5773f7a20105d484ca93102ad29c7c2029f71d532f8e6991bafa800b8b90.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/images/2fed5773f7a20105d484ca93102ad29c7c2029f71d532f8e6991bafa800b8b90.jpg)

![3292c183a417e155504d2cfdea8809324acb6127aad50d668067d25f5a6692a8.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/images/3292c183a417e155504d2cfdea8809324acb6127aad50d668067d25f5a6692a8.jpg)

![91932b2029e620e3a8bd61019c7351ddbab974e5de23b26b44414eec738a5c68.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/images/91932b2029e620e3a8bd61019c7351ddbab974e5de23b26b44414eec738a5c68.jpg)

![c5fd4e0e0727500411579f495a04b7829df4696841111194480978d5ac42012e.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/images/c5fd4e0e0727500411579f495a04b7829df4696841111194480978d5ac42012e.jpg)

### Tables

![1e555e4cc9e017a475f1069b778e8ba9cc426f3f007d074976d5862c7b8c492f.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/1e555e4cc9e017a475f1069b778e8ba9cc426f3f007d074976d5862c7b8c492f.jpg)

![2449ed82686dd290e57186d59f798e7acc3e888aebf57978ae41d62fc6bc4eae.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/2449ed82686dd290e57186d59f798e7acc3e888aebf57978ae41d62fc6bc4eae.jpg)

![27a5ca1a88a319ebb884f8b91b4508adc65cc0f3fb144333f58e79b1c2b8171f.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/27a5ca1a88a319ebb884f8b91b4508adc65cc0f3fb144333f58e79b1c2b8171f.jpg)

![5b4465cf08b8786e72816f60b4529244ea541e359c9d309b33f1297dcff4e767.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/5b4465cf08b8786e72816f60b4529244ea541e359c9d309b33f1297dcff4e767.jpg)

![8a94e76640fac9340eaa796387051eff2749993bf061a58fcc005bc29d1ed744.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/8a94e76640fac9340eaa796387051eff2749993bf061a58fcc005bc29d1ed744.jpg)

![a89b75a760847c20c9179cdfb5dc8fa8c814d0c2b1f5e2041d3079d0ab021b23.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/a89b75a760847c20c9179cdfb5dc8fa8c814d0c2b1f5e2041d3079d0ab021b23.jpg)

![b201b35ee56527effde32236e9763be8a4bf39a16e50150eb9a72df06a5aced3.jpg](../icml_results/18_Learning%20in%20Structured%20Stackelberg%20Games/tables/b201b35ee56527effde32236e9763be8a4bf39a16e50150eb9a72df06a5aced3.jpg)

<a id="paper-0019"></a>

## Riemannian Metric Matching for Scalable Geometric Modeling of Distributions


### Images

![0865f282e13e4ad600746ab85db5e94770cf452ea3b44009c42fe058613f8049.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/0865f282e13e4ad600746ab85db5e94770cf452ea3b44009c42fe058613f8049.jpg)

![13bc2f26d5fbe9564eaee0b131e0dcd6385f39dbedbb4d29c7fc37b7cf55257a.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/13bc2f26d5fbe9564eaee0b131e0dcd6385f39dbedbb4d29c7fc37b7cf55257a.jpg)

![151abc2cbc2d405437228c81cec6a124c7a9226c77c3c9d9fc53700bfd6600a4.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/151abc2cbc2d405437228c81cec6a124c7a9226c77c3c9d9fc53700bfd6600a4.jpg)

![2d2e87d163148d09cf992f3e24fcd1950af0a2f9be349c285797e05808f372bd.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/2d2e87d163148d09cf992f3e24fcd1950af0a2f9be349c285797e05808f372bd.jpg)

![4538827538a43282dbcd31607076883f265f253c2d504ff1c448415b876247ff.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/4538827538a43282dbcd31607076883f265f253c2d504ff1c448415b876247ff.jpg)

![63f3bf2462c2395e6af64056f8336c16e3d637d6977119690d233ca866e4e5de.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/63f3bf2462c2395e6af64056f8336c16e3d637d6977119690d233ca866e4e5de.jpg)

![785ddf819cea0e4662d13228e3ed3456f0d6b0c5d92dd97a50496c12fb0f1511.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/785ddf819cea0e4662d13228e3ed3456f0d6b0c5d92dd97a50496c12fb0f1511.jpg)

![b7145b7115075c7e8d10e3d05cdd2009180b5466815f3d3082b7169a434f9fea.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/b7145b7115075c7e8d10e3d05cdd2009180b5466815f3d3082b7169a434f9fea.jpg)

![c6c0b8664d90acfe9a8522ac1fc439f80f7a782bc45f65fa5bc6ecef3999dcb7.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/c6c0b8664d90acfe9a8522ac1fc439f80f7a782bc45f65fa5bc6ecef3999dcb7.jpg)

![f7955eb6a3f6d37a366c3a51a074d5bff97715dbd8c92773f3061e8836df984e.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/images/f7955eb6a3f6d37a366c3a51a074d5bff97715dbd8c92773f3061e8836df984e.jpg)

### Tables

![2e06609b9197130def0dda18f3810bea97dff3234635f7b0fdfd2e74e9c5b13d.jpg](../icml_results/19_Riemannian%20Metric%20Matching%20for%20Scalable%20Geometric%20Modeling%20of%20Distributions/tables/2e06609b9197130def0dda18f3810bea97dff3234635f7b0fdfd2e74e9c5b13d.jpg)

<a id="paper-0020"></a>

## Universal Redundancies in Time Series Foundation Models


### Images

![0001ec82860e6a0f0a4c3076384564514a4ca9082cb3757b8afb14e78180b04b.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/0001ec82860e6a0f0a4c3076384564514a4ca9082cb3757b8afb14e78180b04b.jpg)

![15f86bf106a0056a24b2be601804136493b55743f8c6431d747c6e43143cf1d9.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/15f86bf106a0056a24b2be601804136493b55743f8c6431d747c6e43143cf1d9.jpg)

![2e1da2825250d37cca1e4fb1e39b1a69fb70634a932251054e918118ce9d4356.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/2e1da2825250d37cca1e4fb1e39b1a69fb70634a932251054e918118ce9d4356.jpg)

![3033f9eb386c3f5e2e1df1f38fc9962a7057c671b6fef4b7a15ad605edeb3975.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/3033f9eb386c3f5e2e1df1f38fc9962a7057c671b6fef4b7a15ad605edeb3975.jpg)

![3965bb5a69e35c28b9dd5c76c92a484e0d8b249709dfbc8c2f2037f9d85d1db6.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/3965bb5a69e35c28b9dd5c76c92a484e0d8b249709dfbc8c2f2037f9d85d1db6.jpg)

![3b44ba5ed1d6d4b7c5f92a4447e10b24a20d3338b32780cad863acfd0d1d5d24.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/3b44ba5ed1d6d4b7c5f92a4447e10b24a20d3338b32780cad863acfd0d1d5d24.jpg)

![5d13020740aeea20e6da9900b97af4f51a5c8818c89577ab94b468e0fe591686.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/5d13020740aeea20e6da9900b97af4f51a5c8818c89577ab94b468e0fe591686.jpg)

![651f5b1b620065fcdf8b759e53be9fc66816cf090528175d92758dc69ecd6e99.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/651f5b1b620065fcdf8b759e53be9fc66816cf090528175d92758dc69ecd6e99.jpg)

![6a567d51a91cc9ab263fa384a4f41e1f3f9826b32461bea9d6c545fe3a518cd4.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/6a567d51a91cc9ab263fa384a4f41e1f3f9826b32461bea9d6c545fe3a518cd4.jpg)

![714d3358868677e90f278dcb7b8efbe26936cbe2aa9c497c077a4fdcf783bf27.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/714d3358868677e90f278dcb7b8efbe26936cbe2aa9c497c077a4fdcf783bf27.jpg)

![82867a925d507e270a940f4db98734b7ff7d4f0c5702926d3ac0e6f3bd483296.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/82867a925d507e270a940f4db98734b7ff7d4f0c5702926d3ac0e6f3bd483296.jpg)

![8cb4ee938b1a6d5dfc24244356a507c2a7f186985b0749b6840f83da5ced25bb.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/8cb4ee938b1a6d5dfc24244356a507c2a7f186985b0749b6840f83da5ced25bb.jpg)

![97443d0a4a3d70201923ebd9d59b539030083d379e1bc08376f12acfef1dd16e.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/97443d0a4a3d70201923ebd9d59b539030083d379e1bc08376f12acfef1dd16e.jpg)

![9b2cb21b8487db08dff32ff89b23640722c585152b097a93e768e848883f1fc8.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/9b2cb21b8487db08dff32ff89b23640722c585152b097a93e768e848883f1fc8.jpg)

![a982d7f81837ecf174c44fd74d21eb8319c8c04d6c385ec8a121672280451b59.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/a982d7f81837ecf174c44fd74d21eb8319c8c04d6c385ec8a121672280451b59.jpg)

![aaa5b61f9d2212275bb2ebc1ef39a211ec22f9465dcc48001038c73bd0a6f441.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/aaa5b61f9d2212275bb2ebc1ef39a211ec22f9465dcc48001038c73bd0a6f441.jpg)

![ab9ed584a4c6b164e2b20b7bb542a0e3559344434dfe2a225a71ad43b128c8ea.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/ab9ed584a4c6b164e2b20b7bb542a0e3559344434dfe2a225a71ad43b128c8ea.jpg)

![bdddac0689f29118e8af3116df74db18cd76cb30450a6258753447c495b001ee.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/bdddac0689f29118e8af3116df74db18cd76cb30450a6258753447c495b001ee.jpg)

![c18a2d9bfd1ddb499867bddfaedcdb3c93a40de15a20f90e8c5360cdeddd0a59.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/c18a2d9bfd1ddb499867bddfaedcdb3c93a40de15a20f90e8c5360cdeddd0a59.jpg)

![c709789e58cd0304cfd43b6cffb1097abd7a7b828f33bb5dd7c9dbfa42af1d7e.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/c709789e58cd0304cfd43b6cffb1097abd7a7b828f33bb5dd7c9dbfa42af1d7e.jpg)

![d256cb40a80efd5b24bcb1f6028660184f21b229d685be879b4aa573552305f7.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/d256cb40a80efd5b24bcb1f6028660184f21b229d685be879b4aa573552305f7.jpg)

![d2a9b727a2f7e657ae4c4168112b878a49d5f12ad6a587c2068e97c6d9ce4bae.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/d2a9b727a2f7e657ae4c4168112b878a49d5f12ad6a587c2068e97c6d9ce4bae.jpg)

![db826c1da2ccb7c13d29f55cad4bd4df80ffae336fc44ee14af1f541b204b285.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/db826c1da2ccb7c13d29f55cad4bd4df80ffae336fc44ee14af1f541b204b285.jpg)

![e5ea49c28ae8fb018d5254ddb63a0d4aececc5af5e6a5b6be4260b10877d5aae.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/e5ea49c28ae8fb018d5254ddb63a0d4aececc5af5e6a5b6be4260b10877d5aae.jpg)

![e93c8f2ac44d6266c0d2392ea4eb91e728059ce1b2bed48c9f7bfecf5c58e404.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/e93c8f2ac44d6266c0d2392ea4eb91e728059ce1b2bed48c9f7bfecf5c58e404.jpg)

![eadc5e8b59142aacc98ff18e9f89266d5ec47cd5282a22a31dfdf300187ec8cf.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/eadc5e8b59142aacc98ff18e9f89266d5ec47cd5282a22a31dfdf300187ec8cf.jpg)

![f1b194ab80aede3651f444ceed52f40403f1e06c0667325a4265f659a0602eba.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/f1b194ab80aede3651f444ceed52f40403f1e06c0667325a4265f659a0602eba.jpg)

![f89933498ae4b68282f223a2ad212bf93d522c213a40ae4c8d1b1eb8fbf56d52.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/images/f89933498ae4b68282f223a2ad212bf93d522c213a40ae4c8d1b1eb8fbf56d52.jpg)

### Tables

![33e5e3cc0da614178dbbb19fb237cd79c348ed25d7f29680397e146f2fa26217.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/33e5e3cc0da614178dbbb19fb237cd79c348ed25d7f29680397e146f2fa26217.jpg)

![8bf57e978f39ac047a7e8a9b436be15f41bddf93afdae34608c569bef6a5f1d0.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/8bf57e978f39ac047a7e8a9b436be15f41bddf93afdae34608c569bef6a5f1d0.jpg)

![8cda0f490c19651b7d5db9e3d91ca14350caa4be97b609e7706e6a155b4e2e72.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/8cda0f490c19651b7d5db9e3d91ca14350caa4be97b609e7706e6a155b4e2e72.jpg)

![aaad1622f3c2a4b2de2f832f825d45d76f200a1b23aacffddaf12e405491ee10.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/aaad1622f3c2a4b2de2f832f825d45d76f200a1b23aacffddaf12e405491ee10.jpg)

![b3f9e2ca626d8c086e8f36c7a0fca2beae27e777d462b9caf2dcd563427a7bef.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/b3f9e2ca626d8c086e8f36c7a0fca2beae27e777d462b9caf2dcd563427a7bef.jpg)

![cc24c39c735525dff91f039ca64fef78e3805810a1c504ed01974092097534af.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/cc24c39c735525dff91f039ca64fef78e3805810a1c504ed01974092097534af.jpg)

![f3d00494f22be1598629e26ae20c4cab0d9b35cc3a203c34ef6c2f7f53242747.jpg](../icml_results/21_Universal%20Redundancies%20in%20Time%20Series%20Foundation%20Models/tables/f3d00494f22be1598629e26ae20c4cab0d9b35cc3a203c34ef6c2f7f53242747.jpg)

<a id="paper-0021"></a>

## Optimal Rates for Feasible Payoff Set Estimation in Games


### Images

![3cf7cbaac434004a27f327e25e5cdfde1f5a8d1a129c2de62f9fd129cbbb2551.jpg](../icml_results/22_Optimal%20Rates%20for%20Feasible%20Payoff%20Set%20Estimation%20in%20Games/images/3cf7cbaac434004a27f327e25e5cdfde1f5a8d1a129c2de62f9fd129cbbb2551.jpg)

### Tables

![19715b93da71cf6f7c8868766b36b8f402675ae9b13c3afb420954f290d07790.jpg](../icml_results/22_Optimal%20Rates%20for%20Feasible%20Payoff%20Set%20Estimation%20in%20Games/tables/19715b93da71cf6f7c8868766b36b8f402675ae9b13c3afb420954f290d07790.jpg)

<a id="paper-0022"></a>

## On the Origin of Neural Scaling Laws: from Random Graphs to Natural Language


### Images

![0107a0292960dd6fbfd0b691ffcef6c3670c40d4b073d6eb2b4bb65afec9a5b2.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/0107a0292960dd6fbfd0b691ffcef6c3670c40d4b073d6eb2b4bb65afec9a5b2.jpg)

![2dc4d1d94a8d64cfeb172133aa9a655d83cbd34adf3c332e227bf7e4131eb804.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/2dc4d1d94a8d64cfeb172133aa9a655d83cbd34adf3c332e227bf7e4131eb804.jpg)

![2efaedf61e502ad3957affbc43a4a7b6765e659740e5a68dc6726c3e5d5b9ad7.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/2efaedf61e502ad3957affbc43a4a7b6765e659740e5a68dc6726c3e5d5b9ad7.jpg)

![375bc24ae3be5b0750bb7a1f9f0db9f3653c3b95063b9cd2391d675f46238b5e.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/375bc24ae3be5b0750bb7a1f9f0db9f3653c3b95063b9cd2391d675f46238b5e.jpg)

![461e9e3b91dd8eac15f76d5e8e32749281afc313162f58ed59e0ad3385551ee6.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/461e9e3b91dd8eac15f76d5e8e32749281afc313162f58ed59e0ad3385551ee6.jpg)

![4d4304e0f8a6ddf3c390782f39b1861b2977d9a70b9e609b36ce9a5f7a253f45.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/4d4304e0f8a6ddf3c390782f39b1861b2977d9a70b9e609b36ce9a5f7a253f45.jpg)

![5cf986a36936eaaf898bec2383fe636f0354e21671d99b4a804ec2f2b2b9a039.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/5cf986a36936eaaf898bec2383fe636f0354e21671d99b4a804ec2f2b2b9a039.jpg)

![68a64c22e9377ea900a614dd29d9c33889b9fabbbd54c2fefc084fb95f130a6c.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/68a64c22e9377ea900a614dd29d9c33889b9fabbbd54c2fefc084fb95f130a6c.jpg)

![6a4146893d2981b5728d53fee76dd21356b695416c4f9b518351d5aad305373e.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/6a4146893d2981b5728d53fee76dd21356b695416c4f9b518351d5aad305373e.jpg)

![73cbb26729a74dbc7f35c10a2011f59dbdd550840e153cc43c5c20a4c89044b9.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/73cbb26729a74dbc7f35c10a2011f59dbdd550840e153cc43c5c20a4c89044b9.jpg)

![74c3fd39fc16a26092b1084de2353d20423bbe463d7e7afe4d248936fbce8cc0.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/74c3fd39fc16a26092b1084de2353d20423bbe463d7e7afe4d248936fbce8cc0.jpg)

![867b10e12ea3f63b9b238c80812e953406c565ecd74df7afeef0562a6ceb50f5.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/867b10e12ea3f63b9b238c80812e953406c565ecd74df7afeef0562a6ceb50f5.jpg)

![8d2dc48167a76f52ae32fb08cb1bb5ef845c8ccce52da5ec25cfcf393badd6e9.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/8d2dc48167a76f52ae32fb08cb1bb5ef845c8ccce52da5ec25cfcf393badd6e9.jpg)

![95b5cd7cb8c4a3322a60286ed52cb92e4eaee894c7e5e50be38b17e5882ea1e9.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/95b5cd7cb8c4a3322a60286ed52cb92e4eaee894c7e5e50be38b17e5882ea1e9.jpg)

![96e21ff090f52f4a88150a0caf20f9721a2da1783c6f88d1fc7ae61320c725ae.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/96e21ff090f52f4a88150a0caf20f9721a2da1783c6f88d1fc7ae61320c725ae.jpg)

![a95a6d61fcb90f03ce2634092b8860baae357cbd5a161170244ae83fe8921685.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/a95a6d61fcb90f03ce2634092b8860baae357cbd5a161170244ae83fe8921685.jpg)

![b466ab01b4f8572ef90626f0f1e456b20b766c2aada370f4d67165340ac746be.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/b466ab01b4f8572ef90626f0f1e456b20b766c2aada370f4d67165340ac746be.jpg)

![bb72ed6f2ca9536725922f9bc964c68bb32f1b146cdd532da5ca40e9fcef8808.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/bb72ed6f2ca9536725922f9bc964c68bb32f1b146cdd532da5ca40e9fcef8808.jpg)

![c0110a5036c1a2e5d838127db2ac1037067e12b7c50a495e30cc3f777d95b1d1.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/c0110a5036c1a2e5d838127db2ac1037067e12b7c50a495e30cc3f777d95b1d1.jpg)

![daaa167297620e9156d20ee22c37f6c34ebb950d31e196872f45169b83e13b59.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/daaa167297620e9156d20ee22c37f6c34ebb950d31e196872f45169b83e13b59.jpg)

![ddba9770672b8069ab9ba8051ec445c006457704c15ab61f66575245e08176ea.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/ddba9770672b8069ab9ba8051ec445c006457704c15ab61f66575245e08176ea.jpg)

![de8a6fd771cd2482c90f544da8501f15b01c27d480a2dd4b965f2574037b49d5.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/de8a6fd771cd2482c90f544da8501f15b01c27d480a2dd4b965f2574037b49d5.jpg)

![e65848761766b66fdf353b1540fbc05bc4a5d9e4f9473cdf76362f54b2a41040.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/e65848761766b66fdf353b1540fbc05bc4a5d9e4f9473cdf76362f54b2a41040.jpg)

![f05b2eb7554c4855b43c6d899e4d4d0b313e0da80a7aa7919571584807845ca1.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/images/f05b2eb7554c4855b43c6d899e4d4d0b313e0da80a7aa7919571584807845ca1.jpg)

### Tables

![97b05c7daa162a32886609d41f241eca89ec64528de23c48064de9a1f48aac07.jpg](../icml_results/23_On%20the%20Origin%20of%20Neural%20Scaling%20Laws_%20from%20Random%20Graphs%20to%20Natural%20Language/tables/97b05c7daa162a32886609d41f241eca89ec64528de23c48064de9a1f48aac07.jpg)

<a id="paper-0023"></a>

## $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment


### Images

![053f144cae24353e09aae664754a4daeb26ec12209834c984f9b33217cc82946.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/053f144cae24353e09aae664754a4daeb26ec12209834c984f9b33217cc82946.jpg)

![10da5fb0e4179060401c84a52a6e2a92a03ea325ab5d104fccb8a7dc7a3cd43d.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/10da5fb0e4179060401c84a52a6e2a92a03ea325ab5d104fccb8a7dc7a3cd43d.jpg)

![1bd0e322ed56779b3c252698d1985a0d479a515bdc4f2ce17ec314ac53e1c676.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/1bd0e322ed56779b3c252698d1985a0d479a515bdc4f2ce17ec314ac53e1c676.jpg)

![23d71cdac7ae563a4554f079895b354b9f02c516f57011bacbcd0de07a4891ed.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/23d71cdac7ae563a4554f079895b354b9f02c516f57011bacbcd0de07a4891ed.jpg)

![41460d189f88c3d519ce5345fade6c2159921a28737810a6e424bee89b31b10b.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/41460d189f88c3d519ce5345fade6c2159921a28737810a6e424bee89b31b10b.jpg)

![4d4bc2a959cc4ba16a4e0059b37dc2ff60cb08eb7d69176742af733f1cbd36f4.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/4d4bc2a959cc4ba16a4e0059b37dc2ff60cb08eb7d69176742af733f1cbd36f4.jpg)

![4ddc5ac8745e8fe422dd6800234bbc41e9444fe19d2748eb797b273ad08c990f.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/4ddc5ac8745e8fe422dd6800234bbc41e9444fe19d2748eb797b273ad08c990f.jpg)

![5209aad0f3af2785b9dedc15038dfe4178892d2ed5bf1fbb16de666d9cf21cde.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/5209aad0f3af2785b9dedc15038dfe4178892d2ed5bf1fbb16de666d9cf21cde.jpg)

![5764fc54f1cffd64ac7519899c23db6349c22633baa30ce4845b74d9229fd093.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/5764fc54f1cffd64ac7519899c23db6349c22633baa30ce4845b74d9229fd093.jpg)

![63fb810b21f8e280e2c4c078e9a13b5f08d5408d2e983b98d5d6a7aed9afbe16.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/63fb810b21f8e280e2c4c078e9a13b5f08d5408d2e983b98d5d6a7aed9afbe16.jpg)

![75964f5140df58f0c45f994d9be2664183438fa33885de41f92331c52a4a3700.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/75964f5140df58f0c45f994d9be2664183438fa33885de41f92331c52a4a3700.jpg)

![971f6926479a2f3b21c3a3e5bc2fbc0bc4db1d72fe3931a8d131b18627308041.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/971f6926479a2f3b21c3a3e5bc2fbc0bc4db1d72fe3931a8d131b18627308041.jpg)

![a117a6bb02e7206969a917bf1ce6db8b0cf0344581acc9c045c7b4211005249c.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/a117a6bb02e7206969a917bf1ce6db8b0cf0344581acc9c045c7b4211005249c.jpg)

![acf7b5aac168f5164aa14cdb5d727c06cfef15b988b2c18ca915d0da09738761.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/acf7b5aac168f5164aa14cdb5d727c06cfef15b988b2c18ca915d0da09738761.jpg)

![c3824419b7338f91e48064ef1309f329667fa104676a5735509a889e11e079d7.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/images/c3824419b7338f91e48064ef1309f329667fa104676a5735509a889e11e079d7.jpg)

### Tables

![182e08bd623771207f23b857a7254c70690b2093b4655f3f8b5cc731303470b2.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/tables/182e08bd623771207f23b857a7254c70690b2093b4655f3f8b5cc731303470b2.jpg)

![572c22d1f0de45074e506d4875e197632845ec3b5b0878cd8953da2c70b50f90.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/tables/572c22d1f0de45074e506d4875e197632845ec3b5b0878cd8953da2c70b50f90.jpg)

![7f73434dadaeebc65280298e54dd8f5b7916a403e454887367b32eb4a9a54dbc.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/tables/7f73434dadaeebc65280298e54dd8f5b7916a403e454887367b32eb4a9a54dbc.jpg)

![8821669275db8bad4f15ebd4ae03ac9569cec1d0fc60d39208e683c81e6b09a3.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/tables/8821669275db8bad4f15ebd4ae03ac9569cec1d0fc60d39208e683c81e6b09a3.jpg)

![ff84413c7c8c025118a760621d5eee63dfc48366bc707630ccd467ab2893ddf8.jpg](../icml_results/24_%24%CF%84%5E2%24-Bench_%20Evaluating%20Conversational%20Agents%20in%20a%20Dual-Control%20Environment/tables/ff84413c7c8c025118a760621d5eee63dfc48366bc707630ccd467ab2893ddf8.jpg)

<a id="paper-0024"></a>

## Uncovering the Latent Potential of Deep Intermediate Representations


### Images

![1c00e0c78d61d142f89dd74895ad073f80459e19e3bc080f999a977a2edc10fb.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/1c00e0c78d61d142f89dd74895ad073f80459e19e3bc080f999a977a2edc10fb.jpg)

![37fc55066ce1de2b195a9a0c5432c927d7a25e5c0474f612fd0b70c684d36457.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/37fc55066ce1de2b195a9a0c5432c927d7a25e5c0474f612fd0b70c684d36457.jpg)

![62091a24cad19126300891bfd47be3e1d48e18696bcfa6cff1c3612b97f77e24.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/62091a24cad19126300891bfd47be3e1d48e18696bcfa6cff1c3612b97f77e24.jpg)

![646212625dbbc2e6cacef1fdddaefe0bfb5d4b171e40bcbf80fc6f28cef0a56f.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/646212625dbbc2e6cacef1fdddaefe0bfb5d4b171e40bcbf80fc6f28cef0a56f.jpg)

![7f1d3ab2f9522d09e638f2cf38f7eb04c7ffec45d3472359438fd585733a0010.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/7f1d3ab2f9522d09e638f2cf38f7eb04c7ffec45d3472359438fd585733a0010.jpg)

![92ec40b439516e74fb4e971acb782c1b745e1a16016ed3f1ce68b61735b63189.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/92ec40b439516e74fb4e971acb782c1b745e1a16016ed3f1ce68b61735b63189.jpg)

![bea500b4bd4433bab8f59bd09672d0ebbf449bd68f4d2ca68bee8717e4a322d4.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/bea500b4bd4433bab8f59bd09672d0ebbf449bd68f4d2ca68bee8717e4a322d4.jpg)

![c0a9240db98250393e8919f6c508559637ea2f20784ba562c3bdebff5657a0c6.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/c0a9240db98250393e8919f6c508559637ea2f20784ba562c3bdebff5657a0c6.jpg)

![c9f9fcc74761a6030b4cbf7f212ee99f1e1c4c048e6e815ca4042d1552b8b25b.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/c9f9fcc74761a6030b4cbf7f212ee99f1e1c4c048e6e815ca4042d1552b8b25b.jpg)

![f363c2b3909485b7c62e134d17fd3dc4e7322f59386cb9c72d8d8ce5dfdd0f67.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/images/f363c2b3909485b7c62e134d17fd3dc4e7322f59386cb9c72d8d8ce5dfdd0f67.jpg)

### Tables

![0c2e15d80be854c759dbe40ef1cd7365d34fc1557c679e80ef0b79a4ba2ba2f0.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/0c2e15d80be854c759dbe40ef1cd7365d34fc1557c679e80ef0b79a4ba2ba2f0.jpg)

![0c6a6daf1324cbff5288801b0676a7855ed125697c574380266dbfd491d7ed69.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/0c6a6daf1324cbff5288801b0676a7855ed125697c574380266dbfd491d7ed69.jpg)

![129487635a9fee94e5637f0b0661bf9312e0b81e51960fcb00dc5a705a1a0ee9.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/129487635a9fee94e5637f0b0661bf9312e0b81e51960fcb00dc5a705a1a0ee9.jpg)

![13f1f4bf331ec7333a61d060e34fd44d539dc947b0d6e79c0402b706d4eeef70.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/13f1f4bf331ec7333a61d060e34fd44d539dc947b0d6e79c0402b706d4eeef70.jpg)

![18cdd3358142f3ad3b68cb282bf9839fe03e55631aada7a8532a4e5fd35ab99f.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/18cdd3358142f3ad3b68cb282bf9839fe03e55631aada7a8532a4e5fd35ab99f.jpg)

![2290ebf0f7e81e8ae7da38ffa092d26614d2cb5bc3cd2f7571eef5624ead5904.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/2290ebf0f7e81e8ae7da38ffa092d26614d2cb5bc3cd2f7571eef5624ead5904.jpg)

![283ff5be255d57dd14b99aac8d48a1d2602a7960834e1e043465c87af54aead6.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/283ff5be255d57dd14b99aac8d48a1d2602a7960834e1e043465c87af54aead6.jpg)

![305c0f1af0f223054a8f3f1318412b1bda17f2b86765645810fc0f949e004c83.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/305c0f1af0f223054a8f3f1318412b1bda17f2b86765645810fc0f949e004c83.jpg)

![30ae308b3f5be8297f1f2a7b1a8c5eeee6107768dfb0f92746177516c2f5e1b1.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/30ae308b3f5be8297f1f2a7b1a8c5eeee6107768dfb0f92746177516c2f5e1b1.jpg)

![31dcdce8a4ced90aa2da446b01b36131ccd5afd1bc818a8d7188817bba24e277.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/31dcdce8a4ced90aa2da446b01b36131ccd5afd1bc818a8d7188817bba24e277.jpg)

![37db53f12cdd8fd5120f0e6f27b08b05b2d03708d456682ea0fb22945cd1bf86.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/37db53f12cdd8fd5120f0e6f27b08b05b2d03708d456682ea0fb22945cd1bf86.jpg)

![6d41c211ce5ccb95c51d4ffbe03660518cb0eae3232f22d2bdf7a68f6aa0767b.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/6d41c211ce5ccb95c51d4ffbe03660518cb0eae3232f22d2bdf7a68f6aa0767b.jpg)

![6ed68b7229ae7f9c01109d69703fc17525f20d83c0b71ae02da800e198dc4fa5.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/6ed68b7229ae7f9c01109d69703fc17525f20d83c0b71ae02da800e198dc4fa5.jpg)

![890c4b8821f5e2b707438c0b6e2299c4ccf7aac5f2ebd089021a30e38b77ef3f.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/890c4b8821f5e2b707438c0b6e2299c4ccf7aac5f2ebd089021a30e38b77ef3f.jpg)

![963431685055b0c62929a5e2e7ac076f2e6ee2daaaf220397c8431d8050820f2.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/963431685055b0c62929a5e2e7ac076f2e6ee2daaaf220397c8431d8050820f2.jpg)

![9a6ebb403c2a393d757b164dc7ab7b7fb48ffe0e2f63d1ea0bb75a7b19e5aad8.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/9a6ebb403c2a393d757b164dc7ab7b7fb48ffe0e2f63d1ea0bb75a7b19e5aad8.jpg)

![b983c78e3aae815082948a9c05ee9739fa88556d8e1ef8f105c615806b2ea6ab.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/b983c78e3aae815082948a9c05ee9739fa88556d8e1ef8f105c615806b2ea6ab.jpg)

![cddc67b8cfbb4369099d761042be3ab43d0beaa95f828e3262f77f16cf0e7ee3.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/cddc67b8cfbb4369099d761042be3ab43d0beaa95f828e3262f77f16cf0e7ee3.jpg)

![d91763c914eb3ba761c55c43ce51de080d89ad7e2ceb059b62ca0352e77c7a6d.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/d91763c914eb3ba761c55c43ce51de080d89ad7e2ceb059b62ca0352e77c7a6d.jpg)

![eddfaf8d26071c450627c5aa18d9a1c4163e0afc45d57999c5fe88445c2d1d6a.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/eddfaf8d26071c450627c5aa18d9a1c4163e0afc45d57999c5fe88445c2d1d6a.jpg)

![f6d92ee88d9e181b17f384072ddfed1bc8b4d4f590205b14714f205da689724b.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/f6d92ee88d9e181b17f384072ddfed1bc8b4d4f590205b14714f205da689724b.jpg)

![f7ab24f1231445dd2eb3424072caf9c350d8dc8b74f3b655befcfb80c6515254.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/f7ab24f1231445dd2eb3424072caf9c350d8dc8b74f3b655befcfb80c6515254.jpg)

![fbfdcb9fcf2a866d200bf65b7fd9645b87a215c669d56e985b2fd51df42a4633.jpg](../icml_results/25_Uncovering%20the%20Latent%20Potential%20of%20Deep%20Intermediate%20Representations/tables/fbfdcb9fcf2a866d200bf65b7fd9645b87a215c669d56e985b2fd51df42a4633.jpg)

<a id="paper-0025"></a>

## INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic


### Images

![24044f595a5062558938981efa219a9dda4562adc6473604a4bc7020658546cf.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/images/24044f595a5062558938981efa219a9dda4562adc6473604a4bc7020658546cf.jpg)

![6a6f9a8f6e00ecde36b86bd2795281c698316873442ed38661d80b283776e3d9.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/images/6a6f9a8f6e00ecde36b86bd2795281c698316873442ed38661d80b283776e3d9.jpg)

![9b28cf113dbd0590348f9c0d9958fbc9ee43847dec2ecd1b7b53f025c5f86404.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/images/9b28cf113dbd0590348f9c0d9958fbc9ee43847dec2ecd1b7b53f025c5f86404.jpg)

![e9ac37217879e3ec30ba32c159d11a489aec3bc3a9f18cc3f627c911a5d21a38.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/images/e9ac37217879e3ec30ba32c159d11a489aec3bc3a9f18cc3f627c911a5d21a38.jpg)

![fc40701c9c8d72f04437756e127b401af470a47107194b0a2cfda5382e844681.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/images/fc40701c9c8d72f04437756e127b401af470a47107194b0a2cfda5382e844681.jpg)

### Tables

![146f235e57f703de7dd295376248ef14a1ff3a6d1eed91ae1af33412293afdce.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/146f235e57f703de7dd295376248ef14a1ff3a6d1eed91ae1af33412293afdce.jpg)

![180f323857909b5c7e45c3368bbd928c43f7e11702c137422931d001f4076e30.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/180f323857909b5c7e45c3368bbd928c43f7e11702c137422931d001f4076e30.jpg)

![19955f3d14f731083ac00476430bf5f816b2ac20cc4cd584e2151208bed2d9eb.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/19955f3d14f731083ac00476430bf5f816b2ac20cc4cd584e2151208bed2d9eb.jpg)

![1b58de36a2cdacdf1faf7ccd2de1b2ec69383f5181c03c0602f4f7e2d7a2fef4.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/1b58de36a2cdacdf1faf7ccd2de1b2ec69383f5181c03c0602f4f7e2d7a2fef4.jpg)

![1b5b250e89144df7635cf925d63ff131df2da5e8722ab6fa7e31f57c58ecc4d4.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/1b5b250e89144df7635cf925d63ff131df2da5e8722ab6fa7e31f57c58ecc4d4.jpg)

![1e94f296fc69b75b1c0e83d2ab6d6197e8f4f9b1c90c8e3c029e00a7aca8854a.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/1e94f296fc69b75b1c0e83d2ab6d6197e8f4f9b1c90c8e3c029e00a7aca8854a.jpg)

![210d3465d28ab012dadaaf34293057439e747194104fb03575544885f3abfcc1.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/210d3465d28ab012dadaaf34293057439e747194104fb03575544885f3abfcc1.jpg)

![359cc905bf9f2c9289b69391675e7d985f3eace96f4508899873fbcf6ae20b0e.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/359cc905bf9f2c9289b69391675e7d985f3eace96f4508899873fbcf6ae20b0e.jpg)

![369aeac8592ffbbfead8b15330db01abd32f7912d37d8f5dd4344c2c46cc1a56.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/369aeac8592ffbbfead8b15330db01abd32f7912d37d8f5dd4344c2c46cc1a56.jpg)

![4329b20b91e71d2ec1e115a2948601218c5cc1f9152e12b21ac7952e899f29af.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/4329b20b91e71d2ec1e115a2948601218c5cc1f9152e12b21ac7952e899f29af.jpg)

![496f0b02907e6e558a39f03291c24a9872b6e152a39e318763f95ed5f290a68c.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/496f0b02907e6e558a39f03291c24a9872b6e152a39e318763f95ed5f290a68c.jpg)

![49fca72991aa85247d063b78faccfe7b8afec6353f8405441a6d4ac23eb00ecb.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/49fca72991aa85247d063b78faccfe7b8afec6353f8405441a6d4ac23eb00ecb.jpg)

![4bb8a62369f3108c084d204e9856a946eb5b8ec6b8b2b99699055091b2a0fd63.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/4bb8a62369f3108c084d204e9856a946eb5b8ec6b8b2b99699055091b2a0fd63.jpg)

![51d26e337fe00d51d091e02604a6bbc32208a40d6e9f08405760198528598c6e.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/51d26e337fe00d51d091e02604a6bbc32208a40d6e9f08405760198528598c6e.jpg)

![558a2e38ac919f5d37031f4bc360925772d173ece9a14579b3e74474d44402aa.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/558a2e38ac919f5d37031f4bc360925772d173ece9a14579b3e74474d44402aa.jpg)

![5b19ca4edcb7993572c5e4964c326c88eece9f4d61add6331a23cdc4855e4ddd.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/5b19ca4edcb7993572c5e4964c326c88eece9f4d61add6331a23cdc4855e4ddd.jpg)

![60d869cce72e60b804cfc79c30387468764720e7c0cc38bf6964dab1257f897e.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/60d869cce72e60b804cfc79c30387468764720e7c0cc38bf6964dab1257f897e.jpg)

![65a44df32a6d0531bdc1f297d2dead662ccc285d78867603613500044bf7c651.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/65a44df32a6d0531bdc1f297d2dead662ccc285d78867603613500044bf7c651.jpg)

![66d39a0fd5e9eb709e0085f9a72a52ec049c4525a2b6d584b45959d19d75940d.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/66d39a0fd5e9eb709e0085f9a72a52ec049c4525a2b6d584b45959d19d75940d.jpg)

![6cac8713202fc4fc1e4df6976b58f26fd2c1d4833a788861268438760b65ad38.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/6cac8713202fc4fc1e4df6976b58f26fd2c1d4833a788861268438760b65ad38.jpg)

![7da52820ea7f291765ab05a738d6f3822e0daf6f9b10a23989c6f018dd7ec3b0.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/7da52820ea7f291765ab05a738d6f3822e0daf6f9b10a23989c6f018dd7ec3b0.jpg)

![7e4dd809c4cdf6934764162baa3075c8070673c19bad7d00231c7ce9d22252ce.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/7e4dd809c4cdf6934764162baa3075c8070673c19bad7d00231c7ce9d22252ce.jpg)

![7f6e099fb9952261bfa0606dcfdd7fba4525b5233065f4d2c3cefd1dd495fc8e.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/7f6e099fb9952261bfa0606dcfdd7fba4525b5233065f4d2c3cefd1dd495fc8e.jpg)

![8d56cf5653bef582be0b1171609a191ef751aa0273f6893ca9cf80a3fff57b0a.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/8d56cf5653bef582be0b1171609a191ef751aa0273f6893ca9cf80a3fff57b0a.jpg)

![947108138fb27d5c2fcd92c030e5c24fe67d2fa247e526e5fbb493276bc80311.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/947108138fb27d5c2fcd92c030e5c24fe67d2fa247e526e5fbb493276bc80311.jpg)

![95b8ec5c75f5d068499a1b0a5c668f3aa627b445f332cefe996309556e0f8c93.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/95b8ec5c75f5d068499a1b0a5c668f3aa627b445f332cefe996309556e0f8c93.jpg)

![a96584506159594cab6c764b6b0fc7076bfe54fafe20c494f46109bd1eed9d3e.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/a96584506159594cab6c764b6b0fc7076bfe54fafe20c494f46109bd1eed9d3e.jpg)

![b7188ff6cc06f1dcb64d4d3f4fa2d976602f960dc6289acb5ae70e0b70f01ced.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/b7188ff6cc06f1dcb64d4d3f4fa2d976602f960dc6289acb5ae70e0b70f01ced.jpg)

![be3f75e99425ba948bcaf57493e4e17eb21669d23d988e306024071fdb2ddb31.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/be3f75e99425ba948bcaf57493e4e17eb21669d23d988e306024071fdb2ddb31.jpg)

![c452a6371dca74eb6ddd40f1c612531bb9a0442db62d7c05010d62b5a1a9394e.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/c452a6371dca74eb6ddd40f1c612531bb9a0442db62d7c05010d62b5a1a9394e.jpg)

![c4dce1d1ce9a6f30f4c7e5cb9b01f17e142529c43c10b1aa6805a4d5c975e5e7.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/c4dce1d1ce9a6f30f4c7e5cb9b01f17e142529c43c10b1aa6805a4d5c975e5e7.jpg)

![ce5a2942704eba2f3b3f56f580efe1e83d6fe901200cfdf5ca16451fd0d64940.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/ce5a2942704eba2f3b3f56f580efe1e83d6fe901200cfdf5ca16451fd0d64940.jpg)

![d52d942117316c35604d4d4c118c91dd3820e7c981000b8c2d94c097c6cf7ab3.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/d52d942117316c35604d4d4c118c91dd3820e7c981000b8c2d94c097c6cf7ab3.jpg)

![d8e86ca95a6a0c6b4b3778e774c42cdd443737108f277213bbed24bb79e57fd6.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/d8e86ca95a6a0c6b4b3778e774c42cdd443737108f277213bbed24bb79e57fd6.jpg)

![e1a78b61544f2408a663aaa13c8cb41787b8c7b27b002a456ea697c477a67db5.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/e1a78b61544f2408a663aaa13c8cb41787b8c7b27b002a456ea697c477a67db5.jpg)

![e25532c2e7513b45748703ae5baa3cf571d946a8a9c51a2c20cc9b1cadb34750.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/e25532c2e7513b45748703ae5baa3cf571d946a8a9c51a2c20cc9b1cadb34750.jpg)

![e3778114e06767ce8e5993d7cb841c4064506ab94e21eb13f673f4489d042468.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/e3778114e06767ce8e5993d7cb841c4064506ab94e21eb13f673f4489d042468.jpg)

![e4bc0e8a1f73a546673607cb526346fbc7d83afb08db4b97a27dbb2a7ab6e0c0.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/e4bc0e8a1f73a546673607cb526346fbc7d83afb08db4b97a27dbb2a7ab6e0c0.jpg)

![e6b695a6accdb429d8e2e20c22239e1adf39021dc59a3c16beb17ccc940dee72.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/e6b695a6accdb429d8e2e20c22239e1adf39021dc59a3c16beb17ccc940dee72.jpg)

![eb550f9fbdc6ac495dec82c5a5009522136122491700209418858cb73009c67c.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/eb550f9fbdc6ac495dec82c5a5009522136122491700209418858cb73009c67c.jpg)

![f314d936a27f5b2e132136b66243ebfd1233532e2d1466417502e871bd494052.jpg](../icml_results/26_INDUCTION_%20Finite-Structure%20Concept%20Synthesis%20in%20First-Order%20Logic/tables/f314d936a27f5b2e132136b66243ebfd1233532e2d1466417502e871bd494052.jpg)

<a id="paper-0026"></a>

## The Relative Instability of Model Comparison with Cross-validation


### Images

![0853cbcb86df2209ce05457348f8d3cb6eba19b7d7bb09cd9c242d54300e63df.jpg](../icml_results/27_The%20Relative%20Instability%20of%20Model%20Comparison%20with%20Cross-validation/images/0853cbcb86df2209ce05457348f8d3cb6eba19b7d7bb09cd9c242d54300e63df.jpg)

![2eed275f5c0ce8e7b89aa4bd8a9531fd11faa3fcb50e452ca848926f15585182.jpg](../icml_results/27_The%20Relative%20Instability%20of%20Model%20Comparison%20with%20Cross-validation/images/2eed275f5c0ce8e7b89aa4bd8a9531fd11faa3fcb50e452ca848926f15585182.jpg)

![458e6746c566e3ae89f7c35ddcd59cc64b3b1bb7f13d3781482ed5eac5e6fe92.jpg](../icml_results/27_The%20Relative%20Instability%20of%20Model%20Comparison%20with%20Cross-validation/images/458e6746c566e3ae89f7c35ddcd59cc64b3b1bb7f13d3781482ed5eac5e6fe92.jpg)

![8f1209cfdf227107657cf58bb561de6c5ad2bd5b712d2d566e6f97afaaa81f80.jpg](../icml_results/27_The%20Relative%20Instability%20of%20Model%20Comparison%20with%20Cross-validation/images/8f1209cfdf227107657cf58bb561de6c5ad2bd5b712d2d566e6f97afaaa81f80.jpg)

![90e4190ececa0d37ed89ef4b9e40e4b9d70f2ec124698beebffdb72e558d8aa2.jpg](../icml_results/27_The%20Relative%20Instability%20of%20Model%20Comparison%20with%20Cross-validation/images/90e4190ececa0d37ed89ef4b9e40e4b9d70f2ec124698beebffdb72e558d8aa2.jpg)

![b0bf3d845c59810ed8d6fa639461de4c3d81f6c7040b67546d70fbe6d9101bcd.jpg](../icml_results/27_The%20Relative%20Instability%20of%20Model%20Comparison%20with%20Cross-validation/images/b0bf3d845c59810ed8d6fa639461de4c3d81f6c7040b67546d70fbe6d9101bcd.jpg)

<a id="paper-0027"></a>

## CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning


### Images

![05f25fe752a03c3b0420329c4c228853d24669be1f983d35de0830345febf280.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/05f25fe752a03c3b0420329c4c228853d24669be1f983d35de0830345febf280.jpg)

![2d297b284589c5fbd4d900c31336b57b7039e534e4d3a45e78fd06f88cdef8e7.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/2d297b284589c5fbd4d900c31336b57b7039e534e4d3a45e78fd06f88cdef8e7.jpg)

![3ef4d59149a0c1ef1951883df88bb09db368bfcef38143985e33b78c3676e688.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/3ef4d59149a0c1ef1951883df88bb09db368bfcef38143985e33b78c3676e688.jpg)

![602869137290bb9b0d81e9a19c6c9fe3afa9d43a5ca7b4c3033814a1d92374da.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/602869137290bb9b0d81e9a19c6c9fe3afa9d43a5ca7b4c3033814a1d92374da.jpg)

![6a8016abd947206c1d19b1f195f92dacb23f927b43cabe4a0fa2988576871a28.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/6a8016abd947206c1d19b1f195f92dacb23f927b43cabe4a0fa2988576871a28.jpg)

![7ae6ba5d176ad00424148dcf4e0cd3aa2eb50240c85b9a4779fd9982ce2e0dfe.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/7ae6ba5d176ad00424148dcf4e0cd3aa2eb50240c85b9a4779fd9982ce2e0dfe.jpg)

![7f7b2ba17ac759a8fa61f0b8bd0f3d8bee6eb9cd2e810e9554449131c8bdeac7.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/7f7b2ba17ac759a8fa61f0b8bd0f3d8bee6eb9cd2e810e9554449131c8bdeac7.jpg)

![b7ecd0bbe5f5abbd1015d8290827685e45b2ce312a32f874961846e117d0fb6e.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/b7ecd0bbe5f5abbd1015d8290827685e45b2ce312a32f874961846e117d0fb6e.jpg)

![c64a9aa932b703fba92795d8653f489e74c6485c7925f70ff22a1473f3d291d1.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/c64a9aa932b703fba92795d8653f489e74c6485c7925f70ff22a1473f3d291d1.jpg)

![e74e8f1a60978d9d2b7c9bbe37170bdb0bca3d11c0739f8fd47914289c0ea27b.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/e74e8f1a60978d9d2b7c9bbe37170bdb0bca3d11c0739f8fd47914289c0ea27b.jpg)

![ec6e4cb3a62cfbc1313526b9265ae5c430d30d5fe42193eda0d39154e010ee9e.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/images/ec6e4cb3a62cfbc1313526b9265ae5c430d30d5fe42193eda0d39154e010ee9e.jpg)

### Tables

![198f5ae50520879b87234456741e8b0117e0c8650721b22c8e2dabb4149c68db.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/tables/198f5ae50520879b87234456741e8b0117e0c8650721b22c8e2dabb4149c68db.jpg)

![446afc2c1cd9c950f89b25962908f8e79d058982d073c1a6845e8fa2b7ca1b0b.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/tables/446afc2c1cd9c950f89b25962908f8e79d058982d073c1a6845e8fa2b7ca1b0b.jpg)

![49c43155e2c04d145ee543971a75415054e6d9e9b4482b854fff91fe677acc21.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/tables/49c43155e2c04d145ee543971a75415054e6d9e9b4482b854fff91fe677acc21.jpg)

![796d16a71619a88ebaa5e427ea922fcbb940954461e362823839324e47352c1d.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/tables/796d16a71619a88ebaa5e427ea922fcbb940954461e362823839324e47352c1d.jpg)

![f872ca0834327c0bce6f43b7141cff23410a164f3f957312b45b382543d9905c.jpg](../icml_results/28_CSPO_%20Constraint-Sensitive%20Policy%20Optimization%20for%20Safe%20Reinforcement%20Learning/tables/f872ca0834327c0bce6f43b7141cff23410a164f3f957312b45b382543d9905c.jpg)

<a id="paper-0028"></a>

## Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes


### Images

![316dcb6203567893a2e06bbb1cace5661ebb42da2f190c01ff75aee2a975ed2b.jpg](../icml_results/29_Dynamic%20Programming%20for%20Epistemic%20Uncertainty%20in%20Markov%20Decision%20Processes/images/316dcb6203567893a2e06bbb1cace5661ebb42da2f190c01ff75aee2a975ed2b.jpg)

![c33c2d58c0ce5e608d3a39de69a0729990cbf3fbd8d6edf216bb0a9907812345.jpg](../icml_results/29_Dynamic%20Programming%20for%20Epistemic%20Uncertainty%20in%20Markov%20Decision%20Processes/images/c33c2d58c0ce5e608d3a39de69a0729990cbf3fbd8d6edf216bb0a9907812345.jpg)

![eacd59a983b437ce0a3db158ca2baa7c24be07773931c43c05b85e9141b478d6.jpg](../icml_results/29_Dynamic%20Programming%20for%20Epistemic%20Uncertainty%20in%20Markov%20Decision%20Processes/images/eacd59a983b437ce0a3db158ca2baa7c24be07773931c43c05b85e9141b478d6.jpg)

### Tables

![7e780f1344179141d3cb6fd0f5fdd34de35fe8fa2174cadd7ce39ffd73bfe9eb.jpg](../icml_results/29_Dynamic%20Programming%20for%20Epistemic%20Uncertainty%20in%20Markov%20Decision%20Processes/tables/7e780f1344179141d3cb6fd0f5fdd34de35fe8fa2174cadd7ce39ffd73bfe9eb.jpg)

![a620a87dd9853d1a4ece6d4d3c6c01645cf46d82563daaed21e65100563353e1.jpg](../icml_results/29_Dynamic%20Programming%20for%20Epistemic%20Uncertainty%20in%20Markov%20Decision%20Processes/tables/a620a87dd9853d1a4ece6d4d3c6c01645cf46d82563daaed21e65100563353e1.jpg)

<a id="paper-0029"></a>

## Skill Neologisms: Towards Skill-based Continual Learning


### Images

![0d35211636572b01ec15fedfca5e87105fb420c9eb3599d63186e386dfa4d40d.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/0d35211636572b01ec15fedfca5e87105fb420c9eb3599d63186e386dfa4d40d.jpg)

![33e9f2f3f94659b0db7f3c7cb319768a4dc05cd3b8db47b0b2c6243ae529854c.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/33e9f2f3f94659b0db7f3c7cb319768a4dc05cd3b8db47b0b2c6243ae529854c.jpg)

![3bc3d834b860add018ddbf4631e1e2f2669cfe94eecda557cb0a148b8e9e3bb4.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/3bc3d834b860add018ddbf4631e1e2f2669cfe94eecda557cb0a148b8e9e3bb4.jpg)

![59e775af16e3a7692d0ddf882773b7da19b27b189518f5f3d5cd973c08440814.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/59e775af16e3a7692d0ddf882773b7da19b27b189518f5f3d5cd973c08440814.jpg)

![87d377ba36a111b4b2c6899900badd19c0984466d70e3886680116ccae17ff20.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/87d377ba36a111b4b2c6899900badd19c0984466d70e3886680116ccae17ff20.jpg)

![a3440735c22237a7ee36d53679dc69fe776db286aa23451b88055cc5be327639.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/a3440735c22237a7ee36d53679dc69fe776db286aa23451b88055cc5be327639.jpg)

![cc13b67ea912d21e1908f5541dd5133fc6035d1c78eca7ef874d2b140d58872b.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/cc13b67ea912d21e1908f5541dd5133fc6035d1c78eca7ef874d2b140d58872b.jpg)

![da03770d6e82493c572b14f01e887c7c3dc8f92c9a2b579ca5e36e1ee9dfacfc.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/da03770d6e82493c572b14f01e887c7c3dc8f92c9a2b579ca5e36e1ee9dfacfc.jpg)

![dda2382459c5e97b3786dd9aa916d8fe1c4865b4ca974acd37024d16b4964bfb.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/dda2382459c5e97b3786dd9aa916d8fe1c4865b4ca974acd37024d16b4964bfb.jpg)

![ddb6dad62c5f81dddc57afc1e8bf524cc0ab06dfd0144bd991a8859444fdebd0.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/ddb6dad62c5f81dddc57afc1e8bf524cc0ab06dfd0144bd991a8859444fdebd0.jpg)

![f9de92aece5fa7d9634303f742248624edcc2c489d343f11468f619e13f01c8c.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/images/f9de92aece5fa7d9634303f742248624edcc2c489d343f11468f619e13f01c8c.jpg)

### Tables

![02efc89e457af67917f1602cea74833842437c737257edb915cc6cf60642bbf1.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/02efc89e457af67917f1602cea74833842437c737257edb915cc6cf60642bbf1.jpg)

![16ab0ad30017a9eb2c33ff99fd4ee1103316f7e9ed694b5e101eba34a8921fb9.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/16ab0ad30017a9eb2c33ff99fd4ee1103316f7e9ed694b5e101eba34a8921fb9.jpg)

![19d7847ce27d8c7a55f308e8809d0b7868a83a76de42bd59aa7719e4f08f6b85.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/19d7847ce27d8c7a55f308e8809d0b7868a83a76de42bd59aa7719e4f08f6b85.jpg)

![30ac431b7918d8775b8a47bb1ab1d916a1f093540a3985ab7bc6c3aa5a8e849c.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/30ac431b7918d8775b8a47bb1ab1d916a1f093540a3985ab7bc6c3aa5a8e849c.jpg)

![46be7cbc39beccf07ea41b6039ec946ca492762a92dff824409042bdd6db3a69.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/46be7cbc39beccf07ea41b6039ec946ca492762a92dff824409042bdd6db3a69.jpg)

![46ef94e91204f02e3e3d50f782fa1f0c5757210e37839817a42cf9f5fd17961c.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/46ef94e91204f02e3e3d50f782fa1f0c5757210e37839817a42cf9f5fd17961c.jpg)

![4b72ca0d4ba026b0563993b9e15b2b53d9ff6c49f354e312de9c25d50c069d17.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/4b72ca0d4ba026b0563993b9e15b2b53d9ff6c49f354e312de9c25d50c069d17.jpg)

![60a9eb8abd9b568ff82d2473bb2b8e7a3565f3250d25ae512269f2d5253159b9.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/60a9eb8abd9b568ff82d2473bb2b8e7a3565f3250d25ae512269f2d5253159b9.jpg)

![620506b2c0719a27e07ea270e6141b6e44976f56aad439f26a359ce734901e78.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/620506b2c0719a27e07ea270e6141b6e44976f56aad439f26a359ce734901e78.jpg)

![a21dca8645f925efdac78656bac419d774addb71e1b1e3de352fec818ce0ceaf.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/a21dca8645f925efdac78656bac419d774addb71e1b1e3de352fec818ce0ceaf.jpg)

![b105b485564ec57bff05a39e632c43a37f0f4b9bdb38529491b2fa2e7c9170e0.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/b105b485564ec57bff05a39e632c43a37f0f4b9bdb38529491b2fa2e7c9170e0.jpg)

![b27cf7c2a782a4c0e3aaa7dc274a26647193fa1f6ebe6491046d1ef706f73295.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/b27cf7c2a782a4c0e3aaa7dc274a26647193fa1f6ebe6491046d1ef706f73295.jpg)

![c52eaa98f8a5e43f69e9d32ebf97e1f3387d3bf8bda7146e6af9c7100b091a5f.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/c52eaa98f8a5e43f69e9d32ebf97e1f3387d3bf8bda7146e6af9c7100b091a5f.jpg)

![d5748a81fce24c2bdce2d46c39dcd3a136c1a79f5d90a4cb1ca6f79151b4f7e5.jpg](../icml_results/30_Skill%20Neologisms_%20Towards%20Skill-based%20Continual%20Learning/tables/d5748a81fce24c2bdce2d46c39dcd3a136c1a79f5d90a4cb1ca6f79151b4f7e5.jpg)

<a id="paper-0030"></a>

## STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control


### Images

![112f7f59f8732ceb0456a0546886f32d9f30360e8a2a2377dae08b4ee56ddb0b.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/112f7f59f8732ceb0456a0546886f32d9f30360e8a2a2377dae08b4ee56ddb0b.jpg)

![2d138ce0ac2a82f67aa86b3ffcd2925e4e840be6daf494589a1dc2b732c31af6.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/2d138ce0ac2a82f67aa86b3ffcd2925e4e840be6daf494589a1dc2b732c31af6.jpg)

![346e07c830a679c45529c0d78f17e02722e2b600eedca5bea742b88d8ccb111f.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/346e07c830a679c45529c0d78f17e02722e2b600eedca5bea742b88d8ccb111f.jpg)

![4e1fa7ade4ff91f5e5f96f3077291d146b14b95e7a0b4b7f84f9113a82b6e57a.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/4e1fa7ade4ff91f5e5f96f3077291d146b14b95e7a0b4b7f84f9113a82b6e57a.jpg)

![5df27b7d4793c882b292e4fc42a71bbec9df9be3d4ebfa1b00adc809ae024d3d.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/5df27b7d4793c882b292e4fc42a71bbec9df9be3d4ebfa1b00adc809ae024d3d.jpg)

![5f0b55be9f683f20a586f36ec22869b57e3f4eded81f8a7c2e2425c8d38e4ac8.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/5f0b55be9f683f20a586f36ec22869b57e3f4eded81f8a7c2e2425c8d38e4ac8.jpg)

![66a5e9aee14b33b7aa2910de34b7d7b3e2411c9ef5de980051bae2defdc90ca4.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/66a5e9aee14b33b7aa2910de34b7d7b3e2411c9ef5de980051bae2defdc90ca4.jpg)

![6827eff7eac0c68d4618c5a8b126d77ab99fdeb5ea575d92881301f54d32419f.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/6827eff7eac0c68d4618c5a8b126d77ab99fdeb5ea575d92881301f54d32419f.jpg)

![784b48efcc55b8714feb77d17f7138bbc20fddc40c8c4ed2cd3836caf1daa6f4.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/784b48efcc55b8714feb77d17f7138bbc20fddc40c8c4ed2cd3836caf1daa6f4.jpg)

![7bb39768bc5c56ee02536b4866fa6aea987b1c48235f114309b9b107f3138f8b.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/7bb39768bc5c56ee02536b4866fa6aea987b1c48235f114309b9b107f3138f8b.jpg)

![9e706b54e3be4278dd71c2953bdbe69e8e69d5dbf2e0db693008c8962b08041f.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/9e706b54e3be4278dd71c2953bdbe69e8e69d5dbf2e0db693008c8962b08041f.jpg)

![ca6e0358fe61b5211ac9cf1f416e1373663479f46157a23d428c149c8e4ed170.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/ca6e0358fe61b5211ac9cf1f416e1373663479f46157a23d428c149c8e4ed170.jpg)

![f195a067b811df36ad9b77304d343af149a6404f84747b8f300c93ffcc07a93f.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/images/f195a067b811df36ad9b77304d343af149a6404f84747b8f300c93ffcc07a93f.jpg)

### Tables

![036b461c68c52a89974623edfc0fe3c0f2e4a4db98c0523a11c7064ede36a542.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/036b461c68c52a89974623edfc0fe3c0f2e4a4db98c0523a11c7064ede36a542.jpg)

![2e2ad3ea663e273211cdfebda90be4882767638824fef1f4bf2b69c9d981803b.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/2e2ad3ea663e273211cdfebda90be4882767638824fef1f4bf2b69c9d981803b.jpg)

![2e4097c552f8619ad4ea1598960ac4151805a1e51b3dcf2ecb66a725fad5411c.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/2e4097c552f8619ad4ea1598960ac4151805a1e51b3dcf2ecb66a725fad5411c.jpg)

![30b7c414951f7ffcb3e8e471f81ff1ce5240aeefc343a1bfe6a8bbd127265e06.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/30b7c414951f7ffcb3e8e471f81ff1ce5240aeefc343a1bfe6a8bbd127265e06.jpg)

![3869ba2e8f612708e6b5797576f6cb5712b6d5d1808506a28d25cad9aa08aabe.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/3869ba2e8f612708e6b5797576f6cb5712b6d5d1808506a28d25cad9aa08aabe.jpg)

![4c57ad9dc21ac8f6faf1b0f0bee54c8fc7798ca541814846f274965a63b06889.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/4c57ad9dc21ac8f6faf1b0f0bee54c8fc7798ca541814846f274965a63b06889.jpg)

![5be7654ab5e51625ce82c77bc50e20e2bb1614835c8f47c0529885d414d83574.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/5be7654ab5e51625ce82c77bc50e20e2bb1614835c8f47c0529885d414d83574.jpg)

![91ed3b3359ebe63e36b2965072b5141bd2ea46dc40cefc2fac4e2f660eb10960.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/91ed3b3359ebe63e36b2965072b5141bd2ea46dc40cefc2fac4e2f660eb10960.jpg)

![9deeeba408efba442d79753d5436958d52bb423245fdfae23c294066380e9dc4.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/9deeeba408efba442d79753d5436958d52bb423245fdfae23c294066380e9dc4.jpg)

![c977b92ce6f06c1eb78a92dc2d57f59a7b759178e479888c9d7b3a7cef37131c.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/c977b92ce6f06c1eb78a92dc2d57f59a7b759178e479888c9d7b3a7cef37131c.jpg)

![d07cdaaa72c9a4d084541e15bd12e1819466ac7285c8115373ce29fc56587813.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/d07cdaaa72c9a4d084541e15bd12e1819466ac7285c8115373ce29fc56587813.jpg)

![d6f49a105c3a0a16d10a471d3b37f5dc78617823a3d7c28a615f29a63b24844e.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/d6f49a105c3a0a16d10a471d3b37f5dc78617823a3d7c28a615f29a63b24844e.jpg)

![e145dd8f25313dda387595872802daf17e41df10a65bbd42d8b6a556381b0190.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/e145dd8f25313dda387595872802daf17e41df10a65bbd42d8b6a556381b0190.jpg)

![fce7b73e036ed3f8454d1752b63d866327ecb57f30d4ce71fd4f6c47656c5ae2.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/fce7b73e036ed3f8454d1752b63d866327ecb57f30d4ce71fd4f6c47656c5ae2.jpg)

![ff3debbfd6803d63cfe1becf514ba1a541e7cade74d4f48b1c4c8bcbaf3113da.jpg](../icml_results/31_STAR-KV_%20Low-Rank%20KV%20Cache%20Compression%20via%20Soft%20Thresholding%20for%20Adaptive%20Rank%20Control/tables/ff3debbfd6803d63cfe1becf514ba1a541e7cade74d4f48b1c4c8bcbaf3113da.jpg)

<a id="paper-0031"></a>

## Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning


### Images

![6b6cafd9c5403e2ece82e6c8d3bbdb687d19eb5b7508d3dc34062c959716f235.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/6b6cafd9c5403e2ece82e6c8d3bbdb687d19eb5b7508d3dc34062c959716f235.jpg)

![8005f4b89c714eef1106407fb0b078fe762165812d87c719e1f438636d6fe457.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/8005f4b89c714eef1106407fb0b078fe762165812d87c719e1f438636d6fe457.jpg)

![8a3d2245d6c08ad34f03e2ed85ed66accc15682fa4fb3da8ac305faa653f882b.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/8a3d2245d6c08ad34f03e2ed85ed66accc15682fa4fb3da8ac305faa653f882b.jpg)

![960512cceee3e3d8d8fad984110a6c244b592f9ba7af0717c282677b53ddec5a.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/960512cceee3e3d8d8fad984110a6c244b592f9ba7af0717c282677b53ddec5a.jpg)

![cc3adca658f0efde4fe41f1484c56754963c0fed33699be7f5ff724a238df153.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/cc3adca658f0efde4fe41f1484c56754963c0fed33699be7f5ff724a238df153.jpg)

![cc56f144f31e75af2ff168233eb98b6487f7bcb53716a871aab8919e27603f25.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/cc56f144f31e75af2ff168233eb98b6487f7bcb53716a871aab8919e27603f25.jpg)

![d0d91f03b000221845c2c9b0eec1ae607274d2fe0680e7002e191289ea09470d.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/d0d91f03b000221845c2c9b0eec1ae607274d2fe0680e7002e191289ea09470d.jpg)

![d3d123a2976757f62665d50c13c789a9c8d851b35c68487ca9bbb7ebbc342ad9.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/d3d123a2976757f62665d50c13c789a9c8d851b35c68487ca9bbb7ebbc342ad9.jpg)

![e40870c5db765fd82f973c25a9883c269d75cdd79915278301385d55463917db.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/images/e40870c5db765fd82f973c25a9883c269d75cdd79915278301385d55463917db.jpg)

### Tables

![1f4f66e704fc704dde16e632bc6297ae6a682b2a05666109f547a3a8c30660ed.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/tables/1f4f66e704fc704dde16e632bc6297ae6a682b2a05666109f547a3a8c30660ed.jpg)

![2b48b4821f6bf4ab68d39d0d88209247a360989fc763fa0373d137b47a97e52d.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/tables/2b48b4821f6bf4ab68d39d0d88209247a360989fc763fa0373d137b47a97e52d.jpg)

![7e633caf0ce823b8e0410b6a5464bababd32f9a211a6971f6bf1e164609e870b.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/tables/7e633caf0ce823b8e0410b6a5464bababd32f9a211a6971f6bf1e164609e870b.jpg)

![8527cf7075a757618f679d4eba92b71063a379acea51c11e3345b86f93894fa1.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/tables/8527cf7075a757618f679d4eba92b71063a379acea51c11e3345b86f93894fa1.jpg)

![9c443cd30ba62138e2f92f29e5382f6e9019e1f57e2d6c2361086a7bb98e3406.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/tables/9c443cd30ba62138e2f92f29e5382f6e9019e1f57e2d6c2361086a7bb98e3406.jpg)

![c7b73ff37a39c6fa90bf292074fbced227f45b00766ef7eae17a4832f97fe6b5.jpg](../icml_results/32_Reward%20and%20Guidance%20through%20Rubrics_%20Promoting%20Exploration%20to%20Improve%20Multi-Domain%20Reasoning/tables/c7b73ff37a39c6fa90bf292074fbced227f45b00766ef7eae17a4832f97fe6b5.jpg)

<a id="paper-0032"></a>

## Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers


### Images

![04b0ee17309de6bf2e55f6aa2420e87d2613fcb4658127d072947b6ad259ae57.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/04b0ee17309de6bf2e55f6aa2420e87d2613fcb4658127d072947b6ad259ae57.jpg)

![06d846c24586983460f532d88b85a4a247b1f635af3c277980250bb0440527aa.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/06d846c24586983460f532d88b85a4a247b1f635af3c277980250bb0440527aa.jpg)

![09ba34839a418a708b8405752d06aab9987fa6867f2e9513893193d680735ec6.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/09ba34839a418a708b8405752d06aab9987fa6867f2e9513893193d680735ec6.jpg)

![0ae69faf9ea5046036dcb21fd8cf5837270fbd23d976c5ef3ace257d7d39b7e9.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/0ae69faf9ea5046036dcb21fd8cf5837270fbd23d976c5ef3ace257d7d39b7e9.jpg)

![12704092cc53187184765a6d26668cdd0ab19c39e0d4bb95dfd0dd0c649b69fa.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/12704092cc53187184765a6d26668cdd0ab19c39e0d4bb95dfd0dd0c649b69fa.jpg)

![29b7dfee2ab1bb5c25f3fd0424bd61ea66ded65465b075ddd84f3a490e226cd4.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/29b7dfee2ab1bb5c25f3fd0424bd61ea66ded65465b075ddd84f3a490e226cd4.jpg)

![3bd009b88adb384782da21fad5f3f4b1a5c7db09701058c40c1cb08c1c003964.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/3bd009b88adb384782da21fad5f3f4b1a5c7db09701058c40c1cb08c1c003964.jpg)

![4c85729df58ebbdd9e06a75c9e8bb7295636f22f0e6d1f657a211eb366c5f5fa.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/4c85729df58ebbdd9e06a75c9e8bb7295636f22f0e6d1f657a211eb366c5f5fa.jpg)

![57c5efdeeec215c44a1ad5082f0d7d13b1e758d1647e1fb03cf411ca9dc10873.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/57c5efdeeec215c44a1ad5082f0d7d13b1e758d1647e1fb03cf411ca9dc10873.jpg)

![7c18b06da50e62bdb3846db358fe529c82b2e399ec5f770faa8d3243fea4dc3b.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/7c18b06da50e62bdb3846db358fe529c82b2e399ec5f770faa8d3243fea4dc3b.jpg)

![8d0dd1411788a378ce427a81b80a750e06f1083f4e72abe9f8cc14d24a414d7d.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/8d0dd1411788a378ce427a81b80a750e06f1083f4e72abe9f8cc14d24a414d7d.jpg)

![a59e69230e9d6d23d977c5544837dddc331d88cfae66270860ea78bc92324562.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/a59e69230e9d6d23d977c5544837dddc331d88cfae66270860ea78bc92324562.jpg)

![b223ad4c701f3de6ed26c29b85a97cf017e81f9ffba1b53dc1c4d08fd257ebc9.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/b223ad4c701f3de6ed26c29b85a97cf017e81f9ffba1b53dc1c4d08fd257ebc9.jpg)

![bba2514266062be46e30a3026db9b9ca592b66b533d7c3a411eb3547fddfece7.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/bba2514266062be46e30a3026db9b9ca592b66b533d7c3a411eb3547fddfece7.jpg)

![c0fcf239eeab18118de7ceb7558cf0779f31361db9738920c0778be0d24c0bf2.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/c0fcf239eeab18118de7ceb7558cf0779f31361db9738920c0778be0d24c0bf2.jpg)

![c197553217088200885a40bbd1c3ee997689514d05c7347f77b9245a2348ecce.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/c197553217088200885a40bbd1c3ee997689514d05c7347f77b9245a2348ecce.jpg)

![c8c0ec37e134151154944d0594af30b38612750c9009a26a565d43aa1a88250c.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/c8c0ec37e134151154944d0594af30b38612750c9009a26a565d43aa1a88250c.jpg)

![ceb87635c0d34670b4c35c5112da06a9286ac055458fd5013514ef0aa6cc2e40.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/ceb87635c0d34670b4c35c5112da06a9286ac055458fd5013514ef0aa6cc2e40.jpg)

![d399e70d94b2298dce4c76a87d7ad1fc21122e7bd626757ffffb4e166ca9a1bf.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/d399e70d94b2298dce4c76a87d7ad1fc21122e7bd626757ffffb4e166ca9a1bf.jpg)

![d4712a1205f0cfe0d285f47e9f24d6df2b4c1862e50d209f32a81a3586929301.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/d4712a1205f0cfe0d285f47e9f24d6df2b4c1862e50d209f32a81a3586929301.jpg)

![e708d5af40c559b9229f8c6c4ee04c20418210bdd34dd2a2bb3abb868008fbea.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/e708d5af40c559b9229f8c6c4ee04c20418210bdd34dd2a2bb3abb868008fbea.jpg)

![fdd63583508e2524722a02b2388a572393157bf55fcd5ecc43ad6f1451a655d8.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/images/fdd63583508e2524722a02b2388a572393157bf55fcd5ecc43ad6f1451a655d8.jpg)

### Tables

![189afcd1f1b23297d9669fa7251d905834d6852c752c1f7e55684d0a81967b4a.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/189afcd1f1b23297d9669fa7251d905834d6852c752c1f7e55684d0a81967b4a.jpg)

![37ca3746a49c6ee113c1e9b3c11197f7326af9a8e2a170be33468bb613f5e842.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/37ca3746a49c6ee113c1e9b3c11197f7326af9a8e2a170be33468bb613f5e842.jpg)

![7a3c093431033c557fb40974be81e32e1b005af9d0c32df575b1faaf8641485e.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/7a3c093431033c557fb40974be81e32e1b005af9d0c32df575b1faaf8641485e.jpg)

![887072f0f7c040884db2b0fd94e934eab8b20862a3a14ffde5aaabde8285c187.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/887072f0f7c040884db2b0fd94e934eab8b20862a3a14ffde5aaabde8285c187.jpg)

![8fd52ac41ee480e259d6fcf3782a934f72c1a15b370f15414cfe6789e56ffaff.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/8fd52ac41ee480e259d6fcf3782a934f72c1a15b370f15414cfe6789e56ffaff.jpg)

![b0872dd4796d404d0cc8ffdab820aeb2f989e670acc292a488451cb78ea2c14a.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/b0872dd4796d404d0cc8ffdab820aeb2f989e670acc292a488451cb78ea2c14a.jpg)

![b61d0ed1b350a36dff3e945e6de3463a9addb2336b2dbcedf5f5a6b92a2fbdba.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/b61d0ed1b350a36dff3e945e6de3463a9addb2336b2dbcedf5f5a6b92a2fbdba.jpg)

![be2e9b3f1443a5525ea4a5a49b9ddc9421f1384646aed6a901f5d39d307b7aff.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/be2e9b3f1443a5525ea4a5a49b9ddc9421f1384646aed6a901f5d39d307b7aff.jpg)

![d5b16f6ce1c67acf97dadacf166951785225f0855ef7837538f5acd6f86b5a39.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/d5b16f6ce1c67acf97dadacf166951785225f0855ef7837538f5acd6f86b5a39.jpg)

![ea4837c98b757c486cee71ec5d8a763a4baf4ade4a0df54c70be8b851ca14f8b.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/ea4837c98b757c486cee71ec5d8a763a4baf4ade4a0df54c70be8b851ca14f8b.jpg)

![eb3e3e47dcc4ec0a722e1cb4740015407654a0816c588d67eff9ad04a82db24f.jpg](../icml_results/33_Rex_%20A%20Family%20of%20Reversible%20Exponential%20%28Stochastic%29%20Runge-Kutta%20Solvers/tables/eb3e3e47dcc4ec0a722e1cb4740015407654a0816c588d67eff9ad04a82db24f.jpg)

<a id="paper-0033"></a>

## Optimal Transport under Group Fairness Constraints

### Images

![0eb65273df68d0cd52f8d85ae2f4080dfaffee4b6c52b2e49de12a909c93a716.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/0eb65273df68d0cd52f8d85ae2f4080dfaffee4b6c52b2e49de12a909c93a716.jpg)

![37f4e8a1371b4649448a58ab7e4fe1762cb8aebf900cf8c3a6146afd0315e092.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/37f4e8a1371b4649448a58ab7e4fe1762cb8aebf900cf8c3a6146afd0315e092.jpg)

![579ca2555662f76378c71b0b8616a1fb8ea06f667ce7d241c89abfce2f9b9044.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/579ca2555662f76378c71b0b8616a1fb8ea06f667ce7d241c89abfce2f9b9044.jpg)

![90f6e20734e494d3b9e8699de8916f21b07faca2db55230ce3808d1aacc31b04.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/90f6e20734e494d3b9e8699de8916f21b07faca2db55230ce3808d1aacc31b04.jpg)

![b67adccf75311204bd100fe95d7f198fa2ca5e6f7992bf581b6842a5839db3fd.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/b67adccf75311204bd100fe95d7f198fa2ca5e6f7992bf581b6842a5839db3fd.jpg)

![cde0128bcbfb8003cfa7c1fb199ae3dcb93cd84308480cdc06b959be4b01dc9e.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/cde0128bcbfb8003cfa7c1fb199ae3dcb93cd84308480cdc06b959be4b01dc9e.jpg)

![d463b34c8a3a837e84151cf27f0414a4167376833771cebdb604d99028fcbdb1.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/d463b34c8a3a837e84151cf27f0414a4167376833771cebdb604d99028fcbdb1.jpg)

![ec1b812b3e0bfba3559d8975923cd9389ba2aef2006caac179623ec3138bd617.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/ec1b812b3e0bfba3559d8975923cd9389ba2aef2006caac179623ec3138bd617.jpg)

### Tables

![d4e83d725967d13e5627c63b712fe1d18232461ef290e6f5868478fa612316b5.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/tables/d4e83d725967d13e5627c63b712fe1d18232461ef290e6f5868478fa612316b5.jpg)
