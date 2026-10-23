# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 4 of 197

## 目录 (Table of Contents)

1. [Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression](#paper-0100)
2. [Eigenvectors of Experts are Training-free Non-collapsing Routers](#paper-0101)
3. [Rare Event Analysis of Large Language Models](#paper-0102)
4. [From Distribution to Geometry: Stable Graph Generalization via Invariant Barycenters](#paper-0103)
5. [On the Difficulty of Learning a Meta-network for Training Data Selection](#paper-0104)
6. [Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](#paper-0105)
7. [Mixtures Closest To A Given Measure: A Semidefinite Programming Approach](#paper-0106)
8. [HOBIT: Hardness Optimized Batch Sampling for InfoNCE Training](#paper-0107)
9. [FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs](#paper-0108)
10. [Learning Randomized Reductions](#paper-0109)
11. [Evaluating Robustness of Reasoning Models on Parameterized Logical Problems](#paper-0110)
12. [XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations](#paper-0111)
13. [Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking](#paper-0112)
14. [Clustering in Deep Stochastic Transformers](#paper-0113)
15. [Multimodal Nested Learning for Decoupled and Coordinated Optimization](#paper-0114)
16. [L2G-NET: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](#paper-0115)
17. [Exact Functional ANOVA Decomposition for Categorical Inputs Models](#paper-0116)
18. [Learning to Execute Graph Algorithms Exactly with Graph Neural Networks](#paper-0117)
19. [On Efficient Scaling of GNNs via IO-Aware Layers Implementations](#paper-0118)
20. [Rethinking LLM Ensembling from the Perspective of Mixture Models](#paper-0119)
21. [VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding](#paper-0120)
22. [Self-Soupervision: Cooking Model Soups without Labels](#paper-0121)
23. [Demystifying Entropy Control in LLM RL Training: Theoretical Analysis and Dynamic Scheduling](#paper-0122)
24. [Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights](#paper-0123)
25. [EntroKV: Entropy-Guided Dynamic Budget Allocation for KV-Cache Compression](#paper-0124)
26. [Excited Pfaffians: Generalized Neural Wave Functions Across Structure and State](#paper-0125)
27. [EEmo-Logic: A Unified Dataset and Multi-Stage Framework for Comprehensive Image-Evoked Emotion Assessment](#paper-0126)
28. [Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence](#paper-0127)
29. [DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos](#paper-0128)
30. [Optimal structure learning and conditional independence testing](#paper-0129)
31. [Biased Generalization in Diffusion Models](#paper-0130)
32. [High-accuracy and dimension-free sampling with diffusions](#paper-0131)
33. [Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models](#paper-0132)

---


<a id="paper-0100"></a>

## Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression

### Images

![0e426bcdfe8f504a9cf714626a9b64a35c9ce5d275b69321911a75ad28683cfc.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/0e426bcdfe8f504a9cf714626a9b64a35c9ce5d275b69321911a75ad28683cfc.jpg)

![1f61c121973b96913052c93664dda562f857884882fc26fc6a1d49380d9ed46e.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/1f61c121973b96913052c93664dda562f857884882fc26fc6a1d49380d9ed46e.jpg)

![2180e2b4cd682faa10ff998faddaf10ba80107905c86bf0b1618677ad1c87dcb.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/2180e2b4cd682faa10ff998faddaf10ba80107905c86bf0b1618677ad1c87dcb.jpg)

![5d884d5e38a3d7206af3f0d575ef617126d761ed4b12a33b523caf1be1ba721f.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/5d884d5e38a3d7206af3f0d575ef617126d761ed4b12a33b523caf1be1ba721f.jpg)

![7cf9d5bb37a3fd621431bd1485c2f645fdd1bda054a7b640735a9e4a1f80db55.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/7cf9d5bb37a3fd621431bd1485c2f645fdd1bda054a7b640735a9e4a1f80db55.jpg)

![a8e0a7263c10a340a09fa658319b60657521b65488d70040f98a551cf258102e.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/a8e0a7263c10a340a09fa658319b60657521b65488d70040f98a551cf258102e.jpg)

![c6fc26bd066b42f2b10c08e04c1c4f8904db4a3c103857265604dac5718eae40.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/c6fc26bd066b42f2b10c08e04c1c4f8904db4a3c103857265604dac5718eae40.jpg)

![e0ff7c1553de3d47a890217d819d1473b0c389eeac3629dc38685719b16a8a86.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/e0ff7c1553de3d47a890217d819d1473b0c389eeac3629dc38685719b16a8a86.jpg)

![e20b64499a3a1be86c1de45eb81bdf6ee58c1f1b6a4003b0e41c758998419215.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/e20b64499a3a1be86c1de45eb81bdf6ee58c1f1b6a4003b0e41c758998419215.jpg)

![ed6d1158ee0ec3abbbec193e484268a82a5f6577ef2211dc3b86b612a7652c6a.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/ed6d1158ee0ec3abbbec193e484268a82a5f6577ef2211dc3b86b612a7652c6a.jpg)

![ed94bbe9a4c818abdd51ac5dd3bfef3c966943f9ed4f218daaf50b210124e6d6.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/ed94bbe9a4c818abdd51ac5dd3bfef3c966943f9ed4f218daaf50b210124e6d6.jpg)

### Tables

![103597431153595eb96bb27ac15103d7195e3cfc53014fb445b12e5691c29a04.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/103597431153595eb96bb27ac15103d7195e3cfc53014fb445b12e5691c29a04.jpg)

![22961eea71e88366f821996af15508376a584202fa764002811357eb62c1c847.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/22961eea71e88366f821996af15508376a584202fa764002811357eb62c1c847.jpg)

![2ddec07ed883f1a6cc83b59f983d2968707a5c7ab592e46a83f35104bb6107aa.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/2ddec07ed883f1a6cc83b59f983d2968707a5c7ab592e46a83f35104bb6107aa.jpg)

![38b378aa09822d159a454f1bd2afa791972f9a7621f4bf7c5e1d557915900810.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/38b378aa09822d159a454f1bd2afa791972f9a7621f4bf7c5e1d557915900810.jpg)

![3d903430d95f6d392860f70c4840ceb3f3823f9186d3de92df5b32557c0262a2.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/3d903430d95f6d392860f70c4840ceb3f3823f9186d3de92df5b32557c0262a2.jpg)

![47e90165c6bea55863929b6f25e465f2712ceab6ebab9ca1ca1474fb23841ce7.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/47e90165c6bea55863929b6f25e465f2712ceab6ebab9ca1ca1474fb23841ce7.jpg)

![493ce74ae798b54ce74a7b902e7c4650ca386c9b42e03147dc79a68c4949265f.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/493ce74ae798b54ce74a7b902e7c4650ca386c9b42e03147dc79a68c4949265f.jpg)

![7c3ad1694e95409a92729940a29d72c46fd887103c5d66d014de1193c7b87520.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/7c3ad1694e95409a92729940a29d72c46fd887103c5d66d014de1193c7b87520.jpg)

![9705c8612e4de548207e96796a415d6a924ddacecf4a5baa409d575388cce3ec.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/9705c8612e4de548207e96796a415d6a924ddacecf4a5baa409d575388cce3ec.jpg)

![9b626b9f31eca30e719e71056cac3e5add1cdc2ef8f5dbb98e87a71d0201102d.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/9b626b9f31eca30e719e71056cac3e5add1cdc2ef8f5dbb98e87a71d0201102d.jpg)

## Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression


### Images

![03e535251fd8fe82db280bdbf73ce71d75dff4b1f5fae8bdfd32cbbd6b620101.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/03e535251fd8fe82db280bdbf73ce71d75dff4b1f5fae8bdfd32cbbd6b620101.jpg)

![06074418943596105fc4a33c4cdfc31ac8cc25fbbe1b7c6686ba759d93fdd255.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/06074418943596105fc4a33c4cdfc31ac8cc25fbbe1b7c6686ba759d93fdd255.jpg)

![1827c877941aa8d2cb5795667ea7adc3f9c3d30adf1f7b8fb1a058eb9db41d22.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/1827c877941aa8d2cb5795667ea7adc3f9c3d30adf1f7b8fb1a058eb9db41d22.jpg)

![285567ff6043869d9b2699962e75b9212db8b1a13cea0d9287d62a3d3b1e48c1.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/285567ff6043869d9b2699962e75b9212db8b1a13cea0d9287d62a3d3b1e48c1.jpg)

![3d2fda15bbc1623ce5e3116945f7c56c74a21b5d19e0e7541c2b0be80cffa3c1.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/3d2fda15bbc1623ce5e3116945f7c56c74a21b5d19e0e7541c2b0be80cffa3c1.jpg)

![5974417580fe464ca0430fbebe718d9a10e2f130a066197d9d0679d01573bb76.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/5974417580fe464ca0430fbebe718d9a10e2f130a066197d9d0679d01573bb76.jpg)

![649e483972201e5725a10a93c906c5c4641efb6b3bea95a5e09bee1eecc3d324.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/649e483972201e5725a10a93c906c5c4641efb6b3bea95a5e09bee1eecc3d324.jpg)

![679f78a4a4dfaad80e9f122b27faa21018a3ec45c0beb535b97411bd04f24346.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/679f78a4a4dfaad80e9f122b27faa21018a3ec45c0beb535b97411bd04f24346.jpg)

![73a9cd2d821d8c69080f7010602c0ed9f3d822070d84405d9d01e78a750838ec.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/73a9cd2d821d8c69080f7010602c0ed9f3d822070d84405d9d01e78a750838ec.jpg)

![9170b15cd09f69abc9f24e7ea744ccb9ecc1fbd0a503cd104ab133f61ef8a20a.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/9170b15cd09f69abc9f24e7ea744ccb9ecc1fbd0a503cd104ab133f61ef8a20a.jpg)

![ab6127c9a08a9b8343c985a00944a0964ecf4fe5c7b68a2f0e49ae6550d87922.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/ab6127c9a08a9b8343c985a00944a0964ecf4fe5c7b68a2f0e49ae6550d87922.jpg)

![b321b2966f3d207b2d99b91c143d673520da5ceb1268a942a31b783ebd8c0372.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/b321b2966f3d207b2d99b91c143d673520da5ceb1268a942a31b783ebd8c0372.jpg)

![dcba5b1a149759b5fd7fb6faf24c96a58ce670b0c59c812d2df1332979fd2969.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/dcba5b1a149759b5fd7fb6faf24c96a58ce670b0c59c812d2df1332979fd2969.jpg)

![eb66c0ee1d1802efa91665566208dfbd7ff08a118391121e9c0f1e1987d1fe07.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/eb66c0ee1d1802efa91665566208dfbd7ff08a118391121e9c0f1e1987d1fe07.jpg)

![f5c8042cd5e425a2af165441fe79459c35251749fdd2659443fd125afb6232c1.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/f5c8042cd5e425a2af165441fe79459c35251749fdd2659443fd125afb6232c1.jpg)

![fbbb2d63f4b36f95711c379fe0b4349b9bdb9540e2c14d689324964dc29b017d.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/images/fbbb2d63f4b36f95711c379fe0b4349b9bdb9540e2c14d689324964dc29b017d.jpg)

### Tables

![1f9413adaa7028eabb078d460e93f7840d08500eadd5b4843d26df9b8ac4450b.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/1f9413adaa7028eabb078d460e93f7840d08500eadd5b4843d26df9b8ac4450b.jpg)

![3cd1e6ae6fddd5ee3a06ba1a916204a610147b51f5353dc7c39f8a0a303d0780.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/3cd1e6ae6fddd5ee3a06ba1a916204a610147b51f5353dc7c39f8a0a303d0780.jpg)

![437eaa12a58fa7e9549d4c58db47064a81f66e469ffd6bf142d6b89b9cf1b64a.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/437eaa12a58fa7e9549d4c58db47064a81f66e469ffd6bf142d6b89b9cf1b64a.jpg)

![4ea888be2b52ef343782bb683e8ed9249f65ad942845a1aaf85f908d7217ddfb.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/4ea888be2b52ef343782bb683e8ed9249f65ad942845a1aaf85f908d7217ddfb.jpg)

![531d8d1b94dc06d84f77eb80765751dd916d351833a813c26f6a55e1320db2fe.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/531d8d1b94dc06d84f77eb80765751dd916d351833a813c26f6a55e1320db2fe.jpg)

![55163594e0268399a13611a9191d085267884707f7af6c140c68bb51e153a78e.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/55163594e0268399a13611a9191d085267884707f7af6c140c68bb51e153a78e.jpg)

![58285c0c0af173e80a2538f312941ab387cc84933a894042de08a5f83c3c6426.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/58285c0c0af173e80a2538f312941ab387cc84933a894042de08a5f83c3c6426.jpg)

![69871d77222cd317a5f220793a85eec3a664e1a5f4ae854dd9abe1c85af0a38f.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/69871d77222cd317a5f220793a85eec3a664e1a5f4ae854dd9abe1c85af0a38f.jpg)

![756db95627bbaac937defc57d3a665b2d99453bec00f19d986f93b1fdcf2a500.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/756db95627bbaac937defc57d3a665b2d99453bec00f19d986f93b1fdcf2a500.jpg)

![7777665b955d50ecb57f8f2c9e9970c57bd76bd9f427f787f4488bb5502804e6.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/7777665b955d50ecb57f8f2c9e9970c57bd76bd9f427f787f4488bb5502804e6.jpg)

![82b80faa3233a82073adc8e0ef351eb30d67ed72bdc3f5d860a4bb4f2ed2a8cf.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/82b80faa3233a82073adc8e0ef351eb30d67ed72bdc3f5d860a4bb4f2ed2a8cf.jpg)

![9952d5c750741bf7fbc73dff82c42c5bf58b5af70d669a506e387d5ea6cc92a3.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/9952d5c750741bf7fbc73dff82c42c5bf58b5af70d669a506e387d5ea6cc92a3.jpg)

![9967e164a85311a48f5400121b4d5cb1a32d903e3be3213123fe53e27b6791d6.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/9967e164a85311a48f5400121b4d5cb1a32d903e3be3213123fe53e27b6791d6.jpg)

![9a583a18eb4ec3102076c5590f90be32ddb9184d12e6fda4bab4138349088238.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/9a583a18eb4ec3102076c5590f90be32ddb9184d12e6fda4bab4138349088238.jpg)

![bcbf41f7a803df18e89563200c40e699c87eff8e78d341d6f9a380cfbca6e4a3.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/bcbf41f7a803df18e89563200c40e699c87eff8e78d341d6f9a380cfbca6e4a3.jpg)

![c21dcce848be670d389088a866ac2ecdc61b4fceb3e19c45ff2bc4221fc98aa6.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/c21dcce848be670d389088a866ac2ecdc61b4fceb3e19c45ff2bc4221fc98aa6.jpg)

![ccbc1f172f47f538092f4d86fcb1058833359b731d09e5099948e5d702c5ef09.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/ccbc1f172f47f538092f4d86fcb1058833359b731d09e5099948e5d702c5ef09.jpg)

![d1848c525a7e461ceb848837b667b855284974140d9519470f3e75bb17f2f272.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/d1848c525a7e461ceb848837b667b855284974140d9519470f3e75bb17f2f272.jpg)

![ea26ec02486e11bcb9beab2f71e285b5427d94cadc9eaa628da9dc03d8ace775.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/ea26ec02486e11bcb9beab2f71e285b5427d94cadc9eaa628da9dc03d8ace775.jpg)

![f52be1b8965aee1316b976fa0c5acef85e32b091fdd22c07a59baa1f86bb1bc3.jpg](../icml_results/101_Attribution-Guided%20and%20Coverage-Maximized%20Pruning%20for%20Structural%20MoE%20Compression/tables/f52be1b8965aee1316b976fa0c5acef85e32b091fdd22c07a59baa1f86bb1bc3.jpg)

<a id="paper-0101"></a>

## Eigenvectors of Experts are Training-free Non-collapsing Routers


### Images

![012eb9f125f8f0f00d452e1ed4aff578b500fe2e9a5203cf514670568e5103e8.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/012eb9f125f8f0f00d452e1ed4aff578b500fe2e9a5203cf514670568e5103e8.jpg)

![1ec0015283ad6bdf5d4443eef1dede2d87a1c227fb3dd117bf0f3b0cf83c7d85.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/1ec0015283ad6bdf5d4443eef1dede2d87a1c227fb3dd117bf0f3b0cf83c7d85.jpg)

![3c6af892ebd83e342367e97f1fcd479dec3bde3b2378813a949bb975a48e4067.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/3c6af892ebd83e342367e97f1fcd479dec3bde3b2378813a949bb975a48e4067.jpg)

![4641ff6c8717e3f44f6ea054593abc650e23aaf225e6ccce2ca6f5c7cebcbc01.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/4641ff6c8717e3f44f6ea054593abc650e23aaf225e6ccce2ca6f5c7cebcbc01.jpg)

![48f4529db745bdbca29f5a9517d94949df287456d9e7d2ebd5a144e01f27d1ce.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/48f4529db745bdbca29f5a9517d94949df287456d9e7d2ebd5a144e01f27d1ce.jpg)

![624ba818d3b3cfa5ee6d68e3c097dd7afc19fcbf6e3a2da47e147acb6fca7d81.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/624ba818d3b3cfa5ee6d68e3c097dd7afc19fcbf6e3a2da47e147acb6fca7d81.jpg)

![a08ec20ec6e71611ed75e657a74f0ff0011104002f033ce8c8583b7734502730.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/a08ec20ec6e71611ed75e657a74f0ff0011104002f033ce8c8583b7734502730.jpg)

![b2a00d39fe86bd741303b728f34c78f5ef332644d22befe3018b89608535af1e.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/images/b2a00d39fe86bd741303b728f34c78f5ef332644d22befe3018b89608535af1e.jpg)

### Tables

![269874970826cf8ee5a04c891ccc125bacc751d102ee4fc1b9849a9f8765c6da.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/269874970826cf8ee5a04c891ccc125bacc751d102ee4fc1b9849a9f8765c6da.jpg)

![393e75eae6ffbba883fa737a8a1062e785ce9f4146408241cbb21fbfe2900764.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/393e75eae6ffbba883fa737a8a1062e785ce9f4146408241cbb21fbfe2900764.jpg)

![50d7dba7ef71418cd99ff6da5288d2e1fa77a9e803c4617c132b42e3b361c13b.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/50d7dba7ef71418cd99ff6da5288d2e1fa77a9e803c4617c132b42e3b361c13b.jpg)

![55c805d9af8157b681a7c7b7ef43b6e2e9d384e670ce2d6a9b797e41d81c0bd9.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/55c805d9af8157b681a7c7b7ef43b6e2e9d384e670ce2d6a9b797e41d81c0bd9.jpg)

![66fcbac508da09e0411a4a18a35278eab777d57041832ce1f3e04a33b3203730.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/66fcbac508da09e0411a4a18a35278eab777d57041832ce1f3e04a33b3203730.jpg)

![770556239519e85f3b24010f9124f76379a26280031e6306e789090c1520b2b5.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/770556239519e85f3b24010f9124f76379a26280031e6306e789090c1520b2b5.jpg)

![8024c56487ee74dffc083c782e8af3a60d682c8cf804763a9aa214a4402ca5a0.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/8024c56487ee74dffc083c782e8af3a60d682c8cf804763a9aa214a4402ca5a0.jpg)

![984a37adc6f3e2f859fd7e47fd9c13acc268caedc277d873b77a3218eb543e21.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/984a37adc6f3e2f859fd7e47fd9c13acc268caedc277d873b77a3218eb543e21.jpg)

![9ae0bd1efde522446ef3bd26603441f44d61f9661333ca2a1a1def58b28ce0af.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/9ae0bd1efde522446ef3bd26603441f44d61f9661333ca2a1a1def58b28ce0af.jpg)

![9b5ffb97c4cdfb493f55f20ee1aedfbedb42a9cee9038502fd6df8fa286ecf95.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/9b5ffb97c4cdfb493f55f20ee1aedfbedb42a9cee9038502fd6df8fa286ecf95.jpg)

![9db1895ac834778e64d00eeed9908638b777b3808ab96b1335b87c6ca44f9e90.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/9db1895ac834778e64d00eeed9908638b777b3808ab96b1335b87c6ca44f9e90.jpg)

![ab8856551b137f3de2b888a9fe544d8d0fe8c597b082c42c295ddf6031695546.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/ab8856551b137f3de2b888a9fe544d8d0fe8c597b082c42c295ddf6031695546.jpg)

![c131425123b577d130fd761ab3d9dc968ee062a39685237fc7dbf858bd5bb985.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/c131425123b577d130fd761ab3d9dc968ee062a39685237fc7dbf858bd5bb985.jpg)

![c6e99c3df8bf82571a4241be6981b5fd3cf491709bea0a9b369e106c76e363c4.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/c6e99c3df8bf82571a4241be6981b5fd3cf491709bea0a9b369e106c76e363c4.jpg)

![f7512b2cc2bad96599ed3e4d33fc66f576b1d1bb4d47d35eea0ba310aec200df.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/f7512b2cc2bad96599ed3e4d33fc66f576b1d1bb4d47d35eea0ba310aec200df.jpg)

![fd9f1fd06b7eaa069bab6e15873869970daaec8a7f8e573b673ffdd14160a3ae.jpg](../icml_results/102_Eigenvectors%20of%20Experts%20are%20Training-free%20Non-collapsing%20Routers/tables/fd9f1fd06b7eaa069bab6e15873869970daaec8a7f8e573b673ffdd14160a3ae.jpg)

<a id="paper-0102"></a>

## Rare Event Analysis of Large Language Models


### Images

![445da65ad7dcb16a821a953e2fa892668daa6fa745210612370cb3e2d62c0ebf.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/445da65ad7dcb16a821a953e2fa892668daa6fa745210612370cb3e2d62c0ebf.jpg)

![543c49b1a82f29d8666b8a6bce816c1be947fc81609b509229d07cce3ba8518d.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/543c49b1a82f29d8666b8a6bce816c1be947fc81609b509229d07cce3ba8518d.jpg)

![6428a48665fd88fa84b19971741e4efc8137cf4379770d155ca2570c205052b6.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/6428a48665fd88fa84b19971741e4efc8137cf4379770d155ca2570c205052b6.jpg)

![7dd1527ec1764731ded8285b1c5da8eca28840c51e506b16250fc2922a7e2a1d.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/7dd1527ec1764731ded8285b1c5da8eca28840c51e506b16250fc2922a7e2a1d.jpg)

![9a55a2ba1928659297032b57b20bd16874c49af98b45b7726331aa6d732dc803.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/9a55a2ba1928659297032b57b20bd16874c49af98b45b7726331aa6d732dc803.jpg)

![ace40fd5241e5d208ec1aa970ff978d05c820d70f10dfe24ef8aa1e63c876c2b.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/ace40fd5241e5d208ec1aa970ff978d05c820d70f10dfe24ef8aa1e63c876c2b.jpg)

![c9d2a1172f7573f50799f355d9c447af2d2af40568bfe5238a86d6d1e59d42ab.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/images/c9d2a1172f7573f50799f355d9c447af2d2af40568bfe5238a86d6d1e59d42ab.jpg)

### Tables

![deefcc040b93259d4ddcdd9ee669d9cca3b495310ddc47e68ccc5fad880037e2.jpg](../icml_results/103_Rare%20Event%20Analysis%20of%20Large%20Language%20Models/tables/deefcc040b93259d4ddcdd9ee669d9cca3b495310ddc47e68ccc5fad880037e2.jpg)

<a id="paper-0103"></a>

## From Distribution to Geometry: Stable Graph Generalization via Invariant Barycenters


### Images

![121c6e7b680325ab1d63fea74d17d178405ac005140a2b434b3d548b119a4de0.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/121c6e7b680325ab1d63fea74d17d178405ac005140a2b434b3d548b119a4de0.jpg)

![12854ce204865e9f6df9df8e951b3bcb5c65ebbf1e4defd326ef5855d3485367.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/12854ce204865e9f6df9df8e951b3bcb5c65ebbf1e4defd326ef5855d3485367.jpg)

![1d9d14f46d3904bdf697ee200c74dc81e9c37c5d10e2bd8407cd6c8505432e3d.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/1d9d14f46d3904bdf697ee200c74dc81e9c37c5d10e2bd8407cd6c8505432e3d.jpg)

![3bddd5aaf896da6c913afee885a94c46da664928040f32af4690803e2f00a3c3.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/3bddd5aaf896da6c913afee885a94c46da664928040f32af4690803e2f00a3c3.jpg)

![45ad5b2e14c545865b5223215aeb3d1f90b183852043a6b32e2f05f3e58428a3.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/45ad5b2e14c545865b5223215aeb3d1f90b183852043a6b32e2f05f3e58428a3.jpg)

![496d2da4343d56d9b275c7cc5c6a3ff15d8de7d8a595fac869cdfcbcb016b2de.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/496d2da4343d56d9b275c7cc5c6a3ff15d8de7d8a595fac869cdfcbcb016b2de.jpg)

![7800d4bb6267dd9430a4996bfb55e7e6ed5dc866bab7f7423a37e627d8b1fb01.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/7800d4bb6267dd9430a4996bfb55e7e6ed5dc866bab7f7423a37e627d8b1fb01.jpg)

![8547e922a6282c562ec69088bc4fc18da83e1c5ac958a56b1ae9277c4a7541b8.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/8547e922a6282c562ec69088bc4fc18da83e1c5ac958a56b1ae9277c4a7541b8.jpg)

![aadec6610ba48b79e4d8f4d83e736b3817548c6b7b383724bc9acbace2d4e5c9.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/aadec6610ba48b79e4d8f4d83e736b3817548c6b7b383724bc9acbace2d4e5c9.jpg)

![c513bab1a643c32f5106cbacc8b802b5c022a7f19c8175456b81cb9098cd2035.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/c513bab1a643c32f5106cbacc8b802b5c022a7f19c8175456b81cb9098cd2035.jpg)

![e0fe39d805b65de8cb54871a928a42fcfa1aa12dc6a61f08b50f489bb2413fb9.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/images/e0fe39d805b65de8cb54871a928a42fcfa1aa12dc6a61f08b50f489bb2413fb9.jpg)

### Tables

![0eca9c55799300b7833fe37a977dc46f001aa774aa981030e00780307c1b705c.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/tables/0eca9c55799300b7833fe37a977dc46f001aa774aa981030e00780307c1b705c.jpg)

![4cec6ac5243487b28a480ea0b49c89a17002e2abca8454d0f86444daa725ffd7.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/tables/4cec6ac5243487b28a480ea0b49c89a17002e2abca8454d0f86444daa725ffd7.jpg)

![d923265be6ec6f6374c6ae788dd225648c761a98706c10bfbc09082c7522eead.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/tables/d923265be6ec6f6374c6ae788dd225648c761a98706c10bfbc09082c7522eead.jpg)

![de9877fe730fa1c505ad52e8713bdc1d228284fe665866ff2e0ac9b73652a70a.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/tables/de9877fe730fa1c505ad52e8713bdc1d228284fe665866ff2e0ac9b73652a70a.jpg)

![eca55f9b4b838b97736c89b5c6f3792b533b2ecfa4a961eadde2938a46825f47.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/tables/eca55f9b4b838b97736c89b5c6f3792b533b2ecfa4a961eadde2938a46825f47.jpg)

![ef1fe9f98412f8374e68f42ac46fed778e6dd12d2e977badeb75316d1166f518.jpg](../icml_results/104_From%20Distribution%20to%20Geometry_%20Stable%20Graph%20Generalization%20via%20Invariant%20Barycenters/tables/ef1fe9f98412f8374e68f42ac46fed778e6dd12d2e977badeb75316d1166f518.jpg)

<a id="paper-0104"></a>

## On the Difficulty of Learning a Meta-network for Training Data Selection


### Images

![44fc019da50516b6ca56ee24d4e961fc9009c28feece16e21661e13f646ecdfd.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/images/44fc019da50516b6ca56ee24d4e961fc9009c28feece16e21661e13f646ecdfd.jpg)

![7e873441339c75ee91c2fe248cc3138e6e66b5a2323212f0f821ce1889221cea.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/images/7e873441339c75ee91c2fe248cc3138e6e66b5a2323212f0f821ce1889221cea.jpg)

![ad7f053b30373ed7e6af8cbbb54a1b60f0260071c62c463b004cce06b75aef7a.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/images/ad7f053b30373ed7e6af8cbbb54a1b60f0260071c62c463b004cce06b75aef7a.jpg)

![d4d7f4f3c932636f710865ebff322b70123f014c71637484d3c2a75c79beecdf.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/images/d4d7f4f3c932636f710865ebff322b70123f014c71637484d3c2a75c79beecdf.jpg)

### Tables

![29afda53f8028b5b11da60803d80ac3983d60c1a6081832186541242d00984ab.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/tables/29afda53f8028b5b11da60803d80ac3983d60c1a6081832186541242d00984ab.jpg)

![3d54e322e4e2064751ef6dde9f91bf1ee4ed1bc32054c559faaff1016a042516.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/tables/3d54e322e4e2064751ef6dde9f91bf1ee4ed1bc32054c559faaff1016a042516.jpg)

![6e0f99661e889838fdbab88dcd4d8907aa877ea644c6e533757beedeb2fd3515.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/tables/6e0f99661e889838fdbab88dcd4d8907aa877ea644c6e533757beedeb2fd3515.jpg)

![70d04134071ccad9f84427cecbe3eb16d5dfcd3b44f9f53ab4e418557daf6752.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/tables/70d04134071ccad9f84427cecbe3eb16d5dfcd3b44f9f53ab4e418557daf6752.jpg)

![d77ded261089f6b24d4185483a37a72cec38070d083ac5b8489e2991cb54178d.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/tables/d77ded261089f6b24d4185483a37a72cec38070d083ac5b8489e2991cb54178d.jpg)

![fe88705a1e99dad1eb08e5be25d0401157923f0a79526a1d5c7590c1b03ecda9.jpg](../icml_results/105_On%20the%20Difficulty%20of%20Learning%20a%20Meta-network%20for%20Training%20Data%20Selection/tables/fe88705a1e99dad1eb08e5be25d0401157923f0a79526a1d5c7590c1b03ecda9.jpg)

<a id="paper-0105"></a>

## Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling


### Images

![0e1e675c75d2a97127739d55f378b403730daff6319126750019b07e745f9747.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/0e1e675c75d2a97127739d55f378b403730daff6319126750019b07e745f9747.jpg)

![1d0555763f0dc09fb22e99dc7f7a91034b3313dec3f8fcb3d934f46c0abda901.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/1d0555763f0dc09fb22e99dc7f7a91034b3313dec3f8fcb3d934f46c0abda901.jpg)

![4f4fbdf10a831290fea248b636ac7f23d095b00857fddcb575d793934ed5e09f.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/4f4fbdf10a831290fea248b636ac7f23d095b00857fddcb575d793934ed5e09f.jpg)

![543b273595ba4084ce7441d5cd2097201d63d29c8bf28420ea5dd9501a7c9e16.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/543b273595ba4084ce7441d5cd2097201d63d29c8bf28420ea5dd9501a7c9e16.jpg)

![82c63b36ca392249d84a0c6fd5c9e40d6ba5675c99dc7167d326711f3441147b.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/82c63b36ca392249d84a0c6fd5c9e40d6ba5675c99dc7167d326711f3441147b.jpg)

![9086d2e3c5bb481df1e7e15f5d36ed6d359421d26b8e4fc5203d468bdda50cdf.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/9086d2e3c5bb481df1e7e15f5d36ed6d359421d26b8e4fc5203d468bdda50cdf.jpg)

![95b1eb84483c565daf98ce69d219666279c9478b1d0a5b88650bcbe8ab217e03.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/95b1eb84483c565daf98ce69d219666279c9478b1d0a5b88650bcbe8ab217e03.jpg)

![973235e0674aca029a89eb71deca5db0b8650cec7a1c7df9577a18cb26752f8b.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/973235e0674aca029a89eb71deca5db0b8650cec7a1c7df9577a18cb26752f8b.jpg)

![abb6a50633a7ba7a098152590f0dacf8d0b84484afae940c40551d4ee719f438.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/abb6a50633a7ba7a098152590f0dacf8d0b84484afae940c40551d4ee719f438.jpg)

![e5679b7f4b2ba5c89b782c48d01fcbb0449d238957f8e784f7bc23c6606387ff.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/images/e5679b7f4b2ba5c89b782c48d01fcbb0449d238957f8e784f7bc23c6606387ff.jpg)

### Tables

![2147ad9a7809db6c89f35e0e9cbc7a28a0a1cec676a68813beef76325fee0edc.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/2147ad9a7809db6c89f35e0e9cbc7a28a0a1cec676a68813beef76325fee0edc.jpg)

![40f169fb41952596c3483ecfe44d0ca4d5a5c7d44960803cce03f9ac90984119.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/40f169fb41952596c3483ecfe44d0ca4d5a5c7d44960803cce03f9ac90984119.jpg)

![5ef7aa1a6c46ef23b612d9e73f5004ae5abf9fa10ec340a7a1cabe3bf0c060ae.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/5ef7aa1a6c46ef23b612d9e73f5004ae5abf9fa10ec340a7a1cabe3bf0c060ae.jpg)

![61066f89022df0dbec43e40b08381c17d4320aa8c61952d675a7bf68d359e41f.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/61066f89022df0dbec43e40b08381c17d4320aa8c61952d675a7bf68d359e41f.jpg)

![6c95751f33cf958b0e798ce76879053d6ea1fa85d787974e4f68484753ffb5e9.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/6c95751f33cf958b0e798ce76879053d6ea1fa85d787974e4f68484753ffb5e9.jpg)

![b17f6e0b90ca4907544d5efbc14b149562f924de3d4c5996b9b97e4ad3c176d9.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/b17f6e0b90ca4907544d5efbc14b149562f924de3d4c5996b9b97e4ad3c176d9.jpg)

![b7e6e1cacfbb70da5b0d4f9a4774fd9b0e1f79d4680ef494436de45d9ea5d808.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/b7e6e1cacfbb70da5b0d4f9a4774fd9b0e1f79d4680ef494436de45d9ea5d808.jpg)

![c01798431976bd6b9c7a91e6b1529ff2a38cf539f4a855ab1a4dd7d0207f6c6d.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/c01798431976bd6b9c7a91e6b1529ff2a38cf539f4a855ab1a4dd7d0207f6c6d.jpg)

![d806c37aefb5d135b5ef0d63d38aa8bf27bfa62ed95a2289d391043c5b258637.jpg](../icml_results/106_Mitigating%20Reward%20Hacking%20in%20RLHF%20via%20Bayesian%20Non-negative%20Reward%20Modeling/tables/d806c37aefb5d135b5ef0d63d38aa8bf27bfa62ed95a2289d391043c5b258637.jpg)

<a id="paper-0106"></a>

## Mixtures Closest To A Given Measure: A Semidefinite Programming Approach


### Images

![9e447f8c620553c0b4416d0ec9d8981ae4f7997d7751136b89ba7c8a09d1eed6.jpg](../icml_results/107_Mixtures%20Closest%20To%20A%20Given%20Measure_%20A%20Semidefinite%20Programming%20Approach/images/9e447f8c620553c0b4416d0ec9d8981ae4f7997d7751136b89ba7c8a09d1eed6.jpg)

![b0302e7fe0aead56b267be9f871a894a46503cd34b1ae365ed9a00a7fd7a72b7.jpg](../icml_results/107_Mixtures%20Closest%20To%20A%20Given%20Measure_%20A%20Semidefinite%20Programming%20Approach/images/b0302e7fe0aead56b267be9f871a894a46503cd34b1ae365ed9a00a7fd7a72b7.jpg)

![bda45ef21bfffa5934da09aea5b10bab1dd0b739e7dbcb338b5c477c3bb76c61.jpg](../icml_results/107_Mixtures%20Closest%20To%20A%20Given%20Measure_%20A%20Semidefinite%20Programming%20Approach/images/bda45ef21bfffa5934da09aea5b10bab1dd0b739e7dbcb338b5c477c3bb76c61.jpg)

![be654a93440054079cb8e311690c5f5afa9ceae61a5b2e05aa35d7896f4e866b.jpg](../icml_results/107_Mixtures%20Closest%20To%20A%20Given%20Measure_%20A%20Semidefinite%20Programming%20Approach/images/be654a93440054079cb8e311690c5f5afa9ceae61a5b2e05aa35d7896f4e866b.jpg)

### Tables

![30bb539dfe962d35444de8622b575dc6f7a9913f70632955847384e06875acdd.jpg](../icml_results/107_Mixtures%20Closest%20To%20A%20Given%20Measure_%20A%20Semidefinite%20Programming%20Approach/tables/30bb539dfe962d35444de8622b575dc6f7a9913f70632955847384e06875acdd.jpg)

<a id="paper-0107"></a>

## HOBIT: Hardness Optimized Batch Sampling for InfoNCE Training


### Images

![29db948cea05e7283abe0088d51a7bd8e2f1147a425186d35905fd4756b01f51.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/images/29db948cea05e7283abe0088d51a7bd8e2f1147a425186d35905fd4756b01f51.jpg)

![3617441569f030d094b9cde9e27fceec6d55015a5b86107aa3a0eb577d2e3e96.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/images/3617441569f030d094b9cde9e27fceec6d55015a5b86107aa3a0eb577d2e3e96.jpg)

![369ab8f1a3e29fbb4faef66371ee26271a5f5dc44db6badb9adaeac8f580dd92.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/images/369ab8f1a3e29fbb4faef66371ee26271a5f5dc44db6badb9adaeac8f580dd92.jpg)

![596cbb00a7bbf0b5220b67e270f0d8efb110d35c3cbbddefc4c12737a64ac057.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/images/596cbb00a7bbf0b5220b67e270f0d8efb110d35c3cbbddefc4c12737a64ac057.jpg)

![cf751df66aea07557506fb02b730414c209d64257afdfffd7e55edd10814b36e.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/images/cf751df66aea07557506fb02b730414c209d64257afdfffd7e55edd10814b36e.jpg)

### Tables

![020a275997dd6aa1969545a1ecfa23990d4df70f572a44bb4c5444125c684ec2.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/020a275997dd6aa1969545a1ecfa23990d4df70f572a44bb4c5444125c684ec2.jpg)

![0a65f225d11695b9decd07fd6131aadb21f8587f4072300c24d7df5e29be078e.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/0a65f225d11695b9decd07fd6131aadb21f8587f4072300c24d7df5e29be078e.jpg)

![40acb5d1e2753d6278396637de225c7f3a6d339502020051edb613a9147ca159.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/40acb5d1e2753d6278396637de225c7f3a6d339502020051edb613a9147ca159.jpg)

![48bc3ba4fcba8c2fa81dd1128ccdf54336dce2007f1d9e4637afa7692498f752.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/48bc3ba4fcba8c2fa81dd1128ccdf54336dce2007f1d9e4637afa7692498f752.jpg)

![4d13f3e4b7e17d7f9995d7bf1bbeaf6239dfcc38bc23c422be378f964cf066ba.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/4d13f3e4b7e17d7f9995d7bf1bbeaf6239dfcc38bc23c422be378f964cf066ba.jpg)

![5ebbb1e313cb12b2c2f6341890aff646c7408b2d15c9e529e8b3aaa95d8c506a.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/5ebbb1e313cb12b2c2f6341890aff646c7408b2d15c9e529e8b3aaa95d8c506a.jpg)

![ad8f23214ed25b521f9589b86b6503a0d0efe1abef7bdc446bfe0fda101ce91f.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/ad8f23214ed25b521f9589b86b6503a0d0efe1abef7bdc446bfe0fda101ce91f.jpg)

![afcbb6b0199a3f2a1fa0a3a0ec7cde6b89af06edc7712c1adc70b91381ec29e1.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/afcbb6b0199a3f2a1fa0a3a0ec7cde6b89af06edc7712c1adc70b91381ec29e1.jpg)

![c506905babe16e1e895b7591ea3efd5f850ce1e8594e03fe806dc159eded57b8.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/c506905babe16e1e895b7591ea3efd5f850ce1e8594e03fe806dc159eded57b8.jpg)

![d2c2d64247d64ffabb9f03615862bfc0bb9c71f5c582a77e2399fe8ea5c60965.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/d2c2d64247d64ffabb9f03615862bfc0bb9c71f5c582a77e2399fe8ea5c60965.jpg)

![d72c4e73dcb896889e9d2ab9d8c83ff17f70697ed90661a7ee1d16b5f8c9cf69.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/d72c4e73dcb896889e9d2ab9d8c83ff17f70697ed90661a7ee1d16b5f8c9cf69.jpg)

![ded9da60812dfa4bbdeb41734f7e29dfc78d039b7f1f57d363de7e1f190163e7.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/ded9da60812dfa4bbdeb41734f7e29dfc78d039b7f1f57d363de7e1f190163e7.jpg)

![e21fabecacfdd60b00c53dfffc809403aecda1df434a6d86be93129c504f44f5.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/e21fabecacfdd60b00c53dfffc809403aecda1df434a6d86be93129c504f44f5.jpg)

![fcb9795ad60fa8a69a093997e432c0a25b06572768f3b3c008b7bf41b33938e0.jpg](../icml_results/108_HOBIT_%20Hardness%20Optimized%20Batch%20Sampling%20for%20InfoNCE%20Training/tables/fcb9795ad60fa8a69a093997e432c0a25b06572768f3b3c008b7bf41b33938e0.jpg)

<a id="paper-0108"></a>

## FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs


### Images

![022265e08d0e6fe36ba0148783d90df9f7393132462cc91487552bfffabfbf74.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/022265e08d0e6fe36ba0148783d90df9f7393132462cc91487552bfffabfbf74.jpg)

![02c4349f3458dc86b52675f9eb5e4a5a308164e814df55fb569dc60e3a60546b.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/02c4349f3458dc86b52675f9eb5e4a5a308164e814df55fb569dc60e3a60546b.jpg)

![05d59e3a7588d82e88796a1f6dc8a9b3ed51c0efd91be1f78c6c20a733ab2935.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/05d59e3a7588d82e88796a1f6dc8a9b3ed51c0efd91be1f78c6c20a733ab2935.jpg)

![07895eacb643614f76256b10ff8cfdc2519ad413447a4e3ab3da04b599625b65.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/07895eacb643614f76256b10ff8cfdc2519ad413447a4e3ab3da04b599625b65.jpg)

![1260f91a0b79712074a019537d27bda9f867e620a3631d491f4b9474558a2bf3.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/1260f91a0b79712074a019537d27bda9f867e620a3631d491f4b9474558a2bf3.jpg)

![13b8937e3c3fede3a0fff45044218eb02582f8236373950d10e116b1ec916b3d.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/13b8937e3c3fede3a0fff45044218eb02582f8236373950d10e116b1ec916b3d.jpg)

![1856381fe836dafbd75084b99fc6c781a339ef4e825ad8847fe21e15ed19fa77.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/1856381fe836dafbd75084b99fc6c781a339ef4e825ad8847fe21e15ed19fa77.jpg)

![1d3ab1eaccf225d795d582c988ca3d7cad043f600504a93533ce7253d2f4c653.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/1d3ab1eaccf225d795d582c988ca3d7cad043f600504a93533ce7253d2f4c653.jpg)

![1d9c7d55efa311486c384ba60e1bc9250245ff50d8489a4bb30b93903f49675d.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/1d9c7d55efa311486c384ba60e1bc9250245ff50d8489a4bb30b93903f49675d.jpg)

![274e5c02d65b7b0b9a4dc244dfdfb2bca7242906689c1ab1eab7bfca7720f4cc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/274e5c02d65b7b0b9a4dc244dfdfb2bca7242906689c1ab1eab7bfca7720f4cc.jpg)

![279d3096aace076f7091d4fdc1d76353223b98219f59e424c598160221e2d7ba.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/279d3096aace076f7091d4fdc1d76353223b98219f59e424c598160221e2d7ba.jpg)

![2a394e1b3798baabc2fa46c0376636c96945855c425264c567555e01e48a9bd8.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/2a394e1b3798baabc2fa46c0376636c96945855c425264c567555e01e48a9bd8.jpg)

![2a7da8c322d5464707e4a7e996f68ccd3e1e20e4ab56fe192b256173005334df.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/2a7da8c322d5464707e4a7e996f68ccd3e1e20e4ab56fe192b256173005334df.jpg)

![2c67a2c6397bdf2c927b065099a9c7d0ce94cd9971ff466ccfe222ec3ddabddc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/2c67a2c6397bdf2c927b065099a9c7d0ce94cd9971ff466ccfe222ec3ddabddc.jpg)

![2faa379ee775a58ce15a5b5cd568ad54d7c3bcb6a4c41fec2019b75e1c8d65bc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/2faa379ee775a58ce15a5b5cd568ad54d7c3bcb6a4c41fec2019b75e1c8d65bc.jpg)

![304e2dc84e3432f7e9595743e6f056a2aeab48050517d44ea0ca87fa9681a05e.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/304e2dc84e3432f7e9595743e6f056a2aeab48050517d44ea0ca87fa9681a05e.jpg)

![32bc17d94ca011bb716b1cd0882a002e34090789cfd480051ffd31f69c722003.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/32bc17d94ca011bb716b1cd0882a002e34090789cfd480051ffd31f69c722003.jpg)

![36c66b84960bfc4f78c34139b28901ab73c5f729939972a74f3cd9eb9d72063c.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/36c66b84960bfc4f78c34139b28901ab73c5f729939972a74f3cd9eb9d72063c.jpg)

![39262019a92253624a2c6b73531bc508da5d79707ea1bc2a2194a69a4a13aa84.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/39262019a92253624a2c6b73531bc508da5d79707ea1bc2a2194a69a4a13aa84.jpg)

![3b731d7c60529e94035240ceb3f56b9e6e64abba7f0d55cc42f066df3f9f6c25.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/3b731d7c60529e94035240ceb3f56b9e6e64abba7f0d55cc42f066df3f9f6c25.jpg)

![3cb2dbeff44b7e22f6149b64762f2e3796f56442ccd4fdd6a4b0306494c254b4.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/3cb2dbeff44b7e22f6149b64762f2e3796f56442ccd4fdd6a4b0306494c254b4.jpg)

![3cbfef194c3c126560a48f92f0dc797cef29db5ccbd59cfdd7f4961d4d879c61.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/3cbfef194c3c126560a48f92f0dc797cef29db5ccbd59cfdd7f4961d4d879c61.jpg)

![3d47d8c2629046c6c6b2dd16b81c002adf2229d9ae7322f75db77877675349f9.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/3d47d8c2629046c6c6b2dd16b81c002adf2229d9ae7322f75db77877675349f9.jpg)

![3e105ad479be196634c3350b51b1228720fda307ba42b373bd4585471019e97f.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/3e105ad479be196634c3350b51b1228720fda307ba42b373bd4585471019e97f.jpg)

![3e69d05a851e9c38063493f3722fa2e71a054a92a15aa77df178bbc1c1bd27d2.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/3e69d05a851e9c38063493f3722fa2e71a054a92a15aa77df178bbc1c1bd27d2.jpg)

![40320de4fa4e42bb586d63d3ddb8caefc835d67f076394a803f0a270e4e13078.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/40320de4fa4e42bb586d63d3ddb8caefc835d67f076394a803f0a270e4e13078.jpg)

![42d672d64d3b654300f65449b5009a26f551ce45750ee273869536bb20855135.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/42d672d64d3b654300f65449b5009a26f551ce45750ee273869536bb20855135.jpg)

![48ecdc47dc35ca8aea50e4fb2225d6cb873740b57cdb7e007030f4cdad3f4c6d.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/48ecdc47dc35ca8aea50e4fb2225d6cb873740b57cdb7e007030f4cdad3f4c6d.jpg)

![490af34cc1df3cd1e0b774a5abf58f22a538cb6ce9ca19a0f58fd28442f40efa.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/490af34cc1df3cd1e0b774a5abf58f22a538cb6ce9ca19a0f58fd28442f40efa.jpg)

![4a7bde13882c59df8882f79005faab26933e6a017b87bd8b005890fb06ed4233.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/4a7bde13882c59df8882f79005faab26933e6a017b87bd8b005890fb06ed4233.jpg)

![4b259980d58364a3160bafbbdac414cb72a3311dde3722e4aab489ab7d9e8f2a.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/4b259980d58364a3160bafbbdac414cb72a3311dde3722e4aab489ab7d9e8f2a.jpg)

![4c4994698eb5e5f9f542264c8d40224c02846d9a382299b2f8280e92e693d581.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/4c4994698eb5e5f9f542264c8d40224c02846d9a382299b2f8280e92e693d581.jpg)

![4d87819e9e61ac3723e36fc6d7bf040bf9525a804cc9617683be9ef7435ca0b1.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/4d87819e9e61ac3723e36fc6d7bf040bf9525a804cc9617683be9ef7435ca0b1.jpg)

![4da51d41134fdaaefceec410f67a08011e3820fe5ca2cd13ec658f7faab115ab.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/4da51d41134fdaaefceec410f67a08011e3820fe5ca2cd13ec658f7faab115ab.jpg)

![4fc186f375391fe4cc30e688f8594bef1cb60292ec24f1f268139c403590f1af.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/4fc186f375391fe4cc30e688f8594bef1cb60292ec24f1f268139c403590f1af.jpg)

![5036cb3febd160a346ed2b215e577b1769302b0c7ee61d4a369bc4f0c142aba3.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/5036cb3febd160a346ed2b215e577b1769302b0c7ee61d4a369bc4f0c142aba3.jpg)

![52c994322bbc58c51fa2f58d81e38f797044bf6eeb219bab22f05d80811ee0eb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/52c994322bbc58c51fa2f58d81e38f797044bf6eeb219bab22f05d80811ee0eb.jpg)

![53a18e482e0cdbe1fa31c60664dd56e5774ccdc7b30b11cd09b61138997251dc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/53a18e482e0cdbe1fa31c60664dd56e5774ccdc7b30b11cd09b61138997251dc.jpg)

![556d171bc01184521b5d2fee3a130d90d2ae9575e7be7b652185ddd37b7b0391.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/556d171bc01184521b5d2fee3a130d90d2ae9575e7be7b652185ddd37b7b0391.jpg)

![55f4e6f67a32b008f4ba2805a66ef50c1da451b341db1ce962704a5693e29127.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/55f4e6f67a32b008f4ba2805a66ef50c1da451b341db1ce962704a5693e29127.jpg)

![5aa71965cdba2dcdfe7512b9b92ff98c65fc6debbc2a32b0b66268ed982127e2.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/5aa71965cdba2dcdfe7512b9b92ff98c65fc6debbc2a32b0b66268ed982127e2.jpg)

![5ac379df190f18f3f5db416287d9514d92150216940684ea01e31cb36cbbca42.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/5ac379df190f18f3f5db416287d9514d92150216940684ea01e31cb36cbbca42.jpg)

![601af28ef8b505a9046e73c905d1f3eebcbdbf1db8a535ed78bf91783e42b1a2.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/601af28ef8b505a9046e73c905d1f3eebcbdbf1db8a535ed78bf91783e42b1a2.jpg)

![605ec37a3cdfbb42722d73e0f33fe5c1ea7aa32b6d1482389833f2a1139382c1.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/605ec37a3cdfbb42722d73e0f33fe5c1ea7aa32b6d1482389833f2a1139382c1.jpg)

![626600418cdab8c7e955cba3ffc1130632e5d4a0cdb5212f1ff997096b4369f4.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/626600418cdab8c7e955cba3ffc1130632e5d4a0cdb5212f1ff997096b4369f4.jpg)

![675e27c021d417fac341f56e2b378614b4b34486a3c7af2b04f2f1df3868adbe.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/675e27c021d417fac341f56e2b378614b4b34486a3c7af2b04f2f1df3868adbe.jpg)

![677039d738c8b49a20ac8ddfd42c5e1843329b7f491718b60e4781f8cb17fc46.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/677039d738c8b49a20ac8ddfd42c5e1843329b7f491718b60e4781f8cb17fc46.jpg)

![67f02ddb7b7bee16780df90d56907ed50c7060f943a7e79b1b99d18ec3cac5eb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/67f02ddb7b7bee16780df90d56907ed50c7060f943a7e79b1b99d18ec3cac5eb.jpg)

![688f693e75c7a2d5279d27327592474a11fae32c4a75f03d259ebfd65344ed3d.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/688f693e75c7a2d5279d27327592474a11fae32c4a75f03d259ebfd65344ed3d.jpg)

![69c1f0a7b238ab5d5febff92318a5436a0b19f030fada61c2ae95e4dbd0fb8c3.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/69c1f0a7b238ab5d5febff92318a5436a0b19f030fada61c2ae95e4dbd0fb8c3.jpg)

![6b88d29bce0144197052794683248f6b9d0d686ef70a15ed9ca6df7afaf84e06.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/6b88d29bce0144197052794683248f6b9d0d686ef70a15ed9ca6df7afaf84e06.jpg)

![6c6f6c46c837e0dcabb3953a3f136db1555bcc608a94081aca30b45728cc573b.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/6c6f6c46c837e0dcabb3953a3f136db1555bcc608a94081aca30b45728cc573b.jpg)

![70a4e67951517a3cf9016b728999cf097408681f2ed99d3fd518511e6c9c88cc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/70a4e67951517a3cf9016b728999cf097408681f2ed99d3fd518511e6c9c88cc.jpg)

![723e4013457579a1d16fe511d8cf979295cdfe8436f738cd06c91a876a362dfd.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/723e4013457579a1d16fe511d8cf979295cdfe8436f738cd06c91a876a362dfd.jpg)

![73305653c7fef9c194309c5015c4d5d85eb6950de2a7d5c0dee8f484d3c1d1f2.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/73305653c7fef9c194309c5015c4d5d85eb6950de2a7d5c0dee8f484d3c1d1f2.jpg)

![76758fdef93b211e112d367dc15bc4309aeed7b901b9fadc2608ed67b1067285.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/76758fdef93b211e112d367dc15bc4309aeed7b901b9fadc2608ed67b1067285.jpg)

![787ca40b1fbe5c5b563341040e99f5a370406af18ff84e6875944d443f9b48c4.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/787ca40b1fbe5c5b563341040e99f5a370406af18ff84e6875944d443f9b48c4.jpg)

![789a7133fe70079d6684a0604223ad4e3813bf2b2f9ab3f1c46ea5b12ab84c69.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/789a7133fe70079d6684a0604223ad4e3813bf2b2f9ab3f1c46ea5b12ab84c69.jpg)

![7998f54ffe65ed42f7fbd318ca9e84a2ee3f02f26415b6431d3df606b7a2e783.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/7998f54ffe65ed42f7fbd318ca9e84a2ee3f02f26415b6431d3df606b7a2e783.jpg)

![79a672c5d80402d2a5655cfeb0cb236f24066e104e03a43281fa16ee922e9706.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/79a672c5d80402d2a5655cfeb0cb236f24066e104e03a43281fa16ee922e9706.jpg)

![7ae60b198cfdbced905ab04bc7505925eac9b9fb650e8fabbb101f06408b9930.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/7ae60b198cfdbced905ab04bc7505925eac9b9fb650e8fabbb101f06408b9930.jpg)

![7ca1f7f8d93408fa22cfe871cb621325fdb7bb9607f776feac60f391f2b428f6.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/7ca1f7f8d93408fa22cfe871cb621325fdb7bb9607f776feac60f391f2b428f6.jpg)

![7d2c06df03065dc7831a38bd7e8a9ec97fb6e4b14f8cbed7cd676cd45450a1bd.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/7d2c06df03065dc7831a38bd7e8a9ec97fb6e4b14f8cbed7cd676cd45450a1bd.jpg)

![7d7bc71fbc96f0470aa529f7db47c984cd6f8c894673d8dd7a81cd64f3c9ea30.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/7d7bc71fbc96f0470aa529f7db47c984cd6f8c894673d8dd7a81cd64f3c9ea30.jpg)

![7da7c117d17b7532308800e86d93c2fd03b5bf7789a6b5b3c221dca372bf58bb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/7da7c117d17b7532308800e86d93c2fd03b5bf7789a6b5b3c221dca372bf58bb.jpg)

![8088baa3f4a336ce97f769d83569a813e94834173203026a59cb22cfadfb7508.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8088baa3f4a336ce97f769d83569a813e94834173203026a59cb22cfadfb7508.jpg)

![811378a31087589f2310f08d567142b5e2a18bbcfd5035fd7760a9c3a9e06da3.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/811378a31087589f2310f08d567142b5e2a18bbcfd5035fd7760a9c3a9e06da3.jpg)

![873dc940f2532dd5f02229f455a922e7ded0300c0c0230ac09866458436b1c8d.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/873dc940f2532dd5f02229f455a922e7ded0300c0c0230ac09866458436b1c8d.jpg)

![879adb3eb590f6d94bf1eb357794c54d4efd5b987eb57646135ba512b5baa940.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/879adb3eb590f6d94bf1eb357794c54d4efd5b987eb57646135ba512b5baa940.jpg)

![888d8860669964854695cdab3ebf7ad25727c395d87967ab7e1482d2b45da436.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/888d8860669964854695cdab3ebf7ad25727c395d87967ab7e1482d2b45da436.jpg)

![892fe826b124d67e858dfd83b9d049591d50c6a44c6b324a00b6597a7c0a3fe8.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/892fe826b124d67e858dfd83b9d049591d50c6a44c6b324a00b6597a7c0a3fe8.jpg)

![89fb0370eee1e653768d12007ab02b3c61fb62d4ff38cd6aee1f25493eb98c52.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/89fb0370eee1e653768d12007ab02b3c61fb62d4ff38cd6aee1f25493eb98c52.jpg)

![8a11c838e81784a2a72e82338852e8a1e3a5432f186b03ae0d2d8317f9afe4d5.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8a11c838e81784a2a72e82338852e8a1e3a5432f186b03ae0d2d8317f9afe4d5.jpg)

![8bc44ab5c2d543c383c97f25f851f732144400a20ed2a445e982562880aae393.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8bc44ab5c2d543c383c97f25f851f732144400a20ed2a445e982562880aae393.jpg)

![8d55b731b141ddb592dbb7463193d57b4803953a6f259beb689992af5a769565.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8d55b731b141ddb592dbb7463193d57b4803953a6f259beb689992af5a769565.jpg)

![8e639d96ceaf5ad276a25cf51bb8bb0651d908107f2c40bfed89e67370bd78d5.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8e639d96ceaf5ad276a25cf51bb8bb0651d908107f2c40bfed89e67370bd78d5.jpg)

![8e656e3e2c7ec80345b5b365b1ff38f9704eeabb0b7131314e208e2a9fdb5e09.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8e656e3e2c7ec80345b5b365b1ff38f9704eeabb0b7131314e208e2a9fdb5e09.jpg)

![8ec4dc2cb9138bb79a1aa81f747c10da1dd80c9e1b293b2a1e69b66e56787fe4.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/8ec4dc2cb9138bb79a1aa81f747c10da1dd80c9e1b293b2a1e69b66e56787fe4.jpg)

![90f0eb28856f2b9616ba520198540d87eae11e7e1ed7936f93d0c5d3d803016d.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/90f0eb28856f2b9616ba520198540d87eae11e7e1ed7936f93d0c5d3d803016d.jpg)

![9328ae269006502770ac247f68c7bd39edbb4b6f3c6b3f1341857210fa12d6fb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/9328ae269006502770ac247f68c7bd39edbb4b6f3c6b3f1341857210fa12d6fb.jpg)

![94adc7da14db8247198b57d4019d195b1c024008fc72ee715be51e70757907bc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/94adc7da14db8247198b57d4019d195b1c024008fc72ee715be51e70757907bc.jpg)

![9516ceaaf78a297682256a55e9bd55ba323f01294ec9127ec6cbab17b30d2837.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/9516ceaaf78a297682256a55e9bd55ba323f01294ec9127ec6cbab17b30d2837.jpg)

![9893e5624627132b6ff784e75fa0c7b7f988248c44fb46bc6d8a0840f1f9019a.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/9893e5624627132b6ff784e75fa0c7b7f988248c44fb46bc6d8a0840f1f9019a.jpg)

![989bdde15bee4b12233e7455a591db8b70092632dd4c4eea0512b649c15cc206.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/989bdde15bee4b12233e7455a591db8b70092632dd4c4eea0512b649c15cc206.jpg)

![9dc2ed0791078c4980a852e4619edf294df696735e80acf03cb0c7fd11fcd83a.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/9dc2ed0791078c4980a852e4619edf294df696735e80acf03cb0c7fd11fcd83a.jpg)

![9e10a17935eddfa39b8c400d28b88f998b1ef60d35b068751ef8b99109e55019.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/9e10a17935eddfa39b8c400d28b88f998b1ef60d35b068751ef8b99109e55019.jpg)

![a0b44372a9eea061c51956ffdac7852facb13a96661694a10b06f378ecef1c26.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a0b44372a9eea061c51956ffdac7852facb13a96661694a10b06f378ecef1c26.jpg)

![a0d10622173a60a077bf22e84d3b49e6ce2518543f5a11c9337ad11da3fddaf9.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a0d10622173a60a077bf22e84d3b49e6ce2518543f5a11c9337ad11da3fddaf9.jpg)

![a10283f1d22e89273cdd1a576d7f99229f9d85743e9bde0aad7cac179d4e62d3.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a10283f1d22e89273cdd1a576d7f99229f9d85743e9bde0aad7cac179d4e62d3.jpg)

![a44007207de725508e746c67fe4eea8baa7c3dbde92540cac81a23349e826e2f.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a44007207de725508e746c67fe4eea8baa7c3dbde92540cac81a23349e826e2f.jpg)

![a5196b60e2bc7b5386116b8ac782a778003eccd0c797f691dbaf8af98013bd74.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a5196b60e2bc7b5386116b8ac782a778003eccd0c797f691dbaf8af98013bd74.jpg)

![a51cb7241b70836fc26ef7bd69ce1e62c87caaf8870abfba30a946e03d603e7e.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a51cb7241b70836fc26ef7bd69ce1e62c87caaf8870abfba30a946e03d603e7e.jpg)

![a52f5560a25d52d72a21c2c4b4cbf69d70342db7cc03b1312324f6c163d691dc.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a52f5560a25d52d72a21c2c4b4cbf69d70342db7cc03b1312324f6c163d691dc.jpg)

![a6c7664cf12ccc21d9d1bf68b76468d11cde5bae6858fa5aea4d3f1b0b123261.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/a6c7664cf12ccc21d9d1bf68b76468d11cde5bae6858fa5aea4d3f1b0b123261.jpg)

![ae3578fea7141020a013e356098681d3c513db73c4148b219444654261b7a961.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/ae3578fea7141020a013e356098681d3c513db73c4148b219444654261b7a961.jpg)

![b31160e657048b1d4ea6a6cfd9442276aa28c832c5cc600a7df1380961cab4b9.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/b31160e657048b1d4ea6a6cfd9442276aa28c832c5cc600a7df1380961cab4b9.jpg)

![b8d4ac9e1f912ef0035523c0b39a7e386cd4dd2164b3e646ffca44cb160838b1.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/b8d4ac9e1f912ef0035523c0b39a7e386cd4dd2164b3e646ffca44cb160838b1.jpg)

![b94e9468d8dc336f7ef503befcd01232ab1151fb2d5aaa463f4604cc870df5fb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/b94e9468d8dc336f7ef503befcd01232ab1151fb2d5aaa463f4604cc870df5fb.jpg)

![b9953303c49e0bdd3661e09a96e4e5655dcb402a34baa76e5592a49e130fe749.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/b9953303c49e0bdd3661e09a96e4e5655dcb402a34baa76e5592a49e130fe749.jpg)

![ba1aee716e16a1b1c3697a251d63f03ad140e353b19a54e67143519e917b8849.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/ba1aee716e16a1b1c3697a251d63f03ad140e353b19a54e67143519e917b8849.jpg)

![bdeaec15a36a8ae5dd2111517f0177e057905f588a8f43df8594647efb36a8bf.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/bdeaec15a36a8ae5dd2111517f0177e057905f588a8f43df8594647efb36a8bf.jpg)

![be1d55c98347b58b229bc66e7c8b4a7dedac6d65a648c2068c76f08e6a54c223.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/be1d55c98347b58b229bc66e7c8b4a7dedac6d65a648c2068c76f08e6a54c223.jpg)

![befd2e9bd4959ac22c2049c447ec99a75e5a3fdd5046ef886194a83cbe5a3e29.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/befd2e9bd4959ac22c2049c447ec99a75e5a3fdd5046ef886194a83cbe5a3e29.jpg)

![c01cd4eaaa9951712f32bc1e57ba245ecf053c823ab0a5cd0427d6dfdb88aa6c.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/c01cd4eaaa9951712f32bc1e57ba245ecf053c823ab0a5cd0427d6dfdb88aa6c.jpg)

![c1d71eadaee562281e66809050404b592dde9a776df8bfb4c288f679328ce8ed.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/c1d71eadaee562281e66809050404b592dde9a776df8bfb4c288f679328ce8ed.jpg)

![c2fb779cc2afd46e2df5165813e5f3c3906e2ce28ef43d3d471cf3fc2b0523fb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/c2fb779cc2afd46e2df5165813e5f3c3906e2ce28ef43d3d471cf3fc2b0523fb.jpg)

![c739b2585c767df5cb9b63bb73c8bc349b9a5d3f9e045ffd375ee9fb7e096fb9.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/c739b2585c767df5cb9b63bb73c8bc349b9a5d3f9e045ffd375ee9fb7e096fb9.jpg)

![c788cfc3e8455a90cedc1cba81810da9e637238e52e8464acb6a5970e7230280.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/c788cfc3e8455a90cedc1cba81810da9e637238e52e8464acb6a5970e7230280.jpg)

![cb216f1935869de156bc10d0c6f3fe89ebcc3eb9721086ad1ef0a481a9c2060f.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/cb216f1935869de156bc10d0c6f3fe89ebcc3eb9721086ad1ef0a481a9c2060f.jpg)

![cd2b2129073b25ae7717fee1c4733d524bab99b1197e383b13dce06e85efe1bb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/cd2b2129073b25ae7717fee1c4733d524bab99b1197e383b13dce06e85efe1bb.jpg)

![d108a42d6bc708eff97fc3b4a79ca3df3d5a0daa5d867f718dd7ff85ff24af49.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/d108a42d6bc708eff97fc3b4a79ca3df3d5a0daa5d867f718dd7ff85ff24af49.jpg)

![d387a77c14991cc0b89771deb36efd5a77fc381c068f70289ca633302df44503.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/d387a77c14991cc0b89771deb36efd5a77fc381c068f70289ca633302df44503.jpg)

![d3cecce643d954180ce05e2f245d96c5e05daa46f516b4d671f836dd466bf9d0.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/d3cecce643d954180ce05e2f245d96c5e05daa46f516b4d671f836dd466bf9d0.jpg)

![d4453f803c993aa4fbcf17927922af9a759c82a7262ac67f423070576e2498e7.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/d4453f803c993aa4fbcf17927922af9a759c82a7262ac67f423070576e2498e7.jpg)

![dc2d2e45652736a5716cd51f72eb73c87c50f74e28986d767e7349b64f2620c8.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/dc2d2e45652736a5716cd51f72eb73c87c50f74e28986d767e7349b64f2620c8.jpg)

![de6c350de411f7b4094be41d630bf4ab6ced4214cc16c25d5b3f48d9a3b5d26c.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/de6c350de411f7b4094be41d630bf4ab6ced4214cc16c25d5b3f48d9a3b5d26c.jpg)

![e06a413249cf97fa15090281f35b4767ec1f1d72bb5e1c86ee984a4cede1872a.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/e06a413249cf97fa15090281f35b4767ec1f1d72bb5e1c86ee984a4cede1872a.jpg)

![e23ec75e541dde20009ac4b6b528b4320a7a187d77a8323f8b2cc50c3a7787c4.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/e23ec75e541dde20009ac4b6b528b4320a7a187d77a8323f8b2cc50c3a7787c4.jpg)

![e2bbfdb0000dd88a096a2dbf9a090196221de030ebcea47d14fbe9ca2e9dfdce.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/e2bbfdb0000dd88a096a2dbf9a090196221de030ebcea47d14fbe9ca2e9dfdce.jpg)

![e30155acfb0876a3b6d94e2e56b0cc360a3f6b71eeb740e3cc2444b1ccc27227.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/e30155acfb0876a3b6d94e2e56b0cc360a3f6b71eeb740e3cc2444b1ccc27227.jpg)

![e3e1da038b78a6b4697f33f4e6a56c93048731ff5d877af0d4a29c557100fb66.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/e3e1da038b78a6b4697f33f4e6a56c93048731ff5d877af0d4a29c557100fb66.jpg)

![e748fa1e636dc0bc00d3db83f28749b2186dbc54a6cc49da0de3e1e3009ecea6.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/e748fa1e636dc0bc00d3db83f28749b2186dbc54a6cc49da0de3e1e3009ecea6.jpg)

![eae0af8e57e62fb3cfc64d48d247c53baad799b1b3c9e0b2f91f8113452fedfd.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/eae0af8e57e62fb3cfc64d48d247c53baad799b1b3c9e0b2f91f8113452fedfd.jpg)

![f115fbffd14ba3c6d9ae8be332c91707388d343f50da454a684e3968dded795c.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/f115fbffd14ba3c6d9ae8be332c91707388d343f50da454a684e3968dded795c.jpg)

![f39065df94a5df8c32aefafa512de24de776b559aeec92ba35eb749dc3b25050.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/f39065df94a5df8c32aefafa512de24de776b559aeec92ba35eb749dc3b25050.jpg)

![f6a351366be4c15c9e3528078727e3a8e5257b0937ea07c4221e0a2b48ab2737.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/f6a351366be4c15c9e3528078727e3a8e5257b0937ea07c4221e0a2b48ab2737.jpg)

![f6b840b51707fadf66694087721c88df20b0a4a7b0c8d3f248407eefe04f90cb.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/f6b840b51707fadf66694087721c88df20b0a4a7b0c8d3f248407eefe04f90cb.jpg)

![f852f784c718480d82f52916a81793dc4c895c0e60a50b551636be5dec0ede18.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/f852f784c718480d82f52916a81793dc4c895c0e60a50b551636be5dec0ede18.jpg)

![fa7e3989b02093a32e36cb1fcb46d995ce71486fa0e04d5c354a8cb1256bf9d5.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/fa7e3989b02093a32e36cb1fcb46d995ce71486fa0e04d5c354a8cb1256bf9d5.jpg)

![fbd2bb8a4e66784e413fd0945bebeb1787154c09f22c702d96f97a63332d9448.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/fbd2bb8a4e66784e413fd0945bebeb1787154c09f22c702d96f97a63332d9448.jpg)

![fe4f6f35cba81f1358f9832e967fef54f38c24fb7491671315ea44c23a0eb244.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/fe4f6f35cba81f1358f9832e967fef54f38c24fb7491671315ea44c23a0eb244.jpg)

![fe84532a81d403484af08aaece10c41a9bdd2c23d446c6e43d6038fc97a1fc76.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/fe84532a81d403484af08aaece10c41a9bdd2c23d446c6e43d6038fc97a1fc76.jpg)

![ff0f74a6635395fb6b7ddb87b256f31ba4b80333459d7b3bf70c103ee7b38e4f.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/ff0f74a6635395fb6b7ddb87b256f31ba4b80333459d7b3bf70c103ee7b38e4f.jpg)

![ffcaaaf4a0b8fd9a920eeb6b36bbeee7485732c245895fd8eaf19fab49d902f7.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/images/ffcaaaf4a0b8fd9a920eeb6b36bbeee7485732c245895fd8eaf19fab49d902f7.jpg)

### Tables

![43135b906244f535c16947868ac76a09f173ed7b9037838b4445f411a95d552a.jpg](../icml_results/109_FlashSketch_%20Sketch-Kernel%20Co-Design%20for%20Fast%20Sparse%20Sketching%20on%20GPUs/tables/43135b906244f535c16947868ac76a09f173ed7b9037838b4445f411a95d552a.jpg)

<a id="paper-0109"></a>

## Learning Randomized Reductions


### Images

![52c8b93e361aa9f76ec599dafdcc8c77f59ca301affc5078d8006ac2b7f782c1.jpg](../icml_results/111_Learning%20Randomized%20Reductions/images/52c8b93e361aa9f76ec599dafdcc8c77f59ca301affc5078d8006ac2b7f782c1.jpg)

![5c4093e2aaf986f1f73a5030a0449bf79ab654a962e557831c2b87fce1807bc1.jpg](../icml_results/111_Learning%20Randomized%20Reductions/images/5c4093e2aaf986f1f73a5030a0449bf79ab654a962e557831c2b87fce1807bc1.jpg)

![c45f952483c9fec62a5fcb30884d1f7d6b4f5cf1de0d6bf086196675caa56e5e.jpg](../icml_results/111_Learning%20Randomized%20Reductions/images/c45f952483c9fec62a5fcb30884d1f7d6b4f5cf1de0d6bf086196675caa56e5e.jpg)

![e3f85487d58e737a2168782e7d45aceb3d9b4419008c5ee25cc75b86a425d1b1.jpg](../icml_results/111_Learning%20Randomized%20Reductions/images/e3f85487d58e737a2168782e7d45aceb3d9b4419008c5ee25cc75b86a425d1b1.jpg)

![f6601874f9c0ec72b3c7fac3817746131caa442b2bd59d7b1935baad0d1af4bf.jpg](../icml_results/111_Learning%20Randomized%20Reductions/images/f6601874f9c0ec72b3c7fac3817746131caa442b2bd59d7b1935baad0d1af4bf.jpg)

### Tables

![018cf3846bef0f8a87a613ceb610ff374c7dedb1bc984f24731df996cc169778.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/018cf3846bef0f8a87a613ceb610ff374c7dedb1bc984f24731df996cc169778.jpg)

![07ed65b0075b4c08638789f6ce599aaac91485bb09e2b73fb8f6d8a3d8a1d8aa.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/07ed65b0075b4c08638789f6ce599aaac91485bb09e2b73fb8f6d8a3d8a1d8aa.jpg)

![09612dc657426d8e198805d10a561c4c2f2a3d72a7f4c2639cb838355ea39cf9.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/09612dc657426d8e198805d10a561c4c2f2a3d72a7f4c2639cb838355ea39cf9.jpg)

![0a30b9428c23cadd1faf97df0c75cad3c371088172eebe927f25b6ce6fafa4de.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/0a30b9428c23cadd1faf97df0c75cad3c371088172eebe927f25b6ce6fafa4de.jpg)

![1222690313ad45bb907e542b45aa15772069b2dc075e240162e69e00caa024c3.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/1222690313ad45bb907e542b45aa15772069b2dc075e240162e69e00caa024c3.jpg)

![23944c4ec7ecb7f999be9b15f4b5f844fe1eb9bd7cf2c52ffbffae550987a9c8.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/23944c4ec7ecb7f999be9b15f4b5f844fe1eb9bd7cf2c52ffbffae550987a9c8.jpg)

![29a0fe2783ea5c3bf185368fad805a6d1e92f18d23722883aae1ad03ba920864.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/29a0fe2783ea5c3bf185368fad805a6d1e92f18d23722883aae1ad03ba920864.jpg)

![2b8598379e6f9e5ebb2ab70015694ccf8c70ca9557f215d9aa08dace2a940005.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/2b8598379e6f9e5ebb2ab70015694ccf8c70ca9557f215d9aa08dace2a940005.jpg)

![40ceb695ef1f6a21950aec3ae66a5949c74a8d1e9076c627057152aa97c2b39a.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/40ceb695ef1f6a21950aec3ae66a5949c74a8d1e9076c627057152aa97c2b39a.jpg)

![418167af352752157e92e121ef5dbf827f621e11261116b0f5bf1ff5d63bb54b.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/418167af352752157e92e121ef5dbf827f621e11261116b0f5bf1ff5d63bb54b.jpg)

![41a73e4ad814c85e6d9234af83198e97f21b904ec6799ceed1b2a122a444d114.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/41a73e4ad814c85e6d9234af83198e97f21b904ec6799ceed1b2a122a444d114.jpg)

![5213a2147fe25b55bb6cc23d54d5bedef351366504a6748f5c35914164a1a202.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/5213a2147fe25b55bb6cc23d54d5bedef351366504a6748f5c35914164a1a202.jpg)

![57474ef301bd79d07e283ce7456c7bf07b6017c5ffcb468db6b29fb697d6dfaa.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/57474ef301bd79d07e283ce7456c7bf07b6017c5ffcb468db6b29fb697d6dfaa.jpg)

![5bba9ff4e49a5a607ba1b0381f4dc4eaa8f02650909b4562486cc1f06734055f.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/5bba9ff4e49a5a607ba1b0381f4dc4eaa8f02650909b4562486cc1f06734055f.jpg)

![5caa888b5c6e79bd19a6d66aa8ea6b8d222c10ae45e961686a5fde821ab0d2e5.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/5caa888b5c6e79bd19a6d66aa8ea6b8d222c10ae45e961686a5fde821ab0d2e5.jpg)

![698926cfb3abdba54930932c263395d19d000d6bfb3b8eb28f1887d46fcf07e4.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/698926cfb3abdba54930932c263395d19d000d6bfb3b8eb28f1887d46fcf07e4.jpg)

![69ed67d45dd3696a211eae5540e6d1b1d6ad26b2d173e0e083e93492013cbbad.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/69ed67d45dd3696a211eae5540e6d1b1d6ad26b2d173e0e083e93492013cbbad.jpg)

![6bb88c57174253c945e4d1454d275404577cd656c288fd8dae9cd6ae85976912.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/6bb88c57174253c945e4d1454d275404577cd656c288fd8dae9cd6ae85976912.jpg)

![6e2ce698c630023a4540c840790df02aba1c1e226d7f8aa009b8f4cc7ad186c6.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/6e2ce698c630023a4540c840790df02aba1c1e226d7f8aa009b8f4cc7ad186c6.jpg)

![721dbcdaee3195b49eae295156fa180d1cbd291740f4979324d3cd934de15c11.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/721dbcdaee3195b49eae295156fa180d1cbd291740f4979324d3cd934de15c11.jpg)

![78ad65d7f4b161c5dacb0315505abbe4ab19246088b6c13c45c85a612c441c98.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/78ad65d7f4b161c5dacb0315505abbe4ab19246088b6c13c45c85a612c441c98.jpg)

![7a74885e7971a67da0e84973239f49d860c20b1e17dd52913d4c2ea0dbb804bd.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/7a74885e7971a67da0e84973239f49d860c20b1e17dd52913d4c2ea0dbb804bd.jpg)

![8987443a2e4706c4247957f8253a33757721b4ee326453020fdc73d323f0690f.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/8987443a2e4706c4247957f8253a33757721b4ee326453020fdc73d323f0690f.jpg)

![89f8e6bb2e179a9840f075ace4ca39572a39a4e037b1dfb2fd18a3ea81ab7a97.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/89f8e6bb2e179a9840f075ace4ca39572a39a4e037b1dfb2fd18a3ea81ab7a97.jpg)

![8b5c89180d021e2066e531fe381bcfd96410c67d9ddf533497c217babddfad11.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/8b5c89180d021e2066e531fe381bcfd96410c67d9ddf533497c217babddfad11.jpg)

![9692838062920cdfb0908a9a8c53ced68faaa81fe7c2f6b14c0d44a2cb928f76.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/9692838062920cdfb0908a9a8c53ced68faaa81fe7c2f6b14c0d44a2cb928f76.jpg)

![9c5c588e360174c36cee4fa3799fd361472d182ef3921256d090fa179e499ceb.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/9c5c588e360174c36cee4fa3799fd361472d182ef3921256d090fa179e499ceb.jpg)

![a1953ec69361960908706c732a9f72e529768537dc366c42ac5f157908ab1484.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/a1953ec69361960908706c732a9f72e529768537dc366c42ac5f157908ab1484.jpg)

![a1f7fd7ea7dee71107dd1285eed16a52694e9409f2c2132b4551563e4d846b2a.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/a1f7fd7ea7dee71107dd1285eed16a52694e9409f2c2132b4551563e4d846b2a.jpg)

![a6d34df7797d37b7a0fdd836642303321a0644197e8c8dd139c97396111767ca.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/a6d34df7797d37b7a0fdd836642303321a0644197e8c8dd139c97396111767ca.jpg)

![a92044578ebdaca9e2a8c86d0d498b7f2de51062ea925ba430e0ec204ec37826.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/a92044578ebdaca9e2a8c86d0d498b7f2de51062ea925ba430e0ec204ec37826.jpg)

![aacc68a591c7196b79a511b3d1279f99c09df78299f9b9897c305f1c1f128fb6.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/aacc68a591c7196b79a511b3d1279f99c09df78299f9b9897c305f1c1f128fb6.jpg)

![ac847f93a1789dc03359443a974fb435b5a6c558553b48cbf23d0a911e78bea7.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/ac847f93a1789dc03359443a974fb435b5a6c558553b48cbf23d0a911e78bea7.jpg)

![addbd9d4bf3d6ba508c2de4400a95995ac1e20ca153a47ff5c641332af8bc10f.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/addbd9d4bf3d6ba508c2de4400a95995ac1e20ca153a47ff5c641332af8bc10f.jpg)

![b889c67a4c73f8b77fd6d6a04a65e06cf85ee5346916806b9ddc5e2c9e540ea1.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/b889c67a4c73f8b77fd6d6a04a65e06cf85ee5346916806b9ddc5e2c9e540ea1.jpg)

![bae6cb780a303ead05b92f71a420cdeae39a13039f900d1fec8e8836002f830b.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/bae6cb780a303ead05b92f71a420cdeae39a13039f900d1fec8e8836002f830b.jpg)

![bb73ab094adf6a7f2df01fd73ea3c9e3849980cc103ad693eb9404ee726c68f4.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/bb73ab094adf6a7f2df01fd73ea3c9e3849980cc103ad693eb9404ee726c68f4.jpg)

![bc059fbcc7171a6f167fc4b6583a1fa6c818cdba75549b18a649aade6ba5b1f2.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/bc059fbcc7171a6f167fc4b6583a1fa6c818cdba75549b18a649aade6ba5b1f2.jpg)

![c302bd5990c20ba78f23baf5e38e9fbb2d94d4b944264d8dff65d6110058840a.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/c302bd5990c20ba78f23baf5e38e9fbb2d94d4b944264d8dff65d6110058840a.jpg)

![c6b01499e40a4678ecce7a8139970682c07a50503d6db034753ad47da64172d2.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/c6b01499e40a4678ecce7a8139970682c07a50503d6db034753ad47da64172d2.jpg)

![cbed22c8c60578c6b486aafddbf323b795c131972721469f6b2c087d30dea930.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/cbed22c8c60578c6b486aafddbf323b795c131972721469f6b2c087d30dea930.jpg)

![cd87edad630fe7472ea82e3642f06531ea2f90599b20122b42560aa5925b1344.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/cd87edad630fe7472ea82e3642f06531ea2f90599b20122b42560aa5925b1344.jpg)

![ce1219ec2b88ac8b481370986c53322606b4c985f0a74c24f19e1d0949952f55.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/ce1219ec2b88ac8b481370986c53322606b4c985f0a74c24f19e1d0949952f55.jpg)

![ce7773b6095eb73525b3d53fde92691fbd60131a5d13cf2ec6fb87ac17158d6a.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/ce7773b6095eb73525b3d53fde92691fbd60131a5d13cf2ec6fb87ac17158d6a.jpg)

![d191a1ba950e0b5f50203121d39644293445e5ae9fdd5ff9c207aaef82ff2857.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/d191a1ba950e0b5f50203121d39644293445e5ae9fdd5ff9c207aaef82ff2857.jpg)

![d1b27ad81be8287da5534d64c5c5092407cff6e3d4d05511e93330a191309c5a.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/d1b27ad81be8287da5534d64c5c5092407cff6e3d4d05511e93330a191309c5a.jpg)

![d5173874e0a76e68ac3c585347e73fa8c140e84b86e509021295c87acd45fba1.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/d5173874e0a76e68ac3c585347e73fa8c140e84b86e509021295c87acd45fba1.jpg)

![d699bc7d9338f6df140e9bd76586977e3a3b8a70fcb9c489214eb56b654e52ac.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/d699bc7d9338f6df140e9bd76586977e3a3b8a70fcb9c489214eb56b654e52ac.jpg)

![d824779b91cdd894c1d8c25988891cb1163499f58ed4100f5a69178bff78d2c8.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/d824779b91cdd894c1d8c25988891cb1163499f58ed4100f5a69178bff78d2c8.jpg)

![d8b2ff0bb370241fc63fe87ee33d3e08b15f280c3c574aaeae2d1f8651431ba1.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/d8b2ff0bb370241fc63fe87ee33d3e08b15f280c3c574aaeae2d1f8651431ba1.jpg)

![dcad805fb27f7ea86f8dc4a2e03e6a6cac0ff4047347127d2081b8a9e25c75c5.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/dcad805fb27f7ea86f8dc4a2e03e6a6cac0ff4047347127d2081b8a9e25c75c5.jpg)

![e2f223b8b85354b467062bdd12292b546b0d2889080c786f514360f2a08169ed.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/e2f223b8b85354b467062bdd12292b546b0d2889080c786f514360f2a08169ed.jpg)

![e3e59e4b7c050676b61c47983180fcc2a3e7ed6960d207e4a3a0b70aa5dc6006.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/e3e59e4b7c050676b61c47983180fcc2a3e7ed6960d207e4a3a0b70aa5dc6006.jpg)

![e473f1c3733ecf6b063e7200c61b83a6402fd1e20a3946423c2d5e1a1ec7eefb.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/e473f1c3733ecf6b063e7200c61b83a6402fd1e20a3946423c2d5e1a1ec7eefb.jpg)

![e51d7c85ec50ba6d53c447e3b3ffc0209f4a7aff1ab03d3e6a8139b289de9732.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/e51d7c85ec50ba6d53c447e3b3ffc0209f4a7aff1ab03d3e6a8139b289de9732.jpg)

![f27cf2f08c481c02581f363ccdbd861add8f0fa2800871dcf931979d40caf3bd.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/f27cf2f08c481c02581f363ccdbd861add8f0fa2800871dcf931979d40caf3bd.jpg)

![f5c10d4f3b9e022b282b5d6ba2c4a63bf06c1aee3f2b5280f176ff595eff5f3e.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/f5c10d4f3b9e022b282b5d6ba2c4a63bf06c1aee3f2b5280f176ff595eff5f3e.jpg)

![f5f491f3fcd38b1d05f9d8135d0e763ffd1a6c0d093f64c5dfb6badd1a9215a5.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/f5f491f3fcd38b1d05f9d8135d0e763ffd1a6c0d093f64c5dfb6badd1a9215a5.jpg)

![f9172190631e269aa88f7420212417cde8f2f5eda017733a8dd7071ab79f158d.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/f9172190631e269aa88f7420212417cde8f2f5eda017733a8dd7071ab79f158d.jpg)

![f9c5cd9c4584392359cf4780af4e245d3b4c35e0d153ed881b87e9efe4e19ee8.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/f9c5cd9c4584392359cf4780af4e245d3b4c35e0d153ed881b87e9efe4e19ee8.jpg)

![facfedfa9c625f36dfa8ad638d9467e9c7721ffbeb9b3c6e22b8f0ebdaf7684c.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/facfedfa9c625f36dfa8ad638d9467e9c7721ffbeb9b3c6e22b8f0ebdaf7684c.jpg)

![fd76bf7fd4bbe0ed57d9fc7053884727a611d90b426891408c415d80c32301de.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/fd76bf7fd4bbe0ed57d9fc7053884727a611d90b426891408c415d80c32301de.jpg)

![fd7c91f22839147c9f92dae5289356fdf1afe949807b341fea70742f16b7673e.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/fd7c91f22839147c9f92dae5289356fdf1afe949807b341fea70742f16b7673e.jpg)

![fdbd51b74295d68ebc0545072ebb4629521c4462bff63f9c7734fc855d66ad05.jpg](../icml_results/111_Learning%20Randomized%20Reductions/tables/fdbd51b74295d68ebc0545072ebb4629521c4462bff63f9c7734fc855d66ad05.jpg)

<a id="paper-0110"></a>

## Evaluating Robustness of Reasoning Models on Parameterized Logical Problems


### Images

![069ccdedf9302d7732b9d1b837ec16d939ce16159606be56b89a0fa4a3e36c8b.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/069ccdedf9302d7732b9d1b837ec16d939ce16159606be56b89a0fa4a3e36c8b.jpg)

![0742e10db3814c1ef7d878ac16f5967967cdd107d811bd96651a5a90d088c89a.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/0742e10db3814c1ef7d878ac16f5967967cdd107d811bd96651a5a90d088c89a.jpg)

![0b56e555e84c0d507d79aa0699ca9e0b337a2379341bca4b3e144136da2bb339.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/0b56e555e84c0d507d79aa0699ca9e0b337a2379341bca4b3e144136da2bb339.jpg)

![0e9f9d51864fadfeed078a784d32951013174616fcf99be7c98859ca1a14597c.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/0e9f9d51864fadfeed078a784d32951013174616fcf99be7c98859ca1a14597c.jpg)

![28f4cfea860256634164f65d82ee344bf419739abc4be75f151e7007817f9e7b.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/28f4cfea860256634164f65d82ee344bf419739abc4be75f151e7007817f9e7b.jpg)

![616a97ed0878919afe5dedb9fb84ff8d0bc31c8babfbf425daa17421b0402b57.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/616a97ed0878919afe5dedb9fb84ff8d0bc31c8babfbf425daa17421b0402b57.jpg)

![b583757216dc04c8e2235b5fd31264ff29069c282de7031e04f81164db6bb96c.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/b583757216dc04c8e2235b5fd31264ff29069c282de7031e04f81164db6bb96c.jpg)

![b810eaf83faa561ca97960adc01832c079c0a71003d54873560c79043ae8ff2a.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/b810eaf83faa561ca97960adc01832c079c0a71003d54873560c79043ae8ff2a.jpg)

![cdadf26cb44054e8507066831fbfcb7a8e6f73bfbc88c638735da2255bcc3861.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/cdadf26cb44054e8507066831fbfcb7a8e6f73bfbc88c638735da2255bcc3861.jpg)

![dba5b6d4dd5aa49fd7fa9f6559a281e7f91ba17a9aafc490ee655ed0603bab83.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/images/dba5b6d4dd5aa49fd7fa9f6559a281e7f91ba17a9aafc490ee655ed0603bab83.jpg)

### Tables

![0ed4bebe87ccd74d15b4f3f16db691d08e796a007c5608d7c7b4121b22dffb03.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/0ed4bebe87ccd74d15b4f3f16db691d08e796a007c5608d7c7b4121b22dffb03.jpg)

![1bbd0e8f6af90ef1c73e680b0bb8504d30a4929bd38626e84ec40652d203af57.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/1bbd0e8f6af90ef1c73e680b0bb8504d30a4929bd38626e84ec40652d203af57.jpg)

![3a70113da9818579be44ba7220987889a2930ec181096c9b554214c87974dd7e.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/3a70113da9818579be44ba7220987889a2930ec181096c9b554214c87974dd7e.jpg)

![6dc57e880ed3b821d0f8e51e342a70c0c21ddc46ed21a538ddfc2e7dc8d0b300.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/6dc57e880ed3b821d0f8e51e342a70c0c21ddc46ed21a538ddfc2e7dc8d0b300.jpg)

![7d7aac70dccc14445ddf29b3884b2689609e68e39d431b8da705a6536570df65.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/7d7aac70dccc14445ddf29b3884b2689609e68e39d431b8da705a6536570df65.jpg)

![7da7dcf93051e8dfc3e84a3517e46b0541585ab2dc855fd7a5f7b4b0f8478f47.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/7da7dcf93051e8dfc3e84a3517e46b0541585ab2dc855fd7a5f7b4b0f8478f47.jpg)

![8fdf97870a9399481aa507b01d8fea0d1f888f41db5fad61fb330196574620e3.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/8fdf97870a9399481aa507b01d8fea0d1f888f41db5fad61fb330196574620e3.jpg)

![a73ae4fb4e5ce72538147e49dd86ce5e05cc11e3fae837f3f7c711b0c91ba6af.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/a73ae4fb4e5ce72538147e49dd86ce5e05cc11e3fae837f3f7c711b0c91ba6af.jpg)

![db0c7877719fbf90b32015ba9e3e7369ae355092364cb5c0fa051a25669a5011.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/db0c7877719fbf90b32015ba9e3e7369ae355092364cb5c0fa051a25669a5011.jpg)

![efe6828539c589e86bcf96c7dc8d6f6629267a5d0b18d8f5851f017218e57869.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/efe6828539c589e86bcf96c7dc8d6f6629267a5d0b18d8f5851f017218e57869.jpg)

![f20c1ad374309f78be8d13216c9064e78daa2f6ee40afba5cbda375213d3b37c.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/f20c1ad374309f78be8d13216c9064e78daa2f6ee40afba5cbda375213d3b37c.jpg)

![fa0b9bb41f005a45f7555a79d72fde9846290c934868e04f74c31e425a224ebc.jpg](../icml_results/112_Evaluating%20Robustness%20of%20Reasoning%20Models%20on%20Parameterized%20Logical%20Problems/tables/fa0b9bb41f005a45f7555a79d72fde9846290c934868e04f74c31e425a224ebc.jpg)

<a id="paper-0111"></a>

## XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations


### Images

![11c70782ab41866ee8fb48928960c9a37025845029650cebc5f23d395a273b53.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/11c70782ab41866ee8fb48928960c9a37025845029650cebc5f23d395a273b53.jpg)

![146f3862655e0963f5fc3808ff4e16da7272407bc8f82951f4a165c36836b6a4.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/146f3862655e0963f5fc3808ff4e16da7272407bc8f82951f4a165c36836b6a4.jpg)

![160ec14081b1e745faf17ca636569906c0248be948a946e418e60bb083fcca94.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/160ec14081b1e745faf17ca636569906c0248be948a946e418e60bb083fcca94.jpg)

![22da4bdfcaf907c77dd77b9759a5bd62f7025726415609d549481ce2d827d175.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/22da4bdfcaf907c77dd77b9759a5bd62f7025726415609d549481ce2d827d175.jpg)

![2b2da823710a853ff07b1ae85a997f61c01735a8865392eb3cb3c5b3e421ea44.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/2b2da823710a853ff07b1ae85a997f61c01735a8865392eb3cb3c5b3e421ea44.jpg)

![35609ffd35ebebd1d72ad3bcd917aed9b7f7ae7081ad2ff905a03169314f750d.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/35609ffd35ebebd1d72ad3bcd917aed9b7f7ae7081ad2ff905a03169314f750d.jpg)

![49f39de5627fab651e238ae61e032a7e3dd754c38a59c8eec61ef33ae071f48b.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/49f39de5627fab651e238ae61e032a7e3dd754c38a59c8eec61ef33ae071f48b.jpg)

![507b3facf8f8f041be6c59cbc2ed4fd2bf8b287836c3cc112353d9f8852933d7.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/507b3facf8f8f041be6c59cbc2ed4fd2bf8b287836c3cc112353d9f8852933d7.jpg)

![64837d0307349cfb8ea1a1cb87ba347d3d287871c8f2521581ab35d5de913d3f.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/64837d0307349cfb8ea1a1cb87ba347d3d287871c8f2521581ab35d5de913d3f.jpg)

![735c9541f176f2dd019ae5f3a12d26bf9fc4a23045adbf064c8e95539e2ca2f7.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/735c9541f176f2dd019ae5f3a12d26bf9fc4a23045adbf064c8e95539e2ca2f7.jpg)

![75b2f9fc89fa9bc220b6a8f9837a5ffbe77902019ae7b81849f791810e9ee12b.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/75b2f9fc89fa9bc220b6a8f9837a5ffbe77902019ae7b81849f791810e9ee12b.jpg)

![94a692f629b3b44718801ea3c3a09d86c28d228f9ae07926b0d8225c1a84aede.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/94a692f629b3b44718801ea3c3a09d86c28d228f9ae07926b0d8225c1a84aede.jpg)

![ac29a31ba842acf25d05c7c1c9917b03a13528db7534d42bc9a10e4ff86bd624.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/ac29a31ba842acf25d05c7c1c9917b03a13528db7534d42bc9a10e4ff86bd624.jpg)

![bc69a72214a928e695607683940e9009502d2c887b77be781d25368c4fdb38a9.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/bc69a72214a928e695607683940e9009502d2c887b77be781d25368c4fdb38a9.jpg)

![c987de6d9dd34f2e9f817b0abb47134cc785671fa0b60b190a1f206875219e21.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/c987de6d9dd34f2e9f817b0abb47134cc785671fa0b60b190a1f206875219e21.jpg)

![d3c19809c5371421ead16ef29db9048dda2fd3f3a0352df081d7242e048b7e0f.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/d3c19809c5371421ead16ef29db9048dda2fd3f3a0352df081d7242e048b7e0f.jpg)

![e02994da8967c5f01b467676fc61008785e34c39011bb0d5c6a7661bc614ba56.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/e02994da8967c5f01b467676fc61008785e34c39011bb0d5c6a7661bc614ba56.jpg)

![ed594ebcf478b0f17d9e19a669e330d05c7bf8263b75eb7025544004071855ec.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/ed594ebcf478b0f17d9e19a669e330d05c7bf8263b75eb7025544004071855ec.jpg)

![f01919cf617e9bdda8581ceb1ff6bdf17742f5926faac2637297e5dfdbc8b201.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/f01919cf617e9bdda8581ceb1ff6bdf17742f5926faac2637297e5dfdbc8b201.jpg)

![f38deea312ecd063f46dc0790bae4b0536f76df5dca35348596e2c4c94f89e2b.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/f38deea312ecd063f46dc0790bae4b0536f76df5dca35348596e2c4c94f89e2b.jpg)

![f654c1cc998c8d8c15f2d131f076c3528c1a553c5cf56da34c91b54fc3439ca7.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/images/f654c1cc998c8d8c15f2d131f076c3528c1a553c5cf56da34c91b54fc3439ca7.jpg)

### Tables

![05ee11a1d43e04c88eb342be079d337b49f642cbd47e085c68ccd63b6b8256a6.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/05ee11a1d43e04c88eb342be079d337b49f642cbd47e085c68ccd63b6b8256a6.jpg)

![0fb3beba7ae41ead0535916b49703eebc638435b27a50803e9a2d19f4cc3654c.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/0fb3beba7ae41ead0535916b49703eebc638435b27a50803e9a2d19f4cc3654c.jpg)

![10e6f746950d65abe21c89a234af0b75505acf4972bdb6247ccbe434fc980b12.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/10e6f746950d65abe21c89a234af0b75505acf4972bdb6247ccbe434fc980b12.jpg)

![13efc4ac43f9a09cf46853aa59e40a0344d2a2e60d23bc0bb2e6cc2305099c26.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/13efc4ac43f9a09cf46853aa59e40a0344d2a2e60d23bc0bb2e6cc2305099c26.jpg)

![28a9874c6558810d99d12efbbb3c5d2dad989ed684ee5137e09f7138b0e9e7cb.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/28a9874c6558810d99d12efbbb3c5d2dad989ed684ee5137e09f7138b0e9e7cb.jpg)

![2dcc1ccd530644f67458807ae70e182f102a1726f4d78d900aab2f2ed9cc414e.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/2dcc1ccd530644f67458807ae70e182f102a1726f4d78d900aab2f2ed9cc414e.jpg)

![3f0312db3243e53c96c923d351622330303c51b8acae6ad492df750641758785.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/3f0312db3243e53c96c923d351622330303c51b8acae6ad492df750641758785.jpg)

![49334e588e388155385e57bbb294efbc5841db6be15678e16711d7508abfc247.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/49334e588e388155385e57bbb294efbc5841db6be15678e16711d7508abfc247.jpg)

![499639719e1ef8500a21b1e2f2a905de877e847c739fa8b8a5870fae78f09510.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/499639719e1ef8500a21b1e2f2a905de877e847c739fa8b8a5870fae78f09510.jpg)

![4a6331e4248bf81712b6440aa099ab42d1d8eb531c028d206e53d13380578420.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/4a6331e4248bf81712b6440aa099ab42d1d8eb531c028d206e53d13380578420.jpg)

![6476cc361ae644ea5d071a34597d34d74109316731c6a7d109b98d89c26aac0b.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/6476cc361ae644ea5d071a34597d34d74109316731c6a7d109b98d89c26aac0b.jpg)

![65e6ee4699e23e48ec4a3464a50b322a5cf982059181082fa0d0ce5acd44a134.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/65e6ee4699e23e48ec4a3464a50b322a5cf982059181082fa0d0ce5acd44a134.jpg)

![66b9f4330b9c16627a5bc91918b672c2548771e18705f06ff83ce7003b969d8a.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/66b9f4330b9c16627a5bc91918b672c2548771e18705f06ff83ce7003b969d8a.jpg)

![674d10e2aed5e111fb3fd16f7028b058de74ec5b43de5fdb367baa4d9d435b0f.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/674d10e2aed5e111fb3fd16f7028b058de74ec5b43de5fdb367baa4d9d435b0f.jpg)

![68d1e33c6793626bf198e9a46e0b513b79ed17a340fe57434c25869b1cea3314.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/68d1e33c6793626bf198e9a46e0b513b79ed17a340fe57434c25869b1cea3314.jpg)

![6c241639c737796eb6597bc3d7c58912922ad4d62b625c6aa784964a77ba8f7a.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/6c241639c737796eb6597bc3d7c58912922ad4d62b625c6aa784964a77ba8f7a.jpg)

![6f8f6feec739e3c39ecd4f8107ee2daccd153b9490679a8683af6db6b7e27e19.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/6f8f6feec739e3c39ecd4f8107ee2daccd153b9490679a8683af6db6b7e27e19.jpg)

![925d97cbb69447d00dfa1c2db59d79d1ae9939106a0870f86ff4b0c38dfe3ec1.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/925d97cbb69447d00dfa1c2db59d79d1ae9939106a0870f86ff4b0c38dfe3ec1.jpg)

![9d8b69cbfaf1b98ef525c14af9d8d6c4ecc615268151e9a154b901909dfd6565.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/9d8b69cbfaf1b98ef525c14af9d8d6c4ecc615268151e9a154b901909dfd6565.jpg)

![a5f01b65a343c710e7c05793f27f13210c6cdf2b9d70c803ee20c1233e376a0a.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/a5f01b65a343c710e7c05793f27f13210c6cdf2b9d70c803ee20c1233e376a0a.jpg)

![b0bb52332bef1ab2aa4d16ab1b50908786a116081740221cb0e29763159cbf45.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/b0bb52332bef1ab2aa4d16ab1b50908786a116081740221cb0e29763159cbf45.jpg)

![b68e1be482f169dee1ca75846ae7d6247d93f789c529772e340c0964da14f192.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/b68e1be482f169dee1ca75846ae7d6247d93f789c529772e340c0964da14f192.jpg)

![c4a7a508629415865bf025b304263e0d1917af3b82fd814608241d98fbbba56f.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/c4a7a508629415865bf025b304263e0d1917af3b82fd814608241d98fbbba56f.jpg)

![c62842108ca22b888a184643d8e9bfc0d0d91c7ad8f48395278a4caeb31b58f6.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/c62842108ca22b888a184643d8e9bfc0d0d91c7ad8f48395278a4caeb31b58f6.jpg)

![d736223c63d8b2f4f67b6954c471317d783603dad03f5ed1b4daa96e2e00b7c5.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/d736223c63d8b2f4f67b6954c471317d783603dad03f5ed1b4daa96e2e00b7c5.jpg)

![e51c7f3d996b149b64187ae0fe754752e6816e4b840a10a09a084f7c8222d5f6.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/e51c7f3d996b149b64187ae0fe754752e6816e4b840a10a09a084f7c8222d5f6.jpg)

![e8eea46c421471d5a536f0bf4445664ba85bb35d43fea59bd1b7ef1ab4ab11e4.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/e8eea46c421471d5a536f0bf4445664ba85bb35d43fea59bd1b7ef1ab4ab11e4.jpg)

![ea527883f06e3990bf7f88d8d2853545ff53acee3dbc78d1e020f03ada0c049e.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/ea527883f06e3990bf7f88d8d2853545ff53acee3dbc78d1e020f03ada0c049e.jpg)

![f572610c2a020293e5d33be506e42e8aa24b5ea140b1a7a9d862c7a0fb4424b2.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/f572610c2a020293e5d33be506e42e8aa24b5ea140b1a7a9d862c7a0fb4424b2.jpg)

![fe17482db8c0deca56e5006dce0b4d8c43df315e5453cb0d26456fe16c30ba7e.jpg](../icml_results/113_XR-1_%20Towards%20Versatile%20Vision-Language-Action%20Models%20via%20Learning%20Unified%20Vision-Motion%20Representat/tables/fe17482db8c0deca56e5006dce0b4d8c43df315e5453cb0d26456fe16c30ba7e.jpg)

<a id="paper-0112"></a>

## Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking


### Images

![01a5935780f8ddab2a1f158e1235e00e49d389d1a3dafd6a09e2d85aa37a17fd.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/images/01a5935780f8ddab2a1f158e1235e00e49d389d1a3dafd6a09e2d85aa37a17fd.jpg)

![38363bb89cc20b331fa9e2ec25d8409ef8d98f888bd0072bd687c3546e94d042.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/images/38363bb89cc20b331fa9e2ec25d8409ef8d98f888bd0072bd687c3546e94d042.jpg)

![52cabba7f8a4b4547bcbfdbe7ffd0d15b6220aa1429b34db1fbaa75b7bde3bf0.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/images/52cabba7f8a4b4547bcbfdbe7ffd0d15b6220aa1429b34db1fbaa75b7bde3bf0.jpg)

![6eb3d30f436b9b8a3e4c3660a3ea7f0611057fb3a2b068b214fc31f68ba66587.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/images/6eb3d30f436b9b8a3e4c3660a3ea7f0611057fb3a2b068b214fc31f68ba66587.jpg)

![bc063015af9c479645f4da3195f00a4dbc74445c2d53ae05d0d1e2a82677b4ab.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/images/bc063015af9c479645f4da3195f00a4dbc74445c2d53ae05d0d1e2a82677b4ab.jpg)

![e8244c6c7ab339c6efb4ff6e13405e9d4cf0c5057e072d8217dd575132ed0c54.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/images/e8244c6c7ab339c6efb4ff6e13405e9d4cf0c5057e072d8217dd575132ed0c54.jpg)

### Tables

![379299f854d3b88b095646f0c650c4b8868b0c6476ee77d65391e8335b2ef0cd.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/379299f854d3b88b095646f0c650c4b8868b0c6476ee77d65391e8335b2ef0cd.jpg)

![4bc5eea511dc2a963adfab9c0dff0b3d369f626ed10b12448ea70220c95babea.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/4bc5eea511dc2a963adfab9c0dff0b3d369f626ed10b12448ea70220c95babea.jpg)

![5d88bc9977eb6312a8eb5e6cb0f48e85a3bc8c985852aa0cb7c7971c3660c265.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/5d88bc9977eb6312a8eb5e6cb0f48e85a3bc8c985852aa0cb7c7971c3660c265.jpg)

![6f82bcb9d8973e806d623ad28597a66140f385d3554f218f444c9e6475f4a92d.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/6f82bcb9d8973e806d623ad28597a66140f385d3554f218f444c9e6475f4a92d.jpg)

![8101cbec22f2128344df190edcf1a9672b4e3a12aa609e4d3f0b9482a67abf55.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/8101cbec22f2128344df190edcf1a9672b4e3a12aa609e4d3f0b9482a67abf55.jpg)

![d5b081b6428613204604269149235712182fb177b4a97395a034d141f720aed6.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/d5b081b6428613204604269149235712182fb177b4a97395a034d141f720aed6.jpg)

![dae2bb1d2fcc773b5d5cefe2ba5041060e68f109bcf658387d4acfccda35c8ce.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/dae2bb1d2fcc773b5d5cefe2ba5041060e68f109bcf658387d4acfccda35c8ce.jpg)

![e7ccd59d3c883eb4e6ae4bdd186ecad3d8aee7a56cbb3a8effaca8e383ef599d.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/e7ccd59d3c883eb4e6ae4bdd186ecad3d8aee7a56cbb3a8effaca8e383ef599d.jpg)

![f2bf6bb875c8f07e36d7fae2ae0200789ff5049fa397a4c7e6fb093271aa611f.jpg](../icml_results/114_Jailbreak%20Foundry_%20From%20Papers%20to%20Runnable%20Attacks%20for%20Reproducible%20Benchmarking/tables/f2bf6bb875c8f07e36d7fae2ae0200789ff5049fa397a4c7e6fb093271aa611f.jpg)

<a id="paper-0113"></a>

## Clustering in Deep Stochastic Transformers


### Images

![006e7b4d63558a8d8638ee8bb02af8916333e30071663551bde921b023ad0352.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/006e7b4d63558a8d8638ee8bb02af8916333e30071663551bde921b023ad0352.jpg)

![1410c61c0606a1b75f8faeda4bb62668e38abe9d63bfe7d8f2c14609627f488c.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/1410c61c0606a1b75f8faeda4bb62668e38abe9d63bfe7d8f2c14609627f488c.jpg)

![1c6447d588f4f22a8c2b5628749b388d94f99de2fb06215bd060fa385640fa6c.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/1c6447d588f4f22a8c2b5628749b388d94f99de2fb06215bd060fa385640fa6c.jpg)

![1db3c22f91b2f99dc29904323a115c9b3a3af6485abaebaa916f3ddd58205d6d.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/1db3c22f91b2f99dc29904323a115c9b3a3af6485abaebaa916f3ddd58205d6d.jpg)

![441b154c4d79a33481e2230c19c25d5988328da376439d5ea1f2567722391b0f.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/441b154c4d79a33481e2230c19c25d5988328da376439d5ea1f2567722391b0f.jpg)

![45141006e74f73ae86571a4797432cea3aba9003d6e533415646b2cb3aa8ff5f.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/45141006e74f73ae86571a4797432cea3aba9003d6e533415646b2cb3aa8ff5f.jpg)

![6aad18ac20ecee1faaabe21bf67d8b63e709d9dd82335cfdf8d63a589365db25.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/6aad18ac20ecee1faaabe21bf67d8b63e709d9dd82335cfdf8d63a589365db25.jpg)

![77eaff276b34490a60e2d43e8f23dc2a0bd81cbd880fcb2f58d62c8a0be69cdf.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/77eaff276b34490a60e2d43e8f23dc2a0bd81cbd880fcb2f58d62c8a0be69cdf.jpg)

![a0f74c1de2d30982344e851eeffcf1a3bc509f529eb187edc5db8893910296dd.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/a0f74c1de2d30982344e851eeffcf1a3bc509f529eb187edc5db8893910296dd.jpg)

![ae75718f4c29301033e4f079c19eddd9fc69a4347962df418bf364a37b53f6af.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/ae75718f4c29301033e4f079c19eddd9fc69a4347962df418bf364a37b53f6af.jpg)

![c25ccfced620a97a672f8e11588e7e597b60eeaa948824f00b9b71a62b8b2546.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/c25ccfced620a97a672f8e11588e7e597b60eeaa948824f00b9b71a62b8b2546.jpg)

![fe56acaff3243f78583da09024ead552d2c8366299253fe73c299fdf51554432.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/images/fe56acaff3243f78583da09024ead552d2c8366299253fe73c299fdf51554432.jpg)

### Tables

![70df8d358ee9a3e8ed7162cf1031e07bf3fccee5ea443aa1bd0e3ea1f434d2b3.jpg](../icml_results/115_Clustering%20in%20Deep%20Stochastic%20Transformers/tables/70df8d358ee9a3e8ed7162cf1031e07bf3fccee5ea443aa1bd0e3ea1f434d2b3.jpg)

<a id="paper-0114"></a>

## Multimodal Nested Learning for Decoupled and Coordinated Optimization


### Images

![035dc0ebc7af09cc255200b0c2a41a412def415f3416ff13ab08e6f6192fc9b6.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/035dc0ebc7af09cc255200b0c2a41a412def415f3416ff13ab08e6f6192fc9b6.jpg)

![0496b38420cf9f73fe08cf30a6679d5237d1c0c52ed83dd697497f2ae8924b2e.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/0496b38420cf9f73fe08cf30a6679d5237d1c0c52ed83dd697497f2ae8924b2e.jpg)

![29b3cdca63358de0408b63d4157e741a16a6dd99313338c65ca2e9b475ab353b.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/29b3cdca63358de0408b63d4157e741a16a6dd99313338c65ca2e9b475ab353b.jpg)

![719fadace97de539523a8c05709d4c0232712cda69a12a273217cc5a17f7bfd6.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/719fadace97de539523a8c05709d4c0232712cda69a12a273217cc5a17f7bfd6.jpg)

![8c607c3d6e646a46fe5692a47ed958a2d190eb6e7ae164a92e8b6c8f97a6a9f7.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/8c607c3d6e646a46fe5692a47ed958a2d190eb6e7ae164a92e8b6c8f97a6a9f7.jpg)

![a322ea37921fabd0d04252c49f2ea99185ff77bee7be931264eeac833765c19a.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/a322ea37921fabd0d04252c49f2ea99185ff77bee7be931264eeac833765c19a.jpg)

![ba635fce2ab4bb86e2257ae9ad1fd74f640face65a49f3761f5ad231684572d4.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/ba635fce2ab4bb86e2257ae9ad1fd74f640face65a49f3761f5ad231684572d4.jpg)

![d1d76ff5d8d4b3eca7511d9e95f1c12ccb973b99d1e4c9e474dbb7dc770c39cb.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/d1d76ff5d8d4b3eca7511d9e95f1c12ccb973b99d1e4c9e474dbb7dc770c39cb.jpg)

![d39aa9acbc35786fe1de0934ac6af2aee96f3984c7f8cd96a9108dfb8eab8fd5.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/images/d39aa9acbc35786fe1de0934ac6af2aee96f3984c7f8cd96a9108dfb8eab8fd5.jpg)

### Tables

![0a171c216188038e7c6f81e2dffe70814131c1c7c489acf2db3f265cfb1d2fad.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/tables/0a171c216188038e7c6f81e2dffe70814131c1c7c489acf2db3f265cfb1d2fad.jpg)

![752f6419eb2086720812bb527047fb5979308318ac6b81820e69084cb14fbcbd.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/tables/752f6419eb2086720812bb527047fb5979308318ac6b81820e69084cb14fbcbd.jpg)

![a3bd9311f3b458ee0d3e930a2ce86eb7f27708e8c4ab04e14884b4e4ec25fd1d.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/tables/a3bd9311f3b458ee0d3e930a2ce86eb7f27708e8c4ab04e14884b4e4ec25fd1d.jpg)

![e8ee71677451980a6155f81a479e4e2be713be15f7238f89a1fe446d484a4334.jpg](../icml_results/116_Multimodal%20Nested%20Learning%20for%20Decoupled%20and%20Coordinated%20Optimization/tables/e8ee71677451980a6155f81a479e4e2be713be15f7238f89a1fe446d484a4334.jpg)

<a id="paper-0115"></a>

## L2G-NET: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations


### Images

![26de3cea6547a6d546c184c15a0556df77b414e9403ffca1f2fae9f523ba85d5.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/26de3cea6547a6d546c184c15a0556df77b414e9403ffca1f2fae9f523ba85d5.jpg)

![646be7f520f3480dee765dd975f3ef977d60cc58755c58c9e286b4260eb86d18.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/646be7f520f3480dee765dd975f3ef977d60cc58755c58c9e286b4260eb86d18.jpg)

![646dce413b9dba4546b24739a792cd6d00e8e186f423952e8b8ca13de09f2084.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/646dce413b9dba4546b24739a792cd6d00e8e186f423952e8b8ca13de09f2084.jpg)

![7322b16b01b821ea93f0af48154730e57b556739fc59ebf44e1937003f610e34.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/7322b16b01b821ea93f0af48154730e57b556739fc59ebf44e1937003f610e34.jpg)

![93209b01116d74dd5f248ba90679c65b4e0da1466147b2c007e45cba593ab80a.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/93209b01116d74dd5f248ba90679c65b4e0da1466147b2c007e45cba593ab80a.jpg)

![97709cb2ea8f1e934b10c1951f5102333d5a9b1f4adb85b965ebe8506e4d463c.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/97709cb2ea8f1e934b10c1951f5102333d5a9b1f4adb85b965ebe8506e4d463c.jpg)

![acb9b16aa2fe9616144170851b34e6112b77e77ec17be8c9de4095c98767ae9d.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/acb9b16aa2fe9616144170851b34e6112b77e77ec17be8c9de4095c98767ae9d.jpg)

![b2ac5d7edf487726485c23e5c4a1d576cbf40bd839b566ada921fda2607bbb1e.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/b2ac5d7edf487726485c23e5c4a1d576cbf40bd839b566ada921fda2607bbb1e.jpg)

![b53b4c2eab6542aa87ea90ba2306cb94930a1283de3473eabc4893157fa39716.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/b53b4c2eab6542aa87ea90ba2306cb94930a1283de3473eabc4893157fa39716.jpg)

![d16977952f0a2c7e08eee3ab0bab5ed87990ab278a22f73c40bcfb3331a553fc.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/images/d16977952f0a2c7e08eee3ab0bab5ed87990ab278a22f73c40bcfb3331a553fc.jpg)

### Tables

![05b5ebf9f090642a89f70cd9a93221215339b7ca837ac05f81e0fbb29313899a.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/05b5ebf9f090642a89f70cd9a93221215339b7ca837ac05f81e0fbb29313899a.jpg)

![069ffc3872d43fd8f9ea4727bd5fe16d4d9f4397e272d535be9b00902bb81ec7.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/069ffc3872d43fd8f9ea4727bd5fe16d4d9f4397e272d535be9b00902bb81ec7.jpg)

![06f5b1398c8c1a7483c411ba7c1581655c386a5c072b639e1139a10f8cd15ed8.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/06f5b1398c8c1a7483c411ba7c1581655c386a5c072b639e1139a10f8cd15ed8.jpg)

![35f215ae84089959de7615c219e1e6240b366ab1ea311f4ecedf25b707291c14.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/35f215ae84089959de7615c219e1e6240b366ab1ea311f4ecedf25b707291c14.jpg)

![3fbfc90f61e75eb6fde36abe27cf20eddbfcac217c0f36d33d5ad60e8ef20bc7.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/3fbfc90f61e75eb6fde36abe27cf20eddbfcac217c0f36d33d5ad60e8ef20bc7.jpg)

![44dc1a121e5aead3d7f1ad0b356e5391ee73b274b3f96e796d5ea21098b80ffd.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/44dc1a121e5aead3d7f1ad0b356e5391ee73b274b3f96e796d5ea21098b80ffd.jpg)

![9351aa81bd65d0a9b4332d033f9673646ec563216673b807a4f9a35fa843c09a.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/9351aa81bd65d0a9b4332d033f9673646ec563216673b807a4f9a35fa843c09a.jpg)

![954d86f6371e0e52eac24df131c029b439f015799c76cf2042af65fc02055ccf.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/954d86f6371e0e52eac24df131c029b439f015799c76cf2042af65fc02055ccf.jpg)

![ad94c820559b6bbfb6c185f856be5f8ba3e97e1ab90bf4d5b7570111875f746a.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/ad94c820559b6bbfb6c185f856be5f8ba3e97e1ab90bf4d5b7570111875f746a.jpg)

![b1779a5cee8116e5ed76a5d9f6948500a313339d03deedefa1fc03f1ee5f177e.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/b1779a5cee8116e5ed76a5d9f6948500a313339d03deedefa1fc03f1ee5f177e.jpg)

![cd5be63e1d4a8f7e976546831ae0e4da4c5e4cfcf768a41d712bc1d89a142169.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/cd5be63e1d4a8f7e976546831ae0e4da4c5e4cfcf768a41d712bc1d89a142169.jpg)

![cd5cb563c0fdd679eacbdce5ca65699c3cf4d4ca3f9c98e52bf4864309c7f8b2.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/cd5cb563c0fdd679eacbdce5ca65699c3cf4d4ca3f9c98e52bf4864309c7f8b2.jpg)

![ea9d4a2506f3262811264b2589cf395154eb5eeff641041d7623399a6bc21158.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/ea9d4a2506f3262811264b2589cf395154eb5eeff641041d7623399a6bc21158.jpg)

![fcf29b65ae5d930a231f42aa54add37cdbf6a4e15a3da6c3a08f50aa75679214.jpg](../icml_results/117_L2G-NET_%20Local%20to%20Global%20Spectral%20Graph%20Neural%20Networks%20via%20Cauchy%20Factorizations/tables/fcf29b65ae5d930a231f42aa54add37cdbf6a4e15a3da6c3a08f50aa75679214.jpg)

<a id="paper-0116"></a>

## Exact Functional ANOVA Decomposition for Categorical Inputs Models


### Images

![01586b0d3f4eb5ce4b9c2a6d92d4304af282e1a0e627ebe9535ddd90fe311fff.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/images/01586b0d3f4eb5ce4b9c2a6d92d4304af282e1a0e627ebe9535ddd90fe311fff.jpg)

![2abf6ccaf84d746a59b2ef7a01f57f36e1536915ef3ee4358d0b184f279690c6.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/images/2abf6ccaf84d746a59b2ef7a01f57f36e1536915ef3ee4358d0b184f279690c6.jpg)

![527d1903cede9c2cd21fd7a4f05e234dd6f6731d5ab3dc79361578ff3cc31644.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/images/527d1903cede9c2cd21fd7a4f05e234dd6f6731d5ab3dc79361578ff3cc31644.jpg)

### Tables

![019b1a4fa127ae58e1f8e56186f354c666e84d263defd9f2354273020bb443c9.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/019b1a4fa127ae58e1f8e56186f354c666e84d263defd9f2354273020bb443c9.jpg)

![1e19a8a5f38f5b262d06d5aca004faf6c1006912b67f86a87ee2e911d99ff8d4.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/1e19a8a5f38f5b262d06d5aca004faf6c1006912b67f86a87ee2e911d99ff8d4.jpg)

![84c99007b24ba00796c70ca0be80d748242cb42e05233a20ab540a281a14be84.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/84c99007b24ba00796c70ca0be80d748242cb42e05233a20ab540a281a14be84.jpg)

![b8f535ca2f13b9556b75aec162359a627b19f532cfadfe817b49a1ed0c012195.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/b8f535ca2f13b9556b75aec162359a627b19f532cfadfe817b49a1ed0c012195.jpg)

![bcb9a4eda228a0d4dfa535a0af1bfb206874d15b16a4a4aa79b3998645f01fed.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/bcb9a4eda228a0d4dfa535a0af1bfb206874d15b16a4a4aa79b3998645f01fed.jpg)

![be5a007df2ed403464b611f301d83f9cea0750cd3a8bc68414af3c103fd649da.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/be5a007df2ed403464b611f301d83f9cea0750cd3a8bc68414af3c103fd649da.jpg)

![c134b04f4e558671d7b45bd84a47711379325841c2580590e39f5a1b401a8142.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/c134b04f4e558671d7b45bd84a47711379325841c2580590e39f5a1b401a8142.jpg)

![e0b174d4878c3623e736c20c8a6fa78c9182ea5ad7b2e9c93136142f19e84afc.jpg](../icml_results/118_Exact%20Functional%20ANOVA%20Decomposition%20for%20Categorical%20Inputs%20Models/tables/e0b174d4878c3623e736c20c8a6fa78c9182ea5ad7b2e9c93136142f19e84afc.jpg)

<a id="paper-0117"></a>

## Learning to Execute Graph Algorithms Exactly with Graph Neural Networks


### Images

![1779698c433614980459525dd502396c9520c9842246e6477a1800ccbdf74a42.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/1779698c433614980459525dd502396c9520c9842246e6477a1800ccbdf74a42.jpg)

![2f7d2f680b2bbc9f0413f49d6c6cb76404eb24f526f66f8d43b996f298391ddf.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/2f7d2f680b2bbc9f0413f49d6c6cb76404eb24f526f66f8d43b996f298391ddf.jpg)

![3a99fce7fe6786fe7a9fc9c9272983c88163cd97313d125c2283cab02294279b.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/3a99fce7fe6786fe7a9fc9c9272983c88163cd97313d125c2283cab02294279b.jpg)

![436cc502213fe7fc908625c34e0c4a62960118ef315cc5956f75bfb371c04ed8.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/436cc502213fe7fc908625c34e0c4a62960118ef315cc5956f75bfb371c04ed8.jpg)

![487a32365aff871dc2dab4d5e304d427d11557d2d333e58935b2a157345a6dd6.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/487a32365aff871dc2dab4d5e304d427d11557d2d333e58935b2a157345a6dd6.jpg)

![72b12899191a5f2b4bab4d92844511a39dc35a3b81a702a4e4a3be0c1f0a370c.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/72b12899191a5f2b4bab4d92844511a39dc35a3b81a702a4e4a3be0c1f0a370c.jpg)

![7a1a6ce840a29f6228fc1590ab4b9741ca6c2a55111904e2b398db85f88b3431.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/7a1a6ce840a29f6228fc1590ab4b9741ca6c2a55111904e2b398db85f88b3431.jpg)

![7b22b513511a2720e2ef9f153f0df95ef79173cea337fa4072d582696bd8b360.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/7b22b513511a2720e2ef9f153f0df95ef79173cea337fa4072d582696bd8b360.jpg)

![84ed74eff67b33bae9b48280edbeec171cf46300e681d1246966c19d7e4e296a.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/84ed74eff67b33bae9b48280edbeec171cf46300e681d1246966c19d7e4e296a.jpg)

![876b08c6a1dd48b7e2146f8e2db8bbb769f994e36974af04e7e448ee42fd9b4b.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/876b08c6a1dd48b7e2146f8e2db8bbb769f994e36974af04e7e448ee42fd9b4b.jpg)

![a8bc9b9c06e7b299f48cf8723fb9120c04f90aff321edb3a42588bc5916b5282.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/a8bc9b9c06e7b299f48cf8723fb9120c04f90aff321edb3a42588bc5916b5282.jpg)

![b5e0c4ea39d8d5d61c53ea501a5d33b9489c355f0a0935055bc62bba65202a21.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/b5e0c4ea39d8d5d61c53ea501a5d33b9489c355f0a0935055bc62bba65202a21.jpg)

![bf17292e375b60c39a45e392f249ee7612f9affe5bac08192b6ed66e0b30a77b.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/bf17292e375b60c39a45e392f249ee7612f9affe5bac08192b6ed66e0b30a77b.jpg)

![c62d2e6c552c67b88a445a1733da0dcf1d899c87f82b6877a068a329b2445786.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/c62d2e6c552c67b88a445a1733da0dcf1d899c87f82b6877a068a329b2445786.jpg)

![d2fbd0b7ee474dbd649247093fe70bc4582233952f3c79fb787c9ccffb1fb129.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/d2fbd0b7ee474dbd649247093fe70bc4582233952f3c79fb787c9ccffb1fb129.jpg)

![f29e93f2a44b783982ea9bc80669f6c199bfae8b10bd530f5595624c69a6bc8b.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/images/f29e93f2a44b783982ea9bc80669f6c199bfae8b10bd530f5595624c69a6bc8b.jpg)

### Tables

![0fad265d8a1157d9f5fd5ad9db33e84c120624dcc1824fd66a2adbca8642e150.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/0fad265d8a1157d9f5fd5ad9db33e84c120624dcc1824fd66a2adbca8642e150.jpg)

![12ce3b688a0c856de335acf5e987604f4247adfd35089c156e1f222fbb9cf6f3.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/12ce3b688a0c856de335acf5e987604f4247adfd35089c156e1f222fbb9cf6f3.jpg)

![2a550eb50379ab121c45ab2939630d5ecd759f0e50572f4408c7ed691a85b3b4.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/2a550eb50379ab121c45ab2939630d5ecd759f0e50572f4408c7ed691a85b3b4.jpg)

![3486827d818af7f15e745b330d6a855913af32f3f6aa1fa19d235852d04a10d6.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/3486827d818af7f15e745b330d6a855913af32f3f6aa1fa19d235852d04a10d6.jpg)

![3d3d2a49723c6b5c2762a2154d6a18096d975cf73a3884f73e634919ef349d75.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/3d3d2a49723c6b5c2762a2154d6a18096d975cf73a3884f73e634919ef349d75.jpg)

![5d5a9de69c78822f8de8fc14a469fa4ba49983541c5d0a77290f9704511db559.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/5d5a9de69c78822f8de8fc14a469fa4ba49983541c5d0a77290f9704511db559.jpg)

![68e171e1913359d98298392803cf86ba35cf93565255aa167e7c2ff02ccc5108.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/68e171e1913359d98298392803cf86ba35cf93565255aa167e7c2ff02ccc5108.jpg)

![7a83605e945d492bb012fd5e72ff6920eeb641d68850cba8aafa7b422553b5bd.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/7a83605e945d492bb012fd5e72ff6920eeb641d68850cba8aafa7b422553b5bd.jpg)

![8a69d1b856eb5bdb12b1f8cd581e786a2d409a8f5137619eb4e53bc34291819f.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/8a69d1b856eb5bdb12b1f8cd581e786a2d409a8f5137619eb4e53bc34291819f.jpg)

![90b3f4e320860492272f2a863ef6f905d5fe4843a43895e8d08a548fa9bb4ee2.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/90b3f4e320860492272f2a863ef6f905d5fe4843a43895e8d08a548fa9bb4ee2.jpg)

![bb075d982e8e17dcc2a0d9daa816fe5187964d3e3d491fecdccde76d188bbb2b.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/bb075d982e8e17dcc2a0d9daa816fe5187964d3e3d491fecdccde76d188bbb2b.jpg)

![bd45909eca578baa267edbc4ce5e3d510a84af16414e15131e86a6cce9465e72.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/bd45909eca578baa267edbc4ce5e3d510a84af16414e15131e86a6cce9465e72.jpg)

![d3a4754c367bc6445bbce038b71040a1d459cf0827e9bcd59dd922e0583493d9.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/d3a4754c367bc6445bbce038b71040a1d459cf0827e9bcd59dd922e0583493d9.jpg)

![e5d99bde2b62713f9e5504fd252a76797d0c4d19b8c7a0d1a7cf85ff7b5b59a7.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/e5d99bde2b62713f9e5504fd252a76797d0c4d19b8c7a0d1a7cf85ff7b5b59a7.jpg)

![e76398cff0064080d7204ef502a4056cc255f457b28c4a3e76e90e163a6cae77.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/e76398cff0064080d7204ef502a4056cc255f457b28c4a3e76e90e163a6cae77.jpg)

![e7fd87c3f9beb30e36da52c41455ffeff65d761f5c648265c1b9e1a9299b68ac.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/e7fd87c3f9beb30e36da52c41455ffeff65d761f5c648265c1b9e1a9299b68ac.jpg)

![f034c62b23ac081a29dae655d085ea9773ee859cd4954aa7b3027ef13dfe0dc8.jpg](../icml_results/119_Learning%20to%20Execute%20Graph%20Algorithms%20Exactly%20with%20Graph%20Neural%20Networks/tables/f034c62b23ac081a29dae655d085ea9773ee859cd4954aa7b3027ef13dfe0dc8.jpg)

<a id="paper-0118"></a>

## On Efficient Scaling of GNNs via IO-Aware Layers Implementations


### Images

![454aaa01f1d7233a3a978b3ded93fdf26c9e57cec363a9ea01eae9cb931d7c41.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/images/454aaa01f1d7233a3a978b3ded93fdf26c9e57cec363a9ea01eae9cb931d7c41.jpg)

![c40c4ed6deb72716f1bc0d7ade97b9828021dbff9e9feddf5d654c68132f20c4.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/images/c40c4ed6deb72716f1bc0d7ade97b9828021dbff9e9feddf5d654c68132f20c4.jpg)

### Tables

![0143f56b8ca90cc988caa401c15e68b98fb7f167d936d04980a1e843c346fba6.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/0143f56b8ca90cc988caa401c15e68b98fb7f167d936d04980a1e843c346fba6.jpg)

![05f09bf181d44f49fa6086f965e8ea4e5927a652d528e8b50848ce50c5b864aa.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/05f09bf181d44f49fa6086f965e8ea4e5927a652d528e8b50848ce50c5b864aa.jpg)

![0b5032660be5af91188f4e6e4d901d709c4a909119d56b299f0b32e52cf07c07.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/0b5032660be5af91188f4e6e4d901d709c4a909119d56b299f0b32e52cf07c07.jpg)

![0bca9e5b00f37d5ade57278fa4bf93cfbfb09118214ed33795abf1ddb674cc20.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/0bca9e5b00f37d5ade57278fa4bf93cfbfb09118214ed33795abf1ddb674cc20.jpg)

![0e88999c200a5066dadc7e4d8487660314d5df552b62aa149dda9c211eff21c5.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/0e88999c200a5066dadc7e4d8487660314d5df552b62aa149dda9c211eff21c5.jpg)

![0f6041f14b82c2b3b87f3cf5d2c119418c4d0156a26abe253957fd5acdd4685b.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/0f6041f14b82c2b3b87f3cf5d2c119418c4d0156a26abe253957fd5acdd4685b.jpg)

![1d2038998556865a25b884404f9e2649d02c329de181f7c7d3b4715dd1333c92.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/1d2038998556865a25b884404f9e2649d02c329de181f7c7d3b4715dd1333c92.jpg)

![2036904606cde45ed6e86e9b199f607bf9a194ef5b6d73010d28223c1747697e.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/2036904606cde45ed6e86e9b199f607bf9a194ef5b6d73010d28223c1747697e.jpg)

![2100594db01608736adeec562fef8ecca0735c2adc37871efd5cac96123bae42.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/2100594db01608736adeec562fef8ecca0735c2adc37871efd5cac96123bae42.jpg)

![21aeb1f353d26b7c2b78f3af30a453ff3335c7ee97aa523aa95d0dc60a33a423.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/21aeb1f353d26b7c2b78f3af30a453ff3335c7ee97aa523aa95d0dc60a33a423.jpg)

![23f865bc6829217c91c0b30f3ae15ec201e78db7970e1c27c2a5a0390430c570.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/23f865bc6829217c91c0b30f3ae15ec201e78db7970e1c27c2a5a0390430c570.jpg)

![2547d6addfff9325c166b7ca8883ef54f7ec478ae9efc2840a1b37ffe17d4866.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/2547d6addfff9325c166b7ca8883ef54f7ec478ae9efc2840a1b37ffe17d4866.jpg)

![2e4b7a87382d82d72b6f9353821cbd6e7cf6048a0ed8cdb624f22e2f17c45df9.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/2e4b7a87382d82d72b6f9353821cbd6e7cf6048a0ed8cdb624f22e2f17c45df9.jpg)

![30ed0d10fd7d1ca9bbcc69bc610e8f09d903661aea8f110c561f1c7892cd2617.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/30ed0d10fd7d1ca9bbcc69bc610e8f09d903661aea8f110c561f1c7892cd2617.jpg)

![332d79fac82c468f791295d388b03ef21c74f4fd7d95329a5ecf5c40d2c9809c.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/332d79fac82c468f791295d388b03ef21c74f4fd7d95329a5ecf5c40d2c9809c.jpg)

![33f24725375230e3e50e906113e3f3ad77ce65f9bfad6e6c92a37ca512227cf9.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/33f24725375230e3e50e906113e3f3ad77ce65f9bfad6e6c92a37ca512227cf9.jpg)

![35fd3794fd5ff44cc5793bab876405ac150815d4acda065b876fe09f053a1359.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/35fd3794fd5ff44cc5793bab876405ac150815d4acda065b876fe09f053a1359.jpg)

![39d75cdc194b7c3682a98b757e44c562d99902547cfaeb3bdd65d59875a7f467.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/39d75cdc194b7c3682a98b757e44c562d99902547cfaeb3bdd65d59875a7f467.jpg)

![4644a461e853dbcd0cdb95cdfebcaa97a3d02e83ee12452c6d7fa874698cc734.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/4644a461e853dbcd0cdb95cdfebcaa97a3d02e83ee12452c6d7fa874698cc734.jpg)

![4725b9f369b13d030d14895220bf85521212627eaea9d4f436a5bd8b78ceaa2f.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/4725b9f369b13d030d14895220bf85521212627eaea9d4f436a5bd8b78ceaa2f.jpg)

![4dbe1cfd345f689edf7edee1e815d3b14f01b63b8423229a5a5111e3c7cc7e28.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/4dbe1cfd345f689edf7edee1e815d3b14f01b63b8423229a5a5111e3c7cc7e28.jpg)

![4e32099a86e5f95d854f2ae72051c91f868814637240ad89b4682acd1d7edfa4.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/4e32099a86e5f95d854f2ae72051c91f868814637240ad89b4682acd1d7edfa4.jpg)

![53585cc8084eea556d1216a5c80621f1c4ec5f6f72093160dbf3f9bfba7fced2.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/53585cc8084eea556d1216a5c80621f1c4ec5f6f72093160dbf3f9bfba7fced2.jpg)

![5374b0a879f4615e09a8138396147dbbff8ba4269f71332bdaeb6104463cc0b5.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/5374b0a879f4615e09a8138396147dbbff8ba4269f71332bdaeb6104463cc0b5.jpg)

![596263460f09d81a6d8b0620b54b841e1c4bb6fb22ff13131f91ded5da7b8d9f.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/596263460f09d81a6d8b0620b54b841e1c4bb6fb22ff13131f91ded5da7b8d9f.jpg)

![64eae0bd021dbd4e330e5da12fe2df98008b4532cc612bb8638d1ba92b17ddd7.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/64eae0bd021dbd4e330e5da12fe2df98008b4532cc612bb8638d1ba92b17ddd7.jpg)

![65a5612e020ca86d3d2b395e934e2ba34bab9986d10e1fdea5cad1085512b38c.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/65a5612e020ca86d3d2b395e934e2ba34bab9986d10e1fdea5cad1085512b38c.jpg)

![706d2d3ae9734ba764d7510f658ab8069118a3422f720a9f264df4a479e67104.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/706d2d3ae9734ba764d7510f658ab8069118a3422f720a9f264df4a479e67104.jpg)

![753e12a9549dfef12b4fc19013f77475930a8ae9f5dba76c7bf815ed9dfe5677.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/753e12a9549dfef12b4fc19013f77475930a8ae9f5dba76c7bf815ed9dfe5677.jpg)

![77418a4357cbdc15877b80fda6c534d62a87d93fd71605d36ba9c22f3e5f4f28.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/77418a4357cbdc15877b80fda6c534d62a87d93fd71605d36ba9c22f3e5f4f28.jpg)

![7e0cc1c7224a72424d244e5a60ca2c0e6092bd61e1276fad1d304595ea3698b9.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/7e0cc1c7224a72424d244e5a60ca2c0e6092bd61e1276fad1d304595ea3698b9.jpg)

![7e1de90a8eaeaa411b80172d43775604853c4851488e111812e9892460a9d681.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/7e1de90a8eaeaa411b80172d43775604853c4851488e111812e9892460a9d681.jpg)

![7fb5a660836f487cc8a3818f0f8d2dc6c31818637012ece8e903eb4062f40a80.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/7fb5a660836f487cc8a3818f0f8d2dc6c31818637012ece8e903eb4062f40a80.jpg)

![83268d3561cec5b68a55c335114d0613b8846a0c6a3008256a0d2281f282c481.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/83268d3561cec5b68a55c335114d0613b8846a0c6a3008256a0d2281f282c481.jpg)

![9179ad7ef4bec4a2fc3c4012e5efe85ec3a9ae40a711bf3a97721c11b958b501.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/9179ad7ef4bec4a2fc3c4012e5efe85ec3a9ae40a711bf3a97721c11b958b501.jpg)

![9567843213d20c35b08439112f60285223105887f28debc6c45390190dd13f01.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/9567843213d20c35b08439112f60285223105887f28debc6c45390190dd13f01.jpg)

![985e9294355198badd8644b0557884ff8cf76a76619dd0f3c25a1203693630ba.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/985e9294355198badd8644b0557884ff8cf76a76619dd0f3c25a1203693630ba.jpg)

![987d0071d6f861bbef0c2d7f43f6cb9a152417a2a30291e38a69cba6ad01521a.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/987d0071d6f861bbef0c2d7f43f6cb9a152417a2a30291e38a69cba6ad01521a.jpg)

![9d3c95565bd55dc5ee689b971c2f5ec1b6e9dec614d20bdb036a5552f3b36835.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/9d3c95565bd55dc5ee689b971c2f5ec1b6e9dec614d20bdb036a5552f3b36835.jpg)

![9f0d4d17481dbcd65094967520cfcf3f934792c66d5e7ebf2883168792ed61b1.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/9f0d4d17481dbcd65094967520cfcf3f934792c66d5e7ebf2883168792ed61b1.jpg)

![a0b102ce72d752b486dbdd32e763bb55e7956034a174bfe44cd712d4977cfb30.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/a0b102ce72d752b486dbdd32e763bb55e7956034a174bfe44cd712d4977cfb30.jpg)

![a83967df0380a3a92d20297e6594f5b03b05d583e801f92a38dbc951de25d3e2.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/a83967df0380a3a92d20297e6594f5b03b05d583e801f92a38dbc951de25d3e2.jpg)

![ab1fbf655b8dfd0d2ddf7e0f12c91a7fff95e327d649ca06a4189afbe18bd276.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/ab1fbf655b8dfd0d2ddf7e0f12c91a7fff95e327d649ca06a4189afbe18bd276.jpg)

![ab225f27312ccdca111dba40210b4e0e28d7d675910063a96b59a58f4b6e7f81.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/ab225f27312ccdca111dba40210b4e0e28d7d675910063a96b59a58f4b6e7f81.jpg)

![af17cb901f575bc809d543bea3540d12259c14eb618e266bd876c56652c03345.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/af17cb901f575bc809d543bea3540d12259c14eb618e266bd876c56652c03345.jpg)

![b0f1fe75eabc9b76812499be4504ee5dccf1fa4c966677ef046c9d768f8acac3.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b0f1fe75eabc9b76812499be4504ee5dccf1fa4c966677ef046c9d768f8acac3.jpg)

![b2825c15aef019c08c271ddc0e5fb598092edecbdc85114eec23144d2365d4f4.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b2825c15aef019c08c271ddc0e5fb598092edecbdc85114eec23144d2365d4f4.jpg)

![b6ed7000df9afefd4a876b882605fe4cf127f52234a1540905321bbde20850b9.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b6ed7000df9afefd4a876b882605fe4cf127f52234a1540905321bbde20850b9.jpg)

![b7ab915799134c91b4cf556fb49573db124ebe165c337b7e8831580e940a7f29.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b7ab915799134c91b4cf556fb49573db124ebe165c337b7e8831580e940a7f29.jpg)

![b896cab657c7c7d5d09b270f0f8e634983fe591fbf79130966ec0dc51d03de46.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b896cab657c7c7d5d09b270f0f8e634983fe591fbf79130966ec0dc51d03de46.jpg)

![b958082f78e78da35feab870fe2b8d068aa74930cba1ace87b6eb5198d07eccd.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b958082f78e78da35feab870fe2b8d068aa74930cba1ace87b6eb5198d07eccd.jpg)

![b9f70ea0b96fa8e71f1c385330e2f9fb427364f2291dcc58c24143b10305c07e.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/b9f70ea0b96fa8e71f1c385330e2f9fb427364f2291dcc58c24143b10305c07e.jpg)

![c1d4a033e4134f99350f37e154d4e6d249bc0874da69175cf0a0e7d7055aa823.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/c1d4a033e4134f99350f37e154d4e6d249bc0874da69175cf0a0e7d7055aa823.jpg)

![c70ad029e6878f3211cc0a902c1727e6efb9701bb196d94b3684ebc44d31921c.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/c70ad029e6878f3211cc0a902c1727e6efb9701bb196d94b3684ebc44d31921c.jpg)

![cbc1b984792a608ae5ec072792e1d32b82cf553c4dd19c4aedf106ca73edafca.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/cbc1b984792a608ae5ec072792e1d32b82cf553c4dd19c4aedf106ca73edafca.jpg)

![cd80f21a86e6ff5aa00b20bb2b1c5b5a2e92a37922db9635d8e889b27ea7ad48.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/cd80f21a86e6ff5aa00b20bb2b1c5b5a2e92a37922db9635d8e889b27ea7ad48.jpg)

![d3545ccc1b830ac645d20ea3b405763419951af9070de5a80d2e68b390010bd2.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/d3545ccc1b830ac645d20ea3b405763419951af9070de5a80d2e68b390010bd2.jpg)

![d4b8c2c9ac0cfec941d7d1366e734791df359bad0f274b08ac7b8ff288708995.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/d4b8c2c9ac0cfec941d7d1366e734791df359bad0f274b08ac7b8ff288708995.jpg)

![dab81b6487bc871dad5747f368922d3d3f193f0e1aa74bba00f96f7f75488c93.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/dab81b6487bc871dad5747f368922d3d3f193f0e1aa74bba00f96f7f75488c93.jpg)

![dc72138b1cb708db3329a58ce09145fa6d7a4e0b6c8734b9e987b93eccfb100c.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/dc72138b1cb708db3329a58ce09145fa6d7a4e0b6c8734b9e987b93eccfb100c.jpg)

![dce60900eef1ab8d3b5d788d2b5dd6142e4afe423c5011bbd5fc394827ed23fc.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/dce60900eef1ab8d3b5d788d2b5dd6142e4afe423c5011bbd5fc394827ed23fc.jpg)

![df7ec9b0dcbab252475d377fd68339fc978aaeeb1b5075aba3337a87971286e7.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/df7ec9b0dcbab252475d377fd68339fc978aaeeb1b5075aba3337a87971286e7.jpg)

![e38a34cdef733713dd6abdefeaa07439c2f5d6fbbac9b64e31d1716d4153e3e0.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/e38a34cdef733713dd6abdefeaa07439c2f5d6fbbac9b64e31d1716d4153e3e0.jpg)

![e40063c6a2f662b6df26b14230ea85a44094808f612098d0ac681e08fee7a81a.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/e40063c6a2f662b6df26b14230ea85a44094808f612098d0ac681e08fee7a81a.jpg)

![e611844ac22160ae32fdddd09dedaa9f0d3fde24e51972fa7f8e59a05f15dd7a.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/e611844ac22160ae32fdddd09dedaa9f0d3fde24e51972fa7f8e59a05f15dd7a.jpg)

![e7be41c8489cff28d9ce0710e7fb870eab2f24c6ab41dabfc221acbb7dfd9f07.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/e7be41c8489cff28d9ce0710e7fb870eab2f24c6ab41dabfc221acbb7dfd9f07.jpg)

![e803b97f99cc2e2434088fbc0975bdc3a4b1e8fd97b4e258255acb3e04c65732.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/e803b97f99cc2e2434088fbc0975bdc3a4b1e8fd97b4e258255acb3e04c65732.jpg)

![e933bb24792e07c1c5ff853d3b408dcee71328f8d821966cdaa53290a860b83c.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/e933bb24792e07c1c5ff853d3b408dcee71328f8d821966cdaa53290a860b83c.jpg)

![ec6240b834db78e268542397a9427cbc368965b0c912e2bf36538c0f87495849.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/ec6240b834db78e268542397a9427cbc368965b0c912e2bf36538c0f87495849.jpg)

![fb49b239f9932a5e8b676dfae7e52d9fb2dec3afa65701f9002b4e5f0b6e3088.jpg](../icml_results/120_On%20Efficient%20Scaling%20of%20GNNs%20via%20IO-Aware%20Layers%20Implementations/tables/fb49b239f9932a5e8b676dfae7e52d9fb2dec3afa65701f9002b4e5f0b6e3088.jpg)

<a id="paper-0119"></a>

## Rethinking LLM Ensembling from the Perspective of Mixture Models


### Images

![00ff7528c9064e124f46a244021b5ca113b67f4559921918a4eb715f1128bbc8.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/images/00ff7528c9064e124f46a244021b5ca113b67f4559921918a4eb715f1128bbc8.jpg)

![10c6db6817764af442d3d0e93301de1ba508c7e1225a59b2ba1edba443c502da.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/images/10c6db6817764af442d3d0e93301de1ba508c7e1225a59b2ba1edba443c502da.jpg)

![479f35839232d5bc5334ca3a9fd0d96e14b217690d737aa93874dc15a264588e.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/images/479f35839232d5bc5334ca3a9fd0d96e14b217690d737aa93874dc15a264588e.jpg)

![d8b753b0d118a9ca8f53e60ef4757dbd6ebb1037680bdc7a0b6508eeabbf95ad.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/images/d8b753b0d118a9ca8f53e60ef4757dbd6ebb1037680bdc7a0b6508eeabbf95ad.jpg)

![e0cf96a3fa96def78d3b5138e595cdc7d843e8d4ec98077ddcb188df0f5b92e5.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/images/e0cf96a3fa96def78d3b5138e595cdc7d843e8d4ec98077ddcb188df0f5b92e5.jpg)

### Tables

![44a48a6032661245555841e2170e66fc26de4d49d207d7f3772d952a3731c6a2.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/tables/44a48a6032661245555841e2170e66fc26de4d49d207d7f3772d952a3731c6a2.jpg)

![628653c7fcfdb3d05e1a8a201f2eaf782b4840b370bdc85e79e9aa6c8bae33df.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/tables/628653c7fcfdb3d05e1a8a201f2eaf782b4840b370bdc85e79e9aa6c8bae33df.jpg)

![6fb403bfc6fe4fc0c3f7d0c01e4b54d30fe785f5d3679ae61675bf1ad4e3691a.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/tables/6fb403bfc6fe4fc0c3f7d0c01e4b54d30fe785f5d3679ae61675bf1ad4e3691a.jpg)

![7aa1471494df1231871bbc1e917d554fed801898a23873f87c0afe63deaac4b4.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/tables/7aa1471494df1231871bbc1e917d554fed801898a23873f87c0afe63deaac4b4.jpg)

![c4309cd2e8331cf87e34511f83400d01d579446caac4f42c7f9dcdbfafd4a51a.jpg](../icml_results/121_Rethinking%20LLM%20Ensembling%20from%20the%20Perspective%20of%20Mixture%20Models/tables/c4309cd2e8331cf87e34511f83400d01d579446caac4f42c7f9dcdbfafd4a51a.jpg)

<a id="paper-0120"></a>

## VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding


### Images

![0c2bfa44195e5fa4f3cc4e0d6797d1cf4ab794bec23c3e30f51583780ba6e9ee.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/0c2bfa44195e5fa4f3cc4e0d6797d1cf4ab794bec23c3e30f51583780ba6e9ee.jpg)

![1a3913ef6a506589f5dd669cc8e655212cc135971b3d3c4ebf1a8c2424d40db2.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/1a3913ef6a506589f5dd669cc8e655212cc135971b3d3c4ebf1a8c2424d40db2.jpg)

![4ecb488e042dee35b2437928dd50c6f770a87f4e472d054f43133e8754716b95.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/4ecb488e042dee35b2437928dd50c6f770a87f4e472d054f43133e8754716b95.jpg)

![738f1c9cc07be30f4ebd65db945aa6b37dc57f14d180b38572606c95b41063f9.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/738f1c9cc07be30f4ebd65db945aa6b37dc57f14d180b38572606c95b41063f9.jpg)

![924f5d2550c639762ac7c4032438e1df460edf92b150b6da011ac1fdb68ebda2.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/924f5d2550c639762ac7c4032438e1df460edf92b150b6da011ac1fdb68ebda2.jpg)

![b79f5fa67126db50763dc4df0c276b54c8cb843799ef65bf603f24b9db947b63.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/b79f5fa67126db50763dc4df0c276b54c8cb843799ef65bf603f24b9db947b63.jpg)

![b7db02df1b1f65c36cefe2d1a5b05dc3134ee50bafb26bed07685c557b7d9673.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/b7db02df1b1f65c36cefe2d1a5b05dc3134ee50bafb26bed07685c557b7d9673.jpg)

![c9c75178e6d119cadfb2535eeaeee3f7ad4f9548be934cc59de530aeddd25807.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/c9c75178e6d119cadfb2535eeaeee3f7ad4f9548be934cc59de530aeddd25807.jpg)

![dd7f1f080c0dac3621efe97c1df7c7025ead593e8824b5f586172f0b7210870a.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/dd7f1f080c0dac3621efe97c1df7c7025ead593e8824b5f586172f0b7210870a.jpg)

![e7febf47733ec6abc989b37656ecccead88af1d015ade7a42e418a47ff266e88.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/e7febf47733ec6abc989b37656ecccead88af1d015ade7a42e418a47ff266e88.jpg)

![fbd20d5d0522b36130f9a9ea3981e2cb985a64ab480c8c73f175d10cc439c883.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/images/fbd20d5d0522b36130f9a9ea3981e2cb985a64ab480c8c73f175d10cc439c883.jpg)

### Tables

![0077a227ec16a88ca31d979306c4dc9bbdb3929f743029933fdb8a126f4def10.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/0077a227ec16a88ca31d979306c4dc9bbdb3929f743029933fdb8a126f4def10.jpg)

![07f5cdde536a221b75836a9fe5ef0c575843216b8dbe3e27c912e1642016d086.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/07f5cdde536a221b75836a9fe5ef0c575843216b8dbe3e27c912e1642016d086.jpg)

![1e9aa77cd5dfc40b9501ea52d7be7c5742597fcbb47bec157b0740d6569f05a8.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/1e9aa77cd5dfc40b9501ea52d7be7c5742597fcbb47bec157b0740d6569f05a8.jpg)

![38798a95d6c2bc98b1330473abb37b44e2d319983eb8b790ea7243710d51bfd0.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/38798a95d6c2bc98b1330473abb37b44e2d319983eb8b790ea7243710d51bfd0.jpg)

![3ed1c0dc87b8c78ac64b96b0788c163a645a0535b2abc1a85c26ac978ce5329b.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/3ed1c0dc87b8c78ac64b96b0788c163a645a0535b2abc1a85c26ac978ce5329b.jpg)

![64be993a38dbe1427e0f6b1f60bba939636f5d7eb9f0e08a4bbb745d7c91acf4.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/64be993a38dbe1427e0f6b1f60bba939636f5d7eb9f0e08a4bbb745d7c91acf4.jpg)

![6857b32b78a5ece9a8f44ec0145a1d8c33ba5d77a99af131d975a8d7133745d8.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/6857b32b78a5ece9a8f44ec0145a1d8c33ba5d77a99af131d975a8d7133745d8.jpg)

![7b2513c198cf70ec276c2d8d23e74962a91f4387d6776e8344b3bb79747f913c.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/7b2513c198cf70ec276c2d8d23e74962a91f4387d6776e8344b3bb79747f913c.jpg)

![8b16b7f4162b12d1dd2e31a8fb0231a5ba9276038ad99af5c2d2035bc1857884.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/8b16b7f4162b12d1dd2e31a8fb0231a5ba9276038ad99af5c2d2035bc1857884.jpg)

![93ee3ed2bd3e8ea97857dc70d65590cbe1ef5cd533b797707fcacf32943093e3.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/93ee3ed2bd3e8ea97857dc70d65590cbe1ef5cd533b797707fcacf32943093e3.jpg)

![d0fcb5f303fb228855d7c8cea56ac6acc21a1aa23471a64021314e9b50cf3232.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/d0fcb5f303fb228855d7c8cea56ac6acc21a1aa23471a64021314e9b50cf3232.jpg)

![ef0a870563cb39fd2e5de817f4ff5d5b9e065844bfa86abc44a36fb9597fe0b4.jpg](../icml_results/122_VideoKR_%20Towards%20Knowledge-%20and%20Reasoning-Intensive%20Video%20Understanding/tables/ef0a870563cb39fd2e5de817f4ff5d5b9e065844bfa86abc44a36fb9597fe0b4.jpg)

<a id="paper-0121"></a>

## Self-Soupervision: Cooking Model Soups without Labels


### Images

![53b96d81f4b92edb123f17f74c4fd9b7e32606870fb97a21c504ae152a72cd07.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/images/53b96d81f4b92edb123f17f74c4fd9b7e32606870fb97a21c504ae152a72cd07.jpg)

![578f31e5236b6e2e074a235144879389fe32aec75065190d7f1f6f13ab88ee10.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/images/578f31e5236b6e2e074a235144879389fe32aec75065190d7f1f6f13ab88ee10.jpg)

![b28ad2dc4b02cffb610b489b74a220eca8314588984cf59c1721e9fdf408f0fa.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/images/b28ad2dc4b02cffb610b489b74a220eca8314588984cf59c1721e9fdf408f0fa.jpg)

![d0906bb64d41a5b1f1d23ebf368dc85af521c54b831c0c3686be648dadcdd84e.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/images/d0906bb64d41a5b1f1d23ebf368dc85af521c54b831c0c3686be648dadcdd84e.jpg)

![e9577fd578abfced55347c3c8e0ee3c99ab4cff37fd81d537292a6d19036197d.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/images/e9577fd578abfced55347c3c8e0ee3c99ab4cff37fd81d537292a6d19036197d.jpg)

### Tables

![0b12944622303d4241e8ffada2e69b4bff534679b711a20d356df56a518550b4.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/0b12944622303d4241e8ffada2e69b4bff534679b711a20d356df56a518550b4.jpg)

![15286e90d97b0b99e2a1dc46c4ba43db6d6da7d974e8b87aa4e0aea18c71a0c7.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/15286e90d97b0b99e2a1dc46c4ba43db6d6da7d974e8b87aa4e0aea18c71a0c7.jpg)

![1984b61327005ad5aea1d57a97b7606f695bed8e89373d4449685a4ead21fc72.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/1984b61327005ad5aea1d57a97b7606f695bed8e89373d4449685a4ead21fc72.jpg)

![42d093e5ec63036f6ae5adb859afdaa0e99633191e3cb43081d82df4b66d562d.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/42d093e5ec63036f6ae5adb859afdaa0e99633191e3cb43081d82df4b66d562d.jpg)

![49ff5c8fe0151707304621ec5de284538d38105e2b097a22f5a1a79d9f83ea4b.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/49ff5c8fe0151707304621ec5de284538d38105e2b097a22f5a1a79d9f83ea4b.jpg)

![55281b365e061f2883787c903e49ccc346c13e4ce1badf32022fec4eebcbf45a.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/55281b365e061f2883787c903e49ccc346c13e4ce1badf32022fec4eebcbf45a.jpg)

![5d474069fb68887638bb19efeb800733fe74a9e0ed64aa849a141fca0dcc169c.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/5d474069fb68887638bb19efeb800733fe74a9e0ed64aa849a141fca0dcc169c.jpg)

![6c9fd7d0bcfe357532a5ab65b981bb2df5611f9599d2680e676f9d6c0415beb3.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/6c9fd7d0bcfe357532a5ab65b981bb2df5611f9599d2680e676f9d6c0415beb3.jpg)

![cf5af33a388771ce59a24b2ad9faa3813dd20570e0eab55030b1ac887d7813b7.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/cf5af33a388771ce59a24b2ad9faa3813dd20570e0eab55030b1ac887d7813b7.jpg)

![eddb7946b807e360c2b90616a19f9f56620120914c4f08fbb2bd5eea4057408d.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/eddb7946b807e360c2b90616a19f9f56620120914c4f08fbb2bd5eea4057408d.jpg)

![f01d3a832bd3e4120d6303025563e6e8102643ebd31d057f68e73def577e7e78.jpg](../icml_results/123_Self-Soupervision_%20Cooking%20Model%20Soups%20without%20Labels/tables/f01d3a832bd3e4120d6303025563e6e8102643ebd31d057f68e73def577e7e78.jpg)

<a id="paper-0122"></a>

## Demystifying Entropy Control in LLM RL Training: Theoretical Analysis and Dynamic Scheduling


### Images

![0efcfb62b55a5082a99b1b341afd6d7a01f6376c0898e3f456b8e08cdd07e4ea.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/0efcfb62b55a5082a99b1b341afd6d7a01f6376c0898e3f456b8e08cdd07e4ea.jpg)

![60d9acafb747175b04411e357efb983a3ac7c306f72e231f94ed026d847f5555.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/60d9acafb747175b04411e357efb983a3ac7c306f72e231f94ed026d847f5555.jpg)

![775b6b971b8b716e41d75942bcbb555fab21f716c304637716b29ab52dddaa82.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/775b6b971b8b716e41d75942bcbb555fab21f716c304637716b29ab52dddaa82.jpg)

![8b9f75f6dba02350bce412b7e3a37841715592f09eae721f6bcb63f3431d5de7.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/8b9f75f6dba02350bce412b7e3a37841715592f09eae721f6bcb63f3431d5de7.jpg)

![94b3fb2234f1e9610b910f4b1ca85044fdf3e8c2f04c6d41aaad1d7d604f5dbe.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/94b3fb2234f1e9610b910f4b1ca85044fdf3e8c2f04c6d41aaad1d7d604f5dbe.jpg)

![c2605a856961335356c14513f1259858d2070a85fdd73adb985d3f0187c7c934.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/c2605a856961335356c14513f1259858d2070a85fdd73adb985d3f0187c7c934.jpg)

![e0423fdc7f36db7da8fea2679d946f25ff934aade3c61f2b2af6610b068fae5e.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/e0423fdc7f36db7da8fea2679d946f25ff934aade3c61f2b2af6610b068fae5e.jpg)

![f31bd46b4e52270e91eb8d39f6b0c5f5615731f8f1429e87b66a9db0179c28ee.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/images/f31bd46b4e52270e91eb8d39f6b0c5f5615731f8f1429e87b66a9db0179c28ee.jpg)

### Tables

![1b5b487723bd9f15a9c7591fab92976e654311efce05555f625c7ab5efd09f2b.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/tables/1b5b487723bd9f15a9c7591fab92976e654311efce05555f625c7ab5efd09f2b.jpg)

![59bd20ad0e4a94b203d183470d139467e2189e8647607d8f7c1f277d96138a74.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/tables/59bd20ad0e4a94b203d183470d139467e2189e8647607d8f7c1f277d96138a74.jpg)

![9ddd1c3b823bf985c9d3201cd192e05d790b6ddf64d094a25ceedd900b5798b1.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/tables/9ddd1c3b823bf985c9d3201cd192e05d790b6ddf64d094a25ceedd900b5798b1.jpg)

![a10d493b340497e6a3cecd2272017fa8e746a7692ad77fe809dee6afa04a4940.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/tables/a10d493b340497e6a3cecd2272017fa8e746a7692ad77fe809dee6afa04a4940.jpg)

![be44cd50ede27417a69ccae87420832eaa56dfcd49b20fe6dbba61a6749492ec.jpg](../icml_results/124_Demystifying%20Entropy%20Control%20in%20LLM%20RL%20Training_%20Theoretical%20Analysis%20and%20Dynamic%20Scheduling/tables/be44cd50ede27417a69ccae87420832eaa56dfcd49b20fe6dbba61a6749492ec.jpg)

<a id="paper-0123"></a>

## Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights


### Images

![05363ec9048b385a8482b559732a05809167a9b0a2547310c7002237265227de.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/05363ec9048b385a8482b559732a05809167a9b0a2547310c7002237265227de.jpg)

![0f5b6e5754343b26a5cbe09115c18c8de48f7d497d754ab4dde77d75f391f5c4.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/0f5b6e5754343b26a5cbe09115c18c8de48f7d497d754ab4dde77d75f391f5c4.jpg)

![25101c935fa95ba2d9fbfd18ccd90e817dbe124834ae144eb951b3b9496b022f.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/25101c935fa95ba2d9fbfd18ccd90e817dbe124834ae144eb951b3b9496b022f.jpg)

![2d0818042d5bde486c05af911910beab0e6066f66a0f2a3952632ac36b5a678b.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/2d0818042d5bde486c05af911910beab0e6066f66a0f2a3952632ac36b5a678b.jpg)

![404bd2134d435d1b39c570bcf7203875f780323c53b9d591ed5445b328abb62d.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/404bd2134d435d1b39c570bcf7203875f780323c53b9d591ed5445b328abb62d.jpg)

![4097f7e5ef62852a588a0d5954acb1fa6b3a995e8957f9eee912c2789813532d.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/4097f7e5ef62852a588a0d5954acb1fa6b3a995e8957f9eee912c2789813532d.jpg)

![42d0db3c5fa761c99bf6d08b84cc4e6d92c80e316a59e415de0e48742590cde2.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/42d0db3c5fa761c99bf6d08b84cc4e6d92c80e316a59e415de0e48742590cde2.jpg)

![456a35b441d3177d8099f3e80dc2c28faf62f358b91e94a3c349a7f5e45f9315.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/456a35b441d3177d8099f3e80dc2c28faf62f358b91e94a3c349a7f5e45f9315.jpg)

![5dd746a58de6d66cc80d8ca3f644ece406a5f6f68bd85c76f1acb62b5751ad95.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/5dd746a58de6d66cc80d8ca3f644ece406a5f6f68bd85c76f1acb62b5751ad95.jpg)

![6360645fd7962d521fe265e1ad2060b906c7cd8e396db0d098fda41bd2c32b4d.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/6360645fd7962d521fe265e1ad2060b906c7cd8e396db0d098fda41bd2c32b4d.jpg)

![73e0c62aadb32c4fb01fd48b7e91179d10b58885030076adf560c7c0843c954d.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/73e0c62aadb32c4fb01fd48b7e91179d10b58885030076adf560c7c0843c954d.jpg)

![9c2b0de2eea28341af2b5829620667ab1f81b0bbab2c238581f33eb251174bc4.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/9c2b0de2eea28341af2b5829620667ab1f81b0bbab2c238581f33eb251174bc4.jpg)

![b011bdfb1246fe244a7fc15e719571bda9e4ed7b556fb7351f17b1442d5d5a7f.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/b011bdfb1246fe244a7fc15e719571bda9e4ed7b556fb7351f17b1442d5d5a7f.jpg)

![c0692611cbff22b66fb5d01b36ec59e498d68f2d78759df0c364a15cc8533e94.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/c0692611cbff22b66fb5d01b36ec59e498d68f2d78759df0c364a15cc8533e94.jpg)

![d2ab44fd5ac0191198f41364a78712a03c29fae8685f990b634904d6e47e1889.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/d2ab44fd5ac0191198f41364a78712a03c29fae8685f990b634904d6e47e1889.jpg)

![e4852c29ac2fc5071030152911eac12b82cd1c538b2a0c90c67c75dfdfe3edfb.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/e4852c29ac2fc5071030152911eac12b82cd1c538b2a0c90c67c75dfdfe3edfb.jpg)

![f2eb92b093f19ea8c0da5f00ab19750d0c217151bb6b37349eda95170173e674.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/f2eb92b093f19ea8c0da5f00ab19750d0c217151bb6b37349eda95170173e674.jpg)

![fefb4565c5a01c1400291704ace9225ca50b976c5336c2f8d8951cfdbb3f852f.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/images/fefb4565c5a01c1400291704ace9225ca50b976c5336c2f8d8951cfdbb3f852f.jpg)

### Tables

![3bfbe2137b402c1b2dfc1bef609bbc8c10f312e6fbbb64b28daf67783d86ebc6.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/tables/3bfbe2137b402c1b2dfc1bef609bbc8c10f312e6fbbb64b28daf67783d86ebc6.jpg)

![4dde058c0de50be9063d22e5e06c97d67c95aea04406becc2ceaeff39ca530d1.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/tables/4dde058c0de50be9063d22e5e06c97d67c95aea04406becc2ceaeff39ca530d1.jpg)

![9669ab96e9faf06276201b6c09afc3f7b06fe6f6c313f7595ae3acf9af0514ae.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/tables/9669ab96e9faf06276201b6c09afc3f7b06fe6f6c313f7595ae3acf9af0514ae.jpg)

![fa729e3320f3b4d21479ff1f224f00071a7a7277c67cb1433b96c55946f2cff5.jpg](../icml_results/125_Neural%20Thickets_%20Diverse%20Task%20Experts%20Are%20Dense%20Around%20Pretrained%20Weights/tables/fa729e3320f3b4d21479ff1f224f00071a7a7277c67cb1433b96c55946f2cff5.jpg)

<a id="paper-0124"></a>

## EntroKV: Entropy-Guided Dynamic Budget Allocation for KV-Cache Compression


### Images

![2dee834f7b0bb4a316b9a99e54e1427f4d403fc4fb64a1626d9bdc2a9dadff62.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/2dee834f7b0bb4a316b9a99e54e1427f4d403fc4fb64a1626d9bdc2a9dadff62.jpg)

![3a3736693ca88347f1b65b3e162217a082da17d63bc4786e301c2497fa837fec.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/3a3736693ca88347f1b65b3e162217a082da17d63bc4786e301c2497fa837fec.jpg)

![504362b0259a967638e91a3fe7d0df67b20d904176d2b8398dcd6ca1773b2704.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/504362b0259a967638e91a3fe7d0df67b20d904176d2b8398dcd6ca1773b2704.jpg)

![665e76ea7fd930ff45af9a09c07f39b9f771c1d05efbdd1e8d1dab4b2b437b33.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/665e76ea7fd930ff45af9a09c07f39b9f771c1d05efbdd1e8d1dab4b2b437b33.jpg)

![998699065baa67efbb759018ab6d639a9f4afc5eea1a7494409d0059dc138161.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/998699065baa67efbb759018ab6d639a9f4afc5eea1a7494409d0059dc138161.jpg)

![bd6a005602187a31bdf8a7088666c7b96aa6eb1bb8cb32b98ffc3a97b7cccb3f.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/bd6a005602187a31bdf8a7088666c7b96aa6eb1bb8cb32b98ffc3a97b7cccb3f.jpg)

![cde7de21cedfb6e8f88a4d20525ecca65888617418e51bad4d70f89bccebee47.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/cde7de21cedfb6e8f88a4d20525ecca65888617418e51bad4d70f89bccebee47.jpg)

![dcb79e3c7267656d0f1ed2c83366d9b657ce842ddc65201b5ee4701580f5ff19.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/dcb79e3c7267656d0f1ed2c83366d9b657ce842ddc65201b5ee4701580f5ff19.jpg)

![e5990853cdff1007294a1908ce7e94f968d56446c104126cce4bc01f63ada331.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/images/e5990853cdff1007294a1908ce7e94f968d56446c104126cce4bc01f63ada331.jpg)

### Tables

![07566a71da295a70356dd38183f292320e557254ac9c9e28ba8c13ff31936b45.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/07566a71da295a70356dd38183f292320e557254ac9c9e28ba8c13ff31936b45.jpg)

![0afd7eb968e01721232b03e52c52570b0f796ce0dbb8c3f6ef36fea777e7c82f.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/0afd7eb968e01721232b03e52c52570b0f796ce0dbb8c3f6ef36fea777e7c82f.jpg)

![1756698dbcb297db74597c06e0a7a8808b3c35cbf3fd47658cc34420a4f2bb39.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/1756698dbcb297db74597c06e0a7a8808b3c35cbf3fd47658cc34420a4f2bb39.jpg)

![18a11e92d695fe091d75e22a1f532b620e3c5239d775335da1502f155eea3c6f.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/18a11e92d695fe091d75e22a1f532b620e3c5239d775335da1502f155eea3c6f.jpg)

![269a68ece2c5b9a2f32182615209fdabadb972c0806df50f14c51858d217bee8.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/269a68ece2c5b9a2f32182615209fdabadb972c0806df50f14c51858d217bee8.jpg)

![31cdb1496c9468540feaa90d59a081314bccc7bafe4358f654feb5ef1c8cef74.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/31cdb1496c9468540feaa90d59a081314bccc7bafe4358f654feb5ef1c8cef74.jpg)

![4237baedb2990ee5e4e47b2e21a163fe8d5b2c7e2be4f27df2160d57ba44734f.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/4237baedb2990ee5e4e47b2e21a163fe8d5b2c7e2be4f27df2160d57ba44734f.jpg)

![4b7c4bb575fb65669662bc94dc686f631d3c8498998aa3ea03c8e4d0d077c79f.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/4b7c4bb575fb65669662bc94dc686f631d3c8498998aa3ea03c8e4d0d077c79f.jpg)

![52a010f1b077afbb13f8eae05640a39111b17403a74dd6fdb7415e60efff54e6.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/52a010f1b077afbb13f8eae05640a39111b17403a74dd6fdb7415e60efff54e6.jpg)

![6620ca0e104846206af0bf648ed07807fbbe9b9739b7f81b3a28fe44a2cd808d.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/6620ca0e104846206af0bf648ed07807fbbe9b9739b7f81b3a28fe44a2cd808d.jpg)

![a047c8a533d4ea2d86010df87c6f6ba0f56f84442b01517f5401cf79c65fa7de.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/a047c8a533d4ea2d86010df87c6f6ba0f56f84442b01517f5401cf79c65fa7de.jpg)

![bbd465581c90af1e8a264a092d22a63be4a19e290a1d0a6abcbf8e54aec8f6b6.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/bbd465581c90af1e8a264a092d22a63be4a19e290a1d0a6abcbf8e54aec8f6b6.jpg)

![c5c2c802b80c0fb6982441b43bd4f16c14ea6c8b5fc7ad6c4216d5ce931b2f82.jpg](../icml_results/126_EntroKV_%20Entropy-Guided%20Dynamic%20Budget%20Allocation%20for%20KV-Cache%20Compression/tables/c5c2c802b80c0fb6982441b43bd4f16c14ea6c8b5fc7ad6c4216d5ce931b2f82.jpg)

<a id="paper-0125"></a>

## Excited Pfaffians: Generalized Neural Wave Functions Across Structure and State


### Images

![08c941b386f3b4e84c0a1e3b1803eba05b087672e19f6ffef70efeba191f9aec.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/08c941b386f3b4e84c0a1e3b1803eba05b087672e19f6ffef70efeba191f9aec.jpg)

![14b458e4075a9250c750913c19677955a9f9453736f1715038cbcab38aa2ccfe.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/14b458e4075a9250c750913c19677955a9f9453736f1715038cbcab38aa2ccfe.jpg)

![35b9f3faeb6b72fa7807e66a99b0f0c5d2d4705684f8fa156dc8db71b46f4a2e.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/35b9f3faeb6b72fa7807e66a99b0f0c5d2d4705684f8fa156dc8db71b46f4a2e.jpg)

![509f7851f3f937befc33771ea72c0538a9dbf81125eff209a0ff1fbd59fe008d.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/509f7851f3f937befc33771ea72c0538a9dbf81125eff209a0ff1fbd59fe008d.jpg)

![571cbdb39fd01894ae7f29c9d873a92a49aceda4a0d00beff6e5d1ecff5ce942.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/571cbdb39fd01894ae7f29c9d873a92a49aceda4a0d00beff6e5d1ecff5ce942.jpg)

![6192615828de798d6b9438d7d36581680cb1a8043bbdc75defd952752bb092bb.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/6192615828de798d6b9438d7d36581680cb1a8043bbdc75defd952752bb092bb.jpg)

![62f43a9f9a3a65d5b0334f08b203a33379af1c5f771a5dbbda0e385078c6a445.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/62f43a9f9a3a65d5b0334f08b203a33379af1c5f771a5dbbda0e385078c6a445.jpg)

![74310e66b0fa5c38bf2f63c8af40a356aa992ea08bf856b482803c22db462a06.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/74310e66b0fa5c38bf2f63c8af40a356aa992ea08bf856b482803c22db462a06.jpg)

![9ffa67b65d1db521207b56623ef6ff7b81d433192425d95fe0418a7907777e10.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/9ffa67b65d1db521207b56623ef6ff7b81d433192425d95fe0418a7907777e10.jpg)

![a923ed928a64fa904b5c952b7a53fd14893551e546a853da87629322ad928abb.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/a923ed928a64fa904b5c952b7a53fd14893551e546a853da87629322ad928abb.jpg)

![b364101969c900d8f6b4c1ed100ec934f0ae6cebc35d6742ccd49abb39abaed6.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/b364101969c900d8f6b4c1ed100ec934f0ae6cebc35d6742ccd49abb39abaed6.jpg)

![b73cd45ad9c7f348b01fa16ab4d12a8f282eacfe2d1f1cb6c9f56c98d91449f8.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/b73cd45ad9c7f348b01fa16ab4d12a8f282eacfe2d1f1cb6c9f56c98d91449f8.jpg)

![c77c3128c8988aa1ca9399d70effc37828c1edaacd4509e282e45cfc6e472d5c.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/c77c3128c8988aa1ca9399d70effc37828c1edaacd4509e282e45cfc6e472d5c.jpg)

![c88fada3370735802c11919d5ad395a68e2453e066a447869792d9cda7a10609.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/c88fada3370735802c11919d5ad395a68e2453e066a447869792d9cda7a10609.jpg)

![cb6a3f7d76d56084dcfd0727c29c7cb3c8469dff254d96d10a1156774a924292.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/cb6a3f7d76d56084dcfd0727c29c7cb3c8469dff254d96d10a1156774a924292.jpg)

![dad9d01ce9dc7a20062e7525e0f596f3e782ba8bbbc0a9e1fb1d82a62c5a514a.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/dad9d01ce9dc7a20062e7525e0f596f3e782ba8bbbc0a9e1fb1d82a62c5a514a.jpg)

![f16615231dce465153bf23da58ee37d2acccb627c5f7e0cb58cdcc3d1157b993.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/images/f16615231dce465153bf23da58ee37d2acccb627c5f7e0cb58cdcc3d1157b993.jpg)

### Tables

![02bda85d870469fd9abd72c0f1863179c31ddf13139020815489c105dca88f9e.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/02bda85d870469fd9abd72c0f1863179c31ddf13139020815489c105dca88f9e.jpg)

![0428fb2de8c8f308e3ca5b5daa0b283b5eb49e303659569511338b097278868c.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/0428fb2de8c8f308e3ca5b5daa0b283b5eb49e303659569511338b097278868c.jpg)

![12a240b606b2e394be718e99521ddcd1ba3d6ff291cf121148df298ebe276688.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/12a240b606b2e394be718e99521ddcd1ba3d6ff291cf121148df298ebe276688.jpg)

![18c8c21bba414b047d389011f0824170e3954f18378d7d6e681e7fa2f6025de8.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/18c8c21bba414b047d389011f0824170e3954f18378d7d6e681e7fa2f6025de8.jpg)

![28e87f3c12e99fb052b31969bb6eb87491388ddcd77ef05ff19c166fccfd36f4.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/28e87f3c12e99fb052b31969bb6eb87491388ddcd77ef05ff19c166fccfd36f4.jpg)

![47ea77d7d45380b36d31f7598fb5ab296ec460cb457deecbe85e154f3100d46b.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/47ea77d7d45380b36d31f7598fb5ab296ec460cb457deecbe85e154f3100d46b.jpg)

![4e5a9b004d952ff5989def8b5d9f150a835a94887bb7e5d48710fb28342e8ef0.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/4e5a9b004d952ff5989def8b5d9f150a835a94887bb7e5d48710fb28342e8ef0.jpg)

![53bedd222c6721cd1f517338a541f1404dba9b2096b56bbf813417f3846fd786.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/53bedd222c6721cd1f517338a541f1404dba9b2096b56bbf813417f3846fd786.jpg)

![7ff77c041858013bc6c42e7f88b03370d8073c0e86429f1972492cc053f819c7.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/7ff77c041858013bc6c42e7f88b03370d8073c0e86429f1972492cc053f819c7.jpg)

![844fb6e29f5ac64a6a7306ee524a4278a54e965e34ab164422dac68d93760283.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/844fb6e29f5ac64a6a7306ee524a4278a54e965e34ab164422dac68d93760283.jpg)

![8a317adbec0f67554ea57e84ebacd123450ac5ad59ddce7cd5db6561cd510ba2.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/8a317adbec0f67554ea57e84ebacd123450ac5ad59ddce7cd5db6561cd510ba2.jpg)

![a069f6edf55b3db28ee2cf80c79cb83dea45cc1564aeef7a4099b01fc7974602.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/a069f6edf55b3db28ee2cf80c79cb83dea45cc1564aeef7a4099b01fc7974602.jpg)

![bb2aa17db23e0c54f21b49b34bf7fe5a72a5891cb592b975be311e58dde15bb9.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/bb2aa17db23e0c54f21b49b34bf7fe5a72a5891cb592b975be311e58dde15bb9.jpg)

![bdf37f914504a0c93d052c731ff4b87608a1b1fb33feb6cb49de48b6c0b90ac3.jpg](../icml_results/127_Excited%20Pfaffians_%20Generalized%20Neural%20Wave%20Functions%20Across%20Structure%20and%20State/tables/bdf37f914504a0c93d052c731ff4b87608a1b1fb33feb6cb49de48b6c0b90ac3.jpg)

<a id="paper-0126"></a>

## EEmo-Logic: A Unified Dataset and Multi-Stage Framework for Comprehensive Image-Evoked Emotion Assessment


### Images

![01b05aa1037c7c1810ae88e71c168419205e51bec7fb82162fa95b8c3849f032.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/01b05aa1037c7c1810ae88e71c168419205e51bec7fb82162fa95b8c3849f032.jpg)

![0e0919442f424d8cf384395a7beac9e50b9cd56164aa46910446ff1701361da3.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/0e0919442f424d8cf384395a7beac9e50b9cd56164aa46910446ff1701361da3.jpg)

![128e2a833eede9b30c865331b59eff4149b4a185f9493ee35a55294998f6d8a1.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/128e2a833eede9b30c865331b59eff4149b4a185f9493ee35a55294998f6d8a1.jpg)

![1d3b8fbbee722bcef4ee43039d99bf74200363c1fa57f2e9efc0bf5af0b7592a.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/1d3b8fbbee722bcef4ee43039d99bf74200363c1fa57f2e9efc0bf5af0b7592a.jpg)

![3263af1fdea96c73eacaf1ea800da8119d1a98c9f6d4f3e046ab2dbb285beb9d.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/3263af1fdea96c73eacaf1ea800da8119d1a98c9f6d4f3e046ab2dbb285beb9d.jpg)

![50a34d8034f22eba7a744a4fa307315d27dbe44c7391df728a22f2186af1ec3c.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/50a34d8034f22eba7a744a4fa307315d27dbe44c7391df728a22f2186af1ec3c.jpg)

![5aaa07dcb12d24bd5591fbe2caedcf93953f88ef8f8a6ab8dbbebabff71bbfb7.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/5aaa07dcb12d24bd5591fbe2caedcf93953f88ef8f8a6ab8dbbebabff71bbfb7.jpg)

![67e1d625807108df469d51eb53ceee8f6a10c86cd9f0bfeb21c47f4c79e615b4.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/67e1d625807108df469d51eb53ceee8f6a10c86cd9f0bfeb21c47f4c79e615b4.jpg)

![6df95b4190f7765d91ffe4139db7bfe33e0a60b4c625c9c54a4ae13c1bc55077.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/6df95b4190f7765d91ffe4139db7bfe33e0a60b4c625c9c54a4ae13c1bc55077.jpg)

![73aec809c5ecda2e906e86a93319517b1ffa8aa66262d02267b06babea5da067.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/73aec809c5ecda2e906e86a93319517b1ffa8aa66262d02267b06babea5da067.jpg)

![755f8272f8d0700b8219367fb98e01b094ff1e53b773b5ae015ec4edd40322d1.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/755f8272f8d0700b8219367fb98e01b094ff1e53b773b5ae015ec4edd40322d1.jpg)

![86feac9e827d7419545a64e805acce39d13c75c7d583b79cd281b22d58b73882.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/86feac9e827d7419545a64e805acce39d13c75c7d583b79cd281b22d58b73882.jpg)

![98aa188ab4e090df3f3ce4cc4c2568d7c0fb0d4f2e393457f02d7de1b1db710d.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/98aa188ab4e090df3f3ce4cc4c2568d7c0fb0d4f2e393457f02d7de1b1db710d.jpg)

![ac592ae5b5996e43cb80775e531a011152669d51ecd1e5470c2dbee1850b5edd.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/ac592ae5b5996e43cb80775e531a011152669d51ecd1e5470c2dbee1850b5edd.jpg)

![bef50959ed90d0107ea2dbb0ec9bd239177cf4c39d14edb5aecc0ae5c39a1698.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/bef50959ed90d0107ea2dbb0ec9bd239177cf4c39d14edb5aecc0ae5c39a1698.jpg)

![c0b00729cbd042a02072cb877cdd36400885c4ecd9fbb86bc57c2f31f1facba0.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/c0b00729cbd042a02072cb877cdd36400885c4ecd9fbb86bc57c2f31f1facba0.jpg)

![c1e2a96b4f530bbc93be8ccca906a7261aebc8b734af4b2101a5d094f90fc694.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/c1e2a96b4f530bbc93be8ccca906a7261aebc8b734af4b2101a5d094f90fc694.jpg)

![c34662b2046f672c1588bf77cf333f5153186a0fba7af6feff7148537ed58841.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/c34662b2046f672c1588bf77cf333f5153186a0fba7af6feff7148537ed58841.jpg)

![cffd02d6090faa0020781d9a88c4455d2f5b399f766afa0e098107f90060298d.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/cffd02d6090faa0020781d9a88c4455d2f5b399f766afa0e098107f90060298d.jpg)

![d4dcd717d9ea06a971a61a73e7892bd88d7d546ad181a1beb5e42f9b123b5f35.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/d4dcd717d9ea06a971a61a73e7892bd88d7d546ad181a1beb5e42f9b123b5f35.jpg)

![e9f423fd425ee5fe7951b8a68f59cc81d904418f31d7ba51f56ff36af1bd53aa.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/e9f423fd425ee5fe7951b8a68f59cc81d904418f31d7ba51f56ff36af1bd53aa.jpg)

![ea6598c6f792398c2cd05675a7e15de46af80961bccf7854eefa03ce9a65eb43.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/ea6598c6f792398c2cd05675a7e15de46af80961bccf7854eefa03ce9a65eb43.jpg)

![eb50519fdbc2924db5ed7baa876fbb36382257902d295736889ee50b0a5ef93d.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/eb50519fdbc2924db5ed7baa876fbb36382257902d295736889ee50b0a5ef93d.jpg)

![efc6c54d3c094f009ce08b39163036a187fbff3d4630fabe82d08816e5fa6671.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/efc6c54d3c094f009ce08b39163036a187fbff3d4630fabe82d08816e5fa6671.jpg)

![f6d8b78eabc95d3b14fafbcfc4d9dc8bac9afefc6e21dcaee4e22cbd33547a64.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/images/f6d8b78eabc95d3b14fafbcfc4d9dc8bac9afefc6e21dcaee4e22cbd33547a64.jpg)

### Tables

![0001a4350ba4bbb699e628f157f18568040eb664a076b885167993069367bb8c.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/0001a4350ba4bbb699e628f157f18568040eb664a076b885167993069367bb8c.jpg)

![059810743459558ee2099be839d275137fc19188e28fb7ab5a5f4d1b2e60622e.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/059810743459558ee2099be839d275137fc19188e28fb7ab5a5f4d1b2e60622e.jpg)

![1c3b2e7ed07231e307528c00b92206a41e58397b87eeb03de05d1559b39cc42a.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/1c3b2e7ed07231e307528c00b92206a41e58397b87eeb03de05d1559b39cc42a.jpg)

![27837e4c8836c9915e1a084a2d1c1850ba4bb4e3d122e5f5121000527eb5bfde.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/27837e4c8836c9915e1a084a2d1c1850ba4bb4e3d122e5f5121000527eb5bfde.jpg)

![2eff3ee66034cfe7084022534f37f4727c036b996917ecbaf4e3e30a4859b379.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/2eff3ee66034cfe7084022534f37f4727c036b996917ecbaf4e3e30a4859b379.jpg)

![341f502ee67f667a5da7e29f5907fb584e6876c8584187fe62b4caa3ddd30fe8.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/341f502ee67f667a5da7e29f5907fb584e6876c8584187fe62b4caa3ddd30fe8.jpg)

![358fe40b0c7be0874b7ee38ac1b849043aefa5952ec5d01a567c7333c9acd94d.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/358fe40b0c7be0874b7ee38ac1b849043aefa5952ec5d01a567c7333c9acd94d.jpg)

![4facd999c445376c82a26f17ea3db2e792f1e34655f904b23a43fe932cb1f475.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/4facd999c445376c82a26f17ea3db2e792f1e34655f904b23a43fe932cb1f475.jpg)

![692d49580d9f614d8e77355e38c1c76075a8620bd6525b4f357e437d9a628de2.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/692d49580d9f614d8e77355e38c1c76075a8620bd6525b4f357e437d9a628de2.jpg)

![87a96afe14b0ae3a905336200d28b59f224f710cea27cb750f1584fe332ee4a3.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/87a96afe14b0ae3a905336200d28b59f224f710cea27cb750f1584fe332ee4a3.jpg)

![87b19f1e021ba3859042a7fe42119d6d4e31df6c7ac0b713deaba1e50effdeea.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/87b19f1e021ba3859042a7fe42119d6d4e31df6c7ac0b713deaba1e50effdeea.jpg)

![90a812e3b000aca02ae7b7a212cdc7479c216ed454d628f2ba28d198d3a628f3.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/90a812e3b000aca02ae7b7a212cdc7479c216ed454d628f2ba28d198d3a628f3.jpg)

![9508b81efaf5b6da7333c7df841f09ce3ab4d686e27a1d9b746a7b2adc85ad77.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/9508b81efaf5b6da7333c7df841f09ce3ab4d686e27a1d9b746a7b2adc85ad77.jpg)

![ad5e394571e897e34beb1d1fed99e129792b63eaba2e808b4f81f9d000c51de1.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/ad5e394571e897e34beb1d1fed99e129792b63eaba2e808b4f81f9d000c51de1.jpg)

![c0ac75e63a8f651723575fbbc1be5af4d31a30b772a154f0d0b10618c2fbe40b.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/c0ac75e63a8f651723575fbbc1be5af4d31a30b772a154f0d0b10618c2fbe40b.jpg)

![ccdb72a7101ad60ac9215dfeceb4a10bfbf1cc1419c0bbb15e37d9a74c3046b4.jpg](../icml_results/128_EEmo-Logic_%20A%20Unified%20Dataset%20and%20Multi-Stage%20Framework%20for%20Comprehensive%20Image-Evoked%20Emotion%20Asses/tables/ccdb72a7101ad60ac9215dfeceb4a10bfbf1cc1419c0bbb15e37d9a74c3046b4.jpg)

<a id="paper-0127"></a>

## Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence


### Images

![03b1fa3c8ea7488886e7b47f0a1a710104041c24f866e29f01fd728607287594.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/03b1fa3c8ea7488886e7b47f0a1a710104041c24f866e29f01fd728607287594.jpg)

![1134bfcf32b0fe5ec0b52a69d34dd8f5b14225ed0d1e3d521112e7691ea944a3.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/1134bfcf32b0fe5ec0b52a69d34dd8f5b14225ed0d1e3d521112e7691ea944a3.jpg)

![2cc9dd8a02549bb8d96521e847b9d27de8ce491a54d0e2efb600c7f8bb4db50d.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/2cc9dd8a02549bb8d96521e847b9d27de8ce491a54d0e2efb600c7f8bb4db50d.jpg)

![36f132e9c968fea6d6a55f5b2b33248f84dc0247cd43bd748d9aaeb604dc44b7.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/36f132e9c968fea6d6a55f5b2b33248f84dc0247cd43bd748d9aaeb604dc44b7.jpg)

![3c1cee1baa1c3dc42d5fc2a9c22373f7805aefe72992a3fd7d7a1b8dba960183.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/3c1cee1baa1c3dc42d5fc2a9c22373f7805aefe72992a3fd7d7a1b8dba960183.jpg)

![454aadbda6091b72527bcc2427dc5931fcff3fd2ad734cc589f5524b7483e2b1.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/454aadbda6091b72527bcc2427dc5931fcff3fd2ad734cc589f5524b7483e2b1.jpg)

![4821e3f6b4cd7e36041d876e04742019a809c40df4768397317ac27647084453.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/4821e3f6b4cd7e36041d876e04742019a809c40df4768397317ac27647084453.jpg)

![48f9c2fdf1840c91670b4813b023e8f3ccaaf199b6f058525ed06ec520289bd2.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/48f9c2fdf1840c91670b4813b023e8f3ccaaf199b6f058525ed06ec520289bd2.jpg)

![56c53855845ec95e7cad2685bdb4a50bd025ee816dab853f55045c755702ada0.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/56c53855845ec95e7cad2685bdb4a50bd025ee816dab853f55045c755702ada0.jpg)

![8120d4a3334a5ff9c343450d8218e137f4d29d32c9c4bd9dbd52e12be2573fa3.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/8120d4a3334a5ff9c343450d8218e137f4d29d32c9c4bd9dbd52e12be2573fa3.jpg)

![8c8e0a1381e87e05a180678507c9027272debe5b96de81be453fb244a980ae53.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/8c8e0a1381e87e05a180678507c9027272debe5b96de81be453fb244a980ae53.jpg)

![b8c9844bf6ecfaed9b5800d5bc0a5f0f3e223b65b432b950372418697d999063.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/b8c9844bf6ecfaed9b5800d5bc0a5f0f3e223b65b432b950372418697d999063.jpg)

![c280688f2c27bc43390c3d108f3cfce4adacc9819a9d5c67f37690c5237774fc.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/c280688f2c27bc43390c3d108f3cfce4adacc9819a9d5c67f37690c5237774fc.jpg)

![d814672e9714691a9cbb95ffd2365748876613542c8edec23767a80a06c5091b.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/d814672e9714691a9cbb95ffd2365748876613542c8edec23767a80a06c5091b.jpg)

![dbef854bd7d328cdb01ce53088075a9dd40d5a4c3a64f2703b5e7e6272718da5.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/images/dbef854bd7d328cdb01ce53088075a9dd40d5a4c3a64f2703b5e7e6272718da5.jpg)

### Tables

![04b6c0836e2715895bd9e58083b47c165d9f47bc77df24ba56c3457bdb2e4c60.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/tables/04b6c0836e2715895bd9e58083b47c165d9f47bc77df24ba56c3457bdb2e4c60.jpg)

![648088c1fbb4b505154c9b2122226f08dc0fc818054fdc1f1e36a0e8f0d4832b.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/tables/648088c1fbb4b505154c9b2122226f08dc0fc818054fdc1f1e36a0e8f0d4832b.jpg)

![9f4ccdea3d636637538485e1e64bba82281a7040679f191108176a50c5edbdfd.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/tables/9f4ccdea3d636637538485e1e64bba82281a7040679f191108176a50c5edbdfd.jpg)

![a37226a508132c46a2a9ecf972a79f7e4d2e4ef299c397fe061fab7d4dc18c10.jpg](../icml_results/129_Holi-Spatial_%20Evolving%20Video%20Streams%20into%20Holistic%203D%20Spatial%20Intelligence/tables/a37226a508132c46a2a9ecf972a79f7e4d2e4ef299c397fe061fab7d4dc18c10.jpg)

<a id="paper-0128"></a>

## DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos


### Images

![01da204304b9fc48418ce7e5d82ab0799aeac14281fdc32cd9d14355bbc0c9be.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/01da204304b9fc48418ce7e5d82ab0799aeac14281fdc32cd9d14355bbc0c9be.jpg)

![0fd690a1a5e2a1849698c55093e3331b7f7cc3bb43df5f9c803942d7761c06d2.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/0fd690a1a5e2a1849698c55093e3331b7f7cc3bb43df5f9c803942d7761c06d2.jpg)

![2fcb9a9dbc54461307346bf6d0aa35c29c7f9a6173bbc46e4796b5a0dd98b1fc.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/2fcb9a9dbc54461307346bf6d0aa35c29c7f9a6173bbc46e4796b5a0dd98b1fc.jpg)

![30434328146598615d62ce0c7f41670e86dbc8fcb713456ad4a635c16bc35fa3.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/30434328146598615d62ce0c7f41670e86dbc8fcb713456ad4a635c16bc35fa3.jpg)

![3760a8f5b13120a81f93fcbe3b9c07792a5d3d834ac8eaffb9ada272f53e6cb3.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/3760a8f5b13120a81f93fcbe3b9c07792a5d3d834ac8eaffb9ada272f53e6cb3.jpg)

![3f2a0507edfd70b8c50823754ae0c94c109232517b7b6ae83e68cd8e7cda8e75.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/3f2a0507edfd70b8c50823754ae0c94c109232517b7b6ae83e68cd8e7cda8e75.jpg)

![45f30f15782017e58ac36af24f617f3d8d093f713e62262b407e654918f3e7dd.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/45f30f15782017e58ac36af24f617f3d8d093f713e62262b407e654918f3e7dd.jpg)

![682e067e039910160bb2ec76e906369b25128a9c731fa57e33402c3f0b345de9.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/682e067e039910160bb2ec76e906369b25128a9c731fa57e33402c3f0b345de9.jpg)

![9118863276900e009371e292e0962601461ef78bc8c242fa3f21a0992519d63c.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/9118863276900e009371e292e0962601461ef78bc8c242fa3f21a0992519d63c.jpg)

![9856df24d8b3264461db2b96d879fbc303a05943e3e4217baacfc62132f9acf3.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/9856df24d8b3264461db2b96d879fbc303a05943e3e4217baacfc62132f9acf3.jpg)

![9b09c2bb3c68cdc7c719ae1130f3298e9ff802a042618c831bf1dbfec9f3a343.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/9b09c2bb3c68cdc7c719ae1130f3298e9ff802a042618c831bf1dbfec9f3a343.jpg)

![c225f80499c44e186f40791def190d24e40b886379eafc776ea6fe43c84705c6.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/c225f80499c44e186f40791def190d24e40b886379eafc776ea6fe43c84705c6.jpg)

![e06784c822c6aa7d341be6f38c6113042db457b2dceb2d689ab02354f0f34819.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/e06784c822c6aa7d341be6f38c6113042db457b2dceb2d689ab02354f0f34819.jpg)

![f26bcaca2c05ecc56d9edf6e905cdc89fdb5f5753e2177a3d9e3454af60673d2.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/f26bcaca2c05ecc56d9edf6e905cdc89fdb5f5753e2177a3d9e3454af60673d2.jpg)

![fc546e81918dcfbd5dbefdca6d08c99fd989330c2deabc128dd7c89f550c9b4c.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/images/fc546e81918dcfbd5dbefdca6d08c99fd989330c2deabc128dd7c89f550c9b4c.jpg)

### Tables

![148b19d4a0376b82250b16f8142c6c20b1dcceac92a8f583fa63d8ad474ff616.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/148b19d4a0376b82250b16f8142c6c20b1dcceac92a8f583fa63d8ad474ff616.jpg)

![16d021785272fa1d6b1c7db4b20af2e3a71ea1c0cda4dfde8dd7855c1a2cea17.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/16d021785272fa1d6b1c7db4b20af2e3a71ea1c0cda4dfde8dd7855c1a2cea17.jpg)

![246199c74e03567d26d11061b366bac013396468443ae8fc738c01cec2215958.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/246199c74e03567d26d11061b366bac013396468443ae8fc738c01cec2215958.jpg)

![370d7650b17e6a90cdbb0099bfd12830cad1522ccdee45e28c423375ef61c406.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/370d7650b17e6a90cdbb0099bfd12830cad1522ccdee45e28c423375ef61c406.jpg)

![4f130c17a4973657502f532c106d3b9331cd4397683b4fa2c1f3c400d4393850.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/4f130c17a4973657502f532c106d3b9331cd4397683b4fa2c1f3c400d4393850.jpg)

![a2580e4b99b2d7a8c360be50cd0373c453f90a4e64e3161e4386f59d0d375a4d.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/a2580e4b99b2d7a8c360be50cd0373c453f90a4e64e3161e4386f59d0d375a4d.jpg)

![d5e3d2f1b48d722f5f1ca0624dd596b1f671b9cfc2c91d82795f004022d95faf.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/d5e3d2f1b48d722f5f1ca0624dd596b1f671b9cfc2c91d82795f004022d95faf.jpg)

![ec90aaafa1894745f5dfca445c0260690490d3d91e5138119d602f65ef3d980a.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/ec90aaafa1894745f5dfca445c0260690490d3d91e5138119d602f65ef3d980a.jpg)

![f80a37196f74e241dcc39a258f85d957a794a6e23c13b3310c1f9775292404f1.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/f80a37196f74e241dcc39a258f85d957a794a6e23c13b3310c1f9775292404f1.jpg)

![f83b18e5bdb7f5719ad846fd8fc7d4b32ff557bfd9549ee4ab2b0e0568ee64ae.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/f83b18e5bdb7f5719ad846fd8fc7d4b32ff557bfd9549ee4ab2b0e0568ee64ae.jpg)

![fd90d095ac607526c00a8f7c9927d84864f94aff59bd7ca460d1639ff7acb7e4.jpg](../icml_results/130_DreamDojo_%20A%20Generalist%20Robot%20World%20Model%20from%20Large-Scale%20Human%20Videos/tables/fd90d095ac607526c00a8f7c9927d84864f94aff59bd7ca460d1639ff7acb7e4.jpg)

<a id="paper-0129"></a>

## Optimal structure learning and conditional independence testing


### Images

![3b62649b6a00b7db387e8dc94bfebf29c84493517e480cd5206166117ffec4f8.jpg](../icml_results/131_Optimal%20structure%20learning%20and%20conditional%20independence%20testing/images/3b62649b6a00b7db387e8dc94bfebf29c84493517e480cd5206166117ffec4f8.jpg)

![ec6752f7eaa87206b499edd5b22ec89584f61d1d59628cad018eeae7186ee3ca.jpg](../icml_results/131_Optimal%20structure%20learning%20and%20conditional%20independence%20testing/images/ec6752f7eaa87206b499edd5b22ec89584f61d1d59628cad018eeae7186ee3ca.jpg)

### Tables

![3bb1d5a452f51edcf27355f85b152bd0737d9a9ee0107cf61ee851f2469c862c.jpg](../icml_results/131_Optimal%20structure%20learning%20and%20conditional%20independence%20testing/tables/3bb1d5a452f51edcf27355f85b152bd0737d9a9ee0107cf61ee851f2469c862c.jpg)

![7fb3cd466489ae96b83337811b6b49e2f667c5a417fbeb2ec95deca3c7909145.jpg](../icml_results/131_Optimal%20structure%20learning%20and%20conditional%20independence%20testing/tables/7fb3cd466489ae96b83337811b6b49e2f667c5a417fbeb2ec95deca3c7909145.jpg)

![c26aae4baac756a92b51e3b53e7c2b51af04236cfbbbc969d8458631911024d6.jpg](../icml_results/131_Optimal%20structure%20learning%20and%20conditional%20independence%20testing/tables/c26aae4baac756a92b51e3b53e7c2b51af04236cfbbbc969d8458631911024d6.jpg)

![ff7107590bd2751fe6f1fc8987d04359ca7658461145075f35c2f790ed0dd760.jpg](../icml_results/131_Optimal%20structure%20learning%20and%20conditional%20independence%20testing/tables/ff7107590bd2751fe6f1fc8987d04359ca7658461145075f35c2f790ed0dd760.jpg)

<a id="paper-0130"></a>

## Biased Generalization in Diffusion Models


### Images

![08a8027aae7932909f3037cfe2cef7a7c889cbfd4fac91ae9e5f1123c199fff6.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/08a8027aae7932909f3037cfe2cef7a7c889cbfd4fac91ae9e5f1123c199fff6.jpg)

![1bf00b1ec59ebbc0ff6a97210b16e3dd7460b7b67e902879eaed5bc61c6aadb2.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/1bf00b1ec59ebbc0ff6a97210b16e3dd7460b7b67e902879eaed5bc61c6aadb2.jpg)

![1e7177da40b7da0a2604f2e2708d155eb12e49af2d9452c5ec214d45921ff7cf.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/1e7177da40b7da0a2604f2e2708d155eb12e49af2d9452c5ec214d45921ff7cf.jpg)

![2bd50a1e4e3d3f1443b74db7aab8c586bf6dcd5919729958559f3779ae1e9401.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/2bd50a1e4e3d3f1443b74db7aab8c586bf6dcd5919729958559f3779ae1e9401.jpg)

![9042f71370ce2f636a6ec7d110d07e0cdb441bfd2c413d83b333b1ca67b6e7ab.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/9042f71370ce2f636a6ec7d110d07e0cdb441bfd2c413d83b333b1ca67b6e7ab.jpg)

![9992fddfbcf02afcc364e1ec2b09eb9bb2d716a5548ff2e9240da402397a90fe.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/9992fddfbcf02afcc364e1ec2b09eb9bb2d716a5548ff2e9240da402397a90fe.jpg)

![ad555e12b6f3ab7372a244c4e7f04293aa32db746b03d1a27845f07764f67c98.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/ad555e12b6f3ab7372a244c4e7f04293aa32db746b03d1a27845f07764f67c98.jpg)

![c05c267d4e982f361fa2fef7d44a75cce14b80db995a5b93daf45669c8b4f83d.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/c05c267d4e982f361fa2fef7d44a75cce14b80db995a5b93daf45669c8b4f83d.jpg)

![c3ac927756197834d69ae24ad9bbf67563324e91fbf6686762b1f4ca9b750b5e.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/c3ac927756197834d69ae24ad9bbf67563324e91fbf6686762b1f4ca9b750b5e.jpg)

![db731ea24e7b49bec7bbebb114d7f747016b6952539757be3903e3fcf308b857.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/db731ea24e7b49bec7bbebb114d7f747016b6952539757be3903e3fcf308b857.jpg)

![e8b25c7846e30afcbf7cc5092e0679f652c5ce72ad687f68d7cbae863501ba95.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/e8b25c7846e30afcbf7cc5092e0679f652c5ce72ad687f68d7cbae863501ba95.jpg)

![ed3e5cdb07293064c3c7a32694af5191683fa1ed99b39b7b2fe36cd31c5c83ed.jpg](../icml_results/132_Biased%20Generalization%20in%20Diffusion%20Models/images/ed3e5cdb07293064c3c7a32694af5191683fa1ed99b39b7b2fe36cd31c5c83ed.jpg)

<a id="paper-0131"></a>

## High-accuracy and dimension-free sampling with diffusions


### Tables

![4d176a7c843a8ed4e6cd6a7cf42e9cc0a07b41ae3501c5d38bafc9eb87e4ce2c.jpg](../icml_results/133_High-accuracy%20and%20dimension-free%20sampling%20with%20diffusions/tables/4d176a7c843a8ed4e6cd6a7cf42e9cc0a07b41ae3501c5d38bafc9eb87e4ce2c.jpg)

![5d61ff63ceb9479d6f0d5b1ddfe8aeef8ec0b3b8c6af94ddb3edb4f087f877ca.jpg](../icml_results/133_High-accuracy%20and%20dimension-free%20sampling%20with%20diffusions/tables/5d61ff63ceb9479d6f0d5b1ddfe8aeef8ec0b3b8c6af94ddb3edb4f087f877ca.jpg)

![ae414068965d987dd8956db788f41aec1399d693631c2652457035a072afa3ab.jpg](../icml_results/133_High-accuracy%20and%20dimension-free%20sampling%20with%20diffusions/tables/ae414068965d987dd8956db788f41aec1399d693631c2652457035a072afa3ab.jpg)

<a id="paper-0132"></a>

## Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models

### Images

![069a88cc161fe64a52e594b4a54d48d3e0afd7dd772dc3a38e3162f50a7fdcfb.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/069a88cc161fe64a52e594b4a54d48d3e0afd7dd772dc3a38e3162f50a7fdcfb.jpg)

![08e014c57af086a84954167ffd952becf42854fee55613a7d0a79589e8600eaf.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/08e014c57af086a84954167ffd952becf42854fee55613a7d0a79589e8600eaf.jpg)

![1378044b0615e3f0687eecc7a61904f0cd6274f05739fc3fbf68c3ca2c3a13cf.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/1378044b0615e3f0687eecc7a61904f0cd6274f05739fc3fbf68c3ca2c3a13cf.jpg)

![197ac6c92ebf89f7eb423ee8ffe918ef261743df06d4fdf8639231cd1adeea89.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/197ac6c92ebf89f7eb423ee8ffe918ef261743df06d4fdf8639231cd1adeea89.jpg)

![2fdaaa91e5fd3a8fd809715b1128942b6bd23534673afa8db5e00286c947874e.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/2fdaaa91e5fd3a8fd809715b1128942b6bd23534673afa8db5e00286c947874e.jpg)

![459bb5f4818e056b6ef6efc35153966d69531f74c1336aa01ad1838f067e808c.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/459bb5f4818e056b6ef6efc35153966d69531f74c1336aa01ad1838f067e808c.jpg)

![4e63cf7b30a945f80159384aa78110ae48cbbc5fda7adc665d5aa9fb26838eae.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/4e63cf7b30a945f80159384aa78110ae48cbbc5fda7adc665d5aa9fb26838eae.jpg)

![51a52c85842163cf270d84cbf4d53bfa0f509b757c78a9cfd37bd4caf173a828.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/51a52c85842163cf270d84cbf4d53bfa0f509b757c78a9cfd37bd4caf173a828.jpg)

![54b744651f57cf2b05d4f0467ecd8a7911a45d4cfc5f349daa45b98ca7a0d7aa.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/54b744651f57cf2b05d4f0467ecd8a7911a45d4cfc5f349daa45b98ca7a0d7aa.jpg)

![88a0f5ee74983ebc2920891f1170b25edc98e9048ebe57b5642eecc4fe9cd430.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/88a0f5ee74983ebc2920891f1170b25edc98e9048ebe57b5642eecc4fe9cd430.jpg)

![bed1ffee8e6a171c8fa3d66f4b9fa8d4821eb98317b038e98453175f91ef7022.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/bed1ffee8e6a171c8fa3d66f4b9fa8d4821eb98317b038e98453175f91ef7022.jpg)

![d0a3db5612323b15dc9230f65ac36a5cfd0e1469894ff0d91d32e4d2d3fa8bce.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/d0a3db5612323b15dc9230f65ac36a5cfd0e1469894ff0d91d32e4d2d3fa8bce.jpg)

![ec61dd16eefc50477d4d46d1d69ba8e89a9f9e8e3aaa2584e7c31de8051c6619.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/ec61dd16eefc50477d4d46d1d69ba8e89a9f9e8e3aaa2584e7c31de8051c6619.jpg)

### Tables

![039e39797d5b9e893d9a011cc821b0c44406c1294d554a0dc26a06b26d52364e.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/tables/039e39797d5b9e893d9a011cc821b0c44406c1294d554a0dc26a06b26d52364e.jpg)

![2591e408d6d8b578d3617f60bce862dd952a707f5049b2c871e79dc68b845461.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/tables/2591e408d6d8b578d3617f60bce862dd952a707f5049b2c871e79dc68b845461.jpg)

![f775155e90bbc40781a1e449ee01f6c15105466424975e4c8d38964e288c2f86.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/tables/f775155e90bbc40781a1e449ee01f6c15105466424975e4c8d38964e288c2f86.jpg)
