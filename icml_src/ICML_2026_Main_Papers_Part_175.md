# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 175 of 197

## 目录 (Table of Contents)

1. [Contrastive Symbolic Regression: Aligned Representations, Adaptive Prediction, and Diverse Ensembles](#paper-5743)
2. [ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking](#paper-5744)
3. [MAPS: Memory-Aware Predictive Scheduling Framework for Large Language Model Serving](#paper-5745)
4. [REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge](#paper-5746)
5. [The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms](#paper-5747)
6. [Rethinking Parameter Sharing as Graph Coloring for Structured Compression](#paper-5748)
7. [PSBench: Editing Image via GUI Agents in Photoshop](#paper-5749)
8. [LoRA-DA: Data-Aware Initialization for Low-Rank Adaptation via Asymptotic Analysis](#paper-5750)
9. [Motion-Residual Conflict-Aware Time Reversal for Generative Inbetweening](#paper-5751)
10. [Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](#paper-5752)
11. [Diff-MN: Diffusion Parameterized MoE-NCDE for Continuous Time Series Generation with Irregular Observations](#paper-5753)
12. [Comp-Attn: Present-and-Align Attention for Compositional Video Generation](#paper-5754)
13. [Dimension-Independent Convergence of Underdamped Langevin Monte Carlo in KL Divergence](#paper-5755)
14. [Benign Overfitting in Adversarial Training for Vision Transformers](#paper-5756)
15. [Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch](#paper-5757)
16. [CoMem: Context Management with A Decoupled Long-Context Model](#paper-5758)
17. [Who Gets Credit or Blame? Attributing Accountability in Modern AI Systems](#paper-5759)
18. [LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models](#paper-5760)
19. [WF-Bench: A Benchmark for Neural-Network WaveFunction Expressivity and Scaling Laws](#paper-5761)
20. [One Tool Is Enough: Reinforcement Learning of LLM Agents for Repository-Level Code Navigation](#paper-5762)
21. [Spatiotemporal Imputation with Graph-Informed Flow Matching](#paper-5763)
22. [SOLAR for Offline MARL: Plateau-Triggered Potential Shaping under World-Model Uncertainty](#paper-5764)
23. [DeepAnalyze: Agentic Large Language Models for Autonomous Data Science](#paper-5765)
24. [CoDA-Bench: Can Code Agents Handle Data-Intensive Tasks?](#paper-5766)
25. [Breaking the Synthetic-Real Domain Shortcut for Training-Free Generative Replay-based Class Incremental Learning](#paper-5767)
26. [What If We Let Forecasting Forget? A Sparse Bottleneck for Cross-Variable Dependencies](#paper-5768)
27. [Self-evolving LLM agents with in-distribution Optimization](#paper-5769)
28. [Geometric Coherence Learning for Structuring Value Functions in Plain MDPs](#paper-5770)
29. [AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection](#paper-5771)
30. [Boosting Monocular Metric Depth Estimation via Bokeh Rendering](#paper-5772)
31. [Higher-Order Certified Robustness for Regression](#paper-5773)
32. [From Noise to Control: Parameterized Diffusion Policies](#paper-5774)
33. [Generalization Bounds for Discrete Diffusion: Statistical Advantage of Masking](#paper-5775)

---


<a id="paper-5743"></a>

## Contrastive Symbolic Regression: Aligned Representations, Adaptive Prediction, and Diverse Ensembles

### Images

![11e7746d558bd0f0eee40295a19764c92c1da2d73f98eb4061e32573bbb2a071.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/11e7746d558bd0f0eee40295a19764c92c1da2d73f98eb4061e32573bbb2a071.jpg)

![3138daf48abdd2260f8075d18bbf22628771b9e26e9e56a6ed310d5409e3f2a8.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/3138daf48abdd2260f8075d18bbf22628771b9e26e9e56a6ed310d5409e3f2a8.jpg)

![5dedc7ac9e2e90e6d32f82e6496f26d74d893c7d3607b6e3310173998b9956ea.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/5dedc7ac9e2e90e6d32f82e6496f26d74d893c7d3607b6e3310173998b9956ea.jpg)

![61a58adfa9d4922d2fe175e9b25096bbd43a25d46ee696c0b7eb6b65a65dc326.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/61a58adfa9d4922d2fe175e9b25096bbd43a25d46ee696c0b7eb6b65a65dc326.jpg)

![b08b6ba1c21c48b128cec3d886cc743ef11de9a947699a056c7a7558c99503dd.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/b08b6ba1c21c48b128cec3d886cc743ef11de9a947699a056c7a7558c99503dd.jpg)

### Tables

![0ea6cbcfbced72bfdcdd862fef7a878599a8c53816355545bd1d278876bb6968.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/0ea6cbcfbced72bfdcdd862fef7a878599a8c53816355545bd1d278876bb6968.jpg)

![165a28f0370f70f9a462fb00f8ac24796e50d30c8c908a7d09b2df8954b121a5.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/165a28f0370f70f9a462fb00f8ac24796e50d30c8c908a7d09b2df8954b121a5.jpg)

![30b364418f895fa46a32cbcb645d14b9656324b35e9cd44680ad04dbd4e6804f.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/30b364418f895fa46a32cbcb645d14b9656324b35e9cd44680ad04dbd4e6804f.jpg)

![41f72596d90387d2c7d7fb45ed4aad8bd5ae1ae1c3aeac25e2ed73463713a461.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/41f72596d90387d2c7d7fb45ed4aad8bd5ae1ae1c3aeac25e2ed73463713a461.jpg)

![45fdf22952a35cb02b411c2eccebbc9fb9188226087cd93298d350aa0cd58b8e.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/45fdf22952a35cb02b411c2eccebbc9fb9188226087cd93298d350aa0cd58b8e.jpg)

![547ce36e3afe12ce359965879e4b61354fa872da9cdecfbb3f876428b04a786b.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/547ce36e3afe12ce359965879e4b61354fa872da9cdecfbb3f876428b04a786b.jpg)

![77a675a8ca9f32eca1af8fbe926582e8ce3004e6429143ac58bfac6b571ada51.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/77a675a8ca9f32eca1af8fbe926582e8ce3004e6429143ac58bfac6b571ada51.jpg)

![85f8f869a0184d5367e15053b1f7dcb64dd80dc6807844b36eb8eb5a15a894c9.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/85f8f869a0184d5367e15053b1f7dcb64dd80dc6807844b36eb8eb5a15a894c9.jpg)

![ae35fc794a291e04de7e4495d5c01b8e4639708da5bc09fc4a89863b90e86469.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/ae35fc794a291e04de7e4495d5c01b8e4639708da5bc09fc4a89863b90e86469.jpg)

![c4123fc395f6091a0719780be8c27cec72f3f860acc59ee0439f07dea7e6f7f6.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/c4123fc395f6091a0719780be8c27cec72f3f860acc59ee0439f07dea7e6f7f6.jpg)

![c6957bc8d72105cf8b855777254da99f098f3d84b3ad6e1e472c9d8645489b95.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/c6957bc8d72105cf8b855777254da99f098f3d84b3ad6e1e472c9d8645489b95.jpg)

![cbbcdda193c9975226c820536b193abb488117d666e0f9e1977dd022970f2e92.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/cbbcdda193c9975226c820536b193abb488117d666e0f9e1977dd022970f2e92.jpg)

![d76c1b6a0ffa958c866113f2a91e83b1d1307d104027c5c05148824ca3f2017c.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/d76c1b6a0ffa958c866113f2a91e83b1d1307d104027c5c05148824ca3f2017c.jpg)

![e33b98c97a12e50cb42302ee27587c21b2f8cd5ee6a6ee180b4149d485eb7ff9.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/e33b98c97a12e50cb42302ee27587c21b2f8cd5ee6a6ee180b4149d485eb7ff9.jpg)

![e3c06791c5a792f9d2c07aa8bdccc4e0d64a9b1e611879ef030eac75b099629a.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/e3c06791c5a792f9d2c07aa8bdccc4e0d64a9b1e611879ef030eac75b099629a.jpg)

![ea505811799570acbd6d8be74c82b273757451de95f74fa665065ff52b32f52f.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/ea505811799570acbd6d8be74c82b273757451de95f74fa665065ff52b32f52f.jpg)

![f26b97c1c885d1f4939a0d04981a07894acf92c6e4eb3be825b1c90bd047946f.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/f26b97c1c885d1f4939a0d04981a07894acf92c6e4eb3be825b1c90bd047946f.jpg)

## Contrastive Symbolic Regression: Aligned Representations, Adaptive Prediction, and Diverse Ensembles


### Images

![152cdc50ec6877df327a6db58071a0b4c45deaae78defc50f78a1a76d00ad427.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/152cdc50ec6877df327a6db58071a0b4c45deaae78defc50f78a1a76d00ad427.jpg)

![213cb3c108d3d06093f40cfc4826c929599ffcfe500f1025c5ba74f8aca228a4.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/213cb3c108d3d06093f40cfc4826c929599ffcfe500f1025c5ba74f8aca228a4.jpg)

![2a12f767b707ad61898bc0c5b0b1197a00daba85b92724a65ea53dc0f6e18fe6.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/2a12f767b707ad61898bc0c5b0b1197a00daba85b92724a65ea53dc0f6e18fe6.jpg)

![4063c43755eacff9f7780c3f5231f9475dd54369a95f9c7f68c08226bca755a2.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/4063c43755eacff9f7780c3f5231f9475dd54369a95f9c7f68c08226bca755a2.jpg)

![553bbc1f25a1ec67aa49ad4f598ffc81221cd97f3956194eb6d74a772d4f5af2.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/553bbc1f25a1ec67aa49ad4f598ffc81221cd97f3956194eb6d74a772d4f5af2.jpg)

![77310b5e07b1e6cbf2086410d98d82460368579d29ef483cb39c45e07c06f241.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/77310b5e07b1e6cbf2086410d98d82460368579d29ef483cb39c45e07c06f241.jpg)

![817d5a044e63ef4e81b1746f64598343cea30ef033006a97f06921fbb242968c.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/817d5a044e63ef4e81b1746f64598343cea30ef033006a97f06921fbb242968c.jpg)

![81acd9140587a6f0f4fe1d8b6db4330211bcb6ec4670f3ed421234d0a3b9da99.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/81acd9140587a6f0f4fe1d8b6db4330211bcb6ec4670f3ed421234d0a3b9da99.jpg)

![87bee9e68e1d79d1f02b58635ca01a244d8ddfcacb730de343aafdbd47f0e582.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/87bee9e68e1d79d1f02b58635ca01a244d8ddfcacb730de343aafdbd47f0e582.jpg)

![8f20d2074275f3070cdd69ec86ba91486d352a863187b50b6bed939db5bbcb08.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/8f20d2074275f3070cdd69ec86ba91486d352a863187b50b6bed939db5bbcb08.jpg)

![a18615a1defd9c374ac1dfdca13e187ba1f6d8509a338ea59a81d3bfa08f3f8b.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/a18615a1defd9c374ac1dfdca13e187ba1f6d8509a338ea59a81d3bfa08f3f8b.jpg)

![b83542397523e516babe2d3a6eed6ba6fa487f7dbc20c0f0e0127f3015e1d3ea.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/b83542397523e516babe2d3a6eed6ba6fa487f7dbc20c0f0e0127f3015e1d3ea.jpg)

![be3a93ac5db45d130c6aee8f8f5004cdb91fa858791cd6b91b85795cf9ca587a.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/be3a93ac5db45d130c6aee8f8f5004cdb91fa858791cd6b91b85795cf9ca587a.jpg)

![d3c30c4250469f917dd135ef271d385f5d4c0f096781d8aa9215fd77592905b5.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/d3c30c4250469f917dd135ef271d385f5d4c0f096781d8aa9215fd77592905b5.jpg)

![e18ef75545107f89cd097536c6e8f5fb454e5d86046c158dd73d21e628cc5f71.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/e18ef75545107f89cd097536c6e8f5fb454e5d86046c158dd73d21e628cc5f71.jpg)

![f2c0227d5119428c16b785acbd73963e440f03913426d6b2acc485f0404b171b.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/f2c0227d5119428c16b785acbd73963e440f03913426d6b2acc485f0404b171b.jpg)

![ffe41d7a84ed9ba42038fb3435f6fb31d004d0f79b507dad78c69471f442f2e4.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/images/ffe41d7a84ed9ba42038fb3435f6fb31d004d0f79b507dad78c69471f442f2e4.jpg)

### Tables

![12046b07c8203fc61045629a981878b095d9354df6a9750a5a3c57ec4d0265ec.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/12046b07c8203fc61045629a981878b095d9354df6a9750a5a3c57ec4d0265ec.jpg)

![3543f8a0ba66bb03ce576a33af8b1928f2e5bc0c0f82e50f4b4f38ef0e3f2816.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/3543f8a0ba66bb03ce576a33af8b1928f2e5bc0c0f82e50f4b4f38ef0e3f2816.jpg)

![7abdea97b14fff9c5cb328183174eba739c3d3910a57e6f5ab0ebbe613f689bb.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/7abdea97b14fff9c5cb328183174eba739c3d3910a57e6f5ab0ebbe613f689bb.jpg)

![7bbc6980faa4ac87f1476110c4c35710df729940832d4ff2e159832ab3ee08b7.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/7bbc6980faa4ac87f1476110c4c35710df729940832d4ff2e159832ab3ee08b7.jpg)

![7f8c31797b78d263a3d5c2f3a6469fc5aca4dbeea72f92cb60ce87c18053f904.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/7f8c31797b78d263a3d5c2f3a6469fc5aca4dbeea72f92cb60ce87c18053f904.jpg)

![bcc7178813b8b2c59765cfce34db4b72f4554a039101551a64f9d80878b4bc64.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/bcc7178813b8b2c59765cfce34db4b72f4554a039101551a64f9d80878b4bc64.jpg)

![e13a50f08b0503f7319a457c5ed3f6722925c275307b1b776cfed2bb409e0f73.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/e13a50f08b0503f7319a457c5ed3f6722925c275307b1b776cfed2bb409e0f73.jpg)

![eba285075439112937b23126885e92c01fdd0675350d7a68024d50c611801d7b.jpg](../icml_results/5792_Contrastive%20Symbolic%20Regression_%20Aligned%20Representations%2C%20Adaptive%20Prediction%2C%20and%20Diverse%20Ensembles/tables/eba285075439112937b23126885e92c01fdd0675350d7a68024d50c611801d7b.jpg)

<a id="paper-5744"></a>

## ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking


### Images

![208f93d6e4a2a13b3a0235d5627e1e194f4d7940b336ae0feb4752224bc57cd7.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/images/208f93d6e4a2a13b3a0235d5627e1e194f4d7940b336ae0feb4752224bc57cd7.jpg)

![4830e6706fb1e763d49c17f68fc45f7b7ce3b15b35abf78e5b7a3d5002637399.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/images/4830e6706fb1e763d49c17f68fc45f7b7ce3b15b35abf78e5b7a3d5002637399.jpg)

![6e3e86d32bd1b83467bda22a8fb1787b6e31abbef42b9868dd6fc63ebde159af.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/images/6e3e86d32bd1b83467bda22a8fb1787b6e31abbef42b9868dd6fc63ebde159af.jpg)

![837ffe4cbc43dc9dc5de8da528f69be0a8c67f058990dba572e274d9e5c63f60.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/images/837ffe4cbc43dc9dc5de8da528f69be0a8c67f058990dba572e274d9e5c63f60.jpg)

![a47bed9d04d21150b218b082250083a8fa582c47f8aec00385a12ece964f3914.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/images/a47bed9d04d21150b218b082250083a8fa582c47f8aec00385a12ece964f3914.jpg)

![a80d17d6e2ccc5fe80634d41ec2e3f74f9b1919121b5f89c061b8916a051e1e0.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/images/a80d17d6e2ccc5fe80634d41ec2e3f74f9b1919121b5f89c061b8916a051e1e0.jpg)

### Tables

![0b68984e1604a920aee24d232d8f4d1d8d27693fea51848953b27665f80b0ae4.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/tables/0b68984e1604a920aee24d232d8f4d1d8d27693fea51848953b27665f80b0ae4.jpg)

![87c97b83f22027d26b0b5b1853b8172ce66adbcdf085d304723983f234924c9e.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/tables/87c97b83f22027d26b0b5b1853b8172ce66adbcdf085d304723983f234924c9e.jpg)

![aa9c521b91092f8923456605051c532a23dfb41853ba18dfba52a0b12be65da6.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/tables/aa9c521b91092f8923456605051c532a23dfb41853ba18dfba52a0b12be65da6.jpg)

![b82f03af433adb979455043302a577b6c0340351454151b9610d9715e9571ae9.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/tables/b82f03af433adb979455043302a577b6c0340351454151b9610d9715e9571ae9.jpg)

![eb6cd64b2fb074716a64ce825201ece66b584db94a0e0ad34a4af93ca1a3ef1c.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/tables/eb6cd64b2fb074716a64ce825201ece66b584db94a0e0ad34a4af93ca1a3ef1c.jpg)

![fc724f3613aebd3b916f5958972f49911950dca21a7ea046884398b76121fd3b.jpg](../icml_results/5793_ArenaRL_%20Scaling%20RL%20for%20Open-Ended%20Agents%20via%20Tournament-based%20Relative%20Ranking/tables/fc724f3613aebd3b916f5958972f49911950dca21a7ea046884398b76121fd3b.jpg)

<a id="paper-5745"></a>

## MAPS: Memory-Aware Predictive Scheduling Framework for Large Language Model Serving


### Images

![09a820db570f7806eeaef7a863343bf8c4fe330f68f8bb142b41492f0c9bb800.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/09a820db570f7806eeaef7a863343bf8c4fe330f68f8bb142b41492f0c9bb800.jpg)

![1d1a06f83beb97798b342aa34d901cf1d532516ffc7c7e24a6d05e8a79693dee.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/1d1a06f83beb97798b342aa34d901cf1d532516ffc7c7e24a6d05e8a79693dee.jpg)

![27e0aebaa59dfd6e8cd80a597bf9bd051f8e063b1f9f334f887cdc4a323d5463.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/27e0aebaa59dfd6e8cd80a597bf9bd051f8e063b1f9f334f887cdc4a323d5463.jpg)

![305d5bb9138d9239e1b820b343dcddb379c1a74631399efbe0892a5f45b012ab.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/305d5bb9138d9239e1b820b343dcddb379c1a74631399efbe0892a5f45b012ab.jpg)

![64648afbd106671d215594b690fe2a26566ab37dbec9068655eb31389cd1e2f3.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/64648afbd106671d215594b690fe2a26566ab37dbec9068655eb31389cd1e2f3.jpg)

![8fb112458fba2cb684575c052a095bb65559dc1ba7aaafb0e5c33c02b6d4933b.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/8fb112458fba2cb684575c052a095bb65559dc1ba7aaafb0e5c33c02b6d4933b.jpg)

![9a164597bc63644c5771d633d927c3f443d9fa860dfe544ab5764f81aa9c6271.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/9a164597bc63644c5771d633d927c3f443d9fa860dfe544ab5764f81aa9c6271.jpg)

![9ec0275a5db3d7635dcd8036af2c0163d953bb5cf5cfdaf3eb032f8a575315ec.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/9ec0275a5db3d7635dcd8036af2c0163d953bb5cf5cfdaf3eb032f8a575315ec.jpg)

![acc07f6453f85ed528d99d3bbb9fa239a6a7436eb1eafefde6c549c9fbfef860.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/acc07f6453f85ed528d99d3bbb9fa239a6a7436eb1eafefde6c549c9fbfef860.jpg)

![dca37f2bf8401e056acf10ffcbb457b8aa16df6667a65af0a18528495bff0bfc.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/dca37f2bf8401e056acf10ffcbb457b8aa16df6667a65af0a18528495bff0bfc.jpg)

![e6944f7cbfe842bab208c42f7ff67c722dceba0ed3e3e193d4a0afca8424abfd.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/e6944f7cbfe842bab208c42f7ff67c722dceba0ed3e3e193d4a0afca8424abfd.jpg)

![eac2fb22da4647fa15f0ce5f72999a33fc4eced16e78887cb19836601492c869.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/eac2fb22da4647fa15f0ce5f72999a33fc4eced16e78887cb19836601492c869.jpg)

![eead257227be8600471aad536c19da9d24f6f82854858eb986501d3d6c8979bd.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/eead257227be8600471aad536c19da9d24f6f82854858eb986501d3d6c8979bd.jpg)

![ffd4265f01c0ffaa41467b96c3d4553cd199e913b0abcb8ab58bd7d9e4f06bb6.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/images/ffd4265f01c0ffaa41467b96c3d4553cd199e913b0abcb8ab58bd7d9e4f06bb6.jpg)

### Tables

![690730b8b47bea1b59f79d8d48ff7948291425fa6b6185ef69466949056234a9.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/tables/690730b8b47bea1b59f79d8d48ff7948291425fa6b6185ef69466949056234a9.jpg)

![c8e4f0d5eec5314ae05a10122f44ec2ae52d5f50858e969df19674fba48feebb.jpg](../icml_results/5794_MAPS_%20Memory-Aware%20Predictive%20Scheduling%20Framework%20for%20Large%20Language%20Model%20Serving/tables/c8e4f0d5eec5314ae05a10122f44ec2ae52d5f50858e969df19674fba48feebb.jpg)

<a id="paper-5746"></a>

## REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge


### Images

![2dca349bf9c17eb6be801c5ea54c4d129ba547038d07258cd6884aa7dcfd7874.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/images/2dca349bf9c17eb6be801c5ea54c4d129ba547038d07258cd6884aa7dcfd7874.jpg)

![678d8e5323670c5e733ca9d83dad7a7187ace825e2d1a55b8d755e38089f030a.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/images/678d8e5323670c5e733ca9d83dad7a7187ace825e2d1a55b8d755e38089f030a.jpg)

![c21d1e6ac67f23dc14d59afb65c6bb153c12f065b4bb16a1449d8d983b47407d.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/images/c21d1e6ac67f23dc14d59afb65c6bb153c12f065b4bb16a1449d8d983b47407d.jpg)

![fd902da0c05cb6bd1ebe563fbcfdda93768bd5c8b7c28c31cf5c3897250f44de.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/images/fd902da0c05cb6bd1ebe563fbcfdda93768bd5c8b7c28c31cf5c3897250f44de.jpg)

### Tables

![1258b30bd2ed90f0d01f67f6d614b9ab78970c6dc3c2025c0c2822265b84c0d8.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/1258b30bd2ed90f0d01f67f6d614b9ab78970c6dc3c2025c0c2822265b84c0d8.jpg)

![2d4acfcce67f38282bf951a6a3e00f6380847b0da81dbd6c22f9d8f1d8eab9ea.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/2d4acfcce67f38282bf951a6a3e00f6380847b0da81dbd6c22f9d8f1d8eab9ea.jpg)

![2f850ea445408670f483fef6f95f6260f6e11347f94d17ca6d1e8fbcef60c0bb.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/2f850ea445408670f483fef6f95f6260f6e11347f94d17ca6d1e8fbcef60c0bb.jpg)

![34444be57617827b13aff70c6c0d1f36fc7601d5a0bf3f873fc74a6f72341c0f.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/34444be57617827b13aff70c6c0d1f36fc7601d5a0bf3f873fc74a6f72341c0f.jpg)

![540852e475b07691f99f4d06c2d8273a8faf80774fe5b562b3c566fc6defb6be.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/540852e475b07691f99f4d06c2d8273a8faf80774fe5b562b3c566fc6defb6be.jpg)

![5fa32c3bfcc8d01fc47bdf6f21900ea4c4bf1b28c99da5a0127c24216d7f3605.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/5fa32c3bfcc8d01fc47bdf6f21900ea4c4bf1b28c99da5a0127c24216d7f3605.jpg)

![81f6b7a81f8040b5de5086c0dde44a41c2582ba7c4898655ef6bc27a6c9cac35.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/81f6b7a81f8040b5de5086c0dde44a41c2582ba7c4898655ef6bc27a6c9cac35.jpg)

![90adbe78aab49c705227f7f544d07b147eafcc6d3b0aee423d9c4cc44a881af6.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/90adbe78aab49c705227f7f544d07b147eafcc6d3b0aee423d9c4cc44a881af6.jpg)

![aaeed9ecd4491e11de38e023f1d032dafc7ead0bcf38f44a0761bb3ab7545fd8.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/aaeed9ecd4491e11de38e023f1d032dafc7ead0bcf38f44a0761bb3ab7545fd8.jpg)

![b8078240404a76dc0762cf8e6b92347c633a5ef4e1dc165d2ede744845a71780.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/b8078240404a76dc0762cf8e6b92347c633a5ef4e1dc165d2ede744845a71780.jpg)

![cbfe1d8b71288c0f8660e82f48227c3733e4418c23f0f04de9f5c87f2e487edd.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/cbfe1d8b71288c0f8660e82f48227c3733e4418c23f0f04de9f5c87f2e487edd.jpg)

![e2f7e426bc7b9fad8c1252f70a22ba3a0b4ff84047dae2c110a052732deaef22.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/e2f7e426bc7b9fad8c1252f70a22ba3a0b4ff84047dae2c110a052732deaef22.jpg)

![e9bd816234177805d00715fc037606652d414a531732feb2bb97b3023b8b3fc6.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/e9bd816234177805d00715fc037606652d414a531732feb2bb97b3023b8b3fc6.jpg)

![f577f0c47581d806e1e117e6899ae4eb2794f02e3c3dc6bb19d94a21bbc70654.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/f577f0c47581d806e1e117e6899ae4eb2794f02e3c3dc6bb19d94a21bbc70654.jpg)

![f8fb01250b234023d87e866e1082c7113e04fb4cbf7f66f9e350822d938ad42c.jpg](../icml_results/5795_REAL_%20Regression-Aware%20Reinforcement%20Learning%20for%20LLM-as-a-Judge/tables/f8fb01250b234023d87e866e1082c7113e04fb4cbf7f66f9e350822d938ad42c.jpg)

<a id="paper-5747"></a>

## The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms


### Images

![285bd53fcaf83e5e177616be14e84ea4e84793d793bd7f7da0f1369eabab5d91.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/285bd53fcaf83e5e177616be14e84ea4e84793d793bd7f7da0f1369eabab5d91.jpg)

![54ccec9139a8445b84cbe4a4978132fd962bdb0ed4bfd6080ed98196c0cd5b70.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/54ccec9139a8445b84cbe4a4978132fd962bdb0ed4bfd6080ed98196c0cd5b70.jpg)

![5789a0d32a24377385bc0b39bfd1d498748f170187956af87daf86b0a0c8d002.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/5789a0d32a24377385bc0b39bfd1d498748f170187956af87daf86b0a0c8d002.jpg)

![6438633b6fb857991204a12f4c6aa44d68abf252a91faff64f67f76b078cb86d.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/6438633b6fb857991204a12f4c6aa44d68abf252a91faff64f67f76b078cb86d.jpg)

![6a23253f0da8f9cad63f9005a86462c748415a831975d1ccd8db9b46f5129250.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/6a23253f0da8f9cad63f9005a86462c748415a831975d1ccd8db9b46f5129250.jpg)

![6afa1ab9d1ef53db5385157fff9b75481da492537452aaa956aab75e6cef4d95.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/6afa1ab9d1ef53db5385157fff9b75481da492537452aaa956aab75e6cef4d95.jpg)

![710e5964992ab26c62a249bd48bb587155619320c6fefcb73d6f8d758e5d5d11.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/710e5964992ab26c62a249bd48bb587155619320c6fefcb73d6f8d758e5d5d11.jpg)

![787b899187254392cb4b40c989b022de7177e131da12b65a9abf33a97b0c99f8.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/787b899187254392cb4b40c989b022de7177e131da12b65a9abf33a97b0c99f8.jpg)

![89d3cffd1d0a3f0140ff9ee4370ee1ced359dcba57a1d8a5d871703cc7f1a023.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/89d3cffd1d0a3f0140ff9ee4370ee1ced359dcba57a1d8a5d871703cc7f1a023.jpg)

![daf769525a11ccc3cc7807632e2599f276fdd9388d874d417ab2dbd30addc835.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/images/daf769525a11ccc3cc7807632e2599f276fdd9388d874d417ab2dbd30addc835.jpg)

### Tables

![0187534afcbd8f670a2b23ffb72f054a0edc8cf27861bbfa6103c68958f10da7.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/0187534afcbd8f670a2b23ffb72f054a0edc8cf27861bbfa6103c68958f10da7.jpg)

![0a6e0c58b65d2622a2cdf22e7d1677eb924627356e7fc2e429892f259e1e81a7.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/0a6e0c58b65d2622a2cdf22e7d1677eb924627356e7fc2e429892f259e1e81a7.jpg)

![0fe9386c4d56fce1bf9bc2d426c4919d1b7b54ddbed2544c266d54095f070a91.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/0fe9386c4d56fce1bf9bc2d426c4919d1b7b54ddbed2544c266d54095f070a91.jpg)

![28d2563b484c8d516179614211a83ef6d7e2b34eb4c08a18477fa477404cdfb9.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/28d2563b484c8d516179614211a83ef6d7e2b34eb4c08a18477fa477404cdfb9.jpg)

![36ec5b664fcf5ad2747b7b205489fdb8d65af5df3e442235c378d0b1199a9802.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/36ec5b664fcf5ad2747b7b205489fdb8d65af5df3e442235c378d0b1199a9802.jpg)

![3862d4b709e261497298ca4cb860955fcaa26e4add6bb6745887f6f364fdb727.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/3862d4b709e261497298ca4cb860955fcaa26e4add6bb6745887f6f364fdb727.jpg)

![43d3b3baabb64fc222be01db36142fb4f8f4fdd2ba05c70bb8facef51d329ce8.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/43d3b3baabb64fc222be01db36142fb4f8f4fdd2ba05c70bb8facef51d329ce8.jpg)

![6f0d375fe8055c6c0131d659bea5b53ca777575889b91cf4124b6e29219432d5.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/6f0d375fe8055c6c0131d659bea5b53ca777575889b91cf4124b6e29219432d5.jpg)

![8e02aa8fcf0169011637852321076aa9f9fbdcf7578c12ea96f424e367c43712.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/8e02aa8fcf0169011637852321076aa9f9fbdcf7578c12ea96f424e367c43712.jpg)

![95434f51358dc33da694b5c5c58b03cea36ae8012bb434171b085b440f6669d0.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/95434f51358dc33da694b5c5c58b03cea36ae8012bb434171b085b440f6669d0.jpg)

![b716031b4f4b3f0e097e108703b5f1e65e5707b065e9af3404bef10fcc0a7d63.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/b716031b4f4b3f0e097e108703b5f1e65e5707b065e9af3404bef10fcc0a7d63.jpg)

![bdbfa93383dfd9e074ba90a672d67f15dfbf8736ccf2f07bca84ffec19328543.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/bdbfa93383dfd9e074ba90a672d67f15dfbf8736ccf2f07bca84ffec19328543.jpg)

![c2eb13c8736dac8a2c1879b3b3874608c730708557b5fdfdd610b0ecf8be07d2.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/c2eb13c8736dac8a2c1879b3b3874608c730708557b5fdfdd610b0ecf8be07d2.jpg)

![e8ba0751e9c7121ce2cbfb68f9aecd092404183cab877391766f1dcad5f58d8b.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/e8ba0751e9c7121ce2cbfb68f9aecd092404183cab877391766f1dcad5f58d8b.jpg)

![fe720e798e5d52ace737c889a07fd796cdd430a4af6636ff654f06fb62a1af72.jpg](../icml_results/5796_The%20Generalization%20Spectrum_%20A%20Chromatographic%20Approach%20to%20Evaluating%20Learning%20Algorithms/tables/fe720e798e5d52ace737c889a07fd796cdd430a4af6636ff654f06fb62a1af72.jpg)

<a id="paper-5748"></a>

## Rethinking Parameter Sharing as Graph Coloring for Structured Compression


### Images

![55765b6a15b2faef31b6d602e0cdf73c2983c6598ed253577f095e0619346de1.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/images/55765b6a15b2faef31b6d602e0cdf73c2983c6598ed253577f095e0619346de1.jpg)

![73c687a60809ab293f920a28ef3bfd82ec800845019fa5d6d2c9b666ccc344f6.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/images/73c687a60809ab293f920a28ef3bfd82ec800845019fa5d6d2c9b666ccc344f6.jpg)

![88149ef20cbbceaa53c91a67d0d9baf27f80d4ab1ab1b2c2c261433517bd6caa.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/images/88149ef20cbbceaa53c91a67d0d9baf27f80d4ab1ab1b2c2c261433517bd6caa.jpg)

![ac2d9a6ec14e015e47fc229b9c879b299f68557fcf842703092ac0af916e8447.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/images/ac2d9a6ec14e015e47fc229b9c879b299f68557fcf842703092ac0af916e8447.jpg)

![fa093914395906875704f1320419b1bcf853e7a20eb1df75fed5c111f9fa60b9.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/images/fa093914395906875704f1320419b1bcf853e7a20eb1df75fed5c111f9fa60b9.jpg)

### Tables

![1dac4dd2d7964ba86522ad0933151ae597a33bf938eec23d2e64984176b41fe9.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/1dac4dd2d7964ba86522ad0933151ae597a33bf938eec23d2e64984176b41fe9.jpg)

![422db5e6b481532b66aa0abd1783821e2434784165a9993fa46b1f7737e9800a.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/422db5e6b481532b66aa0abd1783821e2434784165a9993fa46b1f7737e9800a.jpg)

![5aad83483179a6df75d8ea22787cf4fbb5162b94736f941abbccab05c0602c64.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/5aad83483179a6df75d8ea22787cf4fbb5162b94736f941abbccab05c0602c64.jpg)

![5c6f10a7e247d1e9a8cba2abd3290dc61ed01543ed3ee4c2407fc6eab5dfec35.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/5c6f10a7e247d1e9a8cba2abd3290dc61ed01543ed3ee4c2407fc6eab5dfec35.jpg)

![76881e97c268a614f010a210e5426b45c831df4549213cfd0af05cc2da0d5325.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/76881e97c268a614f010a210e5426b45c831df4549213cfd0af05cc2da0d5325.jpg)

![771bfc1599a30223c080521b38cc37483ddb9ce57dbfdc5c31d13f1c725a0cbf.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/771bfc1599a30223c080521b38cc37483ddb9ce57dbfdc5c31d13f1c725a0cbf.jpg)

![90700a06bde6d046820fb3ea2f3c0bf7895a05aa3a48bc2c29335b0f919b8704.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/90700a06bde6d046820fb3ea2f3c0bf7895a05aa3a48bc2c29335b0f919b8704.jpg)

![9aa13c2eaa9699a9f8026ac0a704f7dd735cd4977ec169c9fb3f9bfe24a6be71.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/9aa13c2eaa9699a9f8026ac0a704f7dd735cd4977ec169c9fb3f9bfe24a6be71.jpg)

![ab6b698559d2ade88e9efd2c4174c7ce6f9f536d680e10b77e19a4cbf711ece5.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/ab6b698559d2ade88e9efd2c4174c7ce6f9f536d680e10b77e19a4cbf711ece5.jpg)

![b3eea5000945bd9f7cb938db6736269db35d12ed23262eb61d0778488f08f341.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/b3eea5000945bd9f7cb938db6736269db35d12ed23262eb61d0778488f08f341.jpg)

![e5ed2d17fcf7bab91487cd6c9c80c3ee2d136fc27f2a75bc74b817d3ca22d2c7.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/e5ed2d17fcf7bab91487cd6c9c80c3ee2d136fc27f2a75bc74b817d3ca22d2c7.jpg)

![f1c7af8eec7edc13d6e1809c66788cd6f1631514f851adfafacab1c34403cb8a.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/f1c7af8eec7edc13d6e1809c66788cd6f1631514f851adfafacab1c34403cb8a.jpg)

![f7fdd1f9df568fb345912acf257865f1a6cd2a4c80659357094dd68a9f2b30f1.jpg](../icml_results/5797_Rethinking%20Parameter%20Sharing%20as%20Graph%20Coloring%20for%20Structured%20Compression/tables/f7fdd1f9df568fb345912acf257865f1a6cd2a4c80659357094dd68a9f2b30f1.jpg)

<a id="paper-5749"></a>

## PSBench: Editing Image via GUI Agents in Photoshop


### Images

![041712878be06153fdb851fafc952c16b124bf146799b046fbfdabb9cea0a7cd.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/041712878be06153fdb851fafc952c16b124bf146799b046fbfdabb9cea0a7cd.jpg)

![198739f8d6928e25a14bc99faa0c36c54273182419b4f76e00d208cdb761dcff.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/198739f8d6928e25a14bc99faa0c36c54273182419b4f76e00d208cdb761dcff.jpg)

![55761fe574174aa3ca686f80fe8f0cc93fef5cd6d940d1277f6a4e024d9c7ed8.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/55761fe574174aa3ca686f80fe8f0cc93fef5cd6d940d1277f6a4e024d9c7ed8.jpg)

![5a227a6c166d538851b62e09c379047f424e3cccec7133d68ae6abeb54a7ed02.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/5a227a6c166d538851b62e09c379047f424e3cccec7133d68ae6abeb54a7ed02.jpg)

![5ddf857df4eb015d1fa6d2d3b6ceabe82f65a81cc45f6c6a6950d3392fcda186.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/5ddf857df4eb015d1fa6d2d3b6ceabe82f65a81cc45f6c6a6950d3392fcda186.jpg)

![76f7597cab9b74a08161bc5c9db341ce6d5f2b800c9eab46b4b9c2c01a12de8a.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/76f7597cab9b74a08161bc5c9db341ce6d5f2b800c9eab46b4b9c2c01a12de8a.jpg)

![7b47c0e376f629d5d53b4549e37253a9df673f9ffa4c9c6a79f48cda4ae36a1e.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/7b47c0e376f629d5d53b4549e37253a9df673f9ffa4c9c6a79f48cda4ae36a1e.jpg)

![8105cbf5168c64f032b7e2b875e0a6c99ad2b3f801928852c4b607e823341407.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/8105cbf5168c64f032b7e2b875e0a6c99ad2b3f801928852c4b607e823341407.jpg)

![923f7128f3743c3a90761c477ab25db1ba367f94635ce1004cba67d0cda8e3a2.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/923f7128f3743c3a90761c477ab25db1ba367f94635ce1004cba67d0cda8e3a2.jpg)

![edcce1377fdcf5a62743b734e895f92a09d549e9d3df5999d380ac608b82aa56.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/edcce1377fdcf5a62743b734e895f92a09d549e9d3df5999d380ac608b82aa56.jpg)

![f20b6cd4e389c2b44d7e7024c5d0a5861bfd6b5b6a9b93dffa9a9e804bcd78a2.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/f20b6cd4e389c2b44d7e7024c5d0a5861bfd6b5b6a9b93dffa9a9e804bcd78a2.jpg)

![fdf0e2084d738de8c7ae72b754b6266554f2b16332b3a58e1a9c5e723f6b1787.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/images/fdf0e2084d738de8c7ae72b754b6266554f2b16332b3a58e1a9c5e723f6b1787.jpg)

### Tables

![0f5bdb5fb6f6f34872b0420b0d28a0d22de5f7be25cedf298dd2cb18961d4e4a.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/0f5bdb5fb6f6f34872b0420b0d28a0d22de5f7be25cedf298dd2cb18961d4e4a.jpg)

![1944c468739c17df50203d3abe02272dfc4dd6714838de8c4c1bbd302b2d773f.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/1944c468739c17df50203d3abe02272dfc4dd6714838de8c4c1bbd302b2d773f.jpg)

![1faa566025ac95189d0823d7dbaea59d9e23ff8266b33768e5ce7b0c0916f57e.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/1faa566025ac95189d0823d7dbaea59d9e23ff8266b33768e5ce7b0c0916f57e.jpg)

![30677dd09c65ffeb64108a0685a0964c826748fac187241d51d244a78522bd44.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/30677dd09c65ffeb64108a0685a0964c826748fac187241d51d244a78522bd44.jpg)

![313d9f75e55964bdd9526d7a39a1fa887217fc46a482f454406cfe0bfa942130.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/313d9f75e55964bdd9526d7a39a1fa887217fc46a482f454406cfe0bfa942130.jpg)

![3d9ce72044fbba99798686d5e8f12374b146f657a00d15461a2535979414d623.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/3d9ce72044fbba99798686d5e8f12374b146f657a00d15461a2535979414d623.jpg)

![44863dff77054f598c51cd9b80354f9edde5e2786eaa24122199d8892bcd8b13.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/44863dff77054f598c51cd9b80354f9edde5e2786eaa24122199d8892bcd8b13.jpg)

![4742c587752b50669143c5dc5ee4c53fe8e3034f3f1abd0ef3b4994d943aa155.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/4742c587752b50669143c5dc5ee4c53fe8e3034f3f1abd0ef3b4994d943aa155.jpg)

![4e2c7bc868336d5d49692c51bbbf480ee42b2eb95d8a923d02fdc6cc4019302b.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/4e2c7bc868336d5d49692c51bbbf480ee42b2eb95d8a923d02fdc6cc4019302b.jpg)

![b286d7bf1d9cbee1425dad5b2241d0513f3e6e846a24c9dd6f78e00068207e37.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/b286d7bf1d9cbee1425dad5b2241d0513f3e6e846a24c9dd6f78e00068207e37.jpg)

![b2c51f77b0139e15c519b20be1de776305d3c9de13d7c8e0a3c5ad8bf0f7924d.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/b2c51f77b0139e15c519b20be1de776305d3c9de13d7c8e0a3c5ad8bf0f7924d.jpg)

![c8f5b249c12cb0608ab9aaf647962be6ecfdef7c294b0fe6722a5ec0623619e2.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/c8f5b249c12cb0608ab9aaf647962be6ecfdef7c294b0fe6722a5ec0623619e2.jpg)

![e963dd0c5fc2f63d01a1a619a0c3379e741fd3ef4fca72e159daadb42e2f9cbf.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/e963dd0c5fc2f63d01a1a619a0c3379e741fd3ef4fca72e159daadb42e2f9cbf.jpg)

![f7039d89fb0089dd1f84369f1c38e621c1f83407be0cf5491355b625037c7a8f.jpg](../icml_results/5798_PSBench_%20Editing%20Image%20via%20GUI%20Agents%20in%20Photoshop/tables/f7039d89fb0089dd1f84369f1c38e621c1f83407be0cf5491355b625037c7a8f.jpg)

<a id="paper-5750"></a>

## LoRA-DA: Data-Aware Initialization for Low-Rank Adaptation via Asymptotic Analysis


### Images

![09949636397f944775a13820f7896c60aa50db954dbffafb537dbd7c467c82b9.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/images/09949636397f944775a13820f7896c60aa50db954dbffafb537dbd7c467c82b9.jpg)

![0e799852c4bd7fb6054242d7f834da9c1122bfb7c1ac3f1722af7273eff87baf.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/images/0e799852c4bd7fb6054242d7f834da9c1122bfb7c1ac3f1722af7273eff87baf.jpg)

![8da3376e2a3e7795af1f2ac5b5b871384673e32279c46a3143baf2ee31111aae.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/images/8da3376e2a3e7795af1f2ac5b5b871384673e32279c46a3143baf2ee31111aae.jpg)

![e8bcfdc71c5f1a846b00f61f2e84b678f07c59ef8858352a615fc839437b158b.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/images/e8bcfdc71c5f1a846b00f61f2e84b678f07c59ef8858352a615fc839437b158b.jpg)

### Tables

![166429b809e208410cf1c680403fcfb39f8910dafeff21c688187e77f412e710.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/166429b809e208410cf1c680403fcfb39f8910dafeff21c688187e77f412e710.jpg)

![25aa192693ca8e4565dd7aa8635336d58da888cae62d88c941efee868d28b674.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/25aa192693ca8e4565dd7aa8635336d58da888cae62d88c941efee868d28b674.jpg)

![27de379028b35060280f71b61035c84100fe80abfd0385d505f6db5ac2a60d35.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/27de379028b35060280f71b61035c84100fe80abfd0385d505f6db5ac2a60d35.jpg)

![4bfbf534d7ab3abc8e20fe7f1e6a575f4917c370f4d0ea265342c34ac0002e72.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/4bfbf534d7ab3abc8e20fe7f1e6a575f4917c370f4d0ea265342c34ac0002e72.jpg)

![62597cf7e25debc3ba61ac513046dba68cdddd18d78728ef8772409590a2c2d4.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/62597cf7e25debc3ba61ac513046dba68cdddd18d78728ef8772409590a2c2d4.jpg)

![7775cc56fd919072c47814942fe735bfeb747b0bb43fbb3f5a7164b7e76c0d8d.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/7775cc56fd919072c47814942fe735bfeb747b0bb43fbb3f5a7164b7e76c0d8d.jpg)

![909d50b8902073f81d451577345a8fb4d2276975f8e60e93639e42109643db8f.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/909d50b8902073f81d451577345a8fb4d2276975f8e60e93639e42109643db8f.jpg)

![a673956ec8c4041545f94cc59bc49a6085d76fb740b7bb1a79e36f2e1cb86e04.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/a673956ec8c4041545f94cc59bc49a6085d76fb740b7bb1a79e36f2e1cb86e04.jpg)

![e29cda754d41d6c48a65b3cbd7fb2bc690b21364185d2ae1964c2291a48245d4.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/e29cda754d41d6c48a65b3cbd7fb2bc690b21364185d2ae1964c2291a48245d4.jpg)

![f4619103f2c7592d5ede7c1479442a885fe61d1cb8af21b618bd6b0634ca93cd.jpg](../icml_results/5799_LoRA-DA_%20Data-Aware%20Initialization%20for%20Low-Rank%20Adaptation%20via%20Asymptotic%20Analysis/tables/f4619103f2c7592d5ede7c1479442a885fe61d1cb8af21b618bd6b0634ca93cd.jpg)

<a id="paper-5751"></a>

## Motion-Residual Conflict-Aware Time Reversal for Generative Inbetweening


### Images

![09ae0492f6820401028af634558c6a6fe1e3e3173eca1f44a3b7a071cfdf3eb2.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/09ae0492f6820401028af634558c6a6fe1e3e3173eca1f44a3b7a071cfdf3eb2.jpg)

![3b58a4e40a5a988a1c21344e4cf83972f128d7dba61fcfa9e78bb640dfac1d1f.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/3b58a4e40a5a988a1c21344e4cf83972f128d7dba61fcfa9e78bb640dfac1d1f.jpg)

![494fed607fb5575d8fd30279c60c92135423aa621906b244e504d16ca23ac4f6.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/494fed607fb5575d8fd30279c60c92135423aa621906b244e504d16ca23ac4f6.jpg)

![71f024d2991b6a5a4c4b7dd8228f3b5df77e1b0df6243854692a64e838e5edbe.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/71f024d2991b6a5a4c4b7dd8228f3b5df77e1b0df6243854692a64e838e5edbe.jpg)

![7b96385a10b9b0bba33cd452e79f773c23b07139ee10cfe350b64a223760761c.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/7b96385a10b9b0bba33cd452e79f773c23b07139ee10cfe350b64a223760761c.jpg)

![7f13f93ad751a11a3ce1496459cea1755a41380b553c4decf33dad675702c10e.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/7f13f93ad751a11a3ce1496459cea1755a41380b553c4decf33dad675702c10e.jpg)

![9d39db8f022424eb01e009e17381ae486d2492bbb9ef5e4b172d684fc2f3ac93.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/9d39db8f022424eb01e009e17381ae486d2492bbb9ef5e4b172d684fc2f3ac93.jpg)

![afb6c1a097ebd251cadff415b6797eda617137061dd951bd4199ff1818fbb1c8.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/afb6c1a097ebd251cadff415b6797eda617137061dd951bd4199ff1818fbb1c8.jpg)

![b6931ad0336b510ee44d374a1d6e5c67aac073be120d38bf34ba320c58b04305.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/b6931ad0336b510ee44d374a1d6e5c67aac073be120d38bf34ba320c58b04305.jpg)

![b87d88c20f00ad6633c97513250759bbc600363d342ee83a17ee3afa2424ca00.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/b87d88c20f00ad6633c97513250759bbc600363d342ee83a17ee3afa2424ca00.jpg)

![d2b417cd9f310ec094ceb9b133e8e0cc0d665558d1f1738b0d798b7e93d5c0d1.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/d2b417cd9f310ec094ceb9b133e8e0cc0d665558d1f1738b0d798b7e93d5c0d1.jpg)

![e93b6e0fd1648fc34cfd134fed1ab654050c1734b15cab1fb68d01b802426ca9.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/e93b6e0fd1648fc34cfd134fed1ab654050c1734b15cab1fb68d01b802426ca9.jpg)

![ee58b1799284d5cf6bc71d53b04fb4c58ae237bc29daf413f8f70802a5452cb5.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/ee58b1799284d5cf6bc71d53b04fb4c58ae237bc29daf413f8f70802a5452cb5.jpg)

![f995302e216bcc9137c3369220e7c9188a1b2545b9b27f7520227d6530a44e80.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/images/f995302e216bcc9137c3369220e7c9188a1b2545b9b27f7520227d6530a44e80.jpg)

### Tables

![05ded5999d1489549dfdf6ab8b1392b23f12bc9a3fa5c1ebd46a12ebc6bde0a5.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/05ded5999d1489549dfdf6ab8b1392b23f12bc9a3fa5c1ebd46a12ebc6bde0a5.jpg)

![2c4d2cf78dd11026e10a181d8db6f18bb72c48de4016d6cb6b9328cb0723a688.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/2c4d2cf78dd11026e10a181d8db6f18bb72c48de4016d6cb6b9328cb0723a688.jpg)

![558046aa12d2be72a52c6331decd1ece0db25983759c3caadc6228841aa1f0a1.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/558046aa12d2be72a52c6331decd1ece0db25983759c3caadc6228841aa1f0a1.jpg)

![5dbf5a715e17b8400d584d6c51f3f5c69b8c3eca3dc71813129ca416f6f9a929.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/5dbf5a715e17b8400d584d6c51f3f5c69b8c3eca3dc71813129ca416f6f9a929.jpg)

![691f23b7710fb4bbe564da345d0a36aeb21eb8d78a0f0841195e56bfca0d832f.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/691f23b7710fb4bbe564da345d0a36aeb21eb8d78a0f0841195e56bfca0d832f.jpg)

![7a2b296d7770f7fb4a1fb7c666c204de48a6bbd5052cf88385850f64d29dac4f.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/7a2b296d7770f7fb4a1fb7c666c204de48a6bbd5052cf88385850f64d29dac4f.jpg)

![86d578463bdb327c865d422db72cbbc9423416b68166186b0d6dde4d44efaf3c.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/86d578463bdb327c865d422db72cbbc9423416b68166186b0d6dde4d44efaf3c.jpg)

![9db22a4eac6bc2084e22fc36402eecdbb71a843cd806305aa28a58787f9b7d8c.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/9db22a4eac6bc2084e22fc36402eecdbb71a843cd806305aa28a58787f9b7d8c.jpg)

![ab6b0c9b706c9bdfea096f6e77491759f150793c3099977a0a1e71bdbf0579ac.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/ab6b0c9b706c9bdfea096f6e77491759f150793c3099977a0a1e71bdbf0579ac.jpg)

![f6efdbdd7fef66dfd87cbbef8368bee326ba619ed70e75b004e8635629d6fd42.jpg](../icml_results/5800_Motion-Residual%20Conflict-Aware%20Time%20Reversal%20for%20Generative%20Inbetweening/tables/f6efdbdd7fef66dfd87cbbef8368bee326ba619ed70e75b004e8635629d6fd42.jpg)

<a id="paper-5752"></a>

## Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation


### Images

![09784b3676b9f80c451e17b66031f5eca9da5519284ef01fc93249e2e9da54bd.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/images/09784b3676b9f80c451e17b66031f5eca9da5519284ef01fc93249e2e9da54bd.jpg)

![240ca8790274de8651eb042f69b5f05805df9dfcc72f6234fae860e45de455c6.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/images/240ca8790274de8651eb042f69b5f05805df9dfcc72f6234fae860e45de455c6.jpg)

![42d1e11ca3902493c1ad8cf9f3299f01250159fa57f5e2f450bfb89c6114e7c9.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/images/42d1e11ca3902493c1ad8cf9f3299f01250159fa57f5e2f450bfb89c6114e7c9.jpg)

![6294dc83b40ac85a0f605c6cd5ab5d18b5177e751a2f14a0379d882e0600d1a3.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/images/6294dc83b40ac85a0f605c6cd5ab5d18b5177e751a2f14a0379d882e0600d1a3.jpg)

![b04a9440c0d2d65e24b9bd002f21bded1eb10b86c1a447380898d636c8a44506.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/images/b04a9440c0d2d65e24b9bd002f21bded1eb10b86c1a447380898d636c8a44506.jpg)

### Tables

![00b196a038ea7f199a756d0225f0e187f005f78d608fccafd1a992d53c85fd92.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/00b196a038ea7f199a756d0225f0e187f005f78d608fccafd1a992d53c85fd92.jpg)

![0d84c0c329317d5cb90923da8f3f1e078e6319d4167181dab0e71a84e786fea3.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/0d84c0c329317d5cb90923da8f3f1e078e6319d4167181dab0e71a84e786fea3.jpg)

![3f8dea54f14f6510efacd77bd972c04fe5531186611d180fc57bee1f15c873d1.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/3f8dea54f14f6510efacd77bd972c04fe5531186611d180fc57bee1f15c873d1.jpg)

![4817aba6f7604970b3840b4390b6e96387574510a26a31f15feb671371f5e70f.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/4817aba6f7604970b3840b4390b6e96387574510a26a31f15feb671371f5e70f.jpg)

![6dee0fd5f4ec62436aa1a63fd31d0a5ff52a47eb45b4821c948cd8f0b2d89828.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/6dee0fd5f4ec62436aa1a63fd31d0a5ff52a47eb45b4821c948cd8f0b2d89828.jpg)

![87d4016b623e329f8184dfc70992ed682d7d0c973be31ddb879f9c5313d00189.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/87d4016b623e329f8184dfc70992ed682d7d0c973be31ddb879f9c5313d00189.jpg)

![8d3d1496d09fbcb6d99c63b3363c4ca222f580bd870c26f968a4c597a22826d6.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/8d3d1496d09fbcb6d99c63b3363c4ca222f580bd870c26f968a4c597a22826d6.jpg)

![910aeb9d91367a77e5eea10c920c81a8c80e034028e824922f262f47624eb89d.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/910aeb9d91367a77e5eea10c920c81a8c80e034028e824922f262f47624eb89d.jpg)

![9be7698b9e0fb70188d87d05d14afc3f80992ea4f40beba8e5730b51609aa8fe.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/9be7698b9e0fb70188d87d05d14afc3f80992ea4f40beba8e5730b51609aa8fe.jpg)

![bb268e0d913c893a73d481c958a71d1c6a8a3817c567167e8f9a763695365e71.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/bb268e0d913c893a73d481c958a71d1c6a8a3817c567167e8f9a763695365e71.jpg)

![c751f680a43343e8c37dbe49d045c552d850ba63ebee2303b258d1d4ffb92922.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/c751f680a43343e8c37dbe49d045c552d850ba63ebee2303b258d1d4ffb92922.jpg)

![c9ebc368ae00632c2f909ae9fa93ac6a73f7d56490c5f43e50aabea8c3e78e61.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/c9ebc368ae00632c2f909ae9fa93ac6a73f7d56490c5f43e50aabea8c3e78e61.jpg)

![d1dca7c7037ac3ff4baacd2dfb6322cb51ce717a6af6a8993174903a9455f385.jpg](../icml_results/5801_Semantic-Aware%20Motion%20Encoding%20for%20Topology-Agnostic%20Character%20Animation/tables/d1dca7c7037ac3ff4baacd2dfb6322cb51ce717a6af6a8993174903a9455f385.jpg)

<a id="paper-5753"></a>

## Diff-MN: Diffusion Parameterized MoE-NCDE for Continuous Time Series Generation with Irregular Observations


### Images

![0fc105e0f586c9d09a5574c88579de63f23e48b1a42c54140b24946bbaca1c94.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/0fc105e0f586c9d09a5574c88579de63f23e48b1a42c54140b24946bbaca1c94.jpg)

![116db56c6107f46c37e96a4165669ae53cc68bcbbbab784d0acc55f817f072f5.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/116db56c6107f46c37e96a4165669ae53cc68bcbbbab784d0acc55f817f072f5.jpg)

![120102df554a9e40671c81e9a028e6a335bd44f03255d2871ff053236717aa62.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/120102df554a9e40671c81e9a028e6a335bd44f03255d2871ff053236717aa62.jpg)

![2b6742a6157024a3ae264a1448f72be5421e3242ca1a2c00c118b51a96abf859.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/2b6742a6157024a3ae264a1448f72be5421e3242ca1a2c00c118b51a96abf859.jpg)

![350b60e3e41d0dac7a418c26e34c1bcf9b768f3f5cbd335180ca2e4278e2ed1f.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/350b60e3e41d0dac7a418c26e34c1bcf9b768f3f5cbd335180ca2e4278e2ed1f.jpg)

![40d6bf4307307d1fc4f29ca24207f3635d6c03f3190b2108d19de23131a07dfe.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/40d6bf4307307d1fc4f29ca24207f3635d6c03f3190b2108d19de23131a07dfe.jpg)

![5790ad753f15f435ab7f809b9bd584cc051d218e51602e78e63fcd8ea6ef53d4.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/5790ad753f15f435ab7f809b9bd584cc051d218e51602e78e63fcd8ea6ef53d4.jpg)

![5c4b37e2b318fd2aaf0f350b09a1c181cc3dad0e6bf69bdf3b7fac406347e8e7.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/5c4b37e2b318fd2aaf0f350b09a1c181cc3dad0e6bf69bdf3b7fac406347e8e7.jpg)

![81738b24e30c146c381ebb20af4b6247778b04d383fdb4e49ee0b16091b9b8dd.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/81738b24e30c146c381ebb20af4b6247778b04d383fdb4e49ee0b16091b9b8dd.jpg)

![858556f243d12a7beeb29181d9b5fc4ea590e5d07d44043977a6b15a454a5813.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/858556f243d12a7beeb29181d9b5fc4ea590e5d07d44043977a6b15a454a5813.jpg)

![91ede5cb7f606833334bfb381446c1a67a912800881672941c8b683ba50192a6.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/91ede5cb7f606833334bfb381446c1a67a912800881672941c8b683ba50192a6.jpg)

![a7c29bc8342b050d657aa4aaeb4974f96c4089ec7b94434413f03b9865ad8377.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/a7c29bc8342b050d657aa4aaeb4974f96c4089ec7b94434413f03b9865ad8377.jpg)

![b8b133c4c9df2d10a42b50986b2157028c7a7d109f60744c3f60f2f077438ecf.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/b8b133c4c9df2d10a42b50986b2157028c7a7d109f60744c3f60f2f077438ecf.jpg)

![c83018ec2cae9990c8e2ff7255f6461f7f5eb2062bee91eaddbe2bd46fc6c986.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/c83018ec2cae9990c8e2ff7255f6461f7f5eb2062bee91eaddbe2bd46fc6c986.jpg)

![cefe8b9d0d7683108ddf234cc7157cda3ca11c6bae12cfd262c924043924e8e7.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/cefe8b9d0d7683108ddf234cc7157cda3ca11c6bae12cfd262c924043924e8e7.jpg)

![f063e74fde5620efe10c42bf5a6a31a75d2de99c32e9f717004fb13fc4701d78.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/f063e74fde5620efe10c42bf5a6a31a75d2de99c32e9f717004fb13fc4701d78.jpg)

![f1dc4f6ba1ccd2807cb87c25b3d7efae3fea028a86198b3d82c29572a6bc7976.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/f1dc4f6ba1ccd2807cb87c25b3d7efae3fea028a86198b3d82c29572a6bc7976.jpg)

![f386e6c6fa945cf932b839bb7edbdd882bd13dd1bedb5ffe9f345dfe2a39ce31.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/f386e6c6fa945cf932b839bb7edbdd882bd13dd1bedb5ffe9f345dfe2a39ce31.jpg)

![f65153764292e2cc863f577a048b1472c1d03aa0a07530fba441208ff89c4077.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/f65153764292e2cc863f577a048b1472c1d03aa0a07530fba441208ff89c4077.jpg)

![fdbab80dd14935c6c58e2cd1ad61d1f352fdbeee481b78bc6145e40be366f965.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/images/fdbab80dd14935c6c58e2cd1ad61d1f352fdbeee481b78bc6145e40be366f965.jpg)

### Tables

![0384fc3f33ab81a5197aa4793a29f9d490ee232ec215b624a04145bc31df28ca.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/0384fc3f33ab81a5197aa4793a29f9d490ee232ec215b624a04145bc31df28ca.jpg)

![0d397a9c979ef9805be3c0861faa2708a99456cf998b5d7389676b6fa4754a55.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/0d397a9c979ef9805be3c0861faa2708a99456cf998b5d7389676b6fa4754a55.jpg)

![1e125e50ce59596d73843e48a338c073def9207570885cfcb9c5ea7bfde3e190.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/1e125e50ce59596d73843e48a338c073def9207570885cfcb9c5ea7bfde3e190.jpg)

![1f1c6a2b9aef8be1b3311a3205633eba9bd33a878d54bb1db98bb6468a3ca39c.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/1f1c6a2b9aef8be1b3311a3205633eba9bd33a878d54bb1db98bb6468a3ca39c.jpg)

![25489eeff5b5850f9fb7bb698ca8a7b521f83e30dadcb9b200f51808d9a8690e.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/25489eeff5b5850f9fb7bb698ca8a7b521f83e30dadcb9b200f51808d9a8690e.jpg)

![3e6894b2873aed92850e0c6d0effc095a58511101ec4a2d668e15e3a72521341.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/3e6894b2873aed92850e0c6d0effc095a58511101ec4a2d668e15e3a72521341.jpg)

![8ace41191da61af951dc074740f0c4fe421efb64666e5e2757f44ce758fa8687.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/8ace41191da61af951dc074740f0c4fe421efb64666e5e2757f44ce758fa8687.jpg)

![99a474c20acb108cd0471eca7db10b2af3c5fb142167fe1d7a05ec96bf20fc3b.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/99a474c20acb108cd0471eca7db10b2af3c5fb142167fe1d7a05ec96bf20fc3b.jpg)

![aeb9689552ed338a0553d93a3ccfaa98652b2daeeab0957f3e3ad73f92a0f179.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/aeb9689552ed338a0553d93a3ccfaa98652b2daeeab0957f3e3ad73f92a0f179.jpg)

![baae7e60838020b5487ec3b28c08a0a2dd615579bcf215812048768563ac5e21.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/baae7e60838020b5487ec3b28c08a0a2dd615579bcf215812048768563ac5e21.jpg)

![c537bae955a2c565535af7c7ec70703810a60995b50c90cec7db4a0d014c6ea5.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/c537bae955a2c565535af7c7ec70703810a60995b50c90cec7db4a0d014c6ea5.jpg)

![d4d1ed004f5f814d6fd47fd96656bd8e5808d4e683400296d259c28d1feec516.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/d4d1ed004f5f814d6fd47fd96656bd8e5808d4e683400296d259c28d1feec516.jpg)

![eb8d5057bf9ee5f8ce1351a473f064c26a8e099a2a9f08159bdd3a0b554da601.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/eb8d5057bf9ee5f8ce1351a473f064c26a8e099a2a9f08159bdd3a0b554da601.jpg)

![fd2d90c16282948356ef977aaa249e04747f14bfa8657cf5b0049add4a8c2c16.jpg](../icml_results/5802_Diff-MN_%20Diffusion%20Parameterized%20MoE-NCDE%20for%20Continuous%20Time%20Series%20Generation%20with%20Irregular%20Obser/tables/fd2d90c16282948356ef977aaa249e04747f14bfa8657cf5b0049add4a8c2c16.jpg)

<a id="paper-5754"></a>

## Comp-Attn: Present-and-Align Attention for Compositional Video Generation


### Images

![00f9baf3141d8f05c5dc6d5d538c786cba485e5efe36bb0b3875809d56c7232d.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/00f9baf3141d8f05c5dc6d5d538c786cba485e5efe36bb0b3875809d56c7232d.jpg)

![0292b227d552799a18a76902ec5b120754d0bfa2e2e058d3a702ad470f052f9a.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/0292b227d552799a18a76902ec5b120754d0bfa2e2e058d3a702ad470f052f9a.jpg)

![2335ea55b5a0c4e5cf91ce6acd4edf8d1d3617f7fb5491b4093f5fa68dc435ce.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/2335ea55b5a0c4e5cf91ce6acd4edf8d1d3617f7fb5491b4093f5fa68dc435ce.jpg)

![3111069781fbde2b27b8251ac1887d1b52ae1663c648b6b386b99ef40c5ce7f0.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/3111069781fbde2b27b8251ac1887d1b52ae1663c648b6b386b99ef40c5ce7f0.jpg)

![49c8b3fb6b4674ed22e3e5d960c6ecfaae15ff4ee3b50f95aabc83445f9ee58a.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/49c8b3fb6b4674ed22e3e5d960c6ecfaae15ff4ee3b50f95aabc83445f9ee58a.jpg)

![4d8d3d4c6943878863294f44bdbd1737729c95d4a1c224e3237ecfcdc068d399.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/4d8d3d4c6943878863294f44bdbd1737729c95d4a1c224e3237ecfcdc068d399.jpg)

![72bca898c0295748d7cc0c45089d7bc1269f91b0db5010133a693f83adc1de36.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/72bca898c0295748d7cc0c45089d7bc1269f91b0db5010133a693f83adc1de36.jpg)

![79867d9d6053a74949302ad406c897e9aa992c10e822622d148caa11632d3f2a.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/79867d9d6053a74949302ad406c897e9aa992c10e822622d148caa11632d3f2a.jpg)

![800ea3ea0285cce5a101f81f4e19d06b8f04c128f29469c725e54cca75febc65.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/800ea3ea0285cce5a101f81f4e19d06b8f04c128f29469c725e54cca75febc65.jpg)

![947538aec91acb50b548507c03c3c847c2e317d2f967976f293da585d6243395.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/947538aec91acb50b548507c03c3c847c2e317d2f967976f293da585d6243395.jpg)

![abaf35a36108ffe83c28e4b79a4d246731960b6845a7cd108bed73320aa4c123.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/abaf35a36108ffe83c28e4b79a4d246731960b6845a7cd108bed73320aa4c123.jpg)

![b630003ba194c93168717bd95271f9bc03ca38e01dedf3040837b2e0d6d430b0.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/b630003ba194c93168717bd95271f9bc03ca38e01dedf3040837b2e0d6d430b0.jpg)

![bc34fc308bda913a620bf8511ea52eca703cb1e725b8a632c714bf5f2be8e99b.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/bc34fc308bda913a620bf8511ea52eca703cb1e725b8a632c714bf5f2be8e99b.jpg)

![c14040a5f466bb13651534bb988c2e8e0e63a19a637258685d389128a4aa4a93.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/c14040a5f466bb13651534bb988c2e8e0e63a19a637258685d389128a4aa4a93.jpg)

![cdbf8b87f1ed36fee9f13862be8dd0a972b1c0551beb63961cfd0bddac23cdda.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/cdbf8b87f1ed36fee9f13862be8dd0a972b1c0551beb63961cfd0bddac23cdda.jpg)

![e870b96aefd9e4867508a9fbf8c43047e62e513db1c19cdfae4fedd24d43b13f.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/e870b96aefd9e4867508a9fbf8c43047e62e513db1c19cdfae4fedd24d43b13f.jpg)

![ed652627ce3b7f9b68fcc9c9d8d60920b844b94fb887d58f5caa670336f74ded.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/ed652627ce3b7f9b68fcc9c9d8d60920b844b94fb887d58f5caa670336f74ded.jpg)

![f20a4e660c75c9b34d0f83a8d9b392268783245b88969c23763e81bb81874fd1.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/images/f20a4e660c75c9b34d0f83a8d9b392268783245b88969c23763e81bb81874fd1.jpg)

### Tables

![0dd94c1feda738ef402966f443a627f06c275bf7c6893235a6a1c56a7299d5f6.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/0dd94c1feda738ef402966f443a627f06c275bf7c6893235a6a1c56a7299d5f6.jpg)

![39338bbb94322e7dc284ed472cfdb6d02fb0dbdb291dd56b8b402b135052aa0c.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/39338bbb94322e7dc284ed472cfdb6d02fb0dbdb291dd56b8b402b135052aa0c.jpg)

![64e95c1582e8982c307319d0d0a7f82b8865032ef0c57e6d65af1406bb9627eb.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/64e95c1582e8982c307319d0d0a7f82b8865032ef0c57e6d65af1406bb9627eb.jpg)

![6b69485d5a58a686b0032707a0055ae110d43ec8e8b0214b1f4aaf21401fea66.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/6b69485d5a58a686b0032707a0055ae110d43ec8e8b0214b1f4aaf21401fea66.jpg)

![86bce7d9bb0e2fa645d91f736ac9aa080cb221b446cb6bac6f399f965a745440.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/86bce7d9bb0e2fa645d91f736ac9aa080cb221b446cb6bac6f399f965a745440.jpg)

![a97c2d75127f22904e89bb02b426877c3bc93426b4aa45cbeb0d70897ef6158b.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/a97c2d75127f22904e89bb02b426877c3bc93426b4aa45cbeb0d70897ef6158b.jpg)

![b1912de4904eb63098656a6290105e9864d83c4a8ef571190ef3dfc97ee14d7e.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/b1912de4904eb63098656a6290105e9864d83c4a8ef571190ef3dfc97ee14d7e.jpg)

![c3e01f2ec5c385fef30a4d80b69daa65e429c5a25b19f991d97a1441afe9c2f5.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/c3e01f2ec5c385fef30a4d80b69daa65e429c5a25b19f991d97a1441afe9c2f5.jpg)

![c3e4b408c01f3e3d5185af9241f18055b30a2913da43b4e73e864f6495312773.jpg](../icml_results/5803_Comp-Attn_%20Present-and-Align%20Attention%20for%20Compositional%20Video%20Generation/tables/c3e4b408c01f3e3d5185af9241f18055b30a2913da43b4e73e864f6495312773.jpg)

<a id="paper-5755"></a>

## Dimension-Independent Convergence of Underdamped Langevin Monte Carlo in KL Divergence


### Tables

![820b3b00e7ab55af9f6869f4e710867b5454c242a775fa3030e2b0d0894fd21f.jpg](../icml_results/5804_Dimension-Independent%20Convergence%20of%20Underdamped%20Langevin%20Monte%20Carlo%20in%20KL%20Divergence/tables/820b3b00e7ab55af9f6869f4e710867b5454c242a775fa3030e2b0d0894fd21f.jpg)

<a id="paper-5756"></a>

## Benign Overfitting in Adversarial Training for Vision Transformers


### Images

![0d4b2fff769238568750634dd477b1bf4b22c1bcd2df6b6599ebb9a1b551af8a.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/0d4b2fff769238568750634dd477b1bf4b22c1bcd2df6b6599ebb9a1b551af8a.jpg)

![54803ef2bd480110a3d22a1f07eb6040e15d7c488568b07a756dcab4450d8055.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/54803ef2bd480110a3d22a1f07eb6040e15d7c488568b07a756dcab4450d8055.jpg)

![6c83002713755cce43fea41307cf0167abbc29e65218a5424f3203d936af9091.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/6c83002713755cce43fea41307cf0167abbc29e65218a5424f3203d936af9091.jpg)

![8899f7f0f42e883086d7ccc6b58d5af4ffee33cff5b75f5ec016b28ada4e9a9c.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/8899f7f0f42e883086d7ccc6b58d5af4ffee33cff5b75f5ec016b28ada4e9a9c.jpg)

![b4a4d70ede6efb2967b6a3ed3b4a36d80b5c5f19423fa9107d2ff6888c95d583.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/b4a4d70ede6efb2967b6a3ed3b4a36d80b5c5f19423fa9107d2ff6888c95d583.jpg)

![d931b9fffd927214b1a9b4f2ed18fee41b710bf3cafa9860349035132660ea3e.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/d931b9fffd927214b1a9b4f2ed18fee41b710bf3cafa9860349035132660ea3e.jpg)

![e26b6fb2d991a005aa8a2b364abff5ab4ff65dcf250f26b51fa1a320b2a9b2c1.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/e26b6fb2d991a005aa8a2b364abff5ab4ff65dcf250f26b51fa1a320b2a9b2c1.jpg)

![ff7289c8acf4b664ce0e24ebdcc64f3eb0918a93fc6d3f5eb054a8a8c12f4e6f.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/images/ff7289c8acf4b664ce0e24ebdcc64f3eb0918a93fc6d3f5eb054a8a8c12f4e6f.jpg)

### Tables

![a451bb2ed8ca8a772b48c1cb782fc2a2a0065902ca35d1f08716d5c6e065cccb.jpg](../icml_results/5805_Benign%20Overfitting%20in%20Adversarial%20Training%20for%20Vision%20Transformers/tables/a451bb2ed8ca8a772b48c1cb782fc2a2a0065902ca35d1f08716d5c6e065cccb.jpg)

<a id="paper-5757"></a>

## Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch


### Images

![0e9003113a49feb17302d3fd0686efeee3169e766bb6d20e1115c8d1884e1b29.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/0e9003113a49feb17302d3fd0686efeee3169e766bb6d20e1115c8d1884e1b29.jpg)

![2ed1aef870203c24f983086376399144fbf5c8c81464f350d430d8b9ffd400c3.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/2ed1aef870203c24f983086376399144fbf5c8c81464f350d430d8b9ffd400c3.jpg)

![46463e4cdc2a559f6ea047380d046baa7a7a2cd6aa478cc44639f3e1036d072c.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/46463e4cdc2a559f6ea047380d046baa7a7a2cd6aa478cc44639f3e1036d072c.jpg)

![48cb2e8c7b9c6bf7c4faa71904be6bc341e29682781275eec1013425af6f3c5c.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/48cb2e8c7b9c6bf7c4faa71904be6bc341e29682781275eec1013425af6f3c5c.jpg)

![5489b12e1c3ce42627cd40a33db6b12db77f59bff7939505683e396a3a335dd9.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/5489b12e1c3ce42627cd40a33db6b12db77f59bff7939505683e396a3a335dd9.jpg)

![5f7cd1db7256753f545fc77c7c031237d97b0f26870c69aae1d95f61ea01afc7.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/5f7cd1db7256753f545fc77c7c031237d97b0f26870c69aae1d95f61ea01afc7.jpg)

![7a781c3805f08429e19c682e55f0fb78340a879660d6efc1f090abaa77ab69c0.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/7a781c3805f08429e19c682e55f0fb78340a879660d6efc1f090abaa77ab69c0.jpg)

![88e95211e9c21a7c4e687a3fad20048f71f6a1055dcab3487baa84d7e46b7220.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/88e95211e9c21a7c4e687a3fad20048f71f6a1055dcab3487baa84d7e46b7220.jpg)

![96074a1208ee33124dab791a98110aea663e628ab0b4540441de79f8c89f2822.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/96074a1208ee33124dab791a98110aea663e628ab0b4540441de79f8c89f2822.jpg)

![ab869e190784ee22c47de454f51bcce2ce0ec4cf4ddbe836281f3334fd9e93a9.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/ab869e190784ee22c47de454f51bcce2ce0ec4cf4ddbe836281f3334fd9e93a9.jpg)

![c92eb521b8ed0c0e34f309530e683e71d3555670209875e6fd05b2e17547688c.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/c92eb521b8ed0c0e34f309530e683e71d3555670209875e6fd05b2e17547688c.jpg)

![ca6f29061c33ce01882a01f2077d5042a561f9882074f6d6b4a55f57ae70666a.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/images/ca6f29061c33ce01882a01f2077d5042a561f9882074f6d6b4a55f57ae70666a.jpg)

### Tables

![0079459cb53dd1397b3c42391434ae7b85903f0259741963cfafbc2e8860c93d.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/tables/0079459cb53dd1397b3c42391434ae7b85903f0259741963cfafbc2e8860c93d.jpg)

![1f451b37d3b9075525f72a05ad34d22af11f3bfcc77ba68d3642b8386d4c9ebc.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/tables/1f451b37d3b9075525f72a05ad34d22af11f3bfcc77ba68d3642b8386d4c9ebc.jpg)

![3dbafe5f1bab317a5844e4d11b006496010e6434d7a4d77e4a07eddebf8a287b.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/tables/3dbafe5f1bab317a5844e4d11b006496010e6434d7a4d77e4a07eddebf8a287b.jpg)

![4d1200e2d3cc2f4fd85edf6ee609ee48492f0f412a37bb1282c890f8adaf7368.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/tables/4d1200e2d3cc2f4fd85edf6ee609ee48492f0f412a37bb1282c890f8adaf7368.jpg)

![6e4e5578df678587ab5e617b51186fe870ba33f37097d8faba80e6314d86db3e.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/tables/6e4e5578df678587ab5e617b51186fe870ba33f37097d8faba80e6314d86db3e.jpg)

![8b1aa53a5730b6f5eefd4442106856258aeb844912b9c9289d6110412bbeb593.jpg](../icml_results/5806_Deterministic%20Inference%20across%20Tensor%20Parallel%20Sizes%20That%20Eliminates%20Training-Inference%20Mismatch/tables/8b1aa53a5730b6f5eefd4442106856258aeb844912b9c9289d6110412bbeb593.jpg)

<a id="paper-5758"></a>

## CoMem: Context Management with A Decoupled Long-Context Model


### Images

![12c79ff4f0485bd774015fc9ac9143487aa0fa7f367cb533c8858495279ee718.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/images/12c79ff4f0485bd774015fc9ac9143487aa0fa7f367cb533c8858495279ee718.jpg)

![131da4ec7ad61cebbc6dcacb4dd3647c3308772bc6e12fb045920cf9f98b2671.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/images/131da4ec7ad61cebbc6dcacb4dd3647c3308772bc6e12fb045920cf9f98b2671.jpg)

![4467a60d03d6219ad767682716a1d00521309a7bf78360cf568fe5dac363f8d6.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/images/4467a60d03d6219ad767682716a1d00521309a7bf78360cf568fe5dac363f8d6.jpg)

![5532c31415563083a105d3aaae5d857f0acfb16957e1664075550466275a5b38.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/images/5532c31415563083a105d3aaae5d857f0acfb16957e1664075550466275a5b38.jpg)

![ec83cbed4ab742630e16023134dac4d02285c9a90e8ed0dc59a2a974dc01f533.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/images/ec83cbed4ab742630e16023134dac4d02285c9a90e8ed0dc59a2a974dc01f533.jpg)

### Tables

![048ac4292009ac73840b1f79a8b34b317f4a22f4a462970bd2d13d574ab91f5f.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/048ac4292009ac73840b1f79a8b34b317f4a22f4a462970bd2d13d574ab91f5f.jpg)

![07a51a074c7ddc7afc2a7656f90d026a2ce5bb8ce93afedf5b16b7c8c2d94f77.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/07a51a074c7ddc7afc2a7656f90d026a2ce5bb8ce93afedf5b16b7c8c2d94f77.jpg)

![24fb50483a0d42bd5d7fd1f6747bf10fffed81f1a02d438d58d71372af10a7d1.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/24fb50483a0d42bd5d7fd1f6747bf10fffed81f1a02d438d58d71372af10a7d1.jpg)

![5c26401491a06a5e8c64cd62548702169df04a56574f7392d73c5b9e55076236.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/5c26401491a06a5e8c64cd62548702169df04a56574f7392d73c5b9e55076236.jpg)

![d6820d9ea2feeb91fa83a5b714d39f1d66054bf4e0318bdeaeb6713078ab7e72.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/d6820d9ea2feeb91fa83a5b714d39f1d66054bf4e0318bdeaeb6713078ab7e72.jpg)

![f2365981e13fa9f3f0ae190a6757f21fc04eb6ffd0018af23866032d5e11d9d3.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/f2365981e13fa9f3f0ae190a6757f21fc04eb6ffd0018af23866032d5e11d9d3.jpg)

![f36be7fa67a722ec896c2fb4da082def7e0349109ddeafe2bd87a6f03e9dc3c9.jpg](../icml_results/5807_CoMem_%20Context%20Management%20with%20A%20Decoupled%20Long-Context%20Model/tables/f36be7fa67a722ec896c2fb4da082def7e0349109ddeafe2bd87a6f03e9dc3c9.jpg)

<a id="paper-5759"></a>

## Who Gets Credit or Blame? Attributing Accountability in Modern AI Systems


### Images

![0669c2163327f0ee9dee8d8d4e299e67d498f491f15ecc4dfb83c3d1c666e024.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/0669c2163327f0ee9dee8d8d4e299e67d498f491f15ecc4dfb83c3d1c666e024.jpg)

![23b8d646f83436a9ac2703f19389c0be74eb2cb257778f9a750d635e019375bd.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/23b8d646f83436a9ac2703f19389c0be74eb2cb257778f9a750d635e019375bd.jpg)

![30489ab0f988ae99ca0ee5d417495dc2db600f962a82d48601503725ca004338.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/30489ab0f988ae99ca0ee5d417495dc2db600f962a82d48601503725ca004338.jpg)

![4391604570dc1be43ad2d9446cb9ca4dcf1e6b68a35c0b94fe5d706bf8bb0f10.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/4391604570dc1be43ad2d9446cb9ca4dcf1e6b68a35c0b94fe5d706bf8bb0f10.jpg)

![86bb2427217f277b80269da444704e12f9095be59583bc5c929cbfc8f7725002.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/86bb2427217f277b80269da444704e12f9095be59583bc5c929cbfc8f7725002.jpg)

![9cdb2608e7bde7d5e89e3543aa601652d83be80485d55d59e429a12783ca92de.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/9cdb2608e7bde7d5e89e3543aa601652d83be80485d55d59e429a12783ca92de.jpg)

![c5872f3eb02dfc8d724ffd1cca553a6c5d6832b18ee351fcef1380cae76c9c35.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/images/c5872f3eb02dfc8d724ffd1cca553a6c5d6832b18ee351fcef1380cae76c9c35.jpg)

### Tables

![2948999b34707ef6f37493df3f2d2f98ea4794ff66182e0d5c0274bb35314f31.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/tables/2948999b34707ef6f37493df3f2d2f98ea4794ff66182e0d5c0274bb35314f31.jpg)

![627fd62a906d8815be1fa728fb9dac3efb4386dbedd0832f1661addd4026debf.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/tables/627fd62a906d8815be1fa728fb9dac3efb4386dbedd0832f1661addd4026debf.jpg)

![b6f92c2d32efcecdb1ecb5dcd3ac3b81f1607288ebac7915e42479a28a755085.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/tables/b6f92c2d32efcecdb1ecb5dcd3ac3b81f1607288ebac7915e42479a28a755085.jpg)

![d53e77d553f167e47db987362149c844428890fbf3218c109d4952a6576f6ade.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/tables/d53e77d553f167e47db987362149c844428890fbf3218c109d4952a6576f6ade.jpg)

![e76f3a3b9d44e72cc1c2fb7f7910696fb0db4b75646757321d6b45316402a3bf.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/tables/e76f3a3b9d44e72cc1c2fb7f7910696fb0db4b75646757321d6b45316402a3bf.jpg)

![e7e3d4b222275a04e9d40742b6b92d21126e68532a1efbf0668014089badad3c.jpg](../icml_results/5808_Who%20Gets%20Credit%20or%20Blame_%20Attributing%20Accountability%20in%20Modern%20AI%20Systems/tables/e7e3d4b222275a04e9d40742b6b92d21126e68532a1efbf0668014089badad3c.jpg)

<a id="paper-5760"></a>

## LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models


### Images

![27b71033ca073219586f1a30451b4f07e992945b67f040e99ecf4cf1613b1ab1.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/images/27b71033ca073219586f1a30451b4f07e992945b67f040e99ecf4cf1613b1ab1.jpg)

![330ffa603e81f774c1a1b2a0bb649ae31aaeabd07b977627a45fd8a1cae8c416.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/images/330ffa603e81f774c1a1b2a0bb649ae31aaeabd07b977627a45fd8a1cae8c416.jpg)

![46ee33d5da84e52ffd72ddb55cb383f5eb1bc525ea217a5f1344462b987693df.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/images/46ee33d5da84e52ffd72ddb55cb383f5eb1bc525ea217a5f1344462b987693df.jpg)

![6d67ab0e5634cbcc4ae69ac523fbe7649d847b5fbf39e41a8b2987543ca40168.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/images/6d67ab0e5634cbcc4ae69ac523fbe7649d847b5fbf39e41a8b2987543ca40168.jpg)

![70153b51945695d8996e9f98da52a1f02d55742c135d2b751996ec51e7f8e0b4.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/images/70153b51945695d8996e9f98da52a1f02d55742c135d2b751996ec51e7f8e0b4.jpg)

![af5cfd6a0430dbbfc48e35466f812d67ba408523f85bdab0ba18b25a7a74e435.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/images/af5cfd6a0430dbbfc48e35466f812d67ba408523f85bdab0ba18b25a7a74e435.jpg)

### Tables

![197153a1369b89e4524da04309ba98e82e25c35724af3a97e81377930e09440d.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/197153a1369b89e4524da04309ba98e82e25c35724af3a97e81377930e09440d.jpg)

![48d831b1ef08fb35edc3287090646fb918aee6fe3f89d3151d196f26b214b7d1.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/48d831b1ef08fb35edc3287090646fb918aee6fe3f89d3151d196f26b214b7d1.jpg)

![52f55666cb4b34132bfbae4ddc7e5082491b05168bcb66e69f52f9d5fd55235b.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/52f55666cb4b34132bfbae4ddc7e5082491b05168bcb66e69f52f9d5fd55235b.jpg)

![608cdb01791befa7a3479b1f9e4b2198a57c6958338defeacef9f152c536615d.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/608cdb01791befa7a3479b1f9e4b2198a57c6958338defeacef9f152c536615d.jpg)

![6195a2b88d768a29631fd5bd2f5f5d7d29993089c2b1fb07e9735f6111afecbf.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/6195a2b88d768a29631fd5bd2f5f5d7d29993089c2b1fb07e9735f6111afecbf.jpg)

![675cdd4efea7f3b8521b0648822822dc837fa910fd8fcbd52c7b30812af812df.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/675cdd4efea7f3b8521b0648822822dc837fa910fd8fcbd52c7b30812af812df.jpg)

![8b45fa85ba952a7845822d005f670600c5f862f55f8d80bf05a2c5597d57d160.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/8b45fa85ba952a7845822d005f670600c5f862f55f8d80bf05a2c5597d57d160.jpg)

![8fd0a4dfea3e238911a8d82f86f7acada7c019d10c6c0db66bec7e5e15d92e17.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/8fd0a4dfea3e238911a8d82f86f7acada7c019d10c6c0db66bec7e5e15d92e17.jpg)

![9dc7a50037d2c3d27c4bfcc4fcc1d5802acf7237032d9b7e974e15735f528012.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/9dc7a50037d2c3d27c4bfcc4fcc1d5802acf7237032d9b7e974e15735f528012.jpg)

![b503311c6ff992ffb3351814c820492495fae0174fe6db718e8e1268fc9fba07.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/b503311c6ff992ffb3351814c820492495fae0174fe6db718e8e1268fc9fba07.jpg)

![b90f5ebe123def37e27397b77c288ce2131054fe6e842145eb67d04c73b1f6ac.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/b90f5ebe123def37e27397b77c288ce2131054fe6e842145eb67d04c73b1f6ac.jpg)

![bdff80cb82edd3207ea7ee33dced2f3db2d68d42e7ed810ccef05abba7bbc93c.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/bdff80cb82edd3207ea7ee33dced2f3db2d68d42e7ed810ccef05abba7bbc93c.jpg)

![bee10a30c6b158766baac94b12e8800b7fe3218bb4e067849f14c4e381eafcde.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/bee10a30c6b158766baac94b12e8800b7fe3218bb4e067849f14c4e381eafcde.jpg)

![caff57aea3b6ec6ffb06310ee546154a378d4670bb146f882f4df66a6eddd459.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/caff57aea3b6ec6ffb06310ee546154a378d4670bb146f882f4df66a6eddd459.jpg)

![d9f0381983c787067197e7a9fe63a825466d5e038c2369291515a72148c87b3f.jpg](../icml_results/5809_LARFT_%20Closing%20the%20Cognition-Action%20Gap%20for%20Length%20Instruction%20Following%20in%20Large%20Language%20Models/tables/d9f0381983c787067197e7a9fe63a825466d5e038c2369291515a72148c87b3f.jpg)

<a id="paper-5761"></a>

## WF-Bench: A Benchmark for Neural-Network WaveFunction Expressivity and Scaling Laws


### Images

![06da8d28cde09a6567f0a280bee5559243865b15ea7bc545db010ef12a9a4b6c.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/images/06da8d28cde09a6567f0a280bee5559243865b15ea7bc545db010ef12a9a4b6c.jpg)

![3dccd683cbb36a8c213bfe226a557315f5bf2de03650747211c07da9ee5ab781.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/images/3dccd683cbb36a8c213bfe226a557315f5bf2de03650747211c07da9ee5ab781.jpg)

![5e34cb0daf97cacdf7e666d42d8ffca9b10d3d1b5de42dbd17d655e8ed105310.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/images/5e34cb0daf97cacdf7e666d42d8ffca9b10d3d1b5de42dbd17d655e8ed105310.jpg)

![cfcade7a3c10138df8c0f4a96e4cf3e147934aed2de9bbd187402f2f9b50d473.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/images/cfcade7a3c10138df8c0f4a96e4cf3e147934aed2de9bbd187402f2f9b50d473.jpg)

![d0cd18971923f8aa39d00f51f558e426a6f8be1aa3d4e941c6b6b517353fcef9.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/images/d0cd18971923f8aa39d00f51f558e426a6f8be1aa3d4e941c6b6b517353fcef9.jpg)

![d939ecc8b2708ed6d435dc1f97096728447768f7af81a25ff772f3b64a7d5d17.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/images/d939ecc8b2708ed6d435dc1f97096728447768f7af81a25ff772f3b64a7d5d17.jpg)

### Tables

![5333c684bf4b1557e45a221d5e73ee9b063284aae380cf4cc0825c5151439808.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/tables/5333c684bf4b1557e45a221d5e73ee9b063284aae380cf4cc0825c5151439808.jpg)

![b27f3fbebb41b9bc5c2b981ca7d9373d878f58d52c4789f34e04623c6fa60106.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/tables/b27f3fbebb41b9bc5c2b981ca7d9373d878f58d52c4789f34e04623c6fa60106.jpg)

![b6c544374323a354c22c3b63729777e11fa92b52dfa308d4a61b17a92f4304d1.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/tables/b6c544374323a354c22c3b63729777e11fa92b52dfa308d4a61b17a92f4304d1.jpg)

![c3deff34707a1dfa4eea2f08b0b08025792c2c484bb2840524718ac6853e62c0.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/tables/c3deff34707a1dfa4eea2f08b0b08025792c2c484bb2840524718ac6853e62c0.jpg)

![e5832afea2c6dc2629470e98a99aa4999e46c8fc1a8578d1f494508629c5c797.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/tables/e5832afea2c6dc2629470e98a99aa4999e46c8fc1a8578d1f494508629c5c797.jpg)

![fb94dcc1e62dd8737d42ddde4eb5c9cff866edad03a493115fb0e10dc3ebc7ac.jpg](../icml_results/5810_WF-Bench_%20A%20Benchmark%20for%20Neural-Network%20WaveFunction%20Expressivity%20and%20Scaling%20Laws/tables/fb94dcc1e62dd8737d42ddde4eb5c9cff866edad03a493115fb0e10dc3ebc7ac.jpg)

<a id="paper-5762"></a>

## One Tool Is Enough: Reinforcement Learning of LLM Agents for Repository-Level Code Navigation


### Images

![0a5c39b7fc8fb661c30f3e93904d4d819c1219a24586e5472a00b10ca8bea33a.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/0a5c39b7fc8fb661c30f3e93904d4d819c1219a24586e5472a00b10ca8bea33a.jpg)

![1488b178911f6b7aa3e9462145be8ea220b564c0fb4cbefd81f99d3c2a23d21f.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/1488b178911f6b7aa3e9462145be8ea220b564c0fb4cbefd81f99d3c2a23d21f.jpg)

![245d1739431c248360c08367ed8815c258a04ed77801ae2cb87602c81b6d837d.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/245d1739431c248360c08367ed8815c258a04ed77801ae2cb87602c81b6d837d.jpg)

![57a55025ddf1e4d8c4d0ca6fbd52581f511030eadb5a26f2d07132aeaff08004.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/57a55025ddf1e4d8c4d0ca6fbd52581f511030eadb5a26f2d07132aeaff08004.jpg)

![89dcd9e2a936128295c4e629fb70dddc4fcc30c272456ad2589175b7cee02f6e.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/89dcd9e2a936128295c4e629fb70dddc4fcc30c272456ad2589175b7cee02f6e.jpg)

![8c425e2c1aa384018d543bfdd75fc938e411deb858a4e7f1d93a428f003b41d3.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/8c425e2c1aa384018d543bfdd75fc938e411deb858a4e7f1d93a428f003b41d3.jpg)

![a6eaa17ebda75645d11e4db55b9022a70ada124bad2c81cbd4dc478fbaae8b5b.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/a6eaa17ebda75645d11e4db55b9022a70ada124bad2c81cbd4dc478fbaae8b5b.jpg)

![e39dd272c8a69eafab24b423861b5b43f1123ccd4594b51202e9ab465286bcb9.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/images/e39dd272c8a69eafab24b423861b5b43f1123ccd4594b51202e9ab465286bcb9.jpg)

### Tables

![0297d41583dd2d6cbe8099369e559f21ac699fcf0dc93183443c79116421a3a9.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/0297d41583dd2d6cbe8099369e559f21ac699fcf0dc93183443c79116421a3a9.jpg)

![0fbb8ecfdbdb3891889c2a66467719b714048de43fe08b8ee179d67252f537bb.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/0fbb8ecfdbdb3891889c2a66467719b714048de43fe08b8ee179d67252f537bb.jpg)

![1a4362a22e2c120b64a1cf375ca94ead08bc8da115c9b9d4132c75f00daad499.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/1a4362a22e2c120b64a1cf375ca94ead08bc8da115c9b9d4132c75f00daad499.jpg)

![2caf7100b26c7aaf573fe78c839717411dfd2d479b6678e23daf219087893675.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/2caf7100b26c7aaf573fe78c839717411dfd2d479b6678e23daf219087893675.jpg)

![3a92d9399df2908c9cf194966e54e4cf20626576b27ab015229b3f5ff9180408.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/3a92d9399df2908c9cf194966e54e4cf20626576b27ab015229b3f5ff9180408.jpg)

![3b67b2b0b04db78cf52a6a8234849c5e55f01950d37bd37f883bc91e6e10c2ec.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/3b67b2b0b04db78cf52a6a8234849c5e55f01950d37bd37f883bc91e6e10c2ec.jpg)

![4cb88415c11c2fadaefc955eba58770284194bbf237f8528d03c817f476f9557.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/4cb88415c11c2fadaefc955eba58770284194bbf237f8528d03c817f476f9557.jpg)

![5ca9ccfde84915437414398f656be13a51e85de4f1611b6584cdec7efaae7d51.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/5ca9ccfde84915437414398f656be13a51e85de4f1611b6584cdec7efaae7d51.jpg)

![7118cd39a287322a04bdcbd005a12f4b24f6623831a05060692c77da7bfa3860.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/7118cd39a287322a04bdcbd005a12f4b24f6623831a05060692c77da7bfa3860.jpg)

![71b784bb8e2b721b908debbc3d5af76b5990503a828e1f13ba3328e5da128622.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/71b784bb8e2b721b908debbc3d5af76b5990503a828e1f13ba3328e5da128622.jpg)

![81aeb2213b146afb074897d457e0f4e37d578094d9c63768bf9cc3d51af7bd92.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/81aeb2213b146afb074897d457e0f4e37d578094d9c63768bf9cc3d51af7bd92.jpg)

![9044824eac634954b7397fe84ba221da2510ebb6cd825e204f2d561f996a0f31.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/9044824eac634954b7397fe84ba221da2510ebb6cd825e204f2d561f996a0f31.jpg)

![a5e69d65fbd91f722ea20d148c85281dd58345b82771c381164f9e786a305b3e.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/a5e69d65fbd91f722ea20d148c85281dd58345b82771c381164f9e786a305b3e.jpg)

![d77909ac401d79e9953519dfd5d18b7e60530cd8585e482ae07f67d18610e1fb.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/d77909ac401d79e9953519dfd5d18b7e60530cd8585e482ae07f67d18610e1fb.jpg)

![e3301dad14e53e38dfac586f6e2ea9059c67b01440c78dc16eb91a6a6d0ec478.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/e3301dad14e53e38dfac586f6e2ea9059c67b01440c78dc16eb91a6a6d0ec478.jpg)

![e53f94a02f484af30a7006f416622d250b85b7a0a14c089e7b5aaf8e27ec02ab.jpg](../icml_results/5811_One%20Tool%20Is%20Enough_%20Reinforcement%20Learning%20of%20LLM%20Agents%20for%20Repository-Level%20Code%20Navigation/tables/e53f94a02f484af30a7006f416622d250b85b7a0a14c089e7b5aaf8e27ec02ab.jpg)

<a id="paper-5763"></a>

## Spatiotemporal Imputation with Graph-Informed Flow Matching


### Images

![0fb7d15502a2c430500750216d450147d972020ba846eb2fb05e070070b4b46b.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/images/0fb7d15502a2c430500750216d450147d972020ba846eb2fb05e070070b4b46b.jpg)

![36c28ca4e62f32fe773685448fe898a4bd5c0f3577d4a5004d9818e2532b02cf.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/images/36c28ca4e62f32fe773685448fe898a4bd5c0f3577d4a5004d9818e2532b02cf.jpg)

![4c1a7e1330942cadeecf84696652100c66d8df2928389bfa6a34bf09d977a0ad.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/images/4c1a7e1330942cadeecf84696652100c66d8df2928389bfa6a34bf09d977a0ad.jpg)

![a9c905848152f37cab035635c260eb0aabc0d9166be88ebaf678cd88630a443f.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/images/a9c905848152f37cab035635c260eb0aabc0d9166be88ebaf678cd88630a443f.jpg)

![a9d921db8f8249dba6de1bf0a7255aa0da7e8c38e5932c6e6812534066d3a650.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/images/a9d921db8f8249dba6de1bf0a7255aa0da7e8c38e5932c6e6812534066d3a650.jpg)

### Tables

![1955b4c31c87e276bac0b3f6f0d7ef31c7575eeece8405479d53e09c978067e5.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/1955b4c31c87e276bac0b3f6f0d7ef31c7575eeece8405479d53e09c978067e5.jpg)

![21860799e586ee82872a79270efc32972897a4709e3a499cfed60e8facfa3fe4.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/21860799e586ee82872a79270efc32972897a4709e3a499cfed60e8facfa3fe4.jpg)

![2aa0cd0515c6eeae8b8bfb9ccfcf9e69ed4403971fc00f6189f258934f7b9161.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/2aa0cd0515c6eeae8b8bfb9ccfcf9e69ed4403971fc00f6189f258934f7b9161.jpg)

![3012b91ba744684bc6bceda372665104d09a8cb805372a446de2669e6d9568cb.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/3012b91ba744684bc6bceda372665104d09a8cb805372a446de2669e6d9568cb.jpg)

![5679314ccb241c5ba15b3aeccd8443df042b204e0d18d5fb6f76711f7e362545.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/5679314ccb241c5ba15b3aeccd8443df042b204e0d18d5fb6f76711f7e362545.jpg)

![57b2774442f52be333e265304ec9f14f77f6a77e5e2898fb0bce382e4e6cfeb0.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/57b2774442f52be333e265304ec9f14f77f6a77e5e2898fb0bce382e4e6cfeb0.jpg)

![7b6e7d521b66056a28d1962836153b33de19c3ff6d9f55e6009a67302b171fa0.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/7b6e7d521b66056a28d1962836153b33de19c3ff6d9f55e6009a67302b171fa0.jpg)

![8630d81ef42f249fa37f47d044b99093d4e2d5074c0a987e525e21af1c606a5b.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/8630d81ef42f249fa37f47d044b99093d4e2d5074c0a987e525e21af1c606a5b.jpg)

![a552ed5376af9ee7266ffcacc7c3fe7b5811484abd666d6af889d422fd106098.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/a552ed5376af9ee7266ffcacc7c3fe7b5811484abd666d6af889d422fd106098.jpg)

![c38a76dfc2eaec9c636674784f6b6ad12379a99f7a761f0986dfa8e2754888fa.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/c38a76dfc2eaec9c636674784f6b6ad12379a99f7a761f0986dfa8e2754888fa.jpg)

![fa21924eca9c8a4260c79b3b8b53afa057a2c0630bf69abc660770fe63e9bebf.jpg](../icml_results/5812_Spatiotemporal%20Imputation%20with%20Graph-Informed%20Flow%20Matching/tables/fa21924eca9c8a4260c79b3b8b53afa057a2c0630bf69abc660770fe63e9bebf.jpg)

<a id="paper-5764"></a>

## SOLAR for Offline MARL: Plateau-Triggered Potential Shaping under World-Model Uncertainty


### Images

![35f9502da1122cabf2284b809a7ca95248e38d84644987a38a511c21574325c5.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/images/35f9502da1122cabf2284b809a7ca95248e38d84644987a38a511c21574325c5.jpg)

![3cca03e35f26264dca41d7402deca23e0c2146fa05d037dfd169eba270ea5136.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/images/3cca03e35f26264dca41d7402deca23e0c2146fa05d037dfd169eba270ea5136.jpg)

![a2106af93825a32bd9591bf6992a87e4f29a1de2ef6dcf4338d9a47aba26e72b.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/images/a2106af93825a32bd9591bf6992a87e4f29a1de2ef6dcf4338d9a47aba26e72b.jpg)

![b1e08934949fb084faec64a2563309f41ddd74178847a6c9fa4047a339afe954.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/images/b1e08934949fb084faec64a2563309f41ddd74178847a6c9fa4047a339afe954.jpg)

![d58faf49a6096997b7962cffae94b35077e3694afc58ec6075fd5d2bda044c5d.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/images/d58faf49a6096997b7962cffae94b35077e3694afc58ec6075fd5d2bda044c5d.jpg)

### Tables

![1b545b0e00eb609a732970206902f94840a429c5456230b83f7f50988c026848.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/1b545b0e00eb609a732970206902f94840a429c5456230b83f7f50988c026848.jpg)

![2bbfb3ad99961e1e3174a6db9c93fdf0511a8f3f969f8b27b4cacdc6365e49fa.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/2bbfb3ad99961e1e3174a6db9c93fdf0511a8f3f969f8b27b4cacdc6365e49fa.jpg)

![61259b86f80f1cd004a6329f09dde23307b0d3363c70cd1879155369eff588df.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/61259b86f80f1cd004a6329f09dde23307b0d3363c70cd1879155369eff588df.jpg)

![711337ffe4b572dfffb6cf3bdaa3ebeb0a274b732286ff4a2c6dfd83bb8d54ac.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/711337ffe4b572dfffb6cf3bdaa3ebeb0a274b732286ff4a2c6dfd83bb8d54ac.jpg)

![79c950e341ade0bb300acbcaf72b4a5be75eefda650baf350d96053d1accefe2.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/79c950e341ade0bb300acbcaf72b4a5be75eefda650baf350d96053d1accefe2.jpg)

![c33e2f3227f80d676f56d2c51dde8617fc5e00217fe7dd0784f4a2f72cf65790.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/c33e2f3227f80d676f56d2c51dde8617fc5e00217fe7dd0784f4a2f72cf65790.jpg)

![e270d63a1231be9c27f4d692c6e332a7cd84550d8af0222cebbc02876a25271b.jpg](../icml_results/5813_SOLAR%20for%20Offline%20MARL_%20Plateau-Triggered%20Potential%20Shaping%20under%20World-Model%20Uncertainty/tables/e270d63a1231be9c27f4d692c6e332a7cd84550d8af0222cebbc02876a25271b.jpg)

<a id="paper-5765"></a>

## DeepAnalyze: Agentic Large Language Models for Autonomous Data Science


### Images

![021a47d48263826990739eafef11a530d9d513f4857794b2026a950a4d5811a3.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/021a47d48263826990739eafef11a530d9d513f4857794b2026a950a4d5811a3.jpg)

![0cfb9e4c1b09adf624306ce7be77a71029f27890956f20d4f3ce254207b2f81e.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/0cfb9e4c1b09adf624306ce7be77a71029f27890956f20d4f3ce254207b2f81e.jpg)

![1119aa29028e822e58fa6604a8ea7da68defdd60a14d3aa9966837bc3ae6abbc.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/1119aa29028e822e58fa6604a8ea7da68defdd60a14d3aa9966837bc3ae6abbc.jpg)

![12860a96d77707e7d65ce334fdbe36a0de76e099eb534c635b8a093850603886.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/12860a96d77707e7d65ce334fdbe36a0de76e099eb534c635b8a093850603886.jpg)

![5367bb903781c9462deb8795939998c8a24a3fa3132dabf9a16700da8068750c.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/5367bb903781c9462deb8795939998c8a24a3fa3132dabf9a16700da8068750c.jpg)

![64f88291e35a84285e2f850dc1c9dcdd25737c87e27cb36aeb84a649b2eb0269.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/64f88291e35a84285e2f850dc1c9dcdd25737c87e27cb36aeb84a649b2eb0269.jpg)

![74aeb27211a9a13934b72027205d6f1ea3b8612d3d74e8293e0853860fae1600.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/74aeb27211a9a13934b72027205d6f1ea3b8612d3d74e8293e0853860fae1600.jpg)

![7a47a22db5249448a71599a7df79b7d4ea273938342b853aa1ca84b7e088ce8a.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/7a47a22db5249448a71599a7df79b7d4ea273938342b853aa1ca84b7e088ce8a.jpg)

![82dfa4b5afc39bd1908d0cd40092c2683b3ce5df852934ee7d39745b65da97c1.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/82dfa4b5afc39bd1908d0cd40092c2683b3ce5df852934ee7d39745b65da97c1.jpg)

![a57455bc03e17de7b7951ab6ee78c25104eb89393c33d6e7c9510ad74669d966.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/a57455bc03e17de7b7951ab6ee78c25104eb89393c33d6e7c9510ad74669d966.jpg)

![a9d7ef61a0c763fbb8fd5bea74ac3b844253fc4dbe0ca22bbc15b065411b1e91.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/a9d7ef61a0c763fbb8fd5bea74ac3b844253fc4dbe0ca22bbc15b065411b1e91.jpg)

![aaa25804178381e7a7975efc6fb894b623b18bfd08029a5b855feb42b9dafee5.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/aaa25804178381e7a7975efc6fb894b623b18bfd08029a5b855feb42b9dafee5.jpg)

![bda970133daa4c22985d1da16830eddd6897168e69c9973344706d6f5f8f4452.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/bda970133daa4c22985d1da16830eddd6897168e69c9973344706d6f5f8f4452.jpg)

![d7979abdd02a7b09aa3527b349a13d8f2cd5245edc1003dc089c96ecaf0c4e64.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/d7979abdd02a7b09aa3527b349a13d8f2cd5245edc1003dc089c96ecaf0c4e64.jpg)

![de7660c2b84dd877fa0f73dcb37640de2fb569c57014c0aa6de37533ead8cf86.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/de7660c2b84dd877fa0f73dcb37640de2fb569c57014c0aa6de37533ead8cf86.jpg)

![f2bff88634b51ba8fd85aab26820a8360b25fc560330bb49aba1f82d65d1db2c.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/f2bff88634b51ba8fd85aab26820a8360b25fc560330bb49aba1f82d65d1db2c.jpg)

![f88a46e6405d1424391b9f77c3d7cb823dc13a16d436a0a4a3c50edd51679cf0.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/images/f88a46e6405d1424391b9f77c3d7cb823dc13a16d436a0a4a3c50edd51679cf0.jpg)

### Tables

![1fb65f65e74240a846493be3b9120d649a626b447ca2ccdf74273f6f171c3c06.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/1fb65f65e74240a846493be3b9120d649a626b447ca2ccdf74273f6f171c3c06.jpg)

![26c6b44498779cb6555a4e19cd41059fd99bd0f0c27d37b156ae3ef06ee7941b.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/26c6b44498779cb6555a4e19cd41059fd99bd0f0c27d37b156ae3ef06ee7941b.jpg)

![29601d3f67ad9c4dcc12febb03c6e20afbc0f1049b6ad09ee4a219a41672773b.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/29601d3f67ad9c4dcc12febb03c6e20afbc0f1049b6ad09ee4a219a41672773b.jpg)

![3292aff60e24455e1109a524ca2570dfaf7f12f02f717967258ab0a08517c184.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/3292aff60e24455e1109a524ca2570dfaf7f12f02f717967258ab0a08517c184.jpg)

![33e314fccca70b36f43516447febddd5feb3b74c2b08d2b1319342958a0fd076.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/33e314fccca70b36f43516447febddd5feb3b74c2b08d2b1319342958a0fd076.jpg)

![3e1653736792e2262993b4c6d77a837bcfa2ddd2a6ea5fc9c6db402307012c18.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/3e1653736792e2262993b4c6d77a837bcfa2ddd2a6ea5fc9c6db402307012c18.jpg)

![881b77b111efe3369255a7468afcec3d4a3f8b6af68effd47bc6e570c3ef0312.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/881b77b111efe3369255a7468afcec3d4a3f8b6af68effd47bc6e570c3ef0312.jpg)

![a03ae91438dc8fd90c8d7521023e81b4b9db8981d8ac054d5ef1ecbc398ef20e.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/a03ae91438dc8fd90c8d7521023e81b4b9db8981d8ac054d5ef1ecbc398ef20e.jpg)

![a883fc00c20c4dfa29cb1032714e2c9fa7b2e78578d750a94d9b20d78c38e327.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/a883fc00c20c4dfa29cb1032714e2c9fa7b2e78578d750a94d9b20d78c38e327.jpg)

![a91ba8d87753ce17539e21f42f474a54882418dcbe1da4dd8c703ad5644c2171.jpg](../icml_results/5814_DeepAnalyze_%20Agentic%20Large%20Language%20Models%20for%20Autonomous%20Data%20Science/tables/a91ba8d87753ce17539e21f42f474a54882418dcbe1da4dd8c703ad5644c2171.jpg)

<a id="paper-5766"></a>

## CoDA-Bench: Can Code Agents Handle Data-Intensive Tasks?


### Images

![099bd77455093e5897c9511ce9553d67a0fa79bbcf854b9a7d2066517e4d7a32.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/099bd77455093e5897c9511ce9553d67a0fa79bbcf854b9a7d2066517e4d7a32.jpg)

![1945b8091fcdd574f9547d1811120bf7dba33cf097982234fc453fc9a0e906c1.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/1945b8091fcdd574f9547d1811120bf7dba33cf097982234fc453fc9a0e906c1.jpg)

![299ddc06b380d2f9dc6a49891509fcd95e5c6f7cfbb966ddc7b691cf73fdb17a.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/299ddc06b380d2f9dc6a49891509fcd95e5c6f7cfbb966ddc7b691cf73fdb17a.jpg)

![2b08d2496c954706b67fb2f9612620c9f2fe62c1934f5c43b053a2e5349dfed9.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/2b08d2496c954706b67fb2f9612620c9f2fe62c1934f5c43b053a2e5349dfed9.jpg)

![35279ca54c69828ef3842b5116ca59bcaa135a8b2a885f8d42c271edfd97aca7.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/35279ca54c69828ef3842b5116ca59bcaa135a8b2a885f8d42c271edfd97aca7.jpg)

![45d48c5ea7737c8d5345e1524e28dbb245738f0b1432084ee97fc5c5f23150a2.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/45d48c5ea7737c8d5345e1524e28dbb245738f0b1432084ee97fc5c5f23150a2.jpg)

![8200aa07124f4f702ece79c29c5ec4df7fbc6fcd29f831bd715fb1f319dc06b1.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/8200aa07124f4f702ece79c29c5ec4df7fbc6fcd29f831bd715fb1f319dc06b1.jpg)

![8aa0b7d335a96c51f8a2801a79f6a8da6c1d9ac5f831284391de281b799382d3.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/8aa0b7d335a96c51f8a2801a79f6a8da6c1d9ac5f831284391de281b799382d3.jpg)

![beb16aeb2200b7d859896ee5c2431f31dfcc979200c0052db63a979f6c6f92df.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/beb16aeb2200b7d859896ee5c2431f31dfcc979200c0052db63a979f6c6f92df.jpg)

![da3fdc093c9e4fc4702606db0cb0151b2219956a93fc20632e5f17418193b813.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/da3fdc093c9e4fc4702606db0cb0151b2219956a93fc20632e5f17418193b813.jpg)

![e5ed7ff70a8dcda12386655b435017e340001f6b54a51ab231207f7746a1f205.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/e5ed7ff70a8dcda12386655b435017e340001f6b54a51ab231207f7746a1f205.jpg)

![f59d74690600af1c2fd0459ef5379e8f1a03922bb827b63f5aab41c9a5827117.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/images/f59d74690600af1c2fd0459ef5379e8f1a03922bb827b63f5aab41c9a5827117.jpg)

### Tables

![3e2d7a514e51c12704b861f816c823415e655cb1c904006eb0caff226daa0df0.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/3e2d7a514e51c12704b861f816c823415e655cb1c904006eb0caff226daa0df0.jpg)

![58177b0d66f5d89fe99cb748c7a932690a18120a3a4f4760c31d602b90cf2979.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/58177b0d66f5d89fe99cb748c7a932690a18120a3a4f4760c31d602b90cf2979.jpg)

![77116ef57a856b42954ead8b4a2076407d81206bd971bbcf83a543d2808d9a76.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/77116ef57a856b42954ead8b4a2076407d81206bd971bbcf83a543d2808d9a76.jpg)

![804156333a8d011cf09a063dc0da7636bde17ecfdeb733970f67db03082dbec7.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/804156333a8d011cf09a063dc0da7636bde17ecfdeb733970f67db03082dbec7.jpg)

![829b0a4af01fff5cb8ff9b485a121e79523dcb14b8307296afb285fde04c1589.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/829b0a4af01fff5cb8ff9b485a121e79523dcb14b8307296afb285fde04c1589.jpg)

![be481fd715724a978b1b1384f078b8bb6cc3c073e99c18a3519aa95b9b5a3b04.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/be481fd715724a978b1b1384f078b8bb6cc3c073e99c18a3519aa95b9b5a3b04.jpg)

![cd2a36f2add53c7c933b6319bb6ba9bea7b2daa0dbfbe45ba082070551519ca6.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/cd2a36f2add53c7c933b6319bb6ba9bea7b2daa0dbfbe45ba082070551519ca6.jpg)

![ceb9a7351fdb85eb3d5fb15dc5a29f974ddfb5d68e5b6fd8c6b66ec2f7d2f9e5.jpg](../icml_results/5815_CoDA-Bench_%20Can%20Code%20Agents%20Handle%20Data-Intensive%20Tasks_/tables/ceb9a7351fdb85eb3d5fb15dc5a29f974ddfb5d68e5b6fd8c6b66ec2f7d2f9e5.jpg)

<a id="paper-5767"></a>

## Breaking the Synthetic-Real Domain Shortcut for Training-Free Generative Replay-based Class Incremental Learning


### Images

![072eec483507c84f7495eae1b92bdb555b9a33ddcf785ab854947d20d2aa9343.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/072eec483507c84f7495eae1b92bdb555b9a33ddcf785ab854947d20d2aa9343.jpg)

![1965467851203a526169aae6dfe122afa44e14cf7bdec76658c4d39a7fe9f016.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/1965467851203a526169aae6dfe122afa44e14cf7bdec76658c4d39a7fe9f016.jpg)

![30e7cd946b206311bfdc6ef919294b58942cc15a49c5c6f19b9e7bb471f1e0ee.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/30e7cd946b206311bfdc6ef919294b58942cc15a49c5c6f19b9e7bb471f1e0ee.jpg)

![36cc09dc40d5d9350ec218409341ad75d8e00f78ea161bab6b6c69bf4a1ab9ae.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/36cc09dc40d5d9350ec218409341ad75d8e00f78ea161bab6b6c69bf4a1ab9ae.jpg)

![37394830764c4ce4fd4f6637a9b73bafab5bd00020156e2bc4fad754ad6977cb.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/37394830764c4ce4fd4f6637a9b73bafab5bd00020156e2bc4fad754ad6977cb.jpg)

![3b7938a1ad32ba8330bb1bc68f755a15f4ba00829d0f1f60dba4b95bf7572772.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/3b7938a1ad32ba8330bb1bc68f755a15f4ba00829d0f1f60dba4b95bf7572772.jpg)

![3baeaf4b34b32187be3a2c2f292c75f3e0bfc48f4bf82c31b78d5d85062a1734.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/3baeaf4b34b32187be3a2c2f292c75f3e0bfc48f4bf82c31b78d5d85062a1734.jpg)

![4bed7ba3b9be4b70d32b6257817bb6f2402c3e0d8821a3b0ab149ce333f9544a.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/4bed7ba3b9be4b70d32b6257817bb6f2402c3e0d8821a3b0ab149ce333f9544a.jpg)

![4c72df39cf3dd7e5bdb46d10bd891902c2efa457317764a12e10397ee1588513.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/4c72df39cf3dd7e5bdb46d10bd891902c2efa457317764a12e10397ee1588513.jpg)

![4c8f7fd822af99b8f4700602a6a989a2fbbf063170dde0690d9c102280378c9c.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/4c8f7fd822af99b8f4700602a6a989a2fbbf063170dde0690d9c102280378c9c.jpg)

![691ae5e7c102e8004d47e8d33ba6b09559b9935ad29cea942a68850fa90a493d.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/691ae5e7c102e8004d47e8d33ba6b09559b9935ad29cea942a68850fa90a493d.jpg)

![7dbe8671b1d994ec2a1c6b4200895148816a8caa3046d37b01173c17befed1bd.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/7dbe8671b1d994ec2a1c6b4200895148816a8caa3046d37b01173c17befed1bd.jpg)

![84eefca5bebf4443a7d2bd11c91a9e1147d42472535266b1b3b2ac607666b5f3.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/84eefca5bebf4443a7d2bd11c91a9e1147d42472535266b1b3b2ac607666b5f3.jpg)

![92cdef59859fce16484726fcaf14b9a6069c7444c0f1cc2932bfe9b7c4774900.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/92cdef59859fce16484726fcaf14b9a6069c7444c0f1cc2932bfe9b7c4774900.jpg)

![9cb15c1b26843d331d40fb227690d1eb2da8330c21c4a015ebe27a2e35f9b00a.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/9cb15c1b26843d331d40fb227690d1eb2da8330c21c4a015ebe27a2e35f9b00a.jpg)

![9ebd5d5ea79d919f716c7908b18328fd3a027d1c327349f9d462ab1c8b160010.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/9ebd5d5ea79d919f716c7908b18328fd3a027d1c327349f9d462ab1c8b160010.jpg)

![a111245932fc287a7f102d494bf4345c7a5894034a1b39b94f81c9cae8b4e718.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/a111245932fc287a7f102d494bf4345c7a5894034a1b39b94f81c9cae8b4e718.jpg)

![af815efa4396c194ad90909fe3f9739063b8fe020ed1f960dfa81d9837e114eb.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/af815efa4396c194ad90909fe3f9739063b8fe020ed1f960dfa81d9837e114eb.jpg)

![c2a1fe5e3eaabe675132248cbc3357467e366abf932ec21b2aabda903ebf3703.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/c2a1fe5e3eaabe675132248cbc3357467e366abf932ec21b2aabda903ebf3703.jpg)

![c5f741ee0dcaf87b1a2676e78f2ff9d4b8c648b498879bb23d5ac821656079b7.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/c5f741ee0dcaf87b1a2676e78f2ff9d4b8c648b498879bb23d5ac821656079b7.jpg)

![d7bffb6d0d4d3698d17052df692038e04cefe4cb49cae9c92bb22bfb18df8e75.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/d7bffb6d0d4d3698d17052df692038e04cefe4cb49cae9c92bb22bfb18df8e75.jpg)

![e32b17717f7c0416927424a589d9f1872a3b40f8687acb2a9d1e72156b61f50f.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/e32b17717f7c0416927424a589d9f1872a3b40f8687acb2a9d1e72156b61f50f.jpg)

![f0fe63949fe3910b0630287dd7b80f2c0be2959e3b8d85e8fe2822a5433cfa13.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/f0fe63949fe3910b0630287dd7b80f2c0be2959e3b8d85e8fe2822a5433cfa13.jpg)

![f7859b949f75ffe726b4170df7c6c046b65b1808955cfa8fbfadf2f783930c8f.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/images/f7859b949f75ffe726b4170df7c6c046b65b1808955cfa8fbfadf2f783930c8f.jpg)

### Tables

![03228173228233ea7cda6bc6c960b131c32fa21288919bc3841e2a2fce4e9192.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/03228173228233ea7cda6bc6c960b131c32fa21288919bc3841e2a2fce4e9192.jpg)

![2803ae378ce3401030716352ecbe6c120a650c9a692f0c56c754482dfc34df85.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/2803ae378ce3401030716352ecbe6c120a650c9a692f0c56c754482dfc34df85.jpg)

![4499d70f8bafcff79d3746cf3cee8d577a26bf75142c618d34b113c3d120ec81.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/4499d70f8bafcff79d3746cf3cee8d577a26bf75142c618d34b113c3d120ec81.jpg)

![45567b3cbe4381f1d689d6a8099ccd1721ef780f80789bb477c3b512a1c07acd.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/45567b3cbe4381f1d689d6a8099ccd1721ef780f80789bb477c3b512a1c07acd.jpg)

![4c8f18a509a00eca3f32279ec4a60e6d92d97ae4128bac07474d699f9cd2fd91.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/4c8f18a509a00eca3f32279ec4a60e6d92d97ae4128bac07474d699f9cd2fd91.jpg)

![593376c51c7003a2a0391feb25712c65d9f539d0c0d093d5d8f2a1f60de7975a.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/593376c51c7003a2a0391feb25712c65d9f539d0c0d093d5d8f2a1f60de7975a.jpg)

![72e79a4769bec76338606e617e2667ba23b03fa2f8eeee515207f3dc82fccd6d.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/72e79a4769bec76338606e617e2667ba23b03fa2f8eeee515207f3dc82fccd6d.jpg)

![7a1c983a4cca1b143c3964f0012260a39db93738bdeb13381169dbaff2da753a.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/7a1c983a4cca1b143c3964f0012260a39db93738bdeb13381169dbaff2da753a.jpg)

![7dd2c23ada9778d061c40bd2332dc6ca35cae025009dccc5564cab93a8d4d0b1.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/7dd2c23ada9778d061c40bd2332dc6ca35cae025009dccc5564cab93a8d4d0b1.jpg)

![8a2780392cca2b1f07f41bac4eaf42f575e4f5cf21b30e409c79bb038f42ce49.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/8a2780392cca2b1f07f41bac4eaf42f575e4f5cf21b30e409c79bb038f42ce49.jpg)

![91b46581ffe8b8f55c3f47b0005d6c3b656109b9ea3b05065eaecb5320530eae.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/91b46581ffe8b8f55c3f47b0005d6c3b656109b9ea3b05065eaecb5320530eae.jpg)

![923e141adbd353225fa349e3fd258075269f6346b5e33e00b1bb0b5dc1ba1407.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/923e141adbd353225fa349e3fd258075269f6346b5e33e00b1bb0b5dc1ba1407.jpg)

![9f9a4a70d79465aebebe0a9da25977aaaf8c7f7fc72b887da8ef456ba3a7e599.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/9f9a4a70d79465aebebe0a9da25977aaaf8c7f7fc72b887da8ef456ba3a7e599.jpg)

![a36ab50ed29acc76c76d2950647c112dd3898801788312d5f594f47e71eb9197.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/a36ab50ed29acc76c76d2950647c112dd3898801788312d5f594f47e71eb9197.jpg)

![aaaea2c8bfd5455602b95395ffb914db4c92c389db8675e0db2fdead7c16cd52.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/aaaea2c8bfd5455602b95395ffb914db4c92c389db8675e0db2fdead7c16cd52.jpg)

![b6e61fd30f1581286793a1813215ba3fb12ef0074e2dc5636b0ad10577bba6d5.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/b6e61fd30f1581286793a1813215ba3fb12ef0074e2dc5636b0ad10577bba6d5.jpg)

![b7aba1e386de8157be9bc1c76e525a88ca0768b717271c1fb291c8395fd4360a.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/b7aba1e386de8157be9bc1c76e525a88ca0768b717271c1fb291c8395fd4360a.jpg)

![ba32a737b9ebe26c8a94b46cf4578a8c59ed10aea3bd834709430aecee0209c1.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/ba32a737b9ebe26c8a94b46cf4578a8c59ed10aea3bd834709430aecee0209c1.jpg)

![be1b1ab8f264991b76660f72b5d242d6e4ad1ad7d41ed9592193c50b3b130e21.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/be1b1ab8f264991b76660f72b5d242d6e4ad1ad7d41ed9592193c50b3b130e21.jpg)

![c24fb94b208debe20cf161e6d48e40363415bf0a24c91a2fe824689f482fcf93.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/c24fb94b208debe20cf161e6d48e40363415bf0a24c91a2fe824689f482fcf93.jpg)

![d453a86560bfd0fe931a441b0ba763ca96ef6b79ef3516c7d4f1a574b7f9fd00.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/d453a86560bfd0fe931a441b0ba763ca96ef6b79ef3516c7d4f1a574b7f9fd00.jpg)

![e34a6fb3e545eaf8c29d27666bf351765d80281d2fe4d67c6fa5faf63c38c098.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/e34a6fb3e545eaf8c29d27666bf351765d80281d2fe4d67c6fa5faf63c38c098.jpg)

![ed8486480322729d85c09d2bc028b5ae08e3e60b509f74f1466db3e78a2a2494.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/ed8486480322729d85c09d2bc028b5ae08e3e60b509f74f1466db3e78a2a2494.jpg)

![f61513ed734372ad4ba4a6233482c376bf6286fffe990c1b425cfb0df46a2e74.jpg](../icml_results/5816_Breaking%20the%20Synthetic-Real%20Domain%20Shortcut%20for%20Training-Free%20Generative%20Replay-based%20Class%20Incremen/tables/f61513ed734372ad4ba4a6233482c376bf6286fffe990c1b425cfb0df46a2e74.jpg)

<a id="paper-5768"></a>

## What If We Let Forecasting Forget? A Sparse Bottleneck for Cross-Variable Dependencies


### Images

![000d6e8c2c71162106bc170f1c3824625946ce86c558279477861010b9217b7d.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/000d6e8c2c71162106bc170f1c3824625946ce86c558279477861010b9217b7d.jpg)

![0514c7e6ae479e40892231b9d85d9c69d178a2da4ad766dbc79361a2eae34668.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/0514c7e6ae479e40892231b9d85d9c69d178a2da4ad766dbc79361a2eae34668.jpg)

![20b00dae2f1801b243d378424fc10440a5d24064daeae628f71f6d85d4178ae5.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/20b00dae2f1801b243d378424fc10440a5d24064daeae628f71f6d85d4178ae5.jpg)

![26236ae65792861dc39eed4e76dfad248be010ed3acf9b18d3ebc0425dca9a86.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/26236ae65792861dc39eed4e76dfad248be010ed3acf9b18d3ebc0425dca9a86.jpg)

![6314e5f0e3fdd2bdcbb95b0c4ac82782ab56c703db75115d58b58a4788b95e14.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/6314e5f0e3fdd2bdcbb95b0c4ac82782ab56c703db75115d58b58a4788b95e14.jpg)

![aeb548bf117f8a30cf9fcb19ea233026dc9cd77db3d94077fe197febfb085c3b.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/aeb548bf117f8a30cf9fcb19ea233026dc9cd77db3d94077fe197febfb085c3b.jpg)

![caa6b5e8714c0119896fb2e9f04d132442204528787bcc4d06d9864c1e4a1376.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/caa6b5e8714c0119896fb2e9f04d132442204528787bcc4d06d9864c1e4a1376.jpg)

![d13d28850d5b9dbabfe9c55f527af403bf1a3cb03b91e5808466663c9dbee794.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/d13d28850d5b9dbabfe9c55f527af403bf1a3cb03b91e5808466663c9dbee794.jpg)

![dfafcd16692777ea33923682c33e30ad02081c8c5cba22d75f10442a720ba51c.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/dfafcd16692777ea33923682c33e30ad02081c8c5cba22d75f10442a720ba51c.jpg)

![e834b904007210e0d6706d69fc37c3939bc600fb8b2ddd7df654288b5b44223c.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/e834b904007210e0d6706d69fc37c3939bc600fb8b2ddd7df654288b5b44223c.jpg)

![e9754682706fbdc5a3ae62e0bedca94f041ff1e8282111524619aa5c80293acd.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/e9754682706fbdc5a3ae62e0bedca94f041ff1e8282111524619aa5c80293acd.jpg)

![f21061a3777e6532a4e450f2aac946295ed12d743ba39818b4e240210dc5a764.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/images/f21061a3777e6532a4e450f2aac946295ed12d743ba39818b4e240210dc5a764.jpg)

### Tables

![029420c610d723be118185bed4fd1eaaf586dedf8feacf8036d63a6e07801aa8.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/029420c610d723be118185bed4fd1eaaf586dedf8feacf8036d63a6e07801aa8.jpg)

![07999bd66acb8c33d941d5e3830e7ed659b09edce4a2cd98a3f62529277c5c86.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/07999bd66acb8c33d941d5e3830e7ed659b09edce4a2cd98a3f62529277c5c86.jpg)

![1a85b4362f611dd323aeefedbeaaa465b6a9f5b766e316be6a53a72103a6112e.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/1a85b4362f611dd323aeefedbeaaa465b6a9f5b766e316be6a53a72103a6112e.jpg)

![2407bc7fb0410283297d2242d719efb983c1e6ff3be5840873a7fb3475675e28.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/2407bc7fb0410283297d2242d719efb983c1e6ff3be5840873a7fb3475675e28.jpg)

![2cd4ce4d5cbfb176b8e83d52750b572ea434b82b07022986eaa2c592dd7a2b78.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/2cd4ce4d5cbfb176b8e83d52750b572ea434b82b07022986eaa2c592dd7a2b78.jpg)

![668b2caed6fdcb7ff0aa167fdebac92b750ac02adb8452dd25e47211d47e3625.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/668b2caed6fdcb7ff0aa167fdebac92b750ac02adb8452dd25e47211d47e3625.jpg)

![698e3d5ba4f242a3937c4363905c4d9d8cad5ea7a798ad97bb0e14af744774e8.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/698e3d5ba4f242a3937c4363905c4d9d8cad5ea7a798ad97bb0e14af744774e8.jpg)

![a03aeb410b0f66b3ead9d8b38c7f94b7fedd7dfdaa39036c2663824051dd1eff.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/a03aeb410b0f66b3ead9d8b38c7f94b7fedd7dfdaa39036c2663824051dd1eff.jpg)

![c75642a62e7f09b76c1107b9b46b98e80e63e5216d38286ffd8f6d4730b42749.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/c75642a62e7f09b76c1107b9b46b98e80e63e5216d38286ffd8f6d4730b42749.jpg)

![dfbf0188b8dda45482bcb6f7898a83f1c014942106910a0775e004be11afb897.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/dfbf0188b8dda45482bcb6f7898a83f1c014942106910a0775e004be11afb897.jpg)

![f97d258eca6bf9399d8f3c4c61625deae4af303bd31a8fa7b6270498fee80c0d.jpg](../icml_results/5817_What%20If%20We%20Let%20Forecasting%20Forget_%20A%20Sparse%20Bottleneck%20for%20Cross-Variable%20Dependencies/tables/f97d258eca6bf9399d8f3c4c61625deae4af303bd31a8fa7b6270498fee80c0d.jpg)

<a id="paper-5769"></a>

## Self-evolving LLM agents with in-distribution Optimization


### Images

![5d239b9ab1510bc08c2503f6a159840208f881cf4815e9805403ee4fea2b4a04.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/images/5d239b9ab1510bc08c2503f6a159840208f881cf4815e9805403ee4fea2b4a04.jpg)

![671da5b92ad22ac86ef609b4f4889f35b60fb0a685d8558efbdb3054df8dc0e9.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/images/671da5b92ad22ac86ef609b4f4889f35b60fb0a685d8558efbdb3054df8dc0e9.jpg)

![b168459bd5cb035eb5259fd2ba58c48ea4b7c44c5681d9868116c1a65a201cf2.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/images/b168459bd5cb035eb5259fd2ba58c48ea4b7c44c5681d9868116c1a65a201cf2.jpg)

![d31f5ebd47b8f71f2a380855babdb36290c8236023003a4b0c15f0ee334c36c4.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/images/d31f5ebd47b8f71f2a380855babdb36290c8236023003a4b0c15f0ee334c36c4.jpg)

### Tables

![017b34903d9b3f2b25950d8b488240cb5dc71d38d0266b878033975aee8841dd.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/017b34903d9b3f2b25950d8b488240cb5dc71d38d0266b878033975aee8841dd.jpg)

![1a5649add826d4fa0a2f381f22669d59289e24a44630550b9c0b3a5f7f6cff13.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/1a5649add826d4fa0a2f381f22669d59289e24a44630550b9c0b3a5f7f6cff13.jpg)

![251ee9f40f2edc64b1b898f55daca42f88681232bd13b1e09b223e8e4f359a57.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/251ee9f40f2edc64b1b898f55daca42f88681232bd13b1e09b223e8e4f359a57.jpg)

![2e7eec2a1170a3086ad5a29af81f0ef1a484b554fedeb93c064c67b45ecdff11.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/2e7eec2a1170a3086ad5a29af81f0ef1a484b554fedeb93c064c67b45ecdff11.jpg)

![30d1ad2bf1b5b1dbbbae5f232397dd1e4620d7df75a1368f3d8d5375ab848e00.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/30d1ad2bf1b5b1dbbbae5f232397dd1e4620d7df75a1368f3d8d5375ab848e00.jpg)

![38c67b428191079f22b15d9c7e3b469e0d7645f460ac7786d936b4c7493100a7.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/38c67b428191079f22b15d9c7e3b469e0d7645f460ac7786d936b4c7493100a7.jpg)

![4bdae53d0b0e8497d4cbb54cb223f6417ccca0a3f58aa9958280ca4e062dcbb9.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/4bdae53d0b0e8497d4cbb54cb223f6417ccca0a3f58aa9958280ca4e062dcbb9.jpg)

![4e52c8b96e312eb6829544d8f0a59990dff01072763b0a6875880a4334d8cbf1.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/4e52c8b96e312eb6829544d8f0a59990dff01072763b0a6875880a4334d8cbf1.jpg)

![681fec1b2a07a48a291464c6a9f42f489dded58b8e9d85ae65cb1b840cac0ef3.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/681fec1b2a07a48a291464c6a9f42f489dded58b8e9d85ae65cb1b840cac0ef3.jpg)

![82ec590d6fa0e8a53193444d88a6575c09c9a9ff5f30d35919a63da93ff7a534.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/82ec590d6fa0e8a53193444d88a6575c09c9a9ff5f30d35919a63da93ff7a534.jpg)

![b606ec4416676f6775a3a795daf7b794473f5197b6548a165b2d104e2ebb2b1e.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/b606ec4416676f6775a3a795daf7b794473f5197b6548a165b2d104e2ebb2b1e.jpg)

![debb87c25f453ba769e1a393a7bbe077d0dfb00bd6e94e3ebf177163004df197.jpg](../icml_results/5818_Self-evolving%20LLM%20agents%20with%20in-distribution%20Optimization/tables/debb87c25f453ba769e1a393a7bbe077d0dfb00bd6e94e3ebf177163004df197.jpg)

<a id="paper-5770"></a>

## Geometric Coherence Learning for Structuring Value Functions in Plain MDPs


### Images

![0c6ba21d41728edda759430297a876e69261c3f43a48fb5a7828e1fad963d77f.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/images/0c6ba21d41728edda759430297a876e69261c3f43a48fb5a7828e1fad963d77f.jpg)

![38952b137255d282589d000a7a6c515e9887dbefe4ec436741e2f0a1ce414b50.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/images/38952b137255d282589d000a7a6c515e9887dbefe4ec436741e2f0a1ce414b50.jpg)

![94d2a39af4181d4ff020fc62e3c5878a8b9e2780f150817e1205c63173ef92aa.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/images/94d2a39af4181d4ff020fc62e3c5878a8b9e2780f150817e1205c63173ef92aa.jpg)

![99c84586695a68ab3a83af3100cb261a939938b659741540c0391f4b73e03f88.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/images/99c84586695a68ab3a83af3100cb261a939938b659741540c0391f4b73e03f88.jpg)

![d9177288987f5bb88d73557b88f735764e234605a1bc25533d5155f2593f775f.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/images/d9177288987f5bb88d73557b88f735764e234605a1bc25533d5155f2593f775f.jpg)

![e364ba2860d366d6ea6d376d6cd4a71d145f5359f0a13a65aa896c086b25ca12.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/images/e364ba2860d366d6ea6d376d6cd4a71d145f5359f0a13a65aa896c086b25ca12.jpg)

### Tables

![0ee64ab31fc696c0a522c5d249072766dd276cf81d5f750315ce78c64d98f414.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/tables/0ee64ab31fc696c0a522c5d249072766dd276cf81d5f750315ce78c64d98f414.jpg)

![50a36df882e5cf8faf0a38121e9ff6a2c3755a71be14096bbaf298d13fe54032.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/tables/50a36df882e5cf8faf0a38121e9ff6a2c3755a71be14096bbaf298d13fe54032.jpg)

![d883275414937b57792dbcd73bbbb27cb39c34fadd3d300b5fc88670b388ff97.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/tables/d883275414937b57792dbcd73bbbb27cb39c34fadd3d300b5fc88670b388ff97.jpg)

![db5b2b1da49702980c67599167fb0c29b4cc27d0919ee3c3dbfb6bbeb5728d54.jpg](../icml_results/5819_Geometric%20Coherence%20Learning%20for%20Structuring%20Value%20Functions%20in%20Plain%20MDPs/tables/db5b2b1da49702980c67599167fb0c29b4cc27d0919ee3c3dbfb6bbeb5728d54.jpg)

<a id="paper-5771"></a>

## AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection


### Images

![04606b7614a5128c2d6506eef3c5ddcf4461dc2885b3e0153fd3b2f853c6fab9.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/04606b7614a5128c2d6506eef3c5ddcf4461dc2885b3e0153fd3b2f853c6fab9.jpg)

![3e37ec8af5a9c7eddf5bc3ace5cc8c131e6252015910c6e1b959bddb8d58ec03.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/3e37ec8af5a9c7eddf5bc3ace5cc8c131e6252015910c6e1b959bddb8d58ec03.jpg)

![5e8f1d0b8b3711df9c555dc5c4466d90c53c837af1e534636f8a647dacc5bff0.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/5e8f1d0b8b3711df9c555dc5c4466d90c53c837af1e534636f8a647dacc5bff0.jpg)

![5f7d06ca57326fba53f34778eeb3f8556228add132c6cc5fffb8a6006a2c90e2.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/5f7d06ca57326fba53f34778eeb3f8556228add132c6cc5fffb8a6006a2c90e2.jpg)

![7a4108e6613d13698a95ea441366f34ea0ea94e87f9b587afdc153fc49438ec6.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/7a4108e6613d13698a95ea441366f34ea0ea94e87f9b587afdc153fc49438ec6.jpg)

![86e0df761c23ea91e0822d41c77d6cdfb4a3e777f76ccca8c5ac72804a6dec87.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/86e0df761c23ea91e0822d41c77d6cdfb4a3e777f76ccca8c5ac72804a6dec87.jpg)

![98d06ce0f5b9f2a3845b79ca2b0445a8a06ab9c4ad2fa9239da4a9b96b70a860.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/98d06ce0f5b9f2a3845b79ca2b0445a8a06ab9c4ad2fa9239da4a9b96b70a860.jpg)

![a443f62931370cdee25555b9aa66f6089f101f082c86429f60f1d2b7e2f07ef2.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/a443f62931370cdee25555b9aa66f6089f101f082c86429f60f1d2b7e2f07ef2.jpg)

![aad37a0b17b7a2b101555fdfdd1041fdf7fcb559126acb3b8b6e01fb70e8c7e4.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/aad37a0b17b7a2b101555fdfdd1041fdf7fcb559126acb3b8b6e01fb70e8c7e4.jpg)

![bbb179802aa8546e44a80cf8e902a1e7fba5ade329f8d9e7db41e74d98d4b0e2.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/bbb179802aa8546e44a80cf8e902a1e7fba5ade329f8d9e7db41e74d98d4b0e2.jpg)

![bc10f8d3b4249998b4b131690dfadec8dbdfa902663e361b7d585701e1232573.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/bc10f8d3b4249998b4b131690dfadec8dbdfa902663e361b7d585701e1232573.jpg)

![c1652bed5f8e4cf0d9d07a3f21078ad011e65ede61b371854719c25b9bb0bd24.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/c1652bed5f8e4cf0d9d07a3f21078ad011e65ede61b371854719c25b9bb0bd24.jpg)

![ce41a92bcb398e0094e2127a8f82eb8984b414536f24c0b157cf835b0732c1f9.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/ce41a92bcb398e0094e2127a8f82eb8984b414536f24c0b157cf835b0732c1f9.jpg)

![d88acbcc729a72e711afa46c7336e85ec8da3d858690e8b9ab7b3b7f8d87b2f6.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/images/d88acbcc729a72e711afa46c7336e85ec8da3d858690e8b9ab7b3b7f8d87b2f6.jpg)

### Tables

![1ef65e34056702b0b43e308d9b0956c421e5317ba9775567e49cd94de160ffed.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/1ef65e34056702b0b43e308d9b0956c421e5317ba9775567e49cd94de160ffed.jpg)

![1fe2f010840e5b8607eef26c89aa3b29ec245dc466ceecfad6b9bcb71215b7ae.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/1fe2f010840e5b8607eef26c89aa3b29ec245dc466ceecfad6b9bcb71215b7ae.jpg)

![588dc6d658d68f55e7d8d1ed4bd727f25e91fcf50313aa956a5e44fb7a8ebf04.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/588dc6d658d68f55e7d8d1ed4bd727f25e91fcf50313aa956a5e44fb7a8ebf04.jpg)

![61c4ed99ac4f150ba9144ae6883688a9e6182885905d11320f51939d4217dd40.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/61c4ed99ac4f150ba9144ae6883688a9e6182885905d11320f51939d4217dd40.jpg)

![626399264ab84d6c8a54f0abcf50751f860b8d266b917f440cbbc2f179922e72.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/626399264ab84d6c8a54f0abcf50751f860b8d266b917f440cbbc2f179922e72.jpg)

![67a6fa421fecad923eab01304529e582d7d31c9b43ae8ad196c96f685409b7e2.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/67a6fa421fecad923eab01304529e582d7d31c9b43ae8ad196c96f685409b7e2.jpg)

![79f2a1d29103c760a80fe31b67c8d428f5f8fc2ce142fc944316ab24ca98be3c.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/79f2a1d29103c760a80fe31b67c8d428f5f8fc2ce142fc944316ab24ca98be3c.jpg)

![d3d12effe0de519f2d9ef87f3b07b95a070793d6b73e0758273c746e27ef6f86.jpg](../icml_results/5820_AnomSeer_%20Reinforcing%20Multimodal%20LLMs%20to%20Reason%20for%20Time-Series%20Anomaly%20Detection/tables/d3d12effe0de519f2d9ef87f3b07b95a070793d6b73e0758273c746e27ef6f86.jpg)

<a id="paper-5772"></a>

## Boosting Monocular Metric Depth Estimation via Bokeh Rendering


### Images

![3f752926d0ad2b731c7b8a1403e417eb6bb3ec886d411403073613904a4b4774.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/3f752926d0ad2b731c7b8a1403e417eb6bb3ec886d411403073613904a4b4774.jpg)

![495d102b0ec22915420bdbc85f8c8f0926a0731c72614915620a9a886463d03a.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/495d102b0ec22915420bdbc85f8c8f0926a0731c72614915620a9a886463d03a.jpg)

![4a614afbd746caa62c5b80c124cb46ca676fbf3f02acc0c2b57ead23be15610c.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/4a614afbd746caa62c5b80c124cb46ca676fbf3f02acc0c2b57ead23be15610c.jpg)

![53dccfc1ebe0fc9240b5e4a41251e7f875949fb41ea65279925e6532a90de5f7.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/53dccfc1ebe0fc9240b5e4a41251e7f875949fb41ea65279925e6532a90de5f7.jpg)

![623f70b2bf76147a276d4d407b860f315a150e729cebca73b474a8bae4f884de.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/623f70b2bf76147a276d4d407b860f315a150e729cebca73b474a8bae4f884de.jpg)

![6edd3e44647cb53ea5fa51543eff7d20d75cb3e25f4cf08ae0b8bc8595a5abe5.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/6edd3e44647cb53ea5fa51543eff7d20d75cb3e25f4cf08ae0b8bc8595a5abe5.jpg)

![749794a84982570af1bb1a92e8810eb698e3c4a78ffe8a65f8df6a524d1843fb.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/749794a84982570af1bb1a92e8810eb698e3c4a78ffe8a65f8df6a524d1843fb.jpg)

![9ce83cad49aa99dd884434a674adaa28ec055009890b1a2ee45cac6fc39ec7c2.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/9ce83cad49aa99dd884434a674adaa28ec055009890b1a2ee45cac6fc39ec7c2.jpg)

![b76ff9cf4a3cf3de73ee7e87e2bd3287191345515671e68f27017c88da1278be.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/b76ff9cf4a3cf3de73ee7e87e2bd3287191345515671e68f27017c88da1278be.jpg)

![d42dc334277b17fefdf336a0e65bef184b97c3129a8ceebad3adfe6b5953beda.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/d42dc334277b17fefdf336a0e65bef184b97c3129a8ceebad3adfe6b5953beda.jpg)

![e2dff1a993fc712528e57e74fe06ef1a065e2e52ba91294b0378de97b584949a.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/e2dff1a993fc712528e57e74fe06ef1a065e2e52ba91294b0378de97b584949a.jpg)

![f4f929631312965099fd99694c2d9edd2418579bbcb429ce642cdcd13d8088a4.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/images/f4f929631312965099fd99694c2d9edd2418579bbcb429ce642cdcd13d8088a4.jpg)

### Tables

![17cb4dfaab1908c9dc6e269e7ce00fad218337e6a893d91ccda6a1582862c776.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/17cb4dfaab1908c9dc6e269e7ce00fad218337e6a893d91ccda6a1582862c776.jpg)

![1943891dc5e5ae2eb41cdae6efb114ba03474b3145d218cb79cfe2a3fc0d66f7.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/1943891dc5e5ae2eb41cdae6efb114ba03474b3145d218cb79cfe2a3fc0d66f7.jpg)

![29b368ee8e8bd7e06012d5b64d4a678853662a50f6181fbc2d24852d0040747b.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/29b368ee8e8bd7e06012d5b64d4a678853662a50f6181fbc2d24852d0040747b.jpg)

![47d6e6d9e58fd94c98dc1ede6be14173f849676ebaaf3b2485e9d5fe3f6762b6.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/47d6e6d9e58fd94c98dc1ede6be14173f849676ebaaf3b2485e9d5fe3f6762b6.jpg)

![8f75ecf99e351765c507044936fd6cfac15fcfc13fde7748b41fd681e0919edc.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/8f75ecf99e351765c507044936fd6cfac15fcfc13fde7748b41fd681e0919edc.jpg)

![90acc40d7767f2274d0b113feee45effbc3de947271abfbc5407ce1cfa376a92.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/90acc40d7767f2274d0b113feee45effbc3de947271abfbc5407ce1cfa376a92.jpg)

![b869b55f5de478e4f7f4fc185013f44eef05356f7fe95bfd1d5d3a6d5bfce2a3.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/b869b55f5de478e4f7f4fc185013f44eef05356f7fe95bfd1d5d3a6d5bfce2a3.jpg)

![dc6de7845f5d7eadd1cdd96c8a747ea79587a66e79b88cce0a4b4e0d9fbaf261.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/dc6de7845f5d7eadd1cdd96c8a747ea79587a66e79b88cce0a4b4e0d9fbaf261.jpg)

![ecba82eacd4301ca5adca6cd780339e1483e715aa6bdeab4566960166d0de167.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/ecba82eacd4301ca5adca6cd780339e1483e715aa6bdeab4566960166d0de167.jpg)

![f1d6cd2aa37b0932515d868ba46c7fd15900abaa72573853990b19c54d29a50f.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/f1d6cd2aa37b0932515d868ba46c7fd15900abaa72573853990b19c54d29a50f.jpg)

![fc600ef0661d7aa84853208eeab19004cf50d82c00f60e15f22e247fa630cae2.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/fc600ef0661d7aa84853208eeab19004cf50d82c00f60e15f22e247fa630cae2.jpg)

![fc99525c7c45ea3788345f1f029b3a3df629745594ec784ce0effa6a5b0b2eb5.jpg](../icml_results/5821_Boosting%20Monocular%20Metric%20Depth%20Estimation%20via%20Bokeh%20Rendering/tables/fc99525c7c45ea3788345f1f029b3a3df629745594ec784ce0effa6a5b0b2eb5.jpg)

<a id="paper-5773"></a>

## Higher-Order Certified Robustness for Regression


### Images

![0abd6aec012ff5f1d335ec395d28b43241c04bb5e74e981c60805ffe1c3d8a63.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/images/0abd6aec012ff5f1d335ec395d28b43241c04bb5e74e981c60805ffe1c3d8a63.jpg)

![2dccfaa4f885ab6b6efcd648d15afd916a12e1adc46843291b9f007e22adad20.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/images/2dccfaa4f885ab6b6efcd648d15afd916a12e1adc46843291b9f007e22adad20.jpg)

![654616c5e4542a921f7f0bbb4f506ead1ad860a2e9683cc4b752194bf0ffa130.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/images/654616c5e4542a921f7f0bbb4f506ead1ad860a2e9683cc4b752194bf0ffa130.jpg)

![b10314f5313cc7be30479b9a7dcf6ae1da953b109770fb940f57e81195176652.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/images/b10314f5313cc7be30479b9a7dcf6ae1da953b109770fb940f57e81195176652.jpg)

![b2484be303c54ba3426a8e8f677d2125520a164ff45c3746023553058cfd780b.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/images/b2484be303c54ba3426a8e8f677d2125520a164ff45c3746023553058cfd780b.jpg)

![cc4fecb5387f04e130a1fccd07bd318d9213b06b2d074d12d5e2d9205aa26c0c.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/images/cc4fecb5387f04e130a1fccd07bd318d9213b06b2d074d12d5e2d9205aa26c0c.jpg)

### Tables

![455820f36dfe9e7dc0d06a85564475a0827f8ebd6e1fc6d6b9bf9ace9e4d517a.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/tables/455820f36dfe9e7dc0d06a85564475a0827f8ebd6e1fc6d6b9bf9ace9e4d517a.jpg)

![79ca6b784dbc1c0f793b2df73ff7c89b981a850483e460128393151a7bed283a.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/tables/79ca6b784dbc1c0f793b2df73ff7c89b981a850483e460128393151a7bed283a.jpg)

![aef04df35e5da5a4888067e6cb973a7c2dbc319f49779ae0121aa49974839f7d.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/tables/aef04df35e5da5a4888067e6cb973a7c2dbc319f49779ae0121aa49974839f7d.jpg)

![c62e217ffe666caffcac514f0a6dfa55d9cdc5f09e4013364e535080a13b70a6.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/tables/c62e217ffe666caffcac514f0a6dfa55d9cdc5f09e4013364e535080a13b70a6.jpg)

![e8157e62ed26c20d4e79f978b8ccf056cd8e0f0fcfed0050baf1d29abef2cf53.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/tables/e8157e62ed26c20d4e79f978b8ccf056cd8e0f0fcfed0050baf1d29abef2cf53.jpg)

![fcf80ea81c66e6a0ad0b045fa83b534dc09ca09d0dc69a78fc54e2b59d9e4fd7.jpg](../icml_results/5822_Higher-Order%20Certified%20Robustness%20for%20Regression/tables/fcf80ea81c66e6a0ad0b045fa83b534dc09ca09d0dc69a78fc54e2b59d9e4fd7.jpg)

<a id="paper-5774"></a>

## From Noise to Control: Parameterized Diffusion Policies


### Images

![0cc79268e552462a0bf6267239519729b53ffd6bea4e469f48be55f94af14e5f.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/0cc79268e552462a0bf6267239519729b53ffd6bea4e469f48be55f94af14e5f.jpg)

![2125ff29cc2d270aba19b8a6734fa2a1ad88d23f8af4352126cdb148ddbfd782.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/2125ff29cc2d270aba19b8a6734fa2a1ad88d23f8af4352126cdb148ddbfd782.jpg)

![47ced3a356a824ee1835994a2d3d0a3aeba3c5709cd1299ed135044f8954b52b.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/47ced3a356a824ee1835994a2d3d0a3aeba3c5709cd1299ed135044f8954b52b.jpg)

![567d33871fa7e15d005cb763ad68b6b967f763dac24646f068b0a359fd65c504.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/567d33871fa7e15d005cb763ad68b6b967f763dac24646f068b0a359fd65c504.jpg)

![7193114b3d6397f7d78ff403b79964a0ad060def19581d7ca7a0d30ab4aba40f.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/7193114b3d6397f7d78ff403b79964a0ad060def19581d7ca7a0d30ab4aba40f.jpg)

![7cb9215f21395bdb19604771c642ce4bfd9581af97088b93b867664e7606b25a.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/7cb9215f21395bdb19604771c642ce4bfd9581af97088b93b867664e7606b25a.jpg)

![8564963d4678670c68e9e793c906358c078ed02e07883eab34fafb5a5675873f.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/8564963d4678670c68e9e793c906358c078ed02e07883eab34fafb5a5675873f.jpg)

![a3646a500e7dfee5170c47017986d8ba33f08edc189a592a048f831f782dd1d3.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/a3646a500e7dfee5170c47017986d8ba33f08edc189a592a048f831f782dd1d3.jpg)

![c52fbf25e0f30eb89967f97e3ff8667ce6717fa695889e3cc2471315a15d0e97.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/c52fbf25e0f30eb89967f97e3ff8667ce6717fa695889e3cc2471315a15d0e97.jpg)

![cb6bbfda333db28829ed38db65bc2f4803fca5370de1450857ea35e01f857d96.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/cb6bbfda333db28829ed38db65bc2f4803fca5370de1450857ea35e01f857d96.jpg)

![cff7e5d3f08698418f4011c5d27f64c973b5a5456995c775b3b07ac8e0ac3920.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/cff7e5d3f08698418f4011c5d27f64c973b5a5456995c775b3b07ac8e0ac3920.jpg)

![d4a32850d8af2b54afabefc18cce5d4fdc3bd6996689bc3bcef185208c3a4a38.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/d4a32850d8af2b54afabefc18cce5d4fdc3bd6996689bc3bcef185208c3a4a38.jpg)

![e1b67207b29552acb45a34077689fe417aa64c4f16c912e9b881b93c57921e63.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/e1b67207b29552acb45a34077689fe417aa64c4f16c912e9b881b93c57921e63.jpg)

![f1cc2e5ab03ec405ab59efc325bc9a0c6385158f2b72612b8852a166e16cf97d.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/f1cc2e5ab03ec405ab59efc325bc9a0c6385158f2b72612b8852a166e16cf97d.jpg)

![fad39f1d9d43230443d09e03cdfc887f35988d5a69a2996637d927aea3a4d986.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/images/fad39f1d9d43230443d09e03cdfc887f35988d5a69a2996637d927aea3a4d986.jpg)

### Tables

![1e7842e1e8d318bd2fcf8fd85590c695c7f65f0b0b04740c41a032d88a5538da.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/1e7842e1e8d318bd2fcf8fd85590c695c7f65f0b0b04740c41a032d88a5538da.jpg)

![3480d9c0e7b941ff01151690084a9ecab826ac982c2f45414e4eabc49c762390.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/3480d9c0e7b941ff01151690084a9ecab826ac982c2f45414e4eabc49c762390.jpg)

![6badc3668261a010f70ada165ce879a583c53dcc46bad043fa86531c9431a5ea.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/6badc3668261a010f70ada165ce879a583c53dcc46bad043fa86531c9431a5ea.jpg)

![7e72f8bd4a0284d012b2eab7abf8b1cb60c0264340bdd2e203de6ddf77e7ea7c.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/7e72f8bd4a0284d012b2eab7abf8b1cb60c0264340bdd2e203de6ddf77e7ea7c.jpg)

![d58b205e370e3646f42f997f5a3d0bb17a1f27a39a5e03672bac5abeda5c4c1f.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/d58b205e370e3646f42f997f5a3d0bb17a1f27a39a5e03672bac5abeda5c4c1f.jpg)

![e0328b1d273bfab504f587fbf9e793bd5d1c8a1037d195e090b746dc61d34634.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/e0328b1d273bfab504f587fbf9e793bd5d1c8a1037d195e090b746dc61d34634.jpg)

![e0f0f74f06c38c0001380723d1123e3bef2386bded42461df00a9af77228ade0.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/e0f0f74f06c38c0001380723d1123e3bef2386bded42461df00a9af77228ade0.jpg)

![f8625673c6c2be196e4013db54014bf94e419673d4d0beb9b0bbc4cc7421b674.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/f8625673c6c2be196e4013db54014bf94e419673d4d0beb9b0bbc4cc7421b674.jpg)

![fad68d18e61299f02840d5ecb532fe6506150af3833ad12921d121cc66e2e346.jpg](../icml_results/5823_From%20Noise%20to%20Control_%20Parameterized%20Diffusion%20Policies/tables/fad68d18e61299f02840d5ecb532fe6506150af3833ad12921d121cc66e2e346.jpg)

<a id="paper-5775"></a>

## Generalization Bounds for Discrete Diffusion: Statistical Advantage of Masking

### Tables

![8e6cf7247efc042d6af7d2f168487448aafa707ec35df7baaff99bac92a4f073.jpg](../icml_results/5824_Generalization%20Bounds%20for%20Discrete%20Diffusion_%20Statistical%20Advantage%20of%20Masking/tables/8e6cf7247efc042d6af7d2f168487448aafa707ec35df7baaff99bac92a4f073.jpg)

![f985d156602967d4904421e6bdd024e01ccd686673783c9486f09f142908fd92.jpg](../icml_results/5824_Generalization%20Bounds%20for%20Discrete%20Diffusion_%20Statistical%20Advantage%20of%20Masking/tables/f985d156602967d4904421e6bdd024e01ccd686673783c9486f09f142908fd92.jpg)
