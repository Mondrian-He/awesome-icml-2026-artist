# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 120 of 197

## 目录 (Table of Contents)

1. [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](#paper-3928)
2. [TFTF: Training-Free Targeted Flow for Conditional Sampling](#paper-3929)
3. [What Does Thompson Sampling Optimize?](#paper-3930)
4. [Can Muon Fine-tune Adam-Pretrained Models?](#paper-3931)
5. [Small Generalizable Prompt Predictive Models Can Steer Efficient RL Post-Training of Large Reasoning Models](#paper-3932)
6. [A recipe for scalable attention-based ML potentials: unlocking long-range accuracy with all-to-all node attention](#paper-3933)
7. [Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](#paper-3934)
8. [T-POP: Test-Time Personalization with Online Preference Feedback](#paper-3935)
9. [Interpretability and Generalization Bounds for Learning Spatial Physics](#paper-3936)
10. [Towards a Science of AI Agent Reliability](#paper-3937)
11. [Language Generation with Replay: A Learning-Theoretic View of Model Collapse](#paper-3938)
12. [G$^2$RPO: Geometric GRPO Escaping LLMs’ Reasoning Ruts to Break the Accuracy–Entropy Trade-off](#paper-3939)
13. [HyPOLE: Hyperproperty-Guided Multi-Agent Reinforcement Learning under Partial Observation](#paper-3940)
14. [Causal Identification from Counterfactual Data: Completeness and Bounding Results](#paper-3941)
15. [Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](#paper-3942)
16. [ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition](#paper-3943)
17. [VeriSimpl: Robust Optimization Modeling from Natural Language using Simplification-based Verification](#paper-3944)
18. [STT-LLM: Structural-Temporal Tokenization for Adapting LLMs to Longitudinal Clinical Profiles](#paper-3945)
19. [Accuracy-First Rényi Differential Privacy and Post-Processing Immunity](#paper-3946)
20. [Revisiting OOD Generalization in Programmatic RL](#paper-3947)
21. [Personalized Image Generation via Human-in-the-loop Bayesian Optimization](#paper-3948)
22. [Panini: Continual Learning in Token Space via Structured Memory](#paper-3949)
23. [MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](#paper-3950)
24. [A model of errors in transformers](#paper-3951)
25. [One-step Optimal Transport via Regularized Distribution Matching Distillation](#paper-3952)
26. [$α$-PFN: Fast Entropy Search via In-Context Learning](#paper-3953)
27. [Particle Flow for Learning from Label Proportions](#paper-3954)
28. [Trees to Flows and Back: Unifying Decision Trees and Diffusion Models](#paper-3955)
29. [Reasoning Models Are Test Exploiters: Rethinking Multiple Choice](#paper-3956)
30. [DISSOLVR: An Interpretable and Fast Framework for Aqueous and Organic Solubility Prediction](#paper-3957)
31. [Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents](#paper-3958)
32. [Multi-Task GRPO: Reliable LLM Reasoning Across Tasks](#paper-3959)
33. [TriForces: Augmenting Atomistic GNNs for Transferable Representations](#paper-3960)

---


<a id="paper-3928"></a>

## TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model

### Images

![0d0d8246b2fd43bd69bf191cd5736aa5dd823c2878ac18233185c32ec0d5ad27.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/0d0d8246b2fd43bd69bf191cd5736aa5dd823c2878ac18233185c32ec0d5ad27.jpg)

![221b1c3e4d13789832462152910aeed2450d49d92592bab1bc127660ead8dc01.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/221b1c3e4d13789832462152910aeed2450d49d92592bab1bc127660ead8dc01.jpg)

![315c65b0f60b0a92dcc17a845bec7989b6aca452421e492fd9d50928fb7e1bb7.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/315c65b0f60b0a92dcc17a845bec7989b6aca452421e492fd9d50928fb7e1bb7.jpg)

![552be167666cd7fea47383e6194797f2c435f1dddbbc0c8c1a0de45c8253e21e.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/552be167666cd7fea47383e6194797f2c435f1dddbbc0c8c1a0de45c8253e21e.jpg)

![66900e8185d90c595e5fc1bd7fc42f5a8a987ea6d10eb29eb379686dff96f9c2.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/66900e8185d90c595e5fc1bd7fc42f5a8a987ea6d10eb29eb379686dff96f9c2.jpg)

![97df41e8685d19b09e573b9ba2d84c71d2aa08bbddc0d1cc0686e8783ac00b5a.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/97df41e8685d19b09e573b9ba2d84c71d2aa08bbddc0d1cc0686e8783ac00b5a.jpg)

![9d22702bec828f442a892f37eea100c789552c3de21396c68d152e384041fe16.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/9d22702bec828f442a892f37eea100c789552c3de21396c68d152e384041fe16.jpg)

![a6585d219fcc4ea94ec0d48b862686b1251c26f9980adf8699f98f7165d9b10f.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/a6585d219fcc4ea94ec0d48b862686b1251c26f9980adf8699f98f7165d9b10f.jpg)

![c2108e9c14a66f416b3884f8dcc0cecbb5414c19faa9caff03bafb4895a71520.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/c2108e9c14a66f416b3884f8dcc0cecbb5414c19faa9caff03bafb4895a71520.jpg)

![d2991980a5825b61fe62257d0b28edde360c5461b7d49a3e136b438a9f50cd1a.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/d2991980a5825b61fe62257d0b28edde360c5461b7d49a3e136b438a9f50cd1a.jpg)

![d676bed753433db082d4514a95c61d19a95e750c4edb5154b07570d7bffe5756.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/images/d676bed753433db082d4514a95c61d19a95e750c4edb5154b07570d7bffe5756.jpg)

### Tables

![199c616c14f0c82be639b6af642827ffa4c76d3bc0337cfa361e034e8aad1aaa.jpg](../icml_results/3968_cuRegOT_%20A%20GPU-Accelerated%20Solver%20for%20Entropic-Regularized%20Optimal%20Transport/tables/199c616c14f0c82be639b6af642827ffa4c76d3bc0337cfa361e034e8aad1aaa.jpg)

## TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model


### Images

![00a07d6c7616b634dc16fa7267632997af16c23686b4d8fc2fa2bfe82b052f8c.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/00a07d6c7616b634dc16fa7267632997af16c23686b4d8fc2fa2bfe82b052f8c.jpg)

![03c0e6f644a8f5f4fcb4da5cb75282d28a5ccaf93e0ae66a906dbdc74a038967.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/03c0e6f644a8f5f4fcb4da5cb75282d28a5ccaf93e0ae66a906dbdc74a038967.jpg)

![04c683165d1656291a3bfd84771aa181198e3d6bc0f14cb02740d28caa306cf7.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/04c683165d1656291a3bfd84771aa181198e3d6bc0f14cb02740d28caa306cf7.jpg)

![056b775c538ebf4c5c1aacfe95c62d0889cef93961ac4ecf4dd17a0b3a035817.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/056b775c538ebf4c5c1aacfe95c62d0889cef93961ac4ecf4dd17a0b3a035817.jpg)

![0ceea1a40e765425b280171eb84e5cc1b3be9ba819f952a75e705b6344f79185.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/0ceea1a40e765425b280171eb84e5cc1b3be9ba819f952a75e705b6344f79185.jpg)

![0f677db0559da453b10d2684d75b013ceb559b12c1bbf00af69ccf8cfc429797.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/0f677db0559da453b10d2684d75b013ceb559b12c1bbf00af69ccf8cfc429797.jpg)

![117e9d1058e7d59ca70cd691557a00412ff08a61677d8b8abeb5e0839249f0d8.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/117e9d1058e7d59ca70cd691557a00412ff08a61677d8b8abeb5e0839249f0d8.jpg)

![128dda23c48a86572163de365083412acceec5ccdf380e7d15a818de24d9ffb6.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/128dda23c48a86572163de365083412acceec5ccdf380e7d15a818de24d9ffb6.jpg)

![12fd5663eb9f3a0a3ebddeeb135d1853ae7f28539a61bdb746262efaea6774ab.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/12fd5663eb9f3a0a3ebddeeb135d1853ae7f28539a61bdb746262efaea6774ab.jpg)

![16e48b22a817301a1264e833f6ce66b26ca1ac91638593e2c02095f5fe2f4f46.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/16e48b22a817301a1264e833f6ce66b26ca1ac91638593e2c02095f5fe2f4f46.jpg)

![17350ea4ea86aa2d24adec18d3bcf1169431f94d36e329521383217a0d23a5b6.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/17350ea4ea86aa2d24adec18d3bcf1169431f94d36e329521383217a0d23a5b6.jpg)

![1e9528cfb90fc4623cfb1736893e8c0d956434e8ef33f694a37754fd502e0f76.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/1e9528cfb90fc4623cfb1736893e8c0d956434e8ef33f694a37754fd502e0f76.jpg)

![1f8038d97a1e5c6a87b7bc835460b104328661238c7e7f3a4f7caa60680daf6d.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/1f8038d97a1e5c6a87b7bc835460b104328661238c7e7f3a4f7caa60680daf6d.jpg)

![1f94c3498328d58289c1ca4ff0cfc1415ff94238168712128ac4703050fa2e24.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/1f94c3498328d58289c1ca4ff0cfc1415ff94238168712128ac4703050fa2e24.jpg)

![23c515e203525a3aca2a8e88f06b3ba4a390bbf723c3fd998ae2ebb04b29e284.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/23c515e203525a3aca2a8e88f06b3ba4a390bbf723c3fd998ae2ebb04b29e284.jpg)

![298416d4904680be61cbd7c5a78a5b211ee827777180f827705b054d9c0c52e7.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/298416d4904680be61cbd7c5a78a5b211ee827777180f827705b054d9c0c52e7.jpg)

![2c2ed96562bf66262b0ede6f6162839f87708da2d4b01f6d66f5adf9b32675b0.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/2c2ed96562bf66262b0ede6f6162839f87708da2d4b01f6d66f5adf9b32675b0.jpg)

![2cceb46691a052c6bad0d3604955edd43b7a3f2a58769252b4d8898b2b0c069a.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/2cceb46691a052c6bad0d3604955edd43b7a3f2a58769252b4d8898b2b0c069a.jpg)

![390bf606f0f21e4946c02284eb3a1befa56ba39e11bc3e40b0917105be4ea7df.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/390bf606f0f21e4946c02284eb3a1befa56ba39e11bc3e40b0917105be4ea7df.jpg)

![3a6b8f24ec9c79e00fb19797fb3deeba2dff2e037fbe2c90c1ac1888983aabff.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/3a6b8f24ec9c79e00fb19797fb3deeba2dff2e037fbe2c90c1ac1888983aabff.jpg)

![3b4af9687ec46ca069ce35413102cb7ced23da9133cf52d05474ecc27749c2d9.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/3b4af9687ec46ca069ce35413102cb7ced23da9133cf52d05474ecc27749c2d9.jpg)

![3fcc213f0d1cb514afbfb010f235114ea8675813f1eb1bde5d3217f0452ca444.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/3fcc213f0d1cb514afbfb010f235114ea8675813f1eb1bde5d3217f0452ca444.jpg)

![4288194258a31182a49b18c5bd8df66a11f4460c3ef023a09c3c5e2bca3d090a.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/4288194258a31182a49b18c5bd8df66a11f4460c3ef023a09c3c5e2bca3d090a.jpg)

![4bda550708941b5bd36a2c6061856293dba0caaace2a7a88995024b4be6250a7.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/4bda550708941b5bd36a2c6061856293dba0caaace2a7a88995024b4be6250a7.jpg)

![4c9a5a40232deca3426b81cafba1a2e6ab47adbae749bf682858e514b0d66221.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/4c9a5a40232deca3426b81cafba1a2e6ab47adbae749bf682858e514b0d66221.jpg)

![519c2f09f1070af4740c7ee2781d213dd720c88004dfa21fdd947ecc5a135451.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/519c2f09f1070af4740c7ee2781d213dd720c88004dfa21fdd947ecc5a135451.jpg)

![598cf0a1846cb91868373ea8108ee4ca7b46f42f41b3174d9173db08a6781ad7.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/598cf0a1846cb91868373ea8108ee4ca7b46f42f41b3174d9173db08a6781ad7.jpg)

![5c9a88f5ab099ef0a5f1bd5053eced6bcca76664efe4c96db9116c40bb0197d1.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/5c9a88f5ab099ef0a5f1bd5053eced6bcca76664efe4c96db9116c40bb0197d1.jpg)

![5e9b144d2ceba3907b8b841ff045a0188320bf90fb8931cd3cb1c1a7ca4f9fa5.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/5e9b144d2ceba3907b8b841ff045a0188320bf90fb8931cd3cb1c1a7ca4f9fa5.jpg)

![5f756ab9dc025de17b7ade3c97142dbc323249b13730028846e9bc87773bc6d8.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/5f756ab9dc025de17b7ade3c97142dbc323249b13730028846e9bc87773bc6d8.jpg)

![633bbc63787fa92cf8b4f44b6356e2de8540d2ec85818d04f9593f649d871d4f.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/633bbc63787fa92cf8b4f44b6356e2de8540d2ec85818d04f9593f649d871d4f.jpg)

![63c523e3f266701c7100bd0be591124a8ab16385b170f46b3142fd17017944a4.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/63c523e3f266701c7100bd0be591124a8ab16385b170f46b3142fd17017944a4.jpg)

![7198eb059b7df208426dfc3e02403f30c635441f5960d2ebbecdf7b4ef1c93cc.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/7198eb059b7df208426dfc3e02403f30c635441f5960d2ebbecdf7b4ef1c93cc.jpg)

![719abb9637a0011e2f5d8f7e2c8e6988ce9e92d697ad19491a1b6dea063842b9.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/719abb9637a0011e2f5d8f7e2c8e6988ce9e92d697ad19491a1b6dea063842b9.jpg)

![7633bbace5a21f6a8aad29b5a47185eb8e31c6d2180c492997104d66899f2fb0.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/7633bbace5a21f6a8aad29b5a47185eb8e31c6d2180c492997104d66899f2fb0.jpg)

![813608ac9c835dfbd6043d02993dc03847a1190953091cdda112681e4413f521.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/813608ac9c835dfbd6043d02993dc03847a1190953091cdda112681e4413f521.jpg)

![8d2e14445357ecdd55ae220d47043089f9652d735c757888367f38a39ce744fc.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/8d2e14445357ecdd55ae220d47043089f9652d735c757888367f38a39ce744fc.jpg)

![8e2ac30a1b35f9d9270dbcea32656920baa3c68f5f6821567e4460291123d91c.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/8e2ac30a1b35f9d9270dbcea32656920baa3c68f5f6821567e4460291123d91c.jpg)

![9472c144b98c9648bb7cc3eb5bda5eb1d00300295bcbc9445d36866540d20171.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9472c144b98c9648bb7cc3eb5bda5eb1d00300295bcbc9445d36866540d20171.jpg)

![9863e5325ea0c38b304308d475a20d624fa23a4e3c3b5cb80b3386dd547ee9f2.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9863e5325ea0c38b304308d475a20d624fa23a4e3c3b5cb80b3386dd547ee9f2.jpg)

![994eacb2cbabf549396f07cb5b66ab5b732b74a4ed6e272ca571dbe90a07b09d.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/994eacb2cbabf549396f07cb5b66ab5b732b74a4ed6e272ca571dbe90a07b09d.jpg)

![9a7f07eada7f97a6db8e91b8980b7e9116b4696f1b752c2a8bcf4b013eda82fa.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9a7f07eada7f97a6db8e91b8980b7e9116b4696f1b752c2a8bcf4b013eda82fa.jpg)

![9a81afce72e7b6e45c07d58a4abf21f2814d5d09a1ef65031ab0a657d20aea15.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9a81afce72e7b6e45c07d58a4abf21f2814d5d09a1ef65031ab0a657d20aea15.jpg)

![9dc622785db56cc2ae65403e42c321bf4093b14221792efd003e75cb884956e1.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9dc622785db56cc2ae65403e42c321bf4093b14221792efd003e75cb884956e1.jpg)

![9e42c50af13e7c1860a4bcb007987b83397ca94187dc38200a5818856fb47947.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9e42c50af13e7c1860a4bcb007987b83397ca94187dc38200a5818856fb47947.jpg)

![9e5594bdb019d4dcb074b77593b762fb307843dd9ce488ecb07a93331fd92ab8.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9e5594bdb019d4dcb074b77593b762fb307843dd9ce488ecb07a93331fd92ab8.jpg)

![9ee92624066353044f9df4d1a94d276dc19727aa35b413f89ae9ba3887f72b59.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/9ee92624066353044f9df4d1a94d276dc19727aa35b413f89ae9ba3887f72b59.jpg)

![a24dbcb406da92e751ff222587916943c1fa3afe49470763e2ba81501b814cf5.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/a24dbcb406da92e751ff222587916943c1fa3afe49470763e2ba81501b814cf5.jpg)

![b5f93344b3d25c581cdaea23112831580bfb976635cbd8199a351eb434a6e932.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/b5f93344b3d25c581cdaea23112831580bfb976635cbd8199a351eb434a6e932.jpg)

![b6acd29b4f34b647dd7126654869150190ec6bfd7323380a692b99d21b023615.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/b6acd29b4f34b647dd7126654869150190ec6bfd7323380a692b99d21b023615.jpg)

![b93b03175f9e53ac26a5a2ce4e757a7a02516061aef48f5658c8b40665bb6da4.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/b93b03175f9e53ac26a5a2ce4e757a7a02516061aef48f5658c8b40665bb6da4.jpg)

![b9c0cbcf0eafe6917d984de4bc6b91285ee81188fdc0d0d633d429644b5b02ff.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/b9c0cbcf0eafe6917d984de4bc6b91285ee81188fdc0d0d633d429644b5b02ff.jpg)

![bb7112a041d0eca3fe589686047ccbc4198cf2290500cad29e9ffe1293df17b1.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/bb7112a041d0eca3fe589686047ccbc4198cf2290500cad29e9ffe1293df17b1.jpg)

![bf981e36de5b84ccfe892f48eee115ed8d71fb6ae200f0df706dc0cfdb77ad56.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/bf981e36de5b84ccfe892f48eee115ed8d71fb6ae200f0df706dc0cfdb77ad56.jpg)

![c91bb7aabb0df4955ca7160a2d21b6004c5dd8f6700ecc2f9ad2869174bcc19c.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/c91bb7aabb0df4955ca7160a2d21b6004c5dd8f6700ecc2f9ad2869174bcc19c.jpg)

![cb30bc9928f98dcc14c0249a4f0e55c7568a88c64b80bd156c9f10225483a8bf.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/cb30bc9928f98dcc14c0249a4f0e55c7568a88c64b80bd156c9f10225483a8bf.jpg)

![cf2d62b4769f3db9f6f5db663f353dd122c773fe641cacefb3ed41bb4c8628c4.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/cf2d62b4769f3db9f6f5db663f353dd122c773fe641cacefb3ed41bb4c8628c4.jpg)

![d070f52ef4cf885ccc622219a9e296fe186c2ab21f10f34a4930247af3199ed3.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/d070f52ef4cf885ccc622219a9e296fe186c2ab21f10f34a4930247af3199ed3.jpg)

![d5e6f1b97478450aa039032a28e28fc0d4cabd923690a35f90219f4718c84d6d.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/d5e6f1b97478450aa039032a28e28fc0d4cabd923690a35f90219f4718c84d6d.jpg)

![d8634402726c162041bdaa99a4f114baaa8f703dce37dec511c80ef55deba591.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/d8634402726c162041bdaa99a4f114baaa8f703dce37dec511c80ef55deba591.jpg)

![dba216dcad5534b5177f293451a2ba6bc7269794cfab87b58dc20a72e4776cf7.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/dba216dcad5534b5177f293451a2ba6bc7269794cfab87b58dc20a72e4776cf7.jpg)

![dd7d8deb36072812dec21ea7c0bb14b0f637e77d232ad7b4b2b5694d4d9081f1.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/dd7d8deb36072812dec21ea7c0bb14b0f637e77d232ad7b4b2b5694d4d9081f1.jpg)

![ddfb092c37a8d1797d5b0a6e64cd3e5afc5bc9357c4d8ccf6404ce07cd401fd2.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/ddfb092c37a8d1797d5b0a6e64cd3e5afc5bc9357c4d8ccf6404ce07cd401fd2.jpg)

![e1514989c1e7edaa899d94e68bd12296a2d45a3ffe66781d9e54f773526bf409.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/e1514989c1e7edaa899d94e68bd12296a2d45a3ffe66781d9e54f773526bf409.jpg)

![e6ce0cdfbb6acfd71eee3a434032749bd021187c6bbff596fc8c149bc4caac2f.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/e6ce0cdfbb6acfd71eee3a434032749bd021187c6bbff596fc8c149bc4caac2f.jpg)

![ebec38b70f26cbc3403b531ed1241bcbb3b7d713c6b67eb11041f8c2e9344f0b.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/ebec38b70f26cbc3403b531ed1241bcbb3b7d713c6b67eb11041f8c2e9344f0b.jpg)

![ef5776ccec72e680690f5e5bce644df18acffef867a48a6692201e790e2d0211.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/ef5776ccec72e680690f5e5bce644df18acffef867a48a6692201e790e2d0211.jpg)

![efa9d36ed44602ed8d9a681e2768c598362b7417dc2653ec2a09a0b71011fef6.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/efa9d36ed44602ed8d9a681e2768c598362b7417dc2653ec2a09a0b71011fef6.jpg)

![f4bc625f8304f4f423c765b263beb33ecc73f52f265cf0d36241ab2885d37573.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/f4bc625f8304f4f423c765b263beb33ecc73f52f265cf0d36241ab2885d37573.jpg)

![f8eea7e0a4cff6ad507dc4f580ddaaf87f12c60ffbf4ef0a7ed31b6a925434e9.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/images/f8eea7e0a4cff6ad507dc4f580ddaaf87f12c60ffbf4ef0a7ed31b6a925434e9.jpg)

### Tables

![1619115006d4ac4a9f780389a82c1733fb195c1ac087ec14e8e35000b23688b6.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/1619115006d4ac4a9f780389a82c1733fb195c1ac087ec14e8e35000b23688b6.jpg)

![66dceb46c9d0b694dd35e32b9d74aced8a85eaafbde2c0f6e9d7610367dd2a2c.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/66dceb46c9d0b694dd35e32b9d74aced8a85eaafbde2c0f6e9d7610367dd2a2c.jpg)

![83f6b9179b6ef8e72cd3c5ccf7509fcfe975c369497221d9b44766aaee281608.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/83f6b9179b6ef8e72cd3c5ccf7509fcfe975c369497221d9b44766aaee281608.jpg)

![b663f8a7b7307acfee8dc9be396bd837641e00473bca7d9d3eb70461676d4d89.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/b663f8a7b7307acfee8dc9be396bd837641e00473bca7d9d3eb70461676d4d89.jpg)

![c7b698af1b74b7261f22ecb4a8e3ec1974ccc6a8a4dbd965e6437b980922bf81.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/c7b698af1b74b7261f22ecb4a8e3ec1974ccc6a8a4dbd965e6437b980922bf81.jpg)

![ca251ff753df5ee292549b92eef2ca627cef8bbdc0e4716a7409bbbcd099dd56.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/ca251ff753df5ee292549b92eef2ca627cef8bbdc0e4716a7409bbbcd099dd56.jpg)

![ed8b39e9274b08582a4998e345daf5e3127a76a257cf55e10310d6f621309c81.jpg](../icml_results/3969_TabICLv2_%20A%20Better%2C%20Faster%2C%20Scalable%2C%20and%20Open%20Tabular%20Foundation%20Model/tables/ed8b39e9274b08582a4998e345daf5e3127a76a257cf55e10310d6f621309c81.jpg)

<a id="paper-3929"></a>

## TFTF: Training-Free Targeted Flow for Conditional Sampling


### Images

![38fe6d08eaad69f59f56ae9ca179eb3b58bcdfafed55f10a6a00be7ac6d802de.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/38fe6d08eaad69f59f56ae9ca179eb3b58bcdfafed55f10a6a00be7ac6d802de.jpg)

![3ca7a05525860d82af4789d88abbb196e08103991ce048b7fbd31f3f83e32421.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/3ca7a05525860d82af4789d88abbb196e08103991ce048b7fbd31f3f83e32421.jpg)

![4c1f5422bcfb50317a552b3541965b4027a1a5ee32f35c0c3f9d874866212250.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/4c1f5422bcfb50317a552b3541965b4027a1a5ee32f35c0c3f9d874866212250.jpg)

![61062ba54158225ee7b7bba4c1fbec98595a6a63e951e908db13d391bdbda6bc.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/61062ba54158225ee7b7bba4c1fbec98595a6a63e951e908db13d391bdbda6bc.jpg)

![80fc2c817fd5d98425ae1a6fae0d01b0de4912f19aa35d3feb55dd8f287524ad.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/80fc2c817fd5d98425ae1a6fae0d01b0de4912f19aa35d3feb55dd8f287524ad.jpg)

![85ac43ed0145707eccdc96c8f9efe966345d9d4a43d1e01c404d4a8c201b4a85.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/85ac43ed0145707eccdc96c8f9efe966345d9d4a43d1e01c404d4a8c201b4a85.jpg)

![87a11d0fcb30c4aea95b4ce2e9f6a24e8aeb2d66ce844df9ccd60e4fb07ab8a4.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/87a11d0fcb30c4aea95b4ce2e9f6a24e8aeb2d66ce844df9ccd60e4fb07ab8a4.jpg)

![8c662aa5eca156c547dd46a5f7c438ab0fb02e4e4134f43b756be7c9fa88add6.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/8c662aa5eca156c547dd46a5f7c438ab0fb02e4e4134f43b756be7c9fa88add6.jpg)

![a1e421aab272ff096d21cd6700d6147693fee69f7b867d0ad9719349827e16bf.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/a1e421aab272ff096d21cd6700d6147693fee69f7b867d0ad9719349827e16bf.jpg)

![c4ab7c79be4e3f8ce2dc0346c442ad789838746e7a7884400d9a5ee8c3d2247a.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/c4ab7c79be4e3f8ce2dc0346c442ad789838746e7a7884400d9a5ee8c3d2247a.jpg)

![cb80032bae95cf979f89910520cfc07baf4bd7ada8283795b84c91d83f6f0395.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/cb80032bae95cf979f89910520cfc07baf4bd7ada8283795b84c91d83f6f0395.jpg)

![d8925a3dd6fc00b8077093eee4be5879285d7481e1647640ebbfdc288c9cf2eb.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/d8925a3dd6fc00b8077093eee4be5879285d7481e1647640ebbfdc288c9cf2eb.jpg)

![db31ff3f0fcfcb35d1731ff76ce04c7c27b7653b14d65ccccc24755e835d17ba.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/db31ff3f0fcfcb35d1731ff76ce04c7c27b7653b14d65ccccc24755e835d17ba.jpg)

![f76888820054dfb6122ae7ea7a7331c2e886cd1cb415da02aa8073454a4796ba.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/f76888820054dfb6122ae7ea7a7331c2e886cd1cb415da02aa8073454a4796ba.jpg)

![fe26f8abd77141978825be6ef1361b2cc9de90a1428c2b7707589e045c455b64.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/fe26f8abd77141978825be6ef1361b2cc9de90a1428c2b7707589e045c455b64.jpg)

![ffa8de590e6c5a02162ffae911e2c606cb1b7be824f677b8d2ad628218d066d4.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/images/ffa8de590e6c5a02162ffae911e2c606cb1b7be824f677b8d2ad628218d066d4.jpg)

### Tables

![0928d2cd1e2b2a82b735d32339619a1b9fa7eaa869c06333801c868745ca4eb0.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/tables/0928d2cd1e2b2a82b735d32339619a1b9fa7eaa869c06333801c868745ca4eb0.jpg)

![4a81833e3e2c6130624913a4b57eefa2290ff6a4b822aba37fb1d4a15a0f2720.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/tables/4a81833e3e2c6130624913a4b57eefa2290ff6a4b822aba37fb1d4a15a0f2720.jpg)

![a6eb05a0718dfcacd3b7cfce59202cb9f1dc0f1c3c8e00c1decf0e45a7822b20.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/tables/a6eb05a0718dfcacd3b7cfce59202cb9f1dc0f1c3c8e00c1decf0e45a7822b20.jpg)

![c31d920271dd684b96864ded2463799f93dfb4b086ef4739afbfa35791ad56f7.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/tables/c31d920271dd684b96864ded2463799f93dfb4b086ef4739afbfa35791ad56f7.jpg)

![d1dcf6a097232d1a8817045d1a4d4ccca7b09b23624774eeb15fca1d39e6e2bb.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/tables/d1dcf6a097232d1a8817045d1a4d4ccca7b09b23624774eeb15fca1d39e6e2bb.jpg)

![e5d215942ee33ab7065f1c891eb51a74e7bf4c600d880ead18bef275b04a4b88.jpg](../icml_results/3970_TFTF_%20Training-Free%20Targeted%20Flow%20for%20Conditional%20Sampling/tables/e5d215942ee33ab7065f1c891eb51a74e7bf4c600d880ead18bef275b04a4b88.jpg)

<a id="paper-3930"></a>

## What Does Thompson Sampling Optimize?


### Images

![3bde0640ddfb8eb036471c71f5b409b11315cdc975b47c78a4b5719ee5420b5b.jpg](../icml_results/3971_What%20Does%20Thompson%20Sampling%20Optimize_/images/3bde0640ddfb8eb036471c71f5b409b11315cdc975b47c78a4b5719ee5420b5b.jpg)

![71abb9b8aca22fd3f23404d323a83d85e3efb07ad8dd2f6aca2aa7abb8e6c300.jpg](../icml_results/3971_What%20Does%20Thompson%20Sampling%20Optimize_/images/71abb9b8aca22fd3f23404d323a83d85e3efb07ad8dd2f6aca2aa7abb8e6c300.jpg)

![c7762d333bfe56017dbe78c860479c5ce202a634fb7d5f89614850fbef48fe5e.jpg](../icml_results/3971_What%20Does%20Thompson%20Sampling%20Optimize_/images/c7762d333bfe56017dbe78c860479c5ce202a634fb7d5f89614850fbef48fe5e.jpg)

![c7f6a7783c43ae6198d62617352087930aa4ea46e16ab6b2a28af70c263fb538.jpg](../icml_results/3971_What%20Does%20Thompson%20Sampling%20Optimize_/images/c7f6a7783c43ae6198d62617352087930aa4ea46e16ab6b2a28af70c263fb538.jpg)

<a id="paper-3931"></a>

## Can Muon Fine-tune Adam-Pretrained Models?


### Images

![06ecc7b965294371bc8625820141d74832af661acb3ce4e133cbd5564e35fa2d.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/06ecc7b965294371bc8625820141d74832af661acb3ce4e133cbd5564e35fa2d.jpg)

![14ca73f81f8b5eb480a79f91b33dd057d0ff6fd137f6feb5c711cca4e0a57cb4.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/14ca73f81f8b5eb480a79f91b33dd057d0ff6fd137f6feb5c711cca4e0a57cb4.jpg)

![18893198509890d23252426e3d59a396a96164e8a1b77c152322c6859cce6ab5.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/18893198509890d23252426e3d59a396a96164e8a1b77c152322c6859cce6ab5.jpg)

![251f097c02c7a6b712f1498be56287d77eb2855653f4e63fead71c7aece94772.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/251f097c02c7a6b712f1498be56287d77eb2855653f4e63fead71c7aece94772.jpg)

![260a3d2021df4656b91aa45983850989dd1df450619bf9b54eaab074107a7060.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/260a3d2021df4656b91aa45983850989dd1df450619bf9b54eaab074107a7060.jpg)

![59f140272018036c8de586139d1bc8dd7705685d082b693d77a93b9986be0a9d.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/59f140272018036c8de586139d1bc8dd7705685d082b693d77a93b9986be0a9d.jpg)

![704764b77e9ce01a29bca13a9ab7f273bf7af40b03d95cd38ca98cfa84debcbe.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/704764b77e9ce01a29bca13a9ab7f273bf7af40b03d95cd38ca98cfa84debcbe.jpg)

![783fddc598a8be631b1cb5ce12ae4579fd978ca0b3b2087c31efed5267325dc6.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/783fddc598a8be631b1cb5ce12ae4579fd978ca0b3b2087c31efed5267325dc6.jpg)

![7ecf585b0dd6a9b28021999c11f8fa362f2ff04fc058b53e3e622878e6c9816d.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/7ecf585b0dd6a9b28021999c11f8fa362f2ff04fc058b53e3e622878e6c9816d.jpg)

![8536fad5ceba177814346ee0cbf9cab5acdc117c81260191d0aa34034b63dbeb.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/8536fad5ceba177814346ee0cbf9cab5acdc117c81260191d0aa34034b63dbeb.jpg)

![a5e39921f0386e85b3ab6b4982cbed1d77167a17b93b672b9cc1ad91e464d5d6.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/a5e39921f0386e85b3ab6b4982cbed1d77167a17b93b672b9cc1ad91e464d5d6.jpg)

![ae543e85aef2ae9e08ca905742f02f09d6f74fa6ea27345c89eebaafbae03233.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/ae543e85aef2ae9e08ca905742f02f09d6f74fa6ea27345c89eebaafbae03233.jpg)

![bbee4c5134bbbd05ee56f9912ab28331285192ac027cd3a955b723570bcf46cd.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/bbee4c5134bbbd05ee56f9912ab28331285192ac027cd3a955b723570bcf46cd.jpg)

![dc17b395b9f8cd86ce6c7045003919739be23999b78d4eb3b293c92b4c243601.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/images/dc17b395b9f8cd86ce6c7045003919739be23999b78d4eb3b293c92b4c243601.jpg)

### Tables

![01788e0ed163d4652ae36ae018a27d029c8408663f82189ecb6207c359989cf7.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/01788e0ed163d4652ae36ae018a27d029c8408663f82189ecb6207c359989cf7.jpg)

![10308198e4869efaaf08b679daa1d6170effb609dc726ad01bc72b385961a094.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/10308198e4869efaaf08b679daa1d6170effb609dc726ad01bc72b385961a094.jpg)

![295ff7b2e34af1d6e3a42c10db24745672c42a5fb6fa5a0d822693a0105282b0.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/295ff7b2e34af1d6e3a42c10db24745672c42a5fb6fa5a0d822693a0105282b0.jpg)

![3b887aa29d0ddad8b5aa88ea96064bc41493b9e9315b9aa456cd19a7a4d52010.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/3b887aa29d0ddad8b5aa88ea96064bc41493b9e9315b9aa456cd19a7a4d52010.jpg)

![40d2f5cd8ba45363bb53107f514c37b39ec543d61ecb2c0f754559138746694e.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/40d2f5cd8ba45363bb53107f514c37b39ec543d61ecb2c0f754559138746694e.jpg)

![5032cfeb42d9c86b4745f9f6727b984483bcc0762121026c156bc2ee9af72b31.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/5032cfeb42d9c86b4745f9f6727b984483bcc0762121026c156bc2ee9af72b31.jpg)

![6304b92846952cfe67f4121d0eea8509ba85d87644d06e3060828733d7ae2359.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/6304b92846952cfe67f4121d0eea8509ba85d87644d06e3060828733d7ae2359.jpg)

![8044116bab059b9055a176cfd886410926fd5f6eaea529bece8c3871d653a931.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/8044116bab059b9055a176cfd886410926fd5f6eaea529bece8c3871d653a931.jpg)

![808b22029d84b6d8256674be23069f8c9e7a02e22e742dc7efe43249845638c0.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/808b22029d84b6d8256674be23069f8c9e7a02e22e742dc7efe43249845638c0.jpg)

![8097c5799e05069a8291386a2d98db4cd06c6d4786db1b434f7758ebbe982a8e.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/8097c5799e05069a8291386a2d98db4cd06c6d4786db1b434f7758ebbe982a8e.jpg)

![933bf9d4490de56d8abb4c41c501c7d5185e87f657cccaa9683ab681888136bf.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/933bf9d4490de56d8abb4c41c501c7d5185e87f657cccaa9683ab681888136bf.jpg)

![996501eeb3807722695aaa99abad586de758c246924e19630b37a38453832ed2.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/996501eeb3807722695aaa99abad586de758c246924e19630b37a38453832ed2.jpg)

![9b4098a9ec7652025ba47f31e238a65d2413f40e3e9d7e8a952c69f4bfa60520.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/9b4098a9ec7652025ba47f31e238a65d2413f40e3e9d7e8a952c69f4bfa60520.jpg)

![9e4126487c451db280a37f2caa9e52afa8e392fe80f5f1c8c93a91b9de7876f2.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/9e4126487c451db280a37f2caa9e52afa8e392fe80f5f1c8c93a91b9de7876f2.jpg)

![ca4596bb0793b132d86ef0c38f792fd9ed3f733176e84c4ef7863ee82aca1a62.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/ca4596bb0793b132d86ef0c38f792fd9ed3f733176e84c4ef7863ee82aca1a62.jpg)

![d674be2ac7a76fcd0159c4f20e5fef973baaae1e6acd6680e6ead566f4aa3d2d.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/d674be2ac7a76fcd0159c4f20e5fef973baaae1e6acd6680e6ead566f4aa3d2d.jpg)

![ef904c68d217c46890707ffe8af59b7d0160850aac073995f1a062558b9151a1.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/ef904c68d217c46890707ffe8af59b7d0160850aac073995f1a062558b9151a1.jpg)

![fc5ac1dd20f0762a856c4cf81911b86af5508e7e680eccda8a1a77b11581932f.jpg](../icml_results/3972_Can%20Muon%20Fine-tune%20Adam-Pretrained%20Models_/tables/fc5ac1dd20f0762a856c4cf81911b86af5508e7e680eccda8a1a77b11581932f.jpg)

<a id="paper-3932"></a>

## Small Generalizable Prompt Predictive Models Can Steer Efficient RL Post-Training of Large Reasoning Models


### Images

![06cd5344f3bdfe8d72b5f7da3df85fa3315583442ac32c2b99a9ff1d8e30bc3a.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/06cd5344f3bdfe8d72b5f7da3df85fa3315583442ac32c2b99a9ff1d8e30bc3a.jpg)

![36b1a7ac1c0e0daf6aef45ef5385393e49feafcc2a7a96e7e2cc79ce915526ea.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/36b1a7ac1c0e0daf6aef45ef5385393e49feafcc2a7a96e7e2cc79ce915526ea.jpg)

![5522133f546b26b6c86dee1c41131eee3320e74bf1dcec693600c6b0c6af170a.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/5522133f546b26b6c86dee1c41131eee3320e74bf1dcec693600c6b0c6af170a.jpg)

![58db446b307249452ead9bb00ee294092a9563ed1d1c380c0c008c4a5fd68c81.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/58db446b307249452ead9bb00ee294092a9563ed1d1c380c0c008c4a5fd68c81.jpg)

![826534c3e99e70bf7b07eb7cd5a76a3c70ef491ec5d4d878bab7543e7fdf6d09.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/826534c3e99e70bf7b07eb7cd5a76a3c70ef491ec5d4d878bab7543e7fdf6d09.jpg)

![90309b76151951fc7e0af8a1dc2e384fe02226ad6535a24445f64fffe103e114.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/90309b76151951fc7e0af8a1dc2e384fe02226ad6535a24445f64fffe103e114.jpg)

![a35b409a65b6a8b327ea996fad652f8fea50ebdd94eac5cd69a16e0de53f3512.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/a35b409a65b6a8b327ea996fad652f8fea50ebdd94eac5cd69a16e0de53f3512.jpg)

![c62f2069e786f8b0f5fd947ed84c4c18d9b362856ab2fa85584a9a9d203e0a52.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/c62f2069e786f8b0f5fd947ed84c4c18d9b362856ab2fa85584a9a9d203e0a52.jpg)

![c8755cb5ce035e84c391431a5e93efdd41072fed9372b017d2bd5c8963305cee.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/c8755cb5ce035e84c391431a5e93efdd41072fed9372b017d2bd5c8963305cee.jpg)

![d507931fb5841d605cb2c04365dedf8811537926189ec11a3054c1a790a73c07.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/d507931fb5841d605cb2c04365dedf8811537926189ec11a3054c1a790a73c07.jpg)

![d53b6b46f830407dcd968d6e826e1310fb09871bf080218c44aea3f3f5aa56b8.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/d53b6b46f830407dcd968d6e826e1310fb09871bf080218c44aea3f3f5aa56b8.jpg)

![d6c7027b20892ef1c5a25e81c5611f00e5ebbfeba8f937310833e3075ab9bc79.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/d6c7027b20892ef1c5a25e81c5611f00e5ebbfeba8f937310833e3075ab9bc79.jpg)

![d9653f8456b78fe93c897055d0394eef0227c81c4a90e80be33d029dc857e44b.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/images/d9653f8456b78fe93c897055d0394eef0227c81c4a90e80be33d029dc857e44b.jpg)

### Tables

![284c0ad7265a326cb9c3bd25b96fb34405105a5b2a31c13ea9c28815512a2733.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/284c0ad7265a326cb9c3bd25b96fb34405105a5b2a31c13ea9c28815512a2733.jpg)

![6263f254ec85b397bb3e0f44505ada9c4cdc059b20b0d33bcd5097d02878130d.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/6263f254ec85b397bb3e0f44505ada9c4cdc059b20b0d33bcd5097d02878130d.jpg)

![7207f63800fb9342017a30fcd9bfe36794948c1268a247f777643ed56250cac3.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/7207f63800fb9342017a30fcd9bfe36794948c1268a247f777643ed56250cac3.jpg)

![93f97413e7883520a854833f20dfa695162cccace6cf0bb68e35e74c2b53841e.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/93f97413e7883520a854833f20dfa695162cccace6cf0bb68e35e74c2b53841e.jpg)

![9a7e42e60c28b560102377cea38ac8e8ef6b249683e502e2055df0e4fe025fbd.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/9a7e42e60c28b560102377cea38ac8e8ef6b249683e502e2055df0e4fe025fbd.jpg)

![b88f08151c3756a932cae4ee05a73a2756f0f625ce310ad02ba64100f1a7d714.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/b88f08151c3756a932cae4ee05a73a2756f0f625ce310ad02ba64100f1a7d714.jpg)

![e8a1744bdec4e43f934ec0870b68eaaf31ea682347f2d6c8feb31e6b6307c4d8.jpg](../icml_results/3973_Small%20Generalizable%20Prompt%20Predictive%20Models%20Can%20Steer%20Efficient%20RL%20Post-Training%20of%20Large%20Reasoning/tables/e8a1744bdec4e43f934ec0870b68eaaf31ea682347f2d6c8feb31e6b6307c4d8.jpg)

<a id="paper-3933"></a>

## A recipe for scalable attention-based ML potentials: unlocking long-range accuracy with all-to-all node attention


### Images

![2b726af0dd966bdf73c3dec473e58b49b02543b47606c622eb3563bf41c5d075.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/2b726af0dd966bdf73c3dec473e58b49b02543b47606c622eb3563bf41c5d075.jpg)

![2b9c0fe2f1451a87f163e2b9408be7cfd4865c992189b19c0385fff3472aa4bd.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/2b9c0fe2f1451a87f163e2b9408be7cfd4865c992189b19c0385fff3472aa4bd.jpg)

![4f66ecae1c220b9bfe21baffda897fefbbbe54ac1177cfea68542b6b2c3bbec5.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/4f66ecae1c220b9bfe21baffda897fefbbbe54ac1177cfea68542b6b2c3bbec5.jpg)

![58e6b040c78ece221b1697fe33f97e1dc62e74861844791a87d4a48e1f15f7c1.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/58e6b040c78ece221b1697fe33f97e1dc62e74861844791a87d4a48e1f15f7c1.jpg)

![5e0400effaa676b0f505679a33ec08ae245673e76a2e2ac4420920ffd918b9bd.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/5e0400effaa676b0f505679a33ec08ae245673e76a2e2ac4420920ffd918b9bd.jpg)

![6759bf0beffa09f2d61f3f6e514ba54acd561ad77d8870631fdf5b0a37d45261.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/6759bf0beffa09f2d61f3f6e514ba54acd561ad77d8870631fdf5b0a37d45261.jpg)

![7203b4a4ccfdd0ed7486985e2f89bfdeb5b8145a865100f69725400d00ec7f91.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/7203b4a4ccfdd0ed7486985e2f89bfdeb5b8145a865100f69725400d00ec7f91.jpg)

![799f9ce33b658f30a468afcb3487719b31973cf85d944746d50fa9a67bae639f.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/799f9ce33b658f30a468afcb3487719b31973cf85d944746d50fa9a67bae639f.jpg)

![7f38c06fd9f81b6b85318969d91924336f2e70cd6fc959c726c8bb96821f8469.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/7f38c06fd9f81b6b85318969d91924336f2e70cd6fc959c726c8bb96821f8469.jpg)

![ac6ff3312f61fe032a628756fa88617ed537c00ae7b9fc172901986b006dff34.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/ac6ff3312f61fe032a628756fa88617ed537c00ae7b9fc172901986b006dff34.jpg)

![bb1e42a860ebc9be728cceaf3bfe8a16628d7e011d5ed8fca9862535cab686a7.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/bb1e42a860ebc9be728cceaf3bfe8a16628d7e011d5ed8fca9862535cab686a7.jpg)

![d415cc08702ffa37c12a9e9ce230554a02777ca0bd194fc041765548a82baae0.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/d415cc08702ffa37c12a9e9ce230554a02777ca0bd194fc041765548a82baae0.jpg)

![febc2028342c1af0f6b9d1593c1af07ad276abd505b1f5e758ca096c8c6a7fcc.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/images/febc2028342c1af0f6b9d1593c1af07ad276abd505b1f5e758ca096c8c6a7fcc.jpg)

### Tables

![29770c7ce8d44687931eb57aa6191f09130cc23301d5791b792fd76083766e41.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/29770c7ce8d44687931eb57aa6191f09130cc23301d5791b792fd76083766e41.jpg)

![440a9ac9e501697ebe21a5b71f209abffb7876ff152714f5b90405c98e2fc8a8.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/440a9ac9e501697ebe21a5b71f209abffb7876ff152714f5b90405c98e2fc8a8.jpg)

![8e3ff74d1cbcddb1863c803b1ce255144c4d1a6dce805f9e893840de2bdec640.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/8e3ff74d1cbcddb1863c803b1ce255144c4d1a6dce805f9e893840de2bdec640.jpg)

![9324b273b0ee62f7216bc8adca5d79c52d841f26a9ea164e0f4978f472ab362c.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/9324b273b0ee62f7216bc8adca5d79c52d841f26a9ea164e0f4978f472ab362c.jpg)

![a479f3e498413256ef69bf3170a2fcedc9b66ba4f30b2a3cccedac80607e2188.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/a479f3e498413256ef69bf3170a2fcedc9b66ba4f30b2a3cccedac80607e2188.jpg)

![b78cc48f3e19bb24b1da843e8876abda00423b7cc761a0474303fe83b9028751.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/b78cc48f3e19bb24b1da843e8876abda00423b7cc761a0474303fe83b9028751.jpg)

![b803ed40b384fec67e0044c8270a138e26b5b2aa68e4938e84ed45f1685c8f4c.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/b803ed40b384fec67e0044c8270a138e26b5b2aa68e4938e84ed45f1685c8f4c.jpg)

![d4e575f4cab842b92c4ccb0113712ccf823d20f70ff5e0e90051e3f3fd95deb9.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/d4e575f4cab842b92c4ccb0113712ccf823d20f70ff5e0e90051e3f3fd95deb9.jpg)

![f44a6540ed91fb230ac6cc45b2a7f3778c1b6230484dee4713534dc6b27942e5.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/f44a6540ed91fb230ac6cc45b2a7f3778c1b6230484dee4713534dc6b27942e5.jpg)

![fa670a334db8dd0bfd1764fd053296fdb2561625ed396e4a2e102502e4f97dd4.jpg](../icml_results/3974_A%20recipe%20for%20scalable%20attention-based%20ML%20potentials_%20unlocking%20long-range%20accuracy%20with%20all-to-all%20n/tables/fa670a334db8dd0bfd1764fd053296fdb2561625ed396e4a2e102502e4f97dd4.jpg)

<a id="paper-3934"></a>

## Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation


### Images

![04a5d6c27b2eaa8c42dc4c2638374e2c6c3531adca21c10eb1659664b11a7b2b.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/04a5d6c27b2eaa8c42dc4c2638374e2c6c3531adca21c10eb1659664b11a7b2b.jpg)

![15d7851e9e98c3ee2170ca2680b9ad328da4a8834f374487c56e8f0effba7d1b.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/15d7851e9e98c3ee2170ca2680b9ad328da4a8834f374487c56e8f0effba7d1b.jpg)

![177f459b7b25e5cdff902a97fa87c201f27502475b0b76e76ac13a3d7dace2af.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/177f459b7b25e5cdff902a97fa87c201f27502475b0b76e76ac13a3d7dace2af.jpg)

![22a79db53ae7e43692a1de1909315cd9267e926c4890e39dae2799b5662177ca.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/22a79db53ae7e43692a1de1909315cd9267e926c4890e39dae2799b5662177ca.jpg)

![35902835f85c1ba794962130ead3142c9ed60e5658d4aa7311cc08b20a034599.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/35902835f85c1ba794962130ead3142c9ed60e5658d4aa7311cc08b20a034599.jpg)

![3ab739d243dd0417b586342849bddd01be4f473ef9bf8c85d7d273693e662af3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/3ab739d243dd0417b586342849bddd01be4f473ef9bf8c85d7d273693e662af3.jpg)

![40799bea3fe2809cda968503bb06ce2e5a484c9dc5a28c45e7566b437ef028ea.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/40799bea3fe2809cda968503bb06ce2e5a484c9dc5a28c45e7566b437ef028ea.jpg)

![42b66adbadcc569b4bb77cb1026d6074d88633069905668d9565a44205fbdae2.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/42b66adbadcc569b4bb77cb1026d6074d88633069905668d9565a44205fbdae2.jpg)

![43aba67f0cb8adbe37897ca167ce3dab937ba9a43fa757e89c160dba14ff5acd.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/43aba67f0cb8adbe37897ca167ce3dab937ba9a43fa757e89c160dba14ff5acd.jpg)

![44d9bb3f05b61636ed16d1a8a9edaed0c7099ab22832830b3b380e8b95c3690f.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/44d9bb3f05b61636ed16d1a8a9edaed0c7099ab22832830b3b380e8b95c3690f.jpg)

![460e5c5bc18263d090c35a5d2a8bb0ae5bb3c9b7e104940db31683d1aa34f73c.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/460e5c5bc18263d090c35a5d2a8bb0ae5bb3c9b7e104940db31683d1aa34f73c.jpg)

![46da5ac4ce294a7e6ccce767445c4d9f9a4ed5e2d03cfa066ef9f42dab5f1be3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/46da5ac4ce294a7e6ccce767445c4d9f9a4ed5e2d03cfa066ef9f42dab5f1be3.jpg)

![47204798dbbeea9558a324bc8936c3322d9a142e072edb4bb28618f895885d91.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/47204798dbbeea9558a324bc8936c3322d9a142e072edb4bb28618f895885d91.jpg)

![52f4987ac86d5c8a44992c1af2bd30dcad4f38094e701766e5689de46fb61e24.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/52f4987ac86d5c8a44992c1af2bd30dcad4f38094e701766e5689de46fb61e24.jpg)

![595b2979d746497ce067de6575deca6dfee91ad23e8ba162ae8e24fc987213d3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/595b2979d746497ce067de6575deca6dfee91ad23e8ba162ae8e24fc987213d3.jpg)

![5fe17803c65872efc2e66b50c458cd46b8884c7dc6fa0fc03c48871296f26035.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/5fe17803c65872efc2e66b50c458cd46b8884c7dc6fa0fc03c48871296f26035.jpg)

![6276ee8bbfa60b1146f14f8a19b227c1f6975744ed9687aa188dc5dcc41c58a4.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/6276ee8bbfa60b1146f14f8a19b227c1f6975744ed9687aa188dc5dcc41c58a4.jpg)

![78884a362b370a76e0eb2fce4481319727f72f4090e57a278357b5f037ac22df.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/78884a362b370a76e0eb2fce4481319727f72f4090e57a278357b5f037ac22df.jpg)

![82206dd34ebc5c27da64c41afc1632bd0ab3c320b48e4ee62f1d4b1c4470b2c3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/82206dd34ebc5c27da64c41afc1632bd0ab3c320b48e4ee62f1d4b1c4470b2c3.jpg)

![88f7887542b9bf89c8af42b0fa4b3e98d139fa4fc850765f73f591b204a191ba.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/88f7887542b9bf89c8af42b0fa4b3e98d139fa4fc850765f73f591b204a191ba.jpg)

![be6888b94f988947d414c30c9b5d5104684f6d73d9c9a8cfa23556ce775a36b1.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/be6888b94f988947d414c30c9b5d5104684f6d73d9c9a8cfa23556ce775a36b1.jpg)

![c7517b18ab9140a63bdb82c12855aec8e872092c1d16a520dc09fff0eb4439a3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/c7517b18ab9140a63bdb82c12855aec8e872092c1d16a520dc09fff0eb4439a3.jpg)

![d27acb568501a4d01aad3912cf1ae068daf61db3a716d6ade02f92ce18e52e45.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/d27acb568501a4d01aad3912cf1ae068daf61db3a716d6ade02f92ce18e52e45.jpg)

![e661d91447376c5e4c78c34f3681ea29c448f81e71f0b7d76d924d68e0006ee3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/e661d91447376c5e4c78c34f3681ea29c448f81e71f0b7d76d924d68e0006ee3.jpg)

![e76c6615a4097a8c957ae42e5f1df9a7ab23e6bfad7f3369aba5adc640e464b5.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/e76c6615a4097a8c957ae42e5f1df9a7ab23e6bfad7f3369aba5adc640e464b5.jpg)

![f99d53473d3b8eb01f56bf45ef9d8748a7e3cf1eedac6f85f9c5f779474f063f.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/images/f99d53473d3b8eb01f56bf45ef9d8748a7e3cf1eedac6f85f9c5f779474f063f.jpg)

### Tables

![0e2ef578e3805db0a0e0ee38ea0f0a3f88db5f046435f0b06dd511b5fe5695e1.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/0e2ef578e3805db0a0e0ee38ea0f0a3f88db5f046435f0b06dd511b5fe5695e1.jpg)

![183cd7f9ff41d08e07fb4861646c46d2f0a4ad98b79bdaff3e8e8da5f624f360.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/183cd7f9ff41d08e07fb4861646c46d2f0a4ad98b79bdaff3e8e8da5f624f360.jpg)

![201744389f6e039ee58e0c45410420e567e3bd4b6f554067508f6bd0df91f974.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/201744389f6e039ee58e0c45410420e567e3bd4b6f554067508f6bd0df91f974.jpg)

![383c21330320dd80e8434ed69dd2fd38f1b38bae05132e06c7f363e4577bc9bb.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/383c21330320dd80e8434ed69dd2fd38f1b38bae05132e06c7f363e4577bc9bb.jpg)

![41b52532cf12d1c905c0a48d02e6783a47bf1462c732ca16c10b1a2a562522ed.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/41b52532cf12d1c905c0a48d02e6783a47bf1462c732ca16c10b1a2a562522ed.jpg)

![4500f3909f14debe3d89b9ebf8c78effbbed2a6d6bf64a8aead140968265207c.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/4500f3909f14debe3d89b9ebf8c78effbbed2a6d6bf64a8aead140968265207c.jpg)

![48fb2e4151086456c558d7b8642928595bfbf154bf6bbb149d314f0d54dcc843.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/48fb2e4151086456c558d7b8642928595bfbf154bf6bbb149d314f0d54dcc843.jpg)

![4e84d65acd8078a6ee9dc64cb5d3b12114427a4c59bb98f0247442456a661db1.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/4e84d65acd8078a6ee9dc64cb5d3b12114427a4c59bb98f0247442456a661db1.jpg)

![6808d93915eaa3b02af6260932479d536908417cbffd01b83f6de9b7fc0a2bb3.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/6808d93915eaa3b02af6260932479d536908417cbffd01b83f6de9b7fc0a2bb3.jpg)

![7f96d46477488c501d811ab039c579ab4c21d9b49967aa486e060981ce991e1f.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/7f96d46477488c501d811ab039c579ab4c21d9b49967aa486e060981ce991e1f.jpg)

![84138b10b174c4e36160817d241a14fa19ff260276a6fd8a24c20cc507d803f2.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/84138b10b174c4e36160817d241a14fa19ff260276a6fd8a24c20cc507d803f2.jpg)

![887754a18d5ec0d92a0875909594c5bbc557607ce72287d714e48e62ecd2c817.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/887754a18d5ec0d92a0875909594c5bbc557607ce72287d714e48e62ecd2c817.jpg)

![8b6daaef20957c95de622f6c26edf528a8bfe8cb3d711544c539081cef9149fd.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/8b6daaef20957c95de622f6c26edf528a8bfe8cb3d711544c539081cef9149fd.jpg)

![90de4965469721d215197ca6113e6d6be7bd80dfbb4ed6fc9ad89d03a127749e.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/90de4965469721d215197ca6113e6d6be7bd80dfbb4ed6fc9ad89d03a127749e.jpg)

![96ff840aaccc4e92f4be0c8d16c16f83ef6842b8fb37e346562046c9dcef527f.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/96ff840aaccc4e92f4be0c8d16c16f83ef6842b8fb37e346562046c9dcef527f.jpg)

![9945abc557133ae2d3cc0e3367db4b744b78e046bde821c465012dacfc8f91ce.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/9945abc557133ae2d3cc0e3367db4b744b78e046bde821c465012dacfc8f91ce.jpg)

![a04049655a0c229a936969f13268a82b7567aee8510cdd4fd59de466c11287ce.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/a04049655a0c229a936969f13268a82b7567aee8510cdd4fd59de466c11287ce.jpg)

![af0db1cb8b2513759e3ae9b70e0df54d05923659488a07d8fe7b04c214ab132d.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/af0db1cb8b2513759e3ae9b70e0df54d05923659488a07d8fe7b04c214ab132d.jpg)

![bf42f2dc5ad194c80af50324d16eda10657a4a590fa35b868609238769ef87b6.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/bf42f2dc5ad194c80af50324d16eda10657a4a590fa35b868609238769ef87b6.jpg)

![c470c09c0e5fcc757743401e9a70a1493c3b7eb0abd1b9402c3264d05b74dbdd.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/c470c09c0e5fcc757743401e9a70a1493c3b7eb0abd1b9402c3264d05b74dbdd.jpg)

![dbfcafb129123a13c78b223f35831ecabb659d88bc225947d04d870f34992091.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/dbfcafb129123a13c78b223f35831ecabb659d88bc225947d04d870f34992091.jpg)

![e0678fe1aa549a63eca0b09c28e7a92f323376bec15f194c24ebe283b6a3e86f.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/e0678fe1aa549a63eca0b09c28e7a92f323376bec15f194c24ebe283b6a3e86f.jpg)

![e1e090940083b5e490d38e9c7992e23cf3dce31bb5e48fa12dd48b310078c6db.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/e1e090940083b5e490d38e9c7992e23cf3dce31bb5e48fa12dd48b310078c6db.jpg)

![f315bc367d29352a6d86ec5badd69fab62ae1c05eb60f196b076336cc9647bc6.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/f315bc367d29352a6d86ec5badd69fab62ae1c05eb60f196b076336cc9647bc6.jpg)

![fdfbc8dfa6baf5dae8f03d7a3daf38af3b10b342bf75d7989279350f7d9bd0b9.jpg](../icml_results/3975_Scaling%2C%20Benchmarking%2C%20and%20Reasoning%20of%20Vision-Language%20Agents%20for%20Mobile%20GUI%20Navigation/tables/fdfbc8dfa6baf5dae8f03d7a3daf38af3b10b342bf75d7989279350f7d9bd0b9.jpg)

<a id="paper-3935"></a>

## T-POP: Test-Time Personalization with Online Preference Feedback


### Images

![27d34272f36dcf52682273b0a223a2dcb2cd84229e309950d41c6346e9a712a9.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/images/27d34272f36dcf52682273b0a223a2dcb2cd84229e309950d41c6346e9a712a9.jpg)

![561b59fd5930811d8364e73b458bd6cb3e7a9e3a74b9c94597c2df4d00dbfc41.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/images/561b59fd5930811d8364e73b458bd6cb3e7a9e3a74b9c94597c2df4d00dbfc41.jpg)

![7340b33f9d472feb31d8f8126b035290db5be23accc8539a0f95d3129b297d6a.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/images/7340b33f9d472feb31d8f8126b035290db5be23accc8539a0f95d3129b297d6a.jpg)

![99afa01ef7248905a04d5a34fc9dffd8e5c2ee4f39fd996999368db864dcb14d.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/images/99afa01ef7248905a04d5a34fc9dffd8e5c2ee4f39fd996999368db864dcb14d.jpg)

![daa8658702be535f12325772e81c5ad6ae654cad1945a70ca7336b933708c92e.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/images/daa8658702be535f12325772e81c5ad6ae654cad1945a70ca7336b933708c92e.jpg)

### Tables

![18a84448c451d32779869b8240aaded2be0757afd2aeb5a53ebcfe97ecfbb160.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/18a84448c451d32779869b8240aaded2be0757afd2aeb5a53ebcfe97ecfbb160.jpg)

![418318862ee0d2fd1162f08743c5ad17f1cc47d9ee546cd6b670b4ed4e9a7f49.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/418318862ee0d2fd1162f08743c5ad17f1cc47d9ee546cd6b670b4ed4e9a7f49.jpg)

![6f3317b046b3e2b37338177c37e43a4ca5eccf4a13f7777814dda19d87c17675.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/6f3317b046b3e2b37338177c37e43a4ca5eccf4a13f7777814dda19d87c17675.jpg)

![74c28a7760ab8bce68856ca0c907f02fa8d836a5796212a1af8c5fa24eb2c211.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/74c28a7760ab8bce68856ca0c907f02fa8d836a5796212a1af8c5fa24eb2c211.jpg)

![79d02e35e3518bd509782973e5c47d0f45045dfb3ceec8657d750bee76c6269d.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/79d02e35e3518bd509782973e5c47d0f45045dfb3ceec8657d750bee76c6269d.jpg)

![9bd3f163e16438405518ddff778eccb6289fd270a675f6cfb9a0a0b9279c4ec7.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/9bd3f163e16438405518ddff778eccb6289fd270a675f6cfb9a0a0b9279c4ec7.jpg)

![a5673086e344e572cd60a5999c320ef01f9d901582481147a25b5c6866decf83.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/a5673086e344e572cd60a5999c320ef01f9d901582481147a25b5c6866decf83.jpg)

![b1954a741e267bfa935e73e183f7bb47f83f245c309824d35b69d076544fd514.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/b1954a741e267bfa935e73e183f7bb47f83f245c309824d35b69d076544fd514.jpg)

![c4e72e7ee94d528e8e90ee6a6a87dd7442ab9041325cdded78d86f74aa5f4f38.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/c4e72e7ee94d528e8e90ee6a6a87dd7442ab9041325cdded78d86f74aa5f4f38.jpg)

![c74834c0d57ef547317f254d6add04f9e455436fae04f60bd4b6903f6eaea88b.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/c74834c0d57ef547317f254d6add04f9e455436fae04f60bd4b6903f6eaea88b.jpg)

![c97674ecd44502e4f267281cd41f90e18a46221181bf8e4cf6c460165fa9d6db.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/c97674ecd44502e4f267281cd41f90e18a46221181bf8e4cf6c460165fa9d6db.jpg)

![caf3e58a949b42a83daf93fd7b3df4a0a9d9c222c7cbd57a4ab0c9292fad4dcd.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/caf3e58a949b42a83daf93fd7b3df4a0a9d9c222c7cbd57a4ab0c9292fad4dcd.jpg)

![d241c5a94822ade9f05552f235f544ae5a547a3cd33fba7275b8309fb256845f.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/d241c5a94822ade9f05552f235f544ae5a547a3cd33fba7275b8309fb256845f.jpg)

![ee112e16b9309dc7d764d415ec117f74072c75fab63841e12af45c0938bc45c6.jpg](../icml_results/3976_T-POP_%20Test-Time%20Personalization%20with%20Online%20Preference%20Feedback/tables/ee112e16b9309dc7d764d415ec117f74072c75fab63841e12af45c0938bc45c6.jpg)

<a id="paper-3936"></a>

## Interpretability and Generalization Bounds for Learning Spatial Physics


### Images

![0fad93c5d4d95e6695c3acf93a7909bdd2d582417cf6769b99cf82000ad94157.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/0fad93c5d4d95e6695c3acf93a7909bdd2d582417cf6769b99cf82000ad94157.jpg)

![2539681cbe16dba8448f74728ced2c55356f4d30eec8c6bfaa23ccd2812c620d.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/2539681cbe16dba8448f74728ced2c55356f4d30eec8c6bfaa23ccd2812c620d.jpg)

![28c9a0228f842408668b13a1337e531b0dfd9f930b7ff92932cdec9313f83486.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/28c9a0228f842408668b13a1337e531b0dfd9f930b7ff92932cdec9313f83486.jpg)

![532de3c088e7d3a33f4d2c4ef58d8242a54c72d0c44f722179e874de4b1a3d2f.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/532de3c088e7d3a33f4d2c4ef58d8242a54c72d0c44f722179e874de4b1a3d2f.jpg)

![79787df33ad773ed0169b6afe7058ab75820aafa753799f1287cba4ea9458c8f.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/79787df33ad773ed0169b6afe7058ab75820aafa753799f1287cba4ea9458c8f.jpg)

![8be2fac4c59efa5e599590ce7d614c4db3c922fa0d34e308a3ddffbdef2853c2.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/8be2fac4c59efa5e599590ce7d614c4db3c922fa0d34e308a3ddffbdef2853c2.jpg)

![8ed631dcd07457cff9458b26b9f8825af489c6f896d8efd71996b04159d7e0ad.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/8ed631dcd07457cff9458b26b9f8825af489c6f896d8efd71996b04159d7e0ad.jpg)

![9482f3ab76b4cffdbe9354218212d885984ea81275a3eb010db9b9a5a7bc7bdf.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/9482f3ab76b4cffdbe9354218212d885984ea81275a3eb010db9b9a5a7bc7bdf.jpg)

![b6c73eacf78876ed1fed633eaa91eca2c1a0b41837734f530e8278c204c4d813.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/b6c73eacf78876ed1fed633eaa91eca2c1a0b41837734f530e8278c204c4d813.jpg)

![bb162538e05e2f642e9a681074523a8c1fb519ff8cea8edb64063fd4545a89b3.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/bb162538e05e2f642e9a681074523a8c1fb519ff8cea8edb64063fd4545a89b3.jpg)

![bf03ce5464002db300d265e7d185f5aad37875229825ce2bdd16df7c68be878b.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/bf03ce5464002db300d265e7d185f5aad37875229825ce2bdd16df7c68be878b.jpg)

![dff13dbf3021af20527e2d3a9d29667a23fbf157c6cbc3d7328ff123202f2037.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/dff13dbf3021af20527e2d3a9d29667a23fbf157c6cbc3d7328ff123202f2037.jpg)

![e670399afed3f13771c0180d9db779262fd1ec5087109a495b9f693dbaa5a3d7.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/e670399afed3f13771c0180d9db779262fd1ec5087109a495b9f693dbaa5a3d7.jpg)

![ffc780494d9dde6e553b0ef07194978f07c79f91e21bbeadc3ec37763c592233.jpg](../icml_results/3977_Interpretability%20and%20Generalization%20Bounds%20for%20Learning%20Spatial%20Physics/images/ffc780494d9dde6e553b0ef07194978f07c79f91e21bbeadc3ec37763c592233.jpg)

<a id="paper-3937"></a>

## Towards a Science of AI Agent Reliability


### Images

![0735d570c424ba9e95e0f2c0d8e3c3623442299443a0e720220d470fccbb20b4.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/0735d570c424ba9e95e0f2c0d8e3c3623442299443a0e720220d470fccbb20b4.jpg)

![0f15bde7c48abc78cd152c82528bd84d6f5b6a5834cd5575b73ae4bae233ac5b.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/0f15bde7c48abc78cd152c82528bd84d6f5b6a5834cd5575b73ae4bae233ac5b.jpg)

![10489991e6612241631a5f15fa49e7ad541ad9f08df8dacd5bb3c79b7d8d8f58.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/10489991e6612241631a5f15fa49e7ad541ad9f08df8dacd5bb3c79b7d8d8f58.jpg)

![205d64313eb7cf779d93c0b770072a04a15509e2dded9b71a8549f45bfbe9a25.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/205d64313eb7cf779d93c0b770072a04a15509e2dded9b71a8549f45bfbe9a25.jpg)

![372411e92c6351310756c6ccdc700e75ef8f5cbc85d5c2063c5557d1f8576ce8.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/372411e92c6351310756c6ccdc700e75ef8f5cbc85d5c2063c5557d1f8576ce8.jpg)

![3eec515fb5cb48571b8cae2ade350ee5663008bc3d72275031c0a340fd9a3269.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/3eec515fb5cb48571b8cae2ade350ee5663008bc3d72275031c0a340fd9a3269.jpg)

![594b389c67137f9c5e2b46f723769658bd5eb4c37e584f65cd14af5f034e6c15.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/594b389c67137f9c5e2b46f723769658bd5eb4c37e584f65cd14af5f034e6c15.jpg)

![60d0c79d76b150a848e656d979d2b0986dc7da3010cf725985e9751d61be24ae.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/60d0c79d76b150a848e656d979d2b0986dc7da3010cf725985e9751d61be24ae.jpg)

![82d52727a32dfd38f4010d8607e81becef550e6f61e90e8a0fa813e708ca6d9b.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/82d52727a32dfd38f4010d8607e81becef550e6f61e90e8a0fa813e708ca6d9b.jpg)

![8a2fe046fb97319321508a1dbf8b70dc769f087f3543365e2cd51dbf9d623338.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/8a2fe046fb97319321508a1dbf8b70dc769f087f3543365e2cd51dbf9d623338.jpg)

![9d826cab04aee79ae2ada8ddc1aa69df99b3a6bef723954018a52253757f8afd.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/9d826cab04aee79ae2ada8ddc1aa69df99b3a6bef723954018a52253757f8afd.jpg)

![b1183931ec47ed388af896fabd4345b9f4cc000e2212a93c8e2819b4304874cf.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/b1183931ec47ed388af896fabd4345b9f4cc000e2212a93c8e2819b4304874cf.jpg)

![c11f6902a96b303951af2d0240b16494982577ad4798cca56cc9b27f123960e6.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/c11f6902a96b303951af2d0240b16494982577ad4798cca56cc9b27f123960e6.jpg)

![c686d1fc7791ffa92dd357dae20d32ce0ea09887e0623004f8616dbeb84d6354.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/c686d1fc7791ffa92dd357dae20d32ce0ea09887e0623004f8616dbeb84d6354.jpg)

![cc113b0d796c0f981acc5c728b1b0c5696556a0a900502ad117a09311c037c10.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/cc113b0d796c0f981acc5c728b1b0c5696556a0a900502ad117a09311c037c10.jpg)

![d8f42a8fdc3cc611b362ed69eb987a462306cba870be841390b3515f0e4d2aa3.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/d8f42a8fdc3cc611b362ed69eb987a462306cba870be841390b3515f0e4d2aa3.jpg)

![db9782226e25776fe2f49a4ea7516ccc3cc9b4b4e5cf9ea9f1e71b7d32a935d7.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/db9782226e25776fe2f49a4ea7516ccc3cc9b4b4e5cf9ea9f1e71b7d32a935d7.jpg)

![e7ffa819a2027015446e7f73ba8ebb67481f2183245d4ce18f9bf1459dba098f.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/e7ffa819a2027015446e7f73ba8ebb67481f2183245d4ce18f9bf1459dba098f.jpg)

![f7c7fd3c49495d84d87ff23a8bd50ac6371921bad3a0cf8ad1775db7c1323f80.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/f7c7fd3c49495d84d87ff23a8bd50ac6371921bad3a0cf8ad1775db7c1323f80.jpg)

![fe76f324f013c772ac27313244edfc12712f616c41064c8385f4c92556dc8990.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/images/fe76f324f013c772ac27313244edfc12712f616c41064c8385f4c92556dc8990.jpg)

### Tables

![182bafb941ea9c51d63ab5ec72ac408f03787a912863917c90892dd084835e12.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/182bafb941ea9c51d63ab5ec72ac408f03787a912863917c90892dd084835e12.jpg)

![406cec7ebd2515dd14e0a2baad154b146291c80e96d1e8ac32c163c7955df45b.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/406cec7ebd2515dd14e0a2baad154b146291c80e96d1e8ac32c163c7955df45b.jpg)

![495d6bdb3668fba88406f08f14342591ad43d059fff73511e4ba3001e23c887a.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/495d6bdb3668fba88406f08f14342591ad43d059fff73511e4ba3001e23c887a.jpg)

![5cb27c51ad98b3b77168a907bc8eb2cbea61d1ab874c7513e688c59b35dd4278.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/5cb27c51ad98b3b77168a907bc8eb2cbea61d1ab874c7513e688c59b35dd4278.jpg)

![c3a717fd04f54d98ff370f2d4798fcb6870a25e3f4141a76db50a48268701e6f.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/c3a717fd04f54d98ff370f2d4798fcb6870a25e3f4141a76db50a48268701e6f.jpg)

![cfda11dc469512b5c64c6a2f26565c4e251f91ee467bdbd3bb9a72acfea7b195.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/cfda11dc469512b5c64c6a2f26565c4e251f91ee467bdbd3bb9a72acfea7b195.jpg)

![d9ed7b17dd13953f881d353814dd15ed9534f5f4edc88415fa9e0f24dea33ac1.jpg](../icml_results/3978_Towards%20a%20Science%20of%20AI%20Agent%20Reliability/tables/d9ed7b17dd13953f881d353814dd15ed9534f5f4edc88415fa9e0f24dea33ac1.jpg)

<a id="paper-3938"></a>

## Language Generation with Replay: A Learning-Theoretic View of Model Collapse


### Images

![3b162b9ea53e3e921e517bf87ab4c14e3e7f4bb68911cf405bdb6566c8c07172.jpg](../icml_results/3979_Language%20Generation%20with%20Replay_%20A%20Learning-Theoretic%20View%20of%20Model%20Collapse/images/3b162b9ea53e3e921e517bf87ab4c14e3e7f4bb68911cf405bdb6566c8c07172.jpg)

### Tables

![4f59aa4ca5a2b45855d0bdfe4c443c8d43bdc70e7c82572ace0c1559cea4c1a3.jpg](../icml_results/3979_Language%20Generation%20with%20Replay_%20A%20Learning-Theoretic%20View%20of%20Model%20Collapse/tables/4f59aa4ca5a2b45855d0bdfe4c443c8d43bdc70e7c82572ace0c1559cea4c1a3.jpg)

![58876f71de5b81fe30aeb525ab6c6f4498033152e33f7c64fccbbce3318656d8.jpg](../icml_results/3979_Language%20Generation%20with%20Replay_%20A%20Learning-Theoretic%20View%20of%20Model%20Collapse/tables/58876f71de5b81fe30aeb525ab6c6f4498033152e33f7c64fccbbce3318656d8.jpg)

![6baa3174b82deea066b9daf15d2cb6d534d352d0a1955caec866f695d93b6a4d.jpg](../icml_results/3979_Language%20Generation%20with%20Replay_%20A%20Learning-Theoretic%20View%20of%20Model%20Collapse/tables/6baa3174b82deea066b9daf15d2cb6d534d352d0a1955caec866f695d93b6a4d.jpg)

![c76603f0b69addbd3cc605a8b8fb9f0a5be34dca65aad636f78aee61ed3f8d84.jpg](../icml_results/3979_Language%20Generation%20with%20Replay_%20A%20Learning-Theoretic%20View%20of%20Model%20Collapse/tables/c76603f0b69addbd3cc605a8b8fb9f0a5be34dca65aad636f78aee61ed3f8d84.jpg)

![ddb4172a1bf24c3d19db890b733d12a4f29966c2d0d9a4442d1b8eb779809fda.jpg](../icml_results/3979_Language%20Generation%20with%20Replay_%20A%20Learning-Theoretic%20View%20of%20Model%20Collapse/tables/ddb4172a1bf24c3d19db890b733d12a4f29966c2d0d9a4442d1b8eb779809fda.jpg)

<a id="paper-3939"></a>

## G$^2$RPO: Geometric GRPO Escaping LLMs’ Reasoning Ruts to Break the Accuracy–Entropy Trade-off


### Images

![060ad44ee564c34684a9983e6ac63231529bc3fc9a382e4d36249e35c873deaa.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/060ad44ee564c34684a9983e6ac63231529bc3fc9a382e4d36249e35c873deaa.jpg)

![392e0bdc0e9747e6fdcc821e5dc1b3e4484352b56e32a132751347722c61499e.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/392e0bdc0e9747e6fdcc821e5dc1b3e4484352b56e32a132751347722c61499e.jpg)

![415ec54fdd680e357a8d52efc00e21368516ccda23ea5467446aac5d86e893ea.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/415ec54fdd680e357a8d52efc00e21368516ccda23ea5467446aac5d86e893ea.jpg)

![43a52a60651c4edc39d2d3a056745ff52259e46d3d4bbd7ef4a4483fe7cfec75.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/43a52a60651c4edc39d2d3a056745ff52259e46d3d4bbd7ef4a4483fe7cfec75.jpg)

![5cce3cd4a313aaede69423566abcab5bb36534ab10e9e774d823b0883c5d9127.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/5cce3cd4a313aaede69423566abcab5bb36534ab10e9e774d823b0883c5d9127.jpg)

![8359e7b15b950b58b88adeb344ce3f40631b4ce2e7f781f6440acc8288cdd517.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/8359e7b15b950b58b88adeb344ce3f40631b4ce2e7f781f6440acc8288cdd517.jpg)

![89b6fb2898c4531d9bb6bdd9000789d7db3b1656d674b1aa22ea23c22b667216.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/89b6fb2898c4531d9bb6bdd9000789d7db3b1656d674b1aa22ea23c22b667216.jpg)

![a78444a2d0f33d1aa8d5ddb70d27b422cf94304d49260c2679d5ef501e19f82d.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/a78444a2d0f33d1aa8d5ddb70d27b422cf94304d49260c2679d5ef501e19f82d.jpg)

![bb89ce9bec4889c78976ded0bd7e0b87b0584a9221e6c4de4aa5b1459bee5ac6.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/bb89ce9bec4889c78976ded0bd7e0b87b0584a9221e6c4de4aa5b1459bee5ac6.jpg)

![c323c0606cdf943878f659f38777f003afc39219fc7d7bf8b993bcd0821c8661.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/c323c0606cdf943878f659f38777f003afc39219fc7d7bf8b993bcd0821c8661.jpg)

![c97a17ebf911f0e0f47a5e0790e5fecead4ded27983fc64b4d0bd63e1ce7b0cd.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/c97a17ebf911f0e0f47a5e0790e5fecead4ded27983fc64b4d0bd63e1ce7b0cd.jpg)

![c9ff80ea5685e1d62c45f577c5ef37485c0d7ebb529ed9ce69af981932136c24.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/c9ff80ea5685e1d62c45f577c5ef37485c0d7ebb529ed9ce69af981932136c24.jpg)

![f5a91c84fd3231e080d829a559e7a53d1fdaf3cbe37b624de81182a392931e04.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/f5a91c84fd3231e080d829a559e7a53d1fdaf3cbe37b624de81182a392931e04.jpg)

![f6ce639b562496c89d917f9e5d2ec1ec080738a7a621cc668a8a850de274a380.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/images/f6ce639b562496c89d917f9e5d2ec1ec080738a7a621cc668a8a850de274a380.jpg)

### Tables

![17791ac37b566de2659e003c6397517dad349d5703c28d2d708e5b8d03701627.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/17791ac37b566de2659e003c6397517dad349d5703c28d2d708e5b8d03701627.jpg)

![3c65267a1d9223a1b2ef34b191c14ec4199a864d8189d533c000c3399bcfccd2.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/3c65267a1d9223a1b2ef34b191c14ec4199a864d8189d533c000c3399bcfccd2.jpg)

![451cb0ab318eee9083cfe982333a7d206716586406c89900782e130292347723.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/451cb0ab318eee9083cfe982333a7d206716586406c89900782e130292347723.jpg)

![53d0007c0489237e1cfecd825835f8f3884543197f8d10db16e1115ae6f66896.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/53d0007c0489237e1cfecd825835f8f3884543197f8d10db16e1115ae6f66896.jpg)

![57a2516000df90e6dc2844e142a65d9382c9d4353038c68d73d59e3986dad228.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/57a2516000df90e6dc2844e142a65d9382c9d4353038c68d73d59e3986dad228.jpg)

![6ae74729d9af629f13d87153a83ca810c72d3ed33d51f289ee060e080b64bc1f.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/6ae74729d9af629f13d87153a83ca810c72d3ed33d51f289ee060e080b64bc1f.jpg)

![7579014aa5195fba58ab6c9c4f149c745c92810a2bc35b1e42a534f331b82004.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/7579014aa5195fba58ab6c9c4f149c745c92810a2bc35b1e42a534f331b82004.jpg)

![7c5e22c293d07365edeef67884be05e8e40dd4f737b7989c193684d118c4eef4.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/7c5e22c293d07365edeef67884be05e8e40dd4f737b7989c193684d118c4eef4.jpg)

![92ffdd8ebc038255a71589f02f7fff162e319e2014266d3224a84fc982de54a1.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/92ffdd8ebc038255a71589f02f7fff162e319e2014266d3224a84fc982de54a1.jpg)

![aaf791a489d9328b512e4d6d3ff2204093579e5e2042e467a6dd91aaafc485e9.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/aaf791a489d9328b512e4d6d3ff2204093579e5e2042e467a6dd91aaafc485e9.jpg)

![bb53a11bc72f77033a7815b0fa7ab125d4b3535d2353cbad16e453f2a57a4db3.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/bb53a11bc72f77033a7815b0fa7ab125d4b3535d2353cbad16e453f2a57a4db3.jpg)

![c3695fc4881addb443cbd5f86f8ef11925934c1096b1613ea98f2b016099127f.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/c3695fc4881addb443cbd5f86f8ef11925934c1096b1613ea98f2b016099127f.jpg)

![d127abf2af7e24ace491c80a9990ae0a5aff056c2ab331d2cffe1e128f9d3e2a.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/d127abf2af7e24ace491c80a9990ae0a5aff056c2ab331d2cffe1e128f9d3e2a.jpg)

![d9290f27d0642c2be3e800f508cf0b209e2702ef0e5a6595dd504ff07feb584e.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/d9290f27d0642c2be3e800f508cf0b209e2702ef0e5a6595dd504ff07feb584e.jpg)

![e520d5706d903b184e48f29ba322a0ea53f3a499ea49b7f9a7e3ee428c178b48.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/e520d5706d903b184e48f29ba322a0ea53f3a499ea49b7f9a7e3ee428c178b48.jpg)

![f7c12d8302a23f826bbb3a1528acf8698fe850e66b3e750304a117ee38cd2049.jpg](../icml_results/3980_G%24%5E2%24RPO_%20Geometric%20GRPO%20Escaping%20LLMs%E2%80%99%20Reasoning%20Ruts%20to%20Break%20the%20Accuracy%E2%80%93Entropy%20Trade-off/tables/f7c12d8302a23f826bbb3a1528acf8698fe850e66b3e750304a117ee38cd2049.jpg)

<a id="paper-3940"></a>

## HyPOLE: Hyperproperty-Guided Multi-Agent Reinforcement Learning under Partial Observation


### Images

![0704a14d7257229e0900c5061c2b8d0193a30e3ea0085866f0a7aee30d7e092e.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/0704a14d7257229e0900c5061c2b8d0193a30e3ea0085866f0a7aee30d7e092e.jpg)

![3420dad4989157bc810e153a82d4a2ec6453375bc8f353301189a08980742577.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/3420dad4989157bc810e153a82d4a2ec6453375bc8f353301189a08980742577.jpg)

![3bc6881ccd29e43d901537175c93c6179bf76aad02f9a20ba25f7b86998f9f8c.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/3bc6881ccd29e43d901537175c93c6179bf76aad02f9a20ba25f7b86998f9f8c.jpg)

![3bda5a9436a0479b8d4f1d1f170a0874c204dbde0e832b37fc9c65a4e627582f.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/3bda5a9436a0479b8d4f1d1f170a0874c204dbde0e832b37fc9c65a4e627582f.jpg)

![4784b73462982fcb03fbb24a8806c9ef46c0118a332824a4d82075602d9b00f1.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/4784b73462982fcb03fbb24a8806c9ef46c0118a332824a4d82075602d9b00f1.jpg)

![57e6b49a862a136739066ab0102add8a98dcad58c5b4457d63ef2cf732c3c2f6.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/57e6b49a862a136739066ab0102add8a98dcad58c5b4457d63ef2cf732c3c2f6.jpg)

![5dc5118c977b06496c7cc76301bf6ae152f7ba937e78563acf8f4c4c336e0912.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/5dc5118c977b06496c7cc76301bf6ae152f7ba937e78563acf8f4c4c336e0912.jpg)

![6424ffaba3fbc2c56f1cc59fde611763cacc097cdb91fd8fc9d7eeb2133c9d76.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/6424ffaba3fbc2c56f1cc59fde611763cacc097cdb91fd8fc9d7eeb2133c9d76.jpg)

![69f249b564d1f42431306a95ba264b60baf11d36b5ba8bf180b8816501fa3dae.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/69f249b564d1f42431306a95ba264b60baf11d36b5ba8bf180b8816501fa3dae.jpg)

![8b474f52355669a0ab4f4382b161e023600e6c658007f38b592cd6a973e025bc.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/8b474f52355669a0ab4f4382b161e023600e6c658007f38b592cd6a973e025bc.jpg)

![9cea3a5185942999043000e6ab0b3a124a6d67d5ee1f5b3303003e83813ec4f1.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/9cea3a5185942999043000e6ab0b3a124a6d67d5ee1f5b3303003e83813ec4f1.jpg)

![9f8eb66d7aab824ada782c9ae578921dd27aea16e399ddf10d2a444c294be4b3.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/9f8eb66d7aab824ada782c9ae578921dd27aea16e399ddf10d2a444c294be4b3.jpg)

![b9bc2dc54945f98fd41208b6b66490f88faee8cf403ac24a995821473fc37137.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/b9bc2dc54945f98fd41208b6b66490f88faee8cf403ac24a995821473fc37137.jpg)

![c313d334c5b98e301732349f64358e1dbef2bba23c6cbaf02b3f5bf7e75621bf.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/c313d334c5b98e301732349f64358e1dbef2bba23c6cbaf02b3f5bf7e75621bf.jpg)

![d2283ddab6b5b878b1d0079698ed2dcac4199acc6927ca0f60e8f84f74dc7dc2.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/d2283ddab6b5b878b1d0079698ed2dcac4199acc6927ca0f60e8f84f74dc7dc2.jpg)

![dcca193bf99f0ac3835c0a69f6d23ed03c5cd48e790b3ffc6d37b949c90f5bb4.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/dcca193bf99f0ac3835c0a69f6d23ed03c5cd48e790b3ffc6d37b949c90f5bb4.jpg)

![e86e6bb618ca12d9a4e699641c4d231ebf69b5de52bea3140d10b5946bb57a56.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/e86e6bb618ca12d9a4e699641c4d231ebf69b5de52bea3140d10b5946bb57a56.jpg)

![f5cb63818acad8c853389d6398ee451f699d28e4f7bb4ee0b467544a3170e2d1.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/f5cb63818acad8c853389d6398ee451f699d28e4f7bb4ee0b467544a3170e2d1.jpg)

![fccb1bc22145874adb89a802708e1f110630d206cf3f84815e940c9641562d87.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/images/fccb1bc22145874adb89a802708e1f110630d206cf3f84815e940c9641562d87.jpg)

### Tables

![571b10e49116a7a23fe3c13f43f9152c1a05c43699b9fcf20a3bffac3888c4e8.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/tables/571b10e49116a7a23fe3c13f43f9152c1a05c43699b9fcf20a3bffac3888c4e8.jpg)

![9af0f53a86c66ea585eb96c3495a2f535f4420fa1b7c7ffb4accbc89aa9b9e15.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/tables/9af0f53a86c66ea585eb96c3495a2f535f4420fa1b7c7ffb4accbc89aa9b9e15.jpg)

![cc1323962f6376a7b417182f19e225947ea05bb5f62811a658b6fd2d352481ec.jpg](../icml_results/3981_HyPOLE_%20Hyperproperty-Guided%20Multi-Agent%20Reinforcement%20Learning%20under%20Partial%20Observation/tables/cc1323962f6376a7b417182f19e225947ea05bb5f62811a658b6fd2d352481ec.jpg)

<a id="paper-3941"></a>

## Causal Identification from Counterfactual Data: Completeness and Bounding Results


### Images

![0309700b265a5f1161f7af4ed55273a5bb192d93e53d071a156195adf21345d2.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/0309700b265a5f1161f7af4ed55273a5bb192d93e53d071a156195adf21345d2.jpg)

![1289fe3f3e3f3df078d2f47153dd5e26359f66e6d78c73fb2885d26b3dd291e7.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/1289fe3f3e3f3df078d2f47153dd5e26359f66e6d78c73fb2885d26b3dd291e7.jpg)

![138e5bd164e4e7c4f0bc7acbd49a064aa5f5b0a0a6d8a27d7ae15d5d444d0142.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/138e5bd164e4e7c4f0bc7acbd49a064aa5f5b0a0a6d8a27d7ae15d5d444d0142.jpg)

![1a7a604c46682846c4f2a66a62942dfc15859cf68ea88a6442d0475820884dd5.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/1a7a604c46682846c4f2a66a62942dfc15859cf68ea88a6442d0475820884dd5.jpg)

![2c270fcde74f93c286dde74e1f8ea143b09da8af9a5c59e3e6f3cc0a58aa9a83.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/2c270fcde74f93c286dde74e1f8ea143b09da8af9a5c59e3e6f3cc0a58aa9a83.jpg)

![421820849e91c163b2bab4c026a62dd8ae21ad80090e978de02171369337e491.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/421820849e91c163b2bab4c026a62dd8ae21ad80090e978de02171369337e491.jpg)

![582ee0c5a753613db4e14d5df63e0120b21b673de40581c8d871857880cd7eb6.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/582ee0c5a753613db4e14d5df63e0120b21b673de40581c8d871857880cd7eb6.jpg)

![6ae06b9046dce923e4df76696a734c1c6ad6be73d7ea5dc47a7e4d154e9bde9a.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/6ae06b9046dce923e4df76696a734c1c6ad6be73d7ea5dc47a7e4d154e9bde9a.jpg)

![80e184ae5dc6eef70e52cf2179ba793c61ccc131f28812760f46c2e6d32d8a3b.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/80e184ae5dc6eef70e52cf2179ba793c61ccc131f28812760f46c2e6d32d8a3b.jpg)

![82ea37c51ed71f917ec1c2188f8f70048b50d1ab95fc25ba11ada6780f39c1de.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/82ea37c51ed71f917ec1c2188f8f70048b50d1ab95fc25ba11ada6780f39c1de.jpg)

![93d96c7ff160f3fde465dcc49d5238b9f633e83562013de6afc92217aa171bbf.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/93d96c7ff160f3fde465dcc49d5238b9f633e83562013de6afc92217aa171bbf.jpg)

![9dda0c6a9e0b68e6c969f2fb4d736ecea395047e0e88f57298f2b261c3bed844.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/9dda0c6a9e0b68e6c969f2fb4d736ecea395047e0e88f57298f2b261c3bed844.jpg)

![a02b64f0454c99c70fb44fabbf4cf640453a442e2c0f8b777d96eac83901844b.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/a02b64f0454c99c70fb44fabbf4cf640453a442e2c0f8b777d96eac83901844b.jpg)

![a5bbf0cd0bd4ae0c0e42223d0f630031979d67d8359fa882db9eeaeb7df541a4.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/a5bbf0cd0bd4ae0c0e42223d0f630031979d67d8359fa882db9eeaeb7df541a4.jpg)

![a5f8708c6affefb8c7596ff0b1d91faa4b66bf9bb75f102caf636aa27d541ac1.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/a5f8708c6affefb8c7596ff0b1d91faa4b66bf9bb75f102caf636aa27d541ac1.jpg)

![a999874770df363349cdb6edff5ae4b54c2af35db05aaad6adcb19ad15b7a80d.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/a999874770df363349cdb6edff5ae4b54c2af35db05aaad6adcb19ad15b7a80d.jpg)

![c00f9897dae091205b101e0e731f8ad8edfcb5aacc4c004d3f7d8cbbc1a9a9c6.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/c00f9897dae091205b101e0e731f8ad8edfcb5aacc4c004d3f7d8cbbc1a9a9c6.jpg)

![c560eeb3c6b68751b82cb20b0fcdf5b50f0b560e19d158cb1fff572ad1d4d030.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/c560eeb3c6b68751b82cb20b0fcdf5b50f0b560e19d158cb1fff572ad1d4d030.jpg)

![d2fbff9bb232b09478b2cefc670b0fde4f77c5d32e1a27b07709805b8e1b1288.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/d2fbff9bb232b09478b2cefc670b0fde4f77c5d32e1a27b07709805b8e1b1288.jpg)

![db817beea03e574c94d8c5ece424cf00d5c07bf470869a33d2ad9f05813cbca6.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/db817beea03e574c94d8c5ece424cf00d5c07bf470869a33d2ad9f05813cbca6.jpg)

![e5324fa056128f7ac3f53a9d8ff568fbfa33787e213b2cb23e25cc127dca11c6.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/e5324fa056128f7ac3f53a9d8ff568fbfa33787e213b2cb23e25cc127dca11c6.jpg)

![ebf7643ee7c3fc3386b08278e42b858d8a431d9e3abdc89898d2dbfdd0d5cc9d.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/ebf7643ee7c3fc3386b08278e42b858d8a431d9e3abdc89898d2dbfdd0d5cc9d.jpg)

![f7bf1c7365f35822f31d4057859aa5fbfeda01ad709a07edb83337244b866bc5.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/f7bf1c7365f35822f31d4057859aa5fbfeda01ad709a07edb83337244b866bc5.jpg)

![f7ed24b04ba8aeba60fdad4d8317ff3ca5eceae543a33c505bd7866dbf88c31b.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/f7ed24b04ba8aeba60fdad4d8317ff3ca5eceae543a33c505bd7866dbf88c31b.jpg)

![f8cdb49541233ff7adb65b73cbede4a679cff566adfb898c4f68b778be5c13a1.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/images/f8cdb49541233ff7adb65b73cbede4a679cff566adfb898c4f68b778be5c13a1.jpg)

### Tables

![04f069fe3383e76c96eef76ffdc8bd842197ec35d645f7353e77373c3744f55c.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/tables/04f069fe3383e76c96eef76ffdc8bd842197ec35d645f7353e77373c3744f55c.jpg)

![bc19592cb1fa6f722fcbc8ba96d1db33c5bfef3e7b28abea62a8198755e26df9.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/tables/bc19592cb1fa6f722fcbc8ba96d1db33c5bfef3e7b28abea62a8198755e26df9.jpg)

![c16b2f31bc68aa3ee94474ef502eb7c9880cb879834adb1d606df3760cc228f9.jpg](../icml_results/3982_Causal%20Identification%20from%20Counterfactual%20Data_%20Completeness%20and%20Bounding%20Results/tables/c16b2f31bc68aa3ee94474ef502eb7c9880cb879834adb1d606df3760cc228f9.jpg)

<a id="paper-3942"></a>

## Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models


### Images

![05e2fbb33f2c8d6f63cce9ddc9b942be97bce91de076f70eb0df1b361f0f239e.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/05e2fbb33f2c8d6f63cce9ddc9b942be97bce91de076f70eb0df1b361f0f239e.jpg)

![259074b8286911ff673958bec6043f2432a4bcc18c0602ec26147e449d89a755.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/259074b8286911ff673958bec6043f2432a4bcc18c0602ec26147e449d89a755.jpg)

![43b94d933051b160007d4e15c7886a6ea2227e80abc3302f21ab90b73427a6d2.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/43b94d933051b160007d4e15c7886a6ea2227e80abc3302f21ab90b73427a6d2.jpg)

![978c4de1bd2ea424c1758c7f7fb9faf80411913261a46855746437fb4a9ba507.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/978c4de1bd2ea424c1758c7f7fb9faf80411913261a46855746437fb4a9ba507.jpg)

![a99b28b82a13409a62d283975ea85f9d75543d32370585198fb62c17220be5db.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/a99b28b82a13409a62d283975ea85f9d75543d32370585198fb62c17220be5db.jpg)

![f1c376e9d1db9f73424b0c5c263ef990df56e7cfdf7e376c0bfaf3651cb77774.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/f1c376e9d1db9f73424b0c5c263ef990df56e7cfdf7e376c0bfaf3651cb77774.jpg)

![f5c5fc6cc2c585d774e45271aa31f49929a4f6323da786b1f853588ffe08f1b8.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/images/f5c5fc6cc2c585d774e45271aa31f49929a4f6323da786b1f853588ffe08f1b8.jpg)

### Tables

![08242383668d50e380cab0a609a673819133c86a038ea80338ae8695ddff9c90.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/08242383668d50e380cab0a609a673819133c86a038ea80338ae8695ddff9c90.jpg)

![0d5728249d6a81ae6f110fc03553906ce97cd6f1f42f2645c6aff7c34a53834e.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/0d5728249d6a81ae6f110fc03553906ce97cd6f1f42f2645c6aff7c34a53834e.jpg)

![11b688969dd2ed6c196ef490b9309bfe1101cc68bfe3edb73b6c7e2123555325.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/11b688969dd2ed6c196ef490b9309bfe1101cc68bfe3edb73b6c7e2123555325.jpg)

![29a85c732bffb80bd6e102f4e776a61a1a43ac84c3e826ea5aede9934b498d38.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/29a85c732bffb80bd6e102f4e776a61a1a43ac84c3e826ea5aede9934b498d38.jpg)

![4ea1b63b0cd09ce487455adde97ef81d046a93691f1e4955dcfe51172843627c.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/4ea1b63b0cd09ce487455adde97ef81d046a93691f1e4955dcfe51172843627c.jpg)

![511579b216f190f0e33ed8b59070aba17ba2cf992e19f48e091b65b39b708e10.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/511579b216f190f0e33ed8b59070aba17ba2cf992e19f48e091b65b39b708e10.jpg)

![536a776e1d9875806f962f2f5038eff79066b68aa2d78eae936e30e9f02c6b50.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/536a776e1d9875806f962f2f5038eff79066b68aa2d78eae936e30e9f02c6b50.jpg)

![5af8b4ba7c56be16dd4b2d8f242109724b55854e4460d5484df5b82d1d49877f.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/5af8b4ba7c56be16dd4b2d8f242109724b55854e4460d5484df5b82d1d49877f.jpg)

![635d98915ab3a2e221e4c70f3cf1240f0ba43aadd168b23c6207ba61b5a9e542.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/635d98915ab3a2e221e4c70f3cf1240f0ba43aadd168b23c6207ba61b5a9e542.jpg)

![65d8f16e5d63407673de801630dc746cc94dba505ec91ecd4c371d5c8908a528.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/65d8f16e5d63407673de801630dc746cc94dba505ec91ecd4c371d5c8908a528.jpg)

![6883c0c6535e605d7a051ea2abb4dbe93f757a6b64bfb51d89ad11558ec70c9a.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/6883c0c6535e605d7a051ea2abb4dbe93f757a6b64bfb51d89ad11558ec70c9a.jpg)

![73b7bfc7377c9b9b79183ff6744d4e8bb7bee114cbcda18f18709dc296f61da0.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/73b7bfc7377c9b9b79183ff6744d4e8bb7bee114cbcda18f18709dc296f61da0.jpg)

![7fcab49392a7e9b4b22f6483ca5cb0781473a4842a95801487c90d0e35f28a1c.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/7fcab49392a7e9b4b22f6483ca5cb0781473a4842a95801487c90d0e35f28a1c.jpg)

![814d159219a8015b898601b962fc49eeeb3b4b7b1abba0e7445f4b3743cea9a7.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/814d159219a8015b898601b962fc49eeeb3b4b7b1abba0e7445f4b3743cea9a7.jpg)

![89cb79f563d00b5c66e3943314df1190f3b56bd2ee33703569e6659bd8620c08.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/89cb79f563d00b5c66e3943314df1190f3b56bd2ee33703569e6659bd8620c08.jpg)

![9003500711d1418f62d05b1ce49909c9bb68f637c53869a448425d0e85619e71.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/9003500711d1418f62d05b1ce49909c9bb68f637c53869a448425d0e85619e71.jpg)

![9200c7dee809a2e2419b6673d767447c8696d42e60c16ef640e50233d7c921b0.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/9200c7dee809a2e2419b6673d767447c8696d42e60c16ef640e50233d7c921b0.jpg)

![a03b4c14fb88d90e155a5f70f42b44d7a8859d290d4951bda8651397ff061141.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/a03b4c14fb88d90e155a5f70f42b44d7a8859d290d4951bda8651397ff061141.jpg)

![c71592ef91a5c67fcf1349affc6d7471d413883c6a54c2c6c13ea8785e6f6468.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/c71592ef91a5c67fcf1349affc6d7471d413883c6a54c2c6c13ea8785e6f6468.jpg)

![c9c64c9f8d75fbd6352b950e1d53c11c334afb193cf8ad8aa79f4ab65cfad81e.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/c9c64c9f8d75fbd6352b950e1d53c11c334afb193cf8ad8aa79f4ab65cfad81e.jpg)

![d5085746dbb1eff78f87344a654c431b2d7bc9021ec7743abb26f86638a2d3f9.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/d5085746dbb1eff78f87344a654c431b2d7bc9021ec7743abb26f86638a2d3f9.jpg)

![e1f59a593fbc0de63e1315afe65926e071bd89806a3a1ef158410adbbaa3dd9c.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/e1f59a593fbc0de63e1315afe65926e071bd89806a3a1ef158410adbbaa3dd9c.jpg)

![f395cbe2266dba074b6fafd83bfd2ae3f1ee65348f4741a69ccea038fff03ea1.jpg](../icml_results/3983_Learning%20Long%20Range%20Spatio-Temporal%20Representations%20over%20Continuous%20Time%20Dynamic%20Graphs%20with%20State%20S/tables/f395cbe2266dba074b6fafd83bfd2ae3f1ee65348f4741a69ccea038fff03ea1.jpg)

<a id="paper-3943"></a>

## ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition


### Images

![1ca7602fc102491461bef161ad9cad18aa231b35f5bd8af05639d71b2d4c2e16.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/1ca7602fc102491461bef161ad9cad18aa231b35f5bd8af05639d71b2d4c2e16.jpg)

![20a94e29ca6ae3a41d071484d88ae60f891fbf0362f4a947de30d2ec3c03174d.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/20a94e29ca6ae3a41d071484d88ae60f891fbf0362f4a947de30d2ec3c03174d.jpg)

![28c232997fe1b8623f7b9b105fb380a050de023f099a040dcb2a03935400ac74.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/28c232997fe1b8623f7b9b105fb380a050de023f099a040dcb2a03935400ac74.jpg)

![529f20e2d56f5932e829991c8170e858935f69ec6860a21b23208c672945ca80.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/529f20e2d56f5932e829991c8170e858935f69ec6860a21b23208c672945ca80.jpg)

![8307b05d98a16881b00af25e86fba06763b0ea4e070962687d5f57ad3747b456.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/8307b05d98a16881b00af25e86fba06763b0ea4e070962687d5f57ad3747b456.jpg)

![9bd624ec9b7e78c0ad0286c29ef39c1b2700f10ba0d577389ed794bb805cf583.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/9bd624ec9b7e78c0ad0286c29ef39c1b2700f10ba0d577389ed794bb805cf583.jpg)

![a7e659b7ce45bc761c2524e420ab85120e9a19d1cb68335759824519e6d5e1c9.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/a7e659b7ce45bc761c2524e420ab85120e9a19d1cb68335759824519e6d5e1c9.jpg)

![baf7032f87f18ceea5272fe51909cfa82702e99a44a9af12a7ee793480c31aa3.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/baf7032f87f18ceea5272fe51909cfa82702e99a44a9af12a7ee793480c31aa3.jpg)

![c3667ac8fb6e02f6dc6f6310672f8f9c399433dd44dfe472391a9dd47cdc864c.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/c3667ac8fb6e02f6dc6f6310672f8f9c399433dd44dfe472391a9dd47cdc864c.jpg)

![dc95bf24ddd7dd9298f8f092ff1d27ded1a069da163dbee1e90a53b86f2ef6b4.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/dc95bf24ddd7dd9298f8f092ff1d27ded1a069da163dbee1e90a53b86f2ef6b4.jpg)

![e0f439083013e365c790c61fcc908d77586ad206fbf8e98f7cf81d53c61dfbce.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/e0f439083013e365c790c61fcc908d77586ad206fbf8e98f7cf81d53c61dfbce.jpg)

![e3eb9149fcf513a8c89eddb01c9ea2241a63ca7d3c6cedb2352be2b39f88118f.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/e3eb9149fcf513a8c89eddb01c9ea2241a63ca7d3c6cedb2352be2b39f88118f.jpg)

![e4964abcb7860de50b78ad6c7a9af778ba7402acf722dc04fc0b77ca9c77e76f.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/e4964abcb7860de50b78ad6c7a9af778ba7402acf722dc04fc0b77ca9c77e76f.jpg)

![fe677bf5a05737e25dd5a992f6611475f9412eda3b2b5d61b2e0507f9e196e86.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/fe677bf5a05737e25dd5a992f6611475f9412eda3b2b5d61b2e0507f9e196e86.jpg)

![fef7169a3d7b0c0b30398235d64700925415a287bf314872add58e5f60be0719.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/images/fef7169a3d7b0c0b30398235d64700925415a287bf314872add58e5f60be0719.jpg)

### Tables

![16a2ec33dccb155979e27c614e3360aa2ad0d36997f692633e975aadeb498f57.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/16a2ec33dccb155979e27c614e3360aa2ad0d36997f692633e975aadeb498f57.jpg)

![187261bacd0bfd4294d88a1c18122a358e6adba9d9e68819d50a1bd257c207cc.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/187261bacd0bfd4294d88a1c18122a358e6adba9d9e68819d50a1bd257c207cc.jpg)

![2035e5c3e25a0398200d35a5dcb3ec34030aecd6ceed2ab3dedfc97566bfd814.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/2035e5c3e25a0398200d35a5dcb3ec34030aecd6ceed2ab3dedfc97566bfd814.jpg)

![2839cc7344c9cf6432c96d7c4f2611d96e1a069894c6f0a24a04f4ba3d0a500d.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/2839cc7344c9cf6432c96d7c4f2611d96e1a069894c6f0a24a04f4ba3d0a500d.jpg)

![341a2f65292d09ff007eb8b8e77a248ef41e07ffccfcdd661d8c016507fce6e0.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/341a2f65292d09ff007eb8b8e77a248ef41e07ffccfcdd661d8c016507fce6e0.jpg)

![379a2df218cf7dad62b05c74d9fcb2616c457962911109264b92937b85960aa3.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/379a2df218cf7dad62b05c74d9fcb2616c457962911109264b92937b85960aa3.jpg)

![3aa69a2cdbd76008a1c88203d1855faf6228f3ebb88e9d9c282a956ed2508139.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/3aa69a2cdbd76008a1c88203d1855faf6228f3ebb88e9d9c282a956ed2508139.jpg)

![7183c0b1b340afa9724fd64973b0cad86e764109c1f337931cd1fab8d85722dc.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/7183c0b1b340afa9724fd64973b0cad86e764109c1f337931cd1fab8d85722dc.jpg)

![74458d2f2b0efecfa603f0c4046773a781c56331b6340b3e1a0a010a40b92ee6.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/74458d2f2b0efecfa603f0c4046773a781c56331b6340b3e1a0a010a40b92ee6.jpg)

![75a35ddc290975c9806032c758f76130268eb281173c2507e0ccd71fe7937859.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/75a35ddc290975c9806032c758f76130268eb281173c2507e0ccd71fe7937859.jpg)

![7645f95b8818c756a19c368308aad2d4fd03310034659299a8bc00afcc06b5ed.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/7645f95b8818c756a19c368308aad2d4fd03310034659299a8bc00afcc06b5ed.jpg)

![7ca6f2ddbfa9d2d60ab729af52b782d0212851889d2edd0122ec4d9ea94d6219.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/7ca6f2ddbfa9d2d60ab729af52b782d0212851889d2edd0122ec4d9ea94d6219.jpg)

![7dc4b75587a7a699bf4c3c0f8d780e7bc3c466c51dfab290cf94b3bdab9e798c.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/7dc4b75587a7a699bf4c3c0f8d780e7bc3c466c51dfab290cf94b3bdab9e798c.jpg)

![82fa551117de94aa464ab8151749726ee0df4bdfdcb0314a1afb3b7ad39eacbe.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/82fa551117de94aa464ab8151749726ee0df4bdfdcb0314a1afb3b7ad39eacbe.jpg)

![920f500903a24843aae196455eb46ddcbd8c3782c4f2480e1769bbf89869270c.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/920f500903a24843aae196455eb46ddcbd8c3782c4f2480e1769bbf89869270c.jpg)

![fa17d39fd4d995f284cccf074357cddd6923974d15c8041f33018f72bf0496cc.jpg](../icml_results/3984_ScoreMix_%20Synthetic%20Data%20Generation%20by%20Score%20Composition%20in%20Diffusion%20Models%20Improves%20Recognition/tables/fa17d39fd4d995f284cccf074357cddd6923974d15c8041f33018f72bf0496cc.jpg)

<a id="paper-3944"></a>

## VeriSimpl: Robust Optimization Modeling from Natural Language using Simplification-based Verification


### Images

![ad0b994c1b88bb309f75de4c7b23d149be9927a8e6482f1d5da6f8739f420f13.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/images/ad0b994c1b88bb309f75de4c7b23d149be9927a8e6482f1d5da6f8739f420f13.jpg)

![bad9d439d2a000c383077b3f4bc81ddaec7c6f286a0293b0ecf8e6d067e6ce3a.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/images/bad9d439d2a000c383077b3f4bc81ddaec7c6f286a0293b0ecf8e6d067e6ce3a.jpg)

![f90950ec2a90b4ca6ad9bad55f3e9869d4cd6da2c4bd4866ce55d7632f327327.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/images/f90950ec2a90b4ca6ad9bad55f3e9869d4cd6da2c4bd4866ce55d7632f327327.jpg)

### Tables

![19c903267d71fda5aff1c7e8bf114af4acc37573ff468040d11b9a536979d8c2.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/19c903267d71fda5aff1c7e8bf114af4acc37573ff468040d11b9a536979d8c2.jpg)

![1b33ef22da6124896e8f0c672aaadf4ad0557b0685bc8fb4088bfbaf298f28ff.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/1b33ef22da6124896e8f0c672aaadf4ad0557b0685bc8fb4088bfbaf298f28ff.jpg)

![3d857ed189ca2a24dfcf8be6afc18c3d3794d3a35d9e56556a93f03c49847e4a.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/3d857ed189ca2a24dfcf8be6afc18c3d3794d3a35d9e56556a93f03c49847e4a.jpg)

![48d65ec73bd491d345675aeceee5b0fe9b85ceb88411654f82089ba64478a9b4.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/48d65ec73bd491d345675aeceee5b0fe9b85ceb88411654f82089ba64478a9b4.jpg)

![492a837d972ed732b5ef6978df27406f8e30a0b77de9ce9267aad2bef195eb1d.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/492a837d972ed732b5ef6978df27406f8e30a0b77de9ce9267aad2bef195eb1d.jpg)

![528bc10c9bb830f9f2b3f6538238182f4a65e22cae845addc1c1c4685c8b896c.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/528bc10c9bb830f9f2b3f6538238182f4a65e22cae845addc1c1c4685c8b896c.jpg)

![5a8e47af3f194321d984aa9ac758f67153db76dec331db6ffb7dcbaf2859fecd.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/5a8e47af3f194321d984aa9ac758f67153db76dec331db6ffb7dcbaf2859fecd.jpg)

![7cce3cdd94770edcb84c6b0f7bbd76ab1808d7fed32f510fe051f21736abd79c.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/7cce3cdd94770edcb84c6b0f7bbd76ab1808d7fed32f510fe051f21736abd79c.jpg)

![8f1b4e28f077d15723e1e751592f2437af84b201113f441b84627604591caef7.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/8f1b4e28f077d15723e1e751592f2437af84b201113f441b84627604591caef7.jpg)

![a0690cd14fc0cf741b61a5884512b7e006585d683553469449e70cf5fbad5c46.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/a0690cd14fc0cf741b61a5884512b7e006585d683553469449e70cf5fbad5c46.jpg)

![a2cf6a60d9c49d18cc7832b259c612dd51c86e571df9d53fdc500033f32830f0.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/a2cf6a60d9c49d18cc7832b259c612dd51c86e571df9d53fdc500033f32830f0.jpg)

![b45789f0a716de3490016e562f6dcd2e5cb54aff2522319f4fc89ab834ac7ba9.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/b45789f0a716de3490016e562f6dcd2e5cb54aff2522319f4fc89ab834ac7ba9.jpg)

![bed018878c1bd76bb0faf97b4df295ebe6c598232ae324eac64a6fa590ef5a5f.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/bed018878c1bd76bb0faf97b4df295ebe6c598232ae324eac64a6fa590ef5a5f.jpg)

![bfc13f7f44a6263ea2bcc560b83e0eba786ddf0ddf2f39b307b02cf63f4a1bbb.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/bfc13f7f44a6263ea2bcc560b83e0eba786ddf0ddf2f39b307b02cf63f4a1bbb.jpg)

![c3caeee4f25b3dc0f43257c9769d8b56baaad87f4351efff2dc627c828febc26.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/c3caeee4f25b3dc0f43257c9769d8b56baaad87f4351efff2dc627c828febc26.jpg)

![e16534e74f7a662f8e622f73157ebbaa1755e85a327de9e572650fc0587a6411.jpg](../icml_results/3985_VeriSimpl_%20Robust%20Optimization%20Modeling%20from%20Natural%20Language%20using%20Simplification-based%20Verificatio/tables/e16534e74f7a662f8e622f73157ebbaa1755e85a327de9e572650fc0587a6411.jpg)

<a id="paper-3945"></a>

## STT-LLM: Structural-Temporal Tokenization for Adapting LLMs to Longitudinal Clinical Profiles


### Images

![49e019e9604a04bd61f84a32890957904fd9378c31000815cd1d874610ca1229.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/49e019e9604a04bd61f84a32890957904fd9378c31000815cd1d874610ca1229.jpg)

![4b276b87f144944374fe63f43427ef884e1a3749b877a769573e89fa187ad0b8.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/4b276b87f144944374fe63f43427ef884e1a3749b877a769573e89fa187ad0b8.jpg)

![4fa29d27179781b187f4dd3a09e3183f98c99959abee13f7996b9a444446fa04.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/4fa29d27179781b187f4dd3a09e3183f98c99959abee13f7996b9a444446fa04.jpg)

![56da0ddf44641535706e3449badef85bc00ded90cc14527eb050eed3343908fa.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/56da0ddf44641535706e3449badef85bc00ded90cc14527eb050eed3343908fa.jpg)

![6c1c2fc91e45b5e7c34622af0d05ffbe1a9d4ffffe17ff6385b31f180ea4ab15.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/6c1c2fc91e45b5e7c34622af0d05ffbe1a9d4ffffe17ff6385b31f180ea4ab15.jpg)

![8339ef056ce16928a0759bf9999464b16816aecebefef5721738f402c8d3b8a5.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/8339ef056ce16928a0759bf9999464b16816aecebefef5721738f402c8d3b8a5.jpg)

![871bdfa1d13ff773c36767ab32010faef1e365a06a21c5eb22927b40d2462487.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/871bdfa1d13ff773c36767ab32010faef1e365a06a21c5eb22927b40d2462487.jpg)

![8f0cdfc5524030323cb22b4f1f777350343c5a1cd0177807fd5323c2800f6c8a.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/8f0cdfc5524030323cb22b4f1f777350343c5a1cd0177807fd5323c2800f6c8a.jpg)

![a61a34c0dbe932ae1f32cabaeeb794bbca774ea8ebcbfdd946eb0eacb326efe0.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/a61a34c0dbe932ae1f32cabaeeb794bbca774ea8ebcbfdd946eb0eacb326efe0.jpg)

![be37ec1a92eafbf64210598575996137b7efa61ea656b9bc13161e1badeefb82.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/be37ec1a92eafbf64210598575996137b7efa61ea656b9bc13161e1badeefb82.jpg)

![c3535742f9428af415c0371ce4463f0b584149337c70b921dc3bc0c79e4adf8c.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/c3535742f9428af415c0371ce4463f0b584149337c70b921dc3bc0c79e4adf8c.jpg)

![c4d74234c47c8fc223e7d499ce0fea97ee6244c515691ca4bf248b9935d1a607.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/c4d74234c47c8fc223e7d499ce0fea97ee6244c515691ca4bf248b9935d1a607.jpg)

![e8e48e832e9d596be15abfb332c99c22e865558215f1ba5a86b2fd6c516c3233.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/images/e8e48e832e9d596be15abfb332c99c22e865558215f1ba5a86b2fd6c516c3233.jpg)

### Tables

![00089f6278f3dcfbdfea1c737eb09f9b648ed2d515e5e0e44c36fbc3890d8ff3.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/00089f6278f3dcfbdfea1c737eb09f9b648ed2d515e5e0e44c36fbc3890d8ff3.jpg)

![147d80e3afde29ad9afd69b6085a1dab851366765a5c5f83af05e76a44b39724.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/147d80e3afde29ad9afd69b6085a1dab851366765a5c5f83af05e76a44b39724.jpg)

![1743142fc35d4609c96a02f1d923d50331fdb12bc109604504cf705824517190.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/1743142fc35d4609c96a02f1d923d50331fdb12bc109604504cf705824517190.jpg)

![7b3e33b7bc8874cfbcf58cc677c28d11f63dea1fb6b10beac309c934de524f3e.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/7b3e33b7bc8874cfbcf58cc677c28d11f63dea1fb6b10beac309c934de524f3e.jpg)

![985ac624d153d8e25aad24e141a635d60b8012e94ad3d811969f2c94cb3346c0.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/985ac624d153d8e25aad24e141a635d60b8012e94ad3d811969f2c94cb3346c0.jpg)

![b13ecdcc1649a168971cab75e688ee43fec311561ea61da92c0f856743f26846.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/b13ecdcc1649a168971cab75e688ee43fec311561ea61da92c0f856743f26846.jpg)

![b1dc6f5301c576f5b031be95c36f76cde5d867b1ac6dc547e8eac8b2c1144a62.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/b1dc6f5301c576f5b031be95c36f76cde5d867b1ac6dc547e8eac8b2c1144a62.jpg)

![bb8f0ae00df8c4501b63cb3d781cfe9cdc222a6700cc02859dc773b13cae8371.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/bb8f0ae00df8c4501b63cb3d781cfe9cdc222a6700cc02859dc773b13cae8371.jpg)

![cf2312d4754e97066958d0de93c709cb0a28d67f78e5b5357dd9fe95a347bce4.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/cf2312d4754e97066958d0de93c709cb0a28d67f78e5b5357dd9fe95a347bce4.jpg)

![dc1e90f63ee0aa03b23278e77a39a9e55517dae15e73bb2aea96fab5eb5f81bd.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/dc1e90f63ee0aa03b23278e77a39a9e55517dae15e73bb2aea96fab5eb5f81bd.jpg)

![e325f16b2b321e06157db79fdac2bd064322d6505f4f686fc2a11111be7d59f8.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/e325f16b2b321e06157db79fdac2bd064322d6505f4f686fc2a11111be7d59f8.jpg)

![f3ccc42ebc10899245cbaecc72b3c7833548377da1aa1cc9ef4fab333e24bc76.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/f3ccc42ebc10899245cbaecc72b3c7833548377da1aa1cc9ef4fab333e24bc76.jpg)

![f9d5e471f8fd7574747c2f67094c3370c0ec3a4d59b09e3020075b8ab2e75460.jpg](../icml_results/3986_STT-LLM_%20Structural-Temporal%20Tokenization%20for%20Adapting%20LLMs%20to%20Longitudinal%20Clinical%20Profiles/tables/f9d5e471f8fd7574747c2f67094c3370c0ec3a4d59b09e3020075b8ab2e75460.jpg)

<a id="paper-3946"></a>

## Accuracy-First Rényi Differential Privacy and Post-Processing Immunity


### Images

![371eda3bfc83a857d57fc45123a6c2568dad99fda5f366d27bdc186313691920.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/images/371eda3bfc83a857d57fc45123a6c2568dad99fda5f366d27bdc186313691920.jpg)

![58064e7b88656fb43b81dbb5202846fafe4ce2f6dc4fdb6875c5524ce7842a08.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/images/58064e7b88656fb43b81dbb5202846fafe4ce2f6dc4fdb6875c5524ce7842a08.jpg)

![589c04c4b33723385c755ab8e73fda0e319933c186ad0a366e3b47f19e6a0034.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/images/589c04c4b33723385c755ab8e73fda0e319933c186ad0a366e3b47f19e6a0034.jpg)

![75725c88943ec1b72bbb3ea29fefa6f73e7e278b6050cf1853eeda46274ac4cf.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/images/75725c88943ec1b72bbb3ea29fefa6f73e7e278b6050cf1853eeda46274ac4cf.jpg)

![7e2d497e40b3ee92077487473b49a39d3fb1e87473569e21f974df11f425487f.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/images/7e2d497e40b3ee92077487473b49a39d3fb1e87473569e21f974df11f425487f.jpg)

![f91cd2619e3fb31a7d4bfc38dc03809cafcaea56598b985a7e55e022e67e30d3.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/images/f91cd2619e3fb31a7d4bfc38dc03809cafcaea56598b985a7e55e022e67e30d3.jpg)

### Tables

![4fee18b1316ae8ad76c8bb66f577f590e5d616b454af2910cc9664fea325e26d.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/tables/4fee18b1316ae8ad76c8bb66f577f590e5d616b454af2910cc9664fea325e26d.jpg)

![63eb139b368f7539f24d30260454e89e10cd0ddb1f3ac51f87e6f27d0d591861.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/tables/63eb139b368f7539f24d30260454e89e10cd0ddb1f3ac51f87e6f27d0d591861.jpg)

![7358403f1339cead0997f05cb985ea475482aad54d91974aa67ec09a57d45f4b.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/tables/7358403f1339cead0997f05cb985ea475482aad54d91974aa67ec09a57d45f4b.jpg)

![d627ed35b78cbaa61d699329df172111412f1b4731f18c1ef259ca91b6e1bcd2.jpg](../icml_results/3987_Accuracy-First%20R%C3%A9nyi%20Differential%20Privacy%20and%20Post-Processing%20Immunity/tables/d627ed35b78cbaa61d699329df172111412f1b4731f18c1ef259ca91b6e1bcd2.jpg)

<a id="paper-3947"></a>

## Revisiting OOD Generalization in Programmatic RL


### Images

![0bda9c2c2c29a6b9af0c2574b266d1032c7884fc89a610d2c9327b937627ec9e.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/images/0bda9c2c2c29a6b9af0c2574b266d1032c7884fc89a610d2c9327b937627ec9e.jpg)

![207fb192a04d3595b5cb0b123c21044f3b02d5a4620642f4b1c397f13825e34f.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/images/207fb192a04d3595b5cb0b123c21044f3b02d5a4620642f4b1c397f13825e34f.jpg)

![40ca4f8501351dcda45b6e5c86da502adf4cce06ef6c6c4879e784e41a8fece3.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/images/40ca4f8501351dcda45b6e5c86da502adf4cce06ef6c6c4879e784e41a8fece3.jpg)

![54e4549df2e38e61bba600eb15687bf912fefeab7aef2784b4f601afef3044ae.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/images/54e4549df2e38e61bba600eb15687bf912fefeab7aef2784b4f601afef3044ae.jpg)

![d2b45a88e1b7472eb56c26d2500ec1ee4ff41ca45f88b995cb82edb3be8c26ab.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/images/d2b45a88e1b7472eb56c26d2500ec1ee4ff41ca45f88b995cb82edb3be8c26ab.jpg)

### Tables

![040d6cd7fda4371c443b5aef4f588572d800b1a8a1329b54be100dc6c6e46ec6.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/040d6cd7fda4371c443b5aef4f588572d800b1a8a1329b54be100dc6c6e46ec6.jpg)

![13c26faebe221eca5cb0da9d89b930ee79889c784ad0e658d8bb20278c7944b8.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/13c26faebe221eca5cb0da9d89b930ee79889c784ad0e658d8bb20278c7944b8.jpg)

![369624ab9fa06b544ac3dc321cca147a4adc2821ae84e231ddd5dd48764fa32c.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/369624ab9fa06b544ac3dc321cca147a4adc2821ae84e231ddd5dd48764fa32c.jpg)

![4dde6f59eec9dbcf36af7305c2f31c608f2d0869f9d7932d328a6804902933ba.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/4dde6f59eec9dbcf36af7305c2f31c608f2d0869f9d7932d328a6804902933ba.jpg)

![5284c46ac772a766c2659f09ffd5913e72c42b22b2d36c1427c1a832f83f866d.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/5284c46ac772a766c2659f09ffd5913e72c42b22b2d36c1427c1a832f83f866d.jpg)

![6267a06db26178a4409efdc58015841c6996a2a3f65daf391c0156e0ed448577.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/6267a06db26178a4409efdc58015841c6996a2a3f65daf391c0156e0ed448577.jpg)

![b3c4b1351ef8c5664b0f385022df692acadf3716fc9ee390a6e9a726a122ce45.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/b3c4b1351ef8c5664b0f385022df692acadf3716fc9ee390a6e9a726a122ce45.jpg)

![e34e31d8eab825cd380cca08ef5cfe2faf69a23bf2ff5cde3da8e000f6480714.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/e34e31d8eab825cd380cca08ef5cfe2faf69a23bf2ff5cde3da8e000f6480714.jpg)

![ec0e8e621731e8f7ade970b20b426bd07dc437c9819fdd75db559ed3393072f3.jpg](../icml_results/3988_Revisiting%20OOD%20Generalization%20in%20Programmatic%20RL/tables/ec0e8e621731e8f7ade970b20b426bd07dc437c9819fdd75db559ed3393072f3.jpg)

<a id="paper-3948"></a>

## Personalized Image Generation via Human-in-the-loop Bayesian Optimization


### Images

![0876b1c5b731a42b54155b997881ad898f764f256440efbe9210b7d6d074e4ae.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/0876b1c5b731a42b54155b997881ad898f764f256440efbe9210b7d6d074e4ae.jpg)

![1451405c7e505f51f5e3723673ea5126567d82b31f9e9416ce1eae836c8b6eb8.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/1451405c7e505f51f5e3723673ea5126567d82b31f9e9416ce1eae836c8b6eb8.jpg)

![154a6d860f6289b7d0fe573b307ab8d8f3da7f2dc1af9f6bd2a146fd96ca915e.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/154a6d860f6289b7d0fe573b307ab8d8f3da7f2dc1af9f6bd2a146fd96ca915e.jpg)

![25efcd34873f060c46e0d70a5024eaa720e1aaad58f0a43824cd5c2688ea0748.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/25efcd34873f060c46e0d70a5024eaa720e1aaad58f0a43824cd5c2688ea0748.jpg)

![2958f5aee29f2921d19300248a7a58619d1f75b1747fcd60344193fb34a21335.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/2958f5aee29f2921d19300248a7a58619d1f75b1747fcd60344193fb34a21335.jpg)

![32d62309dea9cc8cd23f1bc1ce6d7dd262fe6463c7928c35a5f71fc6c1a00784.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/32d62309dea9cc8cd23f1bc1ce6d7dd262fe6463c7928c35a5f71fc6c1a00784.jpg)

![3882cdc2d43d99332ecbc0dfb9c5767734f077abef6918962ecb704408eed79b.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/3882cdc2d43d99332ecbc0dfb9c5767734f077abef6918962ecb704408eed79b.jpg)

![4b5fc9ebf73b59e6220ad21ee4add55de6fb495a11bdc436c6efad13f62a627d.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/4b5fc9ebf73b59e6220ad21ee4add55de6fb495a11bdc436c6efad13f62a627d.jpg)

![57148d9fc7049d0477226f9d17a3987f614cca232454148c02dd863290a2900e.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/57148d9fc7049d0477226f9d17a3987f614cca232454148c02dd863290a2900e.jpg)

![64d1376be6801a02ba32b936b4acdcbc0b2c4e6be941cd7bb582fc4765685177.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/64d1376be6801a02ba32b936b4acdcbc0b2c4e6be941cd7bb582fc4765685177.jpg)

![6ae4e5b21c77af90177d8708161b79882838a18c8bd8ecc2d9d5d47f3c536b14.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/6ae4e5b21c77af90177d8708161b79882838a18c8bd8ecc2d9d5d47f3c536b14.jpg)

![6dd7a0319799525171b797381392196220c4acded20c0dd90a17ea50f741583d.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/6dd7a0319799525171b797381392196220c4acded20c0dd90a17ea50f741583d.jpg)

![74b45543b2779894ca59da83ac6677c185eeb32dc8c895045538320562f21b05.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/74b45543b2779894ca59da83ac6677c185eeb32dc8c895045538320562f21b05.jpg)

![91a4baf30835b08ac78f023353fe9dab091594c9d28ea2dcb48b8a68845fc02b.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/91a4baf30835b08ac78f023353fe9dab091594c9d28ea2dcb48b8a68845fc02b.jpg)

![9f189da3d6a1eec913e1d0160cb92fc910e849047afbf70ba172948dce122414.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/9f189da3d6a1eec913e1d0160cb92fc910e849047afbf70ba172948dce122414.jpg)

![a05df8387da2ece0ce5b1ecb850c34067c5e2e42f43002fb787612bb03e807fe.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/a05df8387da2ece0ce5b1ecb850c34067c5e2e42f43002fb787612bb03e807fe.jpg)

![a44e0fe0d592aeb978c147abe42025740569c8f61fcc3cf9e26d56b4de8a2466.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/a44e0fe0d592aeb978c147abe42025740569c8f61fcc3cf9e26d56b4de8a2466.jpg)

![a58d65260db95eb80467a222d84ec67ace0b77eef6bc5f99331b169a3656f375.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/a58d65260db95eb80467a222d84ec67ace0b77eef6bc5f99331b169a3656f375.jpg)

![aaabb1f8dd0c32e2628118a7ba410406f6aac57b56c2c2ed5de5496708c94aea.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/aaabb1f8dd0c32e2628118a7ba410406f6aac57b56c2c2ed5de5496708c94aea.jpg)

![aaef372ff66d53122bd8a03a64fdbb0089e0508493cca43a4bbb11bc4dfb35c6.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/aaef372ff66d53122bd8a03a64fdbb0089e0508493cca43a4bbb11bc4dfb35c6.jpg)

![b7f19e61e88bbe2f3279e4892b88a97e76589e797480b85c572cbcd73ebe88a0.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/b7f19e61e88bbe2f3279e4892b88a97e76589e797480b85c572cbcd73ebe88a0.jpg)

![b9505081685d193c5ea6e43fbc2a6847946ebe8f49ef3278f05495ef04fde510.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/b9505081685d193c5ea6e43fbc2a6847946ebe8f49ef3278f05495ef04fde510.jpg)

![c1219304bf69a5e9de89382ad40cf6bdc610ddaac551a1500d3f7be9f133ff09.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/c1219304bf69a5e9de89382ad40cf6bdc610ddaac551a1500d3f7be9f133ff09.jpg)

![ee4aa021bf7637b3310620e9dedef886e85ea77f5e91278bfa7400bac52e9ead.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/images/ee4aa021bf7637b3310620e9dedef886e85ea77f5e91278bfa7400bac52e9ead.jpg)

### Tables

![0aa28b3165fea5a720ef49ae7012ffa7caacbd8f678762eb2966bedd4c092d02.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/0aa28b3165fea5a720ef49ae7012ffa7caacbd8f678762eb2966bedd4c092d02.jpg)

![0bc786e57bf06a0f82920507bf19b0d6e3004c00e837ac90016fb0e74be6e0b5.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/0bc786e57bf06a0f82920507bf19b0d6e3004c00e837ac90016fb0e74be6e0b5.jpg)

![1340df4921c652590f78fea3e482172b368bc2c70673dbe525d85e7168931ae2.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/1340df4921c652590f78fea3e482172b368bc2c70673dbe525d85e7168931ae2.jpg)

![333691cecbc97debeea0e41c2dcfaf2d4f5a0aa3fa5c252db8fddd1def4ca141.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/333691cecbc97debeea0e41c2dcfaf2d4f5a0aa3fa5c252db8fddd1def4ca141.jpg)

![34c08bf541ada349f2de9a381b5f37a54d2b4fed0e15830690914ab33422013c.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/34c08bf541ada349f2de9a381b5f37a54d2b4fed0e15830690914ab33422013c.jpg)

![3dcb1a87f6d3a094c7a58ffe4094dcf993742b9ec62e340f2c36d2586bcae68a.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/3dcb1a87f6d3a094c7a58ffe4094dcf993742b9ec62e340f2c36d2586bcae68a.jpg)

![519ef67f71ea6c26ffb59e3132a0330a85e0ddd13cc17f0b1c0e130121dfe805.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/519ef67f71ea6c26ffb59e3132a0330a85e0ddd13cc17f0b1c0e130121dfe805.jpg)

![9202263dc98b27327f8cd25d738c9307c5c3c6f38fc57d87cc597d5b61fe07c8.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/9202263dc98b27327f8cd25d738c9307c5c3c6f38fc57d87cc597d5b61fe07c8.jpg)

![93089cc283a282fea730fbb06712ab30da2c2c763ff0170beeb3d0ec2628844f.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/93089cc283a282fea730fbb06712ab30da2c2c763ff0170beeb3d0ec2628844f.jpg)

![9ee5527a98ba50fb6d1d833f63be595976542237f3762cd754356398d027af0e.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/9ee5527a98ba50fb6d1d833f63be595976542237f3762cd754356398d027af0e.jpg)

![e3832c2e3a5fe60334e1f831147713f5915482aab951470a91a315e3a2e255be.jpg](../icml_results/3989_Personalized%20Image%20Generation%20via%20Human-in-the-loop%20Bayesian%20Optimization/tables/e3832c2e3a5fe60334e1f831147713f5915482aab951470a91a315e3a2e255be.jpg)

<a id="paper-3949"></a>

## Panini: Continual Learning in Token Space via Structured Memory


### Images

![248797a23bdf3658c5f21d43d4d5bca68f53ff1f32050648c2a6260aa46925e4.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/images/248797a23bdf3658c5f21d43d4d5bca68f53ff1f32050648c2a6260aa46925e4.jpg)

![2f0594eb46043af04c9b84263d5b0868d6484b3e53a5edbf682cc5f95d23f185.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/images/2f0594eb46043af04c9b84263d5b0868d6484b3e53a5edbf682cc5f95d23f185.jpg)

![5c7b08ce99fc4b849c1a2de79405a9ac7329690ad59b5d06a413f1e4a5b8158c.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/images/5c7b08ce99fc4b849c1a2de79405a9ac7329690ad59b5d06a413f1e4a5b8158c.jpg)

### Tables

![0644ee4b1ea1a22fd236b4ffe0831c8bb62f0dc3a12c1564b5db415f50669b76.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/0644ee4b1ea1a22fd236b4ffe0831c8bb62f0dc3a12c1564b5db415f50669b76.jpg)

![0892a7c30435563e045837a9cab4fe62c4e21f07aec9a3b44eca9283e996d175.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/0892a7c30435563e045837a9cab4fe62c4e21f07aec9a3b44eca9283e996d175.jpg)

![112754db270bff1292fd9410c5804b2d6b6ec2c57b993a4ed4ea1dd18fe3d1e0.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/112754db270bff1292fd9410c5804b2d6b6ec2c57b993a4ed4ea1dd18fe3d1e0.jpg)

![2185f31979a16d5a0338dad7cf70141d0f81cb3dee4abe393ef98c95f9f195a6.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/2185f31979a16d5a0338dad7cf70141d0f81cb3dee4abe393ef98c95f9f195a6.jpg)

![2bfe593dd70cf0c066e27974dbf7bc69f4053bd71190770daf26200a967d9e98.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/2bfe593dd70cf0c066e27974dbf7bc69f4053bd71190770daf26200a967d9e98.jpg)

![3ac4bbd7f3ba308b17d61b8a25174e685bf6b082cd801387b579fc112ef0225c.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/3ac4bbd7f3ba308b17d61b8a25174e685bf6b082cd801387b579fc112ef0225c.jpg)

![40f3c29bce488bd580c44ca60eeaf7467b1b265d5af9357a1b2fbd92ff1c5a2d.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/40f3c29bce488bd580c44ca60eeaf7467b1b265d5af9357a1b2fbd92ff1c5a2d.jpg)

![42c9d62cc34e4194b49a1d4dcd6492e57fae3f91b2edd427668604c74bf48442.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/42c9d62cc34e4194b49a1d4dcd6492e57fae3f91b2edd427668604c74bf48442.jpg)

![4561f9ff860e33b4764316b9ea877a9d0dc1140361dc088474443c513e5c84ac.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/4561f9ff860e33b4764316b9ea877a9d0dc1140361dc088474443c513e5c84ac.jpg)

![48dcaeb8621256fbc8e6af0dd5b37e0a6c6881e1881e7867af9839185d75fef9.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/48dcaeb8621256fbc8e6af0dd5b37e0a6c6881e1881e7867af9839185d75fef9.jpg)

![4c1f536f2892e984903ddc111a94368068e353650d2d1e39ff3c0966ccda2c86.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/4c1f536f2892e984903ddc111a94368068e353650d2d1e39ff3c0966ccda2c86.jpg)

![4dc9b6e526dde5e6bd71a79c509e58689430234a0f0466f3e59e82a17c220a81.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/4dc9b6e526dde5e6bd71a79c509e58689430234a0f0466f3e59e82a17c220a81.jpg)

![5a346a8d6befd5ab45d10787df553e4dbebfa9fdbe11dc3082a1652835ed4538.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/5a346a8d6befd5ab45d10787df553e4dbebfa9fdbe11dc3082a1652835ed4538.jpg)

![6b8a80fabe42ca9f4fafedbcf2cd01a849c244f43dacaf7455ae05d4e320a5cf.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/6b8a80fabe42ca9f4fafedbcf2cd01a849c244f43dacaf7455ae05d4e320a5cf.jpg)

![6d196adaba849483bc38ecc35e89d905bce7a268c9e5b9f598954c033d1470ae.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/6d196adaba849483bc38ecc35e89d905bce7a268c9e5b9f598954c033d1470ae.jpg)

![6ef7b24f6c5cbde3974813f6d61d9f53ad36bacaef28cc5bead2f1c77e4dac33.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/6ef7b24f6c5cbde3974813f6d61d9f53ad36bacaef28cc5bead2f1c77e4dac33.jpg)

![72e09078b06fe0f0759b0ed21eb18326496c0c7e98e09ef1cdd0c2ec2c08775f.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/72e09078b06fe0f0759b0ed21eb18326496c0c7e98e09ef1cdd0c2ec2c08775f.jpg)

![8222fec8551316bdb23357c03f7bd947ae25dd619d1278013212ca3c8da8e275.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/8222fec8551316bdb23357c03f7bd947ae25dd619d1278013212ca3c8da8e275.jpg)

![9778917e151baa15dd868727b36b71760692d4b7cf258f580e68c695d68594ca.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/9778917e151baa15dd868727b36b71760692d4b7cf258f580e68c695d68594ca.jpg)

![97ed98231b7dd0de774d39c3cb791efb4d71c599b2651de1b9911f77808ca207.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/97ed98231b7dd0de774d39c3cb791efb4d71c599b2651de1b9911f77808ca207.jpg)

![a20f0735e7af6c53293c9ce22f2d329cd9e7b27de31bbc2e726ae6c2ee0d3396.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/a20f0735e7af6c53293c9ce22f2d329cd9e7b27de31bbc2e726ae6c2ee0d3396.jpg)

![a4b8ccc0bbf7c0b5c81ed9d448a8c735210cef2d6a300bad87016fc7fda60385.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/a4b8ccc0bbf7c0b5c81ed9d448a8c735210cef2d6a300bad87016fc7fda60385.jpg)

![b287c0cb2578ff06024aab256214518aa6fa0c920ec801c29847c911d932d8ae.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/b287c0cb2578ff06024aab256214518aa6fa0c920ec801c29847c911d932d8ae.jpg)

![bd3c44095b14c322f31cdc45efcdf29403cd295c7a86d8c6147bc46132d31b2e.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/bd3c44095b14c322f31cdc45efcdf29403cd295c7a86d8c6147bc46132d31b2e.jpg)

![c9fff635a7dbee11b2fab5a6144d1fb91535f541edf423be8898a1db838460f7.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/c9fff635a7dbee11b2fab5a6144d1fb91535f541edf423be8898a1db838460f7.jpg)

![cf0000a883073897ddb95351d38a499f8214fae247c86135fc07fc17c46a9eab.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/cf0000a883073897ddb95351d38a499f8214fae247c86135fc07fc17c46a9eab.jpg)

![d1e1af8c2598047aac72422d09c356ed77b676a8960bbaf62671e65298bb5812.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/d1e1af8c2598047aac72422d09c356ed77b676a8960bbaf62671e65298bb5812.jpg)

![e19d2a9df6ce6453609a0a688d126846f2bba8db65e1a9978a3e27d40103cc45.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/e19d2a9df6ce6453609a0a688d126846f2bba8db65e1a9978a3e27d40103cc45.jpg)

![f571b146dac20cebc2850650ec7e95c7be5d598c8b63ffa453e93f3a531ff625.jpg](../icml_results/3990_Panini_%20Continual%20Learning%20in%20Token%20Space%20via%20Structured%20Memory/tables/f571b146dac20cebc2850650ec7e95c7be5d598c8b63ffa453e93f3a531ff625.jpg)

<a id="paper-3950"></a>

## MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio


### Images

![4a8f70c2c39af04a34abff9de159c798fd5eb78608d99e74cc497911af069a85.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/images/4a8f70c2c39af04a34abff9de159c798fd5eb78608d99e74cc497911af069a85.jpg)

![643df233f0ba315193243499fb2f658183f1fc944349e3d31a9bc3fcea080617.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/images/643df233f0ba315193243499fb2f658183f1fc944349e3d31a9bc3fcea080617.jpg)

![846e4630d5c5230bee680048b19aa735a97e3b9d9ae967791b70eed1b54930d2.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/images/846e4630d5c5230bee680048b19aa735a97e3b9d9ae967791b70eed1b54930d2.jpg)

![c389d79057fcd87182a83536cc4c8814e4ad103faba3f9f73edc91a31f9c4333.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/images/c389d79057fcd87182a83536cc4c8814e4ad103faba3f9f73edc91a31f9c4333.jpg)

![d12514fcb5b217f1eef412db3b6dc7aba5fb56c97e521a0164181b294338cebb.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/images/d12514fcb5b217f1eef412db3b6dc7aba5fb56c97e521a0164181b294338cebb.jpg)

### Tables

![09cb5dbce9c639722bdc2f0b2bc0849afc38d0c387bf8a3dbf62c97b67d52b04.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/09cb5dbce9c639722bdc2f0b2bc0849afc38d0c387bf8a3dbf62c97b67d52b04.jpg)

![0af2e369b7e826921877f8952de73ea01e7a2e6273235b7570a22437b6565fcc.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/0af2e369b7e826921877f8952de73ea01e7a2e6273235b7570a22437b6565fcc.jpg)

![1626a5b59ea8701e3e03f524e3136b6c2e3ba9125755ae8ea4b75ad88dd6198e.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/1626a5b59ea8701e3e03f524e3136b6c2e3ba9125755ae8ea4b75ad88dd6198e.jpg)

![398bc460e1708941a3f4ebe0be7aba4fe1485679eaf5369f0d3d8dc5ebef6f5e.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/398bc460e1708941a3f4ebe0be7aba4fe1485679eaf5369f0d3d8dc5ebef6f5e.jpg)

![3a8df8eaef67ad46ad89a9b4667c745049ecf89340b47f5cbde7c8aeb542e564.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/3a8df8eaef67ad46ad89a9b4667c745049ecf89340b47f5cbde7c8aeb542e564.jpg)

![489a74410290ed5139f14e27d25f90a9efc7dd4e23a3349b1e78b6bbe28956a8.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/489a74410290ed5139f14e27d25f90a9efc7dd4e23a3349b1e78b6bbe28956a8.jpg)

![4df1004bfc229ee742bf7a31fe66b9cb1b1d0ca91f34686251f06d609bf60607.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/4df1004bfc229ee742bf7a31fe66b9cb1b1d0ca91f34686251f06d609bf60607.jpg)

![560362d82dd98cef5e32fa84705e6f95031a07d38933764b0f8c6c594c0a9355.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/560362d82dd98cef5e32fa84705e6f95031a07d38933764b0f8c6c594c0a9355.jpg)

![6eb14858f56cc73b1a382e75cacd18619fbd4338dd4c1a7e34bd803f47f0ca41.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/6eb14858f56cc73b1a382e75cacd18619fbd4338dd4c1a7e34bd803f47f0ca41.jpg)

![88651c8c926c41976b69c06eb8af93b5647a5dc9fcc914432fecaac08f604188.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/88651c8c926c41976b69c06eb8af93b5647a5dc9fcc914432fecaac08f604188.jpg)

![9598ae69971ca8f573fd473f6a07c5deb790e9f351d9e3f4cf3a14d196dd028d.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/9598ae69971ca8f573fd473f6a07c5deb790e9f351d9e3f4cf3a14d196dd028d.jpg)

![970e8f9471bff341bc8e1f6118c155f69dd6692331df2ffb3f1a61a8de0563d0.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/970e8f9471bff341bc8e1f6118c155f69dd6692331df2ffb3f1a61a8de0563d0.jpg)

![9e84b54ae34fdf4d5c61778d303f8c9cf938be9c93bdf226a00bd2d3b8a40792.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/9e84b54ae34fdf4d5c61778d303f8c9cf938be9c93bdf226a00bd2d3b8a40792.jpg)

![f3baaade29e6907387cf1b89e7349fe9622c08f2ad4f728c04eef921b6fb4750.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/f3baaade29e6907387cf1b89e7349fe9622c08f2ad4f728c04eef921b6fb4750.jpg)

![fb3fc0eee3a1b1ccec95ae5086fdf13aec237d8c5e298153fd58d71bbd0ab7cf.jpg](../icml_results/3991_MedMosaic_%20A%20Challenging%20Large%20Scale%20Benchmark%20of%20Diverse%20Medical%20Audio/tables/fb3fc0eee3a1b1ccec95ae5086fdf13aec237d8c5e298153fd58d71bbd0ab7cf.jpg)

<a id="paper-3951"></a>

## A model of errors in transformers


### Images

![0ce5aef4c239d8b8de4fbde5e4ba17a3aa0ed787f50d317b03d9e76dfbed5c7f.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/0ce5aef4c239d8b8de4fbde5e4ba17a3aa0ed787f50d317b03d9e76dfbed5c7f.jpg)

![14c2cb37af9a8ce7401df15cfd9180ec5ae3bf46c8262b753bb44b1f2af6ad9f.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/14c2cb37af9a8ce7401df15cfd9180ec5ae3bf46c8262b753bb44b1f2af6ad9f.jpg)

![1ee015d42c7f595362fecc25e66de4269d5da8f2f3942a7f3c158ad9f1a87fa6.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/1ee015d42c7f595362fecc25e66de4269d5da8f2f3942a7f3c158ad9f1a87fa6.jpg)

![223323aa95d87b5702f02c7a269770b5e79f5f5e2084d9acc08e4b1eb0ecad52.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/223323aa95d87b5702f02c7a269770b5e79f5f5e2084d9acc08e4b1eb0ecad52.jpg)

![2d6051d57ecd518ff996fa9dc8ea82ef93f5dd8b72639a24071825fc10d84376.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/2d6051d57ecd518ff996fa9dc8ea82ef93f5dd8b72639a24071825fc10d84376.jpg)

![31cb2288ef937095abf8e950c2407bffde8392bb4d8a7c39ded8130e81379992.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/31cb2288ef937095abf8e950c2407bffde8392bb4d8a7c39ded8130e81379992.jpg)

![361fd8fe776623fdc452090fc0c41b15b2bf2163756d7e0dcca3262d948b3331.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/361fd8fe776623fdc452090fc0c41b15b2bf2163756d7e0dcca3262d948b3331.jpg)

![53a294c55722876447e2e53915230c77c86f9c57fcc969f68f1d2322f424e3c0.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/53a294c55722876447e2e53915230c77c86f9c57fcc969f68f1d2322f424e3c0.jpg)

![6dc888dae0a0f2898da36734de615c4dc160422d07474aac4cc393d38e138b5c.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/6dc888dae0a0f2898da36734de615c4dc160422d07474aac4cc393d38e138b5c.jpg)

![722000e647c488b6775de4a03d6b5d356d9d4c78175781445ab1b05d47ca6b5b.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/722000e647c488b6775de4a03d6b5d356d9d4c78175781445ab1b05d47ca6b5b.jpg)

![8f3a8ce2dfe2d5c08bd939a4af941344c64b585a477dd7e5cdba8a437a035594.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/8f3a8ce2dfe2d5c08bd939a4af941344c64b585a477dd7e5cdba8a437a035594.jpg)

![99e0ba80611c83ec1bfb5f7d16795c2c3f78feabe6ec70644e26e3a4beeb87a3.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/99e0ba80611c83ec1bfb5f7d16795c2c3f78feabe6ec70644e26e3a4beeb87a3.jpg)

![aa00b2fe2fdef58db2ecb98017d6cdf666557ca671b682d7fd7928bb8df5aeed.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/aa00b2fe2fdef58db2ecb98017d6cdf666557ca671b682d7fd7928bb8df5aeed.jpg)

![ad29e87e641d9bee0a0613f602e28f44ba3d3561078533ab0961b7782c3d34ef.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/ad29e87e641d9bee0a0613f602e28f44ba3d3561078533ab0961b7782c3d34ef.jpg)

![afe5c6deb77f226c45c8188d96f1ad15f59ef8f23b41b259e8a2a71002307971.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/afe5c6deb77f226c45c8188d96f1ad15f59ef8f23b41b259e8a2a71002307971.jpg)

![b465e31b8a8db0ffa80f9d0e8f8b8e75054732753f6baece999b63bf17453af1.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/b465e31b8a8db0ffa80f9d0e8f8b8e75054732753f6baece999b63bf17453af1.jpg)

![b4c153b72d778bae89131da10edce198bd27b565ed55ba76136e7c0d3044e03f.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/b4c153b72d778bae89131da10edce198bd27b565ed55ba76136e7c0d3044e03f.jpg)

![c2d6eb997555580b6452d04dc59dced1b6cac443694edb1aa99d39479b00376f.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/c2d6eb997555580b6452d04dc59dced1b6cac443694edb1aa99d39479b00376f.jpg)

![c9e67af1c98c293a46fdfaa322c82ac4412c4c3a38df0e2f222b8b6e239786c8.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/c9e67af1c98c293a46fdfaa322c82ac4412c4c3a38df0e2f222b8b6e239786c8.jpg)

![d47fd5a8f80df2a36af9fbb1af826187015656e295389dbdab16f7f6a7408263.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/d47fd5a8f80df2a36af9fbb1af826187015656e295389dbdab16f7f6a7408263.jpg)

![e0a0fb45b5fd9c104a6d206878dcc7a8361c4a0e540dbad71ec31996f80ee452.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/e0a0fb45b5fd9c104a6d206878dcc7a8361c4a0e540dbad71ec31996f80ee452.jpg)

![e47b52f4ed6c97462edd6c61086fead97cade2cdec4d19e50ff031c6b00ddfe0.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/e47b52f4ed6c97462edd6c61086fead97cade2cdec4d19e50ff031c6b00ddfe0.jpg)

![e5a91b9bacf9f3f55bd14cff4525d79d4ce13e0e3391f76238f3a931df15293d.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/images/e5a91b9bacf9f3f55bd14cff4525d79d4ce13e0e3391f76238f3a931df15293d.jpg)

### Tables

![1ff2d74fe9e532845299438214cccf9d0383852dfba66122fb555cb59020026d.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/tables/1ff2d74fe9e532845299438214cccf9d0383852dfba66122fb555cb59020026d.jpg)

![60defb09cb473bd51a6097f689a5634ea6e0a2e56590beff011c8277881b251b.jpg](../icml_results/3992_A%20model%20of%20errors%20in%20transformers/tables/60defb09cb473bd51a6097f689a5634ea6e0a2e56590beff011c8277881b251b.jpg)

<a id="paper-3952"></a>

## One-step Optimal Transport via Regularized Distribution Matching Distillation


### Images

![0ecff8197aef52deba9185e5ad1b089774c8503eaceeb3ad47c49eff0cdbe784.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/0ecff8197aef52deba9185e5ad1b089774c8503eaceeb3ad47c49eff0cdbe784.jpg)

![20296c3688a9403386cd5c2d6d64f569eecdaea545eb4da6166ceb6beecb048b.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/20296c3688a9403386cd5c2d6d64f569eecdaea545eb4da6166ceb6beecb048b.jpg)

![244c0ba8a137f7a89cb5e025c4ad21986a868d3ad60f058cb7fe6a4c9e696bb8.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/244c0ba8a137f7a89cb5e025c4ad21986a868d3ad60f058cb7fe6a4c9e696bb8.jpg)

![299017f134b82d3ef5b74f47b6c65f26e72ff3cb35b1bbd008784500baa13cea.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/299017f134b82d3ef5b74f47b6c65f26e72ff3cb35b1bbd008784500baa13cea.jpg)

![2c053e14292a140cfe27d83c12f04ae4f98405442dfd2dd8e8ed2c2a9fa78ae7.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/2c053e14292a140cfe27d83c12f04ae4f98405442dfd2dd8e8ed2c2a9fa78ae7.jpg)

![34799b51fbdb5e90bffacc3474808743a81d21d33f3868795e3beb917cc46067.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/34799b51fbdb5e90bffacc3474808743a81d21d33f3868795e3beb917cc46067.jpg)

![36fef4bb5784477f60aa38f26dd8986d9e4959c699674bd73e975de2c57af031.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/36fef4bb5784477f60aa38f26dd8986d9e4959c699674bd73e975de2c57af031.jpg)

![39a863f33b88ad62bafdc8c124af62a6372ffc2f1c690073e1c1aff11144dc99.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/39a863f33b88ad62bafdc8c124af62a6372ffc2f1c690073e1c1aff11144dc99.jpg)

![74d6c2b5540be998626d244b37ce7c5ad38c9f19a6454ba8fb5f93f4cf3ac3dd.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/74d6c2b5540be998626d244b37ce7c5ad38c9f19a6454ba8fb5f93f4cf3ac3dd.jpg)

![78c71de2e5ee66433686d5bf29db57e050d1d4af77c4a3c540475115f3dc2e15.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/78c71de2e5ee66433686d5bf29db57e050d1d4af77c4a3c540475115f3dc2e15.jpg)

![87621aeb1bda0710d2ee6ff0fb821010e2a1fa069e3e6e94256910d6570d1889.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/87621aeb1bda0710d2ee6ff0fb821010e2a1fa069e3e6e94256910d6570d1889.jpg)

![9ac161edf3209bed316490a6e41fdf14a7d051b969fbed27ef4249e3c673168f.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/9ac161edf3209bed316490a6e41fdf14a7d051b969fbed27ef4249e3c673168f.jpg)

![a5c0fe1f1a014c4725867e901c2197c880d228346882988784adc22073f216af.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/a5c0fe1f1a014c4725867e901c2197c880d228346882988784adc22073f216af.jpg)

![ae72db90baaa8a317ebbb74cd87b5d8c7a09b4eac6f3bd026d5c1e98e9d2566b.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/ae72db90baaa8a317ebbb74cd87b5d8c7a09b4eac6f3bd026d5c1e98e9d2566b.jpg)

![b484c668bffbc12a60ed69311dde01be256f0b4d0f57d97c44f3e60bdb2657b7.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/b484c668bffbc12a60ed69311dde01be256f0b4d0f57d97c44f3e60bdb2657b7.jpg)

![bdbc9c030a0752d5fbb80e7dd9a0a532db5facc2fb4a47b27c0a69fdc78a93fa.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/bdbc9c030a0752d5fbb80e7dd9a0a532db5facc2fb4a47b27c0a69fdc78a93fa.jpg)

![bfe96b95eaa2206e3ce805406325147738d30b4391d212d17067add54d6ff107.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/bfe96b95eaa2206e3ce805406325147738d30b4391d212d17067add54d6ff107.jpg)

![cae39ffff126c912e89466deea52f2131646214049f50c85cdae3e194df0bf0d.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/cae39ffff126c912e89466deea52f2131646214049f50c85cdae3e194df0bf0d.jpg)

![cb596e9160ea9252e179e685a5b67aec7d5791ac6a9a051baf886e98a2079d9b.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/cb596e9160ea9252e179e685a5b67aec7d5791ac6a9a051baf886e98a2079d9b.jpg)

![cbf06d5ac2e58b9e53bf8be96201cf4105134dc96d018d2edac32b19296daa4b.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/cbf06d5ac2e58b9e53bf8be96201cf4105134dc96d018d2edac32b19296daa4b.jpg)

![da7b4230c10bc3926299a2b76b5eeab4deb3623264355c9ab06347d3b68506a1.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/da7b4230c10bc3926299a2b76b5eeab4deb3623264355c9ab06347d3b68506a1.jpg)

![e93423f4395c03580e01d273cdca4ea1780131d8211314d8099b30aacff2bdd6.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/e93423f4395c03580e01d273cdca4ea1780131d8211314d8099b30aacff2bdd6.jpg)

![ec0014cac52b937db4a92a94afdf23c272c510ca32ee4743f71c1f58335b64f9.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/ec0014cac52b937db4a92a94afdf23c272c510ca32ee4743f71c1f58335b64f9.jpg)

![f4c9ecf164ec643948a8b544c2ccd784b8681266e483333edab5e8600b41026c.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/images/f4c9ecf164ec643948a8b544c2ccd784b8681266e483333edab5e8600b41026c.jpg)

### Tables

![0ed0e589f83717e35e2bbb885a134d6269621dbe946f970c42747d0f354caebd.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/0ed0e589f83717e35e2bbb885a134d6269621dbe946f970c42747d0f354caebd.jpg)

![30794bf343d83bfbd71761cceed14e3b1d32a3ab4a5f80cfd525dc1057be5819.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/30794bf343d83bfbd71761cceed14e3b1d32a3ab4a5f80cfd525dc1057be5819.jpg)

![6251d18c601a9c95a799b145d5f167b6914efb1d5f872ba37c53c7ac79fb1951.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/6251d18c601a9c95a799b145d5f167b6914efb1d5f872ba37c53c7ac79fb1951.jpg)

![6c6535d29d7972907a4218b73c86f73f076e549096f61d27e15274f2a947a70c.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/6c6535d29d7972907a4218b73c86f73f076e549096f61d27e15274f2a947a70c.jpg)

![77c98ae27b3283e846a1f0723a9e4061f66280ca449a6efd2ef2a6bc3c9f477b.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/77c98ae27b3283e846a1f0723a9e4061f66280ca449a6efd2ef2a6bc3c9f477b.jpg)

![90c54a3c8bdc3a4c0358596d41175bc37b2c1473aaef7c05a4087348ed7ab15a.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/90c54a3c8bdc3a4c0358596d41175bc37b2c1473aaef7c05a4087348ed7ab15a.jpg)

![b480bc778acb66e20b63934b716c90bf6babd2a3f89c82892e608871a12d9f26.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/b480bc778acb66e20b63934b716c90bf6babd2a3f89c82892e608871a12d9f26.jpg)

![f0e13ad7cf45345a5c4088174f387964343d12cba0203ae6055dd2ac90669a13.jpg](../icml_results/3993_One-step%20Optimal%20Transport%20via%20Regularized%20Distribution%20Matching%20Distillation/tables/f0e13ad7cf45345a5c4088174f387964343d12cba0203ae6055dd2ac90669a13.jpg)

<a id="paper-3953"></a>

## $α$-PFN: Fast Entropy Search via In-Context Learning


### Images

![3ff32cc8e143ddffccc67697e45bed7a395d9b0adcf6345300804bc36fbbba68.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/3ff32cc8e143ddffccc67697e45bed7a395d9b0adcf6345300804bc36fbbba68.jpg)

![47760749fc81a0dd85da72b18b3d4df73ab98773744dcc410c4b464f791c3a03.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/47760749fc81a0dd85da72b18b3d4df73ab98773744dcc410c4b464f791c3a03.jpg)

![5b1251ff1aeac2c2fd18a8dc981a2ca81fb88b1a294a782c4aebe1d6af91cb0b.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/5b1251ff1aeac2c2fd18a8dc981a2ca81fb88b1a294a782c4aebe1d6af91cb0b.jpg)

![639b8d5bef791a5c1ac441bb3c5a9b9b3da10fe46be10fcd70dae22a7333ae93.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/639b8d5bef791a5c1ac441bb3c5a9b9b3da10fe46be10fcd70dae22a7333ae93.jpg)

![a15f10a1ea6c1e6fe11d7769c2ea0b0a9337ad8fca8489717f3c4040d8d48c9d.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/a15f10a1ea6c1e6fe11d7769c2ea0b0a9337ad8fca8489717f3c4040d8d48c9d.jpg)

![a5681c7f500a720bd38040c3dbad354f2af7718a8bdb322f46a6c82683c9dbc8.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/a5681c7f500a720bd38040c3dbad354f2af7718a8bdb322f46a6c82683c9dbc8.jpg)

![c2a41c731e8fc0a1460c935678d6afb38492d551c32bdb5df4eee18255ba30fa.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/c2a41c731e8fc0a1460c935678d6afb38492d551c32bdb5df4eee18255ba30fa.jpg)

![f2226777d51f1b7075f8df3380761ec58fbc88e2f7afc40e44b25e5d8157920d.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/f2226777d51f1b7075f8df3380761ec58fbc88e2f7afc40e44b25e5d8157920d.jpg)

![f77554039f7d6ab2d3bd7a025a802bb29864e494f0caf749e72c085c5c8bb290.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/images/f77554039f7d6ab2d3bd7a025a802bb29864e494f0caf749e72c085c5c8bb290.jpg)

### Tables

![42e20c872d6fde8c9cc5b689a10ace4d304e5aa0f75fa9daaac3ae03be18cc32.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/tables/42e20c872d6fde8c9cc5b689a10ace4d304e5aa0f75fa9daaac3ae03be18cc32.jpg)

![dce07fc9b904e511005e5e510c2c2618fa667ef7cf0bbd0e022999d6861695be.jpg](../icml_results/3994_%24%CE%B1%24-PFN_%20Fast%20Entropy%20Search%20via%20In-Context%20Learning/tables/dce07fc9b904e511005e5e510c2c2618fa667ef7cf0bbd0e022999d6861695be.jpg)

<a id="paper-3954"></a>

## Particle Flow for Learning from Label Proportions


### Images

![02591d646fee61e90416e6f234582b73ed017c118544b7865917287408ac88c3.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/02591d646fee61e90416e6f234582b73ed017c118544b7865917287408ac88c3.jpg)

![0857a831e31431dc5756faa8d4eebde66b25ac9ceb4bbb50cff7dc53490aaca4.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/0857a831e31431dc5756faa8d4eebde66b25ac9ceb4bbb50cff7dc53490aaca4.jpg)

![1ab4b29da07b8a1943bfadf48d215aa3a054042998f468edcb3ce429f1f17a26.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/1ab4b29da07b8a1943bfadf48d215aa3a054042998f468edcb3ce429f1f17a26.jpg)

![2f2eaf49acc5fab81e78281568a7200102243556ba57b1bea0415eb2da73aaf1.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/2f2eaf49acc5fab81e78281568a7200102243556ba57b1bea0415eb2da73aaf1.jpg)

![4801b8f6f9f89461196f27fc8e4e9572ca7a4bb36aad3a81ad06f53cd3423f40.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/4801b8f6f9f89461196f27fc8e4e9572ca7a4bb36aad3a81ad06f53cd3423f40.jpg)

![4879b96807fba77dcba1968cc64c92b6a43b9a89dcfcc419bf97d0cba9040556.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/4879b96807fba77dcba1968cc64c92b6a43b9a89dcfcc419bf97d0cba9040556.jpg)

![524a6f48d20f8e352fc273b2e8a903330fb450bac4af8c103cc237ef23676f7f.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/524a6f48d20f8e352fc273b2e8a903330fb450bac4af8c103cc237ef23676f7f.jpg)

![5d92e086c2b2caaf2a290f815f87972fca9e160f8699a9d713eab7f668d3dc5b.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/5d92e086c2b2caaf2a290f815f87972fca9e160f8699a9d713eab7f668d3dc5b.jpg)

![8e55463ec5356641ff126bacfb1f8d4fd153eb1172f1b7404046399c88a8bf95.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/8e55463ec5356641ff126bacfb1f8d4fd153eb1172f1b7404046399c88a8bf95.jpg)

![9168ffe4bc974a4120ff7bddedac245c4efb68478241d484bbdba00f0a8f71fc.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/9168ffe4bc974a4120ff7bddedac245c4efb68478241d484bbdba00f0a8f71fc.jpg)

![96ea149f113229e4e85b4e705433b81b0ac2b4e7984acb20ee6dad940d8f3e5b.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/96ea149f113229e4e85b4e705433b81b0ac2b4e7984acb20ee6dad940d8f3e5b.jpg)

![cd1504eab793abdaa9c547539f502a3bdf36d74bec205ef2e0182678f7aa5e62.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/cd1504eab793abdaa9c547539f502a3bdf36d74bec205ef2e0182678f7aa5e62.jpg)

![e3cfe9b3d5af520b0928fc7c00201bfbedf38cdcab1ece7e94d356b53f5c818e.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/e3cfe9b3d5af520b0928fc7c00201bfbedf38cdcab1ece7e94d356b53f5c818e.jpg)

![ebf0cc734208018d1632dc8ff05f7bc6bedc33a2ea6bddea243577f0e61f4b3a.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/ebf0cc734208018d1632dc8ff05f7bc6bedc33a2ea6bddea243577f0e61f4b3a.jpg)

![f0c77970ac74cb9225c6acd6c6b592d13d8b90582360309aeb9dc36ccd4f211c.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/f0c77970ac74cb9225c6acd6c6b592d13d8b90582360309aeb9dc36ccd4f211c.jpg)

![f698509be8e07d3223e1e14ac1f0f35064358514bc56830d748694214e98458b.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/images/f698509be8e07d3223e1e14ac1f0f35064358514bc56830d748694214e98458b.jpg)

### Tables

![11de56d3294fd1388155f60f9e0fa5708ab81a60a37febd03cb699da7e569c9a.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/11de56d3294fd1388155f60f9e0fa5708ab81a60a37febd03cb699da7e569c9a.jpg)

![38fe7aae6f68af6fe48a2e2647c25fa4fb54f3490c334cabc1cef4312f0cb2c4.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/38fe7aae6f68af6fe48a2e2647c25fa4fb54f3490c334cabc1cef4312f0cb2c4.jpg)

![7987f8c47d0abdb62a2d5b016868197311a0c579c3c8523347a6fd09faf85815.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/7987f8c47d0abdb62a2d5b016868197311a0c579c3c8523347a6fd09faf85815.jpg)

![90267172892716db3d961831a342a7ecb08653eb87f9254d60c8e1f8c3e6e99f.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/90267172892716db3d961831a342a7ecb08653eb87f9254d60c8e1f8c3e6e99f.jpg)

![c29919494d5c3a2ede6f4391af0176cf2d5deb5c57c5648ea6683d0ce7c237c1.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/c29919494d5c3a2ede6f4391af0176cf2d5deb5c57c5648ea6683d0ce7c237c1.jpg)

![d1a80816fae303b27c7d16833a35a1ae15bb77d3a0af0d0fea701cef42d813b5.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/d1a80816fae303b27c7d16833a35a1ae15bb77d3a0af0d0fea701cef42d813b5.jpg)

![eb6442ba0bdf9052160e669898568454d741d720137de600e40a7471508eb81a.jpg](../icml_results/3995_Particle%20Flow%20for%20Learning%20from%20Label%20Proportions/tables/eb6442ba0bdf9052160e669898568454d741d720137de600e40a7471508eb81a.jpg)

<a id="paper-3955"></a>

## Trees to Flows and Back: Unifying Decision Trees and Diffusion Models


### Images

![09381f716e233600f28883e0d3c1bc93119c79de66ac19f1e7dcebd5048de42c.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/09381f716e233600f28883e0d3c1bc93119c79de66ac19f1e7dcebd5048de42c.jpg)

![17a0c149b07137a33b297915601f14b04acd9b84b628088cc05c4bb5fbe7f2f8.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/17a0c149b07137a33b297915601f14b04acd9b84b628088cc05c4bb5fbe7f2f8.jpg)

![38dea2a82e189958c8c0311e34524eed5515c77e6095b4831712a217b5730d21.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/38dea2a82e189958c8c0311e34524eed5515c77e6095b4831712a217b5730d21.jpg)

![513b5c9c282e98c2ac0a6b7f1cf67b592c1427d9791910250f61d8d1bb986a2b.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/513b5c9c282e98c2ac0a6b7f1cf67b592c1427d9791910250f61d8d1bb986a2b.jpg)

![5a61d27964d5ba5579ad38336adfb18c24f10b807fb837deeaf68f7ecb6df529.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/5a61d27964d5ba5579ad38336adfb18c24f10b807fb837deeaf68f7ecb6df529.jpg)

![6b8830b5dc7f8e72e761811fee272c335eeb38170832b7ddf715e6592947475d.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/6b8830b5dc7f8e72e761811fee272c335eeb38170832b7ddf715e6592947475d.jpg)

![79bc609d82984f3caef1af750db122775394b3dd00d255a54ba1385e2dc4c81a.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/79bc609d82984f3caef1af750db122775394b3dd00d255a54ba1385e2dc4c81a.jpg)

![79fd878991c1fb2641685b58ba54c5763d4d8e7813661600261aa2797f9fef37.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/79fd878991c1fb2641685b58ba54c5763d4d8e7813661600261aa2797f9fef37.jpg)

![993bbee87e304ad6be2c3bccd89b9f2dd8806ebed9065989f626ad28123ad5bc.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/993bbee87e304ad6be2c3bccd89b9f2dd8806ebed9065989f626ad28123ad5bc.jpg)

![a96afc85562fa50b18f7245e8a69b44a5d10e93c3373f74a2b365bb11ed36de0.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/a96afc85562fa50b18f7245e8a69b44a5d10e93c3373f74a2b365bb11ed36de0.jpg)

![b2ce9803e1724029d8545beff67a3cccde0b0112bd16558f74d89adf8fb6b936.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/b2ce9803e1724029d8545beff67a3cccde0b0112bd16558f74d89adf8fb6b936.jpg)

![b78b76ab0648685b8329a6f362a492bcd0dc1bc1883ff1f9d82929a3d2944a53.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/b78b76ab0648685b8329a6f362a492bcd0dc1bc1883ff1f9d82929a3d2944a53.jpg)

![c9175833e63f9b4b99a53b132bed0acd88f992d63b214b3d693ac41fd9bbb613.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/c9175833e63f9b4b99a53b132bed0acd88f992d63b214b3d693ac41fd9bbb613.jpg)

![db8813045cf5bfa7b333ff53d57f0acfe819048e4dfe9c4b2d0115560cd4ba5d.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/db8813045cf5bfa7b333ff53d57f0acfe819048e4dfe9c4b2d0115560cd4ba5d.jpg)

![f2061586cc38f2e84bc36a382593772b1cca0450254d24fdc1e29a5b9902cbce.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/images/f2061586cc38f2e84bc36a382593772b1cca0450254d24fdc1e29a5b9902cbce.jpg)

### Tables

![0078c78c0acfdd0ae6f0dfcffc9b9ea8022fe0a1048902370e679da356c64c2e.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/tables/0078c78c0acfdd0ae6f0dfcffc9b9ea8022fe0a1048902370e679da356c64c2e.jpg)

![379668fb701663076e187d95ca2f4a56406d5196083fe658a3d10d73765fb954.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/tables/379668fb701663076e187d95ca2f4a56406d5196083fe658a3d10d73765fb954.jpg)

![3961665d4bb4f340118362f1249f60570c921bd2faef443e8532bfb2c14202af.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/tables/3961665d4bb4f340118362f1249f60570c921bd2faef443e8532bfb2c14202af.jpg)

![5dcf705829ffc4dceb696dd28a0be37a9dc9e7ba200fe0cc48d5391dd539fcd8.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/tables/5dcf705829ffc4dceb696dd28a0be37a9dc9e7ba200fe0cc48d5391dd539fcd8.jpg)

![6d2be2e0c334199c00c5de05d07efd7a1a43b451d370225e77e605d86385f1f1.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/tables/6d2be2e0c334199c00c5de05d07efd7a1a43b451d370225e77e605d86385f1f1.jpg)

![abb3ce2a576f8df2e5cb1eaa8ec99d6179a828051f83b097fb6b00e86e44aa7e.jpg](../icml_results/3996_Trees%20to%20Flows%20and%20Back_%20Unifying%20Decision%20Trees%20and%20Diffusion%20Models/tables/abb3ce2a576f8df2e5cb1eaa8ec99d6179a828051f83b097fb6b00e86e44aa7e.jpg)

<a id="paper-3956"></a>

## Reasoning Models Are Test Exploiters: Rethinking Multiple Choice


### Images

![00caa68d541e19e2af101fa3496287ddbf6bd7116af2e5a8a3a874a8fa0b6731.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/00caa68d541e19e2af101fa3496287ddbf6bd7116af2e5a8a3a874a8fa0b6731.jpg)

![08a22d804550b6fa13b90f6d5db08f89aff14a0ce8656fc8a56893db2a7b3c94.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/08a22d804550b6fa13b90f6d5db08f89aff14a0ce8656fc8a56893db2a7b3c94.jpg)

![302623960072849c6e33b413796452ac0ae8a330b55b18d6197d895bb6a4f8b3.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/302623960072849c6e33b413796452ac0ae8a330b55b18d6197d895bb6a4f8b3.jpg)

![4f3eb775997d34584bd78495107cf89000b422d6e493bb0ebdaab752c32acb73.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/4f3eb775997d34584bd78495107cf89000b422d6e493bb0ebdaab752c32acb73.jpg)

![545c2f1e143953ea8bdfed057315eeafb99bf01bd963d8a5922e428b1047be3d.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/545c2f1e143953ea8bdfed057315eeafb99bf01bd963d8a5922e428b1047be3d.jpg)

![777d20340ff56a4f463ba23878102abe645822120de33a8fec49b7a72d769c38.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/777d20340ff56a4f463ba23878102abe645822120de33a8fec49b7a72d769c38.jpg)

![824b4d927fa5f4ad4b32d4b489b05537ee868ffa32f6a66ad552cdea7a53c24d.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/824b4d927fa5f4ad4b32d4b489b05537ee868ffa32f6a66ad552cdea7a53c24d.jpg)

![874f4fb703874681248896749af075b89c6c4e2df4b6f1eeeda9fff648c80c3c.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/874f4fb703874681248896749af075b89c6c4e2df4b6f1eeeda9fff648c80c3c.jpg)

![94b519ec0765d932a3274acd0eb01525ff6b2aebaaee28a0b6c78794c7f5e6f6.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/94b519ec0765d932a3274acd0eb01525ff6b2aebaaee28a0b6c78794c7f5e6f6.jpg)

![97dc7a3f249bd5af469d25c9a8ee39e799c57f3cd588fe4b7b20defd12720979.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/97dc7a3f249bd5af469d25c9a8ee39e799c57f3cd588fe4b7b20defd12720979.jpg)

![c1611b05f491274eac0b7fea571d6b3514cfc302ee7c26c9988174fc335a1399.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/c1611b05f491274eac0b7fea571d6b3514cfc302ee7c26c9988174fc335a1399.jpg)

![f053a7edc3cbe008b685813b1b403163ae5c42da28755aa43c40c29501b36671.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/images/f053a7edc3cbe008b685813b1b403163ae5c42da28755aa43c40c29501b36671.jpg)

### Tables

![207dc8a1c34073fc1da3b0495cf2aac11673c70efa391a54bcbb86205211a25f.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/207dc8a1c34073fc1da3b0495cf2aac11673c70efa391a54bcbb86205211a25f.jpg)

![3a7321de79ab27a03a4d41f90181bc51b7caaa446f4badeef07d2c2092c6c734.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/3a7321de79ab27a03a4d41f90181bc51b7caaa446f4badeef07d2c2092c6c734.jpg)

![40a379a88da3ccf3329eda80130dfee2cc260f1f8195f2730a758ade1c0839e8.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/40a379a88da3ccf3329eda80130dfee2cc260f1f8195f2730a758ade1c0839e8.jpg)

![4dbdd60e2bf594c1b58bca616cfdc8641d75cc1262fb8a18d4c18cb126e88d28.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/4dbdd60e2bf594c1b58bca616cfdc8641d75cc1262fb8a18d4c18cb126e88d28.jpg)

![64a71c30977db9ff588e6e602cf87a7f8a78740b258b97f299eb04024d76ba2d.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/64a71c30977db9ff588e6e602cf87a7f8a78740b258b97f299eb04024d76ba2d.jpg)

![7604ca67d96f1cbfdfa2b0cc39f3e0986868f05633b716ae339fb40d4118bd61.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/7604ca67d96f1cbfdfa2b0cc39f3e0986868f05633b716ae339fb40d4118bd61.jpg)

![9da633601eca4a142186d7dc9705f7bc58ed12ceb71b713fd740cddc24731cee.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/9da633601eca4a142186d7dc9705f7bc58ed12ceb71b713fd740cddc24731cee.jpg)

![a1d64afb9005fbf2a587557dda5228b3b12fcdb782f63bed1f9cbe6d126f8f88.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/a1d64afb9005fbf2a587557dda5228b3b12fcdb782f63bed1f9cbe6d126f8f88.jpg)

![a2ece2ee01f3b05c86166adeae7d3e74406a8ea983a171e31ae4ec89405103c2.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/a2ece2ee01f3b05c86166adeae7d3e74406a8ea983a171e31ae4ec89405103c2.jpg)

![aa85c247dada87b9ca5137e995d75ae058ad686934b5463e8aecb1bdbda29279.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/aa85c247dada87b9ca5137e995d75ae058ad686934b5463e8aecb1bdbda29279.jpg)

![af674d5d11b80c07fb95caea5515a4cffde378d80c009c0d178d56b97457b1f3.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/af674d5d11b80c07fb95caea5515a4cffde378d80c009c0d178d56b97457b1f3.jpg)

![c4923dfdd5dd7cb11d545d0e2cd7ce289e9f52dae0cc9bdf6e28a65a036242aa.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/c4923dfdd5dd7cb11d545d0e2cd7ce289e9f52dae0cc9bdf6e28a65a036242aa.jpg)

![da8be82977f8991dbb52d5e830e918d2a2688299a83dba2e3655b5af71a254b9.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/da8be82977f8991dbb52d5e830e918d2a2688299a83dba2e3655b5af71a254b9.jpg)

![e6151e96782f463c94ed0111665d43c2b5487aeb7a01c51059adb8471eca6152.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/e6151e96782f463c94ed0111665d43c2b5487aeb7a01c51059adb8471eca6152.jpg)

![f3e5fcf6cc60ae6b7397696d287c57e4925fc92f7661a1f84cd272d6fef17714.jpg](../icml_results/3997_Reasoning%20Models%20Are%20Test%20Exploiters_%20Rethinking%20Multiple%20Choice/tables/f3e5fcf6cc60ae6b7397696d287c57e4925fc92f7661a1f84cd272d6fef17714.jpg)

<a id="paper-3957"></a>

## DISSOLVR: An Interpretable and Fast Framework for Aqueous and Organic Solubility Prediction


### Images

![096b59cfd5b6a7c29783853a26e9b66bf4eab8d02fb73ce44fdcec62f75cc9cc.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/096b59cfd5b6a7c29783853a26e9b66bf4eab8d02fb73ce44fdcec62f75cc9cc.jpg)

![21491b3bfb1e5d4fb77cba9e49ad46dbad6d0263475adf31a31c7719d88e778e.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/21491b3bfb1e5d4fb77cba9e49ad46dbad6d0263475adf31a31c7719d88e778e.jpg)

![42492922493ebe703fafcfd041ff3ea4cb196b94e3052149e1ba3cf866117a86.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/42492922493ebe703fafcfd041ff3ea4cb196b94e3052149e1ba3cf866117a86.jpg)

![4e6e05c471df768a6bfab16e1e665d4f63702015f35cae2cd20b3fc6842c44d1.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/4e6e05c471df768a6bfab16e1e665d4f63702015f35cae2cd20b3fc6842c44d1.jpg)

![84b4f7b9dada20712ee71d1294c2aeb7d7d84f5bdc1c5dcc2199b53b11f2bf38.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/84b4f7b9dada20712ee71d1294c2aeb7d7d84f5bdc1c5dcc2199b53b11f2bf38.jpg)

![cb27d50a0f4d819ce0da645fae4430854f805b01a8f720df1b5fdb536b7eb35c.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/cb27d50a0f4d819ce0da645fae4430854f805b01a8f720df1b5fdb536b7eb35c.jpg)

![cb4a8cd9aacfa2a6495200e7b23be0cac23c08b29f399d64cfd9b8610c63dd36.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/cb4a8cd9aacfa2a6495200e7b23be0cac23c08b29f399d64cfd9b8610c63dd36.jpg)

![cb6d6f2a5773cf24f507ff9214f8b3ca888b1b617a9883bd527d1d463639d5bb.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/cb6d6f2a5773cf24f507ff9214f8b3ca888b1b617a9883bd527d1d463639d5bb.jpg)

![db3aac0cb8ede03efef12dc740759e66f67ae252106a6ae0b96cc17c28f526e7.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/db3aac0cb8ede03efef12dc740759e66f67ae252106a6ae0b96cc17c28f526e7.jpg)

![f4dc1fa81bd7c402f1e165de7fe7c281f7ff2aa0d2a0fa823d1b6136305a32c8.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/f4dc1fa81bd7c402f1e165de7fe7c281f7ff2aa0d2a0fa823d1b6136305a32c8.jpg)

![fbc56c04cfb7f83875e0450ba6436233b04e309d1ea865d7d1d1f8645e8ce5c9.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/images/fbc56c04cfb7f83875e0450ba6436233b04e309d1ea865d7d1d1f8645e8ce5c9.jpg)

### Tables

![1ea79a7c13b6456cb6e076d84a7411baa8c08df69944ccc57716bc3fae037373.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/1ea79a7c13b6456cb6e076d84a7411baa8c08df69944ccc57716bc3fae037373.jpg)

![286700d52583b3b2395f680c8f2fa21fd530ad1f18b577866d3cbb37abac2784.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/286700d52583b3b2395f680c8f2fa21fd530ad1f18b577866d3cbb37abac2784.jpg)

![4f0b13224c35394eb8a2740026a81e4683afddc318845ec6a967f0969d4e1bc9.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/4f0b13224c35394eb8a2740026a81e4683afddc318845ec6a967f0969d4e1bc9.jpg)

![564d4dcc377d98c99917574bfa3fa5f3f592a40d2420018f45b87a7359188148.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/564d4dcc377d98c99917574bfa3fa5f3f592a40d2420018f45b87a7359188148.jpg)

![663605479bac54d4e9c074f6c2dc3c8ccf0903a108c39dcb0607bcbc787c4d51.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/663605479bac54d4e9c074f6c2dc3c8ccf0903a108c39dcb0607bcbc787c4d51.jpg)

![67d3c09a94c7de9f2a70a5ea4aebebdeaeb4bbf6d398c663d07771e9d145cc26.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/67d3c09a94c7de9f2a70a5ea4aebebdeaeb4bbf6d398c663d07771e9d145cc26.jpg)

![70bb14c7f8195ba8ef016d4df676705ea0d2c85e504dfc3063d6df7a97d7152a.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/70bb14c7f8195ba8ef016d4df676705ea0d2c85e504dfc3063d6df7a97d7152a.jpg)

![9d94115408468186a96aea8b9074833e4d9570ed45d0b806126aaf648720f05a.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/9d94115408468186a96aea8b9074833e4d9570ed45d0b806126aaf648720f05a.jpg)

![9e26194f56dd142b1414c5d03285af88cb1397e51c5f33febf87f40992a9edca.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/9e26194f56dd142b1414c5d03285af88cb1397e51c5f33febf87f40992a9edca.jpg)

![9e49a5a7debdc35afdd3e09f05395d81748b908a69c811338bea5c4047ee1ab4.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/9e49a5a7debdc35afdd3e09f05395d81748b908a69c811338bea5c4047ee1ab4.jpg)

![a0be25108378cea13d2e48e096d9e3c0d69a60eba7e3b6c0b31f411e40b2e8f0.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/a0be25108378cea13d2e48e096d9e3c0d69a60eba7e3b6c0b31f411e40b2e8f0.jpg)

![a2b6a728101739e0609bbb4dc39b533f8ccb2d9087fe130329404bc0ee99824e.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/a2b6a728101739e0609bbb4dc39b533f8ccb2d9087fe130329404bc0ee99824e.jpg)

![a9e954a30269eb888b033a6500789378765dd1dc3eeeab88889918e82b476970.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/a9e954a30269eb888b033a6500789378765dd1dc3eeeab88889918e82b476970.jpg)

![b88cb01fbf8cc5bc0e1efc5188ca3e2172387c8bf451d4eae99168d03434f0d4.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/b88cb01fbf8cc5bc0e1efc5188ca3e2172387c8bf451d4eae99168d03434f0d4.jpg)

![e5aa158179c7b4f2e4adbc7a7c769eb2516fadb7a896d897f62890f49dc5e8ad.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/e5aa158179c7b4f2e4adbc7a7c769eb2516fadb7a896d897f62890f49dc5e8ad.jpg)

![ecb8c78b1e291c4a17b0da1101c8e2a99f00a9a8da1e736ad42ac5ffe5861436.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/ecb8c78b1e291c4a17b0da1101c8e2a99f00a9a8da1e736ad42ac5ffe5861436.jpg)

![f2dfc644d5758d63589e0d84b36a7578a1f70d1f6ef96ea8faf014f44139100a.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/f2dfc644d5758d63589e0d84b36a7578a1f70d1f6ef96ea8faf014f44139100a.jpg)

![f6243e50a5423d00b468a4994723c922bda958247b89ea757254a8e7254a7771.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/f6243e50a5423d00b468a4994723c922bda958247b89ea757254a8e7254a7771.jpg)

![fa2983248eb9f5921b4ed05e5ef1e124d1906fed69b2d50ad3809f26c0d2af48.jpg](../icml_results/3998_DISSOLVR_%20An%20Interpretable%20and%20Fast%20Framework%20for%20Aqueous%20and%20Organic%20Solubility%20Prediction/tables/fa2983248eb9f5921b4ed05e5ef1e124d1906fed69b2d50ad3809f26c0d2af48.jpg)

<a id="paper-3958"></a>

## Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents


### Images

![001cb1d62e63d989538de79ab6fd09ebbd9c87b25527dfcf296b73b43a73fb68.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/001cb1d62e63d989538de79ab6fd09ebbd9c87b25527dfcf296b73b43a73fb68.jpg)

![1c503fe037b95b996d6c005401103c4dde5fba6bad86f1a42920017c9991ca79.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/1c503fe037b95b996d6c005401103c4dde5fba6bad86f1a42920017c9991ca79.jpg)

![27095a7e6a7117847b27d24f303491a325b38a28376d3dc5e8ba961a9e1b48f8.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/27095a7e6a7117847b27d24f303491a325b38a28376d3dc5e8ba961a9e1b48f8.jpg)

![313487910d1598b8adc116352dfafe3c961e9fa7306d3596f4717b476d39dfb9.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/313487910d1598b8adc116352dfafe3c961e9fa7306d3596f4717b476d39dfb9.jpg)

![3db67a99ace250a24f488ace19666cc4e7b5a97221910ff048577f3d8606a6bf.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/3db67a99ace250a24f488ace19666cc4e7b5a97221910ff048577f3d8606a6bf.jpg)

![3e5f193e132b91f3ee75da8c28e1154069945614fc8d0fc34f4c154cc0423a47.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/3e5f193e132b91f3ee75da8c28e1154069945614fc8d0fc34f4c154cc0423a47.jpg)

![3e7fe5b8cf20ce1fbeec4711cf5b78176e001d36fdf1e46039512ca180d9c6c8.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/3e7fe5b8cf20ce1fbeec4711cf5b78176e001d36fdf1e46039512ca180d9c6c8.jpg)

![41a9fbbb8f0b6a6a8a04476b1eacc0a2d034705ee2fd72ee6dee5b76fd57f8a1.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/41a9fbbb8f0b6a6a8a04476b1eacc0a2d034705ee2fd72ee6dee5b76fd57f8a1.jpg)

![477cd870641d136b830d002b0834641710ccdac950744517a189b929da84872b.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/477cd870641d136b830d002b0834641710ccdac950744517a189b929da84872b.jpg)

![5173140e348f5020d252413a925628a4f9f2b56360e0019202cdee4cbccbb373.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/5173140e348f5020d252413a925628a4f9f2b56360e0019202cdee4cbccbb373.jpg)

![5727e1dfc3d7d98c5588040978dea7cb8a08f19702890a32ff3881928b218dab.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/5727e1dfc3d7d98c5588040978dea7cb8a08f19702890a32ff3881928b218dab.jpg)

![573e2ba6edc756b60568beb8a0e3daf09211a1807884e94b531728040212c89a.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/573e2ba6edc756b60568beb8a0e3daf09211a1807884e94b531728040212c89a.jpg)

![62c37d43d23f9def3c6e6d70a1239049098686a05df03c204d74e548cbb2665f.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/62c37d43d23f9def3c6e6d70a1239049098686a05df03c204d74e548cbb2665f.jpg)

![68bd9cf0b94e1c9628360b13a309cd558b6995ed94e674e7a6b8c131de237fd5.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/68bd9cf0b94e1c9628360b13a309cd558b6995ed94e674e7a6b8c131de237fd5.jpg)

![699b38c6ec68a63934e6d6d55d2d077580082679d10828a95e1c8dd10438e1b6.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/699b38c6ec68a63934e6d6d55d2d077580082679d10828a95e1c8dd10438e1b6.jpg)

![6cf6adc594536fc373fd142736ed1467577a00738d57276d90287de8d6103139.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/6cf6adc594536fc373fd142736ed1467577a00738d57276d90287de8d6103139.jpg)

![723a871383618ab7830e8e4104b359537e7a9bc12566032814b4d4b8c00149a3.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/723a871383618ab7830e8e4104b359537e7a9bc12566032814b4d4b8c00149a3.jpg)

![731458ab7c1719f8733b3bb2217bc6c865046da5b9e54c9b446cb7de6ae388f6.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/731458ab7c1719f8733b3bb2217bc6c865046da5b9e54c9b446cb7de6ae388f6.jpg)

![758738e8d25ba6d636b9e511b4c90a979c4abdb3f81c87fc103d2e409581ca60.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/758738e8d25ba6d636b9e511b4c90a979c4abdb3f81c87fc103d2e409581ca60.jpg)

![7d11d991b3667cdd1a91c6de3e8310b1fc62175976e788f59daad9cdcce949b8.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/7d11d991b3667cdd1a91c6de3e8310b1fc62175976e788f59daad9cdcce949b8.jpg)

![825cb485b7ad690a31e5ef2461e662555ca4fa4f067609469e61138af8bc1ca2.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/825cb485b7ad690a31e5ef2461e662555ca4fa4f067609469e61138af8bc1ca2.jpg)

![82d22d411111e3b849631c496575be384e1998dc88b6cf78c98fd3bfbddfb063.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/82d22d411111e3b849631c496575be384e1998dc88b6cf78c98fd3bfbddfb063.jpg)

![9954e3436d8205a6ae06f69f11183a9f3fb1bd2ebe1c16494eff8b2e2f43818a.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/9954e3436d8205a6ae06f69f11183a9f3fb1bd2ebe1c16494eff8b2e2f43818a.jpg)

![a038750a009a3a14fb07a316919f3b71ddc812cfbfa763bc531f55bc0e6b14b2.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/a038750a009a3a14fb07a316919f3b71ddc812cfbfa763bc531f55bc0e6b14b2.jpg)

![ba598b196f7dfcbdce4fa052fbbf020d1e0792c1e03b1784f7767caca58f1e04.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/ba598b196f7dfcbdce4fa052fbbf020d1e0792c1e03b1784f7767caca58f1e04.jpg)

![bbe8eca24e0ffa83f1fe7ec48c7e923cbc84b2a33387834ecdad3030c08a7543.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/bbe8eca24e0ffa83f1fe7ec48c7e923cbc84b2a33387834ecdad3030c08a7543.jpg)

![bcbe574fa9af5650321604cbfc3fc93443988a47f3a9bb5a1dd416aa6f6a44d1.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/bcbe574fa9af5650321604cbfc3fc93443988a47f3a9bb5a1dd416aa6f6a44d1.jpg)

![d228ac6c5e08de7c85a3ef8157ae3bc26f87509d0eec5d37822d9a7f11ea8e6d.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/d228ac6c5e08de7c85a3ef8157ae3bc26f87509d0eec5d37822d9a7f11ea8e6d.jpg)

![e703e1934b354f464af126afe881ca524cd4ec54242e626151b68fe1a916f9f6.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/e703e1934b354f464af126afe881ca524cd4ec54242e626151b68fe1a916f9f6.jpg)

![e84a1eb07887351394f19773cacf4d68e4e1f93478baaaa46990997e5001977e.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/e84a1eb07887351394f19773cacf4d68e4e1f93478baaaa46990997e5001977e.jpg)

![fc307f00890630558f057dced67afd92d0a79d9abca881ccfed5d79552246341.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/fc307f00890630558f057dced67afd92d0a79d9abca881ccfed5d79552246341.jpg)

![fd6eaf3e4eef9972f5fae111656c73e17c3f35245ec5f8f51dc40a4c97dfe27c.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/images/fd6eaf3e4eef9972f5fae111656c73e17c3f35245ec5f8f51dc40a4c97dfe27c.jpg)

### Tables

![0483650fd0590f9b1f6608b92f78e73e64663e5206c1a10b48ef6f373ac583c2.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/0483650fd0590f9b1f6608b92f78e73e64663e5206c1a10b48ef6f373ac583c2.jpg)

![1c6b34d98f520d9fae9d691d17c039c5f5834d5946953a1848c12842f944feff.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/1c6b34d98f520d9fae9d691d17c039c5f5834d5946953a1848c12842f944feff.jpg)

![2947e414cba75bd9ca6352c3f0143573046c5e287b304c9d58e3ce30a6cca330.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/2947e414cba75bd9ca6352c3f0143573046c5e287b304c9d58e3ce30a6cca330.jpg)

![304300255331afdb6a2e22226174a8f850bd3e6093b8f78ab6528c45c2f87abb.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/304300255331afdb6a2e22226174a8f850bd3e6093b8f78ab6528c45c2f87abb.jpg)

![3d3f6870ca632dea6381489bf0d14d87bfebdf11df95fce4a1155920c497b9db.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/3d3f6870ca632dea6381489bf0d14d87bfebdf11df95fce4a1155920c497b9db.jpg)

![88dbc16675b7e6ca66579894e08158bc8aba40cc3ec33a9049e09cb0aabb27e2.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/88dbc16675b7e6ca66579894e08158bc8aba40cc3ec33a9049e09cb0aabb27e2.jpg)

![968bfce6891e5cf4624124197ebf6505560c350f27cb4e8c92847c4658c7f5d3.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/968bfce6891e5cf4624124197ebf6505560c350f27cb4e8c92847c4658c7f5d3.jpg)

![a81b47b91093c2aec33caae108635211d36432cc41b1df71653d9de2bb0eb2b1.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/a81b47b91093c2aec33caae108635211d36432cc41b1df71653d9de2bb0eb2b1.jpg)

![bc7f9362a5e887af402e99c5d9c9864578157c2e28d7c7e13e529ce648d72960.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/bc7f9362a5e887af402e99c5d9c9864578157c2e28d7c7e13e529ce648d72960.jpg)

![c61cb28b1bad7882a2c34ccbb8f697df9e0f2893619742487d6d3f02de0d8a95.jpg](../icml_results/3999_Sparks%20of%20Cooperative%20Reasoning_%20LLMs%20as%20Strategic%20Hanabi%20Agents/tables/c61cb28b1bad7882a2c34ccbb8f697df9e0f2893619742487d6d3f02de0d8a95.jpg)

<a id="paper-3959"></a>

## Multi-Task GRPO: Reliable LLM Reasoning Across Tasks


### Images

![04e5814428c6b7ea25fd73fe25fb846cf6146a567639bb42dd7920575093cb1b.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/04e5814428c6b7ea25fd73fe25fb846cf6146a567639bb42dd7920575093cb1b.jpg)

![0c50242516b74988667a08ca46752c2629c23ccab2d85f2685a58ed57adf1613.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/0c50242516b74988667a08ca46752c2629c23ccab2d85f2685a58ed57adf1613.jpg)

![1fc1f3ef597fd46a22ec36ecc15d93546c3ced4b2ed6108cd3222946678397e2.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/1fc1f3ef597fd46a22ec36ecc15d93546c3ced4b2ed6108cd3222946678397e2.jpg)

![28ef6a66cee31b80a05c58cc63af0f76511c92190565a634e801db78dd9d4699.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/28ef6a66cee31b80a05c58cc63af0f76511c92190565a634e801db78dd9d4699.jpg)

![34860476adc18c7033df7700a99203199e7bfa6eb59124b7071c8b40199aacca.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/34860476adc18c7033df7700a99203199e7bfa6eb59124b7071c8b40199aacca.jpg)

![36166f4dca293965dbe9177d539803a29b6df0fc9488200679762d28e4ae172a.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/36166f4dca293965dbe9177d539803a29b6df0fc9488200679762d28e4ae172a.jpg)

![3bc26eed91b3c7777e28954a4cb08662eb4c935860794ac16cbb82f05882c8af.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/3bc26eed91b3c7777e28954a4cb08662eb4c935860794ac16cbb82f05882c8af.jpg)

![4946bacf63cf82e9fd5c5e262e186c71fc16c398db6b22f94bacfce9919a25db.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/4946bacf63cf82e9fd5c5e262e186c71fc16c398db6b22f94bacfce9919a25db.jpg)

![4a1ea80cfeb634d8d44d991543e76c64fad1b597171d746adc595f4884fcb673.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/4a1ea80cfeb634d8d44d991543e76c64fad1b597171d746adc595f4884fcb673.jpg)

![4d6d68bbabc52ac370b6aab2722a48a59212cda015a084e3e41c0dcb1940a773.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/4d6d68bbabc52ac370b6aab2722a48a59212cda015a084e3e41c0dcb1940a773.jpg)

![573f3d226793828fee8bf9823beb8e9cdcc2af061c609dd8db94a72ab7e9e8fb.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/573f3d226793828fee8bf9823beb8e9cdcc2af061c609dd8db94a72ab7e9e8fb.jpg)

![9a96638a967ac38dec05cd31d7cd52a8f88ee75fe66af810d67a7a6b378746b6.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/9a96638a967ac38dec05cd31d7cd52a8f88ee75fe66af810d67a7a6b378746b6.jpg)

![9df1461bdb2517a642fb964271328da952ba96a8bca095c6d4d2eff610c56cc2.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/9df1461bdb2517a642fb964271328da952ba96a8bca095c6d4d2eff610c56cc2.jpg)

![a9f943d05b43d568779fe0e402d19e3388fd68dfbf081999d6b52068980fdfa1.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/a9f943d05b43d568779fe0e402d19e3388fd68dfbf081999d6b52068980fdfa1.jpg)

![ac65142432e5dd242d0a7834749ef02b46e2f1dae7f76266ba42bf19e265f171.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/ac65142432e5dd242d0a7834749ef02b46e2f1dae7f76266ba42bf19e265f171.jpg)

![c77c21b86b544a41f844e6da2c32fc8cf390cfba8c4aa4c896424ff56ee94fd3.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/c77c21b86b544a41f844e6da2c32fc8cf390cfba8c4aa4c896424ff56ee94fd3.jpg)

![c9c36461691a18f5783dbbd9464babd8c09a5fcd2746d6efc1ce6effe775fa7e.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/c9c36461691a18f5783dbbd9464babd8c09a5fcd2746d6efc1ce6effe775fa7e.jpg)

![e61b3784bc951f1665505703c4ce3768fe136f393bf7ee0d5e03410b7ba43a49.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/e61b3784bc951f1665505703c4ce3768fe136f393bf7ee0d5e03410b7ba43a49.jpg)

![ea65a2705e4b18a4dcf21297d03bae6315c6f660f22de03ec1f0d89d9c907ad4.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/ea65a2705e4b18a4dcf21297d03bae6315c6f660f22de03ec1f0d89d9c907ad4.jpg)

![fa8fe941770607b1eecafda6427b4e0b96fd953d4d6810b46f93b30faa015758.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/fa8fe941770607b1eecafda6427b4e0b96fd953d4d6810b46f93b30faa015758.jpg)

![fc127394f8844b7880d978835a5e4e32325f4e0614bfe8e972a9549d256eb7ea.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/images/fc127394f8844b7880d978835a5e4e32325f4e0614bfe8e972a9549d256eb7ea.jpg)

### Tables

![001a09c72b6f252f76bbba6ddb4978bdaf7f3f1d57aaaf3c7bbe86499efb63fa.jpg](../icml_results/4000_Multi-Task%20GRPO_%20Reliable%20LLM%20Reasoning%20Across%20Tasks/tables/001a09c72b6f252f76bbba6ddb4978bdaf7f3f1d57aaaf3c7bbe86499efb63fa.jpg)

<a id="paper-3960"></a>

## TriForces: Augmenting Atomistic GNNs for Transferable Representations

### Images

![0dc83b2afeb144e605e4d45704132a56249f76632a9c9d2f192ff677ddad47d9.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/0dc83b2afeb144e605e4d45704132a56249f76632a9c9d2f192ff677ddad47d9.jpg)

![0ee86dd44f9c8a484b31d6e58ab56b0df8d07469aaa54462dc0a60da800c3933.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/0ee86dd44f9c8a484b31d6e58ab56b0df8d07469aaa54462dc0a60da800c3933.jpg)

![1f1ff3d3ed1e5590f7ec2eaad10653a0f434c48932fd179b067d62b82ce24b3e.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/1f1ff3d3ed1e5590f7ec2eaad10653a0f434c48932fd179b067d62b82ce24b3e.jpg)

![226fce7e228797116dc8253be7c5dba0d32fa5f37f0eaad55f4ab1a761c0e3fe.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/226fce7e228797116dc8253be7c5dba0d32fa5f37f0eaad55f4ab1a761c0e3fe.jpg)

![4ca22b12fe9b62c8e1328b7cf4a97bf312c919a52701583b0b062c020af537df.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/4ca22b12fe9b62c8e1328b7cf4a97bf312c919a52701583b0b062c020af537df.jpg)

![59102f031a1bbb018276b8f90a69700bf8803fc767606c62db046783572882ae.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/59102f031a1bbb018276b8f90a69700bf8803fc767606c62db046783572882ae.jpg)

![5ffbd3073667bdaa397a8d3a3da05a8ce17954e0a71ab7de1959a3b4536896af.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/5ffbd3073667bdaa397a8d3a3da05a8ce17954e0a71ab7de1959a3b4536896af.jpg)

![8cf9cd7e371ffe6ed133f2c05b4ff363120cffa26ca04164180c175f9089abce.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/8cf9cd7e371ffe6ed133f2c05b4ff363120cffa26ca04164180c175f9089abce.jpg)

![932609f5223a9447c14f85ca57cf0ddc2183d6279728e7b0032d2b3819719830.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/932609f5223a9447c14f85ca57cf0ddc2183d6279728e7b0032d2b3819719830.jpg)

![995b1917b8bda035a2f33c49ea3d6e6ff2aec9b4d361c06a8e4e9d85d891e1fa.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/995b1917b8bda035a2f33c49ea3d6e6ff2aec9b4d361c06a8e4e9d85d891e1fa.jpg)

![b10541d01cc46c19dbd453179d2b07f9390bd549fae1804910a6afc92164c6c3.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/b10541d01cc46c19dbd453179d2b07f9390bd549fae1804910a6afc92164c6c3.jpg)

![b78ae16402c84a2d134d6bd4b3d7194d49ccd3a96652341bf570ed3da78b8dc7.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/b78ae16402c84a2d134d6bd4b3d7194d49ccd3a96652341bf570ed3da78b8dc7.jpg)

![ccc34371768e3303bad0c4ccad8108427d4a9cdae3dcc51c34a17a87285b30e2.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/ccc34371768e3303bad0c4ccad8108427d4a9cdae3dcc51c34a17a87285b30e2.jpg)

### Tables

![00b45f687a79b0a7ab4abdcfefe656fe95def6020a8215f5411f115e7b0f11ac.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/00b45f687a79b0a7ab4abdcfefe656fe95def6020a8215f5411f115e7b0f11ac.jpg)

![33f59155efd9caab58ec5b8992fa9030736b5bf9c70fae856014f624c2027282.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/33f59155efd9caab58ec5b8992fa9030736b5bf9c70fae856014f624c2027282.jpg)

![498368de8a67050458f54e1a70ef52b2350a30481b7d9f1e48adcb3f49535469.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/498368de8a67050458f54e1a70ef52b2350a30481b7d9f1e48adcb3f49535469.jpg)

![4bf87f59281172b7b145c691662d55ac03275dfa869174bd9a6ddf779e432028.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/4bf87f59281172b7b145c691662d55ac03275dfa869174bd9a6ddf779e432028.jpg)

![5166ee480e2b17a4d0f5ab50cc3e5271721aabf3eea448276006efb961cd1744.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/5166ee480e2b17a4d0f5ab50cc3e5271721aabf3eea448276006efb961cd1744.jpg)

![593fb21dbaa575b57ad3a172e3c3da2c8dd6631b88d4c50ffaa827cfca55ae07.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/593fb21dbaa575b57ad3a172e3c3da2c8dd6631b88d4c50ffaa827cfca55ae07.jpg)

![6131b45c3ab690279f2d45e7721b908165c41fdf0767c994422df0ca7cfdf1bf.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/6131b45c3ab690279f2d45e7721b908165c41fdf0767c994422df0ca7cfdf1bf.jpg)

![6980baf3b9ebe5cc04ac16e9361c7a3b36437c96b9229ef8a2152fd6a5b517d0.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/6980baf3b9ebe5cc04ac16e9361c7a3b36437c96b9229ef8a2152fd6a5b517d0.jpg)

![863f36f23bb2b5fd7e195f921af63dd14a60137f39d0bbfe8cbba8dff03ebefa.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/863f36f23bb2b5fd7e195f921af63dd14a60137f39d0bbfe8cbba8dff03ebefa.jpg)

![882a9274c1b6662b24ae6b363c98c5f40c7dbf9b3e9aa308e781ad91a50fbc37.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/882a9274c1b6662b24ae6b363c98c5f40c7dbf9b3e9aa308e781ad91a50fbc37.jpg)

![8989ce7b30fa2554a8d888fce9696bba80d4b4279cbbebce3b8b5b80e3751ff3.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/8989ce7b30fa2554a8d888fce9696bba80d4b4279cbbebce3b8b5b80e3751ff3.jpg)

![8a20104c47c9f4b4317dcdd3dd3bcd82eff587bfbd8952b5cc99d0be2f9c3f03.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/8a20104c47c9f4b4317dcdd3dd3bcd82eff587bfbd8952b5cc99d0be2f9c3f03.jpg)

![9fbeb20d3fc66a7a603fd41c55f244e4bd77c8b14a8763054e7c04af8c883c1a.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/9fbeb20d3fc66a7a603fd41c55f244e4bd77c8b14a8763054e7c04af8c883c1a.jpg)

![ad277dac398a749085e4cb45695ccfc26612dc6d7e07109ee6f66e822cd05332.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/ad277dac398a749085e4cb45695ccfc26612dc6d7e07109ee6f66e822cd05332.jpg)

![b6cf10935ffebe8cd8909c562f88cee0be5d4964f42ab45d6884731c5cf5b051.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/b6cf10935ffebe8cd8909c562f88cee0be5d4964f42ab45d6884731c5cf5b051.jpg)

![b7c83770df3b7559d0671d5ab6b9c3dd817530d395eb4a93b925a6bd2d441374.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/b7c83770df3b7559d0671d5ab6b9c3dd817530d395eb4a93b925a6bd2d441374.jpg)

![bf51826c4574641135ac0b0177bd6fc4b61e321067dad91ef7bc6c3328dfe994.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/bf51826c4574641135ac0b0177bd6fc4b61e321067dad91ef7bc6c3328dfe994.jpg)

![d93226d7ed1fcad660f2e85eacae4c898cc732f6e34b55c150b3725d77d0fd96.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/d93226d7ed1fcad660f2e85eacae4c898cc732f6e34b55c150b3725d77d0fd96.jpg)

![f43bdb56a0186c4173756058755087a1f692306413056b8df3e1e5bca9885bb3.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/f43bdb56a0186c4173756058755087a1f692306413056b8df3e1e5bca9885bb3.jpg)

![fe827174a65c1322a26ee35f49f50d37eafa77ad0522c75fc4c6feca1c54efa0.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/fe827174a65c1322a26ee35f49f50d37eafa77ad0522c75fc4c6feca1c54efa0.jpg)

![ffb21fbafb1038a4a702489b983670d4b243d63434746faaf8e70c733b063203.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/ffb21fbafb1038a4a702489b983670d4b243d63434746faaf8e70c733b063203.jpg)
