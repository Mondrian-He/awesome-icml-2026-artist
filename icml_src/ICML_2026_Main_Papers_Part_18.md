# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 18 of 197

## 目录 (Table of Contents)

1. [Minibatch selection for Language Models via Partition Matroid Constrained Gradient Matching](#paper-0562)
2. [A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets](#paper-0563)
3. [A Spiking Heterogeneous Harmonic Resonate-and-Fire State Space Model for Time Series](#paper-0564)
4. [What Does Flow-Matching Bring to TD-Learning?](#paper-0565)
5. [Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA](#paper-0566)
6. [TFRBench: A Reasoning Benchmark for Evaluating Forecasting Systems](#paper-0567)
7. [Med-SegLens: Latent-Level Model Diffing for Interpretable Medical Image Segmentation](#paper-0568)
8. [An Interactive Paradigm for Deep Research](#paper-0569)
9. [PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks](#paper-0570)
10. [PromptDyG: Test-Time Prompt Adaptation on Dynamic Graphs](#paper-0571)
11. [Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information](#paper-0572)
12. [Nested Spatio-Temporal Time Series Forecasting](#paper-0573)
13. [WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](#paper-0574)
14. [Learning to Search and Searching to Learn for Generalization in Planning](#paper-0575)
15. [PRISM: Perception Reasoning Interleaved for Sequential Decision Making.](#paper-0576)
16. [A Theoretical Framework for Statistical Evaluability of Generative Models](#paper-0577)
17. [Evaluating Contextual Illegality: AI Compliance in Corporate Law Scenarios](#paper-0578)
18. [Distributional Active Inference](#paper-0579)
19. [Compute Where it Counts: Self Optimizing Language Models](#paper-0580)
20. [Regression Language Models for Code](#paper-0581)
21. [When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](#paper-0582)
22. [Same Graph Cross-Task Transfer in GNNs: Protocols and Predictors](#paper-0583)
23. [Solving Stochastic Variational Inequalities without the Bounded Variance Assumption](#paper-0584)
24. [Set-Preserving Calibration from Conformal P-Values to E-Values](#paper-0585)
25. [Scaling Small Agents Through Strategy Auctions](#paper-0586)
26. [Attention’s forward pass and Frank-Wolfe](#paper-0587)
27. [Logical Guidance for the Exact Composition of Diffusion Models](#paper-0588)
28. [GEPC: Group-Equivariant Posterior Consistency for Out-of-Distribution Detection in Diffusion Models](#paper-0589)
29. [Improving ML Attacks on LWE with Data Repetition and Stepwise Regression](#paper-0590)
30. [Robust Strategic Classification under Decision-Dependent Cost Uncertainty](#paper-0591)
31. [Why Do We Need Warm-up? A Theoretical Perspective](#paper-0592)
32. [Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning](#paper-0593)
33. [Asymmetric conformal prediction with penalized kernel sum-of-squares](#paper-0594)

---


<a id="paper-0562"></a>

## Minibatch selection for Language Models via Partition Matroid Constrained Gradient Matching

### Images

![1f1d79a29b94bcb74823e340858128ee182e0d0793a73b142451dd209667f2f1.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/1f1d79a29b94bcb74823e340858128ee182e0d0793a73b142451dd209667f2f1.jpg)

![4a94d486f3e499638e58fa53d6b4756962b28610c9b4b2e65addac71139b6628.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/4a94d486f3e499638e58fa53d6b4756962b28610c9b4b2e65addac71139b6628.jpg)

![50ad8dc05b4f337d70dad225cda1fe2768bd8e4b1e73540d0d647cb99ab87927.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/50ad8dc05b4f337d70dad225cda1fe2768bd8e4b1e73540d0d647cb99ab87927.jpg)

![60998502a1814bfd06e6f849cd6b2e7c731b092c75749006f7ed5efd38c0dab2.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/60998502a1814bfd06e6f849cd6b2e7c731b092c75749006f7ed5efd38c0dab2.jpg)

![85289035716cd0729e3a489ff7da1ceb8c5f9c43a44af02ab88d11d88d8e6586.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/85289035716cd0729e3a489ff7da1ceb8c5f9c43a44af02ab88d11d88d8e6586.jpg)

![b4698f3486681da8533fbd8c11c5c99316fcec7435f00afb9c958185973b3d42.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/b4698f3486681da8533fbd8c11c5c99316fcec7435f00afb9c958185973b3d42.jpg)

![be3d213b5e8b7587024ccfee71a6a842e27b918636b814fe0b09ce63bfe86a82.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/be3d213b5e8b7587024ccfee71a6a842e27b918636b814fe0b09ce63bfe86a82.jpg)

![bf4ffb7d9703f6afd07250934c9f4064bfba089c76e611a35e18650bdfed2bad.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/bf4ffb7d9703f6afd07250934c9f4064bfba089c76e611a35e18650bdfed2bad.jpg)

![c1771904cc02151ac73f8c2e9cf281855f7e2f76a23405884a964b26308f1b47.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/c1771904cc02151ac73f8c2e9cf281855f7e2f76a23405884a964b26308f1b47.jpg)

![c998c47bf1bfc75f892063cae8d8067eea6b9c1cc016abf710065dcbd73566f3.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/c998c47bf1bfc75f892063cae8d8067eea6b9c1cc016abf710065dcbd73566f3.jpg)

![d764d4e92c7a17ce2d3022ff827eaec4b475da7d7a64a7a2abd147bbea321d48.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/d764d4e92c7a17ce2d3022ff827eaec4b475da7d7a64a7a2abd147bbea321d48.jpg)

![e0ce0a97dbfb8e85b8f27c85c1daab1f0d3786fd5d5288dda940f45adeb139cc.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/e0ce0a97dbfb8e85b8f27c85c1daab1f0d3786fd5d5288dda940f45adeb139cc.jpg)

![e7ec9b455e439f9ceba5e2512ed4e4e9fcda73b22910aeac56b9918898227085.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/e7ec9b455e439f9ceba5e2512ed4e4e9fcda73b22910aeac56b9918898227085.jpg)

![f4d4e7da78ffd21b258b9ff7fe3e5e77f41e84334a98142b12570f730f6b757c.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/images/f4d4e7da78ffd21b258b9ff7fe3e5e77f41e84334a98142b12570f730f6b757c.jpg)

### Tables

![29157ca130833bbcea0eb19eeabc55fb241ec47e82c506a3094c0ab7c00d1816.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/29157ca130833bbcea0eb19eeabc55fb241ec47e82c506a3094c0ab7c00d1816.jpg)

![2a650c558032c3a281a99ef88b979bde2ced10565fde78ea899cf578461bf5e2.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/2a650c558032c3a281a99ef88b979bde2ced10565fde78ea899cf578461bf5e2.jpg)

![3c75de76d4234516b33ea9fb537748a5854704947ff12a72c90dea83ff0f2555.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/3c75de76d4234516b33ea9fb537748a5854704947ff12a72c90dea83ff0f2555.jpg)

![7f90d9707f671aae07fc5d4288927e59f06ea0022663f84519fce6a25e3eda0b.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/7f90d9707f671aae07fc5d4288927e59f06ea0022663f84519fce6a25e3eda0b.jpg)

![9a564cc04dca68c630088e8a54edf3ba483142dd42b337e3a5db87db5c1844e3.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/9a564cc04dca68c630088e8a54edf3ba483142dd42b337e3a5db87db5c1844e3.jpg)

![b740a7b14e3e7dab5fec0901ad08cf903e612f787086e26bd21c75e39ca149fa.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/b740a7b14e3e7dab5fec0901ad08cf903e612f787086e26bd21c75e39ca149fa.jpg)

![c198c04073bbe7a1b26f9c1b40abafef4169071995c5170e485b8efc81711783.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/c198c04073bbe7a1b26f9c1b40abafef4169071995c5170e485b8efc81711783.jpg)

![d43b5b324c9c97cb9a10950ae4656f24e1a26f2f5d2dde292392bfdcc4f17987.jpg](../icml_results/570_Multi-Objective%20Preference%20Optimization_%20Improving%20Human%20Alignment%20of%20Generative%20Models/tables/d43b5b324c9c97cb9a10950ae4656f24e1a26f2f5d2dde292392bfdcc4f17987.jpg)

## Minibatch selection for Language Models via Partition Matroid Constrained Gradient Matching


### Images

![051acf23fe41eae5743355d567abe4a1515c0b6d77b04176ee0b75947c67e968.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/051acf23fe41eae5743355d567abe4a1515c0b6d77b04176ee0b75947c67e968.jpg)

![157ca977bf4407f8c19c0450a09b97ef6608c9387a9256588f5c2ca0c509b4f3.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/157ca977bf4407f8c19c0450a09b97ef6608c9387a9256588f5c2ca0c509b4f3.jpg)

![2925b2866c87c27b40266ea54f53aefbc4ceed8ace2cb7e50b2dbee140ebb751.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/2925b2866c87c27b40266ea54f53aefbc4ceed8ace2cb7e50b2dbee140ebb751.jpg)

![3ddada4bbc102fe1cb939c6d85516579aa37fb7a7e350aec77c06ccd783e4dfb.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/3ddada4bbc102fe1cb939c6d85516579aa37fb7a7e350aec77c06ccd783e4dfb.jpg)

![46e0c97ac82008f98526b888fec0e5b9d8c17801111726b5d860514cc6a69bca.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/46e0c97ac82008f98526b888fec0e5b9d8c17801111726b5d860514cc6a69bca.jpg)

![753c630538924d42230be68b86332466ebfdee90968024dc7b9ef736dc49e648.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/753c630538924d42230be68b86332466ebfdee90968024dc7b9ef736dc49e648.jpg)

![b66ba4e8e793afd3391f0da084cb3ed4d87efcf05c57233b22e63efcc54f6240.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/b66ba4e8e793afd3391f0da084cb3ed4d87efcf05c57233b22e63efcc54f6240.jpg)

![ba5cfa22a7238d20ffd7d48a1387b4e561a5daf060d4bfe8356cc52300bec93e.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/ba5cfa22a7238d20ffd7d48a1387b4e561a5daf060d4bfe8356cc52300bec93e.jpg)

![dc299b09d703167c97f48c82b3308929a1cf6cca42c4103bf12d953896d0e086.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/dc299b09d703167c97f48c82b3308929a1cf6cca42c4103bf12d953896d0e086.jpg)

![fa756a23ac1b20eb6fbc72197e45ca57da56356df5fb1c8a5ad877ba868140e1.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/fa756a23ac1b20eb6fbc72197e45ca57da56356df5fb1c8a5ad877ba868140e1.jpg)

![fcc9841a7ada22b7277655ae2277dd6c33b412046ca61f3929d8cd382eeae13b.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/images/fcc9841a7ada22b7277655ae2277dd6c33b412046ca61f3929d8cd382eeae13b.jpg)

### Tables

![007b9e186a1f89136717a928ddbc26a06c68ef4d4a0339fa7da3f81b3d5ca963.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/007b9e186a1f89136717a928ddbc26a06c68ef4d4a0339fa7da3f81b3d5ca963.jpg)

![0e5ff99fa9ce6a9e7978cfe3b633d07e94c540d68734af852f02f99034a4ff7f.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/0e5ff99fa9ce6a9e7978cfe3b633d07e94c540d68734af852f02f99034a4ff7f.jpg)

![20225878f88bd21821d49cd140159f55b7caa68ba83adc5438fb8d98f1369727.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/20225878f88bd21821d49cd140159f55b7caa68ba83adc5438fb8d98f1369727.jpg)

![5d9f675819db7e89bfffd8c67b023002c5be5b6214bb60728feb6e0f64dbbec0.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/5d9f675819db7e89bfffd8c67b023002c5be5b6214bb60728feb6e0f64dbbec0.jpg)

![8889970034e6db9e2bf192eba86c002d9ffba147ea586358e550fe2df4677318.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/8889970034e6db9e2bf192eba86c002d9ffba147ea586358e550fe2df4677318.jpg)

![a462238158e04fe3485b3c85404120f08af5d1287b2fc1e7270c51b3e10fb006.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/a462238158e04fe3485b3c85404120f08af5d1287b2fc1e7270c51b3e10fb006.jpg)

![a8c2ad77fa860f4a3449fd22031dd230b7bc5e97d72c76cc73a52e62a0f34f54.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/a8c2ad77fa860f4a3449fd22031dd230b7bc5e97d72c76cc73a52e62a0f34f54.jpg)

![d0acbc14fbde7df9e0421a4b7f6953e795e96d3d3dfe2008e2e656337e25d293.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/d0acbc14fbde7df9e0421a4b7f6953e795e96d3d3dfe2008e2e656337e25d293.jpg)

![ee2f2dad56a7e70e45fddfc8b23065170c72a9b3cedc385f9c233b49ec3edb0c.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/ee2f2dad56a7e70e45fddfc8b23065170c72a9b3cedc385f9c233b49ec3edb0c.jpg)

![f671291f481f72321cfad4e0c478921f9ed94f41f875be301905ec2b2fc7f2ff.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/f671291f481f72321cfad4e0c478921f9ed94f41f875be301905ec2b2fc7f2ff.jpg)

![f749d5d1ed83f6b918644c5cbdaa81e338b350ab35ff97530caf44d5abc8c3c2.jpg](../icml_results/571_Minibatch%20selection%20for%20Language%20Models%20via%20Partition%20Matroid%20Constrained%20Gradient%20Matching/tables/f749d5d1ed83f6b918644c5cbdaa81e338b350ab35ff97530caf44d5abc8c3c2.jpg)

<a id="paper-0563"></a>

## A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets


### Images

![0698474fde3760adaea5231885b48bc70a88a33340bbcf31c9415bf730deb9b5.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/0698474fde3760adaea5231885b48bc70a88a33340bbcf31c9415bf730deb9b5.jpg)

![099ba613fd6405d71001fe101aac044f43e2a8c0a4844e0e53c7b86af68ecd17.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/099ba613fd6405d71001fe101aac044f43e2a8c0a4844e0e53c7b86af68ecd17.jpg)

![0d8b9027d99992169922e6474f74102c41465df1e0bb05d5496d046b655d15fa.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/0d8b9027d99992169922e6474f74102c41465df1e0bb05d5496d046b655d15fa.jpg)

![3469fc497b1e790709de6e0ae094334b1ce3fa5930715f625a7db5ae9ab3b7f9.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/3469fc497b1e790709de6e0ae094334b1ce3fa5930715f625a7db5ae9ab3b7f9.jpg)

![35674996ba6d0f3b3c87a776d6b1cdc685279d7096718d3076ba5b9123e8f364.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/35674996ba6d0f3b3c87a776d6b1cdc685279d7096718d3076ba5b9123e8f364.jpg)

![485b08447ebb9054bee1a1b64755ab9da77a6c9b1a2a92120d00116fdd1d3c90.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/485b08447ebb9054bee1a1b64755ab9da77a6c9b1a2a92120d00116fdd1d3c90.jpg)

![5d460d2eff71409b4fd41c0de17faf990d9dfee02ecba58ea2315c08cbf2e09c.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/5d460d2eff71409b4fd41c0de17faf990d9dfee02ecba58ea2315c08cbf2e09c.jpg)

![6a928ba0bc7a3dbb0a510ce95c666f28479bfca4b966e1d8ef8cedb1ec319f57.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/6a928ba0bc7a3dbb0a510ce95c666f28479bfca4b966e1d8ef8cedb1ec319f57.jpg)

![6eaff1b07290fcf8213e7eda735c0cd4d87ff5182da5fdae79909a4d3b24e929.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/6eaff1b07290fcf8213e7eda735c0cd4d87ff5182da5fdae79909a4d3b24e929.jpg)

![75e5466c6659052ca6d8165852e207acdb2ab5ae630486b1830628e81cdba2aa.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/75e5466c6659052ca6d8165852e207acdb2ab5ae630486b1830628e81cdba2aa.jpg)

![791944e8d987adc72b28df4dfafab491c1c5d7da8c03ee2730e1895329dfac26.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/791944e8d987adc72b28df4dfafab491c1c5d7da8c03ee2730e1895329dfac26.jpg)

![a97a2e24068d45a837a00979628d9c6d59cc9fd03247395a25b9addbc4062289.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/a97a2e24068d45a837a00979628d9c6d59cc9fd03247395a25b9addbc4062289.jpg)

![bad3945caa91e32957a2c2ddb34442c0d6523ac80c82cf166583db0105d4eb8b.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/bad3945caa91e32957a2c2ddb34442c0d6523ac80c82cf166583db0105d4eb8b.jpg)

![db17b09d63ba02db0f96cc403a471827f5212ae9af975148ff6c6aee4d8dee77.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/db17b09d63ba02db0f96cc403a471827f5212ae9af975148ff6c6aee4d8dee77.jpg)

![ef38e0ae84e702e18d3308f2aa71179b1fe62b52981fd2e09f1cf7d630712082.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/ef38e0ae84e702e18d3308f2aa71179b1fe62b52981fd2e09f1cf7d630712082.jpg)

![fa28f0ed7cc4b907a2b1a9b585fd92e3af655fbd7ff64508d796434b4754fe99.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/fa28f0ed7cc4b907a2b1a9b585fd92e3af655fbd7ff64508d796434b4754fe99.jpg)

![fa9f4d1a6356791078a8bfd16e7a5aa3c0cb0bc6df82182212a5402f78ed2fcf.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/fa9f4d1a6356791078a8bfd16e7a5aa3c0cb0bc6df82182212a5402f78ed2fcf.jpg)

![ff4781a6947942a4b2f36558a533af9a32d23b8a82a6ca6e20fbeeb42ca4fd66.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/images/ff4781a6947942a4b2f36558a533af9a32d23b8a82a6ca6e20fbeeb42ca4fd66.jpg)

### Tables

![10be6015985653cb84872874de1c4fd3477d3c9e95dcc39ebfdb22e709ffbc88.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/10be6015985653cb84872874de1c4fd3477d3c9e95dcc39ebfdb22e709ffbc88.jpg)

![152a98c27d5ab77462661a1e231263c2db76e195a659451f15a525a7923e67a2.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/152a98c27d5ab77462661a1e231263c2db76e195a659451f15a525a7923e67a2.jpg)

![335cfb08b4cc27069850ccf2d85915263b24b4160945b5a5d5acbc554d5713f8.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/335cfb08b4cc27069850ccf2d85915263b24b4160945b5a5d5acbc554d5713f8.jpg)

![3df07c459384b572a2b25b733711c73c4eae85d52ce3c757ba9421639f3f2086.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/3df07c459384b572a2b25b733711c73c4eae85d52ce3c757ba9421639f3f2086.jpg)

![4325cba61da3d5f136791784eb9808e5ed660f70c6f88602f1f7dc2390d0b673.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/4325cba61da3d5f136791784eb9808e5ed660f70c6f88602f1f7dc2390d0b673.jpg)

![5e3ecfa1bb702bb8b27004486459ad75125ecc9979498a833bea2f80d9fe8e7b.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/5e3ecfa1bb702bb8b27004486459ad75125ecc9979498a833bea2f80d9fe8e7b.jpg)

![6140d74bf3b94f229308359d6816fe26b159886055d23488810aaaa5f614c3c4.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/6140d74bf3b94f229308359d6816fe26b159886055d23488810aaaa5f614c3c4.jpg)

![9a6dfaae68ee0fd327c8909e7cc60f0e8b4223db95e6abef4a091e52d01648cd.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/9a6dfaae68ee0fd327c8909e7cc60f0e8b4223db95e6abef4a091e52d01648cd.jpg)

![d5b3d732d86aef8eb83dd198238910241827e619d3098071667a7e088905c971.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/d5b3d732d86aef8eb83dd198238910241827e619d3098071667a7e088905c971.jpg)

![da2103fb164ac4e8efeaaa82b85d3e1c1a40848b3da9a1638f9edf10d540b7a1.jpg](../icml_results/572_A%20Benchmark%20and%20Framework%20for%20Evaluating%20Next%20Action%20Predictions%20in%20Spreadsheets/tables/da2103fb164ac4e8efeaaa82b85d3e1c1a40848b3da9a1638f9edf10d540b7a1.jpg)

<a id="paper-0564"></a>

## A Spiking Heterogeneous Harmonic Resonate-and-Fire State Space Model for Time Series


### Images

![297905fe44d4b0ae354fa92c744fc6e9c3a644389045ddfe07a73b1c75489c55.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/297905fe44d4b0ae354fa92c744fc6e9c3a644389045ddfe07a73b1c75489c55.jpg)

![2b35affc85b3b0a73e76d75f49cb7defd821850eac7ca683c11f6fc71cc13b79.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/2b35affc85b3b0a73e76d75f49cb7defd821850eac7ca683c11f6fc71cc13b79.jpg)

![4a5fddfaa35fc26308950a48ece743f4041c86781763d180e0042d24d9ca5930.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/4a5fddfaa35fc26308950a48ece743f4041c86781763d180e0042d24d9ca5930.jpg)

![537f193c8ca5245a327a8be5c61ce4cee5a24829cb7f6a9647a56fb60a722908.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/537f193c8ca5245a327a8be5c61ce4cee5a24829cb7f6a9647a56fb60a722908.jpg)

![6c61a1ffb15442ba6a217cef5ef795a4bc0323e0063bb78bc6661f78582793f6.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/6c61a1ffb15442ba6a217cef5ef795a4bc0323e0063bb78bc6661f78582793f6.jpg)

![7b7dc4de071e9d03d2de30fc15a6bc5b72b36422a22399023d86c5667332cb71.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/7b7dc4de071e9d03d2de30fc15a6bc5b72b36422a22399023d86c5667332cb71.jpg)

![90b48d5acb6d192187755c2e3a267520b22db27429df4ba6060c223dedbdbb2d.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/90b48d5acb6d192187755c2e3a267520b22db27429df4ba6060c223dedbdbb2d.jpg)

![bb4907853b5ab439edb9d170916979a8cc44093134d3cbafcc15565638f55645.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/bb4907853b5ab439edb9d170916979a8cc44093134d3cbafcc15565638f55645.jpg)

![e8c83724be226e1facc0851ccc95725bc340769e3b1d8a364a85b76395ede2d5.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/e8c83724be226e1facc0851ccc95725bc340769e3b1d8a364a85b76395ede2d5.jpg)

![e94e0f75bf02275440721d680a1d56a33da23c410e252895493562f20bf72e30.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/images/e94e0f75bf02275440721d680a1d56a33da23c410e252895493562f20bf72e30.jpg)

### Tables

![0d5c69a2ec4bc1cc06cda0bf2feb7ae6d9c9efb28b2ed98b6429f8f1902cc27c.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/0d5c69a2ec4bc1cc06cda0bf2feb7ae6d9c9efb28b2ed98b6429f8f1902cc27c.jpg)

![1dd78f25b70efd9630a73174bfdfb7a8f92c8c5398bdb5b0428b0c43d0940654.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/1dd78f25b70efd9630a73174bfdfb7a8f92c8c5398bdb5b0428b0c43d0940654.jpg)

![31790e646ae76e078c2f0e2bf8a97304ef73be6d30d20d075c103d87ff588605.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/31790e646ae76e078c2f0e2bf8a97304ef73be6d30d20d075c103d87ff588605.jpg)

![39dd9663a2d5f2abd620c9320a83c99cc10e9bfef92df8150f0c36899184c5ca.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/39dd9663a2d5f2abd620c9320a83c99cc10e9bfef92df8150f0c36899184c5ca.jpg)

![457c35366436ec206d7752f05de0a122fa45b9f4d2108c3bcc15294a5777bf7f.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/457c35366436ec206d7752f05de0a122fa45b9f4d2108c3bcc15294a5777bf7f.jpg)

![4b6a0cf31862f6330bbe0278274c746f0f8461cc774befecb0a5f88ad884ca7a.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/4b6a0cf31862f6330bbe0278274c746f0f8461cc774befecb0a5f88ad884ca7a.jpg)

![6b6e9b232e9932d48218a93f20f8f525ff63888202d3092a3796254005878c09.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/6b6e9b232e9932d48218a93f20f8f525ff63888202d3092a3796254005878c09.jpg)

![759ea3a0689d2a008a57298c968deb2482c47c40173f6a8d846e1f434546643d.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/759ea3a0689d2a008a57298c968deb2482c47c40173f6a8d846e1f434546643d.jpg)

![77002a909cd4ffaccc34b0ae98450aacf46f157f940c82fee45167a258dafe2e.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/77002a909cd4ffaccc34b0ae98450aacf46f157f940c82fee45167a258dafe2e.jpg)

![790b3cf36f34190a64b4d6b660cb4dd22dcb35b1f96a1faf29fbafdc68b51ef6.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/790b3cf36f34190a64b4d6b660cb4dd22dcb35b1f96a1faf29fbafdc68b51ef6.jpg)

![8bb7a1c3dcc44238a2184a275c2029b5a0aa9bcc919bf7b90ef1048ed1d7ae68.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/8bb7a1c3dcc44238a2184a275c2029b5a0aa9bcc919bf7b90ef1048ed1d7ae68.jpg)

![af37d1fe7dadc679fd48fcae910387959c58f49a47dc0e34e570e7e5ed2e3090.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/af37d1fe7dadc679fd48fcae910387959c58f49a47dc0e34e570e7e5ed2e3090.jpg)

![c34d911ae1afce9411a60a4cf5917ed566248b4a7f909e83d2a3ca7f0b25e339.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/c34d911ae1afce9411a60a4cf5917ed566248b4a7f909e83d2a3ca7f0b25e339.jpg)

![cfcb57f26167a53233dc6adffd4245bb87883974e495b17393a1e791c5062c24.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/cfcb57f26167a53233dc6adffd4245bb87883974e495b17393a1e791c5062c24.jpg)

![d4d519925af83d094963397a2e365e3149a59e61903608cbf7e4b22c5ad502c3.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/d4d519925af83d094963397a2e365e3149a59e61903608cbf7e4b22c5ad502c3.jpg)

![e5297fefc7f35134cc5124bcb269863c75b342aa210bb71c98dc0e25b6ebd49d.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/e5297fefc7f35134cc5124bcb269863c75b342aa210bb71c98dc0e25b6ebd49d.jpg)

![ed07c522db600f0b6c04c7759a5bfed6ff4e0601c4a29228723bd7a333668cea.jpg](../icml_results/573_A%20Spiking%20Heterogeneous%20Harmonic%20Resonate-and-Fire%20State%20Space%20Model%20for%20Time%20Series/tables/ed07c522db600f0b6c04c7759a5bfed6ff4e0601c4a29228723bd7a333668cea.jpg)

<a id="paper-0565"></a>

## What Does Flow-Matching Bring to TD-Learning?


### Images

![0d6e84c696224fbd91c6a8d1875b166d31ead37e6e6f70e71859ddf849009802.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/0d6e84c696224fbd91c6a8d1875b166d31ead37e6e6f70e71859ddf849009802.jpg)

![270902cdeb4a1591a51b1777eecb75023acab412d0d54783c16d102a0e644056.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/270902cdeb4a1591a51b1777eecb75023acab412d0d54783c16d102a0e644056.jpg)

![4096c6ff7b36941f1237023290062c65fd3984bf9696efb22366daed2a0925d5.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/4096c6ff7b36941f1237023290062c65fd3984bf9696efb22366daed2a0925d5.jpg)

![45fbbef5e21515cd8daeca65d2f5b3d045c06784dc384c96f144021a5a41d7fa.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/45fbbef5e21515cd8daeca65d2f5b3d045c06784dc384c96f144021a5a41d7fa.jpg)

![4f90af6fdebde80bf0f5b49aee27813b93dfef559ddf083ac94d38e7546dcac7.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/4f90af6fdebde80bf0f5b49aee27813b93dfef559ddf083ac94d38e7546dcac7.jpg)

![522784bb8aba8fe3ca180b6e7f4515f325a55186e05602f85df71a5ba5a1895e.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/522784bb8aba8fe3ca180b6e7f4515f325a55186e05602f85df71a5ba5a1895e.jpg)

![6d6f24327a52715b6aec611aba3efb75206a564ca75e68f2e6e02899f0536c7e.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/6d6f24327a52715b6aec611aba3efb75206a564ca75e68f2e6e02899f0536c7e.jpg)

![7ca610dd998be80d90cda1eb952a0dce2d82517006a86607e99d66915d5121ab.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/7ca610dd998be80d90cda1eb952a0dce2d82517006a86607e99d66915d5121ab.jpg)

![93e6235a8835c0ec156973c007142db92e1b2751606cfc8596c4e8d4b56beb2f.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/93e6235a8835c0ec156973c007142db92e1b2751606cfc8596c4e8d4b56beb2f.jpg)

![d2636af13042e6c17d1850a86f595b14f1c1f567a41e8c16ca58279d9bb67cd6.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/d2636af13042e6c17d1850a86f595b14f1c1f567a41e8c16ca58279d9bb67cd6.jpg)

![d5255dbd5ec6ae38340794829d42d24d5c9dcceced1bf7c85ccf790982b46030.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/images/d5255dbd5ec6ae38340794829d42d24d5c9dcceced1bf7c85ccf790982b46030.jpg)

### Tables

![b7afe649822aca056ecda5586ffb6431fa56691fa0ff73e0b4a7b2b908c5fe4c.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/tables/b7afe649822aca056ecda5586ffb6431fa56691fa0ff73e0b4a7b2b908c5fe4c.jpg)

![fa2ec09f555f0f078e34c70fe08cf397206d78bf198ca7b312c98041f06ec06b.jpg](../icml_results/574_What%20Does%20Flow-Matching%20Bring%20to%20TD-Learning_/tables/fa2ec09f555f0f078e34c70fe08cf397206d78bf198ca7b312c98041f06ec06b.jpg)

<a id="paper-0566"></a>

## Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA


### Images

![47eb60c19890bc4494cc89a348ff0a52bb4d595d7a4461245ed8e36f5296e8cf.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/images/47eb60c19890bc4494cc89a348ff0a52bb4d595d7a4461245ed8e36f5296e8cf.jpg)

![4eb0315b1f1babfd6116311d1b71778c49523d48d6d787c35087a707b08321bd.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/images/4eb0315b1f1babfd6116311d1b71778c49523d48d6d787c35087a707b08321bd.jpg)

![5acfedc2c7836840cd9a89d471afb2783803408061bf2836b9b3c4b1627593f8.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/images/5acfedc2c7836840cd9a89d471afb2783803408061bf2836b9b3c4b1627593f8.jpg)

![dd7b8dd9308df3882cbd1153ccf8276ea6aaa86b5a80bebb72d6bb0afbfdac4b.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/images/dd7b8dd9308df3882cbd1153ccf8276ea6aaa86b5a80bebb72d6bb0afbfdac4b.jpg)

### Tables

![995981068884e8c8c90c054cafb08a76f7fe07124c6b602a53d95dc8e52ffe2c.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/tables/995981068884e8c8c90c054cafb08a76f7fe07124c6b602a53d95dc8e52ffe2c.jpg)

![be4d8d0cabadad7848702877f63c718bcd64b1990f4c6190dc8c669e5ddf09ea.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/tables/be4d8d0cabadad7848702877f63c718bcd64b1990f4c6190dc8c669e5ddf09ea.jpg)

![dbf4bc27b957e119c179f3f78bfdc665c6ac3ed7dc7e9296432c619316889aee.jpg](../icml_results/575_Improved%20Analysis%20of%20the%20Accelerated%20Noisy%20Power%20Method%20with%20Applications%20to%20Decentralized%20PCA/tables/dbf4bc27b957e119c179f3f78bfdc665c6ac3ed7dc7e9296432c619316889aee.jpg)

<a id="paper-0567"></a>

## TFRBench: A Reasoning Benchmark for Evaluating Forecasting Systems


### Images

![12ccbb794aaed03466d8990ca9fe86220debac28c3c7732649b9fa79bf574c9b.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/12ccbb794aaed03466d8990ca9fe86220debac28c3c7732649b9fa79bf574c9b.jpg)

![1529c0f5314a723406674ce840514b8f50a3c5543b28e7b8b92a29ad8d3f8516.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/1529c0f5314a723406674ce840514b8f50a3c5543b28e7b8b92a29ad8d3f8516.jpg)

![1ee12a45c3c2e5d1f2b137851c42db9cd01b1a3388ede07decec87d1a6c26580.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/1ee12a45c3c2e5d1f2b137851c42db9cd01b1a3388ede07decec87d1a6c26580.jpg)

![2d4cd0c6156591a110aeb64b19c4388fd3b4c83d54ea36caf5f967ee8fe5e32a.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/2d4cd0c6156591a110aeb64b19c4388fd3b4c83d54ea36caf5f967ee8fe5e32a.jpg)

![2e372f580c2efd0afea24ed826c40dc5660e2606d9aac3e7c55156e4f1f8fa45.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/2e372f580c2efd0afea24ed826c40dc5660e2606d9aac3e7c55156e4f1f8fa45.jpg)

![32281ab767d90f737236d76d9775757a857d3330f4e7e5e4490bf39ac54f5f88.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/32281ab767d90f737236d76d9775757a857d3330f4e7e5e4490bf39ac54f5f88.jpg)

![32e91d570e18b8e9c011252d83a3949d8c6fcc30eae936c901816550443db3fd.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/32e91d570e18b8e9c011252d83a3949d8c6fcc30eae936c901816550443db3fd.jpg)

![3fe9f94541a0169f74317072cee3856e3c17bda39232d669de1cfeed6832b743.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/3fe9f94541a0169f74317072cee3856e3c17bda39232d669de1cfeed6832b743.jpg)

![4baabbb90f6dee0a00dbb0ec75012e54a93ee08137b9fd1c8eddd2afa67a5b3b.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/4baabbb90f6dee0a00dbb0ec75012e54a93ee08137b9fd1c8eddd2afa67a5b3b.jpg)

![54687dd9d782e0826a84b72f60d6cf330525b3eedca05c97d1a75b1afb0ff679.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/54687dd9d782e0826a84b72f60d6cf330525b3eedca05c97d1a75b1afb0ff679.jpg)

![7c73f8250d469f7bce2577506f437114efa8c464457327b3ed237b3f8787691f.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/7c73f8250d469f7bce2577506f437114efa8c464457327b3ed237b3f8787691f.jpg)

![82ecc85440d0666ed79dd915e79c9a0c68eef302c05c15304a72d5c0be68a57e.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/82ecc85440d0666ed79dd915e79c9a0c68eef302c05c15304a72d5c0be68a57e.jpg)

![927c69e8c93048362ca3c72bf0bb41de1ce20e6478d370005508f4d2b9d887ee.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/927c69e8c93048362ca3c72bf0bb41de1ce20e6478d370005508f4d2b9d887ee.jpg)

![a4b1c5d046a15035905df7b250064d53446d38de696e742aac9992ccadc13e9e.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/a4b1c5d046a15035905df7b250064d53446d38de696e742aac9992ccadc13e9e.jpg)

![b219516b16cb823c49748c733b129c604afef2bf08c42b292be73e527b780ae6.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/b219516b16cb823c49748c733b129c604afef2bf08c42b292be73e527b780ae6.jpg)

![b4880405e8d8bf468ad4f02d68db3378b6383fc5a95c4bd6f86d4fcd7b57a795.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/b4880405e8d8bf468ad4f02d68db3378b6383fc5a95c4bd6f86d4fcd7b57a795.jpg)

![b7f218ac5da7b09e020be2db0dbc758ca9814676d04330bffd1c294b6ee5e854.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/b7f218ac5da7b09e020be2db0dbc758ca9814676d04330bffd1c294b6ee5e854.jpg)

![bf6b73e80c8ce6e4cda45c886b8ddf85aff9781412dd47dedd77b3d484f0f4c1.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/bf6b73e80c8ce6e4cda45c886b8ddf85aff9781412dd47dedd77b3d484f0f4c1.jpg)

![cd9a687d76b481c2bece5239822b780061164b2ae2bda6ec4c4cf96977f7924a.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/cd9a687d76b481c2bece5239822b780061164b2ae2bda6ec4c4cf96977f7924a.jpg)

![d0dc27220fb09b1af6b29531ceecb368d295f143266b2f6c0c4e0edc5802f2de.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/d0dc27220fb09b1af6b29531ceecb368d295f143266b2f6c0c4e0edc5802f2de.jpg)

![e0086305a0fb8aa75a4177e23264131d8f1f723debf57e7b4eac81e5cc704bca.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/e0086305a0fb8aa75a4177e23264131d8f1f723debf57e7b4eac81e5cc704bca.jpg)

![e2c049165720e63b9ed4b0f25ce01fccd23fbba61bee6e2a95168e19ec3de9a5.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/e2c049165720e63b9ed4b0f25ce01fccd23fbba61bee6e2a95168e19ec3de9a5.jpg)

![e7158cd069d142668cf0ed7b2bbaea6c7709bc651d5382372f0e9969ba2b7086.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/e7158cd069d142668cf0ed7b2bbaea6c7709bc651d5382372f0e9969ba2b7086.jpg)

![ecb001814385bfdff159a62dca387f527944fb8bf65f26c8b8ea797dc46b4f0b.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/images/ecb001814385bfdff159a62dca387f527944fb8bf65f26c8b8ea797dc46b4f0b.jpg)

### Tables

![01678a72207642458fe4da2d13a02b024425c3aad4dc0636ea7428399f885258.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/01678a72207642458fe4da2d13a02b024425c3aad4dc0636ea7428399f885258.jpg)

![0f6cb29992b387457af06c47e32a6041b8cdb51c1b18fd078730a865b8ec7d44.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/0f6cb29992b387457af06c47e32a6041b8cdb51c1b18fd078730a865b8ec7d44.jpg)

![18fbc63b874cbe9e38a3c5cc47472e678390cf6e594c1c6daa4f857ae6d68d8a.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/18fbc63b874cbe9e38a3c5cc47472e678390cf6e594c1c6daa4f857ae6d68d8a.jpg)

![1bc095fcad7bfd27120fe3d81b2913fbebcf3c8b313b6b14db998cc521ff7ce3.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/1bc095fcad7bfd27120fe3d81b2913fbebcf3c8b313b6b14db998cc521ff7ce3.jpg)

![2d5b33ecaee1250835c31bfc4a6e8ef42cae9fc4c4501ae09443ac1326137c6f.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/2d5b33ecaee1250835c31bfc4a6e8ef42cae9fc4c4501ae09443ac1326137c6f.jpg)

![3d84f350e8a83186cf1aebe27b26a04b0880fd0b7dfaf57a6cf86e07e8e50044.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/3d84f350e8a83186cf1aebe27b26a04b0880fd0b7dfaf57a6cf86e07e8e50044.jpg)

![4d499376d84e186ba2ab352b6be8d0badb7c40dddd73f4c3a0fb0346b05b7225.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/4d499376d84e186ba2ab352b6be8d0badb7c40dddd73f4c3a0fb0346b05b7225.jpg)

![711b4bc7ac796828c13cd1fc4f2e55fdbc9f1a51880de8726cba9f8bdf15886f.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/711b4bc7ac796828c13cd1fc4f2e55fdbc9f1a51880de8726cba9f8bdf15886f.jpg)

![bfb8e81bb69c8bb58b4582cfa31e9224f4489f25747aa4e73780e343791a9358.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/bfb8e81bb69c8bb58b4582cfa31e9224f4489f25747aa4e73780e343791a9358.jpg)

![c7975f772ff361d0a9b087c0a2cf5ba2796e27b2338fccdca29516d371537909.jpg](../icml_results/576_TFRBench_%20A%20Reasoning%20Benchmark%20for%20Evaluating%20Forecasting%20Systems/tables/c7975f772ff361d0a9b087c0a2cf5ba2796e27b2338fccdca29516d371537909.jpg)

<a id="paper-0568"></a>

## Med-SegLens: Latent-Level Model Diffing for Interpretable Medical Image Segmentation


### Images

![13c6ac73ecb48289980dce4c6c76723a814b37430683450efca875998f3e52cc.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/13c6ac73ecb48289980dce4c6c76723a814b37430683450efca875998f3e52cc.jpg)

![3abd7f9ab44d0105aaed2861b2630118b06248e7bd038d330d140720dae88974.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/3abd7f9ab44d0105aaed2861b2630118b06248e7bd038d330d140720dae88974.jpg)

![41853051ef7fac783918ae1d58480af35961480c5156480089154f220b70d0bb.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/41853051ef7fac783918ae1d58480af35961480c5156480089154f220b70d0bb.jpg)

![5d0a5c65cb10eb3bc7f9e58bf6cde9797214fca917e8970118c545c60a5442ec.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/5d0a5c65cb10eb3bc7f9e58bf6cde9797214fca917e8970118c545c60a5442ec.jpg)

![5e4c220024a377924f98ce2a3cc27154743ebd2098a9f5df0f5958ac0bbb082b.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/5e4c220024a377924f98ce2a3cc27154743ebd2098a9f5df0f5958ac0bbb082b.jpg)

![6ee350a8f3989ce79b9d10d40ddfa806de61bc63e4e51139f15c28d935c79f60.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/6ee350a8f3989ce79b9d10d40ddfa806de61bc63e4e51139f15c28d935c79f60.jpg)

![8e30d3581f109699985713a3410480af588d4e3133b8b301a2aef9a537d150bc.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/8e30d3581f109699985713a3410480af588d4e3133b8b301a2aef9a537d150bc.jpg)

![9a072caad65e9fbb636d26255bc753222f16d9c1d56e43969f09e9dfc16dfa5f.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/9a072caad65e9fbb636d26255bc753222f16d9c1d56e43969f09e9dfc16dfa5f.jpg)

![a5c769854c6bee87bb171fe0fbdfd9cf4cff5cbbc8086e817b1174e47f7c1cd8.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/a5c769854c6bee87bb171fe0fbdfd9cf4cff5cbbc8086e817b1174e47f7c1cd8.jpg)

![ac5d5e4a5e593c29432b29d1df4ad645086c4deb9539a2ad551c6404e7738c22.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/ac5d5e4a5e593c29432b29d1df4ad645086c4deb9539a2ad551c6404e7738c22.jpg)

![bd7ee0d3da78d5c06b1746e5f0bf31512141ae93b9f135b32d6d9c628bda9098.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/bd7ee0d3da78d5c06b1746e5f0bf31512141ae93b9f135b32d6d9c628bda9098.jpg)

![cb1f01cb07674cb1a7a7cc00481b7a7c88b955b54f96f615e37d74c43f1e2b21.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/cb1f01cb07674cb1a7a7cc00481b7a7c88b955b54f96f615e37d74c43f1e2b21.jpg)

![cde4f6aa98c9bf7d9a290867fbf6fb0f296720125da8c02b3a3837f99ed26584.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/cde4f6aa98c9bf7d9a290867fbf6fb0f296720125da8c02b3a3837f99ed26584.jpg)

![dc7c2434cf4505833967da2e5b94bd767bed21eabbddbbf5103a085fb4939879.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/dc7c2434cf4505833967da2e5b94bd767bed21eabbddbbf5103a085fb4939879.jpg)

![e67d8b88f9f98219a88412b1475be852a30d5f9a5722e0e95a07d55f2f03e763.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/e67d8b88f9f98219a88412b1475be852a30d5f9a5722e0e95a07d55f2f03e763.jpg)

![eb80f94785289b75859bb82ddde5707c924b9e620f4a827b3e56aad7b4b67cd6.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/eb80f94785289b75859bb82ddde5707c924b9e620f4a827b3e56aad7b4b67cd6.jpg)

![f358f55f98e63e3488226932e4f2f7e594d2f013d26b6034ec58870dfc6c6929.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/images/f358f55f98e63e3488226932e4f2f7e594d2f013d26b6034ec58870dfc6c6929.jpg)

### Tables

![06637d5ad41dc1e20b2c5bc8ddc1095233bef434a6b45d68e7cc48dbd33e23cd.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/06637d5ad41dc1e20b2c5bc8ddc1095233bef434a6b45d68e7cc48dbd33e23cd.jpg)

![4523a93fbd91bb97819bf538a7727fa1a98a1f3362fa6a15e905aec234358c55.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/4523a93fbd91bb97819bf538a7727fa1a98a1f3362fa6a15e905aec234358c55.jpg)

![5bbe0f12a77bad10805fc6930d2782d00ed2a9e1cddde2bb975730d2d9e2a7d8.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/5bbe0f12a77bad10805fc6930d2782d00ed2a9e1cddde2bb975730d2d9e2a7d8.jpg)

![82746af5e8e99eeb41e1cc75f1ca920ed57eae57481da517110a05aa4e81f695.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/82746af5e8e99eeb41e1cc75f1ca920ed57eae57481da517110a05aa4e81f695.jpg)

![82ee7f76ce7bf312fdcb14383c96962e51a8fcf51c16c739d9337db4946b7ed2.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/82ee7f76ce7bf312fdcb14383c96962e51a8fcf51c16c739d9337db4946b7ed2.jpg)

![92daf4863cb86a8e4512c25f6ecfca46ac3e97569884efaf6de7b239804d8180.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/92daf4863cb86a8e4512c25f6ecfca46ac3e97569884efaf6de7b239804d8180.jpg)

![9b4310307aaac029259d2e96dbaedc4da38d784c06965b69f56a2035e24411bf.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/9b4310307aaac029259d2e96dbaedc4da38d784c06965b69f56a2035e24411bf.jpg)

![ba973110cb4b660fdfbd173aae3a022fe075765a8a744350d766b6f9f1b17bea.jpg](../icml_results/577_Med-SegLens_%20Latent-Level%20Model%20Diffing%20for%20Interpretable%20Medical%20Image%20Segmentation/tables/ba973110cb4b660fdfbd173aae3a022fe075765a8a744350d766b6f9f1b17bea.jpg)

<a id="paper-0569"></a>

## An Interactive Paradigm for Deep Research


### Images

![64a887c5305b7698975b168b5a75d9f46e1a6e36b58c8a0696e6f5121c04cd17.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/64a887c5305b7698975b168b5a75d9f46e1a6e36b58c8a0696e6f5121c04cd17.jpg)

![6526f5448c2431852f85b55155b37d3cc6e2bf0491a8ce2bb32af9420bb261f0.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/6526f5448c2431852f85b55155b37d3cc6e2bf0491a8ce2bb32af9420bb261f0.jpg)

![654982c5d8b0d05319223c4dce5de59c18f7d5f9d14d6bbebefe43561d9c39b5.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/654982c5d8b0d05319223c4dce5de59c18f7d5f9d14d6bbebefe43561d9c39b5.jpg)

![7638ba006e68e63b73a07886e74b7f49c94cde7925de53331e1604afecef5189.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/7638ba006e68e63b73a07886e74b7f49c94cde7925de53331e1604afecef5189.jpg)

![768ac732840e6b8f32147586e5a1fbc6547948e0e158ed1ba823f926125a294b.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/768ac732840e6b8f32147586e5a1fbc6547948e0e158ed1ba823f926125a294b.jpg)

![84e90f1c2d905206894081a8c8cb0622edf8065fbe938cb9b4a41ec86fb11383.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/84e90f1c2d905206894081a8c8cb0622edf8065fbe938cb9b4a41ec86fb11383.jpg)

![aca05f9c2c508022629a143134fab570c73b1af024415ac79bf460ad9693f380.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/aca05f9c2c508022629a143134fab570c73b1af024415ac79bf460ad9693f380.jpg)

![b184b63380bdf640e86363d52f7ac1984025fcdd0b8501c990b1205674adf366.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/b184b63380bdf640e86363d52f7ac1984025fcdd0b8501c990b1205674adf366.jpg)

![c54c1d1f0dbb17e1748914ffbf162cf659c43e3c3c414858d89d71bb25110379.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/c54c1d1f0dbb17e1748914ffbf162cf659c43e3c3c414858d89d71bb25110379.jpg)

![f28f808cbc79df431382f8c8d5d4ed2520fdafeda3e8dad8062a6fb68a4872ee.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/images/f28f808cbc79df431382f8c8d5d4ed2520fdafeda3e8dad8062a6fb68a4872ee.jpg)

### Tables

![2393125e4679fc38641e3d5f9462893f87b66603d3a3383d5225d0d059340864.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/tables/2393125e4679fc38641e3d5f9462893f87b66603d3a3383d5225d0d059340864.jpg)

![3591ebf779f7df1171716a7018f7524cb2313eb4f2234427bdb91748f0627002.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/tables/3591ebf779f7df1171716a7018f7524cb2313eb4f2234427bdb91748f0627002.jpg)

![55cd8ec58d3ee2ce996f33ea829031438aab9a6cc06019fdcabfa99450ba3ad0.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/tables/55cd8ec58d3ee2ce996f33ea829031438aab9a6cc06019fdcabfa99450ba3ad0.jpg)

![b94d4482cc33d6fbc58786caaaca581cdee79a5b288baf922840f46d3b1ce95b.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/tables/b94d4482cc33d6fbc58786caaaca581cdee79a5b288baf922840f46d3b1ce95b.jpg)

![c1b76a9e90dfd264037b410abe903722d6cd09db3640767eb2c9c8929843f129.jpg](../icml_results/578_An%20Interactive%20Paradigm%20for%20Deep%20Research/tables/c1b76a9e90dfd264037b410abe903722d6cd09db3640767eb2c9c8929843f129.jpg)

<a id="paper-0570"></a>

## PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks


### Images

![0c61d4e80d373334999b9a80f33436708980540949e5c1553797ba86bc43a030.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/images/0c61d4e80d373334999b9a80f33436708980540949e5c1553797ba86bc43a030.jpg)

![1eafdff846d0f48861e729cf794e60d6e3d1954ba35ee70203824f879110911a.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/images/1eafdff846d0f48861e729cf794e60d6e3d1954ba35ee70203824f879110911a.jpg)

![2cb58c9c396adb791dc8b0498eba0d3742aed5b8d72631ffb3746e5b282bed6b.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/images/2cb58c9c396adb791dc8b0498eba0d3742aed5b8d72631ffb3746e5b282bed6b.jpg)

![4a2218605ad8e176a98708360ed743e8fe8712c08552a127c41cf6c53f2dd2b3.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/images/4a2218605ad8e176a98708360ed743e8fe8712c08552a127c41cf6c53f2dd2b3.jpg)

### Tables

![166bef3a5a858b026ef4bce77950c2978ee96f993d712a6dc757fcf190ab6cd8.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/166bef3a5a858b026ef4bce77950c2978ee96f993d712a6dc757fcf190ab6cd8.jpg)

![1d154f801ca7b00684a752380024b494870183b37365acfa758867c7273db9da.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/1d154f801ca7b00684a752380024b494870183b37365acfa758867c7273db9da.jpg)

![23366b4be76cc1f007ef39373ca2a99807e69441aa151885debabf83bb8a137a.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/23366b4be76cc1f007ef39373ca2a99807e69441aa151885debabf83bb8a137a.jpg)

![54cb6f4df964cc369391f163dc8f591ffb9d6f17922479b2ddce6fc0c38648cc.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/54cb6f4df964cc369391f163dc8f591ffb9d6f17922479b2ddce6fc0c38648cc.jpg)

![6a5e80742e04be2e7f5e43e58e5a054814564221231b0f832f201b14d3db0859.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/6a5e80742e04be2e7f5e43e58e5a054814564221231b0f832f201b14d3db0859.jpg)

![7df2cf93f722bbda8fc305e157dd9098f3161b96348fb4258f0c06d9a35d48fc.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/7df2cf93f722bbda8fc305e157dd9098f3161b96348fb4258f0c06d9a35d48fc.jpg)

![9312164f1b2351cc9274d71ead352bd6a0fd28cf6cf2402b7abe200fcb7bbea3.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/9312164f1b2351cc9274d71ead352bd6a0fd28cf6cf2402b7abe200fcb7bbea3.jpg)

![b70feb41cbe9236cfbcf958b9cd48690d2b3891cb2a180692092617ea4fbb12a.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/b70feb41cbe9236cfbcf958b9cd48690d2b3891cb2a180692092617ea4fbb12a.jpg)

![bf20400f3e3af56591735ab24b170c53ac008a3456a7610293e76fa4625a2ee1.jpg](../icml_results/579_PASA_%20A%20Principled%20Embedding-Space%20Watermarking%20Approach%20for%20LLM-Generated%20Text%20under%20Semantic-Invar/tables/bf20400f3e3af56591735ab24b170c53ac008a3456a7610293e76fa4625a2ee1.jpg)

<a id="paper-0571"></a>

## PromptDyG: Test-Time Prompt Adaptation on Dynamic Graphs


### Images

![1645e36469ba80b6bb13a2db53c7a823c43b5abbde96627581000c1d66e09561.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/1645e36469ba80b6bb13a2db53c7a823c43b5abbde96627581000c1d66e09561.jpg)

![316c3e6cc6ce76d116c203c3795902607eaf837bb5876a64ee5c15922879402b.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/316c3e6cc6ce76d116c203c3795902607eaf837bb5876a64ee5c15922879402b.jpg)

![4e4c8466c2270f3d6e91efe9cd166e083de01d73ae1c126bdfb6bfe646e3b198.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/4e4c8466c2270f3d6e91efe9cd166e083de01d73ae1c126bdfb6bfe646e3b198.jpg)

![5f3dd3f9e92745a3bff8280dc51527fb1f3977da55deaaf7ea40b4dfa68af755.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/5f3dd3f9e92745a3bff8280dc51527fb1f3977da55deaaf7ea40b4dfa68af755.jpg)

![66acbfcf9e3c67da531e9b325b7738df5efa57b2174dd535ee183b5bd886b564.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/66acbfcf9e3c67da531e9b325b7738df5efa57b2174dd535ee183b5bd886b564.jpg)

![6f777e32a5b491a095a92c946f0d1cc662125f520bc3cf820245c532fe5f82be.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/6f777e32a5b491a095a92c946f0d1cc662125f520bc3cf820245c532fe5f82be.jpg)

![7553ba9d4ca5c7088f0060e4a25aaed2005c7f38f97faa74aab4f26d359f95d2.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/7553ba9d4ca5c7088f0060e4a25aaed2005c7f38f97faa74aab4f26d359f95d2.jpg)

![7af09032c5813c372a6fd6c8b2bc49f52046009a9b787d77229f51b9fe8b474f.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/7af09032c5813c372a6fd6c8b2bc49f52046009a9b787d77229f51b9fe8b474f.jpg)

![a99d6700a77680416843447f596867d1077a45d68af2a0fb88fce1f3b39a1d70.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/a99d6700a77680416843447f596867d1077a45d68af2a0fb88fce1f3b39a1d70.jpg)

![b1413dab30a82dee1ad029d362fc5652df896e7fc953c674ceefd6efdab3dd6b.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/b1413dab30a82dee1ad029d362fc5652df896e7fc953c674ceefd6efdab3dd6b.jpg)

![e5dcc1263db2e01511fe36df46a87373672505eeb09e0c07c8b09e061a384235.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/images/e5dcc1263db2e01511fe36df46a87373672505eeb09e0c07c8b09e061a384235.jpg)

### Tables

![1783d28a9f6442d35c53421d398e00dbfc47e8367bda21ae645368b56e7e4c7f.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/tables/1783d28a9f6442d35c53421d398e00dbfc47e8367bda21ae645368b56e7e4c7f.jpg)

![24a2302cee3d061102a612c0cc3a68bfd7cf040b5f9bf8bcea90b1ef294fc361.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/tables/24a2302cee3d061102a612c0cc3a68bfd7cf040b5f9bf8bcea90b1ef294fc361.jpg)

![3a3f3f3010033d24bb4759516a91f958bb64a8ff215585a68852ff77ac326a80.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/tables/3a3f3f3010033d24bb4759516a91f958bb64a8ff215585a68852ff77ac326a80.jpg)

![4608395c04fd6edbe52310bd78a027f8c3da71893550a3ea373f31369a3edbc6.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/tables/4608395c04fd6edbe52310bd78a027f8c3da71893550a3ea373f31369a3edbc6.jpg)

![5e55d9333b643e25cfec04fddf5c6faa392e5ecd1464465451b2a23cb7c9c160.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/tables/5e55d9333b643e25cfec04fddf5c6faa392e5ecd1464465451b2a23cb7c9c160.jpg)

![a697a192c8aab6073741816210daaf9491e55939283230a0885460c201de50b2.jpg](../icml_results/580_PromptDyG_%20Test-Time%20Prompt%20Adaptation%20on%20Dynamic%20Graphs/tables/a697a192c8aab6073741816210daaf9491e55939283230a0885460c201de50b2.jpg)

<a id="paper-0572"></a>

## Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information


### Images

![617911d86fac10d75095fcab332d9376b39801999a8c07e4455fbf0b4377e7bf.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/images/617911d86fac10d75095fcab332d9376b39801999a8c07e4455fbf0b4377e7bf.jpg)

![cb833535a7f23485fe65c0a8dd6c3bea3e880eff7face5205ebf7fb1e23ff071.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/images/cb833535a7f23485fe65c0a8dd6c3bea3e880eff7face5205ebf7fb1e23ff071.jpg)

### Tables

![04a62e1215142cfafffed97bb5d6b1a0f9f6fbb0d3098a8aa14d9e6210bfe38d.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/04a62e1215142cfafffed97bb5d6b1a0f9f6fbb0d3098a8aa14d9e6210bfe38d.jpg)

![0642900a7456d7e7b7b64a39ecf6a92c140b028162182402a588ae90df611ef8.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/0642900a7456d7e7b7b64a39ecf6a92c140b028162182402a588ae90df611ef8.jpg)

![08f04597975cbd36e4a7a03a10a8a3259774368569ee21502b611a5a1add7961.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/08f04597975cbd36e4a7a03a10a8a3259774368569ee21502b611a5a1add7961.jpg)

![18c8a7555ac2d876db176a5cdfbdaaea4a50b2776a91610cbe1cacd924da9fe5.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/18c8a7555ac2d876db176a5cdfbdaaea4a50b2776a91610cbe1cacd924da9fe5.jpg)

![2a137612fdfafc7ab1afb128c54599f38985983d072c2939bfb2d44256274c8b.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/2a137612fdfafc7ab1afb128c54599f38985983d072c2939bfb2d44256274c8b.jpg)

![39afc3ca1267e667ec07468b09bab71023b1bbe05e31c86614b11bd42f4fec0b.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/39afc3ca1267e667ec07468b09bab71023b1bbe05e31c86614b11bd42f4fec0b.jpg)

![49dd4304c15f8643ae4f4bcfb344867afcf67537cccef7dac41bc87ae57f285a.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/49dd4304c15f8643ae4f4bcfb344867afcf67537cccef7dac41bc87ae57f285a.jpg)

![537d964b0415653edbb38fd9c7310472de82800508650af863587cd09cf7b9a8.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/537d964b0415653edbb38fd9c7310472de82800508650af863587cd09cf7b9a8.jpg)

![60097c3cb7c3faab8a18c7a919228c79e325256eb6bc157fd98a9c010c9fe14b.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/60097c3cb7c3faab8a18c7a919228c79e325256eb6bc157fd98a9c010c9fe14b.jpg)

![8848c40a1933dced8711c5768dc17fab2ae3fcedc34693e2afa82f3d29583890.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/8848c40a1933dced8711c5768dc17fab2ae3fcedc34693e2afa82f3d29583890.jpg)

![943e6fb0cc17f533989881cda8cfe7f73036545a27fb7dcde161918cd3bee5f0.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/943e6fb0cc17f533989881cda8cfe7f73036545a27fb7dcde161918cd3bee5f0.jpg)

![de48d39c22c22a7dd646338c4467ce7d53e89ad15ca256596747e15fb3c90778.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/de48d39c22c22a7dd646338c4467ce7d53e89ad15ca256596747e15fb3c90778.jpg)

![f412e01c4b826301e7f1b857c97ec87ab944ae924e694fc1807632a41d8fd97d.jpg](../icml_results/581_Beyond%20Majority%20Voting_%20LLM%20Aggregation%20by%20Leveraging%20Higher-Order%20Information/tables/f412e01c4b826301e7f1b857c97ec87ab944ae924e694fc1807632a41d8fd97d.jpg)

<a id="paper-0573"></a>

## Nested Spatio-Temporal Time Series Forecasting


### Images

![0d55a2c6c2073138db8c8bc6a500686e6410a8e6795c315e63088114f0972e5f.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/images/0d55a2c6c2073138db8c8bc6a500686e6410a8e6795c315e63088114f0972e5f.jpg)

![0f18ecb8c401a70991b96e89bd02884ea4c3a2304fc7a8f66de3d8e43f692b2d.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/images/0f18ecb8c401a70991b96e89bd02884ea4c3a2304fc7a8f66de3d8e43f692b2d.jpg)

![2309ecbea0c448494ce7f87fc8f0b9386270d38b513a39b104f1321787e49398.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/images/2309ecbea0c448494ce7f87fc8f0b9386270d38b513a39b104f1321787e49398.jpg)

![826dc6184126ebb2e2b1e8c443b56e049550104abfca56d6e01756f49a33c7dc.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/images/826dc6184126ebb2e2b1e8c443b56e049550104abfca56d6e01756f49a33c7dc.jpg)

![a6493c4ec38b45fdd6cf0fe6b80eb17c40446d0309721fbfc04e789e27a1f275.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/images/a6493c4ec38b45fdd6cf0fe6b80eb17c40446d0309721fbfc04e789e27a1f275.jpg)

![f42d5c006a6bc92605e10cc215e8b5a46b779003f7b53b84e1b9ea2582af2475.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/images/f42d5c006a6bc92605e10cc215e8b5a46b779003f7b53b84e1b9ea2582af2475.jpg)

### Tables

![0b57c70051a0c865d1a86d68cf097e768da31a3d66f122b2c805492bccabe059.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/0b57c70051a0c865d1a86d68cf097e768da31a3d66f122b2c805492bccabe059.jpg)

![1150d81e3afea8d1503214eaa75ac6e568ae045d727c65897e99c9c492af8eaa.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/1150d81e3afea8d1503214eaa75ac6e568ae045d727c65897e99c9c492af8eaa.jpg)

![426aee8f79f6e2ee3db4146243d5fd879f667a25a70146e471cfad579ccc2bf9.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/426aee8f79f6e2ee3db4146243d5fd879f667a25a70146e471cfad579ccc2bf9.jpg)

![462a9296c56a9598d1ec4e67412fb1d64feadc97b2c22943394f21a49ce1ab00.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/462a9296c56a9598d1ec4e67412fb1d64feadc97b2c22943394f21a49ce1ab00.jpg)

![4812a6c0b1d423d521faa38c0350b50a89e4a0f1f45167da0f995e3d92230f30.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/4812a6c0b1d423d521faa38c0350b50a89e4a0f1f45167da0f995e3d92230f30.jpg)

![50cc9a37d614bc2c12da594853fc5cc417cc779b8571d3d0f3042715dc072573.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/50cc9a37d614bc2c12da594853fc5cc417cc779b8571d3d0f3042715dc072573.jpg)

![718b7135fbdf034754c097f5f40813bf318586d953676e39add2f4bca82865b8.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/718b7135fbdf034754c097f5f40813bf318586d953676e39add2f4bca82865b8.jpg)

![87422b3aede58307e2371ee28e3768f2e67f5907ca59f7b2875f4c8d34149d1f.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/87422b3aede58307e2371ee28e3768f2e67f5907ca59f7b2875f4c8d34149d1f.jpg)

![8796fb0f8ac7f7cc9ea89dbe54a6ac72e41958b4b9c8d128ff470e4a004e2717.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/8796fb0f8ac7f7cc9ea89dbe54a6ac72e41958b4b9c8d128ff470e4a004e2717.jpg)

![96b9fe4b035bb90ac7fcf19d252489f06e1dcbcef59bf5b9f09ce47887e10d0d.jpg](../icml_results/582_Nested%20Spatio-Temporal%20Time%20Series%20Forecasting/tables/96b9fe4b035bb90ac7fcf19d252489f06e1dcbcef59bf5b9f09ce47887e10d0d.jpg)

<a id="paper-0574"></a>

## WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling


### Images

![001068a2bee0c4450df9ec0b13d63307c3a164a95df3d6681b8d29a0feeb426d.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/001068a2bee0c4450df9ec0b13d63307c3a164a95df3d6681b8d29a0feeb426d.jpg)

![0f1bbb29959a3b094ea01580dd2cd2f97ad1cad41ef7becea5d65aece4d893b7.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/0f1bbb29959a3b094ea01580dd2cd2f97ad1cad41ef7becea5d65aece4d893b7.jpg)

![15d6229398f888f0f4981f51cb615e5c005ca8cee5c9e79ac0b9b3903e43a8b8.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/15d6229398f888f0f4981f51cb615e5c005ca8cee5c9e79ac0b9b3903e43a8b8.jpg)

![2333f7fa2321d2bf33e427dc4fe8992f4934b70e70c08e8430b737ac0bca3542.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/2333f7fa2321d2bf33e427dc4fe8992f4934b70e70c08e8430b737ac0bca3542.jpg)

![2596bb154e842f879f5382226dd5c10753377bcc5093b760d81b1d8e6be39d84.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/2596bb154e842f879f5382226dd5c10753377bcc5093b760d81b1d8e6be39d84.jpg)

![291d435f7717f2d415056adfcd718dc8ac07476f842b40e8051d0715195322d8.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/291d435f7717f2d415056adfcd718dc8ac07476f842b40e8051d0715195322d8.jpg)

![2acd6d86e02bcfc893d3b60ad90f9612a9e596b966740679eb2b41225a95c0a3.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/2acd6d86e02bcfc893d3b60ad90f9612a9e596b966740679eb2b41225a95c0a3.jpg)

![32eb676894bd8a99a357c3d0366ce052875134ab44a1137390305925478d6e6f.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/32eb676894bd8a99a357c3d0366ce052875134ab44a1137390305925478d6e6f.jpg)

![3329c739bd41121972961f21f2ea58eacfc22debf5546cc1c8103d22dfb95177.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/3329c739bd41121972961f21f2ea58eacfc22debf5546cc1c8103d22dfb95177.jpg)

![3933c82377ae75e3cbd4a69911aaca781dc3ada8ec3eee24572b363c058c296e.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/3933c82377ae75e3cbd4a69911aaca781dc3ada8ec3eee24572b363c058c296e.jpg)

![3a4d9eab46e8540fb32ac218003199c009057f19ad58c1b170913759dc74aec8.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/3a4d9eab46e8540fb32ac218003199c009057f19ad58c1b170913759dc74aec8.jpg)

![42e9fb786562b30fd04d6762e5a299925b7171d3dfdb07b24f85f78c28686880.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/42e9fb786562b30fd04d6762e5a299925b7171d3dfdb07b24f85f78c28686880.jpg)

![4696d16e4860fe1a3c3f292b5726e9f342c6dd612b53d3c97d95c8d42029e3be.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/4696d16e4860fe1a3c3f292b5726e9f342c6dd612b53d3c97d95c8d42029e3be.jpg)

![4a767c338e0794ba7b874b644516913dcb4843ed9392c308c2412a3d24866d2f.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/4a767c338e0794ba7b874b644516913dcb4843ed9392c308c2412a3d24866d2f.jpg)

![54991fa533f91c86cf1b11fda7d81debbfe3e332beb7b2d97435a6dfbc2f491e.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/54991fa533f91c86cf1b11fda7d81debbfe3e332beb7b2d97435a6dfbc2f491e.jpg)

![771c507227ba7519b3802a1fb39c2e54b20625ea939b5014ef3af2e635ca6e09.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/771c507227ba7519b3802a1fb39c2e54b20625ea939b5014ef3af2e635ca6e09.jpg)

![acf2446042244bb42c82d2015dc4b4312dc0d9f50ec0546a4af61f265e08ad42.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/acf2446042244bb42c82d2015dc4b4312dc0d9f50ec0546a4af61f265e08ad42.jpg)

![b8456915a248ab1ff861e89c1072fe59463c9d1ff994a4f2598678324405174b.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/b8456915a248ab1ff861e89c1072fe59463c9d1ff994a4f2598678324405174b.jpg)

![cf8ac836d2e605966755727243447d1427c8049e7ebbf015054351f1cc3b0ad5.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/cf8ac836d2e605966755727243447d1427c8049e7ebbf015054351f1cc3b0ad5.jpg)

![dffec7996ee1362bec84f1b93b7566feecf4a6d1ac390695e72a814474566e20.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/dffec7996ee1362bec84f1b93b7566feecf4a6d1ac390695e72a814474566e20.jpg)

![f95b02b31a645a481d03ab6e35cbfd1ab8a16704b56ba9b839e031e900e650c3.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/images/f95b02b31a645a481d03ab6e35cbfd1ab8a16704b56ba9b839e031e900e650c3.jpg)

### Tables

![293bfe5eb8ba3d963a5a192ade77d24a2e5fccba34239297b1625ecfd694dbaf.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/293bfe5eb8ba3d963a5a192ade77d24a2e5fccba34239297b1625ecfd694dbaf.jpg)

![3ef7f7237b864a1c7ba4cd6e994139b738210e74aa7bf491b0e5262a1e18f173.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/3ef7f7237b864a1c7ba4cd6e994139b738210e74aa7bf491b0e5262a1e18f173.jpg)

![3f9f0698a9b0e00bcfb3ec2bb7c59e8d768046d95d2363fd04dfc7085e22ff0a.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/3f9f0698a9b0e00bcfb3ec2bb7c59e8d768046d95d2363fd04dfc7085e22ff0a.jpg)

![4f8c28ab6fc0b22ca6e4aee984e39fea12d92b1df7f8bfc7d18e5a49719d6862.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/4f8c28ab6fc0b22ca6e4aee984e39fea12d92b1df7f8bfc7d18e5a49719d6862.jpg)

![7f0de1496b45eaf7ea43bd11dc68c8c79fd34954f366207821d456b2f2f35fb9.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/7f0de1496b45eaf7ea43bd11dc68c8c79fd34954f366207821d456b2f2f35fb9.jpg)

![cade1f0f11612451a3b392668ae04a1b9d28c7eb6d74add7dad147fe440d479f.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/cade1f0f11612451a3b392668ae04a1b9d28c7eb6d74add7dad147fe440d479f.jpg)

![d607dfe3fe9f5cd2bbe0fa9813fad6c3fbc8d31d6732634f9849a0c97c21a1cd.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/d607dfe3fe9f5cd2bbe0fa9813fad6c3fbc8d31d6732634f9849a0c97c21a1cd.jpg)

![f91ccdfea7d2baccbf0f384aa762948f6fb8e36cc9ea451c1ab19b2746b1800e.jpg](../icml_results/583_WIND_%20Weather%20Inverse%20Diffusion%20for%20Zero-Shot%20Atmospheric%20Modeling/tables/f91ccdfea7d2baccbf0f384aa762948f6fb8e36cc9ea451c1ab19b2746b1800e.jpg)

<a id="paper-0575"></a>

## Learning to Search and Searching to Learn for Generalization in Planning


### Images

![0c76fa284a5e5944a7f89001380ded6f3ed7ccace3f5879ad9a701340f3625e8.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/images/0c76fa284a5e5944a7f89001380ded6f3ed7ccace3f5879ad9a701340f3625e8.jpg)

![8585854f5f26a7b7140f6b7102ce4bc09810a1af9e4e363bcae372de2de55ca2.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/images/8585854f5f26a7b7140f6b7102ce4bc09810a1af9e4e363bcae372de2de55ca2.jpg)

![a1d433129d91d647e4f69032c5fe1c973afaf61afffc1e8695c71f0ea6493b38.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/images/a1d433129d91d647e4f69032c5fe1c973afaf61afffc1e8695c71f0ea6493b38.jpg)

### Tables

![27f2fbbd4380b6b054e51031137154a2116e94f03ca64a26771ad72572041a70.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/27f2fbbd4380b6b054e51031137154a2116e94f03ca64a26771ad72572041a70.jpg)

![532fe891c0e6efa0422f46375f8c85bc9e92cd95100067a401f5e93c25bd8b77.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/532fe891c0e6efa0422f46375f8c85bc9e92cd95100067a401f5e93c25bd8b77.jpg)

![68fe8ac7a57c644ccbb81b74672876470357bbf5e7dea823d5c02f2f75652da2.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/68fe8ac7a57c644ccbb81b74672876470357bbf5e7dea823d5c02f2f75652da2.jpg)

![acf61ca4162d77b9eac55f9c1c3457f90e7686d4851e407967a611ea65ee53c4.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/acf61ca4162d77b9eac55f9c1c3457f90e7686d4851e407967a611ea65ee53c4.jpg)

![b4293452a861f4660df4c92bb780550ac3ab4cefcc6ae8066c3b1253e9fcd39e.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/b4293452a861f4660df4c92bb780550ac3ab4cefcc6ae8066c3b1253e9fcd39e.jpg)

![c216edf697952c29778dae42e97482ced51bf536a965f053df5fe000c96ad928.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/c216edf697952c29778dae42e97482ced51bf536a965f053df5fe000c96ad928.jpg)

![dfcd8827d06e0f6627ac1ecc37d7a19fe2f9f33fb744fd11dca74fc260e7f377.jpg](../icml_results/584_Learning%20to%20Search%20and%20Searching%20to%20Learn%20for%20Generalization%20in%20Planning/tables/dfcd8827d06e0f6627ac1ecc37d7a19fe2f9f33fb744fd11dca74fc260e7f377.jpg)

<a id="paper-0576"></a>

## PRISM: Perception Reasoning Interleaved for Sequential Decision Making.


### Images

![102a40859d07a4941bc4066d0e296eb42ee12038399d5b58366a452123052452.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/102a40859d07a4941bc4066d0e296eb42ee12038399d5b58366a452123052452.jpg)

![14365986cf6aa6bc8c381d9ac3e219f5c2d0c804fbafe69715157acaf21d405d.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/14365986cf6aa6bc8c381d9ac3e219f5c2d0c804fbafe69715157acaf21d405d.jpg)

![2c6770ea251da74eacd322cf0543ba8c68c8630a050f7403182a98c8cbdfd010.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/2c6770ea251da74eacd322cf0543ba8c68c8630a050f7403182a98c8cbdfd010.jpg)

![4bd6e45c1f11f365af80dddc97707c02b49a79bddcc387f830e53d0ae2624162.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/4bd6e45c1f11f365af80dddc97707c02b49a79bddcc387f830e53d0ae2624162.jpg)

![5c4e584f5a8a77f70e4ae935a9d9f07952b31406df0a8e3bb547c7beb393e2f1.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/5c4e584f5a8a77f70e4ae935a9d9f07952b31406df0a8e3bb547c7beb393e2f1.jpg)

![8a93f8782c330fa9bba788ffc709d8587ea8eb9dc90edd0425e0a477d16aad52.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/8a93f8782c330fa9bba788ffc709d8587ea8eb9dc90edd0425e0a477d16aad52.jpg)

![9ae6faf451fe84471956b60329917f0a28f9a85d6bd31fdb04b073585d834970.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/9ae6faf451fe84471956b60329917f0a28f9a85d6bd31fdb04b073585d834970.jpg)

![a4c87417265bdb62da998c8d82fc2c4a8bac59c3250e3c85598c16316dae2988.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/a4c87417265bdb62da998c8d82fc2c4a8bac59c3250e3c85598c16316dae2988.jpg)

![bd54baf4de5102e584dfeab6cf8fdbd37d0740fe066e74d44432be03a4bb7b50.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/bd54baf4de5102e584dfeab6cf8fdbd37d0740fe066e74d44432be03a4bb7b50.jpg)

![e43d1492323c47afd92ee9d7d665fcc75aa2b8eb0a870a09e29285e3f07e67e6.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/e43d1492323c47afd92ee9d7d665fcc75aa2b8eb0a870a09e29285e3f07e67e6.jpg)

![e6115d16048f0aecc12aab542254f1af6616d9c94efe2069a640374458173776.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/e6115d16048f0aecc12aab542254f1af6616d9c94efe2069a640374458173776.jpg)

![edf122dc43ba93b96088151ec4d1f0dadf2b245dc78bf446d57828680c7d1aff.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/edf122dc43ba93b96088151ec4d1f0dadf2b245dc78bf446d57828680c7d1aff.jpg)

![f4941f71154cfae1be3398a912119ff94c16506db100315f7373b3badd69cb9f.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./images/f4941f71154cfae1be3398a912119ff94c16506db100315f7373b3badd69cb9f.jpg)

### Tables

![1c89b1409d5c4ef64da59a80e7de3e996d53b821c44c9bb770661c58399b3178.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/1c89b1409d5c4ef64da59a80e7de3e996d53b821c44c9bb770661c58399b3178.jpg)

![389b655378c084cbdefedc7127c45bc099eb6d022f8386851082e4a93ea28259.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/389b655378c084cbdefedc7127c45bc099eb6d022f8386851082e4a93ea28259.jpg)

![546acc83a9020f60411722e37b8046acb3a88b65a751f8c2a54660ee47d51eef.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/546acc83a9020f60411722e37b8046acb3a88b65a751f8c2a54660ee47d51eef.jpg)

![5c89d4944a53d13d1eb600c9765b359a8499344ea3f4f83e9b21ef77e275195b.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/5c89d4944a53d13d1eb600c9765b359a8499344ea3f4f83e9b21ef77e275195b.jpg)

![67cf059794b20ed3c73de20af2917088ebeab9a9d06c932f851272a498a6a0ca.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/67cf059794b20ed3c73de20af2917088ebeab9a9d06c932f851272a498a6a0ca.jpg)

![794fb9776da614991f7ab1078d682f1a415f26678fdd7a40e5a4ac23e184c17e.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/794fb9776da614991f7ab1078d682f1a415f26678fdd7a40e5a4ac23e184c17e.jpg)

![7f1fccdae5404a0d36cdbd2a9552407d7e5b076eed09cbb405cc11ff7c15a292.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/7f1fccdae5404a0d36cdbd2a9552407d7e5b076eed09cbb405cc11ff7c15a292.jpg)

![7fd09eeaa17d55f3fb738250dc976c45cc9f99817cad049badd92c95697fedf8.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/7fd09eeaa17d55f3fb738250dc976c45cc9f99817cad049badd92c95697fedf8.jpg)

![80c417163f6e6a6c6a382481fa02b258f7405ecb23150454d339a5df49015735.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/80c417163f6e6a6c6a382481fa02b258f7405ecb23150454d339a5df49015735.jpg)

![e38df3b254fe115639ef17116f55fd2863235b3a18a0510baa449555925a60b6.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/e38df3b254fe115639ef17116f55fd2863235b3a18a0510baa449555925a60b6.jpg)

![efe9b3706404277d761d752573312461ab3cc1d42bbe4159b479a4e4272d4bbc.jpg](../icml_results/585_PRISM_%20Perception%20Reasoning%20Interleaved%20for%20Sequential%20Decision%20Making./tables/efe9b3706404277d761d752573312461ab3cc1d42bbe4159b479a4e4272d4bbc.jpg)

<a id="paper-0577"></a>

## A Theoretical Framework for Statistical Evaluability of Generative Models


### Tables

![4d034ff3b9affcf2521bf25303a16d88ed221ad364d09ed00e41cfe336911eb4.jpg](../icml_results/586_A%20Theoretical%20Framework%20for%20Statistical%20Evaluability%20of%20Generative%20Models/tables/4d034ff3b9affcf2521bf25303a16d88ed221ad364d09ed00e41cfe336911eb4.jpg)

<a id="paper-0578"></a>

## Evaluating Contextual Illegality: AI Compliance in Corporate Law Scenarios


### Images

![1aa9e8dbd0bcb4935565b5f35cf19dfaa21f0258116f3f8e67e41906c1092e8c.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/1aa9e8dbd0bcb4935565b5f35cf19dfaa21f0258116f3f8e67e41906c1092e8c.jpg)

![3aa6e36e07b5714fced7727d56103dfce8b8ef2b0a00069516ce779a06470173.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/3aa6e36e07b5714fced7727d56103dfce8b8ef2b0a00069516ce779a06470173.jpg)

![3b71f75c94f9538feff4e5dff92b3769c00da2da598c0bcf7df1193d2acf0f96.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/3b71f75c94f9538feff4e5dff92b3769c00da2da598c0bcf7df1193d2acf0f96.jpg)

![45388b17b7204868c5a5a72c99e31fc91987c18b4e49f70f04dd2b48f8f12418.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/45388b17b7204868c5a5a72c99e31fc91987c18b4e49f70f04dd2b48f8f12418.jpg)

![48d76e6121d9e511f8ee8f351b20297b3646abcaf50407e197e2c4dff2b69fad.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/48d76e6121d9e511f8ee8f351b20297b3646abcaf50407e197e2c4dff2b69fad.jpg)

![4dacbdc1d8d4316d1093b1a62c453846bf86a1a0920c218f5846cab10a132708.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/4dacbdc1d8d4316d1093b1a62c453846bf86a1a0920c218f5846cab10a132708.jpg)

![6d314cac86e76486fcdf9288fbc9b064d0b99d497c5b645ae2dd2cb953de97e7.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/6d314cac86e76486fcdf9288fbc9b064d0b99d497c5b645ae2dd2cb953de97e7.jpg)

![9680630af4fc277a6b535ecce38f3522229f6867384273a34803f0c8a7d72528.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/9680630af4fc277a6b535ecce38f3522229f6867384273a34803f0c8a7d72528.jpg)

![9699e9eef52f1d666fcf9b1364c354e2a4e9d54a609dfffc806118b40678686f.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/9699e9eef52f1d666fcf9b1364c354e2a4e9d54a609dfffc806118b40678686f.jpg)

![9e51bab035117b96d80fa5b2b28388df3538d2463435ab64db7fa982da316702.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/9e51bab035117b96d80fa5b2b28388df3538d2463435ab64db7fa982da316702.jpg)

![b214b49e9f78a015902c4d1653f05b0f023a72c66a68cb2a17b6f9a355b45a98.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/b214b49e9f78a015902c4d1653f05b0f023a72c66a68cb2a17b6f9a355b45a98.jpg)

![b9af4f249eb25abe697f3aff90c10e23e7c8fc5d8cc068e1095a330a49aeda83.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/b9af4f249eb25abe697f3aff90c10e23e7c8fc5d8cc068e1095a330a49aeda83.jpg)

![ca1a68a05044094232b9d515afff9f0bc31d0b1df296ff5db41e20d46c74e343.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/ca1a68a05044094232b9d515afff9f0bc31d0b1df296ff5db41e20d46c74e343.jpg)

![ea33de55c09cc44d06511f2469511e8dc970c922908ffe7f44ff2813cf3e788a.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/ea33de55c09cc44d06511f2469511e8dc970c922908ffe7f44ff2813cf3e788a.jpg)

![eba3b0a2bb860bc93f335440f57ca497b9684ddade7fbb3c12213c0737a7e819.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/eba3b0a2bb860bc93f335440f57ca497b9684ddade7fbb3c12213c0737a7e819.jpg)

![f0e09071e93492b496bb5180755dc353d1747d267605726f0cc68ef736a56e90.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/f0e09071e93492b496bb5180755dc353d1747d267605726f0cc68ef736a56e90.jpg)

![f51b8601a4cb09213003833023b78e20a630f25548cd817b7ca94ba41f09baea.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/images/f51b8601a4cb09213003833023b78e20a630f25548cd817b7ca94ba41f09baea.jpg)

### Tables

![1f7ade3426a14edec8d86976c57edc2fdcc1743760845c8626708262dff2636d.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/1f7ade3426a14edec8d86976c57edc2fdcc1743760845c8626708262dff2636d.jpg)

![21314dcf42171d95168879f9c435751101455fe8f62eba91646f8dbba7629dbe.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/21314dcf42171d95168879f9c435751101455fe8f62eba91646f8dbba7629dbe.jpg)

![23ff6708e16dc97cefb38741164b0138fa56bcbe4862b64df63a892cc730e7a9.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/23ff6708e16dc97cefb38741164b0138fa56bcbe4862b64df63a892cc730e7a9.jpg)

![2945d9aadb66f95bf059ca46496ee834e4741aee878d509453f3bfcdcdeb9a35.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/2945d9aadb66f95bf059ca46496ee834e4741aee878d509453f3bfcdcdeb9a35.jpg)

![2b3b7ef0ef9bcd914ad73ca0a3bd4d0150089dcdca2c8b68bdd4f83527bf48c6.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/2b3b7ef0ef9bcd914ad73ca0a3bd4d0150089dcdca2c8b68bdd4f83527bf48c6.jpg)

![30a7f2a398c1e56e23cfab70d926cbb01f57e68234159b34db579a098177fe09.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/30a7f2a398c1e56e23cfab70d926cbb01f57e68234159b34db579a098177fe09.jpg)

![3bd7d32b12bfc86d48d52af8b26c8b5dd5b3face817e29d0cb62c91c36f019f5.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/3bd7d32b12bfc86d48d52af8b26c8b5dd5b3face817e29d0cb62c91c36f019f5.jpg)

![408ed62226ec6fe0afcdeaedc911bec270977736880c7a4f2a2c4deb1601d3a4.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/408ed62226ec6fe0afcdeaedc911bec270977736880c7a4f2a2c4deb1601d3a4.jpg)

![68db2292c9adf06414475e387003bdd88a5e0b9e42e0ea22499462d2679d1738.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/68db2292c9adf06414475e387003bdd88a5e0b9e42e0ea22499462d2679d1738.jpg)

![6ac5d06b573eda21fc8a8d478ccc3c16080e471c3686c5d220b1721da2399209.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/6ac5d06b573eda21fc8a8d478ccc3c16080e471c3686c5d220b1721da2399209.jpg)

![7647fd04c67fafca4fc434e61528eaee5bc289b3fe7d9b74c62594e379cdf488.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/7647fd04c67fafca4fc434e61528eaee5bc289b3fe7d9b74c62594e379cdf488.jpg)

![885947b7912665c18c8ae14412026a8fabe47a6de19988e0db4bc4c3420bb4b6.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/885947b7912665c18c8ae14412026a8fabe47a6de19988e0db4bc4c3420bb4b6.jpg)

![8d6bcbbda1f3c32944d262096e59cd4dc525c82b4090bba7edb7ccd145129e4a.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/8d6bcbbda1f3c32944d262096e59cd4dc525c82b4090bba7edb7ccd145129e4a.jpg)

![a39bf499fae76003ebdc1a22023437d53ca71e35762b1254263cae05390baab8.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/a39bf499fae76003ebdc1a22023437d53ca71e35762b1254263cae05390baab8.jpg)

![c7f63feb54800c1ef9559a92e0e37e19e436fa7bfd20fc74e8e8d3d279dcc352.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/c7f63feb54800c1ef9559a92e0e37e19e436fa7bfd20fc74e8e8d3d279dcc352.jpg)

![f6476f0bbde86e016f456adb3aafe7721c432e986edef93bc70fabc8a7b4f919.jpg](../icml_results/587_Evaluating%20Contextual%20Illegality_%20AI%20Compliance%20in%20Corporate%20Law%20Scenarios/tables/f6476f0bbde86e016f456adb3aafe7721c432e986edef93bc70fabc8a7b4f919.jpg)

<a id="paper-0579"></a>

## Distributional Active Inference


### Images

![01263765e0ab97d5b90cdb06230b8d5d438e9e3f2c87a91388ff745ba6787ed8.jpg](../icml_results/588_Distributional%20Active%20Inference/images/01263765e0ab97d5b90cdb06230b8d5d438e9e3f2c87a91388ff745ba6787ed8.jpg)

![133c35ea0fba52a65262e29de19f187ae9d862478f2dce7c2e07af8163d1da5b.jpg](../icml_results/588_Distributional%20Active%20Inference/images/133c35ea0fba52a65262e29de19f187ae9d862478f2dce7c2e07af8163d1da5b.jpg)

![3ad5b7e2afd654723457e7292c7fc7723035973b61723f4dd3e35275aa123710.jpg](../icml_results/588_Distributional%20Active%20Inference/images/3ad5b7e2afd654723457e7292c7fc7723035973b61723f4dd3e35275aa123710.jpg)

![4c64818c26af8075e7c2bb17f9a5c8ee4a18ebdc15ad85a90f6c7ba553c59166.jpg](../icml_results/588_Distributional%20Active%20Inference/images/4c64818c26af8075e7c2bb17f9a5c8ee4a18ebdc15ad85a90f6c7ba553c59166.jpg)

![752a4e01e8627aa7bd66adcf3350f34f38d5d925cd41b8bba271f4b8a6aba484.jpg](../icml_results/588_Distributional%20Active%20Inference/images/752a4e01e8627aa7bd66adcf3350f34f38d5d925cd41b8bba271f4b8a6aba484.jpg)

![845e64828fe400956f947f4fecda01be8b95e6ee5b8cb0d11b1d764b27736418.jpg](../icml_results/588_Distributional%20Active%20Inference/images/845e64828fe400956f947f4fecda01be8b95e6ee5b8cb0d11b1d764b27736418.jpg)

![b6f554d8aad11ee298950dcff7311253aa2394c65082670c095457b888eff71f.jpg](../icml_results/588_Distributional%20Active%20Inference/images/b6f554d8aad11ee298950dcff7311253aa2394c65082670c095457b888eff71f.jpg)

![d9267aab5cd4aea182aa3eb9c3d01637e7f134bd72c5cddb541205c93603c9e6.jpg](../icml_results/588_Distributional%20Active%20Inference/images/d9267aab5cd4aea182aa3eb9c3d01637e7f134bd72c5cddb541205c93603c9e6.jpg)

![e53851a3e56849fa76e3fe300f865518939907a8b48c4a46d49c22b8656be587.jpg](../icml_results/588_Distributional%20Active%20Inference/images/e53851a3e56849fa76e3fe300f865518939907a8b48c4a46d49c22b8656be587.jpg)

### Tables

![24ea19d2eab0309782b6fb576b0bb034948cd29ae08ed4e406ad5b1976062ab2.jpg](../icml_results/588_Distributional%20Active%20Inference/tables/24ea19d2eab0309782b6fb576b0bb034948cd29ae08ed4e406ad5b1976062ab2.jpg)

![26a911140ac2548cf7889380c26f39be0916e1ed3ee393ea0fd86afe3466624d.jpg](../icml_results/588_Distributional%20Active%20Inference/tables/26a911140ac2548cf7889380c26f39be0916e1ed3ee393ea0fd86afe3466624d.jpg)

![629217e9020143e245e9237b205e34a94500fe3b32c1d0244322eb3eaac665f5.jpg](../icml_results/588_Distributional%20Active%20Inference/tables/629217e9020143e245e9237b205e34a94500fe3b32c1d0244322eb3eaac665f5.jpg)

![7034e5f6cc4d2368e66b8ecb2833bc964c8a8fdee0130a68ac31faf57c6bf366.jpg](../icml_results/588_Distributional%20Active%20Inference/tables/7034e5f6cc4d2368e66b8ecb2833bc964c8a8fdee0130a68ac31faf57c6bf366.jpg)

![7457779b1e3fd328ef13e34d9b5ab41cf14849f6db28076f6e061502550ed85f.jpg](../icml_results/588_Distributional%20Active%20Inference/tables/7457779b1e3fd328ef13e34d9b5ab41cf14849f6db28076f6e061502550ed85f.jpg)

![e9e8d2f3d396912db2366ee22a16ec187ff9410af86fb001e8f45540c3394b76.jpg](../icml_results/588_Distributional%20Active%20Inference/tables/e9e8d2f3d396912db2366ee22a16ec187ff9410af86fb001e8f45540c3394b76.jpg)

<a id="paper-0580"></a>

## Compute Where it Counts: Self Optimizing Language Models


### Images

![0c7f7946fc1a25953f425049049d6b5d5a1d918736a530849ca782320f2996b3.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/0c7f7946fc1a25953f425049049d6b5d5a1d918736a530849ca782320f2996b3.jpg)

![17cfcbbf139d92ba4129cc6b31c7e0e469e379839bcfe7ac1285e4244e73ed27.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/17cfcbbf139d92ba4129cc6b31c7e0e469e379839bcfe7ac1285e4244e73ed27.jpg)

![43c4055ae14cd3fe2050c4a30ba18cc9a56de29338a89a3021351969b34ffde7.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/43c4055ae14cd3fe2050c4a30ba18cc9a56de29338a89a3021351969b34ffde7.jpg)

![4c06717e2fe402e9e742e16021776bfad2f0b0d66488e6c93f1486b4d9ae81f8.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/4c06717e2fe402e9e742e16021776bfad2f0b0d66488e6c93f1486b4d9ae81f8.jpg)

![5373a59bf76ed038165657f0e3031a392ecb6df358015049d5c03f1653d7301a.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/5373a59bf76ed038165657f0e3031a392ecb6df358015049d5c03f1653d7301a.jpg)

![8951b791a373c5bdf86265b4112515b041044d1bbe81c60bbd8933f4f0c791f0.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/8951b791a373c5bdf86265b4112515b041044d1bbe81c60bbd8933f4f0c791f0.jpg)

![a78ebfc7f721fa0c58d12a32517cc9dd1fe8a83422e40e69dc39012be316e30b.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/a78ebfc7f721fa0c58d12a32517cc9dd1fe8a83422e40e69dc39012be316e30b.jpg)

![c54115133263d2d9da225b60ee367f675b06160c59f63ac14af495be104b88cd.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/c54115133263d2d9da225b60ee367f675b06160c59f63ac14af495be104b88cd.jpg)

![ce282608f7173b9a77c1b7466dec2dffec5f00567e525d63ac857a36b9551f1a.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/ce282608f7173b9a77c1b7466dec2dffec5f00567e525d63ac857a36b9551f1a.jpg)

![d3688afeda8262e828dddaf6e7daca165edd80f6918c336ba976ec776d2678f9.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/d3688afeda8262e828dddaf6e7daca165edd80f6918c336ba976ec776d2678f9.jpg)

![ea22dcc49d17e053d6273062742505091e7bbeb9a78cc89190f1edb92948ffd9.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/images/ea22dcc49d17e053d6273062742505091e7bbeb9a78cc89190f1edb92948ffd9.jpg)

### Tables

![2e084365e691b595cb17da09402baf330cafee04cda9132e253676379ecd675f.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/tables/2e084365e691b595cb17da09402baf330cafee04cda9132e253676379ecd675f.jpg)

![55a348b1fc3779119076a515d5f804d79ef145fbfd712333a3f9bc15423b1138.jpg](../icml_results/589_Compute%20Where%20it%20Counts_%20Self%20Optimizing%20Language%20Models/tables/55a348b1fc3779119076a515d5f804d79ef145fbfd712333a3f9bc15423b1138.jpg)

<a id="paper-0581"></a>

## Regression Language Models for Code


### Images

![1983a3a5036db02c17f5865da8dd55039511c15e99ba92aaa9f0424376472545.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/1983a3a5036db02c17f5865da8dd55039511c15e99ba92aaa9f0424376472545.jpg)

![1baaa9b7315c398a343c391ec5776aa814078fb1956d5993fab879d77658881b.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/1baaa9b7315c398a343c391ec5776aa814078fb1956d5993fab879d77658881b.jpg)

![20e181a2042c099ffc65621ab4e455cf28b4840246c21c5f70c6fd9363125a2d.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/20e181a2042c099ffc65621ab4e455cf28b4840246c21c5f70c6fd9363125a2d.jpg)

![3001b612a73637df36fe60a8d9a4e3d1951abd5039e7aa6f3499f5cf028bd7b5.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/3001b612a73637df36fe60a8d9a4e3d1951abd5039e7aa6f3499f5cf028bd7b5.jpg)

![3e201454de803605b46c4b3f5c9b93d7abf9af97bda83141c4770dee0aa3ee2c.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/3e201454de803605b46c4b3f5c9b93d7abf9af97bda83141c4770dee0aa3ee2c.jpg)

![46056c749f8796266b52bc64200dce004f50d8021fff9a8858789c46d9022b87.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/46056c749f8796266b52bc64200dce004f50d8021fff9a8858789c46d9022b87.jpg)

![53a5424ed9382c99a2d51b7f42aea020666e2f1bdcd667aeea69a56cdee3dabb.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/53a5424ed9382c99a2d51b7f42aea020666e2f1bdcd667aeea69a56cdee3dabb.jpg)

![6457128731988a6ba1990f6229c4dcc724d8f57f0042769c8abafc4008c5061a.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/6457128731988a6ba1990f6229c4dcc724d8f57f0042769c8abafc4008c5061a.jpg)

![67518c054c5e24b61dfe80674bae31e60345977dcaad62a9fcda9727eae6ddef.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/67518c054c5e24b61dfe80674bae31e60345977dcaad62a9fcda9727eae6ddef.jpg)

![a9f68177cfc1c19b21ef8c9874f36e63ff53aeb655a405bc027f09c54090f72a.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/a9f68177cfc1c19b21ef8c9874f36e63ff53aeb655a405bc027f09c54090f72a.jpg)

![b9e96db9f48b35e1ed72f089fe919513945b9aaf94a82fa8c0f17a71bb0f36f9.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/b9e96db9f48b35e1ed72f089fe919513945b9aaf94a82fa8c0f17a71bb0f36f9.jpg)

![d463c69a393b53aec3022f7abaaa3936d81c1f02f37c222530d0687e7ebf4eb7.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/d463c69a393b53aec3022f7abaaa3936d81c1f02f37c222530d0687e7ebf4eb7.jpg)

![d4ddcd5fd2eb19d515d8a6b9cf2b21f6c90dfc9749e6ebbf6dbf14bae33aa32e.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/d4ddcd5fd2eb19d515d8a6b9cf2b21f6c90dfc9749e6ebbf6dbf14bae33aa32e.jpg)

![fc923bdf8e8c7070244b7ee6c18ef95f6ffa207d0743f0dd43410958e4ce1da6.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/fc923bdf8e8c7070244b7ee6c18ef95f6ffa207d0743f0dd43410958e4ce1da6.jpg)

![fe5e4a11822f0b6b921d49f41249bc31c68582891a14e2397c2d7a6e39ef6640.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/images/fe5e4a11822f0b6b921d49f41249bc31c68582891a14e2397c2d7a6e39ef6640.jpg)

### Tables

![20dbb1fa0edc3f34584c16b704428bd904e70ae9069f4cf2822ee4de296f8140.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/20dbb1fa0edc3f34584c16b704428bd904e70ae9069f4cf2822ee4de296f8140.jpg)

![263d79bac7b3d5d571e7eb25c8c66c4c24cbfbaa665690045e27197d43b96c3a.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/263d79bac7b3d5d571e7eb25c8c66c4c24cbfbaa665690045e27197d43b96c3a.jpg)

![4a14dc8e9602d07779ebb2cf6915597b52b9390afa0b76b77aa4e35e4e16a0e7.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/4a14dc8e9602d07779ebb2cf6915597b52b9390afa0b76b77aa4e35e4e16a0e7.jpg)

![69ef9bd6f2fcc7397eb19aab0e14f05e2a53b11518a6e737651bcbd5bfb3c43f.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/69ef9bd6f2fcc7397eb19aab0e14f05e2a53b11518a6e737651bcbd5bfb3c43f.jpg)

![7b00f172abe8a6d6f74f10d6c556c7baca6b72ad3edaa911e3c386e98c0830d8.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/7b00f172abe8a6d6f74f10d6c556c7baca6b72ad3edaa911e3c386e98c0830d8.jpg)

![a4337c9f29d16bbd4ee2d47c6d8d07e87e31696c17e6ab6c3d8ddf2106d089ee.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/a4337c9f29d16bbd4ee2d47c6d8d07e87e31696c17e6ab6c3d8ddf2106d089ee.jpg)

![af4d248c9830f9fcd6d692c462cc04b9754d470cafbf163b48e7e8dfb44a32b2.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/af4d248c9830f9fcd6d692c462cc04b9754d470cafbf163b48e7e8dfb44a32b2.jpg)

![c167d8dc340de9936bb01059e84fa87fd8209d82fef8c87e6060af8a098957a9.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/c167d8dc340de9936bb01059e84fa87fd8209d82fef8c87e6060af8a098957a9.jpg)

![d6bc0bb63ec03a6a3483b0edd9c0f65840d34a1c1f04e09563187302e2720532.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/d6bc0bb63ec03a6a3483b0edd9c0f65840d34a1c1f04e09563187302e2720532.jpg)

![dff3095cb4b186cf5a86677ab4f31b364ae17f9cdbbe474e0e9663588a34710c.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/dff3095cb4b186cf5a86677ab4f31b364ae17f9cdbbe474e0e9663588a34710c.jpg)

![e14f23332c43712dc32c210a0e8296ba9a2aacf2c978bb6edc9c70a51f339107.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/e14f23332c43712dc32c210a0e8296ba9a2aacf2c978bb6edc9c70a51f339107.jpg)

![f365c887769f26918536aeb5d47a7773a4551c466a3cf2ac7a22edc3a1d990d1.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/f365c887769f26918536aeb5d47a7773a4551c466a3cf2ac7a22edc3a1d990d1.jpg)

![f9fd67010da00d280dede8595481ec1ce81995ef52e151f93b3b798556ddcd3a.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/f9fd67010da00d280dede8595481ec1ce81995ef52e151f93b3b798556ddcd3a.jpg)

![fed89c03bfb064a48a96cdf39e151d33dfaf110d4288e4b6906d5a3a364466a4.jpg](../icml_results/590_Regression%20Language%20Models%20for%20Code/tables/fed89c03bfb064a48a96cdf39e151d33dfaf110d4288e4b6906d5a3a364466a4.jpg)

<a id="paper-0582"></a>

## When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation


### Images

![2d25186f6dd51f2cda9fde12ce9a276ea8561b9376d9feafb79baeea5f43cd55.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/images/2d25186f6dd51f2cda9fde12ce9a276ea8561b9376d9feafb79baeea5f43cd55.jpg)

![4c78051d94a5557ae805cdb6635eca8b68690fe39a50bdbcbe1a7573501fa6b6.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/images/4c78051d94a5557ae805cdb6635eca8b68690fe39a50bdbcbe1a7573501fa6b6.jpg)

![50f90b9c80874b4c02be97c311d00153dca90f864c009971daa16cd062876d34.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/images/50f90b9c80874b4c02be97c311d00153dca90f864c009971daa16cd062876d34.jpg)

![5f2cc6dab305208a53dddeb30bdfe51fa33ae480490a029683748f265754887e.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/images/5f2cc6dab305208a53dddeb30bdfe51fa33ae480490a029683748f265754887e.jpg)

![abc247a595a628343943ab7372bca155ff3db667b900ca9caa03a7b619c9eb84.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/images/abc247a595a628343943ab7372bca155ff3db667b900ca9caa03a7b619c9eb84.jpg)

![dbccc945a0a4c84960c7cde8dc30632b850625fca658edc2c2a75271f00ff2ca.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/images/dbccc945a0a4c84960c7cde8dc30632b850625fca658edc2c2a75271f00ff2ca.jpg)

### Tables

![2c35435f1d1670c9f2bb5d6015df8cb6f87aec68eb8ce36a9b0691f891a847ea.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/2c35435f1d1670c9f2bb5d6015df8cb6f87aec68eb8ce36a9b0691f891a847ea.jpg)

![451c03ad78811d9a23465b2a41ba8fb8cca2801c39097b1d056cb8df3d03a637.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/451c03ad78811d9a23465b2a41ba8fb8cca2801c39097b1d056cb8df3d03a637.jpg)

![72aa400f816d03049cf621f30e5f565451eb5a15edd6cb88b0156f939dffceef.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/72aa400f816d03049cf621f30e5f565451eb5a15edd6cb88b0156f939dffceef.jpg)

![92f2377a93db8511723048c899826688a94342e9f89ad39c463f06eb4f267be0.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/92f2377a93db8511723048c899826688a94342e9f89ad39c463f06eb4f267be0.jpg)

![aab1f87ede14220224fe85607991846c4b03d1f704546a8f1a428100d81f0f8a.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/aab1f87ede14220224fe85607991846c4b03d1f704546a8f1a428100d81f0f8a.jpg)

![c06d368f9174c4333604690b0fa020041c5fd4636b04471840acca96b219e5c9.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/c06d368f9174c4333604690b0fa020041c5fd4636b04471840acca96b219e5c9.jpg)

![c891d42b510c07bf2cdc3e7b4ac56dad3dec4d7a1fe2ea668951e6e0e52e51ac.jpg](../icml_results/591_When%20AI%20Benchmarks%20Plateau_%20A%20Systematic%20Study%20of%20Benchmark%20Saturation/tables/c891d42b510c07bf2cdc3e7b4ac56dad3dec4d7a1fe2ea668951e6e0e52e51ac.jpg)

<a id="paper-0583"></a>

## Same Graph Cross-Task Transfer in GNNs: Protocols and Predictors


### Images

![941ef1572bca373a3a43bd413422860d4c168989d9f945cfdc1d22386669fa7f.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/images/941ef1572bca373a3a43bd413422860d4c168989d9f945cfdc1d22386669fa7f.jpg)

![961f4ab3eb2f9fa3a67238d5056b3a16ff1f52626b5acf3b8bc4a7dc5275af4c.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/images/961f4ab3eb2f9fa3a67238d5056b3a16ff1f52626b5acf3b8bc4a7dc5275af4c.jpg)

### Tables

![0ffeb3a0d7dc57bf92dde1dfd30aa440af1323711f2402a5c84233921f494c65.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/0ffeb3a0d7dc57bf92dde1dfd30aa440af1323711f2402a5c84233921f494c65.jpg)

![463aee7f89c92023ecee8bd0c14c8652b1341fdbc03c5764891a64f441515e31.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/463aee7f89c92023ecee8bd0c14c8652b1341fdbc03c5764891a64f441515e31.jpg)

![4cf75c92963b761fe095588a665e8f2d76bc6d5b7cdaed5cbcd1aed7bc8edec0.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/4cf75c92963b761fe095588a665e8f2d76bc6d5b7cdaed5cbcd1aed7bc8edec0.jpg)

![6e4e37457bd9570154a4e79853a36afb9de8be9d7e1288d1a5054d8fc94999c3.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/6e4e37457bd9570154a4e79853a36afb9de8be9d7e1288d1a5054d8fc94999c3.jpg)

![8539c9c8e29b6f520260eb8714e03054ebfdc990d9988de1ce236eb59a8c5c16.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/8539c9c8e29b6f520260eb8714e03054ebfdc990d9988de1ce236eb59a8c5c16.jpg)

![8fb6551fc25587ab121748fded04a60efd17bcb809048407e306cb78a4c93040.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/8fb6551fc25587ab121748fded04a60efd17bcb809048407e306cb78a4c93040.jpg)

![904c9aefb97c5985dd76f045c18b03e29be15a3132d118294af082542cb50bf9.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/904c9aefb97c5985dd76f045c18b03e29be15a3132d118294af082542cb50bf9.jpg)

![9f9c644d767a98b852d76d741ac2c4c65458414d967fca58e348e56a2b553af3.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/9f9c644d767a98b852d76d741ac2c4c65458414d967fca58e348e56a2b553af3.jpg)

![aa1ee41f587e2355ab15c136380bad3dcaff6a9b90c8f5d8a5821aa2163861a6.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/aa1ee41f587e2355ab15c136380bad3dcaff6a9b90c8f5d8a5821aa2163861a6.jpg)

![b096867f55e59de1ff6262adffe64fae93206234bf18114f2f18bb3e1dcca864.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/b096867f55e59de1ff6262adffe64fae93206234bf18114f2f18bb3e1dcca864.jpg)

![be4e046421494d9f97779e454b945a2216e5f393c7747879133e8d4fc6784fd4.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/be4e046421494d9f97779e454b945a2216e5f393c7747879133e8d4fc6784fd4.jpg)

![c0e53a778c4a04a3770f8eda7c87b0519eddae741dd59b4a531a93b23102c961.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/c0e53a778c4a04a3770f8eda7c87b0519eddae741dd59b4a531a93b23102c961.jpg)

![cc4ba49c93d986f6536f64895cf151f525ca9a23ebeea336624a60552f5ad8e3.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/cc4ba49c93d986f6536f64895cf151f525ca9a23ebeea336624a60552f5ad8e3.jpg)

![fd59b0d9215d024f45d4db4a3d66d96ee1b2f3785b93234dcb06da7ef606db50.jpg](../icml_results/592_Same%20Graph%20Cross-Task%20Transfer%20in%20GNNs_%20Protocols%20and%20Predictors/tables/fd59b0d9215d024f45d4db4a3d66d96ee1b2f3785b93234dcb06da7ef606db50.jpg)

<a id="paper-0584"></a>

## Solving Stochastic Variational Inequalities without the Bounded Variance Assumption


### Images

![9470dfe409f2a10f47eb6caf759691d3d1cfa8bea3c5fef6d9725de727218c72.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/images/9470dfe409f2a10f47eb6caf759691d3d1cfa8bea3c5fef6d9725de727218c72.jpg)

![b1a076cdfe072b21346c1a8d37a45f9649edea9cd28c44f72d2c977f77872eea.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/images/b1a076cdfe072b21346c1a8d37a45f9649edea9cd28c44f72d2c977f77872eea.jpg)

![f926f966b1c4e4a03a1b0f5b3525ed517c9f712f6597abc7b1b199c45f9f3ea0.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/images/f926f966b1c4e4a03a1b0f5b3525ed517c9f712f6597abc7b1b199c45f9f3ea0.jpg)

### Tables

![3a1f1b82396f3839ed3c82545559245dc578ae3e21caf3b80b8b206bf7e8a466.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/tables/3a1f1b82396f3839ed3c82545559245dc578ae3e21caf3b80b8b206bf7e8a466.jpg)

![9e28c6658a5b91262308ce4b4dc7fd0a60a8444cda69570e332968f9c37ba76a.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/tables/9e28c6658a5b91262308ce4b4dc7fd0a60a8444cda69570e332968f9c37ba76a.jpg)

![d71fc3ffd7df961ef8ec3cf99f3ae31475d3543c55db308ea28d7a16886d0623.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/tables/d71fc3ffd7df961ef8ec3cf99f3ae31475d3543c55db308ea28d7a16886d0623.jpg)

![df318ec218d60f53adf2fa55d4e50b234e8e174b15406fb52c0e6bcb76eed01d.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/tables/df318ec218d60f53adf2fa55d4e50b234e8e174b15406fb52c0e6bcb76eed01d.jpg)

![e3522040fe63653a8fd5bea4f7949887146040db5fbadde78cf1d9b0c8e5cc94.jpg](../icml_results/593_Solving%20Stochastic%20Variational%20Inequalities%20without%20the%20Bounded%20Variance%20Assumption/tables/e3522040fe63653a8fd5bea4f7949887146040db5fbadde78cf1d9b0c8e5cc94.jpg)

<a id="paper-0585"></a>

## Set-Preserving Calibration from Conformal P-Values to E-Values


### Images

![3a7176274943aa3a218e21ec7cb420bf54b0750eac9fdfc3a5c184c9e0f5c8da.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/images/3a7176274943aa3a218e21ec7cb420bf54b0750eac9fdfc3a5c184c9e0f5c8da.jpg)

![49f9b8db785b39a3af9b7f8bba294a7a1ac2d3d8ecf9fac6d6af8feaf5597d6b.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/images/49f9b8db785b39a3af9b7f8bba294a7a1ac2d3d8ecf9fac6d6af8feaf5597d6b.jpg)

![5da20da2ae7147d115d096841fa6a2e12b9fe1e78e47c84ccf5b7014d8507e49.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/images/5da20da2ae7147d115d096841fa6a2e12b9fe1e78e47c84ccf5b7014d8507e49.jpg)

![b5465ce4c601841c3414889dd949ef049bc06150eb7ecfde78b41e1b61e6099d.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/images/b5465ce4c601841c3414889dd949ef049bc06150eb7ecfde78b41e1b61e6099d.jpg)

### Tables

![066f6d3f2798849661a98973a7570ab5664d130f2d89b621736524b12fe55b12.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/066f6d3f2798849661a98973a7570ab5664d130f2d89b621736524b12fe55b12.jpg)

![3588c9a09b189cabd89eacfd8c5a2bd01dbba0defde887727337dcbd8fc00767.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/3588c9a09b189cabd89eacfd8c5a2bd01dbba0defde887727337dcbd8fc00767.jpg)

![3bdfc1208132ce15317965f6942f58aa9b739faba2fb70c04676cdbae6cfbf8f.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/3bdfc1208132ce15317965f6942f58aa9b739faba2fb70c04676cdbae6cfbf8f.jpg)

![3f29fa40d7ef60b4c6a96064a28519d7f1998f2fb4cfba85cb24545dd114753e.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/3f29fa40d7ef60b4c6a96064a28519d7f1998f2fb4cfba85cb24545dd114753e.jpg)

![44ee79a398d594809febdfc70a70ec53f4e218c2d7a75afc7c66864a6e29fff7.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/44ee79a398d594809febdfc70a70ec53f4e218c2d7a75afc7c66864a6e29fff7.jpg)

![980f5fe8d1240e59df320a790b3484e9f8ba2e844a39183d36292d0a3ba232fc.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/980f5fe8d1240e59df320a790b3484e9f8ba2e844a39183d36292d0a3ba232fc.jpg)

![af6c96e0ee44431becd3716f6a92faa57f32d4b7a44d75c53310e68956ea8ff7.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/af6c96e0ee44431becd3716f6a92faa57f32d4b7a44d75c53310e68956ea8ff7.jpg)

![b28d124e8fba2dd0d27f0ce8292dbbd54674a2f9d1802e9ebbed4cabf1de4f85.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/b28d124e8fba2dd0d27f0ce8292dbbd54674a2f9d1802e9ebbed4cabf1de4f85.jpg)

![b2cceaaee5fb59a221a0fc21905993e654ecbf33f9696b5266fa3511d8b24f29.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/b2cceaaee5fb59a221a0fc21905993e654ecbf33f9696b5266fa3511d8b24f29.jpg)

![b8cad7dc82b4a7ef245b3ad08c50a029c7b41509aad7efa8232ed0bdbfa188be.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/b8cad7dc82b4a7ef245b3ad08c50a029c7b41509aad7efa8232ed0bdbfa188be.jpg)

![d1a24a65ce68b128ebd40f49548af56e0bbcacde2edbfac930c63e02a427ebea.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/d1a24a65ce68b128ebd40f49548af56e0bbcacde2edbfac930c63e02a427ebea.jpg)

![d5e09d5ff57bd2768a25f026ba61cb29804a7572105661b9760cfd31eca1c01d.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/d5e09d5ff57bd2768a25f026ba61cb29804a7572105661b9760cfd31eca1c01d.jpg)

![da73c6141c83946758a5491bd94b5cd9b5f5eda6cefbab9d432b6408b75f3a15.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/da73c6141c83946758a5491bd94b5cd9b5f5eda6cefbab9d432b6408b75f3a15.jpg)

![e9aa001b239f40f6bf15a6231166403574a32e14f2e18c0403c826eddf15665a.jpg](../icml_results/594_Set-Preserving%20Calibration%20from%20Conformal%20P-Values%20to%20E-Values/tables/e9aa001b239f40f6bf15a6231166403574a32e14f2e18c0403c826eddf15665a.jpg)

<a id="paper-0586"></a>

## Scaling Small Agents Through Strategy Auctions


### Images

![0fa2bca315ba6d186baffd2ebaa664aff06f337361a576c3e8fad313894d0f50.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/0fa2bca315ba6d186baffd2ebaa664aff06f337361a576c3e8fad313894d0f50.jpg)

![216718efd04b9dcdbb2bf18442b62ad75e556b22bcb1a13aea8f25bf63551deb.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/216718efd04b9dcdbb2bf18442b62ad75e556b22bcb1a13aea8f25bf63551deb.jpg)

![361eeafb93b867e28534fc659d115ed180d2cc6dfdfb12185b347b3fe23921dc.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/361eeafb93b867e28534fc659d115ed180d2cc6dfdfb12185b347b3fe23921dc.jpg)

![4a22e0a9813fe29b0b808609cbde7ebfa063dbaf0916eebd157d830128613fdd.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/4a22e0a9813fe29b0b808609cbde7ebfa063dbaf0916eebd157d830128613fdd.jpg)

![59f4cbdf767967345550c9d787b1d029df49c44e8791a566093fffe833466dc5.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/59f4cbdf767967345550c9d787b1d029df49c44e8791a566093fffe833466dc5.jpg)

![7b38e3993c087b23d7bd71401197847b64d78a74f9b34323ac259828964e4aeb.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/7b38e3993c087b23d7bd71401197847b64d78a74f9b34323ac259828964e4aeb.jpg)

![97e49b8119061cfa0539678f32a1f778e089bef4f71680fe1114e89c38cf92ca.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/97e49b8119061cfa0539678f32a1f778e089bef4f71680fe1114e89c38cf92ca.jpg)

![d2aa86a7c1237b0c202093864ff3eab59f3223ce8da0d6c57e778c00a562072e.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/d2aa86a7c1237b0c202093864ff3eab59f3223ce8da0d6c57e778c00a562072e.jpg)

![eee66536e6be27bf76496486f6a19f4aeac36f33b928d9fee194aba9dec2bd19.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/eee66536e6be27bf76496486f6a19f4aeac36f33b928d9fee194aba9dec2bd19.jpg)

![f5db580972b9aaf392dd155dd43a25fe3863b976d22da92ca4c785b7beabd028.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/images/f5db580972b9aaf392dd155dd43a25fe3863b976d22da92ca4c785b7beabd028.jpg)

### Tables

![1250f4f49ee4288a33f309188cf2916d4e38e2e8bd92349ec6fb8ee8ca3d9fb3.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/1250f4f49ee4288a33f309188cf2916d4e38e2e8bd92349ec6fb8ee8ca3d9fb3.jpg)

![21c37bcd721a297d8271d6af14d2d39da1ab572b78e8b97f0e7d61e976da87ec.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/21c37bcd721a297d8271d6af14d2d39da1ab572b78e8b97f0e7d61e976da87ec.jpg)

![2c5bd05b8c353b9e74e482282e88808f3ce0cffcf10eaeac6b41c32b70e54781.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/2c5bd05b8c353b9e74e482282e88808f3ce0cffcf10eaeac6b41c32b70e54781.jpg)

![32e9ec8fe0502f4e53955c8765b2cff2d692e7c7f012401947a16bcdeb1e400b.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/32e9ec8fe0502f4e53955c8765b2cff2d692e7c7f012401947a16bcdeb1e400b.jpg)

![3495336f5cf000f499af7900cf501c8b56f07a730d513a61b8940402eb1cdef6.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/3495336f5cf000f499af7900cf501c8b56f07a730d513a61b8940402eb1cdef6.jpg)

![3c9d5645884ecb3efce2d2041186536a3d84562a5af521be05439e20fa00b742.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/3c9d5645884ecb3efce2d2041186536a3d84562a5af521be05439e20fa00b742.jpg)

![49c03b4b76e887d4bdbcc15ae0fbb708a2960fc34c61dc04027f92e1f6ddc9ec.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/49c03b4b76e887d4bdbcc15ae0fbb708a2960fc34c61dc04027f92e1f6ddc9ec.jpg)

![52e4c7f9407c55fde653137da13c3a7d427b25f3b961b2df10d1e5385b955b55.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/52e4c7f9407c55fde653137da13c3a7d427b25f3b961b2df10d1e5385b955b55.jpg)

![597e9428ce00d4abbbaeedbc569493678906c8b67880c6240ab5371ece202662.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/597e9428ce00d4abbbaeedbc569493678906c8b67880c6240ab5371ece202662.jpg)

![5b55d0d84718040e023c12a9efcea34a5f737019f5a1de6b635d308874da0b1d.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/5b55d0d84718040e023c12a9efcea34a5f737019f5a1de6b635d308874da0b1d.jpg)

![67b6b08a7f7ccb2c1d503a56c16e32a4995a6b42b19f54eb217bee21699515a1.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/67b6b08a7f7ccb2c1d503a56c16e32a4995a6b42b19f54eb217bee21699515a1.jpg)

![6ea4e78419393a96dc6007d1fcb4548702ae1b844ed1654213e126cc42c2b20d.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/6ea4e78419393a96dc6007d1fcb4548702ae1b844ed1654213e126cc42c2b20d.jpg)

![789b304c545f3866945c6af9a0e947f6cebc7e6e1a5f866185bd3c0d0b4aad41.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/789b304c545f3866945c6af9a0e947f6cebc7e6e1a5f866185bd3c0d0b4aad41.jpg)

![80681cba4e0ed2900f9991641e967c30364af8486a663577a9fd643041d48a24.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/80681cba4e0ed2900f9991641e967c30364af8486a663577a9fd643041d48a24.jpg)

![859b6724020044295c6778ca05b8100ed5d4eec44805dca5b06be21ce7ab210d.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/859b6724020044295c6778ca05b8100ed5d4eec44805dca5b06be21ce7ab210d.jpg)

![87b3ea87f054b35c032a5ed2d01b17fdaab3b23c40a32b391317d91d32f8dc2d.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/87b3ea87f054b35c032a5ed2d01b17fdaab3b23c40a32b391317d91d32f8dc2d.jpg)

![89a248e10110aa738edcf83297d1a57200528964932ed3e1f5938cef5fbe4863.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/89a248e10110aa738edcf83297d1a57200528964932ed3e1f5938cef5fbe4863.jpg)

![a8f2ae827d8adb5563879a43b9ad68b3d17aaf6007ff520e42c9135c149d0c54.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/a8f2ae827d8adb5563879a43b9ad68b3d17aaf6007ff520e42c9135c149d0c54.jpg)

![adc736ab2e29f57241e9a15d2930f0ef992ae9f833a99ef61c654e0d3ab590e2.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/adc736ab2e29f57241e9a15d2930f0ef992ae9f833a99ef61c654e0d3ab590e2.jpg)

![aecf25ddc68f7cb419234786d3a4e2bada2b0b40fff0b89cc1470d628b43ecdc.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/aecf25ddc68f7cb419234786d3a4e2bada2b0b40fff0b89cc1470d628b43ecdc.jpg)

![b0c84a30ba74189f14809a771ec1dbd9a248cce2bb7554a7605fd466587f979a.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/b0c84a30ba74189f14809a771ec1dbd9a248cce2bb7554a7605fd466587f979a.jpg)

![b41a2d89e213f5b509beb5e7710789aba1a2979166b86f238a5185952649ad60.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/b41a2d89e213f5b509beb5e7710789aba1a2979166b86f238a5185952649ad60.jpg)

![c7f9253add97ee6c68ed8c0a8cb81e0da78a7035cf81dd29bd3e39c66cb2b4c5.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/c7f9253add97ee6c68ed8c0a8cb81e0da78a7035cf81dd29bd3e39c66cb2b4c5.jpg)

![cd309d413f3008cd2fd92a544402f3949ab0bc33644ce96dd407aa2f1f76aa27.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/cd309d413f3008cd2fd92a544402f3949ab0bc33644ce96dd407aa2f1f76aa27.jpg)

![d824dd4467ada28ca94abd3325e8fdfa86dbe956dbbf2b1c69ee98ed0c8a6c92.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/d824dd4467ada28ca94abd3325e8fdfa86dbe956dbbf2b1c69ee98ed0c8a6c92.jpg)

![e3f06963c72f3b1b637a5404952ac89cef8acd6cea6d429c56184bdfae0229c1.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/e3f06963c72f3b1b637a5404952ac89cef8acd6cea6d429c56184bdfae0229c1.jpg)

![f86585c50213eda2f019dc11a8124ddb988c7f4533d63046087a1cefc9fd7e54.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/f86585c50213eda2f019dc11a8124ddb988c7f4533d63046087a1cefc9fd7e54.jpg)

![ffb3cfca2998cc3f78396c6749e92a8c8444fcff3af19078fe5b76b9164453cb.jpg](../icml_results/595_Scaling%20Small%20Agents%20Through%20Strategy%20Auctions/tables/ffb3cfca2998cc3f78396c6749e92a8c8444fcff3af19078fe5b76b9164453cb.jpg)

<a id="paper-0587"></a>

## Attention’s forward pass and Frank-Wolfe


### Images

![1cf6a6c0461b7625b433b2990458660bc02d8cdce63cc5ab9effd2c38474a322.jpg](../icml_results/596_Attention%E2%80%99s%20forward%20pass%20and%20Frank-Wolfe/images/1cf6a6c0461b7625b433b2990458660bc02d8cdce63cc5ab9effd2c38474a322.jpg)

![1d5335ba0e69a8e02d82aa0235ee82c585476dc8b3a6656e38068f7ad5359dd4.jpg](../icml_results/596_Attention%E2%80%99s%20forward%20pass%20and%20Frank-Wolfe/images/1d5335ba0e69a8e02d82aa0235ee82c585476dc8b3a6656e38068f7ad5359dd4.jpg)

![326d3262c2db1971673eb03ae2675ab8360e90677792a938825b40b86cf7e93c.jpg](../icml_results/596_Attention%E2%80%99s%20forward%20pass%20and%20Frank-Wolfe/images/326d3262c2db1971673eb03ae2675ab8360e90677792a938825b40b86cf7e93c.jpg)

![6d0177303604c9d2bb5a844e904534c8c1d54df1da635c44503a2bd1a92036fe.jpg](../icml_results/596_Attention%E2%80%99s%20forward%20pass%20and%20Frank-Wolfe/images/6d0177303604c9d2bb5a844e904534c8c1d54df1da635c44503a2bd1a92036fe.jpg)

![87f4486e1c49fcbf4cf0035ade49b0bd18d3af55b3d0c5e3e44ca1bd0e7b7916.jpg](../icml_results/596_Attention%E2%80%99s%20forward%20pass%20and%20Frank-Wolfe/images/87f4486e1c49fcbf4cf0035ade49b0bd18d3af55b3d0c5e3e44ca1bd0e7b7916.jpg)

![e58d3cfbf425bb45890619e429044ab80982f51b85616ec2689b319ef731f449.jpg](../icml_results/596_Attention%E2%80%99s%20forward%20pass%20and%20Frank-Wolfe/images/e58d3cfbf425bb45890619e429044ab80982f51b85616ec2689b319ef731f449.jpg)

<a id="paper-0588"></a>

## Logical Guidance for the Exact Composition of Diffusion Models


### Images

![28871410ecc40e50f64f599b185b34dbff919da0ba118e75873daafa7ccb11b6.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/28871410ecc40e50f64f599b185b34dbff919da0ba118e75873daafa7ccb11b6.jpg)

![38833724c6038dbe0e88b146d903796c92e0e3143d0c299e53ba55d871186884.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/38833724c6038dbe0e88b146d903796c92e0e3143d0c299e53ba55d871186884.jpg)

![3b71ba6b0593408d3c87b5def6d9bda6812db3abaf33b7abcfee5755614ca7fa.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/3b71ba6b0593408d3c87b5def6d9bda6812db3abaf33b7abcfee5755614ca7fa.jpg)

![687bafe5f0ed21c118e01259d2e70cd962b88e3e24a66bba5c5d807908d77aff.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/687bafe5f0ed21c118e01259d2e70cd962b88e3e24a66bba5c5d807908d77aff.jpg)

![959a5659fa09b220ec4c84c3b804ea5392e511c165e630a102927fff4497775e.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/959a5659fa09b220ec4c84c3b804ea5392e511c165e630a102927fff4497775e.jpg)

![9d800d0c85b2b7ae7b19beb1672a89db98e807b52dbaf72e363f737c6333c55b.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/9d800d0c85b2b7ae7b19beb1672a89db98e807b52dbaf72e363f737c6333c55b.jpg)

![bb536e03126383c26697554b79d8cbe66cf64fa7fd7dd854b4ed53c177268e0e.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/bb536e03126383c26697554b79d8cbe66cf64fa7fd7dd854b4ed53c177268e0e.jpg)

![c0c81e92153a7c17ba4213f8ebd2582aa4063674015609f833c332e284a6df51.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/c0c81e92153a7c17ba4213f8ebd2582aa4063674015609f833c332e284a6df51.jpg)

![c811119d197f52f9c1c46bc28c36b60a5f76385ca942a120a452228d8bb49998.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/c811119d197f52f9c1c46bc28c36b60a5f76385ca942a120a452228d8bb49998.jpg)

![c9179ced6cecb7371c8bf026b92535a5e99b35a2231f14f578b4689c33ce23e3.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/c9179ced6cecb7371c8bf026b92535a5e99b35a2231f14f578b4689c33ce23e3.jpg)

![df3fd402f2f983e1f769a8d241180b9ca182de5f2bde77129f0979fcf4a7beb3.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/df3fd402f2f983e1f769a8d241180b9ca182de5f2bde77129f0979fcf4a7beb3.jpg)

![f3d74358e3c63266e07649247122ceb305759f4ee2c2d24589b8e277d4eb8211.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/f3d74358e3c63266e07649247122ceb305759f4ee2c2d24589b8e277d4eb8211.jpg)

![fb748ea921decedaf9afd4b6577110d11447b26aee4d31509716fbff8a022a48.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/images/fb748ea921decedaf9afd4b6577110d11447b26aee4d31509716fbff8a022a48.jpg)

### Tables

![0d4de4ce875518aee53e38e471a70961ec60eb2caec0fc49a4429fafc69aa37e.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/0d4de4ce875518aee53e38e471a70961ec60eb2caec0fc49a4429fafc69aa37e.jpg)

![114d54c7659ed41e55bc1252e0b10c47d4183afea89a8cd3ffc3903c78384f55.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/114d54c7659ed41e55bc1252e0b10c47d4183afea89a8cd3ffc3903c78384f55.jpg)

![172b5248d17dd6a8798a70bc8c3e61ba4746ac2b9ca7ba4b090dd012642f5463.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/172b5248d17dd6a8798a70bc8c3e61ba4746ac2b9ca7ba4b090dd012642f5463.jpg)

![2f60ba51988ae40979506bdff85db3e39439180e24c82818d2886a8c64a3d6ce.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/2f60ba51988ae40979506bdff85db3e39439180e24c82818d2886a8c64a3d6ce.jpg)

![3e280974acd976b2ccb5311c3bed9bdf9d245048e71b424ba524fba3399a3dcb.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/3e280974acd976b2ccb5311c3bed9bdf9d245048e71b424ba524fba3399a3dcb.jpg)

![44245566157086598cedaf227fa3a79db092faf2420290fc5516d89872ebdc0e.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/44245566157086598cedaf227fa3a79db092faf2420290fc5516d89872ebdc0e.jpg)

![65a2afc1d346c5f15569222479d9ff791b64a247fcb557ba1e2ab5d201d7c1d9.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/65a2afc1d346c5f15569222479d9ff791b64a247fcb557ba1e2ab5d201d7c1d9.jpg)

![685ee97330752029abb4b2076588c911a887f1e6ce377a5663b3b07df43b7a42.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/685ee97330752029abb4b2076588c911a887f1e6ce377a5663b3b07df43b7a42.jpg)

![6abb17cadbafa97095c3b59fc38700df3a533c0c3fe9e9bd339bf7290137af6d.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/6abb17cadbafa97095c3b59fc38700df3a533c0c3fe9e9bd339bf7290137af6d.jpg)

![71d3238e36e55fc1cbc3a7966d47b80554da23681fafb729de44ee665d232263.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/71d3238e36e55fc1cbc3a7966d47b80554da23681fafb729de44ee665d232263.jpg)

![73d007e8d5573d6a8bb5e2c278c7a251525ce25e5afb6dc1b214fdca5a0f605a.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/73d007e8d5573d6a8bb5e2c278c7a251525ce25e5afb6dc1b214fdca5a0f605a.jpg)

![74fb9cc45f8805605431a17a0de42dbdcd3d459d88736d1de571f97aaaa575cd.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/74fb9cc45f8805605431a17a0de42dbdcd3d459d88736d1de571f97aaaa575cd.jpg)

![7b7fa089a63f1caf45edec97befa8822df311c51a3a20603dd49c942fafdcd1b.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/7b7fa089a63f1caf45edec97befa8822df311c51a3a20603dd49c942fafdcd1b.jpg)

![8d90afa54bc77f968a8cee15da726e86c176f13cb615c68198bf915992b8d449.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/8d90afa54bc77f968a8cee15da726e86c176f13cb615c68198bf915992b8d449.jpg)

![94a577ed6a9621cf7c50c2579172b585c8565d3e3ec5977b2e2b55eb9ea64bfa.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/94a577ed6a9621cf7c50c2579172b585c8565d3e3ec5977b2e2b55eb9ea64bfa.jpg)

![9504aa7c1f582043dd3f83e5dd2f0f1023f44a99edc8231152039e754be34b70.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/9504aa7c1f582043dd3f83e5dd2f0f1023f44a99edc8231152039e754be34b70.jpg)

![ab9760d1159c440f1ff3318cf3bfb8bd54a994679f6b56c21636c9610c8eff4d.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/ab9760d1159c440f1ff3318cf3bfb8bd54a994679f6b56c21636c9610c8eff4d.jpg)

![bd6d1279c19119dde773165e9abeed0ff756c889f94f31bae7ff5c32596f1264.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/bd6d1279c19119dde773165e9abeed0ff756c889f94f31bae7ff5c32596f1264.jpg)

![c2776bc14f2615ae6d8e2b87e5e4d89969f9964e288c9039b92d61c1e34afaf6.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/c2776bc14f2615ae6d8e2b87e5e4d89969f9964e288c9039b92d61c1e34afaf6.jpg)

![c546c252756257dd965886c9c736d73fac56b68322542f177319d588f9ccd69d.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/c546c252756257dd965886c9c736d73fac56b68322542f177319d588f9ccd69d.jpg)

![ccc28e76c31433cfffd8927c6052fdfecd5f295920a08df49ef030cf5ca6d957.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/ccc28e76c31433cfffd8927c6052fdfecd5f295920a08df49ef030cf5ca6d957.jpg)

![df9e2b6e77b20aecfceb2b11a52deea741f041cabf0607905539c0c03fdbc298.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/df9e2b6e77b20aecfceb2b11a52deea741f041cabf0607905539c0c03fdbc298.jpg)

![ed84e42316d97fdb09641d122cde762f2d90c75863ac7de92f7bd470052d3b83.jpg](../icml_results/597_Logical%20Guidance%20for%20the%20Exact%20Composition%20of%20Diffusion%20Models/tables/ed84e42316d97fdb09641d122cde762f2d90c75863ac7de92f7bd470052d3b83.jpg)

<a id="paper-0589"></a>

## GEPC: Group-Equivariant Posterior Consistency for Out-of-Distribution Detection in Diffusion Models


### Images

![23d384bfee4b1ab92dd3f4cabf759e94995ef880ef9c222afbf5ad6de2aabcc9.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/23d384bfee4b1ab92dd3f4cabf759e94995ef880ef9c222afbf5ad6de2aabcc9.jpg)

![2aa6647caeacd166e68811f9c3951a9111e403c23b057385cdcc115709309d3c.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/2aa6647caeacd166e68811f9c3951a9111e403c23b057385cdcc115709309d3c.jpg)

![3ddc0659bc0b520eabf365f8ddfca71a35258094163aa984c443194c4a885a07.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/3ddc0659bc0b520eabf365f8ddfca71a35258094163aa984c443194c4a885a07.jpg)

![548dfeded8af7a0bc0cb0a6c5f58882426583c858a850f5f8e4ba8470f9cc203.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/548dfeded8af7a0bc0cb0a6c5f58882426583c858a850f5f8e4ba8470f9cc203.jpg)

![63985a19af4cebf5a02152f8fdf63edf86b8b66150ee358c8ba892c5c3ff945c.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/63985a19af4cebf5a02152f8fdf63edf86b8b66150ee358c8ba892c5c3ff945c.jpg)

![b24497a7c907392d7e745cebf851bd9fa2bca56db8bdb8c8c39523be3f97f28a.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/b24497a7c907392d7e745cebf851bd9fa2bca56db8bdb8c8c39523be3f97f28a.jpg)

![ce4620e5a24268ad016b0c8d1329f9d68d912e2468a3a85eb38136f587723493.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/ce4620e5a24268ad016b0c8d1329f9d68d912e2468a3a85eb38136f587723493.jpg)

![fbfa228e990eb6127af141c51ba0a9309275c10337b162a2981fd737a5cfda8c.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/fbfa228e990eb6127af141c51ba0a9309275c10337b162a2981fd737a5cfda8c.jpg)

![fe8c42858581e6d25381a36607fe79989ee7388acf0c6de0532f639bade34ee7.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/images/fe8c42858581e6d25381a36607fe79989ee7388acf0c6de0532f639bade34ee7.jpg)

### Tables

![02f6799e60f6e49498dc8f5c3c5cf361c472956477061457da8ba8dbb261060e.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/02f6799e60f6e49498dc8f5c3c5cf361c472956477061457da8ba8dbb261060e.jpg)

![0401edc1e3d1080362e14030a6dc4a78c139367a5fbb7e3a3edd8e1f48ccd6d2.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/0401edc1e3d1080362e14030a6dc4a78c139367a5fbb7e3a3edd8e1f48ccd6d2.jpg)

![4029f5bcee5ecc69159af6e6c010abfde75ab2edfca8c22d3544685b66f6efc2.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/4029f5bcee5ecc69159af6e6c010abfde75ab2edfca8c22d3544685b66f6efc2.jpg)

![43331dd40fcc59245a18efb07e8fa38603a0b08d022df07f48adadec2fb133c1.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/43331dd40fcc59245a18efb07e8fa38603a0b08d022df07f48adadec2fb133c1.jpg)

![67c04e2bcb82e68920a50e7e0da75b4204157d76901818d0f2a551e6fddcf26c.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/67c04e2bcb82e68920a50e7e0da75b4204157d76901818d0f2a551e6fddcf26c.jpg)

![6bac6e20b8b96bd4745684a38781148b97c876b0da8f8055a2ddac3c07af2a48.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/6bac6e20b8b96bd4745684a38781148b97c876b0da8f8055a2ddac3c07af2a48.jpg)

![a4dd925ebbbb818dd9ed0a8409c73976593cce6cb5176ba501675dcfa9840345.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/a4dd925ebbbb818dd9ed0a8409c73976593cce6cb5176ba501675dcfa9840345.jpg)

![ab66d5894340c7542681504a60e7403e87beea346558e347de03026826069273.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/ab66d5894340c7542681504a60e7403e87beea346558e347de03026826069273.jpg)

![ba90860ca820a11a8e2ef1bcd10dbb154b40521ba81a492eda6000db01c563e4.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/ba90860ca820a11a8e2ef1bcd10dbb154b40521ba81a492eda6000db01c563e4.jpg)

![c7940b3836d7f37b7ef6b3766e8bb4b79932d19c0bc749f377e1e98b530bf7ab.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/c7940b3836d7f37b7ef6b3766e8bb4b79932d19c0bc749f377e1e98b530bf7ab.jpg)

![cf4cba934d15a3d7e030fe46731c0bf98c452d14e0563ab1ac96a5a3b62c9daf.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/cf4cba934d15a3d7e030fe46731c0bf98c452d14e0563ab1ac96a5a3b62c9daf.jpg)

![d8956e09000501b5690fcfbe473d89627b9f2cbed6f3ecce00bd0351d92ffdb2.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/d8956e09000501b5690fcfbe473d89627b9f2cbed6f3ecce00bd0351d92ffdb2.jpg)

![de0fb936274729be11212d9bd98d32be755f80373ef711bf5304daf5eb20ee13.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/de0fb936274729be11212d9bd98d32be755f80373ef711bf5304daf5eb20ee13.jpg)

![f11283b615d2f2e91fadc289614ecc127ff7c94745a0fe36a308d80b67b28c8a.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/f11283b615d2f2e91fadc289614ecc127ff7c94745a0fe36a308d80b67b28c8a.jpg)

![fb0d7a76e3c9c868a3170fafa9b8deb293e816b7c87027b206b8a538a70cbab4.jpg](../icml_results/598_GEPC_%20Group-Equivariant%20Posterior%20Consistency%20for%20Out-of-Distribution%20Detection%20in%20Diffusion%20Models/tables/fb0d7a76e3c9c868a3170fafa9b8deb293e816b7c87027b206b8a538a70cbab4.jpg)

<a id="paper-0590"></a>

## Improving ML Attacks on LWE with Data Repetition and Stepwise Regression


### Images

![3d3c18aff4cdda2b0324718a1c7f02311b82481140dfe2abbb8b096a7a07a488.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/images/3d3c18aff4cdda2b0324718a1c7f02311b82481140dfe2abbb8b096a7a07a488.jpg)

![da354c5aa27f4bb1d02eac462a949cb0e79438f74e8361e2df03dcaf37afaffe.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/images/da354c5aa27f4bb1d02eac462a949cb0e79438f74e8361e2df03dcaf37afaffe.jpg)

### Tables

![039bc673e21e3a1fd660a7f6a08b4906a1ee896ed0d25d55d822757b1cf739af.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/039bc673e21e3a1fd660a7f6a08b4906a1ee896ed0d25d55d822757b1cf739af.jpg)

![05cf7181271bb5c57b2c9717863939e0e26546fbf18524293bcb315236358577.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/05cf7181271bb5c57b2c9717863939e0e26546fbf18524293bcb315236358577.jpg)

![09657193e6a33f9a0ba9e0357db71078528b2f977be0864071c64eafb17fc68b.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/09657193e6a33f9a0ba9e0357db71078528b2f977be0864071c64eafb17fc68b.jpg)

![1e28297c211d03d3082a2733b4442e22af8092eedbe357a039309c946d94878f.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/1e28297c211d03d3082a2733b4442e22af8092eedbe357a039309c946d94878f.jpg)

![4b0998c324378e574456f1f10ad0423ac3f3af9bb425bb7a310bd51dac51f1cf.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/4b0998c324378e574456f1f10ad0423ac3f3af9bb425bb7a310bd51dac51f1cf.jpg)

![5622dc1cd2e0676ead3a648b9d4330ba53e2aa697c5a7ef7ad0c7c0f9c7e821e.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/5622dc1cd2e0676ead3a648b9d4330ba53e2aa697c5a7ef7ad0c7c0f9c7e821e.jpg)

![8e3445c079ddaf8b387380a57fbcb17192c0905c69215133aa295f2afe288193.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/8e3445c079ddaf8b387380a57fbcb17192c0905c69215133aa295f2afe288193.jpg)

![a1ceb664cf8b174a3947d62ea8047c4509a7a658d8eb459b4458218551e2a10c.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/a1ceb664cf8b174a3947d62ea8047c4509a7a658d8eb459b4458218551e2a10c.jpg)

![a438e4fa84c6efcaababad39fb7c0392a6cc68c5b0df0f2e4e7cf12dfcd9b164.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/a438e4fa84c6efcaababad39fb7c0392a6cc68c5b0df0f2e4e7cf12dfcd9b164.jpg)

![a74a8293ada0612bd20ab30b710128560fb389a1c051301ec072cdaea4a002e7.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/a74a8293ada0612bd20ab30b710128560fb389a1c051301ec072cdaea4a002e7.jpg)

![ac74595f60daf0501c6e2365da7cb37d9152b56ab7d7cffd32f990796f85250c.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/ac74595f60daf0501c6e2365da7cb37d9152b56ab7d7cffd32f990796f85250c.jpg)

![c7b0525292c3f38f04c59a2ba5318c2358822a27f3f9e881011b1617e8fa6670.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/c7b0525292c3f38f04c59a2ba5318c2358822a27f3f9e881011b1617e8fa6670.jpg)

![d8204d0b4a7d031da8156d41d007866ec86f35d7622076a7d97a9af9de99734e.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/d8204d0b4a7d031da8156d41d007866ec86f35d7622076a7d97a9af9de99734e.jpg)

![e142822df331f6b55f83a99827c3d7a8ff29a6b20808fc7516ee34623c826199.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e142822df331f6b55f83a99827c3d7a8ff29a6b20808fc7516ee34623c826199.jpg)

![e342fefdc9571c6c2c9ed0458aabdfa4ecb77643c56899536d03d88e8057570f.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e342fefdc9571c6c2c9ed0458aabdfa4ecb77643c56899536d03d88e8057570f.jpg)

![e512f31a48ef408329ae3dc890438024a50f57129f242c13011c071b7139e156.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e512f31a48ef408329ae3dc890438024a50f57129f242c13011c071b7139e156.jpg)

![e7521d3edb0f9998dde8f94ff5c70f965c6fb455c7408a7e02844c030fa60ed5.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e7521d3edb0f9998dde8f94ff5c70f965c6fb455c7408a7e02844c030fa60ed5.jpg)

![e7f6a2d2c6d28be6dc2df5a4d3799307d419404a06a3c9225187100a651c9eab.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e7f6a2d2c6d28be6dc2df5a4d3799307d419404a06a3c9225187100a651c9eab.jpg)

![e825f66dbe0dfa602ac980147d4cad7e9d90a25d85fff97cb4c6d08374f1c009.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e825f66dbe0dfa602ac980147d4cad7e9d90a25d85fff97cb4c6d08374f1c009.jpg)

![e9be96f3ee60a72210cccbe78a43a4ee3bfdb8d990f7519b39008f8890fba9a1.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/e9be96f3ee60a72210cccbe78a43a4ee3bfdb8d990f7519b39008f8890fba9a1.jpg)

![ebacf425c210c37ad0540b0420de98fe197e0524edaeb49e27e6fce8a11fb5ee.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/ebacf425c210c37ad0540b0420de98fe197e0524edaeb49e27e6fce8a11fb5ee.jpg)

![ebd50bee91241a316cb461653a509e64a0a1a68fbfa0d7ccb07ef8322efe99a4.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/ebd50bee91241a316cb461653a509e64a0a1a68fbfa0d7ccb07ef8322efe99a4.jpg)

![f4a4664a00b01db62811081e641842a3648490554d3809ae0bd2ac97db50847b.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/f4a4664a00b01db62811081e641842a3648490554d3809ae0bd2ac97db50847b.jpg)

![fa4e9b59d21fcdf47fa1f08747c50e3742667f206d200d9f5305b8b337d475bf.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/fa4e9b59d21fcdf47fa1f08747c50e3742667f206d200d9f5305b8b337d475bf.jpg)

![ffed53f0c11157d80add2e91e7bcefb6acddbbb2983c6fb307e664bdf3c7a6df.jpg](../icml_results/599_Improving%20ML%20Attacks%20on%20LWE%20with%20Data%20Repetition%20and%20Stepwise%20Regression/tables/ffed53f0c11157d80add2e91e7bcefb6acddbbb2983c6fb307e664bdf3c7a6df.jpg)

<a id="paper-0591"></a>

## Robust Strategic Classification under Decision-Dependent Cost Uncertainty


### Images

![0422494943abc6199d4ef41843446cf225978b16c310146356a2138dbfaca872.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/0422494943abc6199d4ef41843446cf225978b16c310146356a2138dbfaca872.jpg)

![47768a0e406302a6f0d5a627e5df38501ccea03306e0cce359bd46c150628b5c.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/47768a0e406302a6f0d5a627e5df38501ccea03306e0cce359bd46c150628b5c.jpg)

![54360de4972e8bfdce40780c5b37dbae51cf3ec452dfbea10919b01522bb1cc0.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/54360de4972e8bfdce40780c5b37dbae51cf3ec452dfbea10919b01522bb1cc0.jpg)

![5cc424e1e48921f9d295c232f072701d39684a3800673452c4cf511091ce395e.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/5cc424e1e48921f9d295c232f072701d39684a3800673452c4cf511091ce395e.jpg)

![a278476c2cc93cfed826c7602097d7810ad47998ad8280d04af9aa77a0dac94c.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/a278476c2cc93cfed826c7602097d7810ad47998ad8280d04af9aa77a0dac94c.jpg)

![c51a5ead3a0bac7bc08a5f16fe19e18a2943bb9f488d93ec9b4cb590bbcc5627.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/c51a5ead3a0bac7bc08a5f16fe19e18a2943bb9f488d93ec9b4cb590bbcc5627.jpg)

![e6333245dd122989420f59002c19ea98ae2d4bb70877b54080b7683d20615189.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/e6333245dd122989420f59002c19ea98ae2d4bb70877b54080b7683d20615189.jpg)

![f087e3a625466964e91559d88c8e6b8eee19ea47f89a78019b748d112d36a603.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/images/f087e3a625466964e91559d88c8e6b8eee19ea47f89a78019b748d112d36a603.jpg)

### Tables

![185df76cf9d60563d8289d5a4b4fa1bddd08d95c8b299c821ef00aebc81dc9c7.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/185df76cf9d60563d8289d5a4b4fa1bddd08d95c8b299c821ef00aebc81dc9c7.jpg)

![4d34323a15919315bcc2eaaeee6a89e6f79119edeace0d9412c04eeace673091.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/4d34323a15919315bcc2eaaeee6a89e6f79119edeace0d9412c04eeace673091.jpg)

![8252d8d01cb8d1ee65e3f9f00fecdffd768f8e79803f869a1003d2255f5c0e23.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/8252d8d01cb8d1ee65e3f9f00fecdffd768f8e79803f869a1003d2255f5c0e23.jpg)

![a37ebc40a3805a16366a27f6754a9b960314bab61af7fbbccd780ae55331bb3c.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/a37ebc40a3805a16366a27f6754a9b960314bab61af7fbbccd780ae55331bb3c.jpg)

![c71dd3a9354fa3e5d49fec378e61fe1bbdd7a447c5b4aa467ea7fd6912d7f348.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/c71dd3a9354fa3e5d49fec378e61fe1bbdd7a447c5b4aa467ea7fd6912d7f348.jpg)

![c7585455acbccb5c2a89ae58a1a5b43aaeece8179429e8317ceae967214aa36e.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/c7585455acbccb5c2a89ae58a1a5b43aaeece8179429e8317ceae967214aa36e.jpg)

![e5347dd9654d30688f731e95584a0584bcf6d3aa65b44d873ded92a1ee8e741e.jpg](../icml_results/600_Robust%20Strategic%20Classification%20under%20Decision-Dependent%20Cost%20Uncertainty/tables/e5347dd9654d30688f731e95584a0584bcf6d3aa65b44d873ded92a1ee8e741e.jpg)

<a id="paper-0592"></a>

## Why Do We Need Warm-up? A Theoretical Perspective


### Images

![0315b3a8251954b407c7e3ce355acf6328abdb6c1be5e8d8c5b927d3bde42615.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/0315b3a8251954b407c7e3ce355acf6328abdb6c1be5e8d8c5b927d3bde42615.jpg)

![18bb5755cc073293caa2a69c948def2cedf789954476394b1c3998b4e8b5f1a1.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/18bb5755cc073293caa2a69c948def2cedf789954476394b1c3998b4e8b5f1a1.jpg)

![281323d1c9899f7150fdcb483db5ea02fc4732e535d6fa40a97067a64c578180.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/281323d1c9899f7150fdcb483db5ea02fc4732e535d6fa40a97067a64c578180.jpg)

![31d56f3f952e135bbda2ad9bd06f2c2cb06c9d2af5bdc90b1d4df684bba2c919.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/31d56f3f952e135bbda2ad9bd06f2c2cb06c9d2af5bdc90b1d4df684bba2c919.jpg)

![3ba8e4f50def0160cdb1552b045892808a66e73919c03711866caf2724f6f673.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/3ba8e4f50def0160cdb1552b045892808a66e73919c03711866caf2724f6f673.jpg)

![70b77c21b7503e2367975a1dbbead5f207b5feeb3eeea36abbae066ea4377404.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/70b77c21b7503e2367975a1dbbead5f207b5feeb3eeea36abbae066ea4377404.jpg)

![76104c720e0003fc953a44e9b91aaa5178bd3a9969e9d9052ddca3176b6dcc3e.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/76104c720e0003fc953a44e9b91aaa5178bd3a9969e9d9052ddca3176b6dcc3e.jpg)

![8b217529d4eb2bb05a4f76f0f21f174ef5634131f613add4772149cf525b9b44.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/8b217529d4eb2bb05a4f76f0f21f174ef5634131f613add4772149cf525b9b44.jpg)

![9d02ab278373874cdab5263ab7118d554b32e557ca46bbdc608e6eccdbe98636.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/9d02ab278373874cdab5263ab7118d554b32e557ca46bbdc608e6eccdbe98636.jpg)

![c565d3a5814d0faa5d1c5c6f7f25340e4758d04cc3fb995840398d804e1b32b6.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/c565d3a5814d0faa5d1c5c6f7f25340e4758d04cc3fb995840398d804e1b32b6.jpg)

![cb0efc427d63f06dca187775b7b96040be60b465e875bc1c47316c5f3c386f5e.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/cb0efc427d63f06dca187775b7b96040be60b465e875bc1c47316c5f3c386f5e.jpg)

![cbdb991f0b7b601517b756d831dcc5205ea3a675e864626bb394661c4c7a765a.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/cbdb991f0b7b601517b756d831dcc5205ea3a675e864626bb394661c4c7a765a.jpg)

![d1c04174c27aaa52fb8109737d77b5228f629893cf5828df0d43f84a90977350.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/d1c04174c27aaa52fb8109737d77b5228f629893cf5828df0d43f84a90977350.jpg)

![e488d8a482f0b3c57fcc026dc1714d7a97f8eb9201dd07bff8e59bb8d0bb892f.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/images/e488d8a482f0b3c57fcc026dc1714d7a97f8eb9201dd07bff8e59bb8d0bb892f.jpg)

### Tables

![a8f589ee7226daf23bc84e227a0389110e2d703df502db88feb6896906a6e34e.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/tables/a8f589ee7226daf23bc84e227a0389110e2d703df502db88feb6896906a6e34e.jpg)

![f167510a945554317ffc7386d797b54ab7b232f51555ea3e16b496170dab260c.jpg](../icml_results/601_Why%20Do%20We%20Need%20Warm-up_%20A%20Theoretical%20Perspective/tables/f167510a945554317ffc7386d797b54ab7b232f51555ea3e16b496170dab260c.jpg)

<a id="paper-0593"></a>

## Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning


### Images

![111c372438d76dc1d3cf9d9f3068c704ae91a5f82174e1066f693dc3697a4b48.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/images/111c372438d76dc1d3cf9d9f3068c704ae91a5f82174e1066f693dc3697a4b48.jpg)

![4c0c6c50efe08e1027caf0cf0d42d17c940293d5fc569831f1542e65edba9e11.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/images/4c0c6c50efe08e1027caf0cf0d42d17c940293d5fc569831f1542e65edba9e11.jpg)

![6ccefd1f81e16662bc943b687a999a50e2501aa6238faa682c0dbacaa6817e36.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/images/6ccefd1f81e16662bc943b687a999a50e2501aa6238faa682c0dbacaa6817e36.jpg)

![c74e0b8078de719a5a049b9bc2d0e6d3665a4dce2eca4ab2c772d2ad8ad4d58e.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/images/c74e0b8078de719a5a049b9bc2d0e6d3665a4dce2eca4ab2c772d2ad8ad4d58e.jpg)

### Tables

![1cf595ff8481d80df7c2359e34396470ffb66d73d25b8c31226f6a1f2077c7ba.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/1cf595ff8481d80df7c2359e34396470ffb66d73d25b8c31226f6a1f2077c7ba.jpg)

![1f0adadf217478577da14c4e8d18eb4ff566f84ba66de81422f7ebb68247906a.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/1f0adadf217478577da14c4e8d18eb4ff566f84ba66de81422f7ebb68247906a.jpg)

![47082a25a68884e3af658402567e8717d8ccc3323252697c6c94a784d8923047.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/47082a25a68884e3af658402567e8717d8ccc3323252697c6c94a784d8923047.jpg)

![6f6e747f7a4b833f5b8cd2c510c997a515774913189264a0c44078994ab59e50.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/6f6e747f7a4b833f5b8cd2c510c997a515774913189264a0c44078994ab59e50.jpg)

![7462b23cc3ad37e3cd65349331e2d9fb5a747f6a186695a4344b37d6aacdecec.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/7462b23cc3ad37e3cd65349331e2d9fb5a747f6a186695a4344b37d6aacdecec.jpg)

![7a44803278c3cefba5cfd976a4fcc63df0a601a20b01d876b4f6a77cd5ef45ec.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/7a44803278c3cefba5cfd976a4fcc63df0a601a20b01d876b4f6a77cd5ef45ec.jpg)

![a6c4cf7b4bbb3eaf6d3d251f65aa15d5ad44d3e0c8dba5559de0d12d65dfe766.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/a6c4cf7b4bbb3eaf6d3d251f65aa15d5ad44d3e0c8dba5559de0d12d65dfe766.jpg)

![af40d0b87e3329c9844991165aee198622e5bb71b72d4a00ab75a03e63753dcd.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/af40d0b87e3329c9844991165aee198622e5bb71b72d4a00ab75a03e63753dcd.jpg)

![cedf5fbdf60fc3b8e83e8ada6dd0dcd4ea30c2fdf0c0decb6a4bc9a6d05f7ac1.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/cedf5fbdf60fc3b8e83e8ada6dd0dcd4ea30c2fdf0c0decb6a4bc9a6d05f7ac1.jpg)

![e8201a7989370d8b2f4bde90ae7c1b9e2739d252bbd4438d8a6d2c1a3c861c47.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/e8201a7989370d8b2f4bde90ae7c1b9e2739d252bbd4438d8a6d2c1a3c861c47.jpg)

![f687dd6aa3d6a99c8d1d0d0b951052252520785f21cc0652e4302193016d48ed.jpg](../icml_results/602_Hyper-ICL_%20Attention%20Calibration%20with%20Hyperbolic%20Anchor%20Distillation%20for%20Multimodal%20In-Context%20Learn/tables/f687dd6aa3d6a99c8d1d0d0b951052252520785f21cc0652e4302193016d48ed.jpg)

<a id="paper-0594"></a>

## Asymmetric conformal prediction with penalized kernel sum-of-squares

### Images

![085d9786be4b30f2d9ef07aff589583a8a0e8d9938de23c6cd2fccb2a9fa5284.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/085d9786be4b30f2d9ef07aff589583a8a0e8d9938de23c6cd2fccb2a9fa5284.jpg)

![18211c83f263bbb912fe34e181e2c7e0bdcd5fdc5c2034ce5b67c110fc0c856d.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/18211c83f263bbb912fe34e181e2c7e0bdcd5fdc5c2034ce5b67c110fc0c856d.jpg)

![18a40f5eea828aea3c0465a6160d45b55717376e9c561b38aca53e5be4c6b5d2.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/18a40f5eea828aea3c0465a6160d45b55717376e9c561b38aca53e5be4c6b5d2.jpg)

![2f3ffde21f9ae00f11fac0ac0c056142f52f32ac414f713a60384b89b3cac84d.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/2f3ffde21f9ae00f11fac0ac0c056142f52f32ac414f713a60384b89b3cac84d.jpg)

![39448d63279ba40b88369c7de0221e4fdcce5e9d315f2ffff493ac0cdbbb3985.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/39448d63279ba40b88369c7de0221e4fdcce5e9d315f2ffff493ac0cdbbb3985.jpg)

![3c48ad85611ab18fb29ab3ca31934297e86af31d87351d55089794e94d2008fe.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/3c48ad85611ab18fb29ab3ca31934297e86af31d87351d55089794e94d2008fe.jpg)

![3e22026119cdfaa0eeb02b091ad8f74c1df30da2d197ba35da6808e93a5aafdf.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/3e22026119cdfaa0eeb02b091ad8f74c1df30da2d197ba35da6808e93a5aafdf.jpg)

![5512cd96e6077c2965adf85fb48d6c2e2fbfab98973e4abee017421974ddb00f.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/5512cd96e6077c2965adf85fb48d6c2e2fbfab98973e4abee017421974ddb00f.jpg)

![5de9fdc491243473da1fd8ef9c95220ee8f2453615bbeeba8e0deaf814f9609a.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/5de9fdc491243473da1fd8ef9c95220ee8f2453615bbeeba8e0deaf814f9609a.jpg)

![6f36011c8af26009365dadc33c492113b735441a59e777f460ca8b3525f943d2.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/6f36011c8af26009365dadc33c492113b735441a59e777f460ca8b3525f943d2.jpg)

![7c81aecf721b22e365753a8ac3548751735d6daaf5e97a98efb7a2a25780622e.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/7c81aecf721b22e365753a8ac3548751735d6daaf5e97a98efb7a2a25780622e.jpg)

![7c8f592097e8017bfb9da9279eaedccd432802e19aa1f1d19c0cf94a51bb7d79.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/7c8f592097e8017bfb9da9279eaedccd432802e19aa1f1d19c0cf94a51bb7d79.jpg)

![87e9a25ceb86a02330d68cee72105abae0d85a74e1b11c91e17140d1ea0281e2.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/87e9a25ceb86a02330d68cee72105abae0d85a74e1b11c91e17140d1ea0281e2.jpg)

![881f4e95a6ab281fc291c6f37158d36a5b76b7a0f22ec05ce1080c651bd9bbbd.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/881f4e95a6ab281fc291c6f37158d36a5b76b7a0f22ec05ce1080c651bd9bbbd.jpg)

![910475c176db0362c1425b95c80ec11a5118d2997f37a878e0f2d1062c495674.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/910475c176db0362c1425b95c80ec11a5118d2997f37a878e0f2d1062c495674.jpg)

![958a2c23fc102fc594c17833fb05c30e2609eb5985aafd5646b811b4c129c5ed.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/958a2c23fc102fc594c17833fb05c30e2609eb5985aafd5646b811b4c129c5ed.jpg)

![9c1c3c24999893d053845043466abf2827bbbd6d310cad41f028cb02e566940d.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/9c1c3c24999893d053845043466abf2827bbbd6d310cad41f028cb02e566940d.jpg)

![a10ce0a101e8bd443830bc8fd4e32cd331964849433aefaf511768bed1ec89cb.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/a10ce0a101e8bd443830bc8fd4e32cd331964849433aefaf511768bed1ec89cb.jpg)

![a195907d2acaad9f065948f322b1896ff7f9b620157338df507fd918d2f9644d.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/a195907d2acaad9f065948f322b1896ff7f9b620157338df507fd918d2f9644d.jpg)

![af091d395e01c9d0304e0a9bf0c7fc0ccb490dfcd9133767846a5de618806d25.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/af091d395e01c9d0304e0a9bf0c7fc0ccb490dfcd9133767846a5de618806d25.jpg)

![b0e996133efd07dba5c41fe7aaa9e1a8ee89f1fcce03e5ee1255830585b2363e.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/b0e996133efd07dba5c41fe7aaa9e1a8ee89f1fcce03e5ee1255830585b2363e.jpg)

![b77888953e937df621f540d204be6aa3988fb926675646b49de6a505162282ea.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/b77888953e937df621f540d204be6aa3988fb926675646b49de6a505162282ea.jpg)

![bcb2e4a136a17979616c78f1ed7c1bd48ead6c0e2fc8b1b11198963890fe97f1.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/bcb2e4a136a17979616c78f1ed7c1bd48ead6c0e2fc8b1b11198963890fe97f1.jpg)

![cadf1f57fb6ef144149c89232aa29707e59f51b463de90089d41bcd17b8e1676.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/cadf1f57fb6ef144149c89232aa29707e59f51b463de90089d41bcd17b8e1676.jpg)

![d6b5adc5e95472103c7c5b1acf0cf40278a2547969fa4e7f88cf61b6bb24e414.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/d6b5adc5e95472103c7c5b1acf0cf40278a2547969fa4e7f88cf61b6bb24e414.jpg)

![d98895d1eb45c5f58d39dc4cc71ba1691ac58448c73bb56daf928c0ebb5af13e.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/images/d98895d1eb45c5f58d39dc4cc71ba1691ac58448c73bb56daf928c0ebb5af13e.jpg)

### Tables

![2df42eac1905fe357bc71e116ebe84bec04bdcc47e3075f837e92ef41bc022a4.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/2df42eac1905fe357bc71e116ebe84bec04bdcc47e3075f837e92ef41bc022a4.jpg)

![485745df79af39582097b6ac09a947aac97167e662a10e5e21453da0dc3f5e19.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/485745df79af39582097b6ac09a947aac97167e662a10e5e21453da0dc3f5e19.jpg)

![63959bfd2189383bdd7a1c1a17cea4784c91186988d218da703ba19eac449903.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/63959bfd2189383bdd7a1c1a17cea4784c91186988d218da703ba19eac449903.jpg)

![a045606ee09e018a5097457e767e87ca18d6b285838de4aab14a61dfa445943b.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/a045606ee09e018a5097457e767e87ca18d6b285838de4aab14a61dfa445943b.jpg)

![c7eaf259b181b20271831ce3f91ec991b142ae3c65b3dc2d08b24a01c0ac01b1.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/c7eaf259b181b20271831ce3f91ec991b142ae3c65b3dc2d08b24a01c0ac01b1.jpg)

![d9ee4228ef86d3197ab977ac4d3cc3260f755832379844a40067e6e02ef40c51.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/d9ee4228ef86d3197ab977ac4d3cc3260f755832379844a40067e6e02ef40c51.jpg)

![ded8fac15ad50c95d5b70edb29061e8031384e2d7b7d085c5e9b6b3b92d40b66.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/ded8fac15ad50c95d5b70edb29061e8031384e2d7b7d085c5e9b6b3b92d40b66.jpg)

![e87d8d0b6021ccf264896d265aa82f50234b14ea2959774d92e2710fa76f708f.jpg](../icml_results/603_Asymmetric%20conformal%20prediction%20with%20penalized%20kernel%20sum-of-squares/tables/e87d8d0b6021ccf264896d265aa82f50234b14ea2959774d92e2710fa76f708f.jpg)
