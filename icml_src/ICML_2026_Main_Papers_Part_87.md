# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 87 of 197

## 目录 (Table of Contents)

1. [FasterVAR: Plug-and-Play Acceleration for Visual Autoregressive Models](#paper-2839)
2. [Mind the State: Towards Unified, Context-Aware EEG-to-fMRI Synthesis](#paper-2840)
3. [ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](#paper-2841)
4. [Expectation Alignment of Language Models for Real-World User Expectations](#paper-2842)
5. [Retro-Expert: Collaborative Reasoning for Interpretable Retrosynthesis](#paper-2843)
6. [Towards Trustworthy and Identifiable Virtual Face Generation](#paper-2844)
7. [CELL: A Causal Perspective for Fairness-aware Graph Adaptation](#paper-2845)
8. [BOOSTAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](#paper-2846)
9. [Sponge Tool Attack: Stealthy Denial-of-Efficiency against Tool-Augmented Agentic Reasoning](#paper-2847)
10. [Cross-task Calibration for Asynchronous Federated Continual Learning](#paper-2848)
11. [Prism-MoE: Efficient Dense-to-MoE Conversion for Visual Autoregressive Generation](#paper-2849)
12. [LoRe: Adaptive Interaction-Evaluation Routing with Per-step Interaction Budgets for Iterative Graph Solvers](#paper-2850)
13. [GOCM: Single-Step Graph Outlier Synthesis via Origin Consistency Model](#paper-2851)
14. [Differentially Private Continual Release with Relative Error](#paper-2852)
15. [OBJVanish: Prompt-Driven Generation of Physically Realizable 3D LiDAR-Invisible Objects](#paper-2853)
16. [SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding](#paper-2854)
17. [Statistical Consistency and Generalization of Contrastive Representation Learning](#paper-2855)
18. [Disentangling Intent from Role: Adversarial Self-Play for Persona-Invariant Safety Alignment](#paper-2856)
19. [Variational Routing: A Scalable Bayesian Framework for Calibrated Mixture-of-Experts Transformers](#paper-2857)
20. [Structured 4D Latent Predictive Model for Robot Planning](#paper-2858)
21. [Graph is a Substrate Across Data Modalities](#paper-2859)
22. [nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding](#paper-2860)
23. [Neural Modular Physics for Elastic Simulation](#paper-2861)
24. [Object-level Semantic and Spatial Distillation for Open Vocabulary Detection](#paper-2862)
25. [PlugGuard: A Streaming Safeguard for Large Models via Latent Dynamics-Guided Risk Detection](#paper-2863)
26. [HEARTS: Benchmarking LLM Reasoning on Health Time Series](#paper-2864)
27. [Robust Vision-Language Models via Manifold-Adversarial Adapters](#paper-2865)
28. [Reasoning LLM Improves Speaker Recognition in Long-form TV Dramas](#paper-2866)
29. [PRIM: Cooperative Dynamic Token Compression for Efficient Large Multimodal Models](#paper-2867)
30. [Optimizing Visual Generative Models via Distribution-wise Rewards](#paper-2868)
31. [DRFusion: Drift-Resilient Temporally Consistent Infrared–Visible Video Fusion](#paper-2869)
32. [The Optimal Token Baseline: Variance Reduction for Long-Horizon LLM-RL](#paper-2870)
33. [Spatially-Regularized Entropy for Discriminative Token Merging in Fine-Grained Re-Identification](#paper-2871)

---


<a id="paper-2839"></a>

## FasterVAR: Plug-and-Play Acceleration for Visual Autoregressive Models

### Images

![3be7dcc518c193bdb3e3fbec61caf60fc9390cf8192557ff18539a38df719bd8.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/3be7dcc518c193bdb3e3fbec61caf60fc9390cf8192557ff18539a38df719bd8.jpg)

![533ec01102d13fd4220656684493ffa20974e5b3906f9bb2b6cb52377dfbfec4.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/533ec01102d13fd4220656684493ffa20974e5b3906f9bb2b6cb52377dfbfec4.jpg)

![8ab40db10fea2ae42af363ae4ea52d7e6f2abe4ce35357f4ca3e839a9eedfe71.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/8ab40db10fea2ae42af363ae4ea52d7e6f2abe4ce35357f4ca3e839a9eedfe71.jpg)

![91a16100df644b1ba4c1df50c5268b8ec9994c10b68828ba2dcb0cc0fa054e5c.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/91a16100df644b1ba4c1df50c5268b8ec9994c10b68828ba2dcb0cc0fa054e5c.jpg)

![925e169652a9794689a3478b5df15f394f86cacb4c0a0a9e049a7c5c4b0ba7f9.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/925e169652a9794689a3478b5df15f394f86cacb4c0a0a9e049a7c5c4b0ba7f9.jpg)

![a3f477bd534f09af24f559d14dcb64daa773df17b213ecbb89a3f61843e0309e.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/a3f477bd534f09af24f559d14dcb64daa773df17b213ecbb89a3f61843e0309e.jpg)

![a4213671feafdbcc7c62689a93c5c3a12fb0fcdedf07af3e7e8a8baa8ed1fedb.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/a4213671feafdbcc7c62689a93c5c3a12fb0fcdedf07af3e7e8a8baa8ed1fedb.jpg)

![c9c42aca8e790a6192c2e2df5f54378b16365c9c67966ea6cf839a45fa405efd.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/c9c42aca8e790a6192c2e2df5f54378b16365c9c67966ea6cf839a45fa405efd.jpg)

![c9f6bd5dcff53c2eeeb9100b47716736b21509c37e84f86deead6c531f40ed5d.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/c9f6bd5dcff53c2eeeb9100b47716736b21509c37e84f86deead6c531f40ed5d.jpg)

![da48c5d1f1fa0abe3c15bf1c5ffdde73dbb80caece5281ff8920b9cd3280033a.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/da48c5d1f1fa0abe3c15bf1c5ffdde73dbb80caece5281ff8920b9cd3280033a.jpg)

![e28ccd8e1a36a331398fce124c302214d9035d3e0f70e9030681be3ffc266973.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/e28ccd8e1a36a331398fce124c302214d9035d3e0f70e9030681be3ffc266973.jpg)

![e3e96aa042bc8f2d3792d8aec184158f53fcc3e7ed099efbb3cc434a3628ba81.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/e3e96aa042bc8f2d3792d8aec184158f53fcc3e7ed099efbb3cc434a3628ba81.jpg)

### Tables

![0d32ecb9325834d65d3a7299d8adf9f88ce4e90d5d8ca3d17975798a82c93081.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/0d32ecb9325834d65d3a7299d8adf9f88ce4e90d5d8ca3d17975798a82c93081.jpg)

![10b72fce971699af6df02570a905f1e704e41d74f6e476bdbbdfb078eff3c980.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/10b72fce971699af6df02570a905f1e704e41d74f6e476bdbbdfb078eff3c980.jpg)

![1650fffa0b17e3db73ac627eb2415877fa572942ea9a4ed36ae461d969a273d7.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/1650fffa0b17e3db73ac627eb2415877fa572942ea9a4ed36ae461d969a273d7.jpg)

![25a9aa27e94358d1fab1a155a89a5acddbc148eef0ce3cfadd2ac9b51aea3b39.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/25a9aa27e94358d1fab1a155a89a5acddbc148eef0ce3cfadd2ac9b51aea3b39.jpg)

![3ad28466a2f0a6d675e4b9bcc017a6bf3e66f5e07834ab211a539e7ec99eb2c2.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/3ad28466a2f0a6d675e4b9bcc017a6bf3e66f5e07834ab211a539e7ec99eb2c2.jpg)

![4153c8a41d950b5b34eadd51dd84ed2ca5b164640d4f7d3ce04ebe9056ad0ae1.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/4153c8a41d950b5b34eadd51dd84ed2ca5b164640d4f7d3ce04ebe9056ad0ae1.jpg)

![4fa0772b7c12cb3b46caae9bf036434e4c0ab63a7dfd949d1ac3714a554dbbd0.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/4fa0772b7c12cb3b46caae9bf036434e4c0ab63a7dfd949d1ac3714a554dbbd0.jpg)

![62090a1339bdf13b53c18da96374d3fa2dd88dde21eb1fa1b166b0ba27b679ba.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/62090a1339bdf13b53c18da96374d3fa2dd88dde21eb1fa1b166b0ba27b679ba.jpg)

![62e001143c7126f8ec42810d7e5fb9d73606ce1dfe91c240ffdc257510e65b72.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/62e001143c7126f8ec42810d7e5fb9d73606ce1dfe91c240ffdc257510e65b72.jpg)

![6316120e8d004e7003f91b25bce0c96e98ccd3ed6c4ec94c128bc6a3c3342074.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/6316120e8d004e7003f91b25bce0c96e98ccd3ed6c4ec94c128bc6a3c3342074.jpg)

![664c537085c4a1df91b3194be9372236b74d2c8899a4259870131deb73178551.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/664c537085c4a1df91b3194be9372236b74d2c8899a4259870131deb73178551.jpg)

![6d2f35ed2513af73f148c25e9299e9b6186428ed38234b09f33e0637f8e1bf5b.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/6d2f35ed2513af73f148c25e9299e9b6186428ed38234b09f33e0637f8e1bf5b.jpg)

![7215c3c74dac8c49a0162fc04f70358eac3af2d955beee8eae3b464ab2044803.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/7215c3c74dac8c49a0162fc04f70358eac3af2d955beee8eae3b464ab2044803.jpg)

![79b6ab9d4c6d8dfb182c4735cb7d20ee6c94c3631037e763faf9bf0c379bde0d.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/79b6ab9d4c6d8dfb182c4735cb7d20ee6c94c3631037e763faf9bf0c379bde0d.jpg)

![85178ce68a148bd9e85773d841f747e0b73403d1e03b353c6e480f8b4f9f86ed.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/85178ce68a148bd9e85773d841f747e0b73403d1e03b353c6e480f8b4f9f86ed.jpg)

![8eb0d849a2463e67dac54f3a192730d9efae18840af2151093d2107b741cf1cc.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/8eb0d849a2463e67dac54f3a192730d9efae18840af2151093d2107b741cf1cc.jpg)

![922f1a23da71dfe9d32bb8271bfdc5f6483be9846e6f2f2b93a73a90a0895df0.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/922f1a23da71dfe9d32bb8271bfdc5f6483be9846e6f2f2b93a73a90a0895df0.jpg)

![ac6e6e18442211d5d4e0c7dc4525d0580b00305aba94b4a57f73a275c288cd8a.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/ac6e6e18442211d5d4e0c7dc4525d0580b00305aba94b4a57f73a275c288cd8a.jpg)

![ad1dda1ca0c5e0bbc4cbb1ba63525c8eb96673ecc702f625d26746d06b2d9fde.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/ad1dda1ca0c5e0bbc4cbb1ba63525c8eb96673ecc702f625d26746d06b2d9fde.jpg)

![c3ea61419e980f7af4490c104d12d553e48d0121ee634c7fef440906e2a9d4c4.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/c3ea61419e980f7af4490c104d12d553e48d0121ee634c7fef440906e2a9d4c4.jpg)

![c6d28a85620cd19ea850f9c779d6a81754d45ed6207f88b0aa1eb6105226da4d.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/c6d28a85620cd19ea850f9c779d6a81754d45ed6207f88b0aa1eb6105226da4d.jpg)

![c91a745f98294c8b8a38860a7bd495fa696acb90d2780cdebb5fb2a365d33b58.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/c91a745f98294c8b8a38860a7bd495fa696acb90d2780cdebb5fb2a365d33b58.jpg)

![cc5f780c1f32653a4a3fd1abffb49c01e5eff40f65076c866c1f7d34953337ae.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/cc5f780c1f32653a4a3fd1abffb49c01e5eff40f65076c866c1f7d34953337ae.jpg)

![d0b93589c5720f300491a4a11f94f5a0462c5c2b9b7e4c6ba92403f46103ef2c.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/d0b93589c5720f300491a4a11f94f5a0462c5c2b9b7e4c6ba92403f46103ef2c.jpg)

![d0c432f4b76727083cde3371539884d6f0dce2629afb35b4edec7b7e035d6a80.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/d0c432f4b76727083cde3371539884d6f0dce2629afb35b4edec7b7e035d6a80.jpg)

![d4dee49bfea333cf807c3f5de5d5bc940ed668d3e80667269e0e53d2f8f1717e.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/d4dee49bfea333cf807c3f5de5d5bc940ed668d3e80667269e0e53d2f8f1717e.jpg)

![e309791bbf33eaffc317bceba0e05d789c8a83a5852f6a408d9a60d9f0ba5106.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/e309791bbf33eaffc317bceba0e05d789c8a83a5852f6a408d9a60d9f0ba5106.jpg)

![e94915d9eed77e539ee3215b8ba6f8930419e4efde85ab8f8020010d4972dc00.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/e94915d9eed77e539ee3215b8ba6f8930419e4efde85ab8f8020010d4972dc00.jpg)

![fbfabfc4742e82479b05dd54e9f8ba8f632741b6da3c2640d4ca84efab675f96.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/fbfabfc4742e82479b05dd54e9f8ba8f632741b6da3c2640d4ca84efab675f96.jpg)

## FasterVAR: Plug-and-Play Acceleration for Visual Autoregressive Models


### Images

![222f97dcff71238dd7466afb65df9853645572bc758b594b00cfe526a6fca069.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/222f97dcff71238dd7466afb65df9853645572bc758b594b00cfe526a6fca069.jpg)

![25ce5cacd9a029f7af6dd8cbe52ccccd7f5ff8b1aae432b75ab7538656837acc.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/25ce5cacd9a029f7af6dd8cbe52ccccd7f5ff8b1aae432b75ab7538656837acc.jpg)

![39f2c2382152eb8e2da454e8a35ca24aa250d6928c4f5cae5aa5e83a9e29891b.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/39f2c2382152eb8e2da454e8a35ca24aa250d6928c4f5cae5aa5e83a9e29891b.jpg)

![5832874ea30796864d66bd7081303ea76d37d84187cabcf379a05dbf61215157.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/5832874ea30796864d66bd7081303ea76d37d84187cabcf379a05dbf61215157.jpg)

![6da40748f61872cf66945bcac129f1847f8caf9b833f86e5ef58502c39da9e58.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/6da40748f61872cf66945bcac129f1847f8caf9b833f86e5ef58502c39da9e58.jpg)

![7e59fbc135a3542827f0c87c7cdba9b23fdc8fa7408edc0194f1f368c985c100.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/7e59fbc135a3542827f0c87c7cdba9b23fdc8fa7408edc0194f1f368c985c100.jpg)

![84b75f0707b86641d36cddb2bd13116a715cbebbefc9cfe9e1c6227477a0efee.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/84b75f0707b86641d36cddb2bd13116a715cbebbefc9cfe9e1c6227477a0efee.jpg)

![91afe51e3c74c7a1fe443f67f1f63da319db2091e109441d1acfcac91989580e.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/91afe51e3c74c7a1fe443f67f1f63da319db2091e109441d1acfcac91989580e.jpg)

![ad1d0bf8f440487ec742a78ecca43c0a60179b6175335f5303daaf3e3c12ac9b.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/ad1d0bf8f440487ec742a78ecca43c0a60179b6175335f5303daaf3e3c12ac9b.jpg)

![b0dd7baadb9166cacd2d5a5f70da55b027e3c611c34203c6b1e577ca3d5b0a45.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/b0dd7baadb9166cacd2d5a5f70da55b027e3c611c34203c6b1e577ca3d5b0a45.jpg)

![b1c3216b4e5dc1221e5b7b9bff17e6c71a29acdd3c78007f1918029ad34e13e0.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/b1c3216b4e5dc1221e5b7b9bff17e6c71a29acdd3c78007f1918029ad34e13e0.jpg)

![bdfe1efd525e2dc8f9b8e3ac9f8875e7b790d7e04885d8d616bf30e8dd557063.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/bdfe1efd525e2dc8f9b8e3ac9f8875e7b790d7e04885d8d616bf30e8dd557063.jpg)

![c48f4db6ffb3502bfcc3ae80b544bcb68b0d57885c600c624520eed30d618263.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/c48f4db6ffb3502bfcc3ae80b544bcb68b0d57885c600c624520eed30d618263.jpg)

![f8d969a9124ece5975f015595da0d5e9edd0585ef82129ce543b9c7d31d97722.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/f8d969a9124ece5975f015595da0d5e9edd0585ef82129ce543b9c7d31d97722.jpg)

![ff8ea7779f6279d4f663468d604cbfedc788fa3e9758bbb7b00d850a8c14f2d6.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/images/ff8ea7779f6279d4f663468d604cbfedc788fa3e9758bbb7b00d850a8c14f2d6.jpg)

### Tables

![524fcce805ebb97e64b3f5435d7474818756a26c9646d5d79e140213cbc6fcec.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/524fcce805ebb97e64b3f5435d7474818756a26c9646d5d79e140213cbc6fcec.jpg)

![7347687fc32c95a194e4fa0fd395a5a9ec11da7dbea24b0267166809615b3d47.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/7347687fc32c95a194e4fa0fd395a5a9ec11da7dbea24b0267166809615b3d47.jpg)

![7549f0ed3ea2601fd826d95a3348db8f0c2b98279df37cfc64c562f9947009e0.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/7549f0ed3ea2601fd826d95a3348db8f0c2b98279df37cfc64c562f9947009e0.jpg)

![8968df161e47667738fb09e898d976c2473cd0721efe718061783d153cca4437.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/8968df161e47667738fb09e898d976c2473cd0721efe718061783d153cca4437.jpg)

![926a5826beee48f0ebb87f01de4454359eed05e44b8c3b84b4507920a8979810.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/926a5826beee48f0ebb87f01de4454359eed05e44b8c3b84b4507920a8979810.jpg)

![a354605c811bfa6caf69c95c0fd93212d4f5786c7d4a0821398f6e9a8ddae7b1.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/a354605c811bfa6caf69c95c0fd93212d4f5786c7d4a0821398f6e9a8ddae7b1.jpg)

![b1ea02c81f461435a9afddde4c4d0ffeb2a163fe1abfb8f08919345f5ef24b90.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/b1ea02c81f461435a9afddde4c4d0ffeb2a163fe1abfb8f08919345f5ef24b90.jpg)

![b6342453a68f034b7ed8c4367ac0284e4192750957224a282c6ccde193f80d41.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/b6342453a68f034b7ed8c4367ac0284e4192750957224a282c6ccde193f80d41.jpg)

![c98888bec64ca13a7686074712d43f96c745d8e0d9638bc9935cdaceacf61932.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/c98888bec64ca13a7686074712d43f96c745d8e0d9638bc9935cdaceacf61932.jpg)

![d8147720305e3b9375d1910c3ae6cf61430a53bbdacc1137174c604911f0eb5d.jpg](../icml_results/2877_FasterVAR_%20Plug-and-Play%20Acceleration%20for%20Visual%20Autoregressive%20Models/tables/d8147720305e3b9375d1910c3ae6cf61430a53bbdacc1137174c604911f0eb5d.jpg)

<a id="paper-2840"></a>

## Mind the State: Towards Unified, Context-Aware EEG-to-fMRI Synthesis


### Images

![2216576164f0b15f569ddaad185f22832fb3ca13d56800ccd71a8dadb03a799a.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/2216576164f0b15f569ddaad185f22832fb3ca13d56800ccd71a8dadb03a799a.jpg)

![2cc5623050a1ec90216680a2288781e4eda55b1eb5db7c70a147ad1a1bf5ab32.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/2cc5623050a1ec90216680a2288781e4eda55b1eb5db7c70a147ad1a1bf5ab32.jpg)

![35870fd1f3e6a96a8d5846882999a87a718dba100c40b67b388e07236b858e8e.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/35870fd1f3e6a96a8d5846882999a87a718dba100c40b67b388e07236b858e8e.jpg)

![3d8051c6bd3c054fa9b70f6f9ccaf36adfb1c9289568c15d21cf9d4d8472ab0f.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/3d8051c6bd3c054fa9b70f6f9ccaf36adfb1c9289568c15d21cf9d4d8472ab0f.jpg)

![433754559dac21e229442e12bfd17a304e6ec01ef9aa0e181c969cade65c5998.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/433754559dac21e229442e12bfd17a304e6ec01ef9aa0e181c969cade65c5998.jpg)

![4da362ba4d26ac4b5b79e852c0e5adc88376c834fd905f29cacfdac15d240377.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/4da362ba4d26ac4b5b79e852c0e5adc88376c834fd905f29cacfdac15d240377.jpg)

![8cee8fc4575eb9c41bae0623f1901067a07dc850360159e7c38323f1d6b423db.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/8cee8fc4575eb9c41bae0623f1901067a07dc850360159e7c38323f1d6b423db.jpg)

![d1c1ec4e65baf824a635e9159f7a52f2ed27bf5c44d0983481b45810cdee28a0.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/d1c1ec4e65baf824a635e9159f7a52f2ed27bf5c44d0983481b45810cdee28a0.jpg)

![d1da029db40d37b5337aaa3a643dcf54a67e28531ef669eff7fcdc396e6307af.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/d1da029db40d37b5337aaa3a643dcf54a67e28531ef669eff7fcdc396e6307af.jpg)

![f3564757b639337a029ab3509223658af81586c866ea1eb73497a81f1fb68583.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/images/f3564757b639337a029ab3509223658af81586c866ea1eb73497a81f1fb68583.jpg)

### Tables

![2d0c4d068c37a37e0062d37181e6e531720c0ad5a3be9fb23541b7963c9c94b8.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/2d0c4d068c37a37e0062d37181e6e531720c0ad5a3be9fb23541b7963c9c94b8.jpg)

![4c0592d5e27bd9f9b7e1f2d226e8bff86b4c9b664e6123126a130a38ccb949e6.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/4c0592d5e27bd9f9b7e1f2d226e8bff86b4c9b664e6123126a130a38ccb949e6.jpg)

![5670f2ea56d07fe52144d33b6b39dbc6f07d69b512d50e8265ce37715f702b89.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/5670f2ea56d07fe52144d33b6b39dbc6f07d69b512d50e8265ce37715f702b89.jpg)

![5b3370f030c8eab4202ec15a4189a37c8d7fae634f2806f12acb7278e1b6fe6f.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/5b3370f030c8eab4202ec15a4189a37c8d7fae634f2806f12acb7278e1b6fe6f.jpg)

![780f74501498302bb911dd5fc95785ed4763e4be534eb79385ac60d923ab7a71.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/780f74501498302bb911dd5fc95785ed4763e4be534eb79385ac60d923ab7a71.jpg)

![7fe140b7a7562f89dcfbf40d2c820425012fc2bbc788667b7412d090f94f26e5.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/7fe140b7a7562f89dcfbf40d2c820425012fc2bbc788667b7412d090f94f26e5.jpg)

![8092a48cf09bc6e5c85e6536bfb9e51bea13408a1ef6e0f66779f319c4678c43.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/8092a48cf09bc6e5c85e6536bfb9e51bea13408a1ef6e0f66779f319c4678c43.jpg)

![a6be0f4364ca2bdf9c7bbefd73e3c6ccd67407227dd5bcea35cef4e3f1ae09e3.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/a6be0f4364ca2bdf9c7bbefd73e3c6ccd67407227dd5bcea35cef4e3f1ae09e3.jpg)

![ab2dc5e868caa96a71f30e968ce79a111b1bf4a88793414b7c6c6392e54409ee.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/ab2dc5e868caa96a71f30e968ce79a111b1bf4a88793414b7c6c6392e54409ee.jpg)

![bbdc4b77e5234df2e944419c5f2c8d67d5d2f0d78d6be74a46a2b51f065dd01f.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/bbdc4b77e5234df2e944419c5f2c8d67d5d2f0d78d6be74a46a2b51f065dd01f.jpg)

![c1c73fdcf76e00e15b5cc0985b7b191923ac204fcb1d25234f2d8773c80aacef.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/c1c73fdcf76e00e15b5cc0985b7b191923ac204fcb1d25234f2d8773c80aacef.jpg)

![ce457ea0da1d7cff1368c24ebdaaee603170072f97ad29b30dd12b819ab33ac5.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/ce457ea0da1d7cff1368c24ebdaaee603170072f97ad29b30dd12b819ab33ac5.jpg)

![e81f5f4d2dded341ad80c523a32d40911ad1430885ff4b9e7c2e6daa0d9aea62.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/e81f5f4d2dded341ad80c523a32d40911ad1430885ff4b9e7c2e6daa0d9aea62.jpg)

![f953877897f1401cd9e376d5c23f5c1013e36e35898c81e5986a61db389a218e.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/f953877897f1401cd9e376d5c23f5c1013e36e35898c81e5986a61db389a218e.jpg)

![fe4b67b7108d5d8b0c363bc4e40f8d6e2e079e44d0a3491cb48dd22851ec2529.jpg](../icml_results/2878_Mind%20the%20State_%20Towards%20Unified%2C%20Context-Aware%20EEG-to-fMRI%20Synthesis/tables/fe4b67b7108d5d8b0c363bc4e40f8d6e2e079e44d0a3491cb48dd22851ec2529.jpg)

<a id="paper-2841"></a>

## ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards


### Images

![04c510fd16a10833835012b72caf99deb0cbafbad1b6b0f2063cbc1a534bdc72.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/04c510fd16a10833835012b72caf99deb0cbafbad1b6b0f2063cbc1a534bdc72.jpg)

![1ef80f95ee3c12125e4521e95d947c37daffd1ab327cf983ed7564395d91b255.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/1ef80f95ee3c12125e4521e95d947c37daffd1ab327cf983ed7564395d91b255.jpg)

![28e818c1133da913be3c32a3c29980ff24857b08000b8bb7d9fe6ff4bd287fe2.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/28e818c1133da913be3c32a3c29980ff24857b08000b8bb7d9fe6ff4bd287fe2.jpg)

![2ae245e2819039568792c476b6cf5e79cd9da8f8aca6ac025b8b8a8dc86969ef.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/2ae245e2819039568792c476b6cf5e79cd9da8f8aca6ac025b8b8a8dc86969ef.jpg)

![5c4f2757b78601a0eb586529ba67d61fc65d4bbfb8171ed583794ba169b44e8f.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/5c4f2757b78601a0eb586529ba67d61fc65d4bbfb8171ed583794ba169b44e8f.jpg)

![a161c435750832183beff66fbb14282049081637fab1ae8e5e8cac44af8daf99.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/a161c435750832183beff66fbb14282049081637fab1ae8e5e8cac44af8daf99.jpg)

![a5425d679087b4eae8231ed8b9dc7d8b671ffcd1dd5d413ec40e8c78dcb52d0c.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/a5425d679087b4eae8231ed8b9dc7d8b671ffcd1dd5d413ec40e8c78dcb52d0c.jpg)

![b1668232aaf68487b0b757339e6cbb32d04b762ec6159a3d8afae393fb691ab2.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/b1668232aaf68487b0b757339e6cbb32d04b762ec6159a3d8afae393fb691ab2.jpg)

![bd222249cb6f0227ed0d4c9d185cddaa6c12a6073d89823d96719ed2f15001a3.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/bd222249cb6f0227ed0d4c9d185cddaa6c12a6073d89823d96719ed2f15001a3.jpg)

![d1ef47dd62c457783df15788a4d1de0851c53dea78c4bf2cb915c5e1ecfbfe5b.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/images/d1ef47dd62c457783df15788a4d1de0851c53dea78c4bf2cb915c5e1ecfbfe5b.jpg)

### Tables

![0854ed4c918c5c3cba4806423cccfded4c91abac23978a3cdad308d7ec8529e9.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/0854ed4c918c5c3cba4806423cccfded4c91abac23978a3cdad308d7ec8529e9.jpg)

![155b8f592871dac93ab8dcd7e86f8b41afcbff8537f7e4beeb27c9845a00dadc.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/155b8f592871dac93ab8dcd7e86f8b41afcbff8537f7e4beeb27c9845a00dadc.jpg)

![1711bf8933e9ed349d4d492703a5398291209cb28da035df8cba05ecc7e91d64.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/1711bf8933e9ed349d4d492703a5398291209cb28da035df8cba05ecc7e91d64.jpg)

![1abb968e5badadb66ae58dd3d351df54aa952e8170e2cd9e28e3402015b65a56.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/1abb968e5badadb66ae58dd3d351df54aa952e8170e2cd9e28e3402015b65a56.jpg)

![4f9d60fea6ea422b4b3174019657a674ae53bd5f5265fc4aa57861aca8e00849.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/4f9d60fea6ea422b4b3174019657a674ae53bd5f5265fc4aa57861aca8e00849.jpg)

![53d279906847a98fafab4c379319271e9be5444699541b72c4ae4ace1c8abf5a.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/53d279906847a98fafab4c379319271e9be5444699541b72c4ae4ace1c8abf5a.jpg)

![61ca940f87c0bd0c1f58ff04aa6d0e4878c25897bbce329c0b6252fc417a17c6.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/61ca940f87c0bd0c1f58ff04aa6d0e4878c25897bbce329c0b6252fc417a17c6.jpg)

![7534d7f98a17faf7e86afda35e5ba33e894eb81a81235585d911517241c7e3c6.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/7534d7f98a17faf7e86afda35e5ba33e894eb81a81235585d911517241c7e3c6.jpg)

![76e802c44047c90e8a94e86ded87a9e54591578a3d0f68e6a7b07b4b1804957b.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/76e802c44047c90e8a94e86ded87a9e54591578a3d0f68e6a7b07b4b1804957b.jpg)

![981e4cadf55227de84aace53f88d8f9ca1675d0d0a321fa46c39ce49a5fba9a0.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/981e4cadf55227de84aace53f88d8f9ca1675d0d0a321fa46c39ce49a5fba9a0.jpg)

![a8e52608388e0c745c9b755d01951e049a8f867a7b7db944aaada44388fd8a67.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/a8e52608388e0c745c9b755d01951e049a8f867a7b7db944aaada44388fd8a67.jpg)

![a912d1bc374c1f56f240fec333ff64dcbb4420539c1d372390410cdbdd54db36.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/a912d1bc374c1f56f240fec333ff64dcbb4420539c1d372390410cdbdd54db36.jpg)

![b2e1d8c91e9b20e5f3ef50ab39d7c3358235028101c777ec9b7175285dc60fca.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/b2e1d8c91e9b20e5f3ef50ab39d7c3358235028101c777ec9b7175285dc60fca.jpg)

![ba9c35ac086fd93f1500e3b1616f5f35ae795a67b4a42df74b83da792991991e.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/ba9c35ac086fd93f1500e3b1616f5f35ae795a67b4a42df74b83da792991991e.jpg)

![bf62aca07acddb6a39b6bdc19dc21094a09410c286a0032d068fa477a96ab4b2.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/bf62aca07acddb6a39b6bdc19dc21094a09410c286a0032d068fa477a96ab4b2.jpg)

![c9015c1bd0a2465ebd3db612dca4c5d4feab8bf3102fd3e5341a71ad2288bb3c.jpg](../icml_results/2879_ReSeek_%20A%20Self-Correcting%20Framework%20for%20Search%20Agents%20with%20Instructive%20Rewards/tables/c9015c1bd0a2465ebd3db612dca4c5d4feab8bf3102fd3e5341a71ad2288bb3c.jpg)

<a id="paper-2842"></a>

## Expectation Alignment of Language Models for Real-World User Expectations


### Images

![193af5dc363e5a296f3619bf451118327fe3638844083a39e76b013eb50c0848.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/193af5dc363e5a296f3619bf451118327fe3638844083a39e76b013eb50c0848.jpg)

![288a1cb245f7daf90337f1e4fc5b6bc1c2ecfa4081a4a9b0ca46c6558ad85139.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/288a1cb245f7daf90337f1e4fc5b6bc1c2ecfa4081a4a9b0ca46c6558ad85139.jpg)

![293acaad595051552e0d44c7cb0b0c65acdfc479c6a388266bc702df906fb3f2.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/293acaad595051552e0d44c7cb0b0c65acdfc479c6a388266bc702df906fb3f2.jpg)

![29ffd5719ed767e955407c5cde0cfcbdf47dd81d79772e43b9415d78dbd14761.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/29ffd5719ed767e955407c5cde0cfcbdf47dd81d79772e43b9415d78dbd14761.jpg)

![473de29eea10a0a3a86846c95f6005d7d060937c40ed65189fbedfbd32a272ab.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/473de29eea10a0a3a86846c95f6005d7d060937c40ed65189fbedfbd32a272ab.jpg)

![50eed05c81e98dc396a91027a75698682ee921505abe8a42c7749c79644323df.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/50eed05c81e98dc396a91027a75698682ee921505abe8a42c7749c79644323df.jpg)

![611e5b9bb64f63243df6f7666b6aae18a992c8009ab6252ea64c9ee0d569169d.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/611e5b9bb64f63243df6f7666b6aae18a992c8009ab6252ea64c9ee0d569169d.jpg)

![6cc4c1118a146c70c4894249d075e1f8dfa389c28dec0f7438ea3c28ba6938d9.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/6cc4c1118a146c70c4894249d075e1f8dfa389c28dec0f7438ea3c28ba6938d9.jpg)

![7e9e4341b8c26f9b5c8b64de8acf4feea4b86c45f3ebed686e1fd864c41e5f46.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/7e9e4341b8c26f9b5c8b64de8acf4feea4b86c45f3ebed686e1fd864c41e5f46.jpg)

![9e9cf74ce3d91e96319c0fdaff5413a0994211ea10fcfa5aad5b3d4dbccb0696.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/9e9cf74ce3d91e96319c0fdaff5413a0994211ea10fcfa5aad5b3d4dbccb0696.jpg)

![c352f62c49fe75185a9e351617b1d9351f2482e10fca71248c1733f714b13930.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/c352f62c49fe75185a9e351617b1d9351f2482e10fca71248c1733f714b13930.jpg)

![de3276d4f13548abbce024370d668949ecc8e684d9017d550318ef7c6c442c4b.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/de3276d4f13548abbce024370d668949ecc8e684d9017d550318ef7c6c442c4b.jpg)

![e33e436e65669ef6c780fc40c612209af47897b8e78a6b7a83a09697ef448cac.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/images/e33e436e65669ef6c780fc40c612209af47897b8e78a6b7a83a09697ef448cac.jpg)

### Tables

![06634440e4109323a1ffd19eea244b811a9f108d170cdf084051811cb3cc0dfe.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/tables/06634440e4109323a1ffd19eea244b811a9f108d170cdf084051811cb3cc0dfe.jpg)

![07b7193a55d3278a2a956a7ba7405d64526009516d961e02624ee5ba8ff85d6f.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/tables/07b7193a55d3278a2a956a7ba7405d64526009516d961e02624ee5ba8ff85d6f.jpg)

![8838ffa9183e2a7460bb8fffac2de6cf411bc147f5112c51146bc10e3cbae6a9.jpg](../icml_results/2880_Expectation%20Alignment%20of%20Language%20Models%20for%20Real-World%20User%20Expectations/tables/8838ffa9183e2a7460bb8fffac2de6cf411bc147f5112c51146bc10e3cbae6a9.jpg)

<a id="paper-2843"></a>

## Retro-Expert: Collaborative Reasoning for Interpretable Retrosynthesis


### Images

![0695e6db5e72f44e0cc72d144afb3a7e4d12d0bd1e85a29ae76958385b6d6273.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/0695e6db5e72f44e0cc72d144afb3a7e4d12d0bd1e85a29ae76958385b6d6273.jpg)

![0912e11356b11553f12740f713bab6f38ac1ab3d7a633d7069084af13857bee5.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/0912e11356b11553f12740f713bab6f38ac1ab3d7a633d7069084af13857bee5.jpg)

![0eeb9bcb20e018403a16f7df18a8fb81268c60a6b2460de1e44c06019c9f9756.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/0eeb9bcb20e018403a16f7df18a8fb81268c60a6b2460de1e44c06019c9f9756.jpg)

![1575872364434d9157fab1ce3023c738745e4a96d878b754aa51fda7e3df4683.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/1575872364434d9157fab1ce3023c738745e4a96d878b754aa51fda7e3df4683.jpg)

![1f6946ff3e484b63fcd1b5f6c0a02847bf07e16c54bed388b6ef1f0730204db2.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/1f6946ff3e484b63fcd1b5f6c0a02847bf07e16c54bed388b6ef1f0730204db2.jpg)

![220a8d76657c3d53eb2e752b6d998950a319dda2d792bc11c4dc3cba90c91dcb.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/220a8d76657c3d53eb2e752b6d998950a319dda2d792bc11c4dc3cba90c91dcb.jpg)

![325e332a47f8d5249b5199e96763ac202d78d3ae0e6643c79444cc326e191574.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/325e332a47f8d5249b5199e96763ac202d78d3ae0e6643c79444cc326e191574.jpg)

![3972a5f790ab777c92442bd02fe0fdad4498d1ffb96dac3d8dcd60d5da104da0.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/3972a5f790ab777c92442bd02fe0fdad4498d1ffb96dac3d8dcd60d5da104da0.jpg)

![69adc3dcf867f9f924c74aaa5861cf5afe6c8d62cbeb84f6cbef7f6fe4dbf2f8.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/69adc3dcf867f9f924c74aaa5861cf5afe6c8d62cbeb84f6cbef7f6fe4dbf2f8.jpg)

![6be508b64044bb4c3f0abb9bb7fc57ded7350976f7764c8f91feda1ae010b85b.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/6be508b64044bb4c3f0abb9bb7fc57ded7350976f7764c8f91feda1ae010b85b.jpg)

![737f43a13e6bc27cb47500e5ccfa28f727953d520a6f3af058b54c9ba5a1b576.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/737f43a13e6bc27cb47500e5ccfa28f727953d520a6f3af058b54c9ba5a1b576.jpg)

![77796229cb610a1b9f623c98b21146c6fe69cf28d39789223dd3341602e98b78.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/77796229cb610a1b9f623c98b21146c6fe69cf28d39789223dd3341602e98b78.jpg)

![7e571d31218a7cf730d45be4915aaa6fcefb80df9423e96be20d46e1c64b5819.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/7e571d31218a7cf730d45be4915aaa6fcefb80df9423e96be20d46e1c64b5819.jpg)

![bf8bf00a202e1c0ee41087f9cb21ede6d92b4be41f576d0f5fd463be692b103f.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/bf8bf00a202e1c0ee41087f9cb21ede6d92b4be41f576d0f5fd463be692b103f.jpg)

![c08892a0175e0445a8594d5c875fbd7f5ee17b723ebe3aca234b42a4084656cd.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/c08892a0175e0445a8594d5c875fbd7f5ee17b723ebe3aca234b42a4084656cd.jpg)

![c0ecc78a5091a994485b919c1cda31b653a88766237077ea0c08df4f1a3e6a87.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/c0ecc78a5091a994485b919c1cda31b653a88766237077ea0c08df4f1a3e6a87.jpg)

![c8571684363d21fc70e2754d006aaf76a15694e6ac5b9b9d6fd82bc5d7293360.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/c8571684363d21fc70e2754d006aaf76a15694e6ac5b9b9d6fd82bc5d7293360.jpg)

![def91b1c65f7e5a918420a7381a7b9975faefc37d01373d46a48dec5cde64de5.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/def91b1c65f7e5a918420a7381a7b9975faefc37d01373d46a48dec5cde64de5.jpg)

![e21f28fecf5e54ee3ad73ec77da8937715b51c504e0d6964d63a0c93758475f8.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/e21f28fecf5e54ee3ad73ec77da8937715b51c504e0d6964d63a0c93758475f8.jpg)

![e5abc5356a69a313f98727cc09a6cf942ef7bfe96859508f1deefac5b68e260d.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/e5abc5356a69a313f98727cc09a6cf942ef7bfe96859508f1deefac5b68e260d.jpg)

![ed63c0505dd7149be8d3e3fe3716fc18259fd32696b99a1a25b8ffcf02e67577.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/ed63c0505dd7149be8d3e3fe3716fc18259fd32696b99a1a25b8ffcf02e67577.jpg)

![f54c9865f67c2e3f61de2e129d58411a0653839968c0e78ccf06baee84101f8e.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/images/f54c9865f67c2e3f61de2e129d58411a0653839968c0e78ccf06baee84101f8e.jpg)

### Tables

![0711df2c18c1b7d25396cb40d916e8d0273640736285194d26e82ea3b1fdfdaa.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/0711df2c18c1b7d25396cb40d916e8d0273640736285194d26e82ea3b1fdfdaa.jpg)

![0bb9be86f1a60e742a6e3b53d1e7f0739f5eca89f8d55c0d09b35082c761d838.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/0bb9be86f1a60e742a6e3b53d1e7f0739f5eca89f8d55c0d09b35082c761d838.jpg)

![1a8e9536758d7b0664c5ce9f438c69855639f0c13adcd7bdddd6056a6662a42c.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/1a8e9536758d7b0664c5ce9f438c69855639f0c13adcd7bdddd6056a6662a42c.jpg)

![1b5704eff15bc7cd371a1ef35965f5584c1e1cbb7fdeb2ea979e597a71050ef9.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/1b5704eff15bc7cd371a1ef35965f5584c1e1cbb7fdeb2ea979e597a71050ef9.jpg)

![1bd1c912aa5d9874f73b6ab6dac43e6c41cbd57fb19023d44fadf4c573dbb410.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/1bd1c912aa5d9874f73b6ab6dac43e6c41cbd57fb19023d44fadf4c573dbb410.jpg)

![255ecd69315f4700e68cc666430a91f74c62767e1742d14b94a40acd1ddefdea.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/255ecd69315f4700e68cc666430a91f74c62767e1742d14b94a40acd1ddefdea.jpg)

![25a4c6dd3703b713be6e46e7fcb31c1e63d029f2c196907d8f0e67201832d152.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/25a4c6dd3703b713be6e46e7fcb31c1e63d029f2c196907d8f0e67201832d152.jpg)

![41f40cbbb5b70fa369e82482d74aa9ed843770b3daf3a5537bacced9b9c4487e.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/41f40cbbb5b70fa369e82482d74aa9ed843770b3daf3a5537bacced9b9c4487e.jpg)

![49bae24ff344cdf6debad03b6f1a974cc436834e73044cdcdaf00fd7584325cf.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/49bae24ff344cdf6debad03b6f1a974cc436834e73044cdcdaf00fd7584325cf.jpg)

![6fcb10bccaffea8a31b5acff1a09462f0675d6174a7c159c02936b60a406d0f5.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/6fcb10bccaffea8a31b5acff1a09462f0675d6174a7c159c02936b60a406d0f5.jpg)

![7fed91aca4971cf6d6a06705e2ef155ab143ceb483713954ac7bcd706472f319.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/7fed91aca4971cf6d6a06705e2ef155ab143ceb483713954ac7bcd706472f319.jpg)

![95fc3169be5b4c6f99dbcd9ed116c0f8a823fb781f32b57b381d7c1d3bbf7517.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/95fc3169be5b4c6f99dbcd9ed116c0f8a823fb781f32b57b381d7c1d3bbf7517.jpg)

![a0c5eaf2a9bd4da901a58460794db741cfb4a5b46f23e477b249cba9f0cd6643.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/a0c5eaf2a9bd4da901a58460794db741cfb4a5b46f23e477b249cba9f0cd6643.jpg)

![a954e00fffe2f57d05fefe27a2a4178e69ac5cc653f827e54869591ab640b68c.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/a954e00fffe2f57d05fefe27a2a4178e69ac5cc653f827e54869591ab640b68c.jpg)

![d1049bc6bfd14a12b4e9558983618bb237a4e037bce0a2f0e25e214fc7b1cec2.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/d1049bc6bfd14a12b4e9558983618bb237a4e037bce0a2f0e25e214fc7b1cec2.jpg)

![d749b0efb646cb395614aaf5cc710dad7122ba2aae3f7e35fe0f2a6f62e99269.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/d749b0efb646cb395614aaf5cc710dad7122ba2aae3f7e35fe0f2a6f62e99269.jpg)

![dcd48866dd3885ff484fd239b5411a0c7adfcc35e913ce27e03c1eefc9e7aa48.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/dcd48866dd3885ff484fd239b5411a0c7adfcc35e913ce27e03c1eefc9e7aa48.jpg)

![e59dc891e5fed91593cc3ba8e2f23840ba7dd0150fd2ba50278b0473beda7af4.jpg](../icml_results/2881_Retro-Expert_%20Collaborative%20Reasoning%20for%20Interpretable%20Retrosynthesis/tables/e59dc891e5fed91593cc3ba8e2f23840ba7dd0150fd2ba50278b0473beda7af4.jpg)

<a id="paper-2844"></a>

## Towards Trustworthy and Identifiable Virtual Face Generation


### Images

![22fcbfc41a590f305cad5f8eb9b671df008d1440133634764ed933f1941c23f6.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/images/22fcbfc41a590f305cad5f8eb9b671df008d1440133634764ed933f1941c23f6.jpg)

![615574ec09646fc0405d3d51603f2c482970688dd99814d073a4047f4f86361c.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/images/615574ec09646fc0405d3d51603f2c482970688dd99814d073a4047f4f86361c.jpg)

![76a0d13fb061c719740cb5ca1150dc7f45e750b1aa34349a8298afbc409b18ba.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/images/76a0d13fb061c719740cb5ca1150dc7f45e750b1aa34349a8298afbc409b18ba.jpg)

![92281a30cef1744c615e0c74f98c28332fa396bf9030410cbc1acac9e85ed4ed.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/images/92281a30cef1744c615e0c74f98c28332fa396bf9030410cbc1acac9e85ed4ed.jpg)

![99c2c2343002a57e12e17424e400bb03cebe2476f784ed77d8507e655c1a8dae.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/images/99c2c2343002a57e12e17424e400bb03cebe2476f784ed77d8507e655c1a8dae.jpg)

### Tables

![097b9433398df894f357fbfaa53dd4a5e3be825f0c4f9c4340b76f8562aa0c79.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/097b9433398df894f357fbfaa53dd4a5e3be825f0c4f9c4340b76f8562aa0c79.jpg)

![0e48934172e05c1a6ba85d7f93488dc7abd257d54923a4ab5f60ededbc264db1.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/0e48934172e05c1a6ba85d7f93488dc7abd257d54923a4ab5f60ededbc264db1.jpg)

![461b59a21b9eb6ec7d5a5a584b3a162273c7e3cecef87e1af480d94a06901ef8.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/461b59a21b9eb6ec7d5a5a584b3a162273c7e3cecef87e1af480d94a06901ef8.jpg)

![5f6dd7471bf7b638102976d551577a19fc0f8142d24273ec798f3499de25b271.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/5f6dd7471bf7b638102976d551577a19fc0f8142d24273ec798f3499de25b271.jpg)

![8aab38d4025124f0744952dbcd198530fa21efd8e8ce47937ee8910500e7bdce.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/8aab38d4025124f0744952dbcd198530fa21efd8e8ce47937ee8910500e7bdce.jpg)

![aef1ff798b535618750c0a9d75982c727be82c2579939ae14701947203053cee.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/aef1ff798b535618750c0a9d75982c727be82c2579939ae14701947203053cee.jpg)

![fae98164c32cb5bc72a72278d2a59321a3fac8af2d95f14ffeec27af174650c6.jpg](../icml_results/2882_Towards%20Trustworthy%20and%20Identifiable%20Virtual%20Face%20Generation/tables/fae98164c32cb5bc72a72278d2a59321a3fac8af2d95f14ffeec27af174650c6.jpg)

<a id="paper-2845"></a>

## CELL: A Causal Perspective for Fairness-aware Graph Adaptation


### Images

![14b6f8eeda2d112cf238290fe6a60c501e997116c05933cee2566a9f61c8a078.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/14b6f8eeda2d112cf238290fe6a60c501e997116c05933cee2566a9f61c8a078.jpg)

![30ef4cfa91789f67953083668763f0f61e2d0e87176e07479ff8258e63ed2c9f.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/30ef4cfa91789f67953083668763f0f61e2d0e87176e07479ff8258e63ed2c9f.jpg)

![4e30f3d4f1b242e4f4fe9fb94b4a6985d9997eef363726bacbd4b8c711aacbd6.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/4e30f3d4f1b242e4f4fe9fb94b4a6985d9997eef363726bacbd4b8c711aacbd6.jpg)

![67c9c091f37322e9e3ed5b43f45e7d646145db80c65fcdb2d64aeb3fad112826.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/67c9c091f37322e9e3ed5b43f45e7d646145db80c65fcdb2d64aeb3fad112826.jpg)

![9679a3d960aee451315ed3c828d71305c8d16bcede21818e0648621fd18631c0.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/9679a3d960aee451315ed3c828d71305c8d16bcede21818e0648621fd18631c0.jpg)

![9eb1567b55eb3863c8d0d50b3cf01da83097fd8837fee62617fb010e57b450f7.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/9eb1567b55eb3863c8d0d50b3cf01da83097fd8837fee62617fb010e57b450f7.jpg)

![bf31ba8e8aaeb0172a3feee69ef93435cae00d9f0368318a2b5cbb9b35878516.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/bf31ba8e8aaeb0172a3feee69ef93435cae00d9f0368318a2b5cbb9b35878516.jpg)

![ffab2b78c88ca9e9d2aca524fcfd9748b95c1003156f4e4db8349a35fa6241ea.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/images/ffab2b78c88ca9e9d2aca524fcfd9748b95c1003156f4e4db8349a35fa6241ea.jpg)

### Tables

![03305893b6d123930dce845737b8a7cc46058bb6b5a224cb25be90a716cd5748.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/tables/03305893b6d123930dce845737b8a7cc46058bb6b5a224cb25be90a716cd5748.jpg)

![21dc2ab7f81acd0667cbb85b11ae11485e588d4b2a64f9c31029a3596ec4ceab.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/tables/21dc2ab7f81acd0667cbb85b11ae11485e588d4b2a64f9c31029a3596ec4ceab.jpg)

![796440d2e36af4e60c907b80abbe721906ad912a5b84af82a5053546d7367ae4.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/tables/796440d2e36af4e60c907b80abbe721906ad912a5b84af82a5053546d7367ae4.jpg)

![aca219766481553cf48474ded1fdbc6dbcd077c0b2b725d7c2d8c2b1d1ef6a27.jpg](../icml_results/2883_CELL_%20A%20Causal%20Perspective%20for%20Fairness-aware%20Graph%20Adaptation/tables/aca219766481553cf48474ded1fdbc6dbcd077c0b2b725d7c2d8c2b1d1ef6a27.jpg)

<a id="paper-2846"></a>

## BOOSTAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models


### Images

![a7711330c5c5c3d353a1f883f8781d5be04b922952f63d8c757ae0debfd194ad.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/images/a7711330c5c5c3d353a1f883f8781d5be04b922952f63d8c757ae0debfd194ad.jpg)

![cd84d63d555874488fdd65d45e69d76229e5f73760438f740cc33962d4344a2b.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/images/cd84d63d555874488fdd65d45e69d76229e5f73760438f740cc33962d4344a2b.jpg)

### Tables

![0f1105bd8bca87c62061799b66dca59b3d2616bf70050f07f5a8bec67bc6ac71.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/0f1105bd8bca87c62061799b66dca59b3d2616bf70050f07f5a8bec67bc6ac71.jpg)

![19e0460a96f4a8edff3c4022cfcc17ee9c5eb9799c25723db3e6aa8c4f9a6591.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/19e0460a96f4a8edff3c4022cfcc17ee9c5eb9799c25723db3e6aa8c4f9a6591.jpg)

![30b8fd66bcd1d5eda9ef028de9592aaeed2d2f8d4fcfbcc5a8a131eb4e364c81.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/30b8fd66bcd1d5eda9ef028de9592aaeed2d2f8d4fcfbcc5a8a131eb4e364c81.jpg)

![61bacc4d9526350bac03c9ff62311a8c8d18f51019457e80a6a7128a4eb1108d.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/61bacc4d9526350bac03c9ff62311a8c8d18f51019457e80a6a7128a4eb1108d.jpg)

![6ef89834f917d68e112ec70395396ffc89ec04e15633841600b9d3ff2df414b4.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/6ef89834f917d68e112ec70395396ffc89ec04e15633841600b9d3ff2df414b4.jpg)

![72a1c25543be5dcb9c91a590c23a210ad1a84ff8e6e5545c0ca030cbd2a28cdd.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/72a1c25543be5dcb9c91a590c23a210ad1a84ff8e6e5545c0ca030cbd2a28cdd.jpg)

![804344805eca480e164b00468478c1cc6984f4a55a1faa0c3a2af3fdb13f2525.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/804344805eca480e164b00468478c1cc6984f4a55a1faa0c3a2af3fdb13f2525.jpg)

![87bf2bc1033088e7662e844d260bf91265cc60c87b3b85405d9bde915247b939.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/87bf2bc1033088e7662e844d260bf91265cc60c87b3b85405d9bde915247b939.jpg)

![896cc77ec75878940f02ed0671b97a127ad32ee8dd79ab518f5d978cb526ba4b.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/896cc77ec75878940f02ed0671b97a127ad32ee8dd79ab518f5d978cb526ba4b.jpg)

![8abd0037f9aeec638be2ba4892c18b08e210768e0669b6017e2a5fb9d130cb88.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/8abd0037f9aeec638be2ba4892c18b08e210768e0669b6017e2a5fb9d130cb88.jpg)

![8d85126be2901a505137457af81f311e8a8ce9e2a13018b74a1ad7dad354727e.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/8d85126be2901a505137457af81f311e8a8ce9e2a13018b74a1ad7dad354727e.jpg)

![92ede0970367005bb80b7d5c0ae380ec992324840136d824193167a0eac45907.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/92ede0970367005bb80b7d5c0ae380ec992324840136d824193167a0eac45907.jpg)

![9305c9dca4883ad386db0a80299489a928ad2b1ff0b662e13b30cbfdaca20653.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/9305c9dca4883ad386db0a80299489a928ad2b1ff0b662e13b30cbfdaca20653.jpg)

![c60e9645c5b45cde684681f13ecfe7a2c7ad8391410bfec1f7e5717337db8416.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/c60e9645c5b45cde684681f13ecfe7a2c7ad8391410bfec1f7e5717337db8416.jpg)

![db07d4d4e38a0931de04419b4fa2dc57f5fb65743d62dceda40871600ce5efce.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/db07d4d4e38a0931de04419b4fa2dc57f5fb65743d62dceda40871600ce5efce.jpg)

![e0559447a8efeada2df7e8b63a419a4963a57a8b562ba4c05f89be36d5cdbecf.jpg](../icml_results/2884_BOOSTAPR_%20Boosting%20Automated%20Program%20Repair%20via%20Execution-Grounded%20Reinforcement%20Learning%20with%20Dual%20/tables/e0559447a8efeada2df7e8b63a419a4963a57a8b562ba4c05f89be36d5cdbecf.jpg)

<a id="paper-2847"></a>

## Sponge Tool Attack: Stealthy Denial-of-Efficiency against Tool-Augmented Agentic Reasoning


### Images

![0d80cc888eb3e1e00d358ea25ccc3ac7badad072ba45a228f0851e340109a11a.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/0d80cc888eb3e1e00d358ea25ccc3ac7badad072ba45a228f0851e340109a11a.jpg)

![48e21028e46b2d0c541db9d931c28925b46f904445a462b8d49733c4659fe777.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/48e21028e46b2d0c541db9d931c28925b46f904445a462b8d49733c4659fe777.jpg)

![4efc03d175dbe469b7a92705367bf3ce470711d50be40b4726b390125a9e5591.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/4efc03d175dbe469b7a92705367bf3ce470711d50be40b4726b390125a9e5591.jpg)

![639349aaf8fcd0018a0956aac08798f0e59798c85c7bb40dc873123906aca485.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/639349aaf8fcd0018a0956aac08798f0e59798c85c7bb40dc873123906aca485.jpg)

![6429f4aab6af049fcbfe5969d1d706db27e6a8727758f6958ec6f2492043837f.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/6429f4aab6af049fcbfe5969d1d706db27e6a8727758f6958ec6f2492043837f.jpg)

![942de82e744d39ccdd66216a7d76780f364393cdfec578beb9a93cd3aeb07e97.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/942de82e744d39ccdd66216a7d76780f364393cdfec578beb9a93cd3aeb07e97.jpg)

![aaa7539732f4a1894a13c4bfe5e634f63ff25bd063f31154a23c8d57557ded67.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/aaa7539732f4a1894a13c4bfe5e634f63ff25bd063f31154a23c8d57557ded67.jpg)

![b0c3e5ebe8642823070ef5eaf5a0a86a564c50323c20daa44e926d467f75067c.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/b0c3e5ebe8642823070ef5eaf5a0a86a564c50323c20daa44e926d467f75067c.jpg)

![d54507f742f8a1af6c18ea7bc7a508e759aa60ae5494fb10a0b5b7356889351e.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/d54507f742f8a1af6c18ea7bc7a508e759aa60ae5494fb10a0b5b7356889351e.jpg)

![e61941ce27c1792154bcdd3c7e225d25f160818e83b240ea1a6c29b566642f75.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/images/e61941ce27c1792154bcdd3c7e225d25f160818e83b240ea1a6c29b566642f75.jpg)

### Tables

![0d8d26ac14b69c256f2a2de1da62d59b5f68d0d3bcf82501a00a5b26d3ec191f.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/0d8d26ac14b69c256f2a2de1da62d59b5f68d0d3bcf82501a00a5b26d3ec191f.jpg)

![328a8508083bf4643098326e3128c400c79efba548513d9c4fd44a6aaeaef78d.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/328a8508083bf4643098326e3128c400c79efba548513d9c4fd44a6aaeaef78d.jpg)

![48c0b23310b5016b2ba79cd8188d5ff2197a464a0a1c7f76077da4a4a2e6d78a.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/48c0b23310b5016b2ba79cd8188d5ff2197a464a0a1c7f76077da4a4a2e6d78a.jpg)

![614d0505bde5df85b69660568246515e1d215328b7dc5ffa67f0a9d9f2825cf5.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/614d0505bde5df85b69660568246515e1d215328b7dc5ffa67f0a9d9f2825cf5.jpg)

![671562323b20d07e1b36aca017eb3e98b3e7826d50cfc868ec9f67022804b544.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/671562323b20d07e1b36aca017eb3e98b3e7826d50cfc868ec9f67022804b544.jpg)

![c2db619be8176d6556e232f6c31786bab130eebe9f9f91b91738bbfc7b3c163c.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/c2db619be8176d6556e232f6c31786bab130eebe9f9f91b91738bbfc7b3c163c.jpg)

![c4eb66b37ee31aa98ba2f12a9e5a99a20ae154ea607f898f0f10e0b8753e1d14.jpg](../icml_results/2885_Sponge%20Tool%20Attack_%20Stealthy%20Denial-of-Efficiency%20against%20Tool-Augmented%20Agentic%20Reasoning/tables/c4eb66b37ee31aa98ba2f12a9e5a99a20ae154ea607f898f0f10e0b8753e1d14.jpg)

<a id="paper-2848"></a>

## Cross-task Calibration for Asynchronous Federated Continual Learning


### Images

![54639036fbb5b8fe9bda8db953296605da23584f68d2cc9ac99de491426f305b.jpg](../icml_results/2886_Cross-task%20Calibration%20for%20Asynchronous%20Federated%20Continual%20Learning/images/54639036fbb5b8fe9bda8db953296605da23584f68d2cc9ac99de491426f305b.jpg)

![562d2b890dad0d5cc2a10da49292fa44d40c040d7da2c9c9f2fcdfb676025c5d.jpg](../icml_results/2886_Cross-task%20Calibration%20for%20Asynchronous%20Federated%20Continual%20Learning/images/562d2b890dad0d5cc2a10da49292fa44d40c040d7da2c9c9f2fcdfb676025c5d.jpg)

![9e313ddead29c75b088c20782d008249c6ff2916b213e3dd6a53f37e3fb015c4.jpg](../icml_results/2886_Cross-task%20Calibration%20for%20Asynchronous%20Federated%20Continual%20Learning/images/9e313ddead29c75b088c20782d008249c6ff2916b213e3dd6a53f37e3fb015c4.jpg)

![d8391c09f60fb7496c8c7a1d8cfb81f93b29255f1fd8c40df8e8f6f4344daf5e.jpg](../icml_results/2886_Cross-task%20Calibration%20for%20Asynchronous%20Federated%20Continual%20Learning/images/d8391c09f60fb7496c8c7a1d8cfb81f93b29255f1fd8c40df8e8f6f4344daf5e.jpg)

### Tables

![352c515e574311f0950c589295b292aba455e39e6d4a5bc744f9ed7a7b4bf60c.jpg](../icml_results/2886_Cross-task%20Calibration%20for%20Asynchronous%20Federated%20Continual%20Learning/tables/352c515e574311f0950c589295b292aba455e39e6d4a5bc744f9ed7a7b4bf60c.jpg)

![b601f181db86864c56391800f219675b02469a6dac5f1b07ea2fd59ab7bbc1f9.jpg](../icml_results/2886_Cross-task%20Calibration%20for%20Asynchronous%20Federated%20Continual%20Learning/tables/b601f181db86864c56391800f219675b02469a6dac5f1b07ea2fd59ab7bbc1f9.jpg)

<a id="paper-2849"></a>

## Prism-MoE: Efficient Dense-to-MoE Conversion for Visual Autoregressive Generation


### Images

![2d8d072d7703ef197fd2f39e48fa171906533475b9df641b776928b8aecdab66.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/2d8d072d7703ef197fd2f39e48fa171906533475b9df641b776928b8aecdab66.jpg)

![2f11b99efb8d3ca5c558d56a77db8f55220b12c5e88c45a6d0559248bc32d545.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/2f11b99efb8d3ca5c558d56a77db8f55220b12c5e88c45a6d0559248bc32d545.jpg)

![3bf448e92c12c5471efaca81e5c5adf52843b4741e92e6cb8a3cff0da10b5bae.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/3bf448e92c12c5471efaca81e5c5adf52843b4741e92e6cb8a3cff0da10b5bae.jpg)

![3dc1d166d3b44cada92057efcaf3a617fe7245dc8c0fdc0a7925149fe4612a4f.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/3dc1d166d3b44cada92057efcaf3a617fe7245dc8c0fdc0a7925149fe4612a4f.jpg)

![556ca512acf4b40b1f4bc75a5f3a82f5d4eaf3c766617c15d098534ce4eff27f.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/556ca512acf4b40b1f4bc75a5f3a82f5d4eaf3c766617c15d098534ce4eff27f.jpg)

![5669cae299a29e49017445f9e64e6e64a450b8a9b5f59eca0df77597e70a3c49.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/5669cae299a29e49017445f9e64e6e64a450b8a9b5f59eca0df77597e70a3c49.jpg)

![7704cebd711668881f635719e016711cb5b0c3c0e2101eed19bca0f8a80a77f5.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/7704cebd711668881f635719e016711cb5b0c3c0e2101eed19bca0f8a80a77f5.jpg)

![86f8f0af43c4ee2ef54e940a32d5af36821a6545086e3945c694cf1225681f04.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/86f8f0af43c4ee2ef54e940a32d5af36821a6545086e3945c694cf1225681f04.jpg)

![8a36efec0e6e5748761cf03a898b87af2a6f49bf1d80940c99c37e7e8bb621ae.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/8a36efec0e6e5748761cf03a898b87af2a6f49bf1d80940c99c37e7e8bb621ae.jpg)

![ad18843798061aa82262df52ce176f19ef9df259d807f760e595fb58089fa0d4.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/ad18843798061aa82262df52ce176f19ef9df259d807f760e595fb58089fa0d4.jpg)

![b2524adc7eba98490715623084fb045014b395d346d148c0a1a236934942dc32.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/b2524adc7eba98490715623084fb045014b395d346d148c0a1a236934942dc32.jpg)

![b2915fa9d797fd6c05ce2b49d427481c613081805f039ee00a9088741279340e.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/b2915fa9d797fd6c05ce2b49d427481c613081805f039ee00a9088741279340e.jpg)

![e9bf9a5c325fb4f464c2180636b4e63578779fcf18344179024b28878d326cb9.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/e9bf9a5c325fb4f464c2180636b4e63578779fcf18344179024b28878d326cb9.jpg)

![f4fa36020e047ac401aeeab0d425c3298dfcd5dc93b0ce7b171ee8d6b7dcb5c0.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/images/f4fa36020e047ac401aeeab0d425c3298dfcd5dc93b0ce7b171ee8d6b7dcb5c0.jpg)

### Tables

![00365e3675ca239329fecaf4ffff7db7505491092d9121c1354dfc277fbf117d.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/tables/00365e3675ca239329fecaf4ffff7db7505491092d9121c1354dfc277fbf117d.jpg)

![07346284afc6e5a20776a2019ed5fccdfb6d0196b915ca3136e30cd5d53d5fab.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/tables/07346284afc6e5a20776a2019ed5fccdfb6d0196b915ca3136e30cd5d53d5fab.jpg)

![1bd19f4a95eb16f23268d95fb313561c0d3f75a3f00eff8fcd18db8a08122a6f.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/tables/1bd19f4a95eb16f23268d95fb313561c0d3f75a3f00eff8fcd18db8a08122a6f.jpg)

![a9e6537698e294092a9ffc406dd9924612c5644dcc88ff4096a35ad3af1bc309.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/tables/a9e6537698e294092a9ffc406dd9924612c5644dcc88ff4096a35ad3af1bc309.jpg)

![acd33f017e1dd77a4c002c8e2bab27aa0d99766798fbf5db7e50f86af4955b0b.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/tables/acd33f017e1dd77a4c002c8e2bab27aa0d99766798fbf5db7e50f86af4955b0b.jpg)

![ae59551137d0c31b1b075f650fa121a2e647f3f8cdb6903a4350ec78feb12d4a.jpg](../icml_results/2887_Prism-MoE_%20Efficient%20Dense-to-MoE%20Conversion%20for%20Visual%20Autoregressive%20Generation/tables/ae59551137d0c31b1b075f650fa121a2e647f3f8cdb6903a4350ec78feb12d4a.jpg)

<a id="paper-2850"></a>

## LoRe: Adaptive Interaction-Evaluation Routing with Per-step Interaction Budgets for Iterative Graph Solvers


### Images

![132f11efb51a2300b395d836aab88f342c4391c4862b7182d5171f710ef68d17.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/images/132f11efb51a2300b395d836aab88f342c4391c4862b7182d5171f710ef68d17.jpg)

![36c50005328de0f3d7762e2f8cd9f415de3ebf6d01f8b07d62b31761a0ea96e2.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/images/36c50005328de0f3d7762e2f8cd9f415de3ebf6d01f8b07d62b31761a0ea96e2.jpg)

![cfb1f857ba10352cdb9c69bf5f35c31e69fec3e242707e0d37d79ffca862b332.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/images/cfb1f857ba10352cdb9c69bf5f35c31e69fec3e242707e0d37d79ffca862b332.jpg)

![e5d7bbc1a04f17cf1b34bdb56dd91fa985604fe03f9c1babadcc07bac92cd1c1.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/images/e5d7bbc1a04f17cf1b34bdb56dd91fa985604fe03f9c1babadcc07bac92cd1c1.jpg)

### Tables

![1e0837f37b5adef1aa1c2fad458b70197dc28bf4bfe692e4b6f24c4d70777ecc.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/tables/1e0837f37b5adef1aa1c2fad458b70197dc28bf4bfe692e4b6f24c4d70777ecc.jpg)

![3924144a0e635d018f3d33002e3d2fed7a3ca0ef6563afcdf8ad29f4f778ebf8.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/tables/3924144a0e635d018f3d33002e3d2fed7a3ca0ef6563afcdf8ad29f4f778ebf8.jpg)

![75635355d9387cefb6cbe3bdf185d54afd6bcd0613fc94f143b2e4604d5fc1e2.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/tables/75635355d9387cefb6cbe3bdf185d54afd6bcd0613fc94f143b2e4604d5fc1e2.jpg)

![98d08704f19bcd92ef65609e76abd7ea4145402f075737e1c9c05968fa6ee6c4.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/tables/98d08704f19bcd92ef65609e76abd7ea4145402f075737e1c9c05968fa6ee6c4.jpg)

![a2d269779ad80249a06314335e77d7b3eceb3a75479709018dbc1d5ffa76173e.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/tables/a2d269779ad80249a06314335e77d7b3eceb3a75479709018dbc1d5ffa76173e.jpg)

![d0784a5759fd4667d001ed2e5727dbf133472ed396c233c7213b9a02f0e96335.jpg](../icml_results/2888_LoRe_%20Adaptive%20Interaction-Evaluation%20Routing%20with%20Per-step%20Interaction%20Budgets%20for%20Iterative%20Graph%20/tables/d0784a5759fd4667d001ed2e5727dbf133472ed396c233c7213b9a02f0e96335.jpg)

<a id="paper-2851"></a>

## GOCM: Single-Step Graph Outlier Synthesis via Origin Consistency Model


### Images

![2c614e2e29a8a644eca0a2e4081635891c6104bd1f80d9caf5e203b3625bd0f6.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/2c614e2e29a8a644eca0a2e4081635891c6104bd1f80d9caf5e203b3625bd0f6.jpg)

![431da18b8aeec5f72db7ba91d0264a43b91b88fe476151869157ad8b77a8e709.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/431da18b8aeec5f72db7ba91d0264a43b91b88fe476151869157ad8b77a8e709.jpg)

![43c21073e0e36d1297c49634826ca78e4c3bdfca0c27f3802ce0abac289f91bf.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/43c21073e0e36d1297c49634826ca78e4c3bdfca0c27f3802ce0abac289f91bf.jpg)

![9958c31a5b724837e89f4665ca4799956afadd62b009381a5dadcee849d02f67.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/9958c31a5b724837e89f4665ca4799956afadd62b009381a5dadcee849d02f67.jpg)

![ce0e2d3b1eb9cedf5f638194903411a980a025dec06f514db0d2f18336faa15b.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/ce0e2d3b1eb9cedf5f638194903411a980a025dec06f514db0d2f18336faa15b.jpg)

![f5670d48ae76a08df4c67c7535ef860474178ed1d273a0335da6d2fe080984de.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/f5670d48ae76a08df4c67c7535ef860474178ed1d273a0335da6d2fe080984de.jpg)

![f8180aeda443d791db091fc3f789e2e34f2f9545415ca5f00ae9e1215147509f.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/images/f8180aeda443d791db091fc3f789e2e34f2f9545415ca5f00ae9e1215147509f.jpg)

### Tables

![2ae1c1aa79dfcbeb13eff89248324eab9c591be38c9e96ec6ed70d20edae3d40.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/2ae1c1aa79dfcbeb13eff89248324eab9c591be38c9e96ec6ed70d20edae3d40.jpg)

![391b1e791d138e9dc19101e58f82bcf1b0ff573780126afd122ecc8b21426148.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/391b1e791d138e9dc19101e58f82bcf1b0ff573780126afd122ecc8b21426148.jpg)

![3d10600531169226eb12dd18f8e29f509411ae1443c6e10b3b4abbf6dd03afed.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/3d10600531169226eb12dd18f8e29f509411ae1443c6e10b3b4abbf6dd03afed.jpg)

![9673c66dc4c874baa5476a67472c7aa835ba8dedb7ff30ed44cce466aa5c6c66.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/9673c66dc4c874baa5476a67472c7aa835ba8dedb7ff30ed44cce466aa5c6c66.jpg)

![a2bf444dd7ee965af9a973b4c9d17663ee31b3b3dbcc525c60ad3f1ab39c7407.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/a2bf444dd7ee965af9a973b4c9d17663ee31b3b3dbcc525c60ad3f1ab39c7407.jpg)

![a2c7530441a19a43ba27d375ff5159bb9b932d07fcef9a8e3d2f64193f854d61.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/a2c7530441a19a43ba27d375ff5159bb9b932d07fcef9a8e3d2f64193f854d61.jpg)

![b155718f1834abdfbacd75c5d815131893ac3bdc5d53a2e9b9f8b7e7757fef82.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/b155718f1834abdfbacd75c5d815131893ac3bdc5d53a2e9b9f8b7e7757fef82.jpg)

![b236aa77de1b15095a65a554caa2e0f3a6b0df1a5eaa0d568d40e9a7a547ab1a.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/b236aa77de1b15095a65a554caa2e0f3a6b0df1a5eaa0d568d40e9a7a547ab1a.jpg)

![b5d759f625708ad1ce2dcf2aa62570ee3e913b8ea304e7c11c2c303d39432f36.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/b5d759f625708ad1ce2dcf2aa62570ee3e913b8ea304e7c11c2c303d39432f36.jpg)

![c3fbe30b6fe783d9bb66f00b0c5aa02c80c3f04732592fe2809681c4bbd4643f.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/c3fbe30b6fe783d9bb66f00b0c5aa02c80c3f04732592fe2809681c4bbd4643f.jpg)

![d3f12b3ebd5f5d873888a88c70c5761032377b63ea6b73a4881963a88abe1615.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/d3f12b3ebd5f5d873888a88c70c5761032377b63ea6b73a4881963a88abe1615.jpg)

![e93fddc4a0bdc010211fed0f64df014cc79a3f662367734504d814bd974d96f3.jpg](../icml_results/2889_GOCM_%20Single-Step%20Graph%20Outlier%20Synthesis%20via%20Origin%20Consistency%20Model/tables/e93fddc4a0bdc010211fed0f64df014cc79a3f662367734504d814bd974d96f3.jpg)

<a id="paper-2852"></a>

## Differentially Private Continual Release with Relative Error


### Tables

![501657ce07026f4a65d30a9a5a4be6af76bb5bc90ac9ece6044319e188065800.jpg](../icml_results/2890_Differentially%20Private%20Continual%20Release%20with%20Relative%20Error/tables/501657ce07026f4a65d30a9a5a4be6af76bb5bc90ac9ece6044319e188065800.jpg)

<a id="paper-2853"></a>

## OBJVanish: Prompt-Driven Generation of Physically Realizable 3D LiDAR-Invisible Objects


### Images

![017ed97c7d158dd09455f1a4ae92301f63b68424cbd1475a4260545abffc603b.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/017ed97c7d158dd09455f1a4ae92301f63b68424cbd1475a4260545abffc603b.jpg)

![122566a731743d64f0cf6d2fba0919b45761e43e392ae81133100e32d969bb77.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/122566a731743d64f0cf6d2fba0919b45761e43e392ae81133100e32d969bb77.jpg)

![2e485c1fc1c4c476de97f339af00eecc6757c59d31408b3b9a99fdac041fcaab.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/2e485c1fc1c4c476de97f339af00eecc6757c59d31408b3b9a99fdac041fcaab.jpg)

![3b9178db5c2a100ca5f6bc23157fa197a49c3f99323c667e73db9139e433ad57.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/3b9178db5c2a100ca5f6bc23157fa197a49c3f99323c667e73db9139e433ad57.jpg)

![3d0780092fc3df8d0d902e4e68aea0b55f88c22ca5690598b0d6c2c0bf949597.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/3d0780092fc3df8d0d902e4e68aea0b55f88c22ca5690598b0d6c2c0bf949597.jpg)

![4ebaff8c8ec9af978e12669cd8ad711233a6477b3744781ed4272bf4a029f2a5.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/4ebaff8c8ec9af978e12669cd8ad711233a6477b3744781ed4272bf4a029f2a5.jpg)

![533670d3f285ee5d1cab61426bdd3d524ff1917f5bfec50dc81954ed33e6387b.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/533670d3f285ee5d1cab61426bdd3d524ff1917f5bfec50dc81954ed33e6387b.jpg)

![70c2c5fe5d34d3f811c75a7e9a24b9bd45bf8539b6006b23b7a813b9dd55ad31.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/70c2c5fe5d34d3f811c75a7e9a24b9bd45bf8539b6006b23b7a813b9dd55ad31.jpg)

![79975b5c912809c056f84c2552519b929a23b3f48854e33fef7dd25a4857b839.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/79975b5c912809c056f84c2552519b929a23b3f48854e33fef7dd25a4857b839.jpg)

![ae4528e6f0831dbfcf8a883153e715f0039f59d890866909a12d78812fe0ad5e.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/ae4528e6f0831dbfcf8a883153e715f0039f59d890866909a12d78812fe0ad5e.jpg)

![faf0f12f80f5489fffc4d1f8cf16aec20d307d34ba581858619a7e57b089894f.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/images/faf0f12f80f5489fffc4d1f8cf16aec20d307d34ba581858619a7e57b089894f.jpg)

### Tables

![7deb1aa661dff6b1e187f24615677b0d19e54c50249f73a82d0edc9663497a5e.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/7deb1aa661dff6b1e187f24615677b0d19e54c50249f73a82d0edc9663497a5e.jpg)

![8469d2402567957a39f861d354571be3bde3a647a862e91d5ab32d4b53cad4d7.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/8469d2402567957a39f861d354571be3bde3a647a862e91d5ab32d4b53cad4d7.jpg)

![abcea28b6c1edb5e31f2412e4ec1527734d364efdcef4454c634e634bcbf010c.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/abcea28b6c1edb5e31f2412e4ec1527734d364efdcef4454c634e634bcbf010c.jpg)

![c354c51d1f30385819057af098409d1cc198b3f52068b8d5b3691c7c298e8318.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/c354c51d1f30385819057af098409d1cc198b3f52068b8d5b3691c7c298e8318.jpg)

![dce0fe10fdc5723fd9d0018e209c7dcecc870c5321fe4ca3d8880c1345fad3d4.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/dce0fe10fdc5723fd9d0018e209c7dcecc870c5321fe4ca3d8880c1345fad3d4.jpg)

![e4c489c9ac277843c7dba986f390be26ee30210cddcd091bbb0abd08f4189fc3.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/e4c489c9ac277843c7dba986f390be26ee30210cddcd091bbb0abd08f4189fc3.jpg)

![e81e38b5f04f31d44095fa9ebfc206fa7c73844abde9f8977cb2459cf19549f1.jpg](../icml_results/2891_OBJVanish_%20Prompt-Driven%20Generation%20of%20Physically%20Realizable%203D%20LiDAR-Invisible%20Objects/tables/e81e38b5f04f31d44095fa9ebfc206fa7c73844abde9f8977cb2459cf19549f1.jpg)

<a id="paper-2854"></a>

## SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding


### Images

![5c2f9c95d79b9a0a460b91d8a8efd6929a57faf60cec5e55ee9815f77537ef72.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/5c2f9c95d79b9a0a460b91d8a8efd6929a57faf60cec5e55ee9815f77537ef72.jpg)

![7db514d47df61068586c48b84bd30c45bba433815a7348ef455f1b68cc164383.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/7db514d47df61068586c48b84bd30c45bba433815a7348ef455f1b68cc164383.jpg)

![8729fd7835170c20be5a574254585f57c986677c46aae7e6a992ca7305e2067d.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/8729fd7835170c20be5a574254585f57c986677c46aae7e6a992ca7305e2067d.jpg)

![8eb7a16ade120d90ccdf82528a3498ec3561f6ef3e5a0204115ba405cbdedb4c.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/8eb7a16ade120d90ccdf82528a3498ec3561f6ef3e5a0204115ba405cbdedb4c.jpg)

![b7fb24939f2c104c0f02f4ecfe267f43ce8a5396e931104ec7ac9ce51e542887.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/b7fb24939f2c104c0f02f4ecfe267f43ce8a5396e931104ec7ac9ce51e542887.jpg)

![d6c92c6c1efb96460b4b1a2dc9f948427dffd99d3dc9df9b2fc472bf26fe4234.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/d6c92c6c1efb96460b4b1a2dc9f948427dffd99d3dc9df9b2fc472bf26fe4234.jpg)

![ea2a9bde49a888d9b69bc9ef1bedcac8cc44aa08752166a51343418462946a81.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/ea2a9bde49a888d9b69bc9ef1bedcac8cc44aa08752166a51343418462946a81.jpg)

![f8eb9fbc32c4f948bce3181db1626f05125cf75e093db20f452aba4b6b38de19.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/images/f8eb9fbc32c4f948bce3181db1626f05125cf75e093db20f452aba4b6b38de19.jpg)

### Tables

![4b2d57e3a657c777fb2dbedf733836714a67a25ec56038d9646ada25e7dd2922.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/4b2d57e3a657c777fb2dbedf733836714a67a25ec56038d9646ada25e7dd2922.jpg)

![5e3f00bb1361977691fcd40a5612ddaf0daeda8474d4e9056a852e3e617c763d.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/5e3f00bb1361977691fcd40a5612ddaf0daeda8474d4e9056a852e3e617c763d.jpg)

![710d3985b1040a50b7f31eb726af6463b55a76ded75aa1bb172475ba58ade316.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/710d3985b1040a50b7f31eb726af6463b55a76ded75aa1bb172475ba58ade316.jpg)

![7c967a57e5d0003650b7254358130ed7f646c1f10af4be7476ff61dbaeb789cd.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/7c967a57e5d0003650b7254358130ed7f646c1f10af4be7476ff61dbaeb789cd.jpg)

![7c98ac544adc9db42ddc978fb274e7127e58c6447aac7ab5c2fc845b28e32a52.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/7c98ac544adc9db42ddc978fb274e7127e58c6447aac7ab5c2fc845b28e32a52.jpg)

![872d3aaab2091093ee31f6a0acc51a573722b44d78b1d893cfd2a5838386eee9.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/872d3aaab2091093ee31f6a0acc51a573722b44d78b1d893cfd2a5838386eee9.jpg)

![905388f448152f1867c7c980d39accbc2150698f612f483d2864545a17b74848.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/905388f448152f1867c7c980d39accbc2150698f612f483d2864545a17b74848.jpg)

![a85539aa0390ed3ae7c093467f1b589c3c3620b602f5e02cdc186f7f5f3a99ef.jpg](../icml_results/2892_SpecForge_%20A%20Flexible%20and%20Efficient%20Open-Source%20Training%20Framework%20for%20Speculative%20Decoding/tables/a85539aa0390ed3ae7c093467f1b589c3c3620b602f5e02cdc186f7f5f3a99ef.jpg)

<a id="paper-2855"></a>

## Statistical Consistency and Generalization of Contrastive Representation Learning


### Images

![f2f4d0e26b4268b07eef3395a926ea0ea98702e888c3a1d7f687c71911c3d013.jpg](../icml_results/2893_Statistical%20Consistency%20and%20Generalization%20of%20Contrastive%20Representation%20Learning/images/f2f4d0e26b4268b07eef3395a926ea0ea98702e888c3a1d7f687c71911c3d013.jpg)

### Tables

![3b5d16624345db2b34c034e98d9183ed8d17fe31e00ec8f54d7ed31f46cb04b1.jpg](../icml_results/2893_Statistical%20Consistency%20and%20Generalization%20of%20Contrastive%20Representation%20Learning/tables/3b5d16624345db2b34c034e98d9183ed8d17fe31e00ec8f54d7ed31f46cb04b1.jpg)

![923864d2a25dab2741e66145cf9810fd06c673152c9448f4371760e0a8642009.jpg](../icml_results/2893_Statistical%20Consistency%20and%20Generalization%20of%20Contrastive%20Representation%20Learning/tables/923864d2a25dab2741e66145cf9810fd06c673152c9448f4371760e0a8642009.jpg)

<a id="paper-2856"></a>

## Disentangling Intent from Role: Adversarial Self-Play for Persona-Invariant Safety Alignment


### Images

![2063a9b396d2fb082fcabffa845fe08d762a45feaa6ddb6695e130eecad015bb.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/images/2063a9b396d2fb082fcabffa845fe08d762a45feaa6ddb6695e130eecad015bb.jpg)

![2310b3b2e4fcea32ff31155c8bf28b42cea87de5a8d9ab34c65dcede5f11e564.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/images/2310b3b2e4fcea32ff31155c8bf28b42cea87de5a8d9ab34c65dcede5f11e564.jpg)

![5b17e8eabd5275c16a5aaf561997c67fb623ebc8858cfbf4dc037c11573e7fc4.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/images/5b17e8eabd5275c16a5aaf561997c67fb623ebc8858cfbf4dc037c11573e7fc4.jpg)

![a5b4cf5054ba7332c21ef566ca4774eca49f973c5ea899f62213292c7fb77512.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/images/a5b4cf5054ba7332c21ef566ca4774eca49f973c5ea899f62213292c7fb77512.jpg)

![d8dac513d03abf67b43676bf5da324da4cff7acbcc03e60ca3cc2acd50ac4928.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/images/d8dac513d03abf67b43676bf5da324da4cff7acbcc03e60ca3cc2acd50ac4928.jpg)

### Tables

![0382d810f9fe3e6492d41ae0e8901e6abd6af3d6bcf7d1495cc5cee1c5e8285a.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/0382d810f9fe3e6492d41ae0e8901e6abd6af3d6bcf7d1495cc5cee1c5e8285a.jpg)

![086606380524f78036dc1b2a17e2e9026d4deaf1d62d85fe27fe3d35160db1e8.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/086606380524f78036dc1b2a17e2e9026d4deaf1d62d85fe27fe3d35160db1e8.jpg)

![2518a487e1bf72f5ca1334a41b66edc2c502707587f6d2d2c0bf9c257e65afeb.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/2518a487e1bf72f5ca1334a41b66edc2c502707587f6d2d2c0bf9c257e65afeb.jpg)

![485fc0db65210cc84f92a27bfbe9cf94ddb52f2ba297f4db93ee3ebfee582d1b.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/485fc0db65210cc84f92a27bfbe9cf94ddb52f2ba297f4db93ee3ebfee582d1b.jpg)

![55d957071d87936a94d132138714b03d6374be9703b66428aad77137eab0b500.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/55d957071d87936a94d132138714b03d6374be9703b66428aad77137eab0b500.jpg)

![57066d2b269e00eb476df0ec53664a2a20088fc0cbfc69aa165c3129503e70c3.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/57066d2b269e00eb476df0ec53664a2a20088fc0cbfc69aa165c3129503e70c3.jpg)

![6096d29ca0e2ab26126f9a596c39b3f122f393ff9e78393155fb040675ed9a71.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/6096d29ca0e2ab26126f9a596c39b3f122f393ff9e78393155fb040675ed9a71.jpg)

![648a93fe81b354cdaa5f3363068d2764b060c17a45a8f5061230bc5f76f34931.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/648a93fe81b354cdaa5f3363068d2764b060c17a45a8f5061230bc5f76f34931.jpg)

![67fffb0c00fa392ad3ed4b430b04890040a6d041f98a35b6f2cee7659f90e4b0.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/67fffb0c00fa392ad3ed4b430b04890040a6d041f98a35b6f2cee7659f90e4b0.jpg)

![75ce7b6b21f02c23ab85f55b97f843620f1ba470e8c6b04088d9a27e940bb9a5.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/75ce7b6b21f02c23ab85f55b97f843620f1ba470e8c6b04088d9a27e940bb9a5.jpg)

![8bd1c8bb18fb0652dda3127b1a36dadd841e3f678f6631fd31d42160f8e23f11.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/8bd1c8bb18fb0652dda3127b1a36dadd841e3f678f6631fd31d42160f8e23f11.jpg)

![9222e370b4eea9a8126ff75ff4880e23cc843207e43fb21acf215c5870cae746.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/9222e370b4eea9a8126ff75ff4880e23cc843207e43fb21acf215c5870cae746.jpg)

![995bc376f965eb67713d56850a321bdc3f5e6a8833fa2c6b6fd014072dd8d135.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/995bc376f965eb67713d56850a321bdc3f5e6a8833fa2c6b6fd014072dd8d135.jpg)

![a0b67590574e2b9582a1837875fbffbff2f4ee7d4408934266f2aed3d19fddcd.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/a0b67590574e2b9582a1837875fbffbff2f4ee7d4408934266f2aed3d19fddcd.jpg)

![dff424a3493ac18982631684ee466b53b8f543c2767b877ba06172113b7474ea.jpg](../icml_results/2894_Disentangling%20Intent%20from%20Role_%20Adversarial%20Self-Play%20for%20Persona-Invariant%20Safety%20Alignment/tables/dff424a3493ac18982631684ee466b53b8f543c2767b877ba06172113b7474ea.jpg)

<a id="paper-2857"></a>

## Variational Routing: A Scalable Bayesian Framework for Calibrated Mixture-of-Experts Transformers


### Images

![205416ffe45bb1ba69d284892b3092a6cbcd7b087df407daaefa7b1b498ebe66.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/205416ffe45bb1ba69d284892b3092a6cbcd7b087df407daaefa7b1b498ebe66.jpg)

![22c8775ea8dce0c6d03e63ecc32c2edbd627fd40f85102a8beb2bc4075ab1671.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/22c8775ea8dce0c6d03e63ecc32c2edbd627fd40f85102a8beb2bc4075ab1671.jpg)

![339f1db68ad37aa0f2bb86a72165ea59ef59766eefe34ebea4eef8e3913d8839.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/339f1db68ad37aa0f2bb86a72165ea59ef59766eefe34ebea4eef8e3913d8839.jpg)

![3f659803aa65e5df9bfb3e70f62129038e03c75279c1929a072e6fe13243c0f4.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/3f659803aa65e5df9bfb3e70f62129038e03c75279c1929a072e6fe13243c0f4.jpg)

![627d33e25ecd28e871abe9918b8e51fcc44f53bdfe596e5f9fac9bbaea0afa03.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/627d33e25ecd28e871abe9918b8e51fcc44f53bdfe596e5f9fac9bbaea0afa03.jpg)

![6607f054828c53ba2d83adcbaf7cfc1576243bbc5f9bc393f30818fe2b7867bd.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/6607f054828c53ba2d83adcbaf7cfc1576243bbc5f9bc393f30818fe2b7867bd.jpg)

![6da725c3754b90ebb6b49637d71c34b3435a5371654d5bfbecc0596ee0590280.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/6da725c3754b90ebb6b49637d71c34b3435a5371654d5bfbecc0596ee0590280.jpg)

![7d57386a02df5a9c04ea7dcc1a9d7c5b42ebe1d8a996da0cb82a30d6d61e7d68.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/7d57386a02df5a9c04ea7dcc1a9d7c5b42ebe1d8a996da0cb82a30d6d61e7d68.jpg)

![83b00be682769e9dfc1746ccb42bb8d397598bed3d0537b5126d052eb42d6d5f.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/83b00be682769e9dfc1746ccb42bb8d397598bed3d0537b5126d052eb42d6d5f.jpg)

![9d769f2ff4380aa71a4c52b40b0645ff173b5ee6653bf20c19f71c1041727aa8.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/9d769f2ff4380aa71a4c52b40b0645ff173b5ee6653bf20c19f71c1041727aa8.jpg)

![a4937e5bbd7f2fc8537df5489fcea834c25fe099987fc0e982e5feb56782fe81.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/a4937e5bbd7f2fc8537df5489fcea834c25fe099987fc0e982e5feb56782fe81.jpg)

![c5d42104f1969fa7965b81eb01f2b676a26005690dbd5169a93c0cd627473dcd.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/c5d42104f1969fa7965b81eb01f2b676a26005690dbd5169a93c0cd627473dcd.jpg)

![d9cdf5b9df70f67a8f29fcf355144ec065172e0d6d7753aea16b175cb46fd02d.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/d9cdf5b9df70f67a8f29fcf355144ec065172e0d6d7753aea16b175cb46fd02d.jpg)

![dea1f0a3820a53ca0b34bd6293d65842f4fd8b0e117be8138aa2f8a7abae664c.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/images/dea1f0a3820a53ca0b34bd6293d65842f4fd8b0e117be8138aa2f8a7abae664c.jpg)

### Tables

![091af1edb7ba2ef88c53ca395219e24b41786382e274d9b4c8ef39252e3ce93e.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/091af1edb7ba2ef88c53ca395219e24b41786382e274d9b4c8ef39252e3ce93e.jpg)

![1c0478a88259a5f548f0be301f2fd4951afab1db8ebb729d0917aec831551bfb.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/1c0478a88259a5f548f0be301f2fd4951afab1db8ebb729d0917aec831551bfb.jpg)

![2a28fa52af1b18c31f15752d482bce4f69bb3fc08b1d3abe4fbe465022d9f244.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/2a28fa52af1b18c31f15752d482bce4f69bb3fc08b1d3abe4fbe465022d9f244.jpg)

![515171a8bff7cda34b464c13807954376d6c0a680e12fd3a0fcac595032843b1.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/515171a8bff7cda34b464c13807954376d6c0a680e12fd3a0fcac595032843b1.jpg)

![67c9fb66074f8a44795b8022b528ac5608c9be5c73c956a56d8ea37a7e55a6ea.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/67c9fb66074f8a44795b8022b528ac5608c9be5c73c956a56d8ea37a7e55a6ea.jpg)

![7ad6c28e00871141cdff31e42a51ccb2c1b4143a35776135f74ade624d8bc060.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/7ad6c28e00871141cdff31e42a51ccb2c1b4143a35776135f74ade624d8bc060.jpg)

![7c7736deafa426f53af9ad553b693d196cb517a88ee56337a3fda420ed715fd0.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/7c7736deafa426f53af9ad553b693d196cb517a88ee56337a3fda420ed715fd0.jpg)

![84ae9723738eb417322f54613b3b8a893e0d5c6a1d355b425f46370b4ad4f4d3.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/84ae9723738eb417322f54613b3b8a893e0d5c6a1d355b425f46370b4ad4f4d3.jpg)

![ca9e50350fb5019fb9507189c2f9ae2600e9051dbdb18d27309cd696bfa88eeb.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/ca9e50350fb5019fb9507189c2f9ae2600e9051dbdb18d27309cd696bfa88eeb.jpg)

![f7d18c9b4eb6d2839e5bbf7bfd334b75f6e4c9420239f37d21c51c05745e6b26.jpg](../icml_results/2895_Variational%20Routing_%20A%20Scalable%20Bayesian%20Framework%20for%20Calibrated%20Mixture-of-Experts%20Transformers/tables/f7d18c9b4eb6d2839e5bbf7bfd334b75f6e4c9420239f37d21c51c05745e6b26.jpg)

<a id="paper-2858"></a>

## Structured 4D Latent Predictive Model for Robot Planning


### Images

![20cef475596afd153c449c10b4e230b1b1e4259ac79611f1cdada41f0cc5a365.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/images/20cef475596afd153c449c10b4e230b1b1e4259ac79611f1cdada41f0cc5a365.jpg)

![6706c4f72335f0c0430b10eec2695027e8687307eeb929dcd8df7682ac41e250.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/images/6706c4f72335f0c0430b10eec2695027e8687307eeb929dcd8df7682ac41e250.jpg)

![8b74db08ef57d5276798e60e56b75cc5e184b65cc534b42aac2d6ebd9681804a.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/images/8b74db08ef57d5276798e60e56b75cc5e184b65cc534b42aac2d6ebd9681804a.jpg)

![90a6d7f141cc79f729b9588ee5c9814313b5aa9b79f0a5c2c6486cd89099247f.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/images/90a6d7f141cc79f729b9588ee5c9814313b5aa9b79f0a5c2c6486cd89099247f.jpg)

![a0e36f0303c8e2612969a6d7ce0982b59cd5178ce4fac6e0d9fb1791684a0b2c.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/images/a0e36f0303c8e2612969a6d7ce0982b59cd5178ce4fac6e0d9fb1791684a0b2c.jpg)

![bc1f61f5b4c3d57b3e8e6cf70e42ec55f11d19613d0258cfccb09f73e7406d2c.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/images/bc1f61f5b4c3d57b3e8e6cf70e42ec55f11d19613d0258cfccb09f73e7406d2c.jpg)

### Tables

![32d929d999d438dc268bb62a9687828bd8661b6b6e2981981a8ba19b726bac1d.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/32d929d999d438dc268bb62a9687828bd8661b6b6e2981981a8ba19b726bac1d.jpg)

![510ab6b809a29a6deea251c02bac1a8d8198b5758a04b72de77d06421374179a.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/510ab6b809a29a6deea251c02bac1a8d8198b5758a04b72de77d06421374179a.jpg)

![89e13e251908620259be3db5eddc517ac4a67e0302a0af6db20b3271dbe0620e.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/89e13e251908620259be3db5eddc517ac4a67e0302a0af6db20b3271dbe0620e.jpg)

![8db0a5fee2d7c1e302d739f5dab26044c1385d4e1375229ba11780dd3256260b.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/8db0a5fee2d7c1e302d739f5dab26044c1385d4e1375229ba11780dd3256260b.jpg)

![b53560f3e2330f27f8e350fe41cb312f5db0539d844e0101b9d3ce15712a40b7.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/b53560f3e2330f27f8e350fe41cb312f5db0539d844e0101b9d3ce15712a40b7.jpg)

![d27a74c269b27fccaa4348858ef72e73ed00908b086f25eb135d54f6945857ad.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/d27a74c269b27fccaa4348858ef72e73ed00908b086f25eb135d54f6945857ad.jpg)

![e4ac15054da8a304ca1dc58b43bd06ef87e3d2208df2e4ba13012de87298c950.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/e4ac15054da8a304ca1dc58b43bd06ef87e3d2208df2e4ba13012de87298c950.jpg)

![f6d7a337269898e4ce03122444382fc3f9f65c79c1b4be2d4c39d1532b0d1d5e.jpg](../icml_results/2896_Structured%204D%20Latent%20Predictive%20Model%20for%20Robot%20Planning/tables/f6d7a337269898e4ce03122444382fc3f9f65c79c1b4be2d4c39d1532b0d1d5e.jpg)

<a id="paper-2859"></a>

## Graph is a Substrate Across Data Modalities


### Images

![1228ce1be6178b2400bd7a6358075fce0fb71708abc155c23c754ef175b05918.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/images/1228ce1be6178b2400bd7a6358075fce0fb71708abc155c23c754ef175b05918.jpg)

![471de1df0414197db13b581bae06c69ae32b2f516ebef419a7301e0409053c2e.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/images/471de1df0414197db13b581bae06c69ae32b2f516ebef419a7301e0409053c2e.jpg)

![bd33e84a8e60205469e44b897cab07426a17d98ee0181cc3b341cf715a832051.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/images/bd33e84a8e60205469e44b897cab07426a17d98ee0181cc3b341cf715a832051.jpg)

![c311acc3c2efefca8b5aa0895dbf18bd6dc9c94a83d061b0a4925fe2560699aa.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/images/c311acc3c2efefca8b5aa0895dbf18bd6dc9c94a83d061b0a4925fe2560699aa.jpg)

![ee011df9740ef49098277d3bfa01653d00a6d8d5279966fd3edce69641908d38.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/images/ee011df9740ef49098277d3bfa01653d00a6d8d5279966fd3edce69641908d38.jpg)

### Tables

![05b8892d394f9a37bdde0f7848d67fdde36b385998ba6ff7b0683cf68104dfdc.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/05b8892d394f9a37bdde0f7848d67fdde36b385998ba6ff7b0683cf68104dfdc.jpg)

![1e09d36b0e5061c09d967fbbff7d96e4503b97e8813fbd110841042c9f534c99.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/1e09d36b0e5061c09d967fbbff7d96e4503b97e8813fbd110841042c9f534c99.jpg)

![2ceffad5ed39ad8f1e3206470dc8f500aa29061266e262631e034904ec5a50aa.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/2ceffad5ed39ad8f1e3206470dc8f500aa29061266e262631e034904ec5a50aa.jpg)

![3b26a4593ef293bb75baa4b9ecbb3c44a58a9984bf5376a25592600a5ac187e2.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/3b26a4593ef293bb75baa4b9ecbb3c44a58a9984bf5376a25592600a5ac187e2.jpg)

![4576dfa78abb0639b148a1894af35e986f192efdae844dd68bdc9fb87c95ce43.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/4576dfa78abb0639b148a1894af35e986f192efdae844dd68bdc9fb87c95ce43.jpg)

![677e48048f58f10c2cce3b33972201fcf1cd86046b62595e1b64a7de584f699f.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/677e48048f58f10c2cce3b33972201fcf1cd86046b62595e1b64a7de584f699f.jpg)

![67abafba4246d7e61db0cd8bd7a2d1ae0981e2ff28bcabe080c74a8f51bd4b24.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/67abafba4246d7e61db0cd8bd7a2d1ae0981e2ff28bcabe080c74a8f51bd4b24.jpg)

![70f1711cef971f9c2dc815b296879fae2023208fdf8d3cab709ed992cffe2d3c.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/70f1711cef971f9c2dc815b296879fae2023208fdf8d3cab709ed992cffe2d3c.jpg)

![9b0293d3bb43d07e3d0665ed14e74f46f83d055445f72abea06b4587868dd418.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/9b0293d3bb43d07e3d0665ed14e74f46f83d055445f72abea06b4587868dd418.jpg)

![a4feb89659d7415f27292c1c1c956c827f99d73a39ac4f0affa72ad4647f37c8.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/a4feb89659d7415f27292c1c1c956c827f99d73a39ac4f0affa72ad4647f37c8.jpg)

![b48da5b28694f0cfb8ba4942349af075422624f1797ee9ef81945e3234edbf64.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/b48da5b28694f0cfb8ba4942349af075422624f1797ee9ef81945e3234edbf64.jpg)

![c232b1809ceeae4f3dd4ccfa9c4238ea9d2fdbc89b2fb851e5176413d28d9ae9.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/c232b1809ceeae4f3dd4ccfa9c4238ea9d2fdbc89b2fb851e5176413d28d9ae9.jpg)

![c5ad55ae409d095e89a2cb719d315884b227c36908886dbd9747b9ccfa76624c.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/c5ad55ae409d095e89a2cb719d315884b227c36908886dbd9747b9ccfa76624c.jpg)

![cc873921a7d16e56deb14f7357830f15113e4b6defb614d27d65be2420e00242.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/cc873921a7d16e56deb14f7357830f15113e4b6defb614d27d65be2420e00242.jpg)

![cda66e3d5bc319484c6e02b4459817ff6247fc1273a0bc547d6f42881afcca48.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/cda66e3d5bc319484c6e02b4459817ff6247fc1273a0bc547d6f42881afcca48.jpg)

![d01b45930e20ef0ce97cd9ad5095cd1fdb84fc75882c8d03ab720f5547abbdee.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/d01b45930e20ef0ce97cd9ad5095cd1fdb84fc75882c8d03ab720f5547abbdee.jpg)

![ebb02ca233a14cbd588fe30a4511d69858b542a319151d68911d90d819416557.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/ebb02ca233a14cbd588fe30a4511d69858b542a319151d68911d90d819416557.jpg)

![eddb157aa1085a3c750354332ab236c52cbfa143f32f49ddd63b16a8b86f82f6.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/eddb157aa1085a3c750354332ab236c52cbfa143f32f49ddd63b16a8b86f82f6.jpg)

![f1b8eb08962063b115021bccc36f117c8c146fa2cbce3593700340f96cc37f65.jpg](../icml_results/2897_Graph%20is%20a%20Substrate%20Across%20Data%20Modalities/tables/f1b8eb08962063b115021bccc36f117c8c146fa2cbce3593700340f96cc37f65.jpg)

<a id="paper-2860"></a>

## nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding


### Images

![23c1b6c813bea8c3dd9c5f8fc02b4cab39f21197a73e8acd310f7e26ceaef118.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/23c1b6c813bea8c3dd9c5f8fc02b4cab39f21197a73e8acd310f7e26ceaef118.jpg)

![26e129de9b385df601962e0369fe6f40ae819ec58cb4ef112354d8738e538980.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/26e129de9b385df601962e0369fe6f40ae819ec58cb4ef112354d8738e538980.jpg)

![2be2482215d1de1d7dba91e7d58013eff52bbe56dfdd34f3e6e2992c60c6f8ce.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/2be2482215d1de1d7dba91e7d58013eff52bbe56dfdd34f3e6e2992c60c6f8ce.jpg)

![7184add87d63404afa298aaa8b2398ee4ad34ada4707cc7c9a3e13dd6351a16e.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/7184add87d63404afa298aaa8b2398ee4ad34ada4707cc7c9a3e13dd6351a16e.jpg)

![875d0d06d905e975f8564fb20584f9a2b0d44acc2fe2efbef0610584d6657df3.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/875d0d06d905e975f8564fb20584f9a2b0d44acc2fe2efbef0610584d6657df3.jpg)

![9d4a12575d3e9d530854ce6225df65991b33eee869c4f60c6573f10bafa3030d.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/9d4a12575d3e9d530854ce6225df65991b33eee869c4f60c6573f10bafa3030d.jpg)

![ad4d6b9108983d8e06ac85be6c750d60dea758e847efaf39bbccbe23f71add77.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/ad4d6b9108983d8e06ac85be6c750d60dea758e847efaf39bbccbe23f71add77.jpg)

![be2a5ea94f113fa8f8de0ae7b653a99cbc04233c051f0f39fcde2af2787b3cde.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/be2a5ea94f113fa8f8de0ae7b653a99cbc04233c051f0f39fcde2af2787b3cde.jpg)

![d00f8a47d9c46fe0a06ca299f3e48a46be60758329a1db1b115c69478ac09745.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/images/d00f8a47d9c46fe0a06ca299f3e48a46be60758329a1db1b115c69478ac09745.jpg)

### Tables

![2190917a8a35a17edbd567e0f8a62faf0ea2b78a289f5a9251e8bdd12feb0d4c.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/2190917a8a35a17edbd567e0f8a62faf0ea2b78a289f5a9251e8bdd12feb0d4c.jpg)

![4c3b49835e41521d097a5ea36d4bcf54d916ccee7cc729cda59fac4e29cd6bf5.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/4c3b49835e41521d097a5ea36d4bcf54d916ccee7cc729cda59fac4e29cd6bf5.jpg)

![7433858e3a0fb47a718d977c426760bd097504c9f957af582f7cdcb42ccfc06a.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/7433858e3a0fb47a718d977c426760bd097504c9f957af582f7cdcb42ccfc06a.jpg)

![86e129b7c8c0806ceb82a4bac641522e7543e94bf56cb9e74e515364d89c04dd.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/86e129b7c8c0806ceb82a4bac641522e7543e94bf56cb9e74e515364d89c04dd.jpg)

![9f50bfc766be674105fce2aba8f1f5ff9483de72c7e69d625581cdf78a5dfc68.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/9f50bfc766be674105fce2aba8f1f5ff9483de72c7e69d625581cdf78a5dfc68.jpg)

![a47ec00e95fe104c0448b4318a127d203653016824c8766f28a14c2c1501edd1.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/a47ec00e95fe104c0448b4318a127d203653016824c8766f28a14c2c1501edd1.jpg)

![b3f6b8fd2de647e0424ebe4b769047148876a68f3d8d075af7252554c6571c50.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/b3f6b8fd2de647e0424ebe4b769047148876a68f3d8d075af7252554c6571c50.jpg)

![d63cd4bd81eb7652d8894ca8c2e673940adb7720fae8f482704dbfc6a23e8361.jpg](../icml_results/2898_nD-RoPE_%20A%20Generalized%20RoPE%20for%20n-Dimensional%20Position%20Embedding/tables/d63cd4bd81eb7652d8894ca8c2e673940adb7720fae8f482704dbfc6a23e8361.jpg)

<a id="paper-2861"></a>

## Neural Modular Physics for Elastic Simulation


### Images

![2a5634a72694a37e2bdde5c687b395ac4c3727fe97ecf0cc738846b51f21e80a.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/2a5634a72694a37e2bdde5c687b395ac4c3727fe97ecf0cc738846b51f21e80a.jpg)

![696ebd8cf1b829c6a024898b11f3d61fb48136e5d5b5965dd517558edbcaa65d.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/696ebd8cf1b829c6a024898b11f3d61fb48136e5d5b5965dd517558edbcaa65d.jpg)

![6aee963fefe4f2aef40f4c7e3d0ec335715cda727e154301dcdd6887b117663a.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/6aee963fefe4f2aef40f4c7e3d0ec335715cda727e154301dcdd6887b117663a.jpg)

![76b0cc22a9aabe64615847e68b1031b4ae8319e10baf634b4f51d4ba3fca066f.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/76b0cc22a9aabe64615847e68b1031b4ae8319e10baf634b4f51d4ba3fca066f.jpg)

![9b1cca7791b3ea9bf94b15db9aebbd048a1b192592c527dec06666f72d547ed3.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/9b1cca7791b3ea9bf94b15db9aebbd048a1b192592c527dec06666f72d547ed3.jpg)

![9f0726f280a14ba86dc1c1822708c6aacbfdb0cd6422addd04c9a55558510ef2.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/9f0726f280a14ba86dc1c1822708c6aacbfdb0cd6422addd04c9a55558510ef2.jpg)

![ab3b0a9f57290d904c2828edd8e9ccbf34bc9870f770c690b5a00cc28a47fc27.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/ab3b0a9f57290d904c2828edd8e9ccbf34bc9870f770c690b5a00cc28a47fc27.jpg)

![b0d5ed073fe728d439e8c1db3ba5435ff1458c7f482148763120dc0079750720.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/b0d5ed073fe728d439e8c1db3ba5435ff1458c7f482148763120dc0079750720.jpg)

![b5898e93785dd83161a5441673039d979afc16c7993fb33f0673eaca6819aef5.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/b5898e93785dd83161a5441673039d979afc16c7993fb33f0673eaca6819aef5.jpg)

![d097e4ed4e109f40b356c2e92af4c5138579b4fd3442206390ff00fcdf629ffe.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/images/d097e4ed4e109f40b356c2e92af4c5138579b4fd3442206390ff00fcdf629ffe.jpg)

### Tables

![1b80602dc092e448f25b1d66da9109df3a2757302988b43a84b2a727313fd9f2.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/1b80602dc092e448f25b1d66da9109df3a2757302988b43a84b2a727313fd9f2.jpg)

![4df84762f01f3f777104b41b3e1f5d2a57d912501a29e9bb215a3be2ce326d35.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/4df84762f01f3f777104b41b3e1f5d2a57d912501a29e9bb215a3be2ce326d35.jpg)

![54cff66b4714f98f2b1b95d8cef82f20b64a44259a47fa5e463ed68fad4ab251.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/54cff66b4714f98f2b1b95d8cef82f20b64a44259a47fa5e463ed68fad4ab251.jpg)

![58fed96a2d8178d97473dec572e08a2bb487389d386333f5943187709a68ad03.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/58fed96a2d8178d97473dec572e08a2bb487389d386333f5943187709a68ad03.jpg)

![62f299e2f13c65e21fc97c2446f17e8fc621f5e3d27398a4437b16c92cff08bd.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/62f299e2f13c65e21fc97c2446f17e8fc621f5e3d27398a4437b16c92cff08bd.jpg)

![759f3dca32ca2ed5eab812ab778a170876574a3f368f567857909abbfa1084f6.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/759f3dca32ca2ed5eab812ab778a170876574a3f368f567857909abbfa1084f6.jpg)

![a8290fe7c0596950116eba47b05a7bf23cc23f3431af2d501649ac990231ffde.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/a8290fe7c0596950116eba47b05a7bf23cc23f3431af2d501649ac990231ffde.jpg)

![bb95b1eac4009f6fcb6305e3818a6ba2164a206ec2589e21b4619a6c73e80d6e.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/bb95b1eac4009f6fcb6305e3818a6ba2164a206ec2589e21b4619a6c73e80d6e.jpg)

![bea6ec13ddfaf24b386b1cb0929dc47c69685006a453289b765c483cdf333b52.jpg](../icml_results/2899_Neural%20Modular%20Physics%20for%20Elastic%20Simulation/tables/bea6ec13ddfaf24b386b1cb0929dc47c69685006a453289b765c483cdf333b52.jpg)

<a id="paper-2862"></a>

## Object-level Semantic and Spatial Distillation for Open Vocabulary Detection


### Images

![19b1a199f10bc4f41a60af8e547481086997c02d8df6b2dda1fc407adee15918.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/19b1a199f10bc4f41a60af8e547481086997c02d8df6b2dda1fc407adee15918.jpg)

![233bb60bf78c172fe4e48400be331de8dae43d62486f9f79bcf5ca68eaa56619.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/233bb60bf78c172fe4e48400be331de8dae43d62486f9f79bcf5ca68eaa56619.jpg)

![53eee60c25555cefddf25d032f866d39bc8cb7229f1d1a89e264747528f23b25.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/53eee60c25555cefddf25d032f866d39bc8cb7229f1d1a89e264747528f23b25.jpg)

![554437038f60092191e5b9b489683152784b5a64741331a3a954449c55f8a23a.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/554437038f60092191e5b9b489683152784b5a64741331a3a954449c55f8a23a.jpg)

![80ecccf43b26cb8e496330a0e7d27387d4a669bda1bfee7bd09c286b70e8a67f.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/80ecccf43b26cb8e496330a0e7d27387d4a669bda1bfee7bd09c286b70e8a67f.jpg)

![8a6e43c0685b9d315494c6de8d79928de01407135eac3ef73a1aa0a8523a7bc4.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/8a6e43c0685b9d315494c6de8d79928de01407135eac3ef73a1aa0a8523a7bc4.jpg)

![afc248ed8adf6ea80afc2acdb105d6f1cdd7d0f84bac8fbcaf9d87c3a7974f00.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/afc248ed8adf6ea80afc2acdb105d6f1cdd7d0f84bac8fbcaf9d87c3a7974f00.jpg)

![c40ac5b98e34380ae27cce63edfed5221fd0cc84a0e98c948ee30a65508eba93.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/c40ac5b98e34380ae27cce63edfed5221fd0cc84a0e98c948ee30a65508eba93.jpg)

![db2345cdf8a3466c15f4671df08858e75d75bb66308c1b1fec559c9b55756fef.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/images/db2345cdf8a3466c15f4671df08858e75d75bb66308c1b1fec559c9b55756fef.jpg)

### Tables

![698fc501663358db77fd1dec26fc4278179760f930eefbe9a567c3a7f4424730.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/tables/698fc501663358db77fd1dec26fc4278179760f930eefbe9a567c3a7f4424730.jpg)

![9902af46492272a0e0a6e04934bbb6efb190138cf51b0ea989ebb46b844e24b5.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/tables/9902af46492272a0e0a6e04934bbb6efb190138cf51b0ea989ebb46b844e24b5.jpg)

![bc0251b9768a9def841a95d6855e70c9b7fae77d8a926a04fb0682afb7c9d3aa.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/tables/bc0251b9768a9def841a95d6855e70c9b7fae77d8a926a04fb0682afb7c9d3aa.jpg)

![da412330350e93ea1e96ea54664850d4cf5b0a1d00ece875832a48ce3caf4af6.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/tables/da412330350e93ea1e96ea54664850d4cf5b0a1d00ece875832a48ce3caf4af6.jpg)

![fe245fe7007362bae1d95f99679065544eb0baca20c2cc56ba83bb6b8aaa83bd.jpg](../icml_results/2900_Object-level%20Semantic%20and%20Spatial%20Distillation%20for%20Open%20Vocabulary%20Detection/tables/fe245fe7007362bae1d95f99679065544eb0baca20c2cc56ba83bb6b8aaa83bd.jpg)

<a id="paper-2863"></a>

## PlugGuard: A Streaming Safeguard for Large Models via Latent Dynamics-Guided Risk Detection


### Images

![0145a4c8135a6362e79b3b49a79adaec6c2c0f9fb0636a2c6d6d61b8e7cd8ecf.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/0145a4c8135a6362e79b3b49a79adaec6c2c0f9fb0636a2c6d6d61b8e7cd8ecf.jpg)

![23100e623a6d4c38f7ffcb1ed1d45e72a6766ce17e3aeaa30130514008447956.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/23100e623a6d4c38f7ffcb1ed1d45e72a6766ce17e3aeaa30130514008447956.jpg)

![4957c20ce108177e6a68ae0fe4ef65129455df914d5d5797393ca114f761537d.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/4957c20ce108177e6a68ae0fe4ef65129455df914d5d5797393ca114f761537d.jpg)

![700286f10543eedafaa78f4f8a7e4fe7881f2eae607927a646cc59518485bfc0.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/700286f10543eedafaa78f4f8a7e4fe7881f2eae607927a646cc59518485bfc0.jpg)

![93680a9c4bc2add0fe7356819625faf5d4bd3a8156c5a559c42c7db1759eac7c.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/93680a9c4bc2add0fe7356819625faf5d4bd3a8156c5a559c42c7db1759eac7c.jpg)

![94ac2abaa3a89dbd189cf1348deed6d02dded0fc19942a25c8f7423c4f465622.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/94ac2abaa3a89dbd189cf1348deed6d02dded0fc19942a25c8f7423c4f465622.jpg)

![b8c5258346529ef66b444c23f32506825a06e5dd29c8bc11fce21e83ce78bd33.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/b8c5258346529ef66b444c23f32506825a06e5dd29c8bc11fce21e83ce78bd33.jpg)

![dd93598642e814868e4f26b4907f34672a066bb1d2ab2a816799ef09e28e1eef.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/dd93598642e814868e4f26b4907f34672a066bb1d2ab2a816799ef09e28e1eef.jpg)

![ff1aff936662cc4ec26d22ed6c8834015c593d76bc941025a723bc97ee50b9df.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/images/ff1aff936662cc4ec26d22ed6c8834015c593d76bc941025a723bc97ee50b9df.jpg)

### Tables

![015e4e40a7ca73409ac07a47bcc82e52821b8c15da666ba9095959f6b53e1d0d.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/015e4e40a7ca73409ac07a47bcc82e52821b8c15da666ba9095959f6b53e1d0d.jpg)

![0ab3509ea28bc110cab5804b3ba1878f2ab5adc6c0036ebff5f2805ef885945e.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/0ab3509ea28bc110cab5804b3ba1878f2ab5adc6c0036ebff5f2805ef885945e.jpg)

![284c6b5f9cc1cf89b10b7403f654f22bcebf60c5ce660c0ada364aac22461b27.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/284c6b5f9cc1cf89b10b7403f654f22bcebf60c5ce660c0ada364aac22461b27.jpg)

![2b6437359b674e4b193346f0ce2a8756c0611736408e5b762d158cba49fa14f6.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/2b6437359b674e4b193346f0ce2a8756c0611736408e5b762d158cba49fa14f6.jpg)

![2df1fabb09891b003168eb9e7be630992844c27ff88732791afc18c85b1374b9.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/2df1fabb09891b003168eb9e7be630992844c27ff88732791afc18c85b1374b9.jpg)

![303d1db9dd835c572dbcd25ca11ab5a3ba3f2787d70fc68e5982fbf627fa20b9.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/303d1db9dd835c572dbcd25ca11ab5a3ba3f2787d70fc68e5982fbf627fa20b9.jpg)

![3f226944996c0cc4927105f553e504a8a96a15727b556ab6b0b992cfef29e2b5.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/3f226944996c0cc4927105f553e504a8a96a15727b556ab6b0b992cfef29e2b5.jpg)

![5fd877e3c0037e9f8cbba562b88ce135875b8b948fe1d3ec0bfc90aa0b1c4bf6.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/5fd877e3c0037e9f8cbba562b88ce135875b8b948fe1d3ec0bfc90aa0b1c4bf6.jpg)

![68f20f1da8209c7ab0fba5197f66cfdada6a9b904a779c4d8d670243ed161f2b.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/68f20f1da8209c7ab0fba5197f66cfdada6a9b904a779c4d8d670243ed161f2b.jpg)

![aa763be2a1ff1d745c4b8f39bb5eb876f923bcd68d32e284f2dc85507975fc94.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/aa763be2a1ff1d745c4b8f39bb5eb876f923bcd68d32e284f2dc85507975fc94.jpg)

![ac1c3a35706cd7cf41e262ad10c89dad4162947fc94e570a9cdded0114456b64.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/ac1c3a35706cd7cf41e262ad10c89dad4162947fc94e570a9cdded0114456b64.jpg)

![b4dfbbf71a3160afc07b59949a0ecfd9f45a7ee65918f19eeee36b618216a2ff.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/b4dfbbf71a3160afc07b59949a0ecfd9f45a7ee65918f19eeee36b618216a2ff.jpg)

![c9b6c4b7384f6a0a6b9162d3e4c1c9fcb2aafcfee49a61869e8d4a11cb6d9cdc.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/c9b6c4b7384f6a0a6b9162d3e4c1c9fcb2aafcfee49a61869e8d4a11cb6d9cdc.jpg)

![d13e9edcfe0107b6b5e0c3b23fc650708646fd7ef934923a843d0077459cd2f6.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/d13e9edcfe0107b6b5e0c3b23fc650708646fd7ef934923a843d0077459cd2f6.jpg)

![d419dc95582e1e6b21b0160add900c63ff507a5788723c69d882ce10820ec2a3.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/d419dc95582e1e6b21b0160add900c63ff507a5788723c69d882ce10820ec2a3.jpg)

![dc51aa7335243ad6b8720308c9425b48e78cdabd9542e31d72b0b6bc1661b9b1.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/dc51aa7335243ad6b8720308c9425b48e78cdabd9542e31d72b0b6bc1661b9b1.jpg)

![eb2596beeb015515537af829be60212b88130f7a6b70b303c270f0e1cd86a68d.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/eb2596beeb015515537af829be60212b88130f7a6b70b303c270f0e1cd86a68d.jpg)

![ef536432df281563ccc7b6e3103ffd57abe8a5dff5656e170e6c20105fb11281.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/ef536432df281563ccc7b6e3103ffd57abe8a5dff5656e170e6c20105fb11281.jpg)

![f4e6234aca342701ba61445dce4d975636eb9b835f6036d56dc0ffb3dc34d9f3.jpg](../icml_results/2901_PlugGuard_%20A%20Streaming%20Safeguard%20for%20Large%20Models%20via%20Latent%20Dynamics-Guided%20Risk%20Detection/tables/f4e6234aca342701ba61445dce4d975636eb9b835f6036d56dc0ffb3dc34d9f3.jpg)

<a id="paper-2864"></a>

## HEARTS: Benchmarking LLM Reasoning on Health Time Series


### Images

![13c1451f83ab29d07377178dc39e0e0fd6e897faccff847a14bb39a756ca9f85.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/13c1451f83ab29d07377178dc39e0e0fd6e897faccff847a14bb39a756ca9f85.jpg)

![18a472999bc3081db4d3081ac220146b700a49dc6102f08ca2224d97bc8a3517.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/18a472999bc3081db4d3081ac220146b700a49dc6102f08ca2224d97bc8a3517.jpg)

![26a9d7e6ae355c2b1df1ed758fa5a8197bbaffd33dcd89512a6fad08fd0805f3.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/26a9d7e6ae355c2b1df1ed758fa5a8197bbaffd33dcd89512a6fad08fd0805f3.jpg)

![317d288df1f478e2e68e21052344fe752cf4f853413f3fa8790b71ddd41b1e37.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/317d288df1f478e2e68e21052344fe752cf4f853413f3fa8790b71ddd41b1e37.jpg)

![4437f9ae8219f679ddc8c24a93e36ecca8c3415a3c5a988513c637da3495eb24.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/4437f9ae8219f679ddc8c24a93e36ecca8c3415a3c5a988513c637da3495eb24.jpg)

![4dce0808e3c461bbf8946d843be0e47017a0b7b22114ea7d6bc4db904fa9c23b.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/4dce0808e3c461bbf8946d843be0e47017a0b7b22114ea7d6bc4db904fa9c23b.jpg)

![584b97911af23853b41d5549d69cd8e8e4ec1a1b20ac28737af466cf81820d27.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/584b97911af23853b41d5549d69cd8e8e4ec1a1b20ac28737af466cf81820d27.jpg)

![66080497b70ad5f3ddb14cebfab448a2d25e4140c2a327d5713abbcf34710c9f.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/66080497b70ad5f3ddb14cebfab448a2d25e4140c2a327d5713abbcf34710c9f.jpg)

![7c0322eb5c09ac6a5377435b59e496ae354d19cfd890afb691aa7de867c63acb.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/7c0322eb5c09ac6a5377435b59e496ae354d19cfd890afb691aa7de867c63acb.jpg)

![7df97a5cf0cae66e35e61ea56954d1d1d7e41752c204c4e7dbd014e0a49c1f8e.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/7df97a5cf0cae66e35e61ea56954d1d1d7e41752c204c4e7dbd014e0a49c1f8e.jpg)

![832ac5ffe7b263d3927b417134a81e02655b38f728670526c9c9e4c31535be13.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/832ac5ffe7b263d3927b417134a81e02655b38f728670526c9c9e4c31535be13.jpg)

![84d93f5eef03ef9913a7cdfe8ccbfe868ce4383bccc5f1e8b07d1b57e0d7361c.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/84d93f5eef03ef9913a7cdfe8ccbfe868ce4383bccc5f1e8b07d1b57e0d7361c.jpg)

![9097e504b1148820ee7e182717f6ee1fdbd4fa53cdd76451dc1625e1553cbbb7.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/9097e504b1148820ee7e182717f6ee1fdbd4fa53cdd76451dc1625e1553cbbb7.jpg)

![913878a66bd7ebbcaa1b6eba06ce86ac7561a98227d114e65c4d8efdadd65c33.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/913878a66bd7ebbcaa1b6eba06ce86ac7561a98227d114e65c4d8efdadd65c33.jpg)

![979b44016016c6fc1f7841207502088fe4cdaa90975e3236389be1737a00b802.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/979b44016016c6fc1f7841207502088fe4cdaa90975e3236389be1737a00b802.jpg)

![a0127d97bce715d05fb929527d80f87b08aa80bd80cb96c65d76537111c7ae36.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/a0127d97bce715d05fb929527d80f87b08aa80bd80cb96c65d76537111c7ae36.jpg)

![ac00b9cf0b520daedab3a70508fa10c61ae0a2ec0f7271fe1f17f22be130995c.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/ac00b9cf0b520daedab3a70508fa10c61ae0a2ec0f7271fe1f17f22be130995c.jpg)

![b70ecc6f7468564e8d886212d2793d65417466a0d36718220135142df05260bc.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/b70ecc6f7468564e8d886212d2793d65417466a0d36718220135142df05260bc.jpg)

![f7a3676dcd07ea45f22722e5876890b50dc9ed539f5082533ae227b650cc666b.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/images/f7a3676dcd07ea45f22722e5876890b50dc9ed539f5082533ae227b650cc666b.jpg)

### Tables

![0b40a66c53b5d3dffcd29ed2c50453901744305405b4ec085d255c98ccd6b13a.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/0b40a66c53b5d3dffcd29ed2c50453901744305405b4ec085d255c98ccd6b13a.jpg)

![29e1ef84c8dd07c22e3806406a4899212bbf59928325e5d1c087912698096ba2.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/29e1ef84c8dd07c22e3806406a4899212bbf59928325e5d1c087912698096ba2.jpg)

![54869a5206b131559cdb742ec07ef3bd2dedce17a9feff7c2288db641bc6672c.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/54869a5206b131559cdb742ec07ef3bd2dedce17a9feff7c2288db641bc6672c.jpg)

![7b0780fac959c5a2b0e4c70a7b22f8b727615b05698fa9b13dc13acefb276eff.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/7b0780fac959c5a2b0e4c70a7b22f8b727615b05698fa9b13dc13acefb276eff.jpg)

![84343944d54cfa17dfa8659ed1ed3c8d45a5181461890923252484e241bd1787.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/84343944d54cfa17dfa8659ed1ed3c8d45a5181461890923252484e241bd1787.jpg)

![8ceaf350ec5b1ad87ccc1dccbf6d3c8a70ffd892f6a1d8bcda664783d705431b.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/8ceaf350ec5b1ad87ccc1dccbf6d3c8a70ffd892f6a1d8bcda664783d705431b.jpg)

![a5f88540bc69e9d772301d163bc24d25e18e714f25cedf909bb6769e3b3449a0.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/a5f88540bc69e9d772301d163bc24d25e18e714f25cedf909bb6769e3b3449a0.jpg)

![c0de18399dd69285b2a62b08a9c5d36859474463d896a772d6a5a4a6a91e5190.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/c0de18399dd69285b2a62b08a9c5d36859474463d896a772d6a5a4a6a91e5190.jpg)

![d1642c6db5983fe11573c7ba3a79f1733a42411c4dc54981b7fefae81f25fa35.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/d1642c6db5983fe11573c7ba3a79f1733a42411c4dc54981b7fefae81f25fa35.jpg)

![d6796a48012406d445b8813a58db99554e0cae360da59c778a1371933151e4c4.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/d6796a48012406d445b8813a58db99554e0cae360da59c778a1371933151e4c4.jpg)

![f1451965304d768c17c3fd763e303471f2c7a9dfad34b5d98896d4edf310f42d.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/f1451965304d768c17c3fd763e303471f2c7a9dfad34b5d98896d4edf310f42d.jpg)

![f1ed1cd7918ffeb640152034655fb047e01c72ccb4c08a0a2af054ca06270338.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/f1ed1cd7918ffeb640152034655fb047e01c72ccb4c08a0a2af054ca06270338.jpg)

![f33a92f4ca9fbadeb6a4f400767203110e6d02eac69f6a589174cda9cbbcbf2a.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/f33a92f4ca9fbadeb6a4f400767203110e6d02eac69f6a589174cda9cbbcbf2a.jpg)

![f5142c75340c97982bfc6a7a88d662fc48976ff59dbe78ddd6d59a460020b6b8.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/f5142c75340c97982bfc6a7a88d662fc48976ff59dbe78ddd6d59a460020b6b8.jpg)

![f887feaf74be4ecb12e463c0c39fb8c28ac29042fcf9a1172c8f1e2c733e43d6.jpg](../icml_results/2902_HEARTS_%20Benchmarking%20LLM%20Reasoning%20on%20Health%20Time%20Series/tables/f887feaf74be4ecb12e463c0c39fb8c28ac29042fcf9a1172c8f1e2c733e43d6.jpg)

<a id="paper-2865"></a>

## Robust Vision-Language Models via Manifold-Adversarial Adapters


### Images

![3ec9d2e6c5b0418a2fe2c7331f657b2051ead989e5f7de7aa89cad56c97100f0.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/3ec9d2e6c5b0418a2fe2c7331f657b2051ead989e5f7de7aa89cad56c97100f0.jpg)

![52078e8068b03d625863d65abc70cb04019d2d0d2c4f5105a5e756960eef69b8.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/52078e8068b03d625863d65abc70cb04019d2d0d2c4f5105a5e756960eef69b8.jpg)

![752ce30a669cf3ae131519823d87fd360e0b31ed9383826821accb8c551e2e3f.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/752ce30a669cf3ae131519823d87fd360e0b31ed9383826821accb8c551e2e3f.jpg)

![a6268ed45d76118704358fce5f8966d251cc5d2793af31097f38cda6f9223d03.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/a6268ed45d76118704358fce5f8966d251cc5d2793af31097f38cda6f9223d03.jpg)

![c04b57aba1711f0c6d3c8e0df63c6c06847b142465cb80c25a5ce532609a452a.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/c04b57aba1711f0c6d3c8e0df63c6c06847b142465cb80c25a5ce532609a452a.jpg)

![cb5407de005e9dc06b3467dfa71fad4ddec15882e5ef76993717d82c1ebdb4e9.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/cb5407de005e9dc06b3467dfa71fad4ddec15882e5ef76993717d82c1ebdb4e9.jpg)

![d187fa48970a819812a024a677d3476fe4638940c398369dd39c14a9134d8494.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/images/d187fa48970a819812a024a677d3476fe4638940c398369dd39c14a9134d8494.jpg)

### Tables

![142d7ab25cfadd4150feb196af727f45d06866da4d273b60100fecfb19189bdf.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/142d7ab25cfadd4150feb196af727f45d06866da4d273b60100fecfb19189bdf.jpg)

![1eb8191ccc090da6af41077f380bec885b530726ac075ce607875eb29c75a120.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/1eb8191ccc090da6af41077f380bec885b530726ac075ce607875eb29c75a120.jpg)

![2b482f0b5326c71d5eb661ca2f1bfed3e1a7ca3c40bb28017973757813002efa.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/2b482f0b5326c71d5eb661ca2f1bfed3e1a7ca3c40bb28017973757813002efa.jpg)

![369be2c1ae2c4775142076c25da731674e38f7ae702bc8f5bc640e619a81b1f9.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/369be2c1ae2c4775142076c25da731674e38f7ae702bc8f5bc640e619a81b1f9.jpg)

![3ac8cf19ed4d004461b160535812997f7c99756e42aed52316d6d4eed9bd6349.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/3ac8cf19ed4d004461b160535812997f7c99756e42aed52316d6d4eed9bd6349.jpg)

![3d4f0567238960f8174b2cb34af443e4f152ab27b97358d14367019582f3fbe1.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/3d4f0567238960f8174b2cb34af443e4f152ab27b97358d14367019582f3fbe1.jpg)

![3da2e8cc04841210dcc0b0fb2ace45f50bd8bf8ee183ff7c9d4006744856adf3.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/3da2e8cc04841210dcc0b0fb2ace45f50bd8bf8ee183ff7c9d4006744856adf3.jpg)

![3ecd25a0c47b7bba556a797e40dcbba1445a8e2c349600661173bd5cec924a35.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/3ecd25a0c47b7bba556a797e40dcbba1445a8e2c349600661173bd5cec924a35.jpg)

![5a50399c0721688d358355782c9277dc10c1cc692ee4535518d25e49afd5ba24.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/5a50399c0721688d358355782c9277dc10c1cc692ee4535518d25e49afd5ba24.jpg)

![93d8da1525a13618e51dfbc26c5d591dfa24f33d7276cae3c57d68101127c63f.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/93d8da1525a13618e51dfbc26c5d591dfa24f33d7276cae3c57d68101127c63f.jpg)

![9f6524847a0eb4704d8da2be2a73982aaa2eb267d4d41d6967527ed063a2bc5b.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/9f6524847a0eb4704d8da2be2a73982aaa2eb267d4d41d6967527ed063a2bc5b.jpg)

![a4a9fc49db0ba6d18b94b0a0abd104f706f25d0b0d5191e694908f9e9d974db3.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/a4a9fc49db0ba6d18b94b0a0abd104f706f25d0b0d5191e694908f9e9d974db3.jpg)

![c1c60cb6005f102ad7ef5b19638922e88cb1f769f67731fd167d3fab1b695aa7.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/c1c60cb6005f102ad7ef5b19638922e88cb1f769f67731fd167d3fab1b695aa7.jpg)

![c67d04ea6879e4c71345a98b96b4104efb4560bd98af3ede2c778a45b3845c32.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/c67d04ea6879e4c71345a98b96b4104efb4560bd98af3ede2c778a45b3845c32.jpg)

![d7072b4024b40f22ac8955462722c8dfb029d95cb71112db1b13762b7224fcfd.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/d7072b4024b40f22ac8955462722c8dfb029d95cb71112db1b13762b7224fcfd.jpg)

![fc72716c58e594fafd77787a597507ce143f39f74a8b86633f841224d2d04abf.jpg](../icml_results/2903_Robust%20Vision-Language%20Models%20via%20Manifold-Adversarial%20Adapters/tables/fc72716c58e594fafd77787a597507ce143f39f74a8b86633f841224d2d04abf.jpg)

<a id="paper-2866"></a>

## Reasoning LLM Improves Speaker Recognition in Long-form TV Dramas


### Images

![41fb8f82eed7712d38e465dfff70c6785f2dba2a42a37617f4b162ee6db88987.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/images/41fb8f82eed7712d38e465dfff70c6785f2dba2a42a37617f4b162ee6db88987.jpg)

![9eb2806998064e19014716e5fe684bf20be73ab0c81ed7ef0f758ec514d25acb.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/images/9eb2806998064e19014716e5fe684bf20be73ab0c81ed7ef0f758ec514d25acb.jpg)

![d0bf1d65a56961c23610b4da922d23ae7511abb4c0fefda27a46afc12759d115.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/images/d0bf1d65a56961c23610b4da922d23ae7511abb4c0fefda27a46afc12759d115.jpg)

![e3d42ebce088ac4e837328a2b0ee9643a9e8bf2309f0ff47279563ab527215e3.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/images/e3d42ebce088ac4e837328a2b0ee9643a9e8bf2309f0ff47279563ab527215e3.jpg)

### Tables

![528a7e89b2f648678a1da507a2cafbb1d4fd1fea9b5c2f27a667f6ffaee9e2fe.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/528a7e89b2f648678a1da507a2cafbb1d4fd1fea9b5c2f27a667f6ffaee9e2fe.jpg)

![5487f67832d2a16458e35103413e02d6ea68ea669055fb36fd3017be2cee69c9.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/5487f67832d2a16458e35103413e02d6ea68ea669055fb36fd3017be2cee69c9.jpg)

![5a6a88a30c0211ff9e90edfd0e85a01d9302fb01113cd578207bae52b149a300.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/5a6a88a30c0211ff9e90edfd0e85a01d9302fb01113cd578207bae52b149a300.jpg)

![94f09236617479f3b8e89be2d3a1868a6d81b70b67298246c3c70e41ecd31672.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/94f09236617479f3b8e89be2d3a1868a6d81b70b67298246c3c70e41ecd31672.jpg)

![d1f675cd85b6d78cf320c4e521ce345fa1df99fe2c9593083e369c678687cd7c.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/d1f675cd85b6d78cf320c4e521ce345fa1df99fe2c9593083e369c678687cd7c.jpg)

![d676ad1dd63a260c1f2c3a73a6e7b7fe8f2454add8dfd4b7b31ae84bdf4b37d6.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/d676ad1dd63a260c1f2c3a73a6e7b7fe8f2454add8dfd4b7b31ae84bdf4b37d6.jpg)

![e2b33846293a273ef9cc392092e3098f081944e04e3f6e3bc99ae61d290266df.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/e2b33846293a273ef9cc392092e3098f081944e04e3f6e3bc99ae61d290266df.jpg)

![fe3c00de227cb16e5398464519d7057764d1a9fe255cc830741dbc8ea186d662.jpg](../icml_results/2904_Reasoning%20LLM%20Improves%20Speaker%20Recognition%20in%20Long-form%20TV%20Dramas/tables/fe3c00de227cb16e5398464519d7057764d1a9fe255cc830741dbc8ea186d662.jpg)

<a id="paper-2867"></a>

## PRIM: Cooperative Dynamic Token Compression for Efficient Large Multimodal Models


### Images

![096d4b20bbb1832d2f6fe80457a1c5173c34479ac02f9907fc0b13e2823952aa.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/096d4b20bbb1832d2f6fe80457a1c5173c34479ac02f9907fc0b13e2823952aa.jpg)

![3734baaa6862b2bfe94145e6adddf269acd175f2e78e8927429d61ad44d1ed60.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/3734baaa6862b2bfe94145e6adddf269acd175f2e78e8927429d61ad44d1ed60.jpg)

![674468e9e61da09ec26fc69d36332ac89c94407c174e41a5b1fae101f9c95cbd.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/674468e9e61da09ec26fc69d36332ac89c94407c174e41a5b1fae101f9c95cbd.jpg)

![8317cdb937cc0a713b485eb83b547211272f2d7a128c5041c72331ab14293336.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/8317cdb937cc0a713b485eb83b547211272f2d7a128c5041c72331ab14293336.jpg)

![a1c102aac07dcc7eae8d7e2e53a935721ad00841efa9ff69335366bed7d9f816.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/a1c102aac07dcc7eae8d7e2e53a935721ad00841efa9ff69335366bed7d9f816.jpg)

![b3199dbcea8c89b0f8d557bacaf7e15a00cde7fbb6b0ee11fa7679d2a7e4aa28.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/b3199dbcea8c89b0f8d557bacaf7e15a00cde7fbb6b0ee11fa7679d2a7e4aa28.jpg)

![cced478718d24df636bc275057361bc70a5d9d8dc06a007ed892a883fd5a4831.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/cced478718d24df636bc275057361bc70a5d9d8dc06a007ed892a883fd5a4831.jpg)

![d928fff4f4bce780b7666eec31b9738bb4400c6553b0e22be189f206d4b66ddf.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/images/d928fff4f4bce780b7666eec31b9738bb4400c6553b0e22be189f206d4b66ddf.jpg)

### Tables

![173c4a5855f12e63774cdbb9ef6b102bb5241e0fdf3cebcf2795a94de7e98f06.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/173c4a5855f12e63774cdbb9ef6b102bb5241e0fdf3cebcf2795a94de7e98f06.jpg)

![21598794215071a2dbd800763316452bed68b18f768c6a900d785dac88db1827.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/21598794215071a2dbd800763316452bed68b18f768c6a900d785dac88db1827.jpg)

![62a92c511db77e8fe84fb138651210cf46372346d9e9a15f1e96d44cfa97580e.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/62a92c511db77e8fe84fb138651210cf46372346d9e9a15f1e96d44cfa97580e.jpg)

![b20ee0ff180dbd92deba879b8ab05dc2a83a3ad957b6b55eb3d4fab795ee4598.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/b20ee0ff180dbd92deba879b8ab05dc2a83a3ad957b6b55eb3d4fab795ee4598.jpg)

![e7ddc3b641849ab41bd05bb722815e1c9ad58636698ea857116bfdd5ee427f0a.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/e7ddc3b641849ab41bd05bb722815e1c9ad58636698ea857116bfdd5ee427f0a.jpg)

![e88bc8fe07b9bc0e45293504dfb72712983e65068a1d1ef6fe7a0707cf6fa811.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/e88bc8fe07b9bc0e45293504dfb72712983e65068a1d1ef6fe7a0707cf6fa811.jpg)

![f570e353202c89c555c43c8e63ad6778766920c9b0d51abd04910aba4138b177.jpg](../icml_results/2905_PRIM_%20Cooperative%20Dynamic%20Token%20Compression%20for%20Efficient%20Large%20Multimodal%20Models/tables/f570e353202c89c555c43c8e63ad6778766920c9b0d51abd04910aba4138b177.jpg)

<a id="paper-2868"></a>

## Optimizing Visual Generative Models via Distribution-wise Rewards


### Images

![07b9ecb1930c99136b84715ae4104a5000f2f5a3648a873bbae77e1d41e5b9fd.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/07b9ecb1930c99136b84715ae4104a5000f2f5a3648a873bbae77e1d41e5b9fd.jpg)

![07e58e1e80b1faf802aace245f7b6574f109459843c8c28b1d88ac49b374209a.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/07e58e1e80b1faf802aace245f7b6574f109459843c8c28b1d88ac49b374209a.jpg)

![17111fcfc2eb08665fceda1f4ef483a9c363f03589a3dc5ddddf5ef8d24c5a81.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/17111fcfc2eb08665fceda1f4ef483a9c363f03589a3dc5ddddf5ef8d24c5a81.jpg)

![1ea93a83bcc046435d30cfddb5bff39e7f1a6bf8f31e0710e87805755cfc6535.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/1ea93a83bcc046435d30cfddb5bff39e7f1a6bf8f31e0710e87805755cfc6535.jpg)

![5ca6f014158c9d3ad303fcf147fec4842884492409893d5be5754180897de816.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/5ca6f014158c9d3ad303fcf147fec4842884492409893d5be5754180897de816.jpg)

![6415bbc2a064ac0ae380c8e7f958b674b8c57432c5c71f6d2af824cb540c0ff7.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/6415bbc2a064ac0ae380c8e7f958b674b8c57432c5c71f6d2af824cb540c0ff7.jpg)

![8f0069f37448a1aec72549055752c458eecd636fdbdb1cd6840676192026dd65.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/8f0069f37448a1aec72549055752c458eecd636fdbdb1cd6840676192026dd65.jpg)

![da26405349e56c7b6a574dc18cdbb04cfb075502f186d55c8cd091417d3bdae1.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/da26405349e56c7b6a574dc18cdbb04cfb075502f186d55c8cd091417d3bdae1.jpg)

![eaa76c8af4ad1da44140e813d76703a28d1c47135b3a78e17d1e9cd51baa784f.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/eaa76c8af4ad1da44140e813d76703a28d1c47135b3a78e17d1e9cd51baa784f.jpg)

![f75a6aa181b2cfc7d63c543126e3edbf4e1c0c8cb730e40949538f0e72c88c17.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/images/f75a6aa181b2cfc7d63c543126e3edbf4e1c0c8cb730e40949538f0e72c88c17.jpg)

### Tables

![09bd513c1856aee034c9e5351cc08ac5c2e7e6f91510ad01368ab849686ad219.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/tables/09bd513c1856aee034c9e5351cc08ac5c2e7e6f91510ad01368ab849686ad219.jpg)

![1ae532eaf2d5731ab2771a6c763595042dab3c303b6fd27e1977dd4de5fc3211.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/tables/1ae532eaf2d5731ab2771a6c763595042dab3c303b6fd27e1977dd4de5fc3211.jpg)

![20fd48241babc8bff2743944a733c64de49ed7026ba508a100a5e255c2f3315d.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/tables/20fd48241babc8bff2743944a733c64de49ed7026ba508a100a5e255c2f3315d.jpg)

![79b19d22a7579a2daaa5048d16ffd414de13720dd652799421dc6ffb4177436a.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/tables/79b19d22a7579a2daaa5048d16ffd414de13720dd652799421dc6ffb4177436a.jpg)

![c9d9719bd983dbc2fda3d6f3476820cc01034879c2e5d8759a60fbcdd95961d3.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/tables/c9d9719bd983dbc2fda3d6f3476820cc01034879c2e5d8759a60fbcdd95961d3.jpg)

![e8e830dadbbadf1af7d601805cb8e0d6ea3aad9606b75003204251c2f87f3417.jpg](../icml_results/2906_Optimizing%20Visual%20Generative%20Models%20via%20Distribution-wise%20Rewards/tables/e8e830dadbbadf1af7d601805cb8e0d6ea3aad9606b75003204251c2f87f3417.jpg)

<a id="paper-2869"></a>

## DRFusion: Drift-Resilient Temporally Consistent Infrared–Visible Video Fusion


### Images

![26a3114faed7000c5eb62395f708e90ffb9e1bc01d1ade6d41ca64ce900d6a4d.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/26a3114faed7000c5eb62395f708e90ffb9e1bc01d1ade6d41ca64ce900d6a4d.jpg)

![26b1bed82c68effc4c26605c3ada80d5d3d228eac794d7321ce8fdd3cac27dc4.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/26b1bed82c68effc4c26605c3ada80d5d3d228eac794d7321ce8fdd3cac27dc4.jpg)

![2fecf5fb22b86165c14ad46b789bc33a6a27562767685a37d2ee113acb74d60e.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/2fecf5fb22b86165c14ad46b789bc33a6a27562767685a37d2ee113acb74d60e.jpg)

![3c448cd4420ed0f3ddd8304ee53d4ff67ec4cf7886d248cc2a8888ea092a87ba.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/3c448cd4420ed0f3ddd8304ee53d4ff67ec4cf7886d248cc2a8888ea092a87ba.jpg)

![a38a0c294e1a4a9af71078b69cc6d330b40d8ba468726971c44abe3e7b0875cf.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/a38a0c294e1a4a9af71078b69cc6d330b40d8ba468726971c44abe3e7b0875cf.jpg)

![d5ba594f0a04ce72a4038c73e0733c78c5a4a93bb69633e3a6a7d01959ac2d13.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/d5ba594f0a04ce72a4038c73e0733c78c5a4a93bb69633e3a6a7d01959ac2d13.jpg)

![da187c432d98415e521e178860fe3f46fbe5ce28d1d81ca4f90ca92c22642e3d.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/images/da187c432d98415e521e178860fe3f46fbe5ce28d1d81ca4f90ca92c22642e3d.jpg)

### Tables

![8a79bce193e3fcaef78bca1397ee832ef97dd5f447d3a8496929e301c6f9ecbb.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/tables/8a79bce193e3fcaef78bca1397ee832ef97dd5f447d3a8496929e301c6f9ecbb.jpg)

![aadb4af451732558ab2fef6b717a2e7a381b20efb1ec17f3b9d3c9543ca88bc4.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/tables/aadb4af451732558ab2fef6b717a2e7a381b20efb1ec17f3b9d3c9543ca88bc4.jpg)

![bad645c68c1ca84a9d1c0df74a3052eaebffcbd7ece3ab7fa133adaea088355a.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/tables/bad645c68c1ca84a9d1c0df74a3052eaebffcbd7ece3ab7fa133adaea088355a.jpg)

![c13041fd8fb1f3c76c0d686515bba80d9ae8e53dcb123e06501ddf303b406ea0.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/tables/c13041fd8fb1f3c76c0d686515bba80d9ae8e53dcb123e06501ddf303b406ea0.jpg)

![eeeed27cc65ddcff4d0121c36065aaaa8a1ee8fe7cfdfc3517d5fb97de022bcc.jpg](../icml_results/2907_DRFusion_%20Drift-Resilient%20Temporally%20Consistent%20Infrared%E2%80%93Visible%20Video%20Fusion/tables/eeeed27cc65ddcff4d0121c36065aaaa8a1ee8fe7cfdfc3517d5fb97de022bcc.jpg)

<a id="paper-2870"></a>

## The Optimal Token Baseline: Variance Reduction for Long-Horizon LLM-RL


### Images

![049be4cc20fbfdbb526ae39abde4d929fdcdf36f5ddb5eefb67f0ade50a2b0b6.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/049be4cc20fbfdbb526ae39abde4d929fdcdf36f5ddb5eefb67f0ade50a2b0b6.jpg)

![0acd56d0d92a3c452d1158e137fa7ffaea5eeed5b4b6244a9b0bd3b1e87741df.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/0acd56d0d92a3c452d1158e137fa7ffaea5eeed5b4b6244a9b0bd3b1e87741df.jpg)

![13abcabedc84ee3fb9dba871a4a391fd0b8e3e02552ae0a553e83a33f6b4d1a3.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/13abcabedc84ee3fb9dba871a4a391fd0b8e3e02552ae0a553e83a33f6b4d1a3.jpg)

![1577c9f815794139df1fddbfadd354fa87562032ef27cfb9e192260463c29760.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/1577c9f815794139df1fddbfadd354fa87562032ef27cfb9e192260463c29760.jpg)

![4043b9d02c5e1a35ce2e19b5d75853acd9d3114a522b158185c60dc22f7132f8.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/4043b9d02c5e1a35ce2e19b5d75853acd9d3114a522b158185c60dc22f7132f8.jpg)

![59d74cf4d0a74b756287fe3ee0ce8a14526b9ae9cf303435bf89080d059805ae.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/59d74cf4d0a74b756287fe3ee0ce8a14526b9ae9cf303435bf89080d059805ae.jpg)

![5c9438f44db9ff0b8099d3a2915f0b23dda75004a0f858ac5ab1466966a4471f.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/5c9438f44db9ff0b8099d3a2915f0b23dda75004a0f858ac5ab1466966a4471f.jpg)

![7a0688d84cce11d64f2d9cc1266766aecfb8477f92ebfcc0bd81d261ffcd7033.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/7a0688d84cce11d64f2d9cc1266766aecfb8477f92ebfcc0bd81d261ffcd7033.jpg)

![85c337f535a9c9c080dccad93ed7999df88f85faf2ec9860a412d695b01d5ee8.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/85c337f535a9c9c080dccad93ed7999df88f85faf2ec9860a412d695b01d5ee8.jpg)

![85fde85eadc09ef15e5e72d39e218c6423846cc7d52e79c7fa4b7d4ae628e930.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/85fde85eadc09ef15e5e72d39e218c6423846cc7d52e79c7fa4b7d4ae628e930.jpg)

![8c567f5c9d50c19bf70c6d9664e39a18ed80b78b4cc60934cfdebd7ec2b65ff0.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/8c567f5c9d50c19bf70c6d9664e39a18ed80b78b4cc60934cfdebd7ec2b65ff0.jpg)

![9a5ecfab05728dce81322d269991493dd3aa2ad5d54785072eb5b09838227580.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/9a5ecfab05728dce81322d269991493dd3aa2ad5d54785072eb5b09838227580.jpg)

![b209c309093e15cfe80f58831fdef3c48427f173e758fafd981876374aeb640d.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/b209c309093e15cfe80f58831fdef3c48427f173e758fafd981876374aeb640d.jpg)

![b8abe7681475edd7c2009ac7a19cd460a5f200e8334efa565985d9e91a303fef.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/b8abe7681475edd7c2009ac7a19cd460a5f200e8334efa565985d9e91a303fef.jpg)

![bbad9bc38135397179bc23d74528aa0de71cacb377bfe4fbf6368331ed14989d.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/bbad9bc38135397179bc23d74528aa0de71cacb377bfe4fbf6368331ed14989d.jpg)

![c3ab301017dc3493741805df5ced046299d5ad737def0204aeda5904304f5623.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/c3ab301017dc3493741805df5ced046299d5ad737def0204aeda5904304f5623.jpg)

![c3c4ad8212457283828a4adb416a5ad809cad418c6ccf88137700b87355f4053.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/c3c4ad8212457283828a4adb416a5ad809cad418c6ccf88137700b87355f4053.jpg)

![cca5648450d5b18814137059dc1ec44d204c2a68fa8694360759b960ded32b0f.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/cca5648450d5b18814137059dc1ec44d204c2a68fa8694360759b960ded32b0f.jpg)

![d5f617f237849cbdd8f2b1c101951aaf5faf0d49c00b0e945e8667233c69c15e.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/d5f617f237849cbdd8f2b1c101951aaf5faf0d49c00b0e945e8667233c69c15e.jpg)

![da80b5777560834610efe7c2b79cd9d8eae450909885df7c5da2ce23cab35289.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/images/da80b5777560834610efe7c2b79cd9d8eae450909885df7c5da2ce23cab35289.jpg)

### Tables

![0c00e5c006c81d1d4862ba8b6ef69c2db32711b83c4a4f6bbf8a7ef477a091af.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/tables/0c00e5c006c81d1d4862ba8b6ef69c2db32711b83c4a4f6bbf8a7ef477a091af.jpg)

![2abee5bc6a43d1d2a236c40a08e201cd8e5c9e9041b2ee8bf654763aa749adc3.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/tables/2abee5bc6a43d1d2a236c40a08e201cd8e5c9e9041b2ee8bf654763aa749adc3.jpg)

![6218ac6ec07d1a006333b672167447b88828275c12e518167461a48805fd1113.jpg](../icml_results/2908_The%20Optimal%20Token%20Baseline_%20Variance%20Reduction%20for%20Long-Horizon%20LLM-RL/tables/6218ac6ec07d1a006333b672167447b88828275c12e518167461a48805fd1113.jpg)

<a id="paper-2871"></a>

## Spatially-Regularized Entropy for Discriminative Token Merging in Fine-Grained Re-Identification

### Images

![56fb5be429f80483cf70294ccd681f9a93f93d029e0b61db4bba185cb2c2c93b.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/56fb5be429f80483cf70294ccd681f9a93f93d029e0b61db4bba185cb2c2c93b.jpg)

![5e5bd0ac61635c3bcec12f5e056a0b53073e6977ef5cb012673bbf268cb4a93f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/5e5bd0ac61635c3bcec12f5e056a0b53073e6977ef5cb012673bbf268cb4a93f.jpg)

![9c0efcbec9f76b514ba06c9314ff948b1281d07a240d3e647264615211540013.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/9c0efcbec9f76b514ba06c9314ff948b1281d07a240d3e647264615211540013.jpg)

![b500ee35b86a617ff13c9c52a25a6a056b194163e4eb159eea2b459f96a7d151.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/b500ee35b86a617ff13c9c52a25a6a056b194163e4eb159eea2b459f96a7d151.jpg)

![b60f72a42ac9567e6803a3f86226d1a7ab8b364c34773bfaa0255b688772902d.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/b60f72a42ac9567e6803a3f86226d1a7ab8b364c34773bfaa0255b688772902d.jpg)

![dc9bc41cf85f52ea18959a2a9b36b1c875b1525290c5c92d22007bd3578c41ae.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/dc9bc41cf85f52ea18959a2a9b36b1c875b1525290c5c92d22007bd3578c41ae.jpg)

![ec1c558bf251de27d820cd25648a57d57b40d35fdc3babd9e83d47cfeddc00fe.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/ec1c558bf251de27d820cd25648a57d57b40d35fdc3babd9e83d47cfeddc00fe.jpg)

![effaad8e4756b2710423e9660aa2111d285cdefe0d3315a84a1e89d332fcb00d.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/effaad8e4756b2710423e9660aa2111d285cdefe0d3315a84a1e89d332fcb00d.jpg)

![f7eae85d6c7ce9fbfc91dec5a2e59b7a3a8467a060cf735054e2ef2529985d81.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/f7eae85d6c7ce9fbfc91dec5a2e59b7a3a8467a060cf735054e2ef2529985d81.jpg)

### Tables

![052f16907d74ba267ac36af1d00abb70574d1ff5c788c758bef9efc6cce26590.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/052f16907d74ba267ac36af1d00abb70574d1ff5c788c758bef9efc6cce26590.jpg)

![0cf717f989eb030c76d98b57490d4b00665e0973a707f37ba00ede44c66fa5f7.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/0cf717f989eb030c76d98b57490d4b00665e0973a707f37ba00ede44c66fa5f7.jpg)

![190474d364d7dcc464fff43e77cefdf9fbc07ea1641234cd650118785109ce1f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/190474d364d7dcc464fff43e77cefdf9fbc07ea1641234cd650118785109ce1f.jpg)

![2026eb985b8ff85ad0b9b24c92e13ef8d531f421a9455fed2b64017000fae9fd.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/2026eb985b8ff85ad0b9b24c92e13ef8d531f421a9455fed2b64017000fae9fd.jpg)

![52a4c5c637961c24ff2f5812ac69a69817306ebf918e58f8560ee607f5cbbf85.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/52a4c5c637961c24ff2f5812ac69a69817306ebf918e58f8560ee607f5cbbf85.jpg)

![600627ee1aa562b4013b456af008556e01f39c2739e352adfbea0ce2ee9d1ebd.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/600627ee1aa562b4013b456af008556e01f39c2739e352adfbea0ce2ee9d1ebd.jpg)

![8f4723703f85dafe505587f7a24945c76815d5666b64ef8400c9712f1bf97591.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/8f4723703f85dafe505587f7a24945c76815d5666b64ef8400c9712f1bf97591.jpg)

![92f9afe62d23d62aa44536826aeb65bd37d83225e2e8392f6df6f43b03a2dd5f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/92f9afe62d23d62aa44536826aeb65bd37d83225e2e8392f6df6f43b03a2dd5f.jpg)

![937315f0f24888003bc9050cf1aec92189b9d01744514843c47e5028fbd4aecd.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/937315f0f24888003bc9050cf1aec92189b9d01744514843c47e5028fbd4aecd.jpg)

![a2d5517a6af14f164e5e3a61cadd08605f471dbcea3ffac40e2e11fd82ef75f4.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/a2d5517a6af14f164e5e3a61cadd08605f471dbcea3ffac40e2e11fd82ef75f4.jpg)

![d3cd9cf8e54ee69db95f0abd87bb00e2bb9884186daf5886052d25783f63223f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/d3cd9cf8e54ee69db95f0abd87bb00e2bb9884186daf5886052d25783f63223f.jpg)

![f452cc5ec35f4f2aacdc152ee440e4264f1840e71b5ded00ebd2f18761e69fa2.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/f452cc5ec35f4f2aacdc152ee440e4264f1840e71b5ded00ebd2f18761e69fa2.jpg)
