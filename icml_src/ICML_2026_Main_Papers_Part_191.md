# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 191 of 197

## 目录 (Table of Contents)

1. [VideoBrain: Learning Adaptive Frame Sampling for Long Video Understanding](#paper-6271)
2. [Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance](#paper-6272)
3. [Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning](#paper-6273)
4. [OVLR: Efficient, Scalable, and Robust Training via Output-Level Variance-Reduced Likelihood Ratio](#paper-6274)
5. [Leveraging Evidence Priors for Robust Prompt Learning under Noisy Supervision in Vision-Language Models](#paper-6275)
6. [LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads?](#paper-6276)
7. [AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning](#paper-6277)
8. [From Static Constraints to Dynamic Adaptation: Sample-Level Constraint Relaxation for Offline-to-Online Reinforcement Learning](#paper-6278)
9. [Predicting the Order of Upcoming Tokens Improves Language Modeling](#paper-6279)
10. [From Representation to Action: A Unified Laplacian Framework for Spatial Representation and Path Planning](#paper-6280)
11. [IVQ: Structured and Lightweight Vector Quantization via Binary Hierarchical Composition Inspired by $\textitIChing$](#paper-6281)
12. [Enhancing Numerical Prediction in LLMs via Smooth MMD Alignment](#paper-6282)
13. [GLAD: Bidirectional Structure-Attribute Alignment via Latent Graph Diffusion Models](#paper-6283)
14. [From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates](#paper-6284)
15. [AnyBand-Diff: A Unified Remote Sensing Image Generation and Band Repair Framework with Spectral Priors](#paper-6285)
16. [A Random Matrix Theory of Masked Self-Supervised Learning](#paper-6286)
17. [Fast KV Compaction via Attention Matching](#paper-6287)
18. [Position: Predictive Uncertainty Is Not Enough — Joint Distribution for Full Uncertainty Representation](#paper-6288)
19. [Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](#paper-6289)
20. [Position: We Need Large Language Models Optimized For Our Well-Being](#paper-6290)
21. [Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](#paper-6291)
22. [Position: The Alignment Community is Unintentionally Building a Censor’s Toolkit](#paper-6292)
23. [Position: Agentic Systems Should be General](#paper-6293)
24. [Position: AI Leaderboards Are Underserving the Global South: A Case Study from India](#paper-6294)
25. [Position: Stop Automating Peer Review Without Rigorous Evaluation](#paper-6295)
26. [Position: Deployed Reinforcement Learning should be Continual](#paper-6296)
27. [Position: Unplugging a Seemingly Sentient Machine Is the Rational Choice — A Metaphysical Perspective](#paper-6297)
28. [Position: Evaluation of ECG Representations Must Be Fixed](#paper-6298)
29. [Position: LLM-Safety Evaluations Lack Robustness](#paper-6299)
30. [Position: Assistive AI requires Personalized Specialists, not Generalists](#paper-6300)
31. [Position: It is Time to Virtualize Foundation Models with a Self-evolving Operating System Layer](#paper-6301)
32. [Position: Trustworthy AI Suffers from Invariance Conflicts and Causality is The Solution](#paper-6302)
33. [Position: We Need AI Efficiency Incentives for Accessibility and Sustainability](#paper-6303)

---


<a id="paper-6271"></a>

## VideoBrain: Learning Adaptive Frame Sampling for Long Video Understanding

### Images

![1e51f2b059b92d1c7e83e9518a67969ef4effc3056528ca6e1a6941445ef7ae2.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/1e51f2b059b92d1c7e83e9518a67969ef4effc3056528ca6e1a6941445ef7ae2.jpg)

![2e3ff0929b3362ab2d0296d1fe085064a8061397e36a56b509911f3c6e4f3d01.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/2e3ff0929b3362ab2d0296d1fe085064a8061397e36a56b509911f3c6e4f3d01.jpg)

![591d13ac9a4a4c500ac1b289981ccf5654f43f30c39f73daa7db1f9906c3c3ed.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/591d13ac9a4a4c500ac1b289981ccf5654f43f30c39f73daa7db1f9906c3c3ed.jpg)

![6392dcac01952136af6eb19e9a1094f3c6bc40a4ba479e93c2f8da959a06a983.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/6392dcac01952136af6eb19e9a1094f3c6bc40a4ba479e93c2f8da959a06a983.jpg)

![7831a4ed43f4d172ed00d59cabc33ceb857128e158e156124df50b307d4cb21d.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/7831a4ed43f4d172ed00d59cabc33ceb857128e158e156124df50b307d4cb21d.jpg)

![8295d3ee6b5f3b08b744d1f8af2d03972debcd18262888f9744a335655baefd1.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/8295d3ee6b5f3b08b744d1f8af2d03972debcd18262888f9744a335655baefd1.jpg)

![877e0374a23aa51656ec29b606384d2b8e6aa9143e423863bb4e84a5f3504148.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/877e0374a23aa51656ec29b606384d2b8e6aa9143e423863bb4e84a5f3504148.jpg)

![a0374b49a96bdcc76acfbabcdf18a2072f57462f5ce33aa9d721e7f740d18714.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/a0374b49a96bdcc76acfbabcdf18a2072f57462f5ce33aa9d721e7f740d18714.jpg)

![cc4d61eee72d942a8863c25a4fbf7169eca8f764ad8a0f765d396450ae7d45d4.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/cc4d61eee72d942a8863c25a4fbf7169eca8f764ad8a0f765d396450ae7d45d4.jpg)

![d5919df4950c1f65e1eade7304157f2bc408879dd89731f3ae97ba3c04bee66c.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/d5919df4950c1f65e1eade7304157f2bc408879dd89731f3ae97ba3c04bee66c.jpg)

![e4e7e388b53162601fbf915770619dca877084499da1350fb5b777ed41503e93.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/e4e7e388b53162601fbf915770619dca877084499da1350fb5b777ed41503e93.jpg)

### Tables

![00d1808aef0433af537667af171e85ea7777e009d102e80f15afc72a6dc88265.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/00d1808aef0433af537667af171e85ea7777e009d102e80f15afc72a6dc88265.jpg)

![0c1807641bd82e098ee6f5f7a89032459e50e2035e41ccb3fa54f4a4f4213382.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/0c1807641bd82e098ee6f5f7a89032459e50e2035e41ccb3fa54f4a4f4213382.jpg)

![1c5646ea61950a3f5dd99ae7a81fb89526994ba6e0ce5b79de7ab2f8cead4c57.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/1c5646ea61950a3f5dd99ae7a81fb89526994ba6e0ce5b79de7ab2f8cead4c57.jpg)

![299eb66806b91609bc72110ee1836baf7f20a2dc06188d1492c56bc0a540eb78.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/299eb66806b91609bc72110ee1836baf7f20a2dc06188d1492c56bc0a540eb78.jpg)

![2f090011783caad18b0127e3054c777eeba283f9b334c3bfd8c46b1bdafb5f07.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/2f090011783caad18b0127e3054c777eeba283f9b334c3bfd8c46b1bdafb5f07.jpg)

![544ab8bd97bdb848015542b1dc889b7d277488a3a664dff5cc4ba319636735c4.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/544ab8bd97bdb848015542b1dc889b7d277488a3a664dff5cc4ba319636735c4.jpg)

![9285d9eea2b51f46c20edfb0cce377084466e89c7872bdd7302b0ae41a00b653.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/9285d9eea2b51f46c20edfb0cce377084466e89c7872bdd7302b0ae41a00b653.jpg)

![a93c2acc3f3f9e9afde4a7d4ad9fd1a4c06420ed09e0929ca26ee7ca9fd64cda.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/a93c2acc3f3f9e9afde4a7d4ad9fd1a4c06420ed09e0929ca26ee7ca9fd64cda.jpg)

![c7f73a15aa3f5b6195f3e7e971c7609ee06b1ee5f578db5c5e2f0dfbcf1ba9f3.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/c7f73a15aa3f5b6195f3e7e971c7609ee06b1ee5f578db5c5e2f0dfbcf1ba9f3.jpg)

![e923b58c8564186d77639ec9a224feb7c723314baa1ebdb817d28b697245ecc2.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/e923b58c8564186d77639ec9a224feb7c723314baa1ebdb817d28b697245ecc2.jpg)

![f5a24a7478e1f8f3bf5537d3c4f6a22c2beccf54fec3560d823bbbe06e4b6e77.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/f5a24a7478e1f8f3bf5537d3c4f6a22c2beccf54fec3560d823bbbe06e4b6e77.jpg)

![f7ac2c92b3036a61ef928839a243e44a85c16e7fa68083f0228a9a8400943fa9.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/f7ac2c92b3036a61ef928839a243e44a85c16e7fa68083f0228a9a8400943fa9.jpg)

## VideoBrain: Learning Adaptive Frame Sampling for Long Video Understanding


### Images

![06899c23090f8f9cac8df2f4e0d440fd120f66e55f5f3df9e3d5a7905dc92ff8.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/06899c23090f8f9cac8df2f4e0d440fd120f66e55f5f3df9e3d5a7905dc92ff8.jpg)

![326bf0e4940a683b8ba755f2083d034563329a0b4cd5a3e4db5b4ec9504caeec.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/326bf0e4940a683b8ba755f2083d034563329a0b4cd5a3e4db5b4ec9504caeec.jpg)

![44dd3aae8f5e4dc29cc3c1c1edf4d1ad392915fcaa38d2113f2f7d603407ec09.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/44dd3aae8f5e4dc29cc3c1c1edf4d1ad392915fcaa38d2113f2f7d603407ec09.jpg)

![70716df6a9e4e73e0ce9f239f171851d95326979a396fc08dbf16f871cf83e5c.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/70716df6a9e4e73e0ce9f239f171851d95326979a396fc08dbf16f871cf83e5c.jpg)

![78c2f74f9a2e97dcc9543f009597a3f7d5e1bff9fee6c92a4bb24b3eeebaa93b.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/78c2f74f9a2e97dcc9543f009597a3f7d5e1bff9fee6c92a4bb24b3eeebaa93b.jpg)

![78ec4783607f56042396ff103026b0b04350dbbb70bfa82a12ad98197d7cce52.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/78ec4783607f56042396ff103026b0b04350dbbb70bfa82a12ad98197d7cce52.jpg)

![880d089eed4ee8ef7931e2dc30273eef2eba651a3cffdb06863082c9b36e510b.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/880d089eed4ee8ef7931e2dc30273eef2eba651a3cffdb06863082c9b36e510b.jpg)

![8dde0bad4e7ec0b6993732b9f3645d44e039bcb622d82aaefa46d96c0f6e8748.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/8dde0bad4e7ec0b6993732b9f3645d44e039bcb622d82aaefa46d96c0f6e8748.jpg)

![a5e7283ce4f034ae689850e18ec395783625836812302fb662feea9c0ea920cb.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/a5e7283ce4f034ae689850e18ec395783625836812302fb662feea9c0ea920cb.jpg)

![bac4838245251f1ff7f2c43b4c16b62b85690a491045f09a70f8a25ac0bc0fbf.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/bac4838245251f1ff7f2c43b4c16b62b85690a491045f09a70f8a25ac0bc0fbf.jpg)

![d45dcb6484f30785cb0183953b06b70e9758a989ed9011793fe7f45586acac17.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/d45dcb6484f30785cb0183953b06b70e9758a989ed9011793fe7f45586acac17.jpg)

![eb08157c96fe240e1d48d701224164804dfab3a9abb74316af0f4cc30b28748c.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/eb08157c96fe240e1d48d701224164804dfab3a9abb74316af0f4cc30b28748c.jpg)

![ebc0d0d0484d4bf3c70fea438c34b463d802bd0e51f96fe98641d370f0243201.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/ebc0d0d0484d4bf3c70fea438c34b463d802bd0e51f96fe98641d370f0243201.jpg)

![f1ec6b573986cda2b2f44f9ecf1d47eb218f9f6a4b58645b23175cf3bb425844.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/images/f1ec6b573986cda2b2f44f9ecf1d47eb218f9f6a4b58645b23175cf3bb425844.jpg)

### Tables

![1de8a27a54c25b6bf2a8503598d3b16162af3efb3dbc9374d9f9b73328a316ef.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/1de8a27a54c25b6bf2a8503598d3b16162af3efb3dbc9374d9f9b73328a316ef.jpg)

![47ec74e150e8cf44f834705acb88258b44d192d7c46d6eb5d9500cdecd8f4506.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/47ec74e150e8cf44f834705acb88258b44d192d7c46d6eb5d9500cdecd8f4506.jpg)

![8cd5cd06dee32b06114544a1cd9311e9fc53b4ec6d888078d3b3b57984810f1a.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/8cd5cd06dee32b06114544a1cd9311e9fc53b4ec6d888078d3b3b57984810f1a.jpg)

![9b7abed409040a96a1f68b18cf8f4645a441d27c4d690ed0b217b533a503ae97.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/9b7abed409040a96a1f68b18cf8f4645a441d27c4d690ed0b217b533a503ae97.jpg)

![d26bab49b5c81fc7f3bb4ec093f3f19c5551400f3a82ce60f049942ca4f7ea46.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/d26bab49b5c81fc7f3bb4ec093f3f19c5551400f3a82ce60f049942ca4f7ea46.jpg)

![dca713db15f37c78c726eb26ab3be73d470f5adc9440ad4b662f7cb62cd1af60.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/dca713db15f37c78c726eb26ab3be73d470f5adc9440ad4b662f7cb62cd1af60.jpg)

![e65845672ef72a7d5070fdc83d4bf8aebdfc803a88ec6edcf1bb6bd733b0ab15.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/e65845672ef72a7d5070fdc83d4bf8aebdfc803a88ec6edcf1bb6bd733b0ab15.jpg)

![eb22abb4d88ce820f55b0cd35a4a867dfc6bff3d05e1c4b87b5ffd9991c87ab2.jpg](../icml_results/6324_VideoBrain_%20Learning%20Adaptive%20Frame%20Sampling%20for%20Long%20Video%20Understanding/tables/eb22abb4d88ce820f55b0cd35a4a867dfc6bff3d05e1c4b87b5ffd9991c87ab2.jpg)

<a id="paper-6272"></a>

## Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance


### Images

![00d1ea8eb8c9dd1c67a692d08e5e4ad0f94206b3ab42f537b729a3be9f4f749b.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/images/00d1ea8eb8c9dd1c67a692d08e5e4ad0f94206b3ab42f537b729a3be9f4f749b.jpg)

### Tables

![2b43ca4b5882a3c030cd917c210f352bf8c0650dc7a27a95417a5796c5055f1c.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/2b43ca4b5882a3c030cd917c210f352bf8c0650dc7a27a95417a5796c5055f1c.jpg)

![68e01f547ed0c2aabc8b10849e928b082f8de8e058b6fa1b58358476bc54a36b.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/68e01f547ed0c2aabc8b10849e928b082f8de8e058b6fa1b58358476bc54a36b.jpg)

![6df28069ff88068b4f8135cacb3f0ae57b7eb3299c010cddb9939ca65219c4fb.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/6df28069ff88068b4f8135cacb3f0ae57b7eb3299c010cddb9939ca65219c4fb.jpg)

![abd869821200779c374ccbf266d838731f118b28a20e2b5c0d2398aae94a7951.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/abd869821200779c374ccbf266d838731f118b28a20e2b5c0d2398aae94a7951.jpg)

![b20c0c22adf21df26d36b1d5d9acb38c384326f5ea8232b382f04bad21f872b7.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/b20c0c22adf21df26d36b1d5d9acb38c384326f5ea8232b382f04bad21f872b7.jpg)

![e65457a33810ebea8e693d41f6c29b5b198c21bb69e39132f70b12a0096e7be8.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/e65457a33810ebea8e693d41f6c29b5b198c21bb69e39132f70b12a0096e7be8.jpg)

![f9c0a34db7469f5a72b31d2eebba9e53f8467e1c1d7283382e9da4452f45caea.jpg](../icml_results/6325_Variational%20Speculative%20Decoding_%20Rethinking%20Draft%20Training%20from%20Token%20Likelihood%20to%20Sequence%20Accept/tables/f9c0a34db7469f5a72b31d2eebba9e53f8467e1c1d7283382e9da4452f45caea.jpg)

<a id="paper-6273"></a>

## Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning


### Images

![2b7e12af8f4245f167a1e34edc2201f37b9a19da3e8c95e6f838270dfee58b13.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/2b7e12af8f4245f167a1e34edc2201f37b9a19da3e8c95e6f838270dfee58b13.jpg)

![52b401cc2059ec73dad26d8c67db09df45f043e12d2664d56e3054475ae81bf6.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/52b401cc2059ec73dad26d8c67db09df45f043e12d2664d56e3054475ae81bf6.jpg)

![a7315a8916c8fb1249677d263386f59ca301edd68168fda7647b7513d5c9c5a8.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/a7315a8916c8fb1249677d263386f59ca301edd68168fda7647b7513d5c9c5a8.jpg)

![b593dfbde606eda976e1acf67cb0deefd6b1de31d0a94705d025d55974952ca4.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/b593dfbde606eda976e1acf67cb0deefd6b1de31d0a94705d025d55974952ca4.jpg)

![bd1490403b92063c034dffc682b39010fd3fd3108377babcaad65cc2984adeec.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/bd1490403b92063c034dffc682b39010fd3fd3108377babcaad65cc2984adeec.jpg)

![e47d5aae1489c52b0bf17aa16ebca7af442fd3f18788ab1c51c63b587108f71f.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/e47d5aae1489c52b0bf17aa16ebca7af442fd3f18788ab1c51c63b587108f71f.jpg)

![e89a6373422769d5dbc177e25ccb1718005c2c172f89aa1a65e8b1c00589f59f.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/images/e89a6373422769d5dbc177e25ccb1718005c2c172f89aa1a65e8b1c00589f59f.jpg)

### Tables

![02157c59e660ed3a400d9d38675129345fe060f7e0e1454e2babcdc596dd558a.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/02157c59e660ed3a400d9d38675129345fe060f7e0e1454e2babcdc596dd558a.jpg)

![16dcd604723c83acf7f486af2a8dd33cd381e2044f43b89f0e46845f9a033f3a.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/16dcd604723c83acf7f486af2a8dd33cd381e2044f43b89f0e46845f9a033f3a.jpg)

![1e19eb1cdfd5554cb9171e7f707cbab017195029cf90ad8b12ec614c81c40686.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/1e19eb1cdfd5554cb9171e7f707cbab017195029cf90ad8b12ec614c81c40686.jpg)

![21fdce431aba0c17fdcf18ddbb66c0e08f06bde979cf5bebbb2fc6df18b231c4.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/21fdce431aba0c17fdcf18ddbb66c0e08f06bde979cf5bebbb2fc6df18b231c4.jpg)

![34a66f5944335f192d2910e2b3ff97486bf64a3333aef2bbc09c46ccb8806090.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/34a66f5944335f192d2910e2b3ff97486bf64a3333aef2bbc09c46ccb8806090.jpg)

![3783bfe0796859b99c81011285a77142e700dc812934ff064c1bece957c0053b.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/3783bfe0796859b99c81011285a77142e700dc812934ff064c1bece957c0053b.jpg)

![7599ad150392a26d2f40e0859e4e57aa01fe8a9489c1bea3efac50b5b04c876e.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/7599ad150392a26d2f40e0859e4e57aa01fe8a9489c1bea3efac50b5b04c876e.jpg)

![7dc6208e06683134c81d874219c95b4f19d60c9ff38d83f6c5fbb8ee763c8be5.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/7dc6208e06683134c81d874219c95b4f19d60c9ff38d83f6c5fbb8ee763c8be5.jpg)

![7e5b2657a879d9d1bdd3c739d37dbaeb42f65cd58ea2adde57be00c96a9ef0e3.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/7e5b2657a879d9d1bdd3c739d37dbaeb42f65cd58ea2adde57be00c96a9ef0e3.jpg)

![82b3015a867b89915024fe4351729e06e18ac08e51b7ad11a5ca1b93c26fcfbf.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/82b3015a867b89915024fe4351729e06e18ac08e51b7ad11a5ca1b93c26fcfbf.jpg)

![886d7e3e06215a5ddbd462c06ae10e61b5787970a48c53c1b823138285e6fbf5.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/886d7e3e06215a5ddbd462c06ae10e61b5787970a48c53c1b823138285e6fbf5.jpg)

![8e351f3a1aee375008535079ebdcf7fd05be54741b474d86794646d14d847652.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/8e351f3a1aee375008535079ebdcf7fd05be54741b474d86794646d14d847652.jpg)

![adcde2de1680383076b4f6bd27f6246804f25955b0dda5bc850a0a3b6e6fee56.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/adcde2de1680383076b4f6bd27f6246804f25955b0dda5bc850a0a3b6e6fee56.jpg)

![ba00bf88fb99f137078af4e45106299ecdbc231fc85c476a3e6f345dcc128e30.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/ba00bf88fb99f137078af4e45106299ecdbc231fc85c476a3e6f345dcc128e30.jpg)

![bbcc6b8b2647a10d5a4e00cf3eca2d9d4ef7b5c1093f4a04eecacf7d184fa152.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/bbcc6b8b2647a10d5a4e00cf3eca2d9d4ef7b5c1093f4a04eecacf7d184fa152.jpg)

![bf0f7f15258137aa827f701aa97d8ae679b117faa1b2e1c33b34c64dc86cab04.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/bf0f7f15258137aa827f701aa97d8ae679b117faa1b2e1c33b34c64dc86cab04.jpg)

![c3866be9fd7354d52239f726ad734a1796ca321423c9762d034dda1ea5169db8.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/c3866be9fd7354d52239f726ad734a1796ca321423c9762d034dda1ea5169db8.jpg)

![c9c1a47f0b4fcd74ce5c05c9f42e0cd7608191af5389bc80d863726186a1ad4d.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/c9c1a47f0b4fcd74ce5c05c9f42e0cd7608191af5389bc80d863726186a1ad4d.jpg)

![ccd7900e302cd5515279681912196206ef2b36b524a3d18d579745104af4c7f1.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/ccd7900e302cd5515279681912196206ef2b36b524a3d18d579745104af4c7f1.jpg)

![cd2209c61116044e8d3becb61376fc9f79bc907cd56c6741c9c3cc3c59a92371.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/cd2209c61116044e8d3becb61376fc9f79bc907cd56c6741c9c3cc3c59a92371.jpg)

![d095f8f65bdb29eb67c3b3268f13a59a01f7f7e819b0d9cebe0dbc64ff0dca74.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/d095f8f65bdb29eb67c3b3268f13a59a01f7f7e819b0d9cebe0dbc64ff0dca74.jpg)

![dd39a5f083d5620d42862e01c378f32f2234e575f19283d50a137ced4ded5f04.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/dd39a5f083d5620d42862e01c378f32f2234e575f19283d50a137ced4ded5f04.jpg)

![fc3afcf1b25841a3150ea8e5353dac54ee7326b605f3c0dacdfcae52c8bd3bae.jpg](../icml_results/6326_Utility-Diversity%20Aware%20Online%20Batch%20Selection%20for%20LLM%20Supervised%20Fine-tuning/tables/fc3afcf1b25841a3150ea8e5353dac54ee7326b605f3c0dacdfcae52c8bd3bae.jpg)

<a id="paper-6274"></a>

## OVLR: Efficient, Scalable, and Robust Training via Output-Level Variance-Reduced Likelihood Ratio


### Images

![09da6c457b46e689b514bcf85c4721f96fa8b786481b07372786327830c0c770.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/09da6c457b46e689b514bcf85c4721f96fa8b786481b07372786327830c0c770.jpg)

![0defab2242a9d2fed5402016d69f6e98cda618eb4a6977656ac745f3a17997d8.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/0defab2242a9d2fed5402016d69f6e98cda618eb4a6977656ac745f3a17997d8.jpg)

![0ed48d0d4afff796f80d5448aea7fc3ca98d2d8a8277e0cb536fba9f786bea36.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/0ed48d0d4afff796f80d5448aea7fc3ca98d2d8a8277e0cb536fba9f786bea36.jpg)

![4b8179c400403bcb076c7ddf2bf6d204c6fd1cde5904dc849bf4aa4e5e8a77f5.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/4b8179c400403bcb076c7ddf2bf6d204c6fd1cde5904dc849bf4aa4e5e8a77f5.jpg)

![63a90f73c70c4a6657f9f0b85ead5b47df72609dc92896aa585053b5922fda15.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/63a90f73c70c4a6657f9f0b85ead5b47df72609dc92896aa585053b5922fda15.jpg)

![993424ef0b4cf3161b7d81e4f02575250923fa05f9023f2e5ed0073c324aeca6.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/993424ef0b4cf3161b7d81e4f02575250923fa05f9023f2e5ed0073c324aeca6.jpg)

![ad455dbdd3d50e5823b1752caba57531aaf855bef71888067a57c8d328c43979.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/ad455dbdd3d50e5823b1752caba57531aaf855bef71888067a57c8d328c43979.jpg)

![cdc8ebdd8bdce90b7a97e2604a8a0d1bf3cad1c67ba8a8fdbcc72688ea48331e.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/cdc8ebdd8bdce90b7a97e2604a8a0d1bf3cad1c67ba8a8fdbcc72688ea48331e.jpg)

![dd071208932418786cf4a1a3ff54289f776a85a7af4aa5b9d6143efd8ed4ca9c.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/dd071208932418786cf4a1a3ff54289f776a85a7af4aa5b9d6143efd8ed4ca9c.jpg)

![e258b668a7caee3c436b873a0cfa4f6a7f63cce5bf7e2f8d558b61f355997817.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/images/e258b668a7caee3c436b873a0cfa4f6a7f63cce5bf7e2f8d558b61f355997817.jpg)

### Tables

![285cb43554a850455e9435150df340d1bc3d00d5ead77f04bf1f0346cc797861.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/285cb43554a850455e9435150df340d1bc3d00d5ead77f04bf1f0346cc797861.jpg)

![2b8fbe2373afadc79624c2c97e28d73e6e024dbb1ce9e8dbeb07b15526824895.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/2b8fbe2373afadc79624c2c97e28d73e6e024dbb1ce9e8dbeb07b15526824895.jpg)

![4231f043ea0a1e5f21f91d76c466614d780ad0b6efbac4ee498b197b8760655e.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/4231f043ea0a1e5f21f91d76c466614d780ad0b6efbac4ee498b197b8760655e.jpg)

![43aa3f88e7e410a7468f9485d4f84c15ed37920c4d3e44625ca0c7fe11e3704e.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/43aa3f88e7e410a7468f9485d4f84c15ed37920c4d3e44625ca0c7fe11e3704e.jpg)

![44ce7f429bf547f50ae8579d65f5db61e4e1ac79e85f8a76e58dabae11a9d531.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/44ce7f429bf547f50ae8579d65f5db61e4e1ac79e85f8a76e58dabae11a9d531.jpg)

![4e7082f0f634edd48f6bb7a471f19727b7e7f6bd7f1e9899bee83ff0a1b3702d.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/4e7082f0f634edd48f6bb7a471f19727b7e7f6bd7f1e9899bee83ff0a1b3702d.jpg)

![7aabdec96ba428b9379cdea6cc4cdac793b7ecf0ebb4f3da13bf4de1f3cbd5b9.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/7aabdec96ba428b9379cdea6cc4cdac793b7ecf0ebb4f3da13bf4de1f3cbd5b9.jpg)

![7b3d745f48109151e9be71dcb1b5930f59a908c6d83e56445e79031fba8bc929.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/7b3d745f48109151e9be71dcb1b5930f59a908c6d83e56445e79031fba8bc929.jpg)

![80708d576b21071d1be22589553f2d08026923caea8c976047e121e24bc20363.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/80708d576b21071d1be22589553f2d08026923caea8c976047e121e24bc20363.jpg)

![9d621c301c6d5d627db03bf32f3705ded4d4c03a54155a2723bec5b2f9ba3df3.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/9d621c301c6d5d627db03bf32f3705ded4d4c03a54155a2723bec5b2f9ba3df3.jpg)

![aa3faebff76b4d9f2c71c7dca578d3f78e819614b2b19851c66f93d1ff6d03ae.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/aa3faebff76b4d9f2c71c7dca578d3f78e819614b2b19851c66f93d1ff6d03ae.jpg)

![b862a9b73f0d8f55aa7ff51a84022c5b081854c488b0009ffd31faf983b7de32.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/b862a9b73f0d8f55aa7ff51a84022c5b081854c488b0009ffd31faf983b7de32.jpg)

![bf94318f62fc33d449cfba0b6dd59179a0ee3bc80a86288814b2421cb6eb24c1.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/bf94318f62fc33d449cfba0b6dd59179a0ee3bc80a86288814b2421cb6eb24c1.jpg)

![c8f2f1222fd935f6f72effe6cd45e807c26005c15c9efd7104645b9f85eb6848.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/c8f2f1222fd935f6f72effe6cd45e807c26005c15c9efd7104645b9f85eb6848.jpg)

![e54fe9e0d6b7e9ae1143cfd4532bf895a6a5bf434051aeb8fa7888b54697fd3a.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/e54fe9e0d6b7e9ae1143cfd4532bf895a6a5bf434051aeb8fa7888b54697fd3a.jpg)

![f21558f9fcb29f3a68fe23c8f3d564e43d6214535478fa0ed8f8c9da2755ec09.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/f21558f9fcb29f3a68fe23c8f3d564e43d6214535478fa0ed8f8c9da2755ec09.jpg)

![f25fdfed5cdee8a1cd54b69931df9e74011d3dbe8bba2e4741cb70d8fad40300.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/f25fdfed5cdee8a1cd54b69931df9e74011d3dbe8bba2e4741cb70d8fad40300.jpg)

![f5ef899d2835a7e4974e2f13e5895ae34b87edfe780fdb4bb169e4f99acfe7b1.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/f5ef899d2835a7e4974e2f13e5895ae34b87edfe780fdb4bb169e4f99acfe7b1.jpg)

![faa29787d455dfbb6859d72e52977ea58598a94341348e1d135b06f8351efa51.jpg](../icml_results/6327_OVLR_%20Efficient%2C%20Scalable%2C%20and%20Robust%20Training%20via%20Output-Level%20Variance-Reduced%20Likelihood%20Ratio/tables/faa29787d455dfbb6859d72e52977ea58598a94341348e1d135b06f8351efa51.jpg)

<a id="paper-6275"></a>

## Leveraging Evidence Priors for Robust Prompt Learning under Noisy Supervision in Vision-Language Models


### Images

![0fdc794eeac199671123c05df85ce662b8b0b679a03c9a8709374ac5384ba118.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/0fdc794eeac199671123c05df85ce662b8b0b679a03c9a8709374ac5384ba118.jpg)

![1536404dc3986c898882c4b25fd057ff57bcaadd2b16210fea31d03ca76d9148.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/1536404dc3986c898882c4b25fd057ff57bcaadd2b16210fea31d03ca76d9148.jpg)

![289ce93baafce10da241d160265f323d65279acb3eb007d646a7e6cc19e14cfb.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/289ce93baafce10da241d160265f323d65279acb3eb007d646a7e6cc19e14cfb.jpg)

![8698b9df4c37c8fcbf1803045ada8234a380c72d5d7eeeffc50fe59fa36568d5.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/8698b9df4c37c8fcbf1803045ada8234a380c72d5d7eeeffc50fe59fa36568d5.jpg)

![95f2bc5d7564649bd701624cb19ba0d4dfd691a87e0d3ba75be92b948fd0a74f.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/95f2bc5d7564649bd701624cb19ba0d4dfd691a87e0d3ba75be92b948fd0a74f.jpg)

![a18eb4bde556fe8abceeba2a78379db4573d61c86910e36faebc1396983fac56.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/a18eb4bde556fe8abceeba2a78379db4573d61c86910e36faebc1396983fac56.jpg)

![ca53033150e9ff1d779d2572de8ea0fdfbda723b42fc0b33b0888b0dff56453c.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/images/ca53033150e9ff1d779d2572de8ea0fdfbda723b42fc0b33b0888b0dff56453c.jpg)

### Tables

![13b2d7b752a012dc9d028f189e699d0d2bd8e7e290301f6c6d80ed729f1a168a.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/13b2d7b752a012dc9d028f189e699d0d2bd8e7e290301f6c6d80ed729f1a168a.jpg)

![14f963ad25b29722313020a1c0b219feac24f0499789e7d4781532d9f7397888.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/14f963ad25b29722313020a1c0b219feac24f0499789e7d4781532d9f7397888.jpg)

![1ef2970783ea042a62c23b1378eafa73437d0308507c83b3973a4f004d427d33.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/1ef2970783ea042a62c23b1378eafa73437d0308507c83b3973a4f004d427d33.jpg)

![23305ecbcfbc8c86e136763f966ae02080e9e0ed953dd70cdf883568326bd602.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/23305ecbcfbc8c86e136763f966ae02080e9e0ed953dd70cdf883568326bd602.jpg)

![3da5b33d3fe5d9cc5e6602a8e8fce915b051b15fa623bb26f59350435b1a640c.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/3da5b33d3fe5d9cc5e6602a8e8fce915b051b15fa623bb26f59350435b1a640c.jpg)

![40d2c42c82c439ddaa385ba8cf43ed6760723942136bcdddebd784af3540e171.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/40d2c42c82c439ddaa385ba8cf43ed6760723942136bcdddebd784af3540e171.jpg)

![61039ed9eec8af7c715e0477a1917371fbfffc4a43c617051d8391585eeb7864.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/61039ed9eec8af7c715e0477a1917371fbfffc4a43c617051d8391585eeb7864.jpg)

![8387c223f9cb91597d4f43af027057e4f6fe8909da2c82b6f17b9beb5369ffae.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/8387c223f9cb91597d4f43af027057e4f6fe8909da2c82b6f17b9beb5369ffae.jpg)

![8cebd1b5730e04e1feccd18454f688c67b2f2702fe07d4ff0559ddb4ea8674d4.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/8cebd1b5730e04e1feccd18454f688c67b2f2702fe07d4ff0559ddb4ea8674d4.jpg)

![aef4334f99bbbebe0cc14c43626f4d58a570638d0b81835864aed24570a341cf.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/aef4334f99bbbebe0cc14c43626f4d58a570638d0b81835864aed24570a341cf.jpg)

![b4bcfaa30d7071aa32ba0557b569757556e24c303b45df65cbdf6d810b95d367.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/b4bcfaa30d7071aa32ba0557b569757556e24c303b45df65cbdf6d810b95d367.jpg)

![ba5e58cb6c55b9810279c740bf06eadfa90f2c35c376a1e83bede39794e4587a.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/ba5e58cb6c55b9810279c740bf06eadfa90f2c35c376a1e83bede39794e4587a.jpg)

![bcc694c7f57314e41f38135cd0d5b64ef8b65f9eb4a8f6504f58af83e745a9e6.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/bcc694c7f57314e41f38135cd0d5b64ef8b65f9eb4a8f6504f58af83e745a9e6.jpg)

![dcae967a7401be5834227981bb3d55830df2aaa331347d0231cb50cad933c102.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/dcae967a7401be5834227981bb3d55830df2aaa331347d0231cb50cad933c102.jpg)

![f263f4f3d125502412fe5227b946f007b39dac7b74c715627dde9a63d6dc1d64.jpg](../icml_results/6328_Leveraging%20Evidence%20Priors%20for%20Robust%20Prompt%20Learning%20under%20Noisy%20Supervision%20in%20Vision-Language%20Mod/tables/f263f4f3d125502412fe5227b946f007b39dac7b74c715627dde9a63d6dc1d64.jpg)

<a id="paper-6276"></a>

## LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads?


### Images

![0388b3eb15391927403333f181b7a28f2e0cd511bfba5c83d876ae1daf1eadc2.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/0388b3eb15391927403333f181b7a28f2e0cd511bfba5c83d876ae1daf1eadc2.jpg)

![103f0737d86948936fa4e7cb16d278d935c08668efae92cd1a77a354fb28e49c.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/103f0737d86948936fa4e7cb16d278d935c08668efae92cd1a77a354fb28e49c.jpg)

![114d7f972f3a0d567498a8fdef7ffae7bd8887aeb0e3df060c7920888551819f.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/114d7f972f3a0d567498a8fdef7ffae7bd8887aeb0e3df060c7920888551819f.jpg)

![141b079f2bec46870f4ab4c27ec0a5da77e32d6482fc0841934784e2c1da715c.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/141b079f2bec46870f4ab4c27ec0a5da77e32d6482fc0841934784e2c1da715c.jpg)

![2d1fea1973cd15a2d839323d566da0d43d11821106629daf4d111f5683037279.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/2d1fea1973cd15a2d839323d566da0d43d11821106629daf4d111f5683037279.jpg)

![394e54b8fc157ff26e3a71a95947d3c096719dd1f32272c91d80070bbcbb9cb3.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/394e54b8fc157ff26e3a71a95947d3c096719dd1f32272c91d80070bbcbb9cb3.jpg)

![45635bd31602c1910b876f0623ea4b714d8b246439d4c8795dc4d5d128193a91.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/45635bd31602c1910b876f0623ea4b714d8b246439d4c8795dc4d5d128193a91.jpg)

![64d053420bffb2d4bf7a1494bc1bfba626cfd03c9324c36a80e84bf06bbc8459.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/64d053420bffb2d4bf7a1494bc1bfba626cfd03c9324c36a80e84bf06bbc8459.jpg)

![755c89bc2aacaa4432606e572e5e9b319d748e0a507356e08dc9f7732e46c380.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/755c89bc2aacaa4432606e572e5e9b319d748e0a507356e08dc9f7732e46c380.jpg)

![7dbac0ca1da714fbe9fef6a7afcb19736c23fe95d17ce9dfe83bf748a980d560.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/7dbac0ca1da714fbe9fef6a7afcb19736c23fe95d17ce9dfe83bf748a980d560.jpg)

![8214ed2fd8a5faebca1059e72bc1a77327d9dfb3c9408a5765d689245aa1e573.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/8214ed2fd8a5faebca1059e72bc1a77327d9dfb3c9408a5765d689245aa1e573.jpg)

![844f65a8785183e35a295d6f5e77954ef57a75a625002c5cac58732cea483778.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/844f65a8785183e35a295d6f5e77954ef57a75a625002c5cac58732cea483778.jpg)

![85630fa674c5800673e7cecf3a78100bdefa295655fa6eb5c47f3bc0cfcdb973.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/85630fa674c5800673e7cecf3a78100bdefa295655fa6eb5c47f3bc0cfcdb973.jpg)

![8f1b722c02f16594e9f27e6a08e8f045eebd4012fe143abd935c50aaae980999.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/8f1b722c02f16594e9f27e6a08e8f045eebd4012fe143abd935c50aaae980999.jpg)

![904d34e5ce6b622004a7d89d95967eabbe59a8dd17dfc67e80a7d1fc4dce2d4e.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/904d34e5ce6b622004a7d89d95967eabbe59a8dd17dfc67e80a7d1fc4dce2d4e.jpg)

![939f8f7fc18acc92934efde0d175718baa210969e5fdae5efd97603727513c38.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/939f8f7fc18acc92934efde0d175718baa210969e5fdae5efd97603727513c38.jpg)

![9932693f23a3cd0c6b5034ca79392949988297daa6e9434c71c92dd58368011d.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/9932693f23a3cd0c6b5034ca79392949988297daa6e9434c71c92dd58368011d.jpg)

![abb934a047f3f97af73ad2f5632a6d051f1583c0eddd906d859f9a2597d992ee.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/abb934a047f3f97af73ad2f5632a6d051f1583c0eddd906d859f9a2597d992ee.jpg)

![af3505430f3e49d74738ae0d00616bd3dfa22fe55ae71d85f0fe270fb5839962.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/af3505430f3e49d74738ae0d00616bd3dfa22fe55ae71d85f0fe270fb5839962.jpg)

![bc112b00979d4ba0d403c11e1ecb192925268f1023e89f91fa3c6cd9270c0d08.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/bc112b00979d4ba0d403c11e1ecb192925268f1023e89f91fa3c6cd9270c0d08.jpg)

![c53a6d71f261e647f315ef1c5213dbb8d61ea54b0c85344c1b0352640178b18a.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/c53a6d71f261e647f315ef1c5213dbb8d61ea54b0c85344c1b0352640178b18a.jpg)

![ddc80e037ea9865736613c983af6d6038676b5b6d3263af70e35640676382ca8.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/ddc80e037ea9865736613c983af6d6038676b5b6d3263af70e35640676382ca8.jpg)

![e7c893b8b4e991557867e6bb4640becf835a25ce968d571154921a7784a40d18.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/e7c893b8b4e991557867e6bb4640becf835a25ce968d571154921a7784a40d18.jpg)

![f1cd00ca0b6afe2cabc5ee7c6e32da6257792f2ba2beab98ccacc6f7862c9246.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/f1cd00ca0b6afe2cabc5ee7c6e32da6257792f2ba2beab98ccacc6f7862c9246.jpg)

![f73f358fbbe82b539032b0a1636ff7baea49920bffc692d4ae7d1e3b8e09b434.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/images/f73f358fbbe82b539032b0a1636ff7baea49920bffc692d4ae7d1e3b8e09b434.jpg)

### Tables

![077b8f25b05c3bdb30d885de28f8e0957eddf7bb4f90a88fd96fc73fac593445.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/077b8f25b05c3bdb30d885de28f8e0957eddf7bb4f90a88fd96fc73fac593445.jpg)

![1f67221e3fc1c93e83e926b1c5fc61e69f869dde4f5723bab01f065a3307b864.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/1f67221e3fc1c93e83e926b1c5fc61e69f869dde4f5723bab01f065a3307b864.jpg)

![3f74861295758f2dc50a7e472cf96ceb7857ba8cde1e6ef9972c66fca6c27ea8.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/3f74861295758f2dc50a7e472cf96ceb7857ba8cde1e6ef9972c66fca6c27ea8.jpg)

![41956070cd70fac401daccd3d9224b1fdaf9d60830ef8d35ec92126f6bbf4bca.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/41956070cd70fac401daccd3d9224b1fdaf9d60830ef8d35ec92126f6bbf4bca.jpg)

![44904ed11837e8f2afae3bccc451048708f48dc20e79721f1a9f916ef1736172.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/44904ed11837e8f2afae3bccc451048708f48dc20e79721f1a9f916ef1736172.jpg)

![655e7414ec318a9e799509a885608d5c52b6703632d71046cae1163c7aa5c29c.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/655e7414ec318a9e799509a885608d5c52b6703632d71046cae1163c7aa5c29c.jpg)

![68d7994b0b7a7e4f13bf5ad9d4665821d6520d177b691412d596a6428e6f35aa.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/68d7994b0b7a7e4f13bf5ad9d4665821d6520d177b691412d596a6428e6f35aa.jpg)

![91c674387a01b57672f30ba24a49785977c19865f5940a1e2b5dbfdfed832dcd.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/91c674387a01b57672f30ba24a49785977c19865f5940a1e2b5dbfdfed832dcd.jpg)

![9b674e5e27a02fd82d6186641a04ad79debb1c04eff9dff355f7625bae90b14e.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/9b674e5e27a02fd82d6186641a04ad79debb1c04eff9dff355f7625bae90b14e.jpg)

![a20f269ee3017dc2e4c364781a9687be2522c9f5da04b8a3a00f390697048901.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/a20f269ee3017dc2e4c364781a9687be2522c9f5da04b8a3a00f390697048901.jpg)

![b7395d249adc2f282703a83bd2a61efed7a56557c3a271b5fed4c65494337a33.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/b7395d249adc2f282703a83bd2a61efed7a56557c3a271b5fed4c65494337a33.jpg)

![c05c788c04f5ae46fc7e9b86bc6c248434b469a47b6d7bea972c237e7dc189ef.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/c05c788c04f5ae46fc7e9b86bc6c248434b469a47b6d7bea972c237e7dc189ef.jpg)

![c5cc1e7038685a7968e741985a36491e4edd97a2cc5f5b7967560ef7ec59a6c2.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/c5cc1e7038685a7968e741985a36491e4edd97a2cc5f5b7967560ef7ec59a6c2.jpg)

![c7c57d20f857b8b483412865109c551129a2b9ef307cbc8eeab15d473b8f9ee2.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/c7c57d20f857b8b483412865109c551129a2b9ef307cbc8eeab15d473b8f9ee2.jpg)

![cb0c82489a6a23b5352701717d5d5299e6be8f81441197317c72951b192a1b32.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/cb0c82489a6a23b5352701717d5d5299e6be8f81441197317c72951b192a1b32.jpg)

![ced392c9783221d692a18f8cf72dfb7e71729f694350e0d7cf176ee08f73821a.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/ced392c9783221d692a18f8cf72dfb7e71729f694350e0d7cf176ee08f73821a.jpg)

![e8bb053c9098175beccfc45bfa682bd591ede47dfffbef574ce5d11fdf557e34.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/e8bb053c9098175beccfc45bfa682bd591ede47dfffbef574ce5d11fdf557e34.jpg)

![e8ce1d2e272e175c94763d590e90dcfa3868f9ca3340b36676bf51a634004d77.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/e8ce1d2e272e175c94763d590e90dcfa3868f9ca3340b36676bf51a634004d77.jpg)

![f186a73e1a8f8b3ab749fa4df1bd25396eb280036ab0ff9910d9d647c0c60ac4.jpg](../icml_results/6329_LiveOIBench_%20Can%20Large%20Language%20Models%20Outperform%20Human%20Contestants%20in%20Informatics%20Olympiads_/tables/f186a73e1a8f8b3ab749fa4df1bd25396eb280036ab0ff9910d9d647c0c60ac4.jpg)

<a id="paper-6277"></a>

## AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning


### Images

![15956fdbac6461430e7c185d39a3255e47aae4b84091cb5af4b334020e59dee9.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/15956fdbac6461430e7c185d39a3255e47aae4b84091cb5af4b334020e59dee9.jpg)

![22052738ca9becb8a709535a8f8f2ad06fe30ae95ab9156f31d6f63832c7a963.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/22052738ca9becb8a709535a8f8f2ad06fe30ae95ab9156f31d6f63832c7a963.jpg)

![244febfd82fd07cb7071ac5ae0aa3663fd4bd8ece66a7e7815f6ebcc877200c8.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/244febfd82fd07cb7071ac5ae0aa3663fd4bd8ece66a7e7815f6ebcc877200c8.jpg)

![5c1ca727ef05d05b1d5dfe530d0c7a417048b9ecb746c4819121ee3468a55e1c.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/5c1ca727ef05d05b1d5dfe530d0c7a417048b9ecb746c4819121ee3468a55e1c.jpg)

![62782c6eaa2ceed9252f38ff217e7d6b857e895c1a31710d1ce5cab6d236aed7.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/62782c6eaa2ceed9252f38ff217e7d6b857e895c1a31710d1ce5cab6d236aed7.jpg)

![a0f28cd76659c7f59e8914ff5f6a3671112dc5999a605ff4678bc0411c321182.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/a0f28cd76659c7f59e8914ff5f6a3671112dc5999a605ff4678bc0411c321182.jpg)

![b115d0e3a7a197063e0e3ea179c4160a6b72b5016dc031b1857356c5bbac9bed.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/b115d0e3a7a197063e0e3ea179c4160a6b72b5016dc031b1857356c5bbac9bed.jpg)

![b79aa34daef36784739cf58433667def0d858857d7d043a71c434f6ef5f1f805.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/images/b79aa34daef36784739cf58433667def0d858857d7d043a71c434f6ef5f1f805.jpg)

### Tables

![127a5f0726efe0df0e42b72ad197bee1c6a8342a2727c3a1d9f9d62b0f12cd62.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/tables/127a5f0726efe0df0e42b72ad197bee1c6a8342a2727c3a1d9f9d62b0f12cd62.jpg)

![35731527f29620874131a91132a282d5482b28b762b814055d62203e6ea0464d.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/tables/35731527f29620874131a91132a282d5482b28b762b814055d62203e6ea0464d.jpg)

![490f0a2d8cb4ca729187b540c736d5b7fa14dbb5258e371be122a5b32a2e8425.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/tables/490f0a2d8cb4ca729187b540c736d5b7fa14dbb5258e371be122a5b32a2e8425.jpg)

![5bf912a2208e2bde9465b0bff9015be2188be42e61894681dbd7a3e151d7d084.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/tables/5bf912a2208e2bde9465b0bff9015be2188be42e61894681dbd7a3e151d7d084.jpg)

![8622ca8302ab2b85dde3e48fb48c04d7ee1eda305f894f3bf4c1d0e05a437924.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/tables/8622ca8302ab2b85dde3e48fb48c04d7ee1eda305f894f3bf4c1d0e05a437924.jpg)

![90426dd6234343e5c13dac04c8b2692c657a794911ce77a3cf7aa7c398cb1aa9.jpg](../icml_results/6330_AutoTool_%20Dynamic%20Tool%20Selection%20and%20Integration%20for%20Agentic%20Reasoning/tables/90426dd6234343e5c13dac04c8b2692c657a794911ce77a3cf7aa7c398cb1aa9.jpg)

<a id="paper-6278"></a>

## From Static Constraints to Dynamic Adaptation: Sample-Level Constraint Relaxation for Offline-to-Online Reinforcement Learning


### Images

![09e5c649e5591dae338370df985cb599dd9886044e8eed83ee24c5b91fa150e2.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/09e5c649e5591dae338370df985cb599dd9886044e8eed83ee24c5b91fa150e2.jpg)

![1323800a8d7cfafd55ed5c37552fef393a583b40581de42b61a1fdf9a797c97d.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/1323800a8d7cfafd55ed5c37552fef393a583b40581de42b61a1fdf9a797c97d.jpg)

![1fab384edac4c33903e64dd8562e35ced76e8f32bbea8ca883927bfcbd0157c7.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/1fab384edac4c33903e64dd8562e35ced76e8f32bbea8ca883927bfcbd0157c7.jpg)

![43cf316ed03c244778cc79980f2a4f1d87d5276245c500093e2576e17c5927bc.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/43cf316ed03c244778cc79980f2a4f1d87d5276245c500093e2576e17c5927bc.jpg)

![58dc5c5b2a11f5317bd019ae44ce913e6a1cdff6229102175e0738cad1bb3a44.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/58dc5c5b2a11f5317bd019ae44ce913e6a1cdff6229102175e0738cad1bb3a44.jpg)

![675cd014f7752e7e9fac8a90756387f546a2027c7f188b4b35f828645d09b059.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/675cd014f7752e7e9fac8a90756387f546a2027c7f188b4b35f828645d09b059.jpg)

![6e90b6f427b83f7dacc9f045652ee206e4456a301a0c8ec4d648008d3557a623.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/6e90b6f427b83f7dacc9f045652ee206e4456a301a0c8ec4d648008d3557a623.jpg)

![71713dc29aa13cf622fc3029bcf6fed29246787529b4d49819c3de75dd61fce3.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/71713dc29aa13cf622fc3029bcf6fed29246787529b4d49819c3de75dd61fce3.jpg)

![9bdcb3e14ee676d03aab8c43cd15de6a538c0bdb88fea2480e64a6f42b42d9ec.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/9bdcb3e14ee676d03aab8c43cd15de6a538c0bdb88fea2480e64a6f42b42d9ec.jpg)

![a4143e7da23b46f0f3095026c59fc7c59854d0f8fafbf41f030b835474708910.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/a4143e7da23b46f0f3095026c59fc7c59854d0f8fafbf41f030b835474708910.jpg)

![accee11b5c422d5b7292bef98a9ee481bf91679d0efbc5484992dcf331429603.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/accee11b5c422d5b7292bef98a9ee481bf91679d0efbc5484992dcf331429603.jpg)

![c2624e3622c255f1a3e5c00cebd318af4f95fc1a675f480acb0d16f0db064fce.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/c2624e3622c255f1a3e5c00cebd318af4f95fc1a675f480acb0d16f0db064fce.jpg)

![cbb20ac7699d88997d244220c94698af1061ee29b50d85fa2d75cc5ae36a2ad9.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/cbb20ac7699d88997d244220c94698af1061ee29b50d85fa2d75cc5ae36a2ad9.jpg)

![d5ba607c72e7ef18985996f40ebd0e7f86a7e16d68ba7ff3daf4a482a3a9dca2.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/d5ba607c72e7ef18985996f40ebd0e7f86a7e16d68ba7ff3daf4a482a3a9dca2.jpg)

![eaa82770503f4662a28a14296f3d1a17feabd19708c7c38a311d844cf87eddf9.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/eaa82770503f4662a28a14296f3d1a17feabd19708c7c38a311d844cf87eddf9.jpg)

![fc530a1919366754b2141d6243dcfc9d169c94f32b7d7263d419b0aa2d294f45.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/images/fc530a1919366754b2141d6243dcfc9d169c94f32b7d7263d419b0aa2d294f45.jpg)

### Tables

![06df9457f29cd72bb9801596a6c3f3b34e06c6890c1eb6e2373eab22364f7ffb.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/06df9457f29cd72bb9801596a6c3f3b34e06c6890c1eb6e2373eab22364f7ffb.jpg)

![1af725f7dd464b1163327e53047121ec225289ccd2bee34d7580bc0f2bbed882.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/1af725f7dd464b1163327e53047121ec225289ccd2bee34d7580bc0f2bbed882.jpg)

![279a9b3fdf344afaeca516209c147dd8a589a2c09486e8f0491f5243573edcc2.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/279a9b3fdf344afaeca516209c147dd8a589a2c09486e8f0491f5243573edcc2.jpg)

![3c02e881020457951dca17772ed99f8f87b6143e7cc7cb2d81d8e146427dce57.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/3c02e881020457951dca17772ed99f8f87b6143e7cc7cb2d81d8e146427dce57.jpg)

![47dcd22d280a9bec99f8fb1aba956bba100b48e3177eb35d79d10834665d04c3.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/47dcd22d280a9bec99f8fb1aba956bba100b48e3177eb35d79d10834665d04c3.jpg)

![483478a4b8cddaf967cb8ba4389334dec61eb6de6f246e04fc4311cc500555c0.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/483478a4b8cddaf967cb8ba4389334dec61eb6de6f246e04fc4311cc500555c0.jpg)

![48767987e7ee030c98c946d5596b4615bf3a36eac64d6fe409a8f3a6458006c1.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/48767987e7ee030c98c946d5596b4615bf3a36eac64d6fe409a8f3a6458006c1.jpg)

![6e4bf9d0460e5716f13d02986f18f6ae6a71e6da73030bf45a7ef1cd4a4dc696.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/6e4bf9d0460e5716f13d02986f18f6ae6a71e6da73030bf45a7ef1cd4a4dc696.jpg)

![8931217c34cbcbd9e9612e466f3789146443857bf6eb5f45029c402c4c20b82e.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/8931217c34cbcbd9e9612e466f3789146443857bf6eb5f45029c402c4c20b82e.jpg)

![9f4f611ff728c6dc119a833b23f63215c2c451fcc3820572a7abdfe53de63af9.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/9f4f611ff728c6dc119a833b23f63215c2c451fcc3820572a7abdfe53de63af9.jpg)

![a6ccd3475b903304ce7ed1fa6dedeaf19b80ed76bd5c71fd9501e7f7d171391a.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/a6ccd3475b903304ce7ed1fa6dedeaf19b80ed76bd5c71fd9501e7f7d171391a.jpg)

![cc330184d1ce7357f893ecd4d3ff76d494d158e5804e80df73b165572171afa6.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/cc330184d1ce7357f893ecd4d3ff76d494d158e5804e80df73b165572171afa6.jpg)

![d1601258f81a94e479d282988f12363c8d0a25e43ad8ca58db79cd6702d67bfe.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/d1601258f81a94e479d282988f12363c8d0a25e43ad8ca58db79cd6702d67bfe.jpg)

![e608f5eb0a47cae3e2cbfd552f9e345b08a8d7f38758d5da710ee8d45608aefb.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/e608f5eb0a47cae3e2cbfd552f9e345b08a8d7f38758d5da710ee8d45608aefb.jpg)

![e99a883c6aed91942d2c3ba1ad33b27a193667a766f0253dde11f647c0a4b0f3.jpg](../icml_results/6331_From%20Static%20Constraints%20to%20Dynamic%20Adaptation_%20Sample-Level%20Constraint%20Relaxation%20for%20Offline-to-Onl/tables/e99a883c6aed91942d2c3ba1ad33b27a193667a766f0253dde11f647c0a4b0f3.jpg)

<a id="paper-6279"></a>

## Predicting the Order of Upcoming Tokens Improves Language Modeling


### Images

![0f7be77c8e52b629653533e095a4ef84a21417db4c3c08c5d54177103b68ef6d.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/0f7be77c8e52b629653533e095a4ef84a21417db4c3c08c5d54177103b68ef6d.jpg)

![34d5c60ac60cb7122bfec47ea73ff81fa321033c231bff0d4b9d1fef106727b9.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/34d5c60ac60cb7122bfec47ea73ff81fa321033c231bff0d4b9d1fef106727b9.jpg)

![5576e52d54b6462b994fc44945f3947c4f9232c33a4e0f1b0f4394eec55f4444.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/5576e52d54b6462b994fc44945f3947c4f9232c33a4e0f1b0f4394eec55f4444.jpg)

![56b2220dd3163c264a916f912b7eafaee4083ff3a04d5367d6b289310aae6427.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/56b2220dd3163c264a916f912b7eafaee4083ff3a04d5367d6b289310aae6427.jpg)

![6770a2ad5ded7d8e76b96d104bd39aa4e47a4edab46d7a97c43f26504b87cc32.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/6770a2ad5ded7d8e76b96d104bd39aa4e47a4edab46d7a97c43f26504b87cc32.jpg)

![76e9f44b6e398e3103346e6b819cdbacf3a1471d8fd8c60a81ecbbf3728d4b8b.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/76e9f44b6e398e3103346e6b819cdbacf3a1471d8fd8c60a81ecbbf3728d4b8b.jpg)

![96f3bffd3d2271ce501ce2b6708cc70dfb2f91a714e2831aeeb427c118c72f4b.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/96f3bffd3d2271ce501ce2b6708cc70dfb2f91a714e2831aeeb427c118c72f4b.jpg)

![a00dc04b7d9dc3e3d27056f389361cb373724410e6fc05c9bc8e53bf502c9768.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/a00dc04b7d9dc3e3d27056f389361cb373724410e6fc05c9bc8e53bf502c9768.jpg)

![ae9c74237066da9a1a4d415ea14f43c7ee52da22781774f6b0999ba7642f897b.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/ae9c74237066da9a1a4d415ea14f43c7ee52da22781774f6b0999ba7642f897b.jpg)

![b97cef9a947a689297a9aade51dedc09cc12674eac9e1202ad3e125b977896d1.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/b97cef9a947a689297a9aade51dedc09cc12674eac9e1202ad3e125b977896d1.jpg)

![f8aab39bd9fd32ba71338d76700a9e8e1b5b476ecd130b54a352e94bfced6bbd.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/images/f8aab39bd9fd32ba71338d76700a9e8e1b5b476ecd130b54a352e94bfced6bbd.jpg)

### Tables

![2d7c89baa8bf30d7b08c1b7ccd4fecd04ed95492a194e77de62824cb16817624.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/2d7c89baa8bf30d7b08c1b7ccd4fecd04ed95492a194e77de62824cb16817624.jpg)

![6e955b00907d77d7e27eb733eb1d23b1a0615bc2a5c45a890429174ce3be1354.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/6e955b00907d77d7e27eb733eb1d23b1a0615bc2a5c45a890429174ce3be1354.jpg)

![8457b68bfb312572b6b81273a4f3930481656edee80599c2021402894a6c0591.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/8457b68bfb312572b6b81273a4f3930481656edee80599c2021402894a6c0591.jpg)

![866f5b6202748f2a06372674ba66fbb62c39c59464f62dc48ac43e6873736e7f.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/866f5b6202748f2a06372674ba66fbb62c39c59464f62dc48ac43e6873736e7f.jpg)

![93e925cad44c9507bee2906f061b5815cffa2e82a2509da0d668e40eda9644f5.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/93e925cad44c9507bee2906f061b5815cffa2e82a2509da0d668e40eda9644f5.jpg)

![9fd4310f825e080c73960273e99471874c6fce3fa9972b62687004ad29819870.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/9fd4310f825e080c73960273e99471874c6fce3fa9972b62687004ad29819870.jpg)

![a11c995633a2452ef153108a70bc80f0d780957ab75ce09bd29b86197345dce0.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/a11c995633a2452ef153108a70bc80f0d780957ab75ce09bd29b86197345dce0.jpg)

![dca24e0e9b950aa3c8f1b105ed3fcc1873fa92e51e15739d8fce66d5d94cb43e.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/dca24e0e9b950aa3c8f1b105ed3fcc1873fa92e51e15739d8fce66d5d94cb43e.jpg)

![e7707a940625e22ff4773e1c9970d3a3d3909ce213281632ff515bfec2ba0341.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/e7707a940625e22ff4773e1c9970d3a3d3909ce213281632ff515bfec2ba0341.jpg)

![f20d0e30f07046230059846e5e019d144d6448bc17c12cfacb915a83676aaacb.jpg](../icml_results/6332_Predicting%20the%20Order%20of%20Upcoming%20Tokens%20Improves%20Language%20Modeling/tables/f20d0e30f07046230059846e5e019d144d6448bc17c12cfacb915a83676aaacb.jpg)

<a id="paper-6280"></a>

## From Representation to Action: A Unified Laplacian Framework for Spatial Representation and Path Planning


### Images

![353ab39092511449890aa4d28b6aa553db9be5f753f848e4bb4d3970566b7afa.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/images/353ab39092511449890aa4d28b6aa553db9be5f753f848e4bb4d3970566b7afa.jpg)

![3ae6906b20cd8946774ca009ae1e7729a68b4ac417cf2c78fc329ec544d471cd.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/images/3ae6906b20cd8946774ca009ae1e7729a68b4ac417cf2c78fc329ec544d471cd.jpg)

![484e77b499ce6b78c412fef049f6bf20acf4a44c709ef6753ee82d47845322a4.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/images/484e77b499ce6b78c412fef049f6bf20acf4a44c709ef6753ee82d47845322a4.jpg)

![ba34312c9d9a5e5dd62d231db629454915f586e7b71ccf13531db8f9a0cddcb4.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/images/ba34312c9d9a5e5dd62d231db629454915f586e7b71ccf13531db8f9a0cddcb4.jpg)

![c59366067fc2f5343e0932d1b8a86b4cbc77d87bf1652f687687f0f63cce5cce.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/images/c59366067fc2f5343e0932d1b8a86b4cbc77d87bf1652f687687f0f63cce5cce.jpg)

![eda2a416c5803e100bae16a6ececa30b5760e337dc04de00c7b518071cb68c8b.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/images/eda2a416c5803e100bae16a6ececa30b5760e337dc04de00c7b518071cb68c8b.jpg)

### Tables

![01ea1cc5b8e23705d2e59bcc9981af0d43b1aefce8dfa396498b52a0547cc395.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/tables/01ea1cc5b8e23705d2e59bcc9981af0d43b1aefce8dfa396498b52a0547cc395.jpg)

![832e68b7682fd1e359c8581578f33d7a425fd631ab60a519bb0864dc730a2710.jpg](../icml_results/6333_From%20Representation%20to%20Action_%20A%20Unified%20Laplacian%20Framework%20for%20Spatial%20Representation%20and%20Path%20Pla/tables/832e68b7682fd1e359c8581578f33d7a425fd631ab60a519bb0864dc730a2710.jpg)

<a id="paper-6281"></a>

## IVQ: Structured and Lightweight Vector Quantization via Binary Hierarchical Composition Inspired by $\textitIChing$


### Images

![19495e572403ec90e4b5745c253d559e36025ee73216f9d871a94cc02910619f.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/19495e572403ec90e4b5745c253d559e36025ee73216f9d871a94cc02910619f.jpg)

![23e95438f0c4f8a8b9070f031494bb5ee5d57799b298bbc01ca51b8ecb66b6fd.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/23e95438f0c4f8a8b9070f031494bb5ee5d57799b298bbc01ca51b8ecb66b6fd.jpg)

![84a2d9706da898d5f8039a9b63f296fa34233f695bcd928e940d4e089b07cc1d.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/84a2d9706da898d5f8039a9b63f296fa34233f695bcd928e940d4e089b07cc1d.jpg)

![87360bca210eacc38acb3abc833b5e028df36823937ae307399208ecfc4a6d17.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/87360bca210eacc38acb3abc833b5e028df36823937ae307399208ecfc4a6d17.jpg)

![936373aef52ce43e984948c3fb51eeb2776113e8f98a6e990d8b424793684322.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/936373aef52ce43e984948c3fb51eeb2776113e8f98a6e990d8b424793684322.jpg)

![eab4933dec41a838b36c8fe94f6fffd240c5df919ccfa7f64c9f934f57cbc422.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/eab4933dec41a838b36c8fe94f6fffd240c5df919ccfa7f64c9f934f57cbc422.jpg)

![f7e186a4079bea32a297833e8ab458ace462fad9a6afede231e23f7847e70335.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/images/f7e186a4079bea32a297833e8ab458ace462fad9a6afede231e23f7847e70335.jpg)

### Tables

![2a34448b6a5fdcec5200bc2e562e9e652faa4801a578d89c184b594ce1045245.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/2a34448b6a5fdcec5200bc2e562e9e652faa4801a578d89c184b594ce1045245.jpg)

![68b6d06b1654a54dbcf0d040aa8784099322fc3c74ea31072eb8d7e2e5ec4e85.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/68b6d06b1654a54dbcf0d040aa8784099322fc3c74ea31072eb8d7e2e5ec4e85.jpg)

![99f8b164e922f401ec0721db8c439df662b3e4043e6b0f055fa5bafb065fd19d.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/99f8b164e922f401ec0721db8c439df662b3e4043e6b0f055fa5bafb065fd19d.jpg)

![bf5bacaa9c7980dd54b03b42e789c332fd499f34cf939cdafd1702bf3b2ca90a.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/bf5bacaa9c7980dd54b03b42e789c332fd499f34cf939cdafd1702bf3b2ca90a.jpg)

![c2f74ac45cb694e9510c08a33d88e115334fde762d2824beeddaa5d86eddb214.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/c2f74ac45cb694e9510c08a33d88e115334fde762d2824beeddaa5d86eddb214.jpg)

![c80eb643b7105b77a50c369cce8af97a4e04a1563f9b2433e2b2d3b0a929c641.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/c80eb643b7105b77a50c369cce8af97a4e04a1563f9b2433e2b2d3b0a929c641.jpg)

![d291656d6184a54f98b0dd8dbba795788755401842a594cc22e9f779cd505ba4.jpg](../icml_results/6334_IVQ_%20Structured%20and%20Lightweight%20Vector%20Quantization%20via%20Binary%20Hierarchical%20Composition%20Inspired%20by%20/tables/d291656d6184a54f98b0dd8dbba795788755401842a594cc22e9f779cd505ba4.jpg)

<a id="paper-6282"></a>

## Enhancing Numerical Prediction in LLMs via Smooth MMD Alignment


### Images

![368cf26b068f4e0fbddeda2d0a36fe862e5885586b4dbedda146ee1d7c4bb81b.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/368cf26b068f4e0fbddeda2d0a36fe862e5885586b4dbedda146ee1d7c4bb81b.jpg)

![4ff6f371e63da19fa87b49fb680d74747c567a52b7f8e2a963bbadddcfeca849.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/4ff6f371e63da19fa87b49fb680d74747c567a52b7f8e2a963bbadddcfeca849.jpg)

![6950484748cdb00d0eff2f20e2e8f8941866bfa2c2f8128ac4fc0895d398d854.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/6950484748cdb00d0eff2f20e2e8f8941866bfa2c2f8128ac4fc0895d398d854.jpg)

![8fd22d8559b26d6a0c7fb34504b926e2667f2d9790b45131874b477f6ff05393.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/8fd22d8559b26d6a0c7fb34504b926e2667f2d9790b45131874b477f6ff05393.jpg)

![9da7787f3950299aab07d211f4e7c1e9d8d84e9d549d1ce72935ec5179947389.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/9da7787f3950299aab07d211f4e7c1e9d8d84e9d549d1ce72935ec5179947389.jpg)

![b478c7df2227f12928190e6e408ad270ded0381de98f5d478e65e2dc609d13cc.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/b478c7df2227f12928190e6e408ad270ded0381de98f5d478e65e2dc609d13cc.jpg)

![c4a2e5cec1ae51d203439ee880e27f7ebf9e56fe6d2ed67924dbf903870ba66d.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/c4a2e5cec1ae51d203439ee880e27f7ebf9e56fe6d2ed67924dbf903870ba66d.jpg)

![dae8b43947ae254a89b953e2a27d9ea36710e5ce13dae671cf27c946a2320c5d.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/dae8b43947ae254a89b953e2a27d9ea36710e5ce13dae671cf27c946a2320c5d.jpg)

![fc5ada376dad38480717dab80793c7080ac8141b8e9f7902909722f73a17e6f4.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/images/fc5ada376dad38480717dab80793c7080ac8141b8e9f7902909722f73a17e6f4.jpg)

### Tables

![025413849d41a48e33a84ae94e63a1b1d1086f4b15e3ebab1eee83d41bdb1808.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/025413849d41a48e33a84ae94e63a1b1d1086f4b15e3ebab1eee83d41bdb1808.jpg)

![1337480721b0a915445f7f90b2a7672af9e0bd57f6f50120082a4a47aafd2114.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/1337480721b0a915445f7f90b2a7672af9e0bd57f6f50120082a4a47aafd2114.jpg)

![3dd0ce7f71610822584e418631a47c7c6bff62bed2f07007a6a9f234b5b78cab.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/3dd0ce7f71610822584e418631a47c7c6bff62bed2f07007a6a9f234b5b78cab.jpg)

![4c08e64670f9f2a77c7ccd3badb62e1c887fdd00b1c61c8c69becbda82fb51f0.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/4c08e64670f9f2a77c7ccd3badb62e1c887fdd00b1c61c8c69becbda82fb51f0.jpg)

![67f72c34bcc615508ed581110b0333cbfae8c52f4b34fbdd0acc5096be15eb2a.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/67f72c34bcc615508ed581110b0333cbfae8c52f4b34fbdd0acc5096be15eb2a.jpg)

![7b2dc7afcdf095593b9f14554c77c36878c096fc14bb7537a681abce16846b75.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/7b2dc7afcdf095593b9f14554c77c36878c096fc14bb7537a681abce16846b75.jpg)

![9d4557434d54d585c025457b7f57c074efdd518af64f3b661201db00a58e240f.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/9d4557434d54d585c025457b7f57c074efdd518af64f3b661201db00a58e240f.jpg)

![a0b37d1cb0dda2c4f6405c3f7aa4526ae9cebccee6b9f35dc1717d65a89f5958.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/a0b37d1cb0dda2c4f6405c3f7aa4526ae9cebccee6b9f35dc1717d65a89f5958.jpg)

![b1661fb549f0d1190d231d8f1d131944cdb0e02a82390ce3ac36c5b5a22a9686.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/b1661fb549f0d1190d231d8f1d131944cdb0e02a82390ce3ac36c5b5a22a9686.jpg)

![c0b4d7f5f2cd1082dec33ef95d7eecd0d8baf177470d575b84394b9f6b5ad58e.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/c0b4d7f5f2cd1082dec33ef95d7eecd0d8baf177470d575b84394b9f6b5ad58e.jpg)

![cf1d521ca619bb9daef7c50288d3c948b54025f337f51b59c9f2aa22c30e3f2a.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/cf1d521ca619bb9daef7c50288d3c948b54025f337f51b59c9f2aa22c30e3f2a.jpg)

![d2080f4c9786b664ecb0ee0c9bd8fce71ea64195943be3f2630feee4e5625692.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/d2080f4c9786b664ecb0ee0c9bd8fce71ea64195943be3f2630feee4e5625692.jpg)

![dc27b449abacfc6f5169210bb77ddcb8821784220e24aa87576a42dff57dafec.jpg](../icml_results/6335_Enhancing%20Numerical%20Prediction%20in%20LLMs%20via%20Smooth%20MMD%20Alignment/tables/dc27b449abacfc6f5169210bb77ddcb8821784220e24aa87576a42dff57dafec.jpg)

<a id="paper-6283"></a>

## GLAD: Bidirectional Structure-Attribute Alignment via Latent Graph Diffusion Models


### Images

![33a394cb0da3bee0a9334b893df0cdc948d02b77c6592e824e26869382e716a3.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/33a394cb0da3bee0a9334b893df0cdc948d02b77c6592e824e26869382e716a3.jpg)

![387b07f897a9f2a3d3c9445fb8672133dbad3dd8d9154eee8abdee2629fe54cc.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/387b07f897a9f2a3d3c9445fb8672133dbad3dd8d9154eee8abdee2629fe54cc.jpg)

![46c4e3bf057b22aa59a5a499201161b2428542aefc4812f6fb423aa4d5184f59.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/46c4e3bf057b22aa59a5a499201161b2428542aefc4812f6fb423aa4d5184f59.jpg)

![49468d5ab0cd2cd015855bf86d908b8f0608c54b67b90d8cc2b01578162fb9ea.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/49468d5ab0cd2cd015855bf86d908b8f0608c54b67b90d8cc2b01578162fb9ea.jpg)

![4aea4ec05e226265be813ed7073db07d961497c53032b464bf72aaf42b66f19b.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/4aea4ec05e226265be813ed7073db07d961497c53032b464bf72aaf42b66f19b.jpg)

![75aa93b8ffdc122dbb29ee40316681facdd9b88712250f4b3b3840530e062976.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/75aa93b8ffdc122dbb29ee40316681facdd9b88712250f4b3b3840530e062976.jpg)

![8cf61162a931724f49df53cf8b0a773066369527afc0696f532c835060d89dbb.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/8cf61162a931724f49df53cf8b0a773066369527afc0696f532c835060d89dbb.jpg)

![b743292ad97e9836ea132645be9b4ffbaf1bb0526a9e879e6cce3593cf9daa6f.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/b743292ad97e9836ea132645be9b4ffbaf1bb0526a9e879e6cce3593cf9daa6f.jpg)

![cb55512c93b8ad728969072e3e0478851bfb0f52036711a8a38186ad48394852.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/cb55512c93b8ad728969072e3e0478851bfb0f52036711a8a38186ad48394852.jpg)

![e65d902d764590016ab989875cbed879c0c560e8675770cbafdefaf7da971f0b.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/e65d902d764590016ab989875cbed879c0c560e8675770cbafdefaf7da971f0b.jpg)

![e8ea182cddfa90f32af3c4c66716ecbeb812d3da34bff258e7d127523fa963c6.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/e8ea182cddfa90f32af3c4c66716ecbeb812d3da34bff258e7d127523fa963c6.jpg)

![f274f4dc5d7a71101feb1d90f2b1f45d33d2fde29c7cdb362734bcadee221449.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/images/f274f4dc5d7a71101feb1d90f2b1f45d33d2fde29c7cdb362734bcadee221449.jpg)

### Tables

![0e187bd57eab02bb5560ee3ea20c037b767ad82940c3e3b0f06a5e35439f1b62.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/0e187bd57eab02bb5560ee3ea20c037b767ad82940c3e3b0f06a5e35439f1b62.jpg)

![22c0bf4be22b5f3bdcd2631018fc8bb6f6d1023d9182be3dabeee32dd81e7697.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/22c0bf4be22b5f3bdcd2631018fc8bb6f6d1023d9182be3dabeee32dd81e7697.jpg)

![639001b606512ebb1015062581c1eb5a6d2f116b1708495c7b35c2975298c7dd.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/639001b606512ebb1015062581c1eb5a6d2f116b1708495c7b35c2975298c7dd.jpg)

![8c9c3513b93046298ee5c7d57a4ba9a0bf113c71b8c18e3486f130b0f9cf5435.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/8c9c3513b93046298ee5c7d57a4ba9a0bf113c71b8c18e3486f130b0f9cf5435.jpg)

![8cc52a0d94ed26e4a00e28c09c86d0e04360b98ce27e34711b0d939d6838f507.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/8cc52a0d94ed26e4a00e28c09c86d0e04360b98ce27e34711b0d939d6838f507.jpg)

![a67cac1d0679019d22fb4f9b608aaf799b28353fb30455e4c95c246922ba2d28.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/a67cac1d0679019d22fb4f9b608aaf799b28353fb30455e4c95c246922ba2d28.jpg)

![acdc0f42dbe2d502c412feb8bd794b2b323b5b29637bab8c5417eb9e8517c566.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/acdc0f42dbe2d502c412feb8bd794b2b323b5b29637bab8c5417eb9e8517c566.jpg)

![c036d8fe234efcdd4a9fe92065b1df5d15c2ad1959ab5e4704d1fe5da5009a66.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/c036d8fe234efcdd4a9fe92065b1df5d15c2ad1959ab5e4704d1fe5da5009a66.jpg)

![dd5df0b89cadcea75cb9873c2f5f580b35e5c88e0cb567c25dde33bb66174e3b.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/dd5df0b89cadcea75cb9873c2f5f580b35e5c88e0cb567c25dde33bb66174e3b.jpg)

![deef84df6c7f0c72d96cfab2bc492cef291b265ee1c44601a58b36527b4ce762.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/deef84df6c7f0c72d96cfab2bc492cef291b265ee1c44601a58b36527b4ce762.jpg)

![fcd4fe328dc3e1fb1eaf3d7491b2b9575488cf073d3ca9cd35f0cb9ab7edc6f6.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/fcd4fe328dc3e1fb1eaf3d7491b2b9575488cf073d3ca9cd35f0cb9ab7edc6f6.jpg)

![fd428bd27233b448dfcb98f3ed3a6bad645a2df615f09c2184d00d4492bef01c.jpg](../icml_results/6336_GLAD_%20Bidirectional%20Structure-Attribute%20Alignment%20via%20Latent%20Graph%20Diffusion%20Models/tables/fd428bd27233b448dfcb98f3ed3a6bad645a2df615f09c2184d00d4492bef01c.jpg)

<a id="paper-6284"></a>

## From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates


### Images

![10b1fddc4366d75d990bca7f808376ba9fb881fe5dfe3e2d641f0223430d070e.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/10b1fddc4366d75d990bca7f808376ba9fb881fe5dfe3e2d641f0223430d070e.jpg)

![209515430e831492faf21a2f2deaace6f98c79df270012545d8f67be98c019de.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/209515430e831492faf21a2f2deaace6f98c79df270012545d8f67be98c019de.jpg)

![2ddbdddcfe03c0b2e7f02deb45d0e73bd284f1247ad1211f2c1d45b4c9eb2fa2.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/2ddbdddcfe03c0b2e7f02deb45d0e73bd284f1247ad1211f2c1d45b4c9eb2fa2.jpg)

![3597323eff67de396fa0f7d5a6136862a5fe90b1169ba797c51bf2db57185b06.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/3597323eff67de396fa0f7d5a6136862a5fe90b1169ba797c51bf2db57185b06.jpg)

![3c91ae28c959d5ee839a73f8790ec9e02d11e4709b92d325f7ca10f6bcd51ce4.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/3c91ae28c959d5ee839a73f8790ec9e02d11e4709b92d325f7ca10f6bcd51ce4.jpg)

![61f185a866e44f4ab25038f24aa675316104f07572f1db4e8884f99dee518c65.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/61f185a866e44f4ab25038f24aa675316104f07572f1db4e8884f99dee518c65.jpg)

![7c08fbcea266e7fa0c64b0fa80ef3389a79c76368f90f8686229ff7c905eb936.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/7c08fbcea266e7fa0c64b0fa80ef3389a79c76368f90f8686229ff7c905eb936.jpg)

![7f3981ee9d8e49d41e1ac4ad72a2180a86e0c5875a587c30350155a2efa22284.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/7f3981ee9d8e49d41e1ac4ad72a2180a86e0c5875a587c30350155a2efa22284.jpg)

![7fbb2f9c506ae8b1ee0724a6c5596e242b63d828075b743370f336615028fd6d.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/7fbb2f9c506ae8b1ee0724a6c5596e242b63d828075b743370f336615028fd6d.jpg)

![992a9b639af80dc56f4e4ea467363b4d7bced80895da45d7122b3ad8527a8437.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/992a9b639af80dc56f4e4ea467363b4d7bced80895da45d7122b3ad8527a8437.jpg)

![a44b1e7d9e2db0cce8a6a6a2a789a22174467e21bfd089200cb5ca23cd6b5651.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/a44b1e7d9e2db0cce8a6a6a2a789a22174467e21bfd089200cb5ca23cd6b5651.jpg)

![bcd29fe9909f6695737c8f10a9ada5f628f160264c44e346886b5b5da800aa36.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/bcd29fe9909f6695737c8f10a9ada5f628f160264c44e346886b5b5da800aa36.jpg)

![cf531779c5b222b63ba1c4e92e525b54d3c57b2630c511c11530f17c1e3bc473.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/cf531779c5b222b63ba1c4e92e525b54d3c57b2630c511c11530f17c1e3bc473.jpg)

![fa1e98856bfe671700f5caf36d1ca58ea6f5fbfe32406518b8ddbf42cd0330c4.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/fa1e98856bfe671700f5caf36d1ca58ea6f5fbfe32406518b8ddbf42cd0330c4.jpg)

![fd823dc9a61bb7ec4e08a78252cc0e771fdf05bd69ed16167e78df1c74cf0c4d.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/images/fd823dc9a61bb7ec4e08a78252cc0e771fdf05bd69ed16167e78df1c74cf0c4d.jpg)

### Tables

![09198b1de9963b28209655b624345f3e6057bf318b6e15412f67a5832007aadc.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/09198b1de9963b28209655b624345f3e6057bf318b6e15412f67a5832007aadc.jpg)

![7b22d899bf4415a44f6ad6d7c55c0293e89f4aebd04c2ce0d69f801aac7b6257.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/7b22d899bf4415a44f6ad6d7c55c0293e89f4aebd04c2ce0d69f801aac7b6257.jpg)

![8ee6e73b0fd4c97716ee1ff0735b048ee7b65bbe6e9658be5dcac2bbcde43593.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/8ee6e73b0fd4c97716ee1ff0735b048ee7b65bbe6e9658be5dcac2bbcde43593.jpg)

![90512a8f1b78041f8f78e034f4836890e327d536f531ac5bc6ca46f08e0feaa5.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/90512a8f1b78041f8f78e034f4836890e327d536f531ac5bc6ca46f08e0feaa5.jpg)

![b1fb17945cb011b9e6b54366f9f3364c517411c9e2c8f594d1eeff35ceab7da5.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/b1fb17945cb011b9e6b54366f9f3364c517411c9e2c8f594d1eeff35ceab7da5.jpg)

![cce87aa7c4ec5ab850e0eaa44058d307ff4856800ac04b6aff51dac06461c075.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/cce87aa7c4ec5ab850e0eaa44058d307ff4856800ac04b6aff51dac06461c075.jpg)

![df25d8ad7d652c50919909c7abc19aa7802a13f2208ca695c4fc4710bab9bd71.jpg](../icml_results/6337_From%20LLM-Generated%20Conjectures%20to%20Lean%20Formalizations_%20Automated%20Polynomial%20Inequality%20Proving%20via%20S/tables/df25d8ad7d652c50919909c7abc19aa7802a13f2208ca695c4fc4710bab9bd71.jpg)

<a id="paper-6285"></a>

## AnyBand-Diff: A Unified Remote Sensing Image Generation and Band Repair Framework with Spectral Priors


### Images

![0cde110f42e6e8db9c00f352deab8954b5d50b01bffe45746a50edb5ac274a86.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/images/0cde110f42e6e8db9c00f352deab8954b5d50b01bffe45746a50edb5ac274a86.jpg)

![14498e22b9c73efb2163804d08bc8388ec969cdf8e838918523a09a9ad59e4d0.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/images/14498e22b9c73efb2163804d08bc8388ec969cdf8e838918523a09a9ad59e4d0.jpg)

![356876b509ec20fd778edb2eac61af4d985440954c6821c206c028ac36fe2d51.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/images/356876b509ec20fd778edb2eac61af4d985440954c6821c206c028ac36fe2d51.jpg)

![62c6c2b29f2dee4529baab9192843cef7a358492b0a80838479b7586e2ef299c.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/images/62c6c2b29f2dee4529baab9192843cef7a358492b0a80838479b7586e2ef299c.jpg)

### Tables

![32744db9bb36c81f97d88fb1f589cc3fc1dbeae1e1b15e8cd0218f934ff8a4c2.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/32744db9bb36c81f97d88fb1f589cc3fc1dbeae1e1b15e8cd0218f934ff8a4c2.jpg)

![39a39d14ab9356befed63c85318c53f0cabdbc37a5552c2aea7af72ef7b643de.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/39a39d14ab9356befed63c85318c53f0cabdbc37a5552c2aea7af72ef7b643de.jpg)

![618dfd0a840a4822aaac40a4af772d0a129493e98093fe027d31bdf7a4317966.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/618dfd0a840a4822aaac40a4af772d0a129493e98093fe027d31bdf7a4317966.jpg)

![8f441e8acd70730ec2bf008bb1bc7aea08055e5a9a626e78100aac22945551fd.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/8f441e8acd70730ec2bf008bb1bc7aea08055e5a9a626e78100aac22945551fd.jpg)

![b89a3d5f6ea8d8bf06573dce5d431670795316ecb11e19a8e3266f21baed1844.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/b89a3d5f6ea8d8bf06573dce5d431670795316ecb11e19a8e3266f21baed1844.jpg)

![d6d1764f357f3ab52be527aaab06352c5fa9ca57ad4edba1c386213c50e9d634.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/d6d1764f357f3ab52be527aaab06352c5fa9ca57ad4edba1c386213c50e9d634.jpg)

![f409b8883588d0fac7f34f1a1c71bc12d167a0212c4e4821ab60d8f9a50d56c8.jpg](../icml_results/6338_AnyBand-Diff_%20A%20Unified%20Remote%20Sensing%20Image%20Generation%20and%20Band%20Repair%20Framework%20with%20Spectral%20Prio/tables/f409b8883588d0fac7f34f1a1c71bc12d167a0212c4e4821ab60d8f9a50d56c8.jpg)

<a id="paper-6286"></a>

## A Random Matrix Theory of Masked Self-Supervised Learning


### Images

![42421397b98b495d97293b276b52cfb26fca7636be0ceb75eff6b8b5c7fd57d8.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/42421397b98b495d97293b276b52cfb26fca7636be0ceb75eff6b8b5c7fd57d8.jpg)

![7b86d8388db3b675be71f4495e13b1c16696548719f983da037d315419574552.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/7b86d8388db3b675be71f4495e13b1c16696548719f983da037d315419574552.jpg)

![8236496c50b9aa3c02216fed5ef79d2290ca2bd0d237a776bfd86b64830c4655.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/8236496c50b9aa3c02216fed5ef79d2290ca2bd0d237a776bfd86b64830c4655.jpg)

![8b713b7bf8e7c187807a93fa3e897006ff2141d26e68cc42331e5ac114d21667.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/8b713b7bf8e7c187807a93fa3e897006ff2141d26e68cc42331e5ac114d21667.jpg)

![c07f8e283b70f69f8a0adddeac52a94beddb78f6fb20e0a5f222cfa66e41f010.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/c07f8e283b70f69f8a0adddeac52a94beddb78f6fb20e0a5f222cfa66e41f010.jpg)

![c30a60b491cf43451af6a1e49c79a3465092c41378f0c3cfb084b03e5c3f5c81.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/c30a60b491cf43451af6a1e49c79a3465092c41378f0c3cfb084b03e5c3f5c81.jpg)

![d77213ffc7fb325989c216551830062379235c4296fcddcbd3781bafd04f9ec9.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/d77213ffc7fb325989c216551830062379235c4296fcddcbd3781bafd04f9ec9.jpg)

![e084868651eeaf9af20a119a1f2405f1caf2ff2b6e9e13c3fe3c91f7543b3d56.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/e084868651eeaf9af20a119a1f2405f1caf2ff2b6e9e13c3fe3c91f7543b3d56.jpg)

![fb6286242bf98f4cd45e558d72d45430461281ed76aee5f4008e2f98bcf72c5e.jpg](../icml_results/6339_A%20Random%20Matrix%20Theory%20of%20Masked%20Self-Supervised%20Learning/images/fb6286242bf98f4cd45e558d72d45430461281ed76aee5f4008e2f98bcf72c5e.jpg)

<a id="paper-6287"></a>

## Fast KV Compaction via Attention Matching


### Images

![18da2d1258154b860305a72476e1c7a237c41416c550f6a5499b5160eb83b9f0.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/18da2d1258154b860305a72476e1c7a237c41416c550f6a5499b5160eb83b9f0.jpg)

![19aaaffec5ef24fd763c5fa62f2a615da9b9c1fb90135b0f5acc6da6a275e82c.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/19aaaffec5ef24fd763c5fa62f2a615da9b9c1fb90135b0f5acc6da6a275e82c.jpg)

![2da5ccd89fde60e3b43a43ff2e60f27d78b6725ba7b24809306a720a14e2c9ce.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/2da5ccd89fde60e3b43a43ff2e60f27d78b6725ba7b24809306a720a14e2c9ce.jpg)

![487369086f8f873498d99b8e54efcdeb0d699ecfa5e767e64f4c474f85fe472f.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/487369086f8f873498d99b8e54efcdeb0d699ecfa5e767e64f4c474f85fe472f.jpg)

![5f64ce42123e1aa5ca330d7ca2260d6c2b9cf522cd91347154cbec80a8af9098.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/5f64ce42123e1aa5ca330d7ca2260d6c2b9cf522cd91347154cbec80a8af9098.jpg)

![7b6a860a386e8239ecc4ad5912d8ffd40c8fc2e6af4b1b3bf2fbf30bfb8f0c7c.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/7b6a860a386e8239ecc4ad5912d8ffd40c8fc2e6af4b1b3bf2fbf30bfb8f0c7c.jpg)

![80c5fcb309708de5a20257515de805efb1c9fc0aa4a61cbb75aa20f18f4cf3ce.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/80c5fcb309708de5a20257515de805efb1c9fc0aa4a61cbb75aa20f18f4cf3ce.jpg)

![a998dd131f25c2625bb130fbb3f01202ab710d057381d279b504d89e9c3b8f76.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/a998dd131f25c2625bb130fbb3f01202ab710d057381d279b504d89e9c3b8f76.jpg)

![dde8f88c4af7ff1eca067d317a5a874f6097c53895d648cb065b888809b7aa00.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/dde8f88c4af7ff1eca067d317a5a874f6097c53895d648cb065b888809b7aa00.jpg)

![ee05cd7e64633123ef8e6d2404ec7e70e7ff44ce8bbda8582b49756d38717403.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/ee05cd7e64633123ef8e6d2404ec7e70e7ff44ce8bbda8582b49756d38717403.jpg)

![fda803e244e40f7886cd074856956ff0c95bedab22818e1f07229d68011d4fbe.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/images/fda803e244e40f7886cd074856956ff0c95bedab22818e1f07229d68011d4fbe.jpg)

### Tables

![16b5bdde26eb48efb1c82559de6c38782451851eec7f9c4bdc77c81e312fd625.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/16b5bdde26eb48efb1c82559de6c38782451851eec7f9c4bdc77c81e312fd625.jpg)

![2fef3cea69fc3fdf5cc56ee57956cb107f8b66fe9e8e86f2de021f0956214bd8.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/2fef3cea69fc3fdf5cc56ee57956cb107f8b66fe9e8e86f2de021f0956214bd8.jpg)

![5b87e34c1d41eeb9fcf821e26cd9c2ef075513e883c82826ba77a0691d3dae31.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/5b87e34c1d41eeb9fcf821e26cd9c2ef075513e883c82826ba77a0691d3dae31.jpg)

![6443a44eb61336adfa14ea770658e985ec912251bc39bffb275d7977ac26ef43.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/6443a44eb61336adfa14ea770658e985ec912251bc39bffb275d7977ac26ef43.jpg)

![76247d8a17f1a1b155a75049cbe873accea4cb7c8fe8a61e24703c54e6606add.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/76247d8a17f1a1b155a75049cbe873accea4cb7c8fe8a61e24703c54e6606add.jpg)

![7777de9df8c2df89a6453c5b389bd05d9872a19507fca1c94310b53244f6dbfb.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/7777de9df8c2df89a6453c5b389bd05d9872a19507fca1c94310b53244f6dbfb.jpg)

![832563eeba1a2770ac3d6be00415726f8f2e62b2713949910008b0ee449bdfd4.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/832563eeba1a2770ac3d6be00415726f8f2e62b2713949910008b0ee449bdfd4.jpg)

![911bde7c22545d4eef2198914c3d120cd721dac7cf733524a3e83a9932a20621.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/911bde7c22545d4eef2198914c3d120cd721dac7cf733524a3e83a9932a20621.jpg)

![991f38a7d145db6490dddf2b08d9e5d7d3a974e6de46edd3a6887886c251951f.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/991f38a7d145db6490dddf2b08d9e5d7d3a974e6de46edd3a6887886c251951f.jpg)

![bdb2326bc01f5d5160425bbd14ebc73390445f557a6f4f65fc13afeb59fd9e8a.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/bdb2326bc01f5d5160425bbd14ebc73390445f557a6f4f65fc13afeb59fd9e8a.jpg)

![c71b4d84f4e765457782f7d7951993a9c8f837d6e920186c932840f42036c9ce.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/c71b4d84f4e765457782f7d7951993a9c8f837d6e920186c932840f42036c9ce.jpg)

![e2fffab3f4e6c0b3f9f55ad64c30b5059c91c109c2959b833fa1491804b0e7aa.jpg](../icml_results/6340_Fast%20KV%20Compaction%20via%20Attention%20Matching/tables/e2fffab3f4e6c0b3f9f55ad64c30b5059c91c109c2959b833fa1491804b0e7aa.jpg)

<a id="paper-6288"></a>

## Position: Predictive Uncertainty Is Not Enough — Joint Distribution for Full Uncertainty Representation


### Images

![0501c3f26962f979ed15c4578877b1ad9db898c8bb00d7368a4edc1b1cdb59f3.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/0501c3f26962f979ed15c4578877b1ad9db898c8bb00d7368a4edc1b1cdb59f3.jpg)

![206237425af70b96ba7679e740fb39de262c4386db1a1de9396742cde827ee09.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/206237425af70b96ba7679e740fb39de262c4386db1a1de9396742cde827ee09.jpg)

![29502874d806e5869bdd1fa196200045d8c4ebf615c5792d62de938ab45396b1.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/29502874d806e5869bdd1fa196200045d8c4ebf615c5792d62de938ab45396b1.jpg)

![2ca8e4f4f8c9f017fcb77d37b3331393e23410176fd9ac4813a5c14d568a032b.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/2ca8e4f4f8c9f017fcb77d37b3331393e23410176fd9ac4813a5c14d568a032b.jpg)

![2e028020fb99b11e10b4552da6b02c18094cc52c621a7fc8661df994082bcf9a.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/2e028020fb99b11e10b4552da6b02c18094cc52c621a7fc8661df994082bcf9a.jpg)

![48b75ad945768dfb30a772c65fe6117ff18aeeb77b3fc8dbc1f2a152bd92d911.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/48b75ad945768dfb30a772c65fe6117ff18aeeb77b3fc8dbc1f2a152bd92d911.jpg)

![4be0d407a374ea617d5450532304c18dbbd552cb6bf8dedd6ce8313fa86d3605.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/4be0d407a374ea617d5450532304c18dbbd552cb6bf8dedd6ce8313fa86d3605.jpg)

![5a369f3598f3ab78779f44485aefd3169de2308af35aade7ab27f8fd6b14d4a2.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/5a369f3598f3ab78779f44485aefd3169de2308af35aade7ab27f8fd6b14d4a2.jpg)

![63701c5fb17599cb50fa31706bfdf6d198f125e868e942fddaa59f3e7740359d.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/63701c5fb17599cb50fa31706bfdf6d198f125e868e942fddaa59f3e7740359d.jpg)

![b23cbb01af98cb223a168e7c94d27773da99d215a7ea1f5b9074d4a20c3ad412.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/b23cbb01af98cb223a168e7c94d27773da99d215a7ea1f5b9074d4a20c3ad412.jpg)

![ca5d0ca033b19847fa29d71163c7c0e9608c555c8c89dbdb656b77de0906a5aa.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/ca5d0ca033b19847fa29d71163c7c0e9608c555c8c89dbdb656b77de0906a5aa.jpg)

![ccfedf19f67ff0489db29141b29c60e7630fc39717c0659f914bea4810a358e3.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/ccfedf19f67ff0489db29141b29c60e7630fc39717c0659f914bea4810a358e3.jpg)

![db69210161c6a2cb0f5016119214810f1cdf1d1bcd29cd52fad8dd852068e70a.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/db69210161c6a2cb0f5016119214810f1cdf1d1bcd29cd52fad8dd852068e70a.jpg)

![f9cd891bb5e96cd414ce57717fa5ae1b391c9b03784a60ef88e3fc7464f5750a.jpg](../icml_results/6341_Position_%20Predictive%20Uncertainty%20Is%20Not%20Enough%20%E2%80%94%20Joint%20Distribution%20for%20Full%20Uncertainty%20Representat/images/f9cd891bb5e96cd414ce57717fa5ae1b391c9b03784a60ef88e3fc7464f5750a.jpg)

<a id="paper-6289"></a>

## Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives


### Images

![0d69f48a5125e221533f53a6956b640e211a063442b7f232c6ce58916632d299.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/images/0d69f48a5125e221533f53a6956b640e211a063442b7f232c6ce58916632d299.jpg)

![1d2339a16d19450376ab65b59ea08692ff845d907bddbc95bc97e853512aaf74.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/images/1d2339a16d19450376ab65b59ea08692ff845d907bddbc95bc97e853512aaf74.jpg)

![cdbd2aa86449f799d01d3fd0fe566637e91272e08c8bb4096837490975aa94da.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/images/cdbd2aa86449f799d01d3fd0fe566637e91272e08c8bb4096837490975aa94da.jpg)

![fb2a229d9f737eaba0fc3502b5e0fe7d238db901df7d253e7ecb72c69f012216.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/images/fb2a229d9f737eaba0fc3502b5e0fe7d238db901df7d253e7ecb72c69f012216.jpg)

### Tables

![0a82ab8f3a579d431a4805115a9ffa02ee0683a91857b72125bcb3cab9b78a44.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/tables/0a82ab8f3a579d431a4805115a9ffa02ee0683a91857b72125bcb3cab9b78a44.jpg)

![c74f3e4d9d5ae15198095fd94b073a9710306c53da1ca5b0a337d0684da405cd.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/tables/c74f3e4d9d5ae15198095fd94b073a9710306c53da1ca5b0a337d0684da405cd.jpg)

![ca4789ae508abb0a387f65cbae08dc051d5eb0358df3a381820a4c255fb785d1.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/tables/ca4789ae508abb0a387f65cbae08dc051d5eb0358df3a381820a4c255fb785d1.jpg)

![e27853d22e0cb2d0029e6a3cc1ce0d98edf33a7961ae982fd76c1c725484c76b.jpg](../icml_results/6342_Position_%20Machine%20Learning%20for%20Heart%20Transplant%20Allocation%20Policy%20Optimization%20Should%20Account%20for%20In/tables/e27853d22e0cb2d0029e6a3cc1ce0d98edf33a7961ae982fd76c1c725484c76b.jpg)

<a id="paper-6290"></a>

## Position: We Need Large Language Models Optimized For Our Well-Being


### Tables

![9c430df2e8b14bc87d13e2306ade5f3cedac7fecde9ec79bd40052711dc8911e.jpg](../icml_results/6343_Position_%20We%20Need%20Large%20Language%20Models%20Optimized%20For%20Our%20Well-Being/tables/9c430df2e8b14bc87d13e2306ade5f3cedac7fecde9ec79bd40052711dc8911e.jpg)

<a id="paper-6291"></a>

## Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race


### Images

![3b0e9267da051f3f106a70f6f608a12db8446086a154e1c3d1db4f89afb39f7d.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/3b0e9267da051f3f106a70f6f608a12db8446086a154e1c3d1db4f89afb39f7d.jpg)

![7c2a4753cde26d90ceff43afb33ef9524e7b83a1cf79b9061edd5fad60e7dd0f.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/7c2a4753cde26d90ceff43afb33ef9524e7b83a1cf79b9061edd5fad60e7dd0f.jpg)

![984890a052132dfdd65861326ed4bc1c72792906d7da0d2beef2e36e1d569a26.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/984890a052132dfdd65861326ed4bc1c72792906d7da0d2beef2e36e1d569a26.jpg)

![b007a5a3a3577d2736e7c0a6e95689e5223d71cbeca1ab1f02733a614a0afd5a.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/b007a5a3a3577d2736e7c0a6e95689e5223d71cbeca1ab1f02733a614a0afd5a.jpg)

![d43df1abc85e6606db57317675950e36335bbf9b1eadc1dc194e10a8ae64f593.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/d43df1abc85e6606db57317675950e36335bbf9b1eadc1dc194e10a8ae64f593.jpg)

![d834d80b852eb8b345ebf5622aa7e6f99986e83329e146bbb23970793906a259.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/d834d80b852eb8b345ebf5622aa7e6f99986e83329e146bbb23970793906a259.jpg)

![f1749df84ca9c5683d4cdf355565cf7ac9bada83463dba47ff92e150f86e50e8.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/images/f1749df84ca9c5683d4cdf355565cf7ac9bada83463dba47ff92e150f86e50e8.jpg)

### Tables

![26079caa81321152ba4ba06102cf781d62308e05e15a970b7b41826d91c07cf7.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/tables/26079caa81321152ba4ba06102cf781d62308e05e15a970b7b41826d91c07cf7.jpg)

![f2454ab4b9d79bac85ad415e82212d810a04e7d97dca5cd17cdded3b84681f7c.jpg](../icml_results/6344_Position_%20Neglecting%20the%20Sustainability%20of%20AI%20is%20Fuelling%20a%20Global%20AI%20Arms%20Race/tables/f2454ab4b9d79bac85ad415e82212d810a04e7d97dca5cd17cdded3b84681f7c.jpg)

<a id="paper-6292"></a>

## Position: The Alignment Community is Unintentionally Building a Censor’s Toolkit


### Tables

![60bc8c95af689b6dbecd5e42b81cf3b78d5997677ff8aba3a59ac45c811c38c9.jpg](../icml_results/6345_Position_%20The%20Alignment%20Community%20is%20Unintentionally%20Building%20a%20Censor%E2%80%99s%20Toolkit/tables/60bc8c95af689b6dbecd5e42b81cf3b78d5997677ff8aba3a59ac45c811c38c9.jpg)

<a id="paper-6293"></a>

## Position: Agentic Systems Should be General


### Images

![dcee8da61284fbf856aef33c272128e8791b7e5c369f2cc1c9c7e3f1efa29376.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/images/dcee8da61284fbf856aef33c272128e8791b7e5c369f2cc1c9c7e3f1efa29376.jpg)

### Tables

![1e12df6126ee90505e4ecbb3a7b7c9f407774898dbc456efed1142f444d2d6f5.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/tables/1e12df6126ee90505e4ecbb3a7b7c9f407774898dbc456efed1142f444d2d6f5.jpg)

![5785e01c1d18c8c2c9c0fa44252cabd97b7b3d1a09f8505abff8762edea99ff2.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/tables/5785e01c1d18c8c2c9c0fa44252cabd97b7b3d1a09f8505abff8762edea99ff2.jpg)

![601b3a71be3e1c5e80f5997e15ca17199933dbabe46c221c8dc6191ce02d5cc6.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/tables/601b3a71be3e1c5e80f5997e15ca17199933dbabe46c221c8dc6191ce02d5cc6.jpg)

![96ff414188ff9ee0b27d903f1c9afd788578302570f7b4bb541def020a9c1f5c.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/tables/96ff414188ff9ee0b27d903f1c9afd788578302570f7b4bb541def020a9c1f5c.jpg)

![a8d4e0d9019ec6a23776bebf939c0aa98b29c6c7415707470bae5c23a1bede2b.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/tables/a8d4e0d9019ec6a23776bebf939c0aa98b29c6c7415707470bae5c23a1bede2b.jpg)

![ee86e242a9b2fd2a9136983b1b0eebca3c056e01ac34e07cb90be107d2e6d65a.jpg](../icml_results/6346_Position_%20Agentic%20Systems%20Should%20be%20General/tables/ee86e242a9b2fd2a9136983b1b0eebca3c056e01ac34e07cb90be107d2e6d65a.jpg)

<a id="paper-6294"></a>

## Position: AI Leaderboards Are Underserving the Global South: A Case Study from India


### Images

![0990ed5a34d62d9c3e5687c1c2eb90c388b94c5c97112423a3ab5803d33b9022.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/0990ed5a34d62d9c3e5687c1c2eb90c388b94c5c97112423a3ab5803d33b9022.jpg)

![12eba77e13731fd1a0abf3f2d21d2c7a336da4c8174a20fd370f2921efa0db67.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/12eba77e13731fd1a0abf3f2d21d2c7a336da4c8174a20fd370f2921efa0db67.jpg)

![25ea551e2c9a6def7f56d87da348c4d7f0d1b374d8116a50fa4735f9906b3052.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/25ea551e2c9a6def7f56d87da348c4d7f0d1b374d8116a50fa4735f9906b3052.jpg)

![2dfd3aad7e9c35fc66afd971d2d6834fedc23a79e2c4e7363fecd4010c28ecf3.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/2dfd3aad7e9c35fc66afd971d2d6834fedc23a79e2c4e7363fecd4010c28ecf3.jpg)

![3c1b0477f362ae9dc2a718707a4b6117ce9393e061050ade21ed28c771a2c1d3.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/3c1b0477f362ae9dc2a718707a4b6117ce9393e061050ade21ed28c771a2c1d3.jpg)

![3d1547f6b1560d45830d0b7a89dd516174d0b5da485efcda715be2ec34b6adb2.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/3d1547f6b1560d45830d0b7a89dd516174d0b5da485efcda715be2ec34b6adb2.jpg)

![485068e82434b9248fcfadcb13655922856f21cef5612e760deb87592189a7fd.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/485068e82434b9248fcfadcb13655922856f21cef5612e760deb87592189a7fd.jpg)

![56e215870cf6d67f5db257544b3e1042fdf1a9e1cf553072478a058b41933890.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/56e215870cf6d67f5db257544b3e1042fdf1a9e1cf553072478a058b41933890.jpg)

![7d912c986049f68f58b34174a611806cefd681bed4095124d94c84d3ec85e956.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/7d912c986049f68f58b34174a611806cefd681bed4095124d94c84d3ec85e956.jpg)

![85dfab484200f98c79f48fb6cd98ae6e676627b41bdf87402743ea3f9bf5f2d7.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/85dfab484200f98c79f48fb6cd98ae6e676627b41bdf87402743ea3f9bf5f2d7.jpg)

![87a0de2d3258c0844cdfc82b3f6e08a07475973f85fe1eba5b1cf87af80ac250.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/87a0de2d3258c0844cdfc82b3f6e08a07475973f85fe1eba5b1cf87af80ac250.jpg)

![887ed7fb9b8ba49f0a7637757c7803073490fe8962846635150daf4ad8037cce.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/887ed7fb9b8ba49f0a7637757c7803073490fe8962846635150daf4ad8037cce.jpg)

![91654e55f07ff715b581d33043f4c8f31d18bc5616498d854113e800296982e8.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/91654e55f07ff715b581d33043f4c8f31d18bc5616498d854113e800296982e8.jpg)

![9410981b203a2155e26b5c398ad09e1f1ad27806a7502855ec2b3e8ad60651db.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/9410981b203a2155e26b5c398ad09e1f1ad27806a7502855ec2b3e8ad60651db.jpg)

![9d5749b0cc88b7f0aa4f840149804ddbb986590e1875e42c20cab1a94af3915c.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/9d5749b0cc88b7f0aa4f840149804ddbb986590e1875e42c20cab1a94af3915c.jpg)

![b015eb11caad3ed04092dda84f8733e4e8b45dc603dacb3e4cc9d9004a27680f.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/b015eb11caad3ed04092dda84f8733e4e8b45dc603dacb3e4cc9d9004a27680f.jpg)

![d715ade84895dce380e993e47f145259c4396f29a9bab6925ba3c5ec21a69acd.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/d715ade84895dce380e993e47f145259c4396f29a9bab6925ba3c5ec21a69acd.jpg)

![e5ee235d4a46f5540b653be428c0b06afb5c9f75c3d7c37ab97f8987d8daa4b1.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/e5ee235d4a46f5540b653be428c0b06afb5c9f75c3d7c37ab97f8987d8daa4b1.jpg)

![e6e718ce6906d37082de13c9c9c3be9a044d3ef5b5f6e9d3ddc1116e4d6e38a5.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/e6e718ce6906d37082de13c9c9c3be9a044d3ef5b5f6e9d3ddc1116e4d6e38a5.jpg)

![fab8a396a7540f63699d3da4f4655d78013f808c41390146f8110b2c6d0912b5.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/images/fab8a396a7540f63699d3da4f4655d78013f808c41390146f8110b2c6d0912b5.jpg)

### Tables

![1a4410f6a22f3eb4818296db4c68298bcdf30706728eb855e4d51dac4b27519e.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/tables/1a4410f6a22f3eb4818296db4c68298bcdf30706728eb855e4d51dac4b27519e.jpg)

![22816451d502f1adb89dd18bfa417ebbbcae4288c37d5f1a2d47586689f56fba.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/tables/22816451d502f1adb89dd18bfa417ebbbcae4288c37d5f1a2d47586689f56fba.jpg)

![3acc51e1531b0bedaebaef78a0d4e808f7af8ecf84b3193b4c13de8b6bbf570e.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/tables/3acc51e1531b0bedaebaef78a0d4e808f7af8ecf84b3193b4c13de8b6bbf570e.jpg)

![49bd09fa29a33431ccbf37b1662eb2835dd9d63c8c59dca210fde111416f1850.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/tables/49bd09fa29a33431ccbf37b1662eb2835dd9d63c8c59dca210fde111416f1850.jpg)

![eba89fa6b881a2bc9f76cd1cf55e94f4f779fed11004665df80a5d5236f06164.jpg](../icml_results/6347_Position_%20AI%20Leaderboards%20Are%20Underserving%20the%20Global%20South_%20A%20Case%20Study%20from%20India/tables/eba89fa6b881a2bc9f76cd1cf55e94f4f779fed11004665df80a5d5236f06164.jpg)

<a id="paper-6295"></a>

## Position: Stop Automating Peer Review Without Rigorous Evaluation


### Images

![072fabba631b428c2edb4eb7d125190d6eb081704a49cbc495d76fa3acfa51fe.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/072fabba631b428c2edb4eb7d125190d6eb081704a49cbc495d76fa3acfa51fe.jpg)

![19319573ca2716b880b278e0fade54042df24c9f75298b8b254b1947e9528c6e.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/19319573ca2716b880b278e0fade54042df24c9f75298b8b254b1947e9528c6e.jpg)

![25cfaf9be26812c1a9c345d548d1d61c3ed6fbf954af6b44ab65e1b9aa605093.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/25cfaf9be26812c1a9c345d548d1d61c3ed6fbf954af6b44ab65e1b9aa605093.jpg)

![2b436fb19d5d0980b9f6de2446ee0e2714c15e0839324b70296c149157d78ff0.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/2b436fb19d5d0980b9f6de2446ee0e2714c15e0839324b70296c149157d78ff0.jpg)

![77b1ac0989ac8f10baacea4573dcf692051a62e7766951afb17f770a86b4baee.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/77b1ac0989ac8f10baacea4573dcf692051a62e7766951afb17f770a86b4baee.jpg)

![7a95974291d3803c6714abe682d194107e6a8e794b6cba2c0010a839e38bd311.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/7a95974291d3803c6714abe682d194107e6a8e794b6cba2c0010a839e38bd311.jpg)

![7f9781810f1fa7068ed26014544b0f21eb92b2800ef13d83b5cda268b592469f.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/7f9781810f1fa7068ed26014544b0f21eb92b2800ef13d83b5cda268b592469f.jpg)

![972a6815f0e063e40e6bd70b44f71f7486710a10a1209519d3f8656d69dbe1c4.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/972a6815f0e063e40e6bd70b44f71f7486710a10a1209519d3f8656d69dbe1c4.jpg)

![aa712d2ec77a2f93441ecdd946b409f9d9c330aed33940e2284c1cfac8df4763.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/aa712d2ec77a2f93441ecdd946b409f9d9c330aed33940e2284c1cfac8df4763.jpg)

![c05e34dbbeb5400c6d7cf49a57125a55afd8e7f8e6a242073437da43386ee25e.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/c05e34dbbeb5400c6d7cf49a57125a55afd8e7f8e6a242073437da43386ee25e.jpg)

![d10b11b46e5c83aa528191286132464a290d98c771c31f1e37ea831e33eb67d6.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/d10b11b46e5c83aa528191286132464a290d98c771c31f1e37ea831e33eb67d6.jpg)

![d3a99afac16fe6df04e10b008bda13b70718e103ea130740aa2231aac675b857.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/images/d3a99afac16fe6df04e10b008bda13b70718e103ea130740aa2231aac675b857.jpg)

### Tables

![121c32b32c8e8ac28eead5f7c8f0f53d4b29747687f2a0188d88bfb79403c3ec.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/121c32b32c8e8ac28eead5f7c8f0f53d4b29747687f2a0188d88bfb79403c3ec.jpg)

![49abbd61dcaa48cd3395e4ffbf65edfdd9231dab060a1caa31a9425ff6d47ad7.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/49abbd61dcaa48cd3395e4ffbf65edfdd9231dab060a1caa31a9425ff6d47ad7.jpg)

![4d45203e0a1a88631dbde83ef51ac500735c5ce8e8c03823395af591c23b9cfa.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/4d45203e0a1a88631dbde83ef51ac500735c5ce8e8c03823395af591c23b9cfa.jpg)

![59a11ef216e9f10252959105b13529f50464655053ba27a7cf4f4f99094c446f.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/59a11ef216e9f10252959105b13529f50464655053ba27a7cf4f4f99094c446f.jpg)

![653fd52e0dd8b2b3f47dc6a7b0c083c08535ed8c7806afbfd45cb136ec86bb25.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/653fd52e0dd8b2b3f47dc6a7b0c083c08535ed8c7806afbfd45cb136ec86bb25.jpg)

![b650997589b43c4985ac7c6a6068e6f0b51df7b9f5f1a0bd99bae0a0e4b25902.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/b650997589b43c4985ac7c6a6068e6f0b51df7b9f5f1a0bd99bae0a0e4b25902.jpg)

![b8a60313c759aa605a7386c12b6b2aef029755b59b8e4a3ca075f5b6ed1021a8.jpg](../icml_results/6348_Position_%20Stop%20Automating%20Peer%20Review%20Without%20Rigorous%20Evaluation/tables/b8a60313c759aa605a7386c12b6b2aef029755b59b8e4a3ca075f5b6ed1021a8.jpg)

<a id="paper-6296"></a>

## Position: Deployed Reinforcement Learning should be Continual


### Images

![153f9c8af6d724ddbf9ecb9f05daa448b7a76295b8c06dcf000a95090e4a5fa2.jpg](../icml_results/6349_Position_%20Deployed%20Reinforcement%20Learning%20should%20be%20Continual/images/153f9c8af6d724ddbf9ecb9f05daa448b7a76295b8c06dcf000a95090e4a5fa2.jpg)

![5aa677c78adb1638621d975c6dc20bec9b618ef62e5ae9f98ed85e27e7d61adf.jpg](../icml_results/6349_Position_%20Deployed%20Reinforcement%20Learning%20should%20be%20Continual/images/5aa677c78adb1638621d975c6dc20bec9b618ef62e5ae9f98ed85e27e7d61adf.jpg)

![6705473e0abc1a580d3cc41a60ca144b8a876ad6a5ed6bd227012449cd7ab03f.jpg](../icml_results/6349_Position_%20Deployed%20Reinforcement%20Learning%20should%20be%20Continual/images/6705473e0abc1a580d3cc41a60ca144b8a876ad6a5ed6bd227012449cd7ab03f.jpg)

![7be2c851e11e085237e29840ce5e3a00900dac9cb4890fb48cd75bb59c752e23.jpg](../icml_results/6349_Position_%20Deployed%20Reinforcement%20Learning%20should%20be%20Continual/images/7be2c851e11e085237e29840ce5e3a00900dac9cb4890fb48cd75bb59c752e23.jpg)

![b033a9b998b085110b70682380cb4942221ae8a1c8d650338f2e5fbaafe40fec.jpg](../icml_results/6349_Position_%20Deployed%20Reinforcement%20Learning%20should%20be%20Continual/images/b033a9b998b085110b70682380cb4942221ae8a1c8d650338f2e5fbaafe40fec.jpg)

### Tables

![7470d33cd3065968c9dcf25471f176fc368c74c16f7b7c00ea0f1d7550dcdee2.jpg](../icml_results/6349_Position_%20Deployed%20Reinforcement%20Learning%20should%20be%20Continual/tables/7470d33cd3065968c9dcf25471f176fc368c74c16f7b7c00ea0f1d7550dcdee2.jpg)

<a id="paper-6297"></a>

## Position: Unplugging a Seemingly Sentient Machine Is the Rational Choice — A Metaphysical Perspective


### Images

![22a3c778656ce5ab1fe9b133b4cf9faadafd240ad13cc5d10c22458752dca296.jpg](../icml_results/6350_Position_%20Unplugging%20a%20Seemingly%20Sentient%20Machine%20Is%20the%20Rational%20Choice%20%E2%80%94%20A%20Metaphysical%20Perspectiv/images/22a3c778656ce5ab1fe9b133b4cf9faadafd240ad13cc5d10c22458752dca296.jpg)

![432bb655292189643fe127de484c7df030e15bf3621142cc3a3d8c5d5633f228.jpg](../icml_results/6350_Position_%20Unplugging%20a%20Seemingly%20Sentient%20Machine%20Is%20the%20Rational%20Choice%20%E2%80%94%20A%20Metaphysical%20Perspectiv/images/432bb655292189643fe127de484c7df030e15bf3621142cc3a3d8c5d5633f228.jpg)

![9789759afb27f6437655ede011a7af8191ea99b84d90bd58e7dfa9536065b88d.jpg](../icml_results/6350_Position_%20Unplugging%20a%20Seemingly%20Sentient%20Machine%20Is%20the%20Rational%20Choice%20%E2%80%94%20A%20Metaphysical%20Perspectiv/images/9789759afb27f6437655ede011a7af8191ea99b84d90bd58e7dfa9536065b88d.jpg)

### Tables

![6de9f9d9d06fc5dd6bb3047fac18cc305b04f8b37863a4c40d2c40eb5a9b6ec2.jpg](../icml_results/6350_Position_%20Unplugging%20a%20Seemingly%20Sentient%20Machine%20Is%20the%20Rational%20Choice%20%E2%80%94%20A%20Metaphysical%20Perspectiv/tables/6de9f9d9d06fc5dd6bb3047fac18cc305b04f8b37863a4c40d2c40eb5a9b6ec2.jpg)

![986ed87d71d4ebbfed17861b9ba454272ce773009d3e12dcfe9b09cbf1ab0853.jpg](../icml_results/6350_Position_%20Unplugging%20a%20Seemingly%20Sentient%20Machine%20Is%20the%20Rational%20Choice%20%E2%80%94%20A%20Metaphysical%20Perspectiv/tables/986ed87d71d4ebbfed17861b9ba454272ce773009d3e12dcfe9b09cbf1ab0853.jpg)

<a id="paper-6298"></a>

## Position: Evaluation of ECG Representations Must Be Fixed


### Images

![6cd3d3fc1e300d594ad8124808963ec282f64d1678d5fb531482998d597075d2.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/images/6cd3d3fc1e300d594ad8124808963ec282f64d1678d5fb531482998d597075d2.jpg)

### Tables

![0fa63c8982e339947e60fcffaeb203fec79eb4d215d46dbde18c7148cfaddf41.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/0fa63c8982e339947e60fcffaeb203fec79eb4d215d46dbde18c7148cfaddf41.jpg)

![1009d0b4310891139101c92093263f3bc2f8420e84e7df1722667461af63731a.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/1009d0b4310891139101c92093263f3bc2f8420e84e7df1722667461af63731a.jpg)

![17f22730e6871f97473c797547455e7fd684f6e6b25b9978eff64ac1f9c46005.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/17f22730e6871f97473c797547455e7fd684f6e6b25b9978eff64ac1f9c46005.jpg)

![25035eb2a1d08a334eeab1029f263c3102280fbdb3bfd1ffe602ba2c161741da.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/25035eb2a1d08a334eeab1029f263c3102280fbdb3bfd1ffe602ba2c161741da.jpg)

![35d37fa20e113cd82c7470939f792a6f5b3989eec8c9ecd4b6ec2de80ed5ed6e.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/35d37fa20e113cd82c7470939f792a6f5b3989eec8c9ecd4b6ec2de80ed5ed6e.jpg)

![3a305989ef43f92c476e76d9d6ceeb2bac12cb63125361e59a3ca0d2a42c98e9.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/3a305989ef43f92c476e76d9d6ceeb2bac12cb63125361e59a3ca0d2a42c98e9.jpg)

![3c4c93be5616d84769bee06ac58d11e8843c6cd8ef4d822e147f358aea59701b.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/3c4c93be5616d84769bee06ac58d11e8843c6cd8ef4d822e147f358aea59701b.jpg)

![3edfb6fa549309e8be89911d5111b1951306eeff5fee44eec4176080c322af90.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/3edfb6fa549309e8be89911d5111b1951306eeff5fee44eec4176080c322af90.jpg)

![4ddb1df110fae75c6211d830fca944189ed6df407d12f5c3fe427ef9f0491097.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/4ddb1df110fae75c6211d830fca944189ed6df407d12f5c3fe427ef9f0491097.jpg)

![5d64fb11f3decef28b368ba7d1fd3f2b81519a3797367006b7b954aa54f1de17.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/5d64fb11f3decef28b368ba7d1fd3f2b81519a3797367006b7b954aa54f1de17.jpg)

![61b7058449f82174577b13274b5729ca9560928799ce154ca05a09465f45c8cf.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/61b7058449f82174577b13274b5729ca9560928799ce154ca05a09465f45c8cf.jpg)

![654d597cd8f7db5df7fcfa1633a020bfbfe4fec6bac3db5ef54be04f89ab1ef7.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/654d597cd8f7db5df7fcfa1633a020bfbfe4fec6bac3db5ef54be04f89ab1ef7.jpg)

![6789cbd80bdb095fa58c981bd088ea64377854153873b34e80406beb21a29775.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/6789cbd80bdb095fa58c981bd088ea64377854153873b34e80406beb21a29775.jpg)

![6a8f8bb486e90690cf0b2142297e4ce958fa3595f65206cdacc0f0546ebc9b15.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/6a8f8bb486e90690cf0b2142297e4ce958fa3595f65206cdacc0f0546ebc9b15.jpg)

![72082c9d12816e88113033e8563e3fcec6d75f2f8f28a0d5bcc33860f3db5ac3.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/72082c9d12816e88113033e8563e3fcec6d75f2f8f28a0d5bcc33860f3db5ac3.jpg)

![7b28ceb7e937b724de81e453db226cedd9e1da3742ccb24eea34f4b1053c8b48.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/7b28ceb7e937b724de81e453db226cedd9e1da3742ccb24eea34f4b1053c8b48.jpg)

![8f2fd60ac5e37d924846ca3d5c50f7f5d3fb9ca20af3825eb8dd346f5c5ad0f9.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/8f2fd60ac5e37d924846ca3d5c50f7f5d3fb9ca20af3825eb8dd346f5c5ad0f9.jpg)

![a96837bbe670ba52f88ef89413234d1ebb2a0ead0f3eeed600ad01a553952fd9.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/a96837bbe670ba52f88ef89413234d1ebb2a0ead0f3eeed600ad01a553952fd9.jpg)

![aab34462c564914ace4498e9a407d0e7c01dbb23322e2b6b5698bb0cdbfbc888.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/aab34462c564914ace4498e9a407d0e7c01dbb23322e2b6b5698bb0cdbfbc888.jpg)

![abbc758ecd26564a3950ed530774c4cf360d2221d03cf764e428b13e18b37a67.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/abbc758ecd26564a3950ed530774c4cf360d2221d03cf764e428b13e18b37a67.jpg)

![af4fc5b4bac6bd9ac18c818dddb5929ba055f69a5cf14530030bd8cbbbe0c354.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/af4fc5b4bac6bd9ac18c818dddb5929ba055f69a5cf14530030bd8cbbbe0c354.jpg)

![bc527f97f4cc4fd4cbcb20757a464b36955dd8f73d4fa734bd0765883d3d956f.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/bc527f97f4cc4fd4cbcb20757a464b36955dd8f73d4fa734bd0765883d3d956f.jpg)

![bfa52eed2d84183d76f1f38ff169f36842ebf26af918279709fbb12db9eabf4a.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/bfa52eed2d84183d76f1f38ff169f36842ebf26af918279709fbb12db9eabf4a.jpg)

![c26cdba349338198ed1cfc431f8cc534b5bd92b004075963411888217a6ac928.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/c26cdba349338198ed1cfc431f8cc534b5bd92b004075963411888217a6ac928.jpg)

![c2f5b41a083259db73365161737f64f5af794d1e3ca84e6f8ac001b5004bed8d.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/c2f5b41a083259db73365161737f64f5af794d1e3ca84e6f8ac001b5004bed8d.jpg)

![c9a3ce8112f93805dea0efc387cb8d3e277c4e09619204ee378c091262670a00.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/c9a3ce8112f93805dea0efc387cb8d3e277c4e09619204ee378c091262670a00.jpg)

![d08d90a5022cf9a862797f84ffd7ac7640b4d7e1f3014a056ddee6a2d3272352.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/d08d90a5022cf9a862797f84ffd7ac7640b4d7e1f3014a056ddee6a2d3272352.jpg)

![eee575e904ada17b55fad45a95bfc77ddc2e516623005be2c752e6fb40db5aa9.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/eee575e904ada17b55fad45a95bfc77ddc2e516623005be2c752e6fb40db5aa9.jpg)

![facb3f34165af7284a5a6b739cb7ad964fdfaf99e5ab3745138715ada8dcc8ef.jpg](../icml_results/6351_Position_%20Evaluation%20of%20ECG%20Representations%20Must%20Be%20Fixed/tables/facb3f34165af7284a5a6b739cb7ad964fdfaf99e5ab3745138715ada8dcc8ef.jpg)

<a id="paper-6299"></a>

## Position: LLM-Safety Evaluations Lack Robustness


### Images

![1ad067588f0b277f36d244c306485fa8bb7eb039277163d621cc859fda5c9474.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/1ad067588f0b277f36d244c306485fa8bb7eb039277163d621cc859fda5c9474.jpg)

![400f1f4a80df784477bae1d4ca052680ce57fba27d5a3fa784b4f67f919c9ee9.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/400f1f4a80df784477bae1d4ca052680ce57fba27d5a3fa784b4f67f919c9ee9.jpg)

![91cf01b5785fa9db093f24ef59dcb98eb1a304d670fccb4d720c494a5df79f44.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/91cf01b5785fa9db093f24ef59dcb98eb1a304d670fccb4d720c494a5df79f44.jpg)

![a9eb8c8e582880c42c374b96681991f7934eb2642f0600e1c3ee94ff3c867c9c.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/a9eb8c8e582880c42c374b96681991f7934eb2642f0600e1c3ee94ff3c867c9c.jpg)

![b781802b45b15a5082d4040a78de37c7f10380a673916fc962ce0ec181b37f43.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/b781802b45b15a5082d4040a78de37c7f10380a673916fc962ce0ec181b37f43.jpg)

![b9241c6a8f25710649636a4a04d7f260e614a1ff5daa72ecb4bcc76e7aaeb723.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/b9241c6a8f25710649636a4a04d7f260e614a1ff5daa72ecb4bcc76e7aaeb723.jpg)

![ce3adddb210c35b28580a7dfb3b57a72b320f3fc02da2793e32cc743353f7d9f.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/ce3adddb210c35b28580a7dfb3b57a72b320f3fc02da2793e32cc743353f7d9f.jpg)

![d6ff04e29e1765ee1d6fceb043e76aa52abc4166c8cfbddcf2f11f7d0a7708a9.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/images/d6ff04e29e1765ee1d6fceb043e76aa52abc4166c8cfbddcf2f11f7d0a7708a9.jpg)

### Tables

![b8e12008b3702d18707decdbac3bd5c893466d04f63f0a384755e73f2c916fd5.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/tables/b8e12008b3702d18707decdbac3bd5c893466d04f63f0a384755e73f2c916fd5.jpg)

![c0168671cfd334562cfaf684d844b7b65977f943f63ee9441be6daef7baf30c1.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/tables/c0168671cfd334562cfaf684d844b7b65977f943f63ee9441be6daef7baf30c1.jpg)

![f06588d74d8e5fd4c46f5515f39f777d325e87f6b20631248a51af4622256545.jpg](../icml_results/6352_Position_%20LLM-Safety%20Evaluations%20Lack%20Robustness/tables/f06588d74d8e5fd4c46f5515f39f777d325e87f6b20631248a51af4622256545.jpg)

<a id="paper-6300"></a>

## Position: Assistive AI requires Personalized Specialists, not Generalists


### Images

![037955c7e89d56cf0d059e9297fe2c90cb43eaef53f6b18cc7bb1471ff4c0589.jpg](../icml_results/6353_Position_%20Assistive%20AI%20requires%20Personalized%20Specialists%2C%20not%20Generalists/images/037955c7e89d56cf0d059e9297fe2c90cb43eaef53f6b18cc7bb1471ff4c0589.jpg)

![0ce5f7fd78255c18d6de80cec9286b2f240ba352dbc7e60ee282082b602bea35.jpg](../icml_results/6353_Position_%20Assistive%20AI%20requires%20Personalized%20Specialists%2C%20not%20Generalists/images/0ce5f7fd78255c18d6de80cec9286b2f240ba352dbc7e60ee282082b602bea35.jpg)

![717dbf521a4855926b538e10589f3b4dcc020afe0fdf613f543b64639f69b177.jpg](../icml_results/6353_Position_%20Assistive%20AI%20requires%20Personalized%20Specialists%2C%20not%20Generalists/images/717dbf521a4855926b538e10589f3b4dcc020afe0fdf613f543b64639f69b177.jpg)

<a id="paper-6301"></a>

## Position: It is Time to Virtualize Foundation Models with a Self-evolving Operating System Layer


### Images

![0052d095b3fe2e72ee802b195296a5328e99ddb530a83a105be8f2176c1db3be.jpg](../icml_results/6354_Position_%20It%20is%20Time%20to%20Virtualize%20Foundation%20Models%20with%20a%20Self-evolving%20Operating%20System%20Layer/images/0052d095b3fe2e72ee802b195296a5328e99ddb530a83a105be8f2176c1db3be.jpg)

![00dc6d300dae67a69438aa99d5bc89b3ed18b66c4c4d4d2288f894941271f6d5.jpg](../icml_results/6354_Position_%20It%20is%20Time%20to%20Virtualize%20Foundation%20Models%20with%20a%20Self-evolving%20Operating%20System%20Layer/images/00dc6d300dae67a69438aa99d5bc89b3ed18b66c4c4d4d2288f894941271f6d5.jpg)

![0963992614af0a4452110685a0d3feb50700c547ee525f1690eadb8d8a5711b0.jpg](../icml_results/6354_Position_%20It%20is%20Time%20to%20Virtualize%20Foundation%20Models%20with%20a%20Self-evolving%20Operating%20System%20Layer/images/0963992614af0a4452110685a0d3feb50700c547ee525f1690eadb8d8a5711b0.jpg)

![2c6e564b06eb4adc3e8060048658022bd35bc1c97e7f8f9ff8928ba24090f6e5.jpg](../icml_results/6354_Position_%20It%20is%20Time%20to%20Virtualize%20Foundation%20Models%20with%20a%20Self-evolving%20Operating%20System%20Layer/images/2c6e564b06eb4adc3e8060048658022bd35bc1c97e7f8f9ff8928ba24090f6e5.jpg)

![35aae833027193a1a146aaa21697d7ffa634c280931e19d2c3f64ea5f6210f38.jpg](../icml_results/6354_Position_%20It%20is%20Time%20to%20Virtualize%20Foundation%20Models%20with%20a%20Self-evolving%20Operating%20System%20Layer/images/35aae833027193a1a146aaa21697d7ffa634c280931e19d2c3f64ea5f6210f38.jpg)

![ba30b21e4a1753e5c3ebcf7a5535b628db8c3067c248c7c4555c8542d1b09785.jpg](../icml_results/6354_Position_%20It%20is%20Time%20to%20Virtualize%20Foundation%20Models%20with%20a%20Self-evolving%20Operating%20System%20Layer/images/ba30b21e4a1753e5c3ebcf7a5535b628db8c3067c248c7c4555c8542d1b09785.jpg)

<a id="paper-6302"></a>

## Position: Trustworthy AI Suffers from Invariance Conflicts and Causality is The Solution


### Images

![133f1d5fc084769861b91b445a0caba6806b1fde0c4ada3fa666e5c94f0bb47f.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/images/133f1d5fc084769861b91b445a0caba6806b1fde0c4ada3fa666e5c94f0bb47f.jpg)

![5ac8acaaee6dfc7c99393dfe79e0cc2f302e62b471d04599c78c1d444c2e27b7.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/images/5ac8acaaee6dfc7c99393dfe79e0cc2f302e62b471d04599c78c1d444c2e27b7.jpg)

![99df35f3bcfd6f427047f543272151d488700d52ef477ee23d44cc5915102d18.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/images/99df35f3bcfd6f427047f543272151d488700d52ef477ee23d44cc5915102d18.jpg)

![e04e8f7c72dc67d5708af10fcee476447722d854e613f6551168f419c9fdea71.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/images/e04e8f7c72dc67d5708af10fcee476447722d854e613f6551168f419c9fdea71.jpg)

### Tables

![a58cddf1e604a4bec106f3ec0eb19f13ccc1101db45a9b9f4bc86f744ce68192.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/tables/a58cddf1e604a4bec106f3ec0eb19f13ccc1101db45a9b9f4bc86f744ce68192.jpg)

![c5959cdc46b37a80dbbecdcb7dc74b12c2f8682dc92d3021ef859b80cc64fc00.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/tables/c5959cdc46b37a80dbbecdcb7dc74b12c2f8682dc92d3021ef859b80cc64fc00.jpg)

![cc477ef9113d442cf921f273a867522c9f1e3bfc8ff1089876646e782495d38a.jpg](../icml_results/6356_Position_%20Trustworthy%20AI%20Suffers%20from%20Invariance%20Conflicts%20and%20Causality%20is%20The%20Solution/tables/cc477ef9113d442cf921f273a867522c9f1e3bfc8ff1089876646e782495d38a.jpg)

<a id="paper-6303"></a>

## Position: We Need AI Efficiency Incentives for Accessibility and Sustainability

### Images

![182af9a037e991b8f9ba4fce7bd67410f9f1efff47e98adc7f9a0fc9cf3d9767.jpg](../icml_results/6357_Position_%20We%20Need%20AI%20Efficiency%20Incentives%20for%20Accessibility%20and%20Sustainability/images/182af9a037e991b8f9ba4fce7bd67410f9f1efff47e98adc7f9a0fc9cf3d9767.jpg)

![9562a45991e55075ce35eafe51e7eb4d202978f844b7bb4c0a9acbe97fc4c1a1.jpg](../icml_results/6357_Position_%20We%20Need%20AI%20Efficiency%20Incentives%20for%20Accessibility%20and%20Sustainability/images/9562a45991e55075ce35eafe51e7eb4d202978f844b7bb4c0a9acbe97fc4c1a1.jpg)

![d7ed6bed840976288aba25d0f2efe92202cc0b24b752fa3da27382e169d59569.jpg](../icml_results/6357_Position_%20We%20Need%20AI%20Efficiency%20Incentives%20for%20Accessibility%20and%20Sustainability/images/d7ed6bed840976288aba25d0f2efe92202cc0b24b752fa3da27382e169d59569.jpg)
