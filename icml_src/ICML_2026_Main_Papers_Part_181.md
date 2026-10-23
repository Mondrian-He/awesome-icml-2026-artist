# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 181 of 197

## 目录 (Table of Contents)

1. [$V_0$: A Generalist Value Model for Any Policy at State Zero](#paper-5941)
2. [Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following](#paper-5942)
3. [The Fairness Hierarchy: A viewpoint from causal inference](#paper-5943)
4. [Sketch-Based Low-Rank Model Merging with Shared Circulant Transforms](#paper-5944)
5. [Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity](#paper-5945)
6. [LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation](#paper-5946)
7. [Reasoning-VLA: An Efficient and Spatial-Guided General Vision-Language-Action Reasoning Model for Autonomous Driving](#paper-5947)
8. [Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](#paper-5948)
9. [Posterior Mismatch Matters: Adversarial Training for Long-Tailed Robustness](#paper-5949)
10. [Personalized Policy Learning through Discrete Experimentation](#paper-5950)
11. [A Generalist Pair-wise Progress Critic Model for Vision-Language-Action Robots](#paper-5951)
12. [MAGIC: Multi-Granularity Language-Informed Image Clustering](#paper-5952)
13. [Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation](#paper-5953)
14. [PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization](#paper-5954)
15. [Distributed Stochastic $K$-Level Optimization Over Networks](#paper-5955)
16. [STRIDE: Post-Training LLMs to Reason and Refine Bio-Sequences via Edit Trajectories](#paper-5956)
17. [SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising](#paper-5957)
18. [Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process](#paper-5958)
19. [Beyond Single Embedding: Modeling User Preferences as Distribution in Federated Recommendation](#paper-5959)
20. [Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs](#paper-5960)
21. [Joint Navigation and Manipulation Planning with 3D Interaction Chains](#paper-5961)
22. [Optimal Transport with Symmetry Groups](#paper-5962)
23. [Convergence Analysis of Decentralized Hessian-/Jacobian-Free Algorithm for Nonconvex Stochastic Bilevel Optimization](#paper-5963)
24. [Dynamic Multimodal Evaluation via Knowledge-Enhanced Benchmark Evolution](#paper-5964)
25. [Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](#paper-5965)
26. [Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing](#paper-5966)
27. [NITP: Next Implicit Token Prediction for LLM Pre-training](#paper-5967)
28. [XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](#paper-5968)
29. [Generalized Boundary FDR Control under Arbitrary Dependence: An Approach on Closure Principle](#paper-5969)
30. [GPan-LoRA: Gaussian Process Amortized Networks for Bayesian Low-Rank Adaptation in Large Language Models](#paper-5970)
31. [When Embedding-Based Defenses Fail: Rethinking Safety in LLM-Based Multi-Agent Systems](#paper-5971)
32. [Controlled SDEs for Long-Horizon Motion Generation under Latent Decision Uncertainty](#paper-5972)
33. [Subspace-Aware Feature Reshaping for Open-Set Graph Class-Incremental Learning](#paper-5973)

---


<a id="paper-5941"></a>

## $V_0$: A Generalist Value Model for Any Policy at State Zero

### Images

![19650c97e977950113882568cece9ad7b6556fb99995a05cec496114080b2144.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/19650c97e977950113882568cece9ad7b6556fb99995a05cec496114080b2144.jpg)

![2ee79b3e080ef32120b6528aa1b16b5ed5a94c71345daec42eb469c8da0c2a3d.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/2ee79b3e080ef32120b6528aa1b16b5ed5a94c71345daec42eb469c8da0c2a3d.jpg)

![3774039eba0b0cf86285e1c9850f5c8a71b971e72d1b941e6e44eaefca6f5aa3.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/3774039eba0b0cf86285e1c9850f5c8a71b971e72d1b941e6e44eaefca6f5aa3.jpg)

![56363f967aeac445dcbdabd551dadd1c63ba8954023fc45bdc88a2fc7d525a9a.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/56363f967aeac445dcbdabd551dadd1c63ba8954023fc45bdc88a2fc7d525a9a.jpg)

![6afe9b5961ba9d8ae2cd95dfb2165a888432e110eda444e24187b2f2728b94bf.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/6afe9b5961ba9d8ae2cd95dfb2165a888432e110eda444e24187b2f2728b94bf.jpg)

![7b5361967950cb5d490d2436d137c573175bcb8c610125d93aae2464644d1fa5.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/7b5361967950cb5d490d2436d137c573175bcb8c610125d93aae2464644d1fa5.jpg)

![95fc80dd22dd0bf39ae08286f274ddf1b35a0217d54ae9136b7b3c822b4b55fa.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/95fc80dd22dd0bf39ae08286f274ddf1b35a0217d54ae9136b7b3c822b4b55fa.jpg)

![b5b01c84f082f07e56ed7a61b5adcb0e00eadb57bedd4e587da7e521bf41c2c5.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/b5b01c84f082f07e56ed7a61b5adcb0e00eadb57bedd4e587da7e521bf41c2c5.jpg)

![c5be1340424a3bd77c0a97701212fa87b8b1a217df7b7c3061c35734fa3e2a84.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/c5be1340424a3bd77c0a97701212fa87b8b1a217df7b7c3061c35734fa3e2a84.jpg)

![e1f8fc50c480a0533ce5b704a092ca25f4fbfb33dc5826344dd5ca09a2142cd2.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/e1f8fc50c480a0533ce5b704a092ca25f4fbfb33dc5826344dd5ca09a2142cd2.jpg)

![e2b3b2eca9f52c639dc00363ce834bf1d1cc99745033ae7cd7607ce8ea578daf.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/e2b3b2eca9f52c639dc00363ce834bf1d1cc99745033ae7cd7607ce8ea578daf.jpg)

![e7a19b07b133070682ef53daae1939e7858f4c820e47eb0bf00a0f495176bf13.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/e7a19b07b133070682ef53daae1939e7858f4c820e47eb0bf00a0f495176bf13.jpg)

![edbc8997bd75788267dbe1d42e7682d961285c0f0b85bcc4ac6cb867014fb401.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/images/edbc8997bd75788267dbe1d42e7682d961285c0f0b85bcc4ac6cb867014fb401.jpg)

### Tables

![25d2d0f805f9902663475d308cd3a614ef35292ee9e955c019c1b5803ed19f22.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/25d2d0f805f9902663475d308cd3a614ef35292ee9e955c019c1b5803ed19f22.jpg)

![786d948135960e981ca06ce8452fbd6ff28e4305a6fb0bf0f99dda6f380d1bc3.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/786d948135960e981ca06ce8452fbd6ff28e4305a6fb0bf0f99dda6f380d1bc3.jpg)

![7efadef478ceab2be94ead9e2ea45af11fe36024197acf922a282cfaea772518.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/7efadef478ceab2be94ead9e2ea45af11fe36024197acf922a282cfaea772518.jpg)

![90a20a456bd8f1ca8bf36e3deb47c55dd783d163b1f10fb0816c2dec74841e5b.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/90a20a456bd8f1ca8bf36e3deb47c55dd783d163b1f10fb0816c2dec74841e5b.jpg)

![9ad9b8e69cb64405d54e761195aa1f8165e93bfbf1231d7d876a1a6fb5d6f21b.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/9ad9b8e69cb64405d54e761195aa1f8165e93bfbf1231d7d876a1a6fb5d6f21b.jpg)

![bd1ab632c9b728053f67cd3c14b2f8e1738d02e9307c817a3a636c688d0d98b5.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/bd1ab632c9b728053f67cd3c14b2f8e1738d02e9307c817a3a636c688d0d98b5.jpg)

![e08fe0338fe0cb28bad6243a92cb36875d6b558a14f53274ab5dfdd9c5d40a4e.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/e08fe0338fe0cb28bad6243a92cb36875d6b558a14f53274ab5dfdd9c5d40a4e.jpg)

![eebc3d6c87548f933bc7bac865f7f39db4d32b9aa16f8bf861b58ae139a2b5e4.jpg](../icml_results/5989_Hyperbolic%20RQ-VAE%20enhanced%20Generative%20Recommendation%20with%20Differential-Length%20Codebook%20Strategy/tables/eebc3d6c87548f933bc7bac865f7f39db4d32b9aa16f8bf861b58ae139a2b5e4.jpg)

## $V_0$: A Generalist Value Model for Any Policy at State Zero


### Images

![02604304e058274a0ed1fc8caf0bac009f6b3fe40538d6a6401ccda53756f243.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/02604304e058274a0ed1fc8caf0bac009f6b3fe40538d6a6401ccda53756f243.jpg)

![248e48b2d203edb7a7eec255bf32157528d280c76cc83658f04168b3666e950b.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/248e48b2d203edb7a7eec255bf32157528d280c76cc83658f04168b3666e950b.jpg)

![4bbe3007704bc05b08c3e549848dc1792bc33d776b95b149fe15a289d98ee05b.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/4bbe3007704bc05b08c3e549848dc1792bc33d776b95b149fe15a289d98ee05b.jpg)

![52712fc2a0feed470ae4f4a0e8ef269ddb1b2a17954ee76c347439c16dee00bf.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/52712fc2a0feed470ae4f4a0e8ef269ddb1b2a17954ee76c347439c16dee00bf.jpg)

![595a345f295fe3a33ec0bd9bf17eac2f24a6635901adc465dea88c2c0e99b7b3.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/595a345f295fe3a33ec0bd9bf17eac2f24a6635901adc465dea88c2c0e99b7b3.jpg)

![694558b7f5d618681af6e25e0382adc9d6a7a1c530274e404f37148326181cb0.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/694558b7f5d618681af6e25e0382adc9d6a7a1c530274e404f37148326181cb0.jpg)

![830cf28e9f3c6e3e76ac42ca92a30625ec514142febda9d6c31bc6fa88f1bf11.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/830cf28e9f3c6e3e76ac42ca92a30625ec514142febda9d6c31bc6fa88f1bf11.jpg)

![d63ece41a84b3054eda3c7cf5ad67d8d5fa245eab81308260bc91b81a743868f.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/d63ece41a84b3054eda3c7cf5ad67d8d5fa245eab81308260bc91b81a743868f.jpg)

![f2be46cb61a9cf5e49b1a03b204fcd5ae0449ac28afd48975b60bd179edd0e00.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/f2be46cb61a9cf5e49b1a03b204fcd5ae0449ac28afd48975b60bd179edd0e00.jpg)

![fdb8735f7ab62cafb25395d2a713e88c7964ea76ea4d3454a445790fb721ef63.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/images/fdb8735f7ab62cafb25395d2a713e88c7964ea76ea4d3454a445790fb721ef63.jpg)

### Tables

![0c6d5b787a2b4427acb61ae43b68c23f55fa9e951cb35228c2a7a660acf171b7.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/0c6d5b787a2b4427acb61ae43b68c23f55fa9e951cb35228c2a7a660acf171b7.jpg)

![0f33308d47793084384e546f8dd7d78a8d244385bc5dbec436c1a1317e653dfe.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/0f33308d47793084384e546f8dd7d78a8d244385bc5dbec436c1a1317e653dfe.jpg)

![3c445a9e9eeeabdaffa0e85be46968b28f8b4015c5f2d9eaddce30ae643a468a.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/3c445a9e9eeeabdaffa0e85be46968b28f8b4015c5f2d9eaddce30ae643a468a.jpg)

![4656cf32ad8a21917f383409add5cd4eb024ed938245307e3033c85444973780.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/4656cf32ad8a21917f383409add5cd4eb024ed938245307e3033c85444973780.jpg)

![59fbb15b1956c0aa6725417ea866968765238a6dc9ac8cca048c3f9748b8ebb1.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/59fbb15b1956c0aa6725417ea866968765238a6dc9ac8cca048c3f9748b8ebb1.jpg)

![6aae067947a1e76d6d7478b51239adaaf999de45f579d9f775b7ce1cfd07349f.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/6aae067947a1e76d6d7478b51239adaaf999de45f579d9f775b7ce1cfd07349f.jpg)

![77149c9d7929a09513772dd3f6bf2acbca4bf58160deeb0daed9c9a35fb28f2d.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/77149c9d7929a09513772dd3f6bf2acbca4bf58160deeb0daed9c9a35fb28f2d.jpg)

![b16f5869247e648a0a658a8173d7a319b7d0408108fb3a39c51c29042be88a4b.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/b16f5869247e648a0a658a8173d7a319b7d0408108fb3a39c51c29042be88a4b.jpg)

![c97a477ea7dfb972d70ae0d0a86932283844dc3cbaf12b27be5675a368b39ae1.jpg](../icml_results/5990_%24V_0%24_%20A%20Generalist%20Value%20Model%20for%20Any%20Policy%20at%20State%20Zero/tables/c97a477ea7dfb972d70ae0d0a86932283844dc3cbaf12b27be5675a368b39ae1.jpg)

<a id="paper-5942"></a>

## Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following


### Images

![0ca23722c9586053c7d5d8dc73335dc04e6c38d55516c342b80f8186fffe69ab.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/0ca23722c9586053c7d5d8dc73335dc04e6c38d55516c342b80f8186fffe69ab.jpg)

![0e70d7e50316bdd44a486573607d8126b8db70ee833d470009fe41e9f1811f57.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/0e70d7e50316bdd44a486573607d8126b8db70ee833d470009fe41e9f1811f57.jpg)

![147a2e70186543deb7b08db70646c19be62552c27e2e9b817f13411910fd9481.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/147a2e70186543deb7b08db70646c19be62552c27e2e9b817f13411910fd9481.jpg)

![1d8a026351d61befe2250f581091163223b7564e424e40a036265749fd842824.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/1d8a026351d61befe2250f581091163223b7564e424e40a036265749fd842824.jpg)

![379725f36be3e800a1a7b203e28f12caf23099ff763e43238f9d7223e85807bd.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/379725f36be3e800a1a7b203e28f12caf23099ff763e43238f9d7223e85807bd.jpg)

![45882164e7015e889850e273e6b2e827b7e521f17ecf95fa6133dd09331034ed.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/45882164e7015e889850e273e6b2e827b7e521f17ecf95fa6133dd09331034ed.jpg)

![5ccc310e6a612e2add6af78d1d20b1504f397cc50504e7c3ce28761daeb9eb83.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/5ccc310e6a612e2add6af78d1d20b1504f397cc50504e7c3ce28761daeb9eb83.jpg)

![c6c31d34ba564ce807c9b47f44e002d3a3920bd93d7d139b39284dee92a316d8.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/c6c31d34ba564ce807c9b47f44e002d3a3920bd93d7d139b39284dee92a316d8.jpg)

![f9f5e00b5fb315d164aa11b9bc1e001d6ce675ba09089e4a8c4259540766e4f2.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/f9f5e00b5fb315d164aa11b9bc1e001d6ce675ba09089e4a8c4259540766e4f2.jpg)

![fb16c9c7ee3872050dc551aedcefdfd0521ecd0c4f1cc5cf6fcdd5f40bbc760d.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/images/fb16c9c7ee3872050dc551aedcefdfd0521ecd0c4f1cc5cf6fcdd5f40bbc760d.jpg)

### Tables

![0486bb478061d0e25ae8a6415b1205b7d8a08a014ea3480e8088c745721f1ab2.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/0486bb478061d0e25ae8a6415b1205b7d8a08a014ea3480e8088c745721f1ab2.jpg)

![0ca387134947fc83ca8980dcec3cd21f2906a55909c853fa7b97725548764218.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/0ca387134947fc83ca8980dcec3cd21f2906a55909c853fa7b97725548764218.jpg)

![17eb4911b49d8d72ec3684328375aebdb4907ac3214cc37bf3cc76d6ba1c8fab.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/17eb4911b49d8d72ec3684328375aebdb4907ac3214cc37bf3cc76d6ba1c8fab.jpg)

![1895fdec611a8e8cf50fd67d10286dcde1b06597a8f6c3eda430492e3d5f1b05.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/1895fdec611a8e8cf50fd67d10286dcde1b06597a8f6c3eda430492e3d5f1b05.jpg)

![24bd0945ad923814fe6fa890e7935a5052a334056c53a20b9d3260313e3493d2.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/24bd0945ad923814fe6fa890e7935a5052a334056c53a20b9d3260313e3493d2.jpg)

![45bde8efe6fe80558855d9e32e5c6201ece2306258d5f121bf33e0cb7a4a2477.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/45bde8efe6fe80558855d9e32e5c6201ece2306258d5f121bf33e0cb7a4a2477.jpg)

![696b9f8d03f462cd740a263652cd12d0cc0f04eafe4414b32ce4833e61dc9242.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/696b9f8d03f462cd740a263652cd12d0cc0f04eafe4414b32ce4833e61dc9242.jpg)

![74463895c5ffc845be7d86060fb5d8d83d2a76d4cf7a1fc021402d52d8a858dd.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/74463895c5ffc845be7d86060fb5d8d83d2a76d4cf7a1fc021402d52d8a858dd.jpg)

![7c02d3311a5ab10818d515fafc8e92bc8141af82013125039413d33a26cad2c6.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/7c02d3311a5ab10818d515fafc8e92bc8141af82013125039413d33a26cad2c6.jpg)

![852a88dd81c7c72b77e368f876946a9da1cd07a55405255c46b89f765b4361cc.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/852a88dd81c7c72b77e368f876946a9da1cd07a55405255c46b89f765b4361cc.jpg)

![90a39a86d5d18748fad19915a9d601af5e6b08fe4d6a4a445ee568854c2616c5.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/90a39a86d5d18748fad19915a9d601af5e6b08fe4d6a4a445ee568854c2616c5.jpg)

![a8b5ef3c5ed8e2e242a5cd1244136f5f72069503a2eb04357ed2310d15b2cadd.jpg](../icml_results/5991_Replay%20Failures%20as%20Successes_%20Sample-Efficient%20Reinforcement%20Learning%20for%20Instruction%20Following/tables/a8b5ef3c5ed8e2e242a5cd1244136f5f72069503a2eb04357ed2310d15b2cadd.jpg)

<a id="paper-5943"></a>

## The Fairness Hierarchy: A viewpoint from causal inference


### Images

![01034bed96b419eac21130eab8bd981670d6f8eaa1a2729acf9233717ac17e97.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/01034bed96b419eac21130eab8bd981670d6f8eaa1a2729acf9233717ac17e97.jpg)

![105f1fa8f4f13b81fd2898db165c2e868ea03c3dda09b61064212985f4b57bfe.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/105f1fa8f4f13b81fd2898db165c2e868ea03c3dda09b61064212985f4b57bfe.jpg)

![13089c9592e08db5bce034f33c16997fd9e7a6a20c5d44940c1df7a831bc20f4.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/13089c9592e08db5bce034f33c16997fd9e7a6a20c5d44940c1df7a831bc20f4.jpg)

![16b4124b8f6ec00c6f8e208d847c5ff46999f7b3c1946eebcc16a359a77f9b9d.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/16b4124b8f6ec00c6f8e208d847c5ff46999f7b3c1946eebcc16a359a77f9b9d.jpg)

![2078594c437653362aa20d6f33b88f6dd169c5e477d9d62e2ee11396255e5e3e.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/2078594c437653362aa20d6f33b88f6dd169c5e477d9d62e2ee11396255e5e3e.jpg)

![260d2038b195bcb3e2b6e58b980fbe98ff91c098d46efa59c000793f5ec83593.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/260d2038b195bcb3e2b6e58b980fbe98ff91c098d46efa59c000793f5ec83593.jpg)

![29d2ea2b6fe4a0055ad4ce31ad085566bc1d6ad9b3fc6fcbb25072ae8ffe420e.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/29d2ea2b6fe4a0055ad4ce31ad085566bc1d6ad9b3fc6fcbb25072ae8ffe420e.jpg)

![3197570a30b615264f3ab90a6c320328f1c1fe6488fca27d2fa367123a6c8695.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/3197570a30b615264f3ab90a6c320328f1c1fe6488fca27d2fa367123a6c8695.jpg)

![552daddd6993f70f0179683f8d4b35d0152404660ede3477f4951c7eda56cf46.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/images/552daddd6993f70f0179683f8d4b35d0152404660ede3477f4951c7eda56cf46.jpg)

### Tables

![3255e6881d0563dc930a81bc37b5542b1e18afe467123135cd25a6711a0f33f9.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/tables/3255e6881d0563dc930a81bc37b5542b1e18afe467123135cd25a6711a0f33f9.jpg)

![3729f9154c831a5b75afc09f8ffa24042eb1fe47c1e08dffcfd53dc2546a2aaa.jpg](../icml_results/5992_The%20Fairness%20Hierarchy_%20A%20viewpoint%20from%20causal%20inference/tables/3729f9154c831a5b75afc09f8ffa24042eb1fe47c1e08dffcfd53dc2546a2aaa.jpg)

<a id="paper-5944"></a>

## Sketch-Based Low-Rank Model Merging with Shared Circulant Transforms


### Images

![4283c8f9df0fec092bff0fba48e556c70b7ba2c2b477e1971e5b389a01e54c78.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/images/4283c8f9df0fec092bff0fba48e556c70b7ba2c2b477e1971e5b389a01e54c78.jpg)

![78b16e4b42c6bd7a6a21b16bf6dea779721d43e361bcce582728ebf65af6d3e3.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/images/78b16e4b42c6bd7a6a21b16bf6dea779721d43e361bcce582728ebf65af6d3e3.jpg)

![b420c597ab6e164fb0f10c88984ce7c5984d9f5f6e3e29b57640319f17badaa2.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/images/b420c597ab6e164fb0f10c88984ce7c5984d9f5f6e3e29b57640319f17badaa2.jpg)

### Tables

![003db77b2776bd25b784326d6e7b2e4bf102a0066f7ec68822acf2f217f37730.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/tables/003db77b2776bd25b784326d6e7b2e4bf102a0066f7ec68822acf2f217f37730.jpg)

![25273ddffffe11142a14bff540661e537571f40e2512e7101a223b5407dddc22.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/tables/25273ddffffe11142a14bff540661e537571f40e2512e7101a223b5407dddc22.jpg)

![3b90bf70e5dafdb762bedccd6062ac1f979eb9e409de715c24c625a2b3b33a90.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/tables/3b90bf70e5dafdb762bedccd6062ac1f979eb9e409de715c24c625a2b3b33a90.jpg)

![b503f1b7ec9a4c951aed1d2950adc3c42a3e82b32e945ad020f5e7e5fe3e24be.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/tables/b503f1b7ec9a4c951aed1d2950adc3c42a3e82b32e945ad020f5e7e5fe3e24be.jpg)

![fb5b31edc92220d89480fe01c0c686f9e54b22de82d9c8ce46e773a2c0ccddfb.jpg](../icml_results/5993_Sketch-Based%20Low-Rank%20Model%20Merging%20with%20Shared%20Circulant%20Transforms/tables/fb5b31edc92220d89480fe01c0c686f9e54b22de82d9c8ce46e773a2c0ccddfb.jpg)

<a id="paper-5945"></a>

## Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity


### Images

![0857f53821f4fa3723e0552bad4c55ce684471d2d75119932cd0b78db63fd4d1.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/0857f53821f4fa3723e0552bad4c55ce684471d2d75119932cd0b78db63fd4d1.jpg)

![0c15be91d0f218dcdc41a603e100c5da4a0cec656934fa028ed3505f30a19271.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/0c15be91d0f218dcdc41a603e100c5da4a0cec656934fa028ed3505f30a19271.jpg)

![0dd56dda8bc2cdff63ce8bb503077e5b77110f0876d61d8045eba306ea40a8bd.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/0dd56dda8bc2cdff63ce8bb503077e5b77110f0876d61d8045eba306ea40a8bd.jpg)

![1add836b56ce8fc861b92dc6ec4b4aefd08cb4062f83b8c2be9378aab0935228.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/1add836b56ce8fc861b92dc6ec4b4aefd08cb4062f83b8c2be9378aab0935228.jpg)

![1c98fa9882c9944a1a743b9c960e15be29bfba7945afecbddb896b33eff1b8cc.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/1c98fa9882c9944a1a743b9c960e15be29bfba7945afecbddb896b33eff1b8cc.jpg)

![1e8c7d2778934d32802726e55e947c92da5b1606f27f8b5e6557af2440922ac8.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/1e8c7d2778934d32802726e55e947c92da5b1606f27f8b5e6557af2440922ac8.jpg)

![245d4d724f8d706128d6d726d52a0352c15db2ad5bdc8ca6451ffad52b3d51cf.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/245d4d724f8d706128d6d726d52a0352c15db2ad5bdc8ca6451ffad52b3d51cf.jpg)

![2a9448f562b5bf591d73194010dc9dc5d3f67a1ec277c86653af0474be96841f.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/2a9448f562b5bf591d73194010dc9dc5d3f67a1ec277c86653af0474be96841f.jpg)

![45e3de0b326b0725d9446a5b5e196ed568dd1ae6e8e16786ac8d7137cd53b7d5.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/45e3de0b326b0725d9446a5b5e196ed568dd1ae6e8e16786ac8d7137cd53b7d5.jpg)

![485f38038d105cfa53775a58180625d0432aa72952d92e25149eb2eb24b37e50.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/485f38038d105cfa53775a58180625d0432aa72952d92e25149eb2eb24b37e50.jpg)

![4e8375e447cd86b228c9c42b01b5a664105ef3a8d5de5cb52de2381f9f709925.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/4e8375e447cd86b228c9c42b01b5a664105ef3a8d5de5cb52de2381f9f709925.jpg)

![567a5b7b93d7afd80b1f191e5eeaa8a0fbcc7ad445b1a90782822ff1dc6196b6.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/567a5b7b93d7afd80b1f191e5eeaa8a0fbcc7ad445b1a90782822ff1dc6196b6.jpg)

![5b6ff39234915620a8a4de516b783689535e5adf20076808db2465c0e87ae625.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/5b6ff39234915620a8a4de516b783689535e5adf20076808db2465c0e87ae625.jpg)

![61739f189039b9c083dcd3d6f09a455416fbf0181358178cb97f940e6d23c271.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/61739f189039b9c083dcd3d6f09a455416fbf0181358178cb97f940e6d23c271.jpg)

![63251280953e148b3ad0152398d434a960013eb24216a65a79b6996e1ade015a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/63251280953e148b3ad0152398d434a960013eb24216a65a79b6996e1ade015a.jpg)

![6c690c432c26576dedc594c870f76028d19362af8ed24919d984d205d11b727d.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/6c690c432c26576dedc594c870f76028d19362af8ed24919d984d205d11b727d.jpg)

![7006b5d0304540f435946d24b14eec1b9ba59d322ad040501fe0bc3756b68323.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/7006b5d0304540f435946d24b14eec1b9ba59d322ad040501fe0bc3756b68323.jpg)

![79f1da5bf9abeb66681605912d81e1876327b7eba8900ab563dfdbd4c3feb72a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/79f1da5bf9abeb66681605912d81e1876327b7eba8900ab563dfdbd4c3feb72a.jpg)

![86aafbc1071702d126983b639d14625f3990e4ce60ca187373f34e0166e1b695.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/86aafbc1071702d126983b639d14625f3990e4ce60ca187373f34e0166e1b695.jpg)

![87c0d7aa2a0db13970da92653789620d7c0abac33efea380bc4e2ad7a30aaa5a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/87c0d7aa2a0db13970da92653789620d7c0abac33efea380bc4e2ad7a30aaa5a.jpg)

![901afc93ab54aefe0fc48088bdabf7b54d045da4d6f271ca09922a5748dc92de.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/901afc93ab54aefe0fc48088bdabf7b54d045da4d6f271ca09922a5748dc92de.jpg)

![912887fd7b95f0589a4e5a51081f790fae4071fff28d6e3227048d875404b4c4.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/912887fd7b95f0589a4e5a51081f790fae4071fff28d6e3227048d875404b4c4.jpg)

![9758071d8f22807b008d03d2791eb7bc4834f0b4c0a560ea2cdd55e9aaccf1bd.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/9758071d8f22807b008d03d2791eb7bc4834f0b4c0a560ea2cdd55e9aaccf1bd.jpg)

![9fdee3966af17c7aa08beed8915912f214dfb66c6ba69dd3425c4e1560a8fc30.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/9fdee3966af17c7aa08beed8915912f214dfb66c6ba69dd3425c4e1560a8fc30.jpg)

![aa3d9653396bd5cf0ee69b5969c7987b12e9a1c0a1ca0788bca7280bf2237af0.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/aa3d9653396bd5cf0ee69b5969c7987b12e9a1c0a1ca0788bca7280bf2237af0.jpg)

![ad84b42c065643ff8b56182a14d96f13d4c9770507bda6b4b365c922c2430205.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/ad84b42c065643ff8b56182a14d96f13d4c9770507bda6b4b365c922c2430205.jpg)

![b0eac5e43790e885526a95a7f9e070ee4aac36a477d3861d50eb9c1d9f6034e6.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/b0eac5e43790e885526a95a7f9e070ee4aac36a477d3861d50eb9c1d9f6034e6.jpg)

![bfd5b4bce2a08be081c55c72aa1e6e4063162891b0237347412c5ee15f03a04d.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/bfd5b4bce2a08be081c55c72aa1e6e4063162891b0237347412c5ee15f03a04d.jpg)

![c157a4d5ceffcc13653e32a4f515fbc9751626ae1510301ac67ae5b385871021.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/c157a4d5ceffcc13653e32a4f515fbc9751626ae1510301ac67ae5b385871021.jpg)

![ce2cc2ec376730797f3d4bf1b4196c1986cafac571643d0ed60052d9002cb24f.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/ce2cc2ec376730797f3d4bf1b4196c1986cafac571643d0ed60052d9002cb24f.jpg)

![d4c41c5d1d081593f2a7b884771d14e6030678b0354d3500e0428d999c9bd808.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/d4c41c5d1d081593f2a7b884771d14e6030678b0354d3500e0428d999c9bd808.jpg)

![da1e918978e7fcca9f198fd2bc1c4caeb09c072b66f054fbb72b545b82f0fd2a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/da1e918978e7fcca9f198fd2bc1c4caeb09c072b66f054fbb72b545b82f0fd2a.jpg)

![dbeb4bb51fbdc42a50cd5d79ccc0f952f568903a17afb1202f7525f379e2ee5f.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/dbeb4bb51fbdc42a50cd5d79ccc0f952f568903a17afb1202f7525f379e2ee5f.jpg)

![e8af1476d0846088ed4827b1167c3be7a32e1885c147a1650976f33eed524dd8.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/e8af1476d0846088ed4827b1167c3be7a32e1885c147a1650976f33eed524dd8.jpg)

![f9fe98411f11f896418522f03e9c48340f1c4bd3186757d07eb06c002f02d794.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/images/f9fe98411f11f896418522f03e9c48340f1c4bd3186757d07eb06c002f02d794.jpg)

### Tables

![07bbd9dafec4b3b37bd89c0e2d960657a3976593611768e2bd9d69911302c54e.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/07bbd9dafec4b3b37bd89c0e2d960657a3976593611768e2bd9d69911302c54e.jpg)

![0b69d8a62e61f298813b562193c213a4f56c6377b1dcb8ee9209437a4341a8b6.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/0b69d8a62e61f298813b562193c213a4f56c6377b1dcb8ee9209437a4341a8b6.jpg)

![0efae297b5e565498616e091c2ad753d7acdfa3a3757c89a04a7985916b926de.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/0efae297b5e565498616e091c2ad753d7acdfa3a3757c89a04a7985916b926de.jpg)

![12baabbcf4dbb21bf8a1a76c6c0d541f8d67410cd0d529dd63cf513945f6fd7a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/12baabbcf4dbb21bf8a1a76c6c0d541f8d67410cd0d529dd63cf513945f6fd7a.jpg)

![140e74cf8a159a0f61305a15c1d7e8ca6dac0e39decc577602af2ec81a6a8bed.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/140e74cf8a159a0f61305a15c1d7e8ca6dac0e39decc577602af2ec81a6a8bed.jpg)

![15b9899bffbdbe7f0db5920bd608243d3b3a47cf7ebc79a04b2b80711c30215a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/15b9899bffbdbe7f0db5920bd608243d3b3a47cf7ebc79a04b2b80711c30215a.jpg)

![16adb8091d18655ab8d866c6aaaf8ff074574e2135bda74faea66828e1075eda.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/16adb8091d18655ab8d866c6aaaf8ff074574e2135bda74faea66828e1075eda.jpg)

![1be9f47116df17a056d723a9385eaedad0f29796f61002c2bc38ec871956975c.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/1be9f47116df17a056d723a9385eaedad0f29796f61002c2bc38ec871956975c.jpg)

![1e588a4dd64253a646315091402899c503aae5b891ad638b21a10250ea645d6c.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/1e588a4dd64253a646315091402899c503aae5b891ad638b21a10250ea645d6c.jpg)

![2588d656ea764da3e3381ee4df3fd60508a0b51e9e6f67d10ff207a5074b7342.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/2588d656ea764da3e3381ee4df3fd60508a0b51e9e6f67d10ff207a5074b7342.jpg)

![2ef5f00dcc5a2414f88c7bfd65b9f396c88069bd64b79684344e665eebbf0814.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/2ef5f00dcc5a2414f88c7bfd65b9f396c88069bd64b79684344e665eebbf0814.jpg)

![31ad9fd6f4ca694215cbd17b8a7f59c6afce6db75ea7db7efaa69b6ddca275e3.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/31ad9fd6f4ca694215cbd17b8a7f59c6afce6db75ea7db7efaa69b6ddca275e3.jpg)

![3f803a08a3901ba03bfdd25d448481427b446bbfde28c62b1eb7c356090468b2.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/3f803a08a3901ba03bfdd25d448481427b446bbfde28c62b1eb7c356090468b2.jpg)

![463308ab30e39a3d46fae5a7951c59599dd0209083dbd89a1a6981ee57f990d2.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/463308ab30e39a3d46fae5a7951c59599dd0209083dbd89a1a6981ee57f990d2.jpg)

![469894c83a15dc260a7d1a0689786723a7918c250ff3ebc6c344da35e53adbbb.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/469894c83a15dc260a7d1a0689786723a7918c250ff3ebc6c344da35e53adbbb.jpg)

![498479b093ae653cd14013d0a8dcddd2115e36cb2317cedcf3a9feaa64a71dda.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/498479b093ae653cd14013d0a8dcddd2115e36cb2317cedcf3a9feaa64a71dda.jpg)

![4c8af659934ccf48e76bf21e28bb56674a0e0b45198042b64ddfa3761021fe6c.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/4c8af659934ccf48e76bf21e28bb56674a0e0b45198042b64ddfa3761021fe6c.jpg)

![57efb2b8dbf962c5036ed72b38dbcd0a69896d9dd384af6552a74d196beabd6a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/57efb2b8dbf962c5036ed72b38dbcd0a69896d9dd384af6552a74d196beabd6a.jpg)

![591448a359bd7f4568e257b68f7d8768f3bafeca6779e3adc553de209131a639.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/591448a359bd7f4568e257b68f7d8768f3bafeca6779e3adc553de209131a639.jpg)

![5a3af201f723eb2bbc5025414896441e9258cd2d6b04e79b42f005d71a916839.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/5a3af201f723eb2bbc5025414896441e9258cd2d6b04e79b42f005d71a916839.jpg)

![5eb7c88d24a76024f5dbfb96ecf4fd8afcbd5f5fb20f9ab5018d62436168e709.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/5eb7c88d24a76024f5dbfb96ecf4fd8afcbd5f5fb20f9ab5018d62436168e709.jpg)

![5f3c7844bcfc3be11250ed2cde256b1c83aacda24e934879b1fd1c51743aa7be.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/5f3c7844bcfc3be11250ed2cde256b1c83aacda24e934879b1fd1c51743aa7be.jpg)

![61cb4e2df9ce0eb8a59bd647c21a499e8f6a62be0b3b7d91e40d38ab671ddf78.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/61cb4e2df9ce0eb8a59bd647c21a499e8f6a62be0b3b7d91e40d38ab671ddf78.jpg)

![664486a3b4f18182f434fa2eee6ed80e1255be1991374f1fbebad4045cc17e9c.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/664486a3b4f18182f434fa2eee6ed80e1255be1991374f1fbebad4045cc17e9c.jpg)

![66c7fb29d036393d3f146082e3356cb2e3f106b5e5a88f47778009c1105fd8d9.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/66c7fb29d036393d3f146082e3356cb2e3f106b5e5a88f47778009c1105fd8d9.jpg)

![69e33f8a256d87f30c97f01f04650b36cdd135e1b52a3b8beeed5623ee23b5d1.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/69e33f8a256d87f30c97f01f04650b36cdd135e1b52a3b8beeed5623ee23b5d1.jpg)

![7a22dc495192ccb4eef1d7d4d71827cc6e02ba5eb17db8b3a0b3796376724380.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/7a22dc495192ccb4eef1d7d4d71827cc6e02ba5eb17db8b3a0b3796376724380.jpg)

![89673a93873b1d100888339bd5fc3134ac240228e8a06be2cbe87564848a1dcd.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/89673a93873b1d100888339bd5fc3134ac240228e8a06be2cbe87564848a1dcd.jpg)

![8bc5fe95103fdf9b88bd5d23109b95ce508e0461d50c44a92d7e911a5d4c8230.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/8bc5fe95103fdf9b88bd5d23109b95ce508e0461d50c44a92d7e911a5d4c8230.jpg)

![9089fce8b181b452b53b2a3d4958f15749c26f5f32b8bbd7013ed0ee55b6663b.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/9089fce8b181b452b53b2a3d4958f15749c26f5f32b8bbd7013ed0ee55b6663b.jpg)

![916f521dd3db12e39ea7dbed69178b369e6693127c0c7361419689075756da0d.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/916f521dd3db12e39ea7dbed69178b369e6693127c0c7361419689075756da0d.jpg)

![91dee38d9f343422b7750a5510a220d8de58e84939d8dde3303d6a6a7514a583.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/91dee38d9f343422b7750a5510a220d8de58e84939d8dde3303d6a6a7514a583.jpg)

![91f37f18bf14235950cd4834b8c232288beeacb6c1672bcefd6494feb7a4df14.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/91f37f18bf14235950cd4834b8c232288beeacb6c1672bcefd6494feb7a4df14.jpg)

![ab5cff861973d48f538b7397340f277ecbcd13007d40b7352c1730faf943ee81.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/ab5cff861973d48f538b7397340f277ecbcd13007d40b7352c1730faf943ee81.jpg)

![c84593f5f65e3fb5e574dab06a1a65d1225517abd9126277aa344addbd864e92.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/c84593f5f65e3fb5e574dab06a1a65d1225517abd9126277aa344addbd864e92.jpg)

![cc974e28a66cd4850e7ff1da18d8d07954eb8d0ce991add069fd7f28bee2c157.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/cc974e28a66cd4850e7ff1da18d8d07954eb8d0ce991add069fd7f28bee2c157.jpg)

![d6f8aecef6221ff81b7a8f4c657f368c6df1de5ee3c9475a76e5577577921b99.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/d6f8aecef6221ff81b7a8f4c657f368c6df1de5ee3c9475a76e5577577921b99.jpg)

![d9a4cdccbb49a6618ca653868e3aa3914370f8fd309c3b73cea7e2073885a412.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/d9a4cdccbb49a6618ca653868e3aa3914370f8fd309c3b73cea7e2073885a412.jpg)

![e6a3238903cdb86cd4271d00369d84cb6071829e11368eacdbaefe9e13177f0a.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/e6a3238903cdb86cd4271d00369d84cb6071829e11368eacdbaefe9e13177f0a.jpg)

![ec6a063a239f348ea7d5b7726f49732e3c908a8ffb1e8fa64774b78765824bbe.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/ec6a063a239f348ea7d5b7726f49732e3c908a8ffb1e8fa64774b78765824bbe.jpg)

![f19c8fe228b718361c9e88424bc25605bdb19dcd45156ed293644171054c6c6f.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/f19c8fe228b718361c9e88424bc25605bdb19dcd45156ed293644171054c6c6f.jpg)

![f27f7cb20cf2e20cc22df60c1dbe1daf1b1b486c49b0039356ea912fd55ac49c.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/f27f7cb20cf2e20cc22df60c1dbe1daf1b1b486c49b0039356ea912fd55ac49c.jpg)

![f9d07b98c948e5d05674062c87736a0b3ff27c90f0c9a68b8e592d4ba902627e.jpg](../icml_results/5994_Verbalized%20Sampling_%20How%20to%20Mitigate%20Mode%20Collapse%20and%20Unlock%20LLM%20Diversity/tables/f9d07b98c948e5d05674062c87736a0b3ff27c90f0c9a68b8e592d4ba902627e.jpg)

<a id="paper-5946"></a>

## LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation


### Images

![1e46b9322f61cb5f5bb1b3a2f0fe5e9ca449f0ff3b9a3e36073658e7f9b918fe.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/1e46b9322f61cb5f5bb1b3a2f0fe5e9ca449f0ff3b9a3e36073658e7f9b918fe.jpg)

![2fc99fa99aad71f404d7cd764460e04ed0958ad3aac6d5f25d45261cd0bd6b15.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/2fc99fa99aad71f404d7cd764460e04ed0958ad3aac6d5f25d45261cd0bd6b15.jpg)

![76aab5a40ec8ae4c0cff9d5d9feb3fe8b947c87086f0e6bd200a2fbe92e09002.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/76aab5a40ec8ae4c0cff9d5d9feb3fe8b947c87086f0e6bd200a2fbe92e09002.jpg)

![a2d167603e6312b70ac481f8ed807d5cd2afc30b9a6f797ae665ea4743d8a076.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/a2d167603e6312b70ac481f8ed807d5cd2afc30b9a6f797ae665ea4743d8a076.jpg)

![a3848bea575c73825173c0c67f8bd080f1ed2c4ddd3ebe1364486b65c5eb4250.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/a3848bea575c73825173c0c67f8bd080f1ed2c4ddd3ebe1364486b65c5eb4250.jpg)

![a974773a3fa45bb12d195998cae9b29ed889bf4b7af664dcc2481a24bbba6513.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/a974773a3fa45bb12d195998cae9b29ed889bf4b7af664dcc2481a24bbba6513.jpg)

![b2e457eda077660bce8ef1caaf61bc8b6d6d47c069a2d947042c958fb2442e22.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/b2e457eda077660bce8ef1caaf61bc8b6d6d47c069a2d947042c958fb2442e22.jpg)

![d1840ad104016796e4fed73e37ac73a5e5e036535d4bf8694aa2ae55922882aa.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/d1840ad104016796e4fed73e37ac73a5e5e036535d4bf8694aa2ae55922882aa.jpg)

![d97fae08f38c566d6996162567376653c5e58fbabe928a5eb61c72d334692e07.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/d97fae08f38c566d6996162567376653c5e58fbabe928a5eb61c72d334692e07.jpg)

![de84c35485dc33fb6bc4a035e900a2111d4e52ac93ba0969195e98f0e7dcc755.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/images/de84c35485dc33fb6bc4a035e900a2111d4e52ac93ba0969195e98f0e7dcc755.jpg)

### Tables

![1e41f824c8d786520ff2fd894538d42d0e3ff5d7184b784936fd8d0c795dcf9f.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/1e41f824c8d786520ff2fd894538d42d0e3ff5d7184b784936fd8d0c795dcf9f.jpg)

![311ed83810ca78ce87cb945a0b233da6c14b6d522b0ed6463adf0df0905edb82.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/311ed83810ca78ce87cb945a0b233da6c14b6d522b0ed6463adf0df0905edb82.jpg)

![41401d75ff1cf79ca86a41c23bf08ed098cd43fe6f7ae0c37c97c89eb4b594b9.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/41401d75ff1cf79ca86a41c23bf08ed098cd43fe6f7ae0c37c97c89eb4b594b9.jpg)

![49d845da9a19c567e34f87f3282be868534ae0f06f299e65213e20687f770763.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/49d845da9a19c567e34f87f3282be868534ae0f06f299e65213e20687f770763.jpg)

![6df58e0d372b4de0c33ed1422b1e1074e343179552e25b55cd83467b36d13fdb.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/6df58e0d372b4de0c33ed1422b1e1074e343179552e25b55cd83467b36d13fdb.jpg)

![9419e6d04ea9330c4d3f013c071ddb55c79afc051f78fa33ef219c98c1cefa36.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/9419e6d04ea9330c4d3f013c071ddb55c79afc051f78fa33ef219c98c1cefa36.jpg)

![d81246a8da2829cad2025ac91e24e9369c135c98d00079038201acb0207b4f45.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/d81246a8da2829cad2025ac91e24e9369c135c98d00079038201acb0207b4f45.jpg)

![e26afbda72168093fea3b8a688b735f5df09145a0f8fca8900ab0141cfb2b636.jpg](../icml_results/5995_LLM4Cov_%20Execution-Aware%20Agentic%20Learning%20for%20High-coverage%20Testbench%20Generation/tables/e26afbda72168093fea3b8a688b735f5df09145a0f8fca8900ab0141cfb2b636.jpg)

<a id="paper-5947"></a>

## Reasoning-VLA: An Efficient and Spatial-Guided General Vision-Language-Action Reasoning Model for Autonomous Driving


### Images

![0de82ef09a51a5801f48c84c6d09a98bf797e23ed186774cb9c57a3055ccd244.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/images/0de82ef09a51a5801f48c84c6d09a98bf797e23ed186774cb9c57a3055ccd244.jpg)

![1c9503699b6bb7d8d0f9ee3f35e1a7bf8ccabd449bfaacc9f9f758795cd5518e.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/images/1c9503699b6bb7d8d0f9ee3f35e1a7bf8ccabd449bfaacc9f9f758795cd5518e.jpg)

![23cf94ab852211a415a802ac7b1e4e96b9470b66a8112d813b7928b6dc2a838d.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/images/23cf94ab852211a415a802ac7b1e4e96b9470b66a8112d813b7928b6dc2a838d.jpg)

![7994b7e62312326e52fb265631456bc382638455e4aa1c3955fd3918d3e6c79e.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/images/7994b7e62312326e52fb265631456bc382638455e4aa1c3955fd3918d3e6c79e.jpg)

![8cbf4d193beaa94ed6110812a533236f19b4f5e3e220231bd92807a6053fc685.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/images/8cbf4d193beaa94ed6110812a533236f19b4f5e3e220231bd92807a6053fc685.jpg)

### Tables

![182d65a3863d1b9f9bb8e8404a54fc0dd4c046f857cc4f738d92882739a288ec.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/182d65a3863d1b9f9bb8e8404a54fc0dd4c046f857cc4f738d92882739a288ec.jpg)

![2a0750b33a93eaf42a399579fcdb091448831a2d1b4be8085829856f8ef83e1f.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/2a0750b33a93eaf42a399579fcdb091448831a2d1b4be8085829856f8ef83e1f.jpg)

![7bc7ca2b4f59a134969c25f48fe9890ccea95bf1b27f64522e9a7e7e898943a4.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/7bc7ca2b4f59a134969c25f48fe9890ccea95bf1b27f64522e9a7e7e898943a4.jpg)

![81a2d749aea93cbc93a8975cc8567c7b66b884857db1dae69ffd24696619d5f5.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/81a2d749aea93cbc93a8975cc8567c7b66b884857db1dae69ffd24696619d5f5.jpg)

![8b68f0028a839f2afc162390c56c4db6729ee0b054ec0a33515173e43560be8b.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/8b68f0028a839f2afc162390c56c4db6729ee0b054ec0a33515173e43560be8b.jpg)

![a365c23f546104495ecc66716067dc60e8f3880d645ef784714e846a526e1b2c.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/a365c23f546104495ecc66716067dc60e8f3880d645ef784714e846a526e1b2c.jpg)

![a5364955533228571df601eb895323501bcebe41e3dad389e7c9bd6f7a73651b.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/a5364955533228571df601eb895323501bcebe41e3dad389e7c9bd6f7a73651b.jpg)

![b99314e6a35e94a3558a091d491f74987c62b97f499af0994be4056f67c1d063.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/b99314e6a35e94a3558a091d491f74987c62b97f499af0994be4056f67c1d063.jpg)

![d7dd05a86485b3f806cb42cc0646eaa7e5392444421d27c308049e699f81ba0f.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/d7dd05a86485b3f806cb42cc0646eaa7e5392444421d27c308049e699f81ba0f.jpg)

![ebca6d6bd8244e2a3b83e836ddb58b59a37a802c9b4ae84681a01be64a5dee46.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/ebca6d6bd8244e2a3b83e836ddb58b59a37a802c9b4ae84681a01be64a5dee46.jpg)

![f211c464f5f857af5fd39a17e107310b9d740caf2ca29bf52ea99aab9a90b2bb.jpg](../icml_results/5996_Reasoning-VLA_%20An%20Efficient%20and%20Spatial-Guided%20General%20Vision-Language-Action%20Reasoning%20Model%20for%20Au/tables/f211c464f5f857af5fd39a17e107310b9d740caf2ca29bf52ea99aab9a90b2bb.jpg)

<a id="paper-5948"></a>

## Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory


### Images

![52d5d71786bc0ac5e4bcfb1f498fb61e6d120432463c6d6a648c9c4b15fa8e01.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/images/52d5d71786bc0ac5e4bcfb1f498fb61e6d120432463c6d6a648c9c4b15fa8e01.jpg)

![64bdb152f09fe331f87fe20aabf7d82afc183675e93fe32df9a9a57ac6f1a1cc.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/images/64bdb152f09fe331f87fe20aabf7d82afc183675e93fe32df9a9a57ac6f1a1cc.jpg)

![9ec1c383cfb1b72556622413b00dab4a923c7f55a0840333c008654b7cb7ee55.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/images/9ec1c383cfb1b72556622413b00dab4a923c7f55a0840333c008654b7cb7ee55.jpg)

![d4b00a4a63f772053cd502c67ee9df2a9b15704f018df56e204c94086ac916e1.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/images/d4b00a4a63f772053cd502c67ee9df2a9b15704f018df56e204c94086ac916e1.jpg)

![f8f4b86b4be9840d971373c725be19bf8575eee89c7fec74219fbe3408596cc3.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/images/f8f4b86b4be9840d971373c725be19bf8575eee89c7fec74219fbe3408596cc3.jpg)

### Tables

![48eff5c7938d67ed40ed2c5c84f5f17d0f79ee97ea72fa5f7406c8b3e6adeae0.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/48eff5c7938d67ed40ed2c5c84f5f17d0f79ee97ea72fa5f7406c8b3e6adeae0.jpg)

![56f7fd5958493571ef00a483dcd9550782a5b1bbf4b57ac7303ed37a33e674c2.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/56f7fd5958493571ef00a483dcd9550782a5b1bbf4b57ac7303ed37a33e674c2.jpg)

![63092455ce4b783332914ecb19afbd78b8b2780d0c1d27d11f4973c7d58832cb.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/63092455ce4b783332914ecb19afbd78b8b2780d0c1d27d11f4973c7d58832cb.jpg)

![7a55a19c293a233b1ad10eefa042004f0bf0f004779e9a16b6b3766d4cf313ac.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/7a55a19c293a233b1ad10eefa042004f0bf0f004779e9a16b6b3766d4cf313ac.jpg)

![802885d51d9fb1ac1df7c9454c1169b32961e6b076f43fd2881771bbd42b2979.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/802885d51d9fb1ac1df7c9454c1169b32961e6b076f43fd2881771bbd42b2979.jpg)

![87011aa51b6bb98370b07c2fd18e705bff5823ce1e34068f49dbcf928df2a6a4.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/87011aa51b6bb98370b07c2fd18e705bff5823ce1e34068f49dbcf928df2a6a4.jpg)

![a5bd31c128ef3f7deb643cdba43e2fbae60c48ea76ce684edeeb6b24830343dc.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/a5bd31c128ef3f7deb643cdba43e2fbae60c48ea76ce684edeeb6b24830343dc.jpg)

![b106c1f49898bcf5d0b98f85df3cbebdaf31fd3b59269f194ccc83203e45c02a.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/b106c1f49898bcf5d0b98f85df3cbebdaf31fd3b59269f194ccc83203e45c02a.jpg)

![b4d81f6babc591b1689b753877c74331c1011d067e8933a58d1d6f84d62e3fe5.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/b4d81f6babc591b1689b753877c74331c1011d067e8933a58d1d6f84d62e3fe5.jpg)

![c4798c1c87ddd196f9035b2701677ba6577abc92b85f356eef1fe2b32b8a9513.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/c4798c1c87ddd196f9035b2701677ba6577abc92b85f356eef1fe2b32b8a9513.jpg)

![cd4d0409d0634c805cadd70042fa52777259c1a6f2f9f73b56e3c5e88d02cf5d.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/cd4d0409d0634c805cadd70042fa52777259c1a6f2f9f73b56e3c5e88d02cf5d.jpg)

![eeb934db2560340c55e6553c696daa90dcaa13678f2b83e854266ba4859d087e.jpg](../icml_results/5997_Learning%20Query-Aware%20Budget-Tier%20Routing%20for%20Runtime%20Agent%20Memory/tables/eeb934db2560340c55e6553c696daa90dcaa13678f2b83e854266ba4859d087e.jpg)

<a id="paper-5949"></a>

## Posterior Mismatch Matters: Adversarial Training for Long-Tailed Robustness


### Images

![12a76015ea60ab0f8ee46cdc45b3ba8917e8d8008cdadf82ab54a94c48d219b0.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/12a76015ea60ab0f8ee46cdc45b3ba8917e8d8008cdadf82ab54a94c48d219b0.jpg)

![2eef04c758aa30dd8fd84260fbcc2ab3c60b6261cd6be8b960b243f90ce551c0.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/2eef04c758aa30dd8fd84260fbcc2ab3c60b6261cd6be8b960b243f90ce551c0.jpg)

![45cf53547a8517fdcd6de000392fe3c5e902db9ea94f691680af12c547abe961.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/45cf53547a8517fdcd6de000392fe3c5e902db9ea94f691680af12c547abe961.jpg)

![9f084a1bfefdd800f86d6a4aa1d6d2f654b6d98968ed34505c15d517d73836f5.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/9f084a1bfefdd800f86d6a4aa1d6d2f654b6d98968ed34505c15d517d73836f5.jpg)

![b2b0002722ef52d82144e8717bf39b66c313b8b73f975187e9999d49188302d9.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/b2b0002722ef52d82144e8717bf39b66c313b8b73f975187e9999d49188302d9.jpg)

![b8fb3385617ea79d015d285e991a6e6155d2f6663aa2850687c7880b083f444e.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/b8fb3385617ea79d015d285e991a6e6155d2f6663aa2850687c7880b083f444e.jpg)

![bdb7c2703aa79932fc1c46b0bea5375024eedac02b4787e2ae1c59e77f0446b9.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/bdb7c2703aa79932fc1c46b0bea5375024eedac02b4787e2ae1c59e77f0446b9.jpg)

![c320034479337552aef2be5bcd4e334a0d4f7720655710b3e5be05beaabf0b6e.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/c320034479337552aef2be5bcd4e334a0d4f7720655710b3e5be05beaabf0b6e.jpg)

![e7ebe33efbe198c948fc3c3757909c0b0159cfa0157250f15ea8ae1da1ca0aaf.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/e7ebe33efbe198c948fc3c3757909c0b0159cfa0157250f15ea8ae1da1ca0aaf.jpg)

![f8745df4ee938eca8451c13dcb79474b21c1e07a157b8e719b2065fbf707b5f1.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/images/f8745df4ee938eca8451c13dcb79474b21c1e07a157b8e719b2065fbf707b5f1.jpg)

### Tables

![0a49339add6a54bc2054a2599f8ca62e3d4d63bf8a1c94959d10a9257cb3cb7e.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/0a49339add6a54bc2054a2599f8ca62e3d4d63bf8a1c94959d10a9257cb3cb7e.jpg)

![1431afd14d4193410321904b083136e5eb830f3bbb4ec625863a6baf6a32d87b.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/1431afd14d4193410321904b083136e5eb830f3bbb4ec625863a6baf6a32d87b.jpg)

![4a8281924c28eb66690bc09b7fd14733312972ef417fc676e722bf7e9e4bb13f.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/4a8281924c28eb66690bc09b7fd14733312972ef417fc676e722bf7e9e4bb13f.jpg)

![4df6108e6baca980300a7b70b8626b7e10992c316764011d7862a78e858e249f.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/4df6108e6baca980300a7b70b8626b7e10992c316764011d7862a78e858e249f.jpg)

![8f9eecf9ed8ca1d2277f0fef615784e55637eef995c5c928f85b490ad5342b28.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/8f9eecf9ed8ca1d2277f0fef615784e55637eef995c5c928f85b490ad5342b28.jpg)

![f0869892d8a4c72b1badbeee06b6d7d9d9f783aa0ab895e36ea04d937370e8d2.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/f0869892d8a4c72b1badbeee06b6d7d9d9f783aa0ab895e36ea04d937370e8d2.jpg)

![f758ed7a3c02bd9c9ec832e14300fe05beeef17b9f7336cd6e5ad5d3d44e5024.jpg](../icml_results/5998_Posterior%20Mismatch%20Matters_%20Adversarial%20Training%20for%20Long-Tailed%20Robustness/tables/f758ed7a3c02bd9c9ec832e14300fe05beeef17b9f7336cd6e5ad5d3d44e5024.jpg)

<a id="paper-5950"></a>

## Personalized Policy Learning through Discrete Experimentation


### Images

![472ee8d97d7a26c0b6da20ff0112d37a34efdf670cdcb417b349f18b5c3950ce.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/images/472ee8d97d7a26c0b6da20ff0112d37a34efdf670cdcb417b349f18b5c3950ce.jpg)

![aa70aebd852e422458e77c7d7be1110797a9c257dd6d42edae37c2ad1b14ec54.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/images/aa70aebd852e422458e77c7d7be1110797a9c257dd6d42edae37c2ad1b14ec54.jpg)

![b0e647b1f86e5a1fc34979ec3d38c4433d38615fd3bd91cd1a8118f2ef2feff2.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/images/b0e647b1f86e5a1fc34979ec3d38c4433d38615fd3bd91cd1a8118f2ef2feff2.jpg)

### Tables

![5785fb61fe6c0800844f164d889cf9409272d3722d0fba7d2e6522732d6ec069.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/5785fb61fe6c0800844f164d889cf9409272d3722d0fba7d2e6522732d6ec069.jpg)

![820f73607e8188f4d9c9cce6fe71d9bd7a26e27f6ae7db5e9315a90d75afa990.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/820f73607e8188f4d9c9cce6fe71d9bd7a26e27f6ae7db5e9315a90d75afa990.jpg)

![923cfbfa3c66823dd9636abf7c6b6844c4ef38b9f854488a064fb2ef170cf97b.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/923cfbfa3c66823dd9636abf7c6b6844c4ef38b9f854488a064fb2ef170cf97b.jpg)

![ba951c82e4c8188aee9194228e6222f1cdbd84cb12adb058a7b732f064b5ded9.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/ba951c82e4c8188aee9194228e6222f1cdbd84cb12adb058a7b732f064b5ded9.jpg)

![c13c192a2e26295880e7d3cf5626a2806c9df5dbdc4e6c271d79dcb026771f31.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/c13c192a2e26295880e7d3cf5626a2806c9df5dbdc4e6c271d79dcb026771f31.jpg)

![c9ec885ab112b2d808cc5319861c02ef5f8668fdad88b4da5ac2a6692aa5650f.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/c9ec885ab112b2d808cc5319861c02ef5f8668fdad88b4da5ac2a6692aa5650f.jpg)

![d190153244cf2bbc468492b6e8905f8c58d9e7a8bf1abfcf1ac1a77a8ae5e1fe.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/d190153244cf2bbc468492b6e8905f8c58d9e7a8bf1abfcf1ac1a77a8ae5e1fe.jpg)

![d31ca827d5cadcfdc3c28db93413b41caf85e2ce0bcc079cce8bb829d4dd1e38.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/d31ca827d5cadcfdc3c28db93413b41caf85e2ce0bcc079cce8bb829d4dd1e38.jpg)

![dba52a0cb91bd4c101522e103f8ba2ec5647ce7a9dd39c2b1eb38202dd5ee994.jpg](../icml_results/5999_Personalized%20Policy%20Learning%20through%20Discrete%20Experimentation/tables/dba52a0cb91bd4c101522e103f8ba2ec5647ce7a9dd39c2b1eb38202dd5ee994.jpg)

<a id="paper-5951"></a>

## A Generalist Pair-wise Progress Critic Model for Vision-Language-Action Robots


### Images

![7197ba30449a5c7836fe54b7743ad4caac063ddde3a171d8d9fbe6b2a7e08105.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/images/7197ba30449a5c7836fe54b7743ad4caac063ddde3a171d8d9fbe6b2a7e08105.jpg)

![b81a0c1e3248295f72a2f357eb6c763129c1d52059435411066dd2f03499d2a1.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/images/b81a0c1e3248295f72a2f357eb6c763129c1d52059435411066dd2f03499d2a1.jpg)

![be8ad1405c46f183248ffb283bc8f26ce67c646c7db63857f75659b4462b4588.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/images/be8ad1405c46f183248ffb283bc8f26ce67c646c7db63857f75659b4462b4588.jpg)

![c13f0eb9f217218f733131c65176a4362bf92604b19bcbf8019905732c866e1b.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/images/c13f0eb9f217218f733131c65176a4362bf92604b19bcbf8019905732c866e1b.jpg)

![e65d4fed9e0b3d01ee32c45601a3870b2afb61cd2fc62972efdc9ab49c59aec1.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/images/e65d4fed9e0b3d01ee32c45601a3870b2afb61cd2fc62972efdc9ab49c59aec1.jpg)

![f17ae1aecc54902134de030345258e6a42664940487295f47dfc44649f97cfb1.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/images/f17ae1aecc54902134de030345258e6a42664940487295f47dfc44649f97cfb1.jpg)

### Tables

![06d987caf60adc8dd55e841ea5042f427a807cfaf18cd98e11e6ab92f1882810.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/tables/06d987caf60adc8dd55e841ea5042f427a807cfaf18cd98e11e6ab92f1882810.jpg)

![3913589ebc7f86ca15f03acb6a6b38a4d0fec027801cf693783f42f29a975c5c.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/tables/3913589ebc7f86ca15f03acb6a6b38a4d0fec027801cf693783f42f29a975c5c.jpg)

![6ba5423b204153395d57aae169d801fb29b95cae3ca28862fc38e1c667c1b380.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/tables/6ba5423b204153395d57aae169d801fb29b95cae3ca28862fc38e1c667c1b380.jpg)

![92356fe813a50d2b03f016ef0cfa079c1ec9ee06bb8137929c2fc799d84e9582.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/tables/92356fe813a50d2b03f016ef0cfa079c1ec9ee06bb8137929c2fc799d84e9582.jpg)

![9e3d6ea371b90b96b469b95beb1be6464a2ef622753367d87c6f39e988df82c7.jpg](../icml_results/6000_A%20Generalist%20Pair-wise%20Progress%20Critic%20Model%20for%20Vision-Language-Action%20Robots/tables/9e3d6ea371b90b96b469b95beb1be6464a2ef622753367d87c6f39e988df82c7.jpg)

<a id="paper-5952"></a>

## MAGIC: Multi-Granularity Language-Informed Image Clustering


### Images

![17f3f3d4034827d94be73fedb3b23fe264eca80a9548211126cf21db9f06baec.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/17f3f3d4034827d94be73fedb3b23fe264eca80a9548211126cf21db9f06baec.jpg)

![7fa0545460dbe055d274315451c66127e915f87eca9c3424eaa876ba9bebc5fa.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/7fa0545460dbe055d274315451c66127e915f87eca9c3424eaa876ba9bebc5fa.jpg)

![88593ebec75a8d56d8a3cfd9a781ebfbe7211907666b9866dafc997b1ba3d23d.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/88593ebec75a8d56d8a3cfd9a781ebfbe7211907666b9866dafc997b1ba3d23d.jpg)

![9b0a2cdd35ad779165c377b8989cb1e8fc2c3160e81f92ee1a77d1d745acdd83.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/9b0a2cdd35ad779165c377b8989cb1e8fc2c3160e81f92ee1a77d1d745acdd83.jpg)

![c3b9126fdf9b2184141143bf184752bf1392e9164d4bc30beb0ea2e7fad8e782.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/c3b9126fdf9b2184141143bf184752bf1392e9164d4bc30beb0ea2e7fad8e782.jpg)

![e075cb27332e9f4a9e955a7be9b35cfbdcf427cf7e041846d4d790a435405856.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/e075cb27332e9f4a9e955a7be9b35cfbdcf427cf7e041846d4d790a435405856.jpg)

![e3ca4502e555cadddf645d2cc893e50c19eb6ffd5abf3f799bda7e9b0b030949.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/images/e3ca4502e555cadddf645d2cc893e50c19eb6ffd5abf3f799bda7e9b0b030949.jpg)

### Tables

![2967acdf1bdeec165c91e869f0aaa0d97d9304910d42a643dfa3cdf810f8ec12.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/2967acdf1bdeec165c91e869f0aaa0d97d9304910d42a643dfa3cdf810f8ec12.jpg)

![4476a591c2088e7f19bc371d0dec85dcaf46394a9167db8df0341808bf1f7964.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/4476a591c2088e7f19bc371d0dec85dcaf46394a9167db8df0341808bf1f7964.jpg)

![48db31c62449d373e07ac3fb28f2fcd46786c29b5f19aabd6a46ffa8d9c51826.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/48db31c62449d373e07ac3fb28f2fcd46786c29b5f19aabd6a46ffa8d9c51826.jpg)

![4b2772e7e98d11142780356ab40ca83e0c979e1fd241e2489e235c7af513ea8c.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/4b2772e7e98d11142780356ab40ca83e0c979e1fd241e2489e235c7af513ea8c.jpg)

![78512045cb1b04445da215aac348f9cad97f09b8569724962a21cc72b2c4a358.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/78512045cb1b04445da215aac348f9cad97f09b8569724962a21cc72b2c4a358.jpg)

![93b23bb53ee29819d07d6e17c7d5eccbc6866e603e8f88a55bbda8000a79a00e.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/93b23bb53ee29819d07d6e17c7d5eccbc6866e603e8f88a55bbda8000a79a00e.jpg)

![be2f6cb09ec77f2fa923cbe0cfc9c2f2a4d4b2de19dae4cae52ff3b70980189c.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/be2f6cb09ec77f2fa923cbe0cfc9c2f2a4d4b2de19dae4cae52ff3b70980189c.jpg)

![c1773694b0887c00a4da273d4367c2e961aae328921590c182dd95c0a8b17a17.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/c1773694b0887c00a4da273d4367c2e961aae328921590c182dd95c0a8b17a17.jpg)

![c32d40bf22c66f3768fb5c99be4041f584353340267dc698c42dcb790b4dfae1.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/c32d40bf22c66f3768fb5c99be4041f584353340267dc698c42dcb790b4dfae1.jpg)

![f3c6483942706306b470b5e02938daa223489d068150200568990286a850121d.jpg](../icml_results/6001_MAGIC_%20Multi-Granularity%20Language-Informed%20Image%20Clustering/tables/f3c6483942706306b470b5e02938daa223489d068150200568990286a850121d.jpg)

<a id="paper-5953"></a>

## Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation


### Images

![039cc8bcfde9ba3c6e164e0f5d16c2423c401875b0c58813b086eb3a831b72b9.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/039cc8bcfde9ba3c6e164e0f5d16c2423c401875b0c58813b086eb3a831b72b9.jpg)

![16bd41d979504fb2b0af43b16abad5b4ece976b14c8a0b2a2fd64aa3ebb948bd.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/16bd41d979504fb2b0af43b16abad5b4ece976b14c8a0b2a2fd64aa3ebb948bd.jpg)

![1a7de4597c09e2b76b3c51c2fca23a458091c159ea3cea4d90b4409d57a5f426.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/1a7de4597c09e2b76b3c51c2fca23a458091c159ea3cea4d90b4409d57a5f426.jpg)

![26e685194771f3bf7886dd05329fea8488cb3fd8bdbb005a2ad3f4b5537a8bb8.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/26e685194771f3bf7886dd05329fea8488cb3fd8bdbb005a2ad3f4b5537a8bb8.jpg)

![4c17bec32aaeda56b317dade2714e94e377815a7f24280511f718d448e76632e.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/4c17bec32aaeda56b317dade2714e94e377815a7f24280511f718d448e76632e.jpg)

![5012b29ca73f66e99cc3e296f8d2c6c994e5ad6b402feaddb2beb7db76478ff9.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/5012b29ca73f66e99cc3e296f8d2c6c994e5ad6b402feaddb2beb7db76478ff9.jpg)

![88e9cd4da967654e8bf4298c29da899eb3ff8ed9a74673f842ac67262e491d75.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/88e9cd4da967654e8bf4298c29da899eb3ff8ed9a74673f842ac67262e491d75.jpg)

![c1cbd21df3b48d0635f193b7c7ac334f07f54156cb981cafec8c40868de9a2cf.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/c1cbd21df3b48d0635f193b7c7ac334f07f54156cb981cafec8c40868de9a2cf.jpg)

![c1f939570eed8ecfbcc51fde7dc6a9bb4c37acb6d2b58440ac2e8d9199c8772c.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/c1f939570eed8ecfbcc51fde7dc6a9bb4c37acb6d2b58440ac2e8d9199c8772c.jpg)

![c4f20143c289c2fa57b449348b012711aadf60344fc1d3759d76cc5679a9bc41.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/c4f20143c289c2fa57b449348b012711aadf60344fc1d3759d76cc5679a9bc41.jpg)

![d58d7629af9e41a7df56d432601bf0b6bf1a8277af8e1be8ad216845a30ae1b7.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/images/d58d7629af9e41a7df56d432601bf0b6bf1a8277af8e1be8ad216845a30ae1b7.jpg)

### Tables

![0b6842075a41274db2d8f93b64d88158cebde1c99d2b02a0eca9835f92151541.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/0b6842075a41274db2d8f93b64d88158cebde1c99d2b02a0eca9835f92151541.jpg)

![18df7def2fbc1a7edbde750aee9c9bdb0e9ecc44534445f0752b1ae7e40c6447.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/18df7def2fbc1a7edbde750aee9c9bdb0e9ecc44534445f0752b1ae7e40c6447.jpg)

![3716de1aafac20beb4eeaa9fe2909268e9d0d9e8ed5dbe92648a093f6dd8b64d.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/3716de1aafac20beb4eeaa9fe2909268e9d0d9e8ed5dbe92648a093f6dd8b64d.jpg)

![3a2b919bdba5b17761b916a14f565184c9d6b0c2d8098ee19f02c6537f2b7f81.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/3a2b919bdba5b17761b916a14f565184c9d6b0c2d8098ee19f02c6537f2b7f81.jpg)

![40a1e10da9b9ac14b630a64405f43474b60756a8a4614ba109b83ae5d2e03194.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/40a1e10da9b9ac14b630a64405f43474b60756a8a4614ba109b83ae5d2e03194.jpg)

![6a83529d69d1dc500553b13eb5d09e8e22b2fd7eb159d2a8fd6935979bd4528d.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/6a83529d69d1dc500553b13eb5d09e8e22b2fd7eb159d2a8fd6935979bd4528d.jpg)

![a30f09dfc8804945ee29f8ed9c49c4950e6571cd312f1e5851f21bd508efc178.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/a30f09dfc8804945ee29f8ed9c49c4950e6571cd312f1e5851f21bd508efc178.jpg)

![aa310a280a3d0fe8dea322c534feeb9d09953e1a3773200c9bc76705a148430a.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/aa310a280a3d0fe8dea322c534feeb9d09953e1a3773200c9bc76705a148430a.jpg)

![e4dd552794081399bf4d41423e6d5e61aa21b1d869578372b409ffaf639c2f5c.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/e4dd552794081399bf4d41423e6d5e61aa21b1d869578372b409ffaf639c2f5c.jpg)

![e78aebb6502a6b6de53511a3d0b42489b9e009d7fd92e1b2bd3e155d923f6364.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/e78aebb6502a6b6de53511a3d0b42489b9e009d7fd92e1b2bd3e155d923f6364.jpg)

![ef5767be3c7f5b3dc04ea93204c5b584d39c35a7633147b8eafdcbd7807684e6.jpg](../icml_results/6002_Principled%20Synthetic%20Data%20Enables%20the%20First%20Scaling%20Laws%20for%20LLMs%20in%20Recommendation/tables/ef5767be3c7f5b3dc04ea93204c5b584d39c35a7633147b8eafdcbd7807684e6.jpg)

<a id="paper-5954"></a>

## PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization


### Images

![0266309cc4dfe573306d0789bf4036e18f41d65d8811db50d92c3fe73fa97500.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/0266309cc4dfe573306d0789bf4036e18f41d65d8811db50d92c3fe73fa97500.jpg)

![060aa13fe0d93729e647eb38505b4c69688f1ab43cc3943df7e1a56dd574239d.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/060aa13fe0d93729e647eb38505b4c69688f1ab43cc3943df7e1a56dd574239d.jpg)

![0e0d09169039f62aaca81ce8bf12a1d6f9f236a160b00c9818d07eb6165447ee.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/0e0d09169039f62aaca81ce8bf12a1d6f9f236a160b00c9818d07eb6165447ee.jpg)

![143e87b821f78484af18d1a815333fb3d358a4a1d27fb0f42665c50b5fe63e5d.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/143e87b821f78484af18d1a815333fb3d358a4a1d27fb0f42665c50b5fe63e5d.jpg)

![18f82273d8af7181e7e69d913e7c7ccb10535f645c7966dbe2773ad999c1a47f.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/18f82273d8af7181e7e69d913e7c7ccb10535f645c7966dbe2773ad999c1a47f.jpg)

![2747ca035f9ae6401a25a2a422b79438bbe7422c89742009807f6271e8d7f3a6.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/2747ca035f9ae6401a25a2a422b79438bbe7422c89742009807f6271e8d7f3a6.jpg)

![2e3e26d8d2399c1174accfe2075054a87f4b34d4e0ded99d17472b1af94a0d06.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/2e3e26d8d2399c1174accfe2075054a87f4b34d4e0ded99d17472b1af94a0d06.jpg)

![3973766480e52ac12d65672c4765781c34285d8e09ca312afe7df1dbf9ab281e.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/3973766480e52ac12d65672c4765781c34285d8e09ca312afe7df1dbf9ab281e.jpg)

![3ed23271080a0d678248249745de87c7494b3e4bafd60fe7cf14177d38b1db79.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/3ed23271080a0d678248249745de87c7494b3e4bafd60fe7cf14177d38b1db79.jpg)

![48fca9a70acfc55a94ee33b316475186f389a3629334e53ad75272d260fe40e2.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/48fca9a70acfc55a94ee33b316475186f389a3629334e53ad75272d260fe40e2.jpg)

![4f2bd0db4a5552835f99e26c1f1cfcfa8ce5b079dd5127f43dff00a69fb16d70.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/4f2bd0db4a5552835f99e26c1f1cfcfa8ce5b079dd5127f43dff00a69fb16d70.jpg)

![6321d1fa002f27aa753951e7eaea09b90b8f15daad191b30c1b869c6610eadb4.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/6321d1fa002f27aa753951e7eaea09b90b8f15daad191b30c1b869c6610eadb4.jpg)

![71c022c2ecf7b85d1bfdbeffa25a3cee101ac2dc228b472ec2c63363bc899337.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/71c022c2ecf7b85d1bfdbeffa25a3cee101ac2dc228b472ec2c63363bc899337.jpg)

![740e1e25a9fff317c9014160204005cfe6d21c497fdf7e075133b1196f04807c.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/740e1e25a9fff317c9014160204005cfe6d21c497fdf7e075133b1196f04807c.jpg)

![7f1a682cf6d9864d8b08f75331b12b95cd627905110088b98fe26cd8c3531f94.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/7f1a682cf6d9864d8b08f75331b12b95cd627905110088b98fe26cd8c3531f94.jpg)

![809b055a8b8b3fe3432e2b69405aa26c67a710df1d674b7f858e2a27324f9172.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/809b055a8b8b3fe3432e2b69405aa26c67a710df1d674b7f858e2a27324f9172.jpg)

![86de66f2befdd7ac740cb731ce75b402969e11718025ca5b1c5cad0bf9e74e57.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/86de66f2befdd7ac740cb731ce75b402969e11718025ca5b1c5cad0bf9e74e57.jpg)

![880e721a68fe78f0be75f492f54220e32ad4eb2550a65eb1b32872930c4062c0.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/880e721a68fe78f0be75f492f54220e32ad4eb2550a65eb1b32872930c4062c0.jpg)

![8e4a0d011bf4c903923ab51e295fe5da18a490a58daf6d0c107c067ef0358b32.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/8e4a0d011bf4c903923ab51e295fe5da18a490a58daf6d0c107c067ef0358b32.jpg)

![9dae2b31d7738ecf69874be8b04986d82a958c3abe084c32dd3adda6e750e185.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/9dae2b31d7738ecf69874be8b04986d82a958c3abe084c32dd3adda6e750e185.jpg)

![a6779de3d9429fa4ca0fec1efbb6ddb0130891549a6aa3d6c026e83308250bc3.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/a6779de3d9429fa4ca0fec1efbb6ddb0130891549a6aa3d6c026e83308250bc3.jpg)

![b35f6804e7262721dd56ffbcac3b1e5335f935da23d83214adb67b127f47c681.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/b35f6804e7262721dd56ffbcac3b1e5335f935da23d83214adb67b127f47c681.jpg)

![b48debb310b95db4b232c161126f9e6f728323ed3126895fe11eb04ced9ae354.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/b48debb310b95db4b232c161126f9e6f728323ed3126895fe11eb04ced9ae354.jpg)

![ba0b14c753ce673ad062ca5fc4b13072d9929aebea390f7c6eecf78cb19c3230.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/ba0b14c753ce673ad062ca5fc4b13072d9929aebea390f7c6eecf78cb19c3230.jpg)

![ba7d3c306e863c5cec294def233faa14d8ea6150eff67956531ac715e95ecfb3.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/ba7d3c306e863c5cec294def233faa14d8ea6150eff67956531ac715e95ecfb3.jpg)

![bf35375cc625e0306f3281c18d5a9b0af26a5faccd3f1bca4b901ca238dcf643.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/bf35375cc625e0306f3281c18d5a9b0af26a5faccd3f1bca4b901ca238dcf643.jpg)

![bfd3946bd3c8c0fa00f5a6632a7b5bd47c3e48b80eb56f27b150a630a0855439.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/bfd3946bd3c8c0fa00f5a6632a7b5bd47c3e48b80eb56f27b150a630a0855439.jpg)

![c17d881deb5b4bdfacdb0cec1fea4bf48240afa3fbf4b34d84b07904ce1c115e.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/c17d881deb5b4bdfacdb0cec1fea4bf48240afa3fbf4b34d84b07904ce1c115e.jpg)

![c7fa495071c54788750015fdadd53bf7f985b6a4ffd876cff83657e43c909fb7.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/c7fa495071c54788750015fdadd53bf7f985b6a4ffd876cff83657e43c909fb7.jpg)

![d7cb2b739218c049cbc85c93d569dff2dec4ed436fe5902f4160ec0ebe56b0eb.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/d7cb2b739218c049cbc85c93d569dff2dec4ed436fe5902f4160ec0ebe56b0eb.jpg)

![dbd216671144c53f60560982e55acdba6698fc9c62110f4c7d78dd70517a082e.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/dbd216671144c53f60560982e55acdba6698fc9c62110f4c7d78dd70517a082e.jpg)

![dfd3e3a19dbe995bcd7253cbcb43b6ce57f33c9bba2e1c21f62c0d62e0c4f6c8.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/dfd3e3a19dbe995bcd7253cbcb43b6ce57f33c9bba2e1c21f62c0d62e0c4f6c8.jpg)

![e909e7c073551b8a9fd068ab02a7745642ea38d1c94a2c15b50e65449d290439.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/e909e7c073551b8a9fd068ab02a7745642ea38d1c94a2c15b50e65449d290439.jpg)

![ee99510a392780b1aa43c140ce06bc0c6e537e2313aeaab5770e107d69e38cbc.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/images/ee99510a392780b1aa43c140ce06bc0c6e537e2313aeaab5770e107d69e38cbc.jpg)

### Tables

![0d3303a919c0ab2d935b2a62edcf44840ba65c0cf11077611f1e65aae191a0e9.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/0d3303a919c0ab2d935b2a62edcf44840ba65c0cf11077611f1e65aae191a0e9.jpg)

![290cde8158cb30545212ec81515790e10b0a88cb7dd7b3e66d5bea7a0cae308c.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/290cde8158cb30545212ec81515790e10b0a88cb7dd7b3e66d5bea7a0cae308c.jpg)

![31d2358bdc58c723d0b0966bc93802a27737b7aa65f83093c90f2939d6f01058.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/31d2358bdc58c723d0b0966bc93802a27737b7aa65f83093c90f2939d6f01058.jpg)

![3c3b21a276af1344f238a08c7859c2aff85df9966bed687f83108567e66dd2fe.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/3c3b21a276af1344f238a08c7859c2aff85df9966bed687f83108567e66dd2fe.jpg)

![481884a931928148e97e0e3bf9afe963c9c106f41885afe195e326ac80bc6fc9.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/481884a931928148e97e0e3bf9afe963c9c106f41885afe195e326ac80bc6fc9.jpg)

![4a6755c5c8b15a1d7f7d8736e09c8e7bf1a4429b5595865c1c2df57742d9acb6.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/4a6755c5c8b15a1d7f7d8736e09c8e7bf1a4429b5595865c1c2df57742d9acb6.jpg)

![568d57ffe714c2a358eba3df09cd27be6e97531ef7bad04bc9a02d934ddf99c0.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/568d57ffe714c2a358eba3df09cd27be6e97531ef7bad04bc9a02d934ddf99c0.jpg)

![665a32addbc3baba9596d4c1985d25c313ec7e1ea70d95d04c648fc9897db062.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/665a32addbc3baba9596d4c1985d25c313ec7e1ea70d95d04c648fc9897db062.jpg)

![70206716ce6e0c8dd3dcb6af60bee5098abae3863508d0904bf6f18a25fc73a5.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/70206716ce6e0c8dd3dcb6af60bee5098abae3863508d0904bf6f18a25fc73a5.jpg)

![7494383dfcc4c87fdd275aa68626b941f68338e8c1ff570b671e448ddb83b2b1.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/7494383dfcc4c87fdd275aa68626b941f68338e8c1ff570b671e448ddb83b2b1.jpg)

![95946de33650c316263aeb38d2726e6beac519ed5afc26ec0bba0df086cc8704.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/95946de33650c316263aeb38d2726e6beac519ed5afc26ec0bba0df086cc8704.jpg)

![be1e33f7316fad21004c0173713625df4ea1492694af343e93dfe252d16ec0aa.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/be1e33f7316fad21004c0173713625df4ea1492694af343e93dfe252d16ec0aa.jpg)

![cc5fe1522cc4d396c49e9d646512da1aed3299550c138ed4fdeb14ac6f80102c.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/cc5fe1522cc4d396c49e9d646512da1aed3299550c138ed4fdeb14ac6f80102c.jpg)

![d1ef8304cd42b6331af396d87d5425136c05ba279543ba06566505be0b298e26.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/d1ef8304cd42b6331af396d87d5425136c05ba279543ba06566505be0b298e26.jpg)

![d22ecc8d67b7dd852e86492c053145de657da4db1829ac3d78dfef020936ffd0.jpg](../icml_results/6003_PlotCraft_%20Pushing%20the%20Limits%20of%20LLMs%20for%20Complex%20and%20Interactive%20Data%20Visualization/tables/d22ecc8d67b7dd852e86492c053145de657da4db1829ac3d78dfef020936ffd0.jpg)

<a id="paper-5955"></a>

## Distributed Stochastic $K$-Level Optimization Over Networks


### Images

![1e6895d3d3d135ba75e75413a412b09b4cf9fd0c313f29a87e2ca5c3677ed8ac.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/1e6895d3d3d135ba75e75413a412b09b4cf9fd0c313f29a87e2ca5c3677ed8ac.jpg)

![2c860172cfc9db622ec8bb02675bfec37993d54bf770ece672ea6c9c84d997e4.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/2c860172cfc9db622ec8bb02675bfec37993d54bf770ece672ea6c9c84d997e4.jpg)

![2e8bea2d2fa9f444b37fdb168c929c106925cd95c61359871e5559527973dc07.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/2e8bea2d2fa9f444b37fdb168c929c106925cd95c61359871e5559527973dc07.jpg)

![2f92c0c9a4725985827446fd7c1af8be2ec15f466bad7fe242fa2bdae2efff6f.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/2f92c0c9a4725985827446fd7c1af8be2ec15f466bad7fe242fa2bdae2efff6f.jpg)

![3ca17d5c8d16ec28eaf2c6776cb6f8c4e636cd92880f0ef42c2cae08c7357efb.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/3ca17d5c8d16ec28eaf2c6776cb6f8c4e636cd92880f0ef42c2cae08c7357efb.jpg)

![4258a20692d692fdf21f4021c531dc20363ba22196d25b184dc889419f5690bf.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/4258a20692d692fdf21f4021c531dc20363ba22196d25b184dc889419f5690bf.jpg)

![481df35249774ddd00dd2bf2a357ed66a5e53191ea90a3303ed0daa42b2638a8.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/481df35249774ddd00dd2bf2a357ed66a5e53191ea90a3303ed0daa42b2638a8.jpg)

![89bff03f30a46548a25cdc2628106179b68a2053eb7e5ca5517cae991ac16061.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/89bff03f30a46548a25cdc2628106179b68a2053eb7e5ca5517cae991ac16061.jpg)

![8a562a18914db63df7609231261aaf76275932999eba26bf91cd3722757c1fb5.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/8a562a18914db63df7609231261aaf76275932999eba26bf91cd3722757c1fb5.jpg)

![9fa6c8d712df71c68c9ea5f38876b672b499b6141471f3b2e8df609a8a37d415.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/9fa6c8d712df71c68c9ea5f38876b672b499b6141471f3b2e8df609a8a37d415.jpg)

![cef5068327e61f22bcbc91a71a3ccfb77688df9390111ba80c52bf0379d7b5d0.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/cef5068327e61f22bcbc91a71a3ccfb77688df9390111ba80c52bf0379d7b5d0.jpg)

![d7f6bf7169cc4ffb919e2e98c05a0f3b0eb9332c69826b1cfa8332baea9aead3.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/d7f6bf7169cc4ffb919e2e98c05a0f3b0eb9332c69826b1cfa8332baea9aead3.jpg)

![fb561b6f5dfd5a04e1c4f0d95dc939d749f467f899194cefa9743914c023963f.jpg](../icml_results/6004_Distributed%20Stochastic%20%24K%24-Level%20Optimization%20Over%20Networks/images/fb561b6f5dfd5a04e1c4f0d95dc939d749f467f899194cefa9743914c023963f.jpg)

<a id="paper-5956"></a>

## STRIDE: Post-Training LLMs to Reason and Refine Bio-Sequences via Edit Trajectories


### Images

![2375f4842f99d8481f900ec69114b38f5e13892fbb1c5ceca1c23306c9566e94.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/images/2375f4842f99d8481f900ec69114b38f5e13892fbb1c5ceca1c23306c9566e94.jpg)

![932297b048703a1b3d45263fc44a3cfc701795bc574d613f550a956823d28943.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/images/932297b048703a1b3d45263fc44a3cfc701795bc574d613f550a956823d28943.jpg)

### Tables

![2c22d64621305e5bde76891d9e265cedac1b49ab13b5b87d5b33bd89d90a924f.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/2c22d64621305e5bde76891d9e265cedac1b49ab13b5b87d5b33bd89d90a924f.jpg)

![3b49f748306a68b09f08a815cf4d28e33f927ec38aa34057838d95f505e02d29.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/3b49f748306a68b09f08a815cf4d28e33f927ec38aa34057838d95f505e02d29.jpg)

![5ce6666455ab7ae5e60ce91f109239d93888adf01bbb1ea832ef0108c3efee2c.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/5ce6666455ab7ae5e60ce91f109239d93888adf01bbb1ea832ef0108c3efee2c.jpg)

![5eeeac27ce51d7461a41bc4b7765b3dabcc50fce512dc1a54d47a16e32af9111.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/5eeeac27ce51d7461a41bc4b7765b3dabcc50fce512dc1a54d47a16e32af9111.jpg)

![7af149a6ce4c235cccf784539e1c5d8ad192cf3715fbfcc6cf3d9f6561883469.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/7af149a6ce4c235cccf784539e1c5d8ad192cf3715fbfcc6cf3d9f6561883469.jpg)

![96e000a49cf7921afff6e29ad7c792bed09fed7ee4c8684545df9cce4c1babc5.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/96e000a49cf7921afff6e29ad7c792bed09fed7ee4c8684545df9cce4c1babc5.jpg)

![9dc2fb5058bbae42c46f4553c4ac28fa24016aa7ea91df7e8fce0bcb5ca6668c.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/9dc2fb5058bbae42c46f4553c4ac28fa24016aa7ea91df7e8fce0bcb5ca6668c.jpg)

![a1da0dc9a5d74809f564c70b66570f7773ee12f8d376e43b8b47e681b458d05d.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/a1da0dc9a5d74809f564c70b66570f7773ee12f8d376e43b8b47e681b458d05d.jpg)

![a49c5b5f064b28b71a8820058a933e4bb17ea31807992b3886f152da0d39d2ff.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/a49c5b5f064b28b71a8820058a933e4bb17ea31807992b3886f152da0d39d2ff.jpg)

![aa6ae7a4d7a979fe607f325ef631123796984676a14d0c170deb1a92dfef76dc.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/aa6ae7a4d7a979fe607f325ef631123796984676a14d0c170deb1a92dfef76dc.jpg)

![bc52a6ebc522b00eb2724bd58b66e757f6db8aaa0dfe7f114b48f7b949205fb4.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/bc52a6ebc522b00eb2724bd58b66e757f6db8aaa0dfe7f114b48f7b949205fb4.jpg)

![d8e98bf0fe960fde0dc0f8b2ef0de2161c3f2166fbb07ecab806d716581ea068.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/d8e98bf0fe960fde0dc0f8b2ef0de2161c3f2166fbb07ecab806d716581ea068.jpg)

![e3394452e7a73a6eef9ae87d37f336f4f13cd76ce29a92d43a10949055b54e31.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/e3394452e7a73a6eef9ae87d37f336f4f13cd76ce29a92d43a10949055b54e31.jpg)

![ee8fe157dd788a878542ebb36b0fb87ef7c45b3db3fc97e3a7c72c4d132ac67c.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/ee8fe157dd788a878542ebb36b0fb87ef7c45b3db3fc97e3a7c72c4d132ac67c.jpg)

![f2e757603baaece115bf0b66b87897f0da8c86a6f427fe388a9ee899e1d1956f.jpg](../icml_results/6005_STRIDE_%20Post-Training%20LLMs%20to%20Reason%20and%20Refine%20Bio-Sequences%20via%20Edit%20Trajectories/tables/f2e757603baaece115bf0b66b87897f0da8c86a6f427fe388a9ee899e1d1956f.jpg)

<a id="paper-5957"></a>

## SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising


### Images

![1726055f3d400183dfc83421d1947a995de22f387857659541df410c4d3ff1ae.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/1726055f3d400183dfc83421d1947a995de22f387857659541df410c4d3ff1ae.jpg)

![2d591eec347a09bfe6fabb156bcfbe7149a491bd01714d28157f97433ab8aefe.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/2d591eec347a09bfe6fabb156bcfbe7149a491bd01714d28157f97433ab8aefe.jpg)

![4a0caa60329a33b863a47fe8bd4b940ac41f9cb240e29f54000ae24540d78cc0.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/4a0caa60329a33b863a47fe8bd4b940ac41f9cb240e29f54000ae24540d78cc0.jpg)

![5a1eb3e8488a9da7311612013824681d623693fc97f7feb000fd2431e45c559b.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/5a1eb3e8488a9da7311612013824681d623693fc97f7feb000fd2431e45c559b.jpg)

![c58274949d998b91f372145ac09fb9fea2968c6c449d7aedda1a180b9200c77c.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/c58274949d998b91f372145ac09fb9fea2968c6c449d7aedda1a180b9200c77c.jpg)

![e03a6928e94f119b74c58090995bdd0607ba6185868ad6defe659c7affb508fb.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/e03a6928e94f119b74c58090995bdd0607ba6185868ad6defe659c7affb508fb.jpg)

![e22e14271cb9aac1d44164ad38d144a4c8f06b59691c59c7bff7f7a16626208b.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/e22e14271cb9aac1d44164ad38d144a4c8f06b59691c59c7bff7f7a16626208b.jpg)

![fef8587d2be202b00091664944ff182e8bb24b2c44be2b6e9521b318321cf014.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/images/fef8587d2be202b00091664944ff182e8bb24b2c44be2b6e9521b318321cf014.jpg)

### Tables

![041676c2b319ac3122f6e235df479542fcbbcf6a2f4348478f2ba6d4d341c2ac.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/041676c2b319ac3122f6e235df479542fcbbcf6a2f4348478f2ba6d4d341c2ac.jpg)

![27367587d4dde9456f4ddc7eea0a69836eb7e58fb5def70a4911496f1a5e5818.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/27367587d4dde9456f4ddc7eea0a69836eb7e58fb5def70a4911496f1a5e5818.jpg)

![67d43fb3934bb66fe6adda5351f6e29ffe1a09a81bf60f13d699e08c206ed60a.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/67d43fb3934bb66fe6adda5351f6e29ffe1a09a81bf60f13d699e08c206ed60a.jpg)

![68781c8dd1b14fafa1df927d6fd915a2ff5bed4a2e145de8d06dc22b3489b749.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/68781c8dd1b14fafa1df927d6fd915a2ff5bed4a2e145de8d06dc22b3489b749.jpg)

![e3432b1cb7153c7a1aebefda99cc697096aab401a5bdf99a5837221157c4a0c4.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/e3432b1cb7153c7a1aebefda99cc697096aab401a5bdf99a5837221157c4a0c4.jpg)

![e6ef4bd510ef3556b80bd0316ed5f6e6cfe3fa76eb11f84a716c3ee8cc0928cc.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/e6ef4bd510ef3556b80bd0316ed5f6e6cfe3fa76eb11f84a716c3ee8cc0928cc.jpg)

![e84568e879bfaea262cabc7392e1bdd235d89f1715828e3613db7d10ccb72739.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/e84568e879bfaea262cabc7392e1bdd235d89f1715828e3613db7d10ccb72739.jpg)

![f3ec29facd62c8278ad8de1a6b67312aa701c2029aff9b9fe7317cac41686a76.jpg](../icml_results/6006_SIMPC_%20Learning%20Self-Induced%20Mirror-Point%20Consistency%20for%20Unsupervised%20Point%20Cloud%20Denoising/tables/f3ec29facd62c8278ad8de1a6b67312aa701c2029aff9b9fe7317cac41686a76.jpg)

<a id="paper-5958"></a>

## Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process


### Images

![078fb3e9717172516b072b4339431ba3a4a1ceb265e0b11da7dc3926f8c19a62.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/078fb3e9717172516b072b4339431ba3a4a1ceb265e0b11da7dc3926f8c19a62.jpg)

![4b08fe0ffbdd7adfd05e5cdaac0fd5b38cf7768a075ea672280c942cf980e2d6.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/4b08fe0ffbdd7adfd05e5cdaac0fd5b38cf7768a075ea672280c942cf980e2d6.jpg)

![4c22234dca082b80439575e61c628972e38c07575627723e8bf6bbb082db856e.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/4c22234dca082b80439575e61c628972e38c07575627723e8bf6bbb082db856e.jpg)

![59f37f02618da52ffba68aba4d95cc96ac7fe9af20a80d5a8f4424fc2e1ed035.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/59f37f02618da52ffba68aba4d95cc96ac7fe9af20a80d5a8f4424fc2e1ed035.jpg)

![5d10ff2165ba128edff3c82d1d6336694516fc8cd543ebd56e9df71fb63a55c8.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/5d10ff2165ba128edff3c82d1d6336694516fc8cd543ebd56e9df71fb63a55c8.jpg)

![6538f9450313fc9ab222042906599b5b9c6c6a0c29c74d306e19f369609b9af3.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/6538f9450313fc9ab222042906599b5b9c6c6a0c29c74d306e19f369609b9af3.jpg)

![b8959cd48c9db2b73e49fec279095d75e49937bd13865a29054bfc7724f9fea3.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/b8959cd48c9db2b73e49fec279095d75e49937bd13865a29054bfc7724f9fea3.jpg)

![cd83477a52437e70c4bf1eb088b10afc3cb704e0017694d8012ae40a34d65b13.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/cd83477a52437e70c4bf1eb088b10afc3cb704e0017694d8012ae40a34d65b13.jpg)

![dfeb7b3b6711b129f495b01e2dd7b91cd1802dfea9f9e6ea48c99ea578e99750.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/dfeb7b3b6711b129f495b01e2dd7b91cd1802dfea9f9e6ea48c99ea578e99750.jpg)

![e27347f955ca05ed1ded103d8d26e119bbec91283d2683f24c7459447169a185.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/e27347f955ca05ed1ded103d8d26e119bbec91283d2683f24c7459447169a185.jpg)

![ec425cd8397de47960fdbd7c4f8269c24650cf7f53d985b6f229a96a1538124c.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/ec425cd8397de47960fdbd7c4f8269c24650cf7f53d985b6f229a96a1538124c.jpg)

![f02888d021ee4df382f0899021783ff81aa0eeb03877fb19f546d152ebd5c5b6.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/images/f02888d021ee4df382f0899021783ff81aa0eeb03877fb19f546d152ebd5c5b6.jpg)

### Tables

![e46c65eab6dc6a6d6823e9bef2b122fce34de9796ec5f6bd36d236831f8945f6.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/tables/e46c65eab6dc6a6d6823e9bef2b122fce34de9796ec5f6bd36d236831f8945f6.jpg)

![e5934cf4259e12cd3accd263add5349cc339e26e007418c2ee1a49566c5f0057.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/tables/e5934cf4259e12cd3accd263add5349cc339e26e007418c2ee1a49566c5f0057.jpg)

![eae8fa43da032d55bb20cbc7b0f04b9b132e720123cfc069f97afe043188736a.jpg](../icml_results/6007_Fantastic%20Reasoning%20Behaviors%20and%20Where%20to%20Find%20Them_%20Unsupervised%20Discovery%20of%20the%20Reasoning%20Proces/tables/eae8fa43da032d55bb20cbc7b0f04b9b132e720123cfc069f97afe043188736a.jpg)

<a id="paper-5959"></a>

## Beyond Single Embedding: Modeling User Preferences as Distribution in Federated Recommendation


### Images

![0a738c2692094fbb3b2a8916bbb5395df0c9a501cd745d64017953d95b6051c0.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/0a738c2692094fbb3b2a8916bbb5395df0c9a501cd745d64017953d95b6051c0.jpg)

![1aa6cec701e365fbc7bb2607d00ee626a88fc72e03e3c6f22aef70fc3d067c7e.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/1aa6cec701e365fbc7bb2607d00ee626a88fc72e03e3c6f22aef70fc3d067c7e.jpg)

![20bb9706e0365cb34b29d5e1429d9d791821e5a061fff13b6047935792affb58.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/20bb9706e0365cb34b29d5e1429d9d791821e5a061fff13b6047935792affb58.jpg)

![2b118974e71873a600d972ff995988d83d7f89432c83466151b89fe3bad34768.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/2b118974e71873a600d972ff995988d83d7f89432c83466151b89fe3bad34768.jpg)

![48e7e78cfbaadc58d88cfbc133baf000f896237f5954b8060bb85f44147e20ba.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/48e7e78cfbaadc58d88cfbc133baf000f896237f5954b8060bb85f44147e20ba.jpg)

![80e6bf7b8900de5bd0b855588cf36dcf649bb294af4f84797c45b63651bea205.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/80e6bf7b8900de5bd0b855588cf36dcf649bb294af4f84797c45b63651bea205.jpg)

![b0afdfce8d135d74138af509665e5dbdc99c3e4eae7db2cf8448fb0cb86def54.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/b0afdfce8d135d74138af509665e5dbdc99c3e4eae7db2cf8448fb0cb86def54.jpg)

![dd7c7ceed934522bcbccda0f8f37f2d596c32b3597cffb1c4d258c1226df5649.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/dd7c7ceed934522bcbccda0f8f37f2d596c32b3597cffb1c4d258c1226df5649.jpg)

![f506285a1fcc24f4453cf8b5f5b7e83d386a927318c8c65efc7141b187975ddb.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/images/f506285a1fcc24f4453cf8b5f5b7e83d386a927318c8c65efc7141b187975ddb.jpg)

### Tables

![45858f997fc92dd82b84aa1ffb137be11ba8e4a685b3b626c8181da95e851f92.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/tables/45858f997fc92dd82b84aa1ffb137be11ba8e4a685b3b626c8181da95e851f92.jpg)

![64872272df9249387bb3fb1940f634827d5be63c08399ebe5ff378f83d81ff7c.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/tables/64872272df9249387bb3fb1940f634827d5be63c08399ebe5ff378f83d81ff7c.jpg)

![9a44390feaefccca126eecc3773805de622273d01ca13456cc71ee2883fedbef.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/tables/9a44390feaefccca126eecc3773805de622273d01ca13456cc71ee2883fedbef.jpg)

![c493e09c61045cb409ba0fc118fcebc195e083601299c3ce8911c3e7cbb9997d.jpg](../icml_results/6008_Beyond%20Single%20Embedding_%20Modeling%20User%20Preferences%20as%20Distribution%20in%20Federated%20Recommendation/tables/c493e09c61045cb409ba0fc118fcebc195e083601299c3ce8911c3e7cbb9997d.jpg)

<a id="paper-5960"></a>

## Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs


### Images

![50f1660d18eb668b2b8cfbb95522b8106e7d2dd777ca7708a661eebca0f7f7ed.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/images/50f1660d18eb668b2b8cfbb95522b8106e7d2dd777ca7708a661eebca0f7f7ed.jpg)

![662ecaabfc8582ebbc9cdca7f5044f87e91a30f661ed673fde56fd649a464910.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/images/662ecaabfc8582ebbc9cdca7f5044f87e91a30f661ed673fde56fd649a464910.jpg)

![8c8e04556405bb05cb3192d1132684e86ea5ba2fff347ec20fc8718b583647ae.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/images/8c8e04556405bb05cb3192d1132684e86ea5ba2fff347ec20fc8718b583647ae.jpg)

![aae852cb18f2cb5723b0853e50ac5f41a3bc4b005a7c5194e49687138e997e0c.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/images/aae852cb18f2cb5723b0853e50ac5f41a3bc4b005a7c5194e49687138e997e0c.jpg)

![e087943f2e727e4799cbdf68714842fe8d697d37c5f042b4cd16b4f8cd96c2c3.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/images/e087943f2e727e4799cbdf68714842fe8d697d37c5f042b4cd16b4f8cd96c2c3.jpg)

### Tables

![28a39ed6ce11ed19ad0628978cbbb7a96a31c8281df5558c8dcb141ce71910fa.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/28a39ed6ce11ed19ad0628978cbbb7a96a31c8281df5558c8dcb141ce71910fa.jpg)

![3854cb7722601f41f0eda82d698a4b6a9f645c2b3249059dc84863d04b3bb32b.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/3854cb7722601f41f0eda82d698a4b6a9f645c2b3249059dc84863d04b3bb32b.jpg)

![399d60e16ae7a9920508aff496a2664adfc7b3dbdaf01310d748af1955fa18db.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/399d60e16ae7a9920508aff496a2664adfc7b3dbdaf01310d748af1955fa18db.jpg)

![3daa5108930d4802a0e55a491fa689563ae5020cfbef91180d65f85917ed55a5.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/3daa5108930d4802a0e55a491fa689563ae5020cfbef91180d65f85917ed55a5.jpg)

![3f1a8c415a72adf226e8be7edc57259269decc37ffa6496089ba4c4cb7159553.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/3f1a8c415a72adf226e8be7edc57259269decc37ffa6496089ba4c4cb7159553.jpg)

![5577dbee82d3056be1e80f5e9c6855fdc098a1ca6d4645a764173df6ccfb3e17.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/5577dbee82d3056be1e80f5e9c6855fdc098a1ca6d4645a764173df6ccfb3e17.jpg)

![5c35df41fc5e64e7ea4a64fe1088fb94c81e8e950d362887004d3010f8d80fd0.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/5c35df41fc5e64e7ea4a64fe1088fb94c81e8e950d362887004d3010f8d80fd0.jpg)

![8ce854df46054f9e5959cf4ac6c8419ea0191131b4190ba204bbd226963e4a20.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/8ce854df46054f9e5959cf4ac6c8419ea0191131b4190ba204bbd226963e4a20.jpg)

![a7cbac2dd6b522182953ed13ac30e3c8ed4ba21eb3bca07c5cdd54a1f2156d6a.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/a7cbac2dd6b522182953ed13ac30e3c8ed4ba21eb3bca07c5cdd54a1f2156d6a.jpg)

![a921e284264110f57b2dcb02be8d031759835661eacb87368751137844165262.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/a921e284264110f57b2dcb02be8d031759835661eacb87368751137844165262.jpg)

![b41fa9bf0192b6b9cb67fa8ec04a782b5e6b999ed8adda46630c8a2861dc09ee.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/b41fa9bf0192b6b9cb67fa8ec04a782b5e6b999ed8adda46630c8a2861dc09ee.jpg)

![d7c79478ea68d944d7ec2c821584b22a375d47dbc64fdcfeb21e1f977b3bca1c.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/d7c79478ea68d944d7ec2c821584b22a375d47dbc64fdcfeb21e1f977b3bca1c.jpg)

![d9a2dc689cd3d9c290df7c551bda3eb4d389ed0f7887e4bd2da9874c8880623f.jpg](../icml_results/6009_Experience%20is%20the%20Best%20Teacher_%20Motivating%20Effective%20Exploration%20in%20Reinforcement%20Learning%20for%20LLMs/tables/d9a2dc689cd3d9c290df7c551bda3eb4d389ed0f7887e4bd2da9874c8880623f.jpg)

<a id="paper-5961"></a>

## Joint Navigation and Manipulation Planning with 3D Interaction Chains


### Images

![260dde787696159aa2ce7a2aaba5f869e64d07b24c4ac7e56cde649bae4802d5.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/images/260dde787696159aa2ce7a2aaba5f869e64d07b24c4ac7e56cde649bae4802d5.jpg)

![973b069c214fd6e732a65aa5291d5472e20fcc89df816287266cf65a820e4208.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/images/973b069c214fd6e732a65aa5291d5472e20fcc89df816287266cf65a820e4208.jpg)

![ac96bae753e7108200e9108a554d745e4561a6814adc09ac5e291f532975a75b.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/images/ac96bae753e7108200e9108a554d745e4561a6814adc09ac5e291f532975a75b.jpg)

![ae555427cbc3d9f7a531e5e5ec693bbf01d8f3f29e1e36de2569ebf25776e105.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/images/ae555427cbc3d9f7a531e5e5ec693bbf01d8f3f29e1e36de2569ebf25776e105.jpg)

![e7195c5716329dcc6c4d8f0cd4bf772a9669e562285c045a1da13230a7cd7993.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/images/e7195c5716329dcc6c4d8f0cd4bf772a9669e562285c045a1da13230a7cd7993.jpg)

### Tables

![0f55e5f3d1e18ba78699837847079441e097b0be6ec19f672bce3ba7f7e28e9c.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/tables/0f55e5f3d1e18ba78699837847079441e097b0be6ec19f672bce3ba7f7e28e9c.jpg)

![147ea17dc3d2592e5374a85a2cbf6c5367dd72331970227d6332cf4fd8585893.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/tables/147ea17dc3d2592e5374a85a2cbf6c5367dd72331970227d6332cf4fd8585893.jpg)

![1e4e4c98cac6d9d90f92df8b9f253a36ae46e34fcd3a25244ae253a111cb167a.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/tables/1e4e4c98cac6d9d90f92df8b9f253a36ae46e34fcd3a25244ae253a111cb167a.jpg)

![55c0075e039d008155c85ae7ef3bbab561710aeccd2aa9fe29d62de8779e317f.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/tables/55c0075e039d008155c85ae7ef3bbab561710aeccd2aa9fe29d62de8779e317f.jpg)

![7d1eef914b7fd72202e2292d8f2f047b3ab3d3e3838a58ee45fc4b052d5ca9a3.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/tables/7d1eef914b7fd72202e2292d8f2f047b3ab3d3e3838a58ee45fc4b052d5ca9a3.jpg)

![9f35535e932d79b323eab79684daec63ec841cf9594ff46d7da17fbde50057ff.jpg](../icml_results/6010_Joint%20Navigation%20and%20Manipulation%20Planning%20with%203D%20Interaction%20Chains/tables/9f35535e932d79b323eab79684daec63ec841cf9594ff46d7da17fbde50057ff.jpg)

<a id="paper-5962"></a>

## Optimal Transport with Symmetry Groups


### Images

![21343c3120a339bd9b1c7eda56adaad3bf05bab89d877f89a13e769e68db32c8.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/21343c3120a339bd9b1c7eda56adaad3bf05bab89d877f89a13e769e68db32c8.jpg)

![2398ec430acf03e4658860796336646f304259e1e2439fef84eee5a2e5c60386.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/2398ec430acf03e4658860796336646f304259e1e2439fef84eee5a2e5c60386.jpg)

![2a9494d80775edf9fec5ccb99fc2131fbe877b5e0ac94dd0bb2a57f346b0f9b6.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/2a9494d80775edf9fec5ccb99fc2131fbe877b5e0ac94dd0bb2a57f346b0f9b6.jpg)

![35ef44426930a3c8b41b405518301e2507e176326ccf2078f5e9de98b4494964.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/35ef44426930a3c8b41b405518301e2507e176326ccf2078f5e9de98b4494964.jpg)

![4a66fa6607da1e7a7e68d9438f0f37eca2d746392040aeeafa07d6cb386b7394.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/4a66fa6607da1e7a7e68d9438f0f37eca2d746392040aeeafa07d6cb386b7394.jpg)

![6c635e654b4668abbb1ab54a6f5c7b118c2c26d2f5f469604c01f2e569b7f19b.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/6c635e654b4668abbb1ab54a6f5c7b118c2c26d2f5f469604c01f2e569b7f19b.jpg)

![6ffb0abc390ba3c8c8cbce732b1e87341addf868ca9bcba98454236d508f5c0f.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/6ffb0abc390ba3c8c8cbce732b1e87341addf868ca9bcba98454236d508f5c0f.jpg)

![b6abd6472ff15a53cbb836406660b32b197fd8b4f7d6df0c469eff8a252402db.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/b6abd6472ff15a53cbb836406660b32b197fd8b4f7d6df0c469eff8a252402db.jpg)

![e9aa375e499ff585c733b3c076160d8dded602cc4d34b1db83db70f5d0f5e2ec.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/images/e9aa375e499ff585c733b3c076160d8dded602cc4d34b1db83db70f5d0f5e2ec.jpg)

### Tables

![1622c89fae4be09d7f4af1420663b80038d940046796ba5b826dc41d1095fd17.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/1622c89fae4be09d7f4af1420663b80038d940046796ba5b826dc41d1095fd17.jpg)

![30e643f4cdd6ec9ae52f367402363c41499dd2c6e7f1e8cbdface0ed2890d5ae.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/30e643f4cdd6ec9ae52f367402363c41499dd2c6e7f1e8cbdface0ed2890d5ae.jpg)

![3a3316fa07d56caf7eef79412b6bcc64099dedbb431017a0698e9ac7c36f36f5.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/3a3316fa07d56caf7eef79412b6bcc64099dedbb431017a0698e9ac7c36f36f5.jpg)

![95dd72b56f827a0e180c3c33fdf7c426b39784ef444826d831ab594d1c463cb5.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/95dd72b56f827a0e180c3c33fdf7c426b39784ef444826d831ab594d1c463cb5.jpg)

![f1d1a7c02b1a799b65b263533b81775e6f0352b23cbc6247d4fffe8718a99979.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/f1d1a7c02b1a799b65b263533b81775e6f0352b23cbc6247d4fffe8718a99979.jpg)

![f380edf7054c83c7c7153ce04e84ef7ca5b1e66383fd976668156b7c1533e398.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/f380edf7054c83c7c7153ce04e84ef7ca5b1e66383fd976668156b7c1533e398.jpg)

![ff4d59196aca8586d66f8f33e8572c251e337fc7c42fbf0770907b315def4291.jpg](../icml_results/6011_Optimal%20Transport%20with%20Symmetry%20Groups/tables/ff4d59196aca8586d66f8f33e8572c251e337fc7c42fbf0770907b315def4291.jpg)

<a id="paper-5963"></a>

## Convergence Analysis of Decentralized Hessian-/Jacobian-Free Algorithm for Nonconvex Stochastic Bilevel Optimization


### Images

![4a8e0df10711bf01046f0f94a73d9f44056b8bf42244aeb70c2c99a61e06979c.jpg](../icml_results/6012_Convergence%20Analysis%20of%20Decentralized%20Hessian-_Jacobian-Free%20Algorithm%20for%20Nonconvex%20Stochastic%20Bile/images/4a8e0df10711bf01046f0f94a73d9f44056b8bf42244aeb70c2c99a61e06979c.jpg)

![4e2e86939dea5963d83fc82136e8d1e629edc03e70e57d6c1e820a30411229ee.jpg](../icml_results/6012_Convergence%20Analysis%20of%20Decentralized%20Hessian-_Jacobian-Free%20Algorithm%20for%20Nonconvex%20Stochastic%20Bile/images/4e2e86939dea5963d83fc82136e8d1e629edc03e70e57d6c1e820a30411229ee.jpg)

![595ebdcd87e09f55e3cc43a09754b4dfb97da653583383b7d0b915535158c7b6.jpg](../icml_results/6012_Convergence%20Analysis%20of%20Decentralized%20Hessian-_Jacobian-Free%20Algorithm%20for%20Nonconvex%20Stochastic%20Bile/images/595ebdcd87e09f55e3cc43a09754b4dfb97da653583383b7d0b915535158c7b6.jpg)

![71c8a87416ba7987f19eec6c0f7830e248a0648a61586bec0e74f766a1f7ccaa.jpg](../icml_results/6012_Convergence%20Analysis%20of%20Decentralized%20Hessian-_Jacobian-Free%20Algorithm%20for%20Nonconvex%20Stochastic%20Bile/images/71c8a87416ba7987f19eec6c0f7830e248a0648a61586bec0e74f766a1f7ccaa.jpg)

![c643c921712ec294d5e80e005f81227908343d8f215db27d22012598acb7106f.jpg](../icml_results/6012_Convergence%20Analysis%20of%20Decentralized%20Hessian-_Jacobian-Free%20Algorithm%20for%20Nonconvex%20Stochastic%20Bile/images/c643c921712ec294d5e80e005f81227908343d8f215db27d22012598acb7106f.jpg)

<a id="paper-5964"></a>

## Dynamic Multimodal Evaluation via Knowledge-Enhanced Benchmark Evolution


### Images

![138e8800d426a8a3bc0bd15c14074375368d18e7e2848a1b88c41fbf9ea53117.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/138e8800d426a8a3bc0bd15c14074375368d18e7e2848a1b88c41fbf9ea53117.jpg)

![15edfc37b7259f6b725fca88eba2267b76684bd2796a3f3a8a6e84cabad0bc2e.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/15edfc37b7259f6b725fca88eba2267b76684bd2796a3f3a8a6e84cabad0bc2e.jpg)

![260d2c423b320def13632976a3ecc6e82f8420f192186d992e311a67c40b6bba.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/260d2c423b320def13632976a3ecc6e82f8420f192186d992e311a67c40b6bba.jpg)

![62a1e7fed51bf746cc5470a5875a2b829e88c1fa98951472f48fa0bf012bb0ab.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/62a1e7fed51bf746cc5470a5875a2b829e88c1fa98951472f48fa0bf012bb0ab.jpg)

![8e0787242e07b01c051075354e26e7e9aaa6b5217acd522a44aabd36b9313b48.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/8e0787242e07b01c051075354e26e7e9aaa6b5217acd522a44aabd36b9313b48.jpg)

![b1cf7c87a452ce1b867dc35282c96bb7140fac1db2854a9021af8939b46a1df1.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/b1cf7c87a452ce1b867dc35282c96bb7140fac1db2854a9021af8939b46a1df1.jpg)

![c4c658ce27b084f893ba6cd2c66cfee8b6cb4f85ff0ebf76ed52973ca7c3cb01.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/images/c4c658ce27b084f893ba6cd2c66cfee8b6cb4f85ff0ebf76ed52973ca7c3cb01.jpg)

### Tables

![233b90c6a49a65bffd88437d3c6971311d741e0fc8d99447c768d98fe8a10508.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/233b90c6a49a65bffd88437d3c6971311d741e0fc8d99447c768d98fe8a10508.jpg)

![3173c043c1be6192ff8efe0b27de579de3ace3a331a46950a19d5c36b3862ea6.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/3173c043c1be6192ff8efe0b27de579de3ace3a331a46950a19d5c36b3862ea6.jpg)

![51266c3eeea2a9ff7b53f26ab4bd27b90e1faec8481845ee3cba99c3609d9cfe.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/51266c3eeea2a9ff7b53f26ab4bd27b90e1faec8481845ee3cba99c3609d9cfe.jpg)

![65a1d2abe29934eaf0213f2873399e2345235bea7dacbd68c160e77c32de0b63.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/65a1d2abe29934eaf0213f2873399e2345235bea7dacbd68c160e77c32de0b63.jpg)

![7081b8826034b0a01fc3a7817e430d54f0e2d7dfa93d577d99f227b8d30535ff.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/7081b8826034b0a01fc3a7817e430d54f0e2d7dfa93d577d99f227b8d30535ff.jpg)

![94fb16d20722c2d5fcf96d7caee58aaa570535f87a94fd6031281000d52e2a49.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/94fb16d20722c2d5fcf96d7caee58aaa570535f87a94fd6031281000d52e2a49.jpg)

![9d67bcae56eec08809a3d5f812dec9ff9747f1aba8d17a0df11271cd7ab04d8a.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/9d67bcae56eec08809a3d5f812dec9ff9747f1aba8d17a0df11271cd7ab04d8a.jpg)

![9e074e7f56fe19e65457f90f6a4c3945519fd64eb2bcf70edad56b619452f5cb.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/9e074e7f56fe19e65457f90f6a4c3945519fd64eb2bcf70edad56b619452f5cb.jpg)

![b4caa9fdef4affde874ba28ddcd40db13e78537b0a72912856cf04b9750c5f2c.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/b4caa9fdef4affde874ba28ddcd40db13e78537b0a72912856cf04b9750c5f2c.jpg)

![d033678aa99277ca62e792dc5af485f89f156386bf62b85360896ce2f574a20c.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/d033678aa99277ca62e792dc5af485f89f156386bf62b85360896ce2f574a20c.jpg)

![ddaaeec98892f73af3b0faaab75743c76057bd35bdfd9c48fb144f7a7da591aa.jpg](../icml_results/6013_Dynamic%20Multimodal%20Evaluation%20via%20Knowledge-Enhanced%20Benchmark%20Evolution/tables/ddaaeec98892f73af3b0faaab75743c76057bd35bdfd9c48fb144f7a7da591aa.jpg)

<a id="paper-5965"></a>

## Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge


### Images

![3d4c268969d3911a0b1167d4d239845728e9f0d0bc67eb5b3e3ebb9a70428f9c.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/images/3d4c268969d3911a0b1167d4d239845728e9f0d0bc67eb5b3e3ebb9a70428f9c.jpg)

![4c85955ca4441d6f8fe5cdbee96a79a0a2ba32c8ea4489a339bef83d8b27d897.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/images/4c85955ca4441d6f8fe5cdbee96a79a0a2ba32c8ea4489a339bef83d8b27d897.jpg)

![6ac6809be3b4b4920657233e2eb5cf3a766a2bc60b1c46535728ff692b05233a.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/images/6ac6809be3b4b4920657233e2eb5cf3a766a2bc60b1c46535728ff692b05233a.jpg)

![db884da479a049fbec6f42a0d12de486468ed92139495da062000ff6e1efb4c4.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/images/db884da479a049fbec6f42a0d12de486468ed92139495da062000ff6e1efb4c4.jpg)

![f7df166f2e4211272788d5e6798bbdd983a6d49d3d9e9b0beecb7f84543597bb.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/images/f7df166f2e4211272788d5e6798bbdd983a6d49d3d9e9b0beecb7f84543597bb.jpg)

### Tables

![17a5fddba7f2d798849e4145ca0b03754f04df6488e0076a27aee0c604a61fe5.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/17a5fddba7f2d798849e4145ca0b03754f04df6488e0076a27aee0c604a61fe5.jpg)

![1d5af295f99a1d95e2bcc8a259651581ac072b81bd26efc87d450d1b29f34027.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/1d5af295f99a1d95e2bcc8a259651581ac072b81bd26efc87d450d1b29f34027.jpg)

![3a28671d378404b75f44de5bca385b2275233269fda247e9ed96bbc98e353311.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/3a28671d378404b75f44de5bca385b2275233269fda247e9ed96bbc98e353311.jpg)

![4a0d4a1900bb83e66607ca4544ddf50c319b7734ec805b8df2e522f705b5334c.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/4a0d4a1900bb83e66607ca4544ddf50c319b7734ec805b8df2e522f705b5334c.jpg)

![976bc155667f6915df36931610bb680a47ca2d78bdfdf650899aaaf496988618.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/976bc155667f6915df36931610bb680a47ca2d78bdfdf650899aaaf496988618.jpg)

![aa2cb228feff40eb24489153b9eeb7e736eaedabdca177c253f7b974159deba3.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/aa2cb228feff40eb24489153b9eeb7e736eaedabdca177c253f7b974159deba3.jpg)

![d0aa2dacda3b93eee410afc973cb3fc2a7ce6db45322db62912b0b78d7a67197.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/d0aa2dacda3b93eee410afc973cb3fc2a7ce6db45322db62912b0b78d7a67197.jpg)

![f911e4b1f5a630494bbb904b6ecb4e01d4a0682bfc687f606dbbd06147f08653.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/f911e4b1f5a630494bbb904b6ecb4e01d4a0682bfc687f606dbbd06147f08653.jpg)

![f9ca228f6c6d6e295294328c8c2c12149f6d1dc1e424a9e586294d948ce85fd4.jpg](../icml_results/6014_Reasoning%20Is%20Not%20Free_%20Robust%20Adaptive%20Cost-Efficient%20Routing%20for%20LLM-as-a-Judge/tables/f9ca228f6c6d6e295294328c8c2c12149f6d1dc1e424a9e586294d948ce85fd4.jpg)

<a id="paper-5966"></a>

## Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing


### Images

![125d13d98715a7f0307537a2015c1d8dee040364d3edb9f094803adf33a15a64.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/125d13d98715a7f0307537a2015c1d8dee040364d3edb9f094803adf33a15a64.jpg)

![19c1640d8ff0405a51be890b86d8ba07c0298cdf54a536128cf255ba3cc46cdd.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/19c1640d8ff0405a51be890b86d8ba07c0298cdf54a536128cf255ba3cc46cdd.jpg)

![1cdd85d5fab5136a61aa768d82052ea7fb06bf4f7cdd284a2a9ef91647be638e.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/1cdd85d5fab5136a61aa768d82052ea7fb06bf4f7cdd284a2a9ef91647be638e.jpg)

![1ead2d35838240031d7ad4b9761148c8c4b5f9d8614d36d0917b5ec19912587e.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/1ead2d35838240031d7ad4b9761148c8c4b5f9d8614d36d0917b5ec19912587e.jpg)

![3e54a4c5163e015abb2b1814f1d6fa0c702f46742ca1fc37e64897bc59eee275.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/3e54a4c5163e015abb2b1814f1d6fa0c702f46742ca1fc37e64897bc59eee275.jpg)

![42236d4d17ecda36ecf48ac767fadb32128a96d7ad359c11b4e7d91bb0a1f279.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/42236d4d17ecda36ecf48ac767fadb32128a96d7ad359c11b4e7d91bb0a1f279.jpg)

![565169a747f13be335f4f88f8791ace104e152ef3dbf0b01284fcb32be697e23.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/565169a747f13be335f4f88f8791ace104e152ef3dbf0b01284fcb32be697e23.jpg)

![57effa6bb6ff881f162146851088bf5a58bf58c741c75772f32bba424b944b5b.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/57effa6bb6ff881f162146851088bf5a58bf58c741c75772f32bba424b944b5b.jpg)

![72c1a4641ba84e3aba00d61423a923de08bdf9c76af28648d8bccf4f307d570b.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/72c1a4641ba84e3aba00d61423a923de08bdf9c76af28648d8bccf4f307d570b.jpg)

![929f9a6b1edfe575364bb65614a67fa9cd21d9bf9a28bd95adbc9ef444902531.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/929f9a6b1edfe575364bb65614a67fa9cd21d9bf9a28bd95adbc9ef444902531.jpg)

![b035a0d1f76513b2a5d8c3dfb81fc20e92e23d74095e823144f6bb08d973ce56.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/b035a0d1f76513b2a5d8c3dfb81fc20e92e23d74095e823144f6bb08d973ce56.jpg)

![dd9018c10c3b60ceee10f1a91368370e9242c64a469460e05d9c62765db94e07.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/dd9018c10c3b60ceee10f1a91368370e9242c64a469460e05d9c62765db94e07.jpg)

![e78342aade5d591c49117e304cdde5c81359ca868c85694f3602cf2b970ae222.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/e78342aade5d591c49117e304cdde5c81359ca868c85694f3602cf2b970ae222.jpg)

![f6cc3e8b5a62b3c66ae2da9c8e3d8ff3cc85856cdd8ffc2c39e45d1131ad8b25.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/images/f6cc3e8b5a62b3c66ae2da9c8e3d8ff3cc85856cdd8ffc2c39e45d1131ad8b25.jpg)

### Tables

![0ae4c50fb614ddc4f1ffeff268ec1b8d3754fa73c7c4730df4ec767fcdc30fa0.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/0ae4c50fb614ddc4f1ffeff268ec1b8d3754fa73c7c4730df4ec767fcdc30fa0.jpg)

![14cfc8e6bf0891361f489237fd07b97402b593f3fc0af786499693c278c2188d.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/14cfc8e6bf0891361f489237fd07b97402b593f3fc0af786499693c278c2188d.jpg)

![6060c15b52a3106e494ea453108150218994d4743a6d5cd88812effe6c2def85.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/6060c15b52a3106e494ea453108150218994d4743a6d5cd88812effe6c2def85.jpg)

![6f56ffced17334b809d8d1de92fc2d743801ee15747cb5b9244e4ebdbf2195d4.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/6f56ffced17334b809d8d1de92fc2d743801ee15747cb5b9244e4ebdbf2195d4.jpg)

![847fc97e9712a3203221dad71dd758a7731915318755ba468a58bf4af5f997ae.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/847fc97e9712a3203221dad71dd758a7731915318755ba468a58bf4af5f997ae.jpg)

![a9ec93e601e6bd43dc2ecde01aada16a725931893b27f4c29153aa562543e543.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/a9ec93e601e6bd43dc2ecde01aada16a725931893b27f4c29153aa562543e543.jpg)

![ae78bb8838053f4055ab9ef97c75d1b811ddc12d5a921eef5822ec83dda69c41.jpg](../icml_results/6015_Both%20Semantics%20and%20Reconstruction%20Matter_%20Making%20Representation%20Encoders%20Ready%20for%20Text-to-Image%20Gen/tables/ae78bb8838053f4055ab9ef97c75d1b811ddc12d5a921eef5822ec83dda69c41.jpg)

<a id="paper-5967"></a>

## NITP: Next Implicit Token Prediction for LLM Pre-training


### Images

![007476b6bdc4c121bc3e0016b78a8e773573f153651e62416d41de8e9f07c724.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/images/007476b6bdc4c121bc3e0016b78a8e773573f153651e62416d41de8e9f07c724.jpg)

![52082c44169cd033ca25c5c81f81186654234112095a19fa0a05c070277eea10.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/images/52082c44169cd033ca25c5c81f81186654234112095a19fa0a05c070277eea10.jpg)

![7eb9d6f27b81ea975d3d6f4edcffe4a8347fa761671f588ee489a143ba6268df.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/images/7eb9d6f27b81ea975d3d6f4edcffe4a8347fa761671f588ee489a143ba6268df.jpg)

![85a23a89482c4fa90e34740431a31548d6f75a8b930dde65bc66c2a6f90cf4a0.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/images/85a23a89482c4fa90e34740431a31548d6f75a8b930dde65bc66c2a6f90cf4a0.jpg)

![e65da6486202f1e6bdc5074f4b49c781bbe5eac1f49f2b278a9bf47fc13e250b.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/images/e65da6486202f1e6bdc5074f4b49c781bbe5eac1f49f2b278a9bf47fc13e250b.jpg)

### Tables

![44baf90e7da75733e8f0edaaa0be59ca1beb59444f1ef96d7deab501be3c93bd.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/44baf90e7da75733e8f0edaaa0be59ca1beb59444f1ef96d7deab501be3c93bd.jpg)

![69db61094765f240f1841a4cd2917b8ad42f18917ec0bd20095e42af47977a2f.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/69db61094765f240f1841a4cd2917b8ad42f18917ec0bd20095e42af47977a2f.jpg)

![6b040f544e32495d9c5177a113a5e96ccfe627788ba58101895faf868c65ad11.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/6b040f544e32495d9c5177a113a5e96ccfe627788ba58101895faf868c65ad11.jpg)

![8f8479b9d15f97456b4cbd6b9fe1089b347df2deb56f0d9c61e5906f2d06a2c8.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/8f8479b9d15f97456b4cbd6b9fe1089b347df2deb56f0d9c61e5906f2d06a2c8.jpg)

![b34d4481220cc02adfb931d18de7e5de8b7e1943f16f2d84bf8a636b30b33806.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/b34d4481220cc02adfb931d18de7e5de8b7e1943f16f2d84bf8a636b30b33806.jpg)

![f4d2e2ab61285c3b8728c6c5d041b65777bdd86b9b37a8d125bc2a23f08bfcd4.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/f4d2e2ab61285c3b8728c6c5d041b65777bdd86b9b37a8d125bc2a23f08bfcd4.jpg)

![fc6faf3f73ea488696ce81436a4ff5cb9c343cdef9aac579f8ecd1381c67005e.jpg](../icml_results/6016_NITP_%20Next%20Implicit%20Token%20Prediction%20for%20LLM%20Pre-training/tables/fc6faf3f73ea488696ce81436a4ff5cb9c343cdef9aac579f8ecd1381c67005e.jpg)

<a id="paper-5968"></a>

## XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge


### Images

![412c71c864af88c042165f9dd2a9667c6b71e021e89709fff78074a6173a018a.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/412c71c864af88c042165f9dd2a9667c6b71e021e89709fff78074a6173a018a.jpg)

![48a26b13c861eb73bbae0cf5ebad5e832042924993175662456f4f4cf343ae90.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/48a26b13c861eb73bbae0cf5ebad5e832042924993175662456f4f4cf343ae90.jpg)

![8ac90e53bbb3d81cad0d389d98c95ba7981a07bfbb3d0eb7eb4f8ff6a1468dfb.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/8ac90e53bbb3d81cad0d389d98c95ba7981a07bfbb3d0eb7eb4f8ff6a1468dfb.jpg)

![98b52bede594500c8eff70c2734bc9fa65bd12227a9366722c9b1df9b16ca161.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/98b52bede594500c8eff70c2734bc9fa65bd12227a9366722c9b1df9b16ca161.jpg)

![bc3e4235dde23759447b651cf18640c11c4be501416a72a8526d7c221740c5fe.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/bc3e4235dde23759447b651cf18640c11c4be501416a72a8526d7c221740c5fe.jpg)

![e4d86d6c1a3a8caa7583447cde1dc1ba1e59ccf5d554e5696c2bc83cd78091ad.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/e4d86d6c1a3a8caa7583447cde1dc1ba1e59ccf5d554e5696c2bc83cd78091ad.jpg)

![f4a881d393bae427dd45660bfb0bf600cc9e35ea7de7396912276bd6f446fa9f.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/images/f4a881d393bae427dd45660bfb0bf600cc9e35ea7de7396912276bd6f446fa9f.jpg)

### Tables

![312320767ab60da4af05564e76fe47b51c5cd33c092456b0b69ef69b4e37f1c6.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/312320767ab60da4af05564e76fe47b51c5cd33c092456b0b69ef69b4e37f1c6.jpg)

![36fccbbcd5a26e2196a240e01feded5873cf8e02abdedc846e06718c38227677.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/36fccbbcd5a26e2196a240e01feded5873cf8e02abdedc846e06718c38227677.jpg)

![39ab57c70cb7b352ad8e78cd96a1d919c655c717c6cd946c76418a2f99a15446.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/39ab57c70cb7b352ad8e78cd96a1d919c655c717c6cd946c76418a2f99a15446.jpg)

![66494600313fc599332438b0ff2df2eaffcb5ad6f821790c7c19ae78a0df3c88.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/66494600313fc599332438b0ff2df2eaffcb5ad6f821790c7c19ae78a0df3c88.jpg)

![6de8b9d4dce0b1e01e19dbd160ca8ee255c8ed6195d4a8f49443646ac8826e3e.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/6de8b9d4dce0b1e01e19dbd160ca8ee255c8ed6195d4a8f49443646ac8826e3e.jpg)

![7ba3a883619750a68db2b21b8872c85305fc1fffbc3b6778e30b42640f3427c7.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/7ba3a883619750a68db2b21b8872c85305fc1fffbc3b6778e30b42640f3427c7.jpg)

![7e234891e41b3f602bc0644c1ec3074e984d72a51365b362a8dd983126ba79f5.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/7e234891e41b3f602bc0644c1ec3074e984d72a51365b362a8dd983126ba79f5.jpg)

![8c5284257f02cfef7238759fd839d5d8b33201255df863999d534d4124416e10.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/8c5284257f02cfef7238759fd839d5d8b33201255df863999d534d4124416e10.jpg)

![b7dbcc2e9484b25908939aa9d4336eda43e9d3e5943920db8de82f9978dbe2a5.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/b7dbcc2e9484b25908939aa9d4336eda43e9d3e5943920db8de82f9978dbe2a5.jpg)

![be3657c9b8cba10430ef4eea25bfe109c299b82efa1a82a472b0b63b4921a0d6.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/be3657c9b8cba10430ef4eea25bfe109c299b82efa1a82a472b0b63b4921a0d6.jpg)

![ccf665571d6edd587ac6a3faeef4522d576f19b606de3bee73a21ec2f2bd3cba.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/ccf665571d6edd587ac6a3faeef4522d576f19b606de3bee73a21ec2f2bd3cba.jpg)

![dfdb9cf609498c7a226f93f6563c4899b9ebdd44bced1d73d5f52a5e472650ea.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/dfdb9cf609498c7a226f93f6563c4899b9ebdd44bced1d73d5f52a5e472650ea.jpg)

![e92069b839357485646f197d505288ed942216f1cae0c88aef0ada21644bde48.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/e92069b839357485646f197d505288ed942216f1cae0c88aef0ada21644bde48.jpg)

![e99a11aacd64ad54b6d5fdeface007d6d6dc061dfe77df6f5270ad93f02057fe.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/e99a11aacd64ad54b6d5fdeface007d6d6dc061dfe77df6f5270ad93f02057fe.jpg)

![ecc7b91332ee9b38bf7811e079cfe391e9d65e2935cbb2bd5dfb9e35cbba3490.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/ecc7b91332ee9b38bf7811e079cfe391e9d65e2935cbb2bd5dfb9e35cbba3490.jpg)

![f6f1f2ee0e013ba3df29cdec2480082f6b295e3f05cfed804240fcfe96cbf709.jpg](../icml_results/6017_XTransfer_%20Modality-Agnostic%20Few-Shot%20Model%20Transfer%20for%20Human%20Sensing%20at%20the%20Edge/tables/f6f1f2ee0e013ba3df29cdec2480082f6b295e3f05cfed804240fcfe96cbf709.jpg)

<a id="paper-5969"></a>

## Generalized Boundary FDR Control under Arbitrary Dependence: An Approach on Closure Principle


### Images

![3feb43b4ce47ac248a07dd3f611491073489d8436476870261701ee4307f892f.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/images/3feb43b4ce47ac248a07dd3f611491073489d8436476870261701ee4307f892f.jpg)

![50b540fb8bae9e2c618fcdaa6133e84d8080f5f5dc3699d2b0b476ceb4d4fbdf.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/images/50b540fb8bae9e2c618fcdaa6133e84d8080f5f5dc3699d2b0b476ceb4d4fbdf.jpg)

![720f378746504a26cdb59c4e5dc9524d3add212aab2f74f292dac58118c1de24.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/images/720f378746504a26cdb59c4e5dc9524d3add212aab2f74f292dac58118c1de24.jpg)

![745ce27bf2fee05be8df8f8a57b0121a16e286f9eba570ff609497631e1ffa53.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/images/745ce27bf2fee05be8df8f8a57b0121a16e286f9eba570ff609497631e1ffa53.jpg)

![bd094c39a9ba4e12937e5d43a1e5c0f703fa100737b21164c16fbe46f60b9599.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/images/bd094c39a9ba4e12937e5d43a1e5c0f703fa100737b21164c16fbe46f60b9599.jpg)

![e115b1c3106d586b021463bd7400f5e29f82983ab5ab34a0717dbddb40d9ef75.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/images/e115b1c3106d586b021463bd7400f5e29f82983ab5ab34a0717dbddb40d9ef75.jpg)

### Tables

![025444e90a566c2e5dc2dca70dbc977dc0ff23698ee56e7da0f477a8611303eb.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/025444e90a566c2e5dc2dca70dbc977dc0ff23698ee56e7da0f477a8611303eb.jpg)

![0a3c65bbac81414f7b59f29d448ed570de0dc62cf049dfde01639de9ccc6c936.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/0a3c65bbac81414f7b59f29d448ed570de0dc62cf049dfde01639de9ccc6c936.jpg)

![161af03ae5167997309e8ca41ba285cc774d5d05b73a154b826e3a428618ecff.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/161af03ae5167997309e8ca41ba285cc774d5d05b73a154b826e3a428618ecff.jpg)

![241d41256787ff73a44608daecb507bf94b5f9bf71407abf317e5e47b91225c9.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/241d41256787ff73a44608daecb507bf94b5f9bf71407abf317e5e47b91225c9.jpg)

![29bb9e4cfeb9833e10f12379233a52ef70d2f04450e60edf13aa7c5588dded2a.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/29bb9e4cfeb9833e10f12379233a52ef70d2f04450e60edf13aa7c5588dded2a.jpg)

![591b97b16c7fec1fd5e46e05640c938fcc73095bc00021ff3358678cf6022214.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/591b97b16c7fec1fd5e46e05640c938fcc73095bc00021ff3358678cf6022214.jpg)

![cb2817fc54022d9c4a4436cfbada30f69c445e629b1cf3b5f57ca170be72fc22.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/cb2817fc54022d9c4a4436cfbada30f69c445e629b1cf3b5f57ca170be72fc22.jpg)

![ee564bd10dc6b16a04723022c9e1951a3aa4d7ba0892dd05ebdeb8fa7054cc61.jpg](../icml_results/6018_Generalized%20Boundary%20FDR%20Control%20under%20Arbitrary%20Dependence_%20An%20Approach%20on%20Closure%20Principle/tables/ee564bd10dc6b16a04723022c9e1951a3aa4d7ba0892dd05ebdeb8fa7054cc61.jpg)

<a id="paper-5970"></a>

## GPan-LoRA: Gaussian Process Amortized Networks for Bayesian Low-Rank Adaptation in Large Language Models


### Images

![28f8eea353b9f28d850d72baa61a2777d2c6187265f362d22fd9b345b7efbf1b.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/images/28f8eea353b9f28d850d72baa61a2777d2c6187265f362d22fd9b345b7efbf1b.jpg)

![2be5429a367861fec78e31d28794c4ef17e06a467d670dfab9d8076b62efd109.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/images/2be5429a367861fec78e31d28794c4ef17e06a467d670dfab9d8076b62efd109.jpg)

![5410805d377a26bae720c689337d30fdd0601ec9bf572393d632469d1063d642.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/images/5410805d377a26bae720c689337d30fdd0601ec9bf572393d632469d1063d642.jpg)

![bafca576a858a8541893a82aeac65a59da3e041699dc3030007df14b1cd72559.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/images/bafca576a858a8541893a82aeac65a59da3e041699dc3030007df14b1cd72559.jpg)

![efcf7b2f045ec825357f6faa01ed6b779a58ce4f8e423cab9e1e031a0974fe78.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/images/efcf7b2f045ec825357f6faa01ed6b779a58ce4f8e423cab9e1e031a0974fe78.jpg)

### Tables

![01e1c1357c955a8049a359af15c66260ffd3ee5e5a6fd2ca9fb6a304eae4a969.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/01e1c1357c955a8049a359af15c66260ffd3ee5e5a6fd2ca9fb6a304eae4a969.jpg)

![02105d488c40e2e2a373652a9e96421bc6ec21b261e6d60656edaa9aa70b4c1d.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/02105d488c40e2e2a373652a9e96421bc6ec21b261e6d60656edaa9aa70b4c1d.jpg)

![026feac33a9ff90f29fee53879bd5eb32828f57a10f4bf0959b9a49c2d682a3e.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/026feac33a9ff90f29fee53879bd5eb32828f57a10f4bf0959b9a49c2d682a3e.jpg)

![617c3ac5b930d9f5ee57257987e3b4ac90bc0a33b27984a9e74e85045dafb232.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/617c3ac5b930d9f5ee57257987e3b4ac90bc0a33b27984a9e74e85045dafb232.jpg)

![7e350d0075ed48bd175f6ed14f5eebd7481c1d2f4b3ce8be1a6b9ec6b70bca0c.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/7e350d0075ed48bd175f6ed14f5eebd7481c1d2f4b3ce8be1a6b9ec6b70bca0c.jpg)

![861d6478355097f9a1ea7483943f6ddce316aa961d4474fdc0bd80bd65cf5564.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/861d6478355097f9a1ea7483943f6ddce316aa961d4474fdc0bd80bd65cf5564.jpg)

![ad6438c6c3227f2187dac12cfe3ca83994b4d80ebe37c1d2f0e016b46341c0db.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/ad6438c6c3227f2187dac12cfe3ca83994b4d80ebe37c1d2f0e016b46341c0db.jpg)

![be8389725890c75d395f64b3f642163c4fcf6cad0810a846a88d2bd543c2cfcd.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/be8389725890c75d395f64b3f642163c4fcf6cad0810a846a88d2bd543c2cfcd.jpg)

![cf08cd75f0a57f77d9e3610afa890169dedba4fba6603e0450e5bbb24f142091.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/cf08cd75f0a57f77d9e3610afa890169dedba4fba6603e0450e5bbb24f142091.jpg)

![eefaf0416cd07df59b57f4eab156c2f0662b746a2d168aada13bce15211f33d9.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/eefaf0416cd07df59b57f4eab156c2f0662b746a2d168aada13bce15211f33d9.jpg)

![fcfd23b02b660af6c996d386de0e33df32d17323827cca48696de9632f61835a.jpg](../icml_results/6019_GPan-LoRA_%20Gaussian%20Process%20Amortized%20Networks%20for%20Bayesian%20Low-Rank%20Adaptation%20in%20Large%20Language%20Mo/tables/fcfd23b02b660af6c996d386de0e33df32d17323827cca48696de9632f61835a.jpg)

<a id="paper-5971"></a>

## When Embedding-Based Defenses Fail: Rethinking Safety in LLM-Based Multi-Agent Systems


### Images

![07783a450b6022f747ef3884c9b0f7df3fdcb5a2b4d7a04376d2af461226411a.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/images/07783a450b6022f747ef3884c9b0f7df3fdcb5a2b4d7a04376d2af461226411a.jpg)

![f251bc69df4e57f1b9a9cde9d186e0868ec97450be85eb7967e5efdbf9616a0f.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/images/f251bc69df4e57f1b9a9cde9d186e0868ec97450be85eb7967e5efdbf9616a0f.jpg)

### Tables

![3d3bc2f216d3eb570130117627a0e7aea40b73d29285683dfb6017a43a197cfa.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/3d3bc2f216d3eb570130117627a0e7aea40b73d29285683dfb6017a43a197cfa.jpg)

![3f57e8bdc8f4b27a620b22113270a3488fa17fcc3d52cf98aafdabbbc39a025e.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/3f57e8bdc8f4b27a620b22113270a3488fa17fcc3d52cf98aafdabbbc39a025e.jpg)

![46b1ebc3bd7ad760994393a1eea8b989526d5956b16805461cb8393d45e64e8b.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/46b1ebc3bd7ad760994393a1eea8b989526d5956b16805461cb8393d45e64e8b.jpg)

![7a591da0ecb5cde8eeffc27d49ffea43dfcac492b072a6f68d08947390554ad8.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/7a591da0ecb5cde8eeffc27d49ffea43dfcac492b072a6f68d08947390554ad8.jpg)

![a01f2b9f175ca3376c8077d19bda82039238523b8e695971587550059114e2ec.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/a01f2b9f175ca3376c8077d19bda82039238523b8e695971587550059114e2ec.jpg)

![b37ad182ab487511312f96c8363787b2d88233d8408950592be61c27bd4fc53f.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/b37ad182ab487511312f96c8363787b2d88233d8408950592be61c27bd4fc53f.jpg)

![d8d7ccf2cbba8455624b12062e09231c126784bd17a262150fc4a2af611eed88.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/d8d7ccf2cbba8455624b12062e09231c126784bd17a262150fc4a2af611eed88.jpg)

![f6e6f3014d93d4c009b38c8bcc657a66d6fb28005948e0432606aceade9952f5.jpg](../icml_results/6020_When%20Embedding-Based%20Defenses%20Fail_%20Rethinking%20Safety%20in%20LLM-Based%20Multi-Agent%20Systems/tables/f6e6f3014d93d4c009b38c8bcc657a66d6fb28005948e0432606aceade9952f5.jpg)

<a id="paper-5972"></a>

## Controlled SDEs for Long-Horizon Motion Generation under Latent Decision Uncertainty


### Images

![03d5b571c0918d9a8238850e30316d00f437572b514915fab75d82e55077a18b.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/03d5b571c0918d9a8238850e30316d00f437572b514915fab75d82e55077a18b.jpg)

![3415df3cab08c2849d610405f2926c98058f287f1ab681e1f0cfcdf29da4f66d.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/3415df3cab08c2849d610405f2926c98058f287f1ab681e1f0cfcdf29da4f66d.jpg)

![79bd00db1022048cfff058e2be9670af07063e8ab88f63d45a92d71f6dc6d0eb.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/79bd00db1022048cfff058e2be9670af07063e8ab88f63d45a92d71f6dc6d0eb.jpg)

![9d40e96c1a2545ff1d311d3c2aafe89a49474f0ceb6478d20136998fd3cf1d64.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/9d40e96c1a2545ff1d311d3c2aafe89a49474f0ceb6478d20136998fd3cf1d64.jpg)

![ad6ba93ebe8ad1e66b83ccec3c9ae1428cfa9f4c62438375dc91875ef2bb30c3.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/ad6ba93ebe8ad1e66b83ccec3c9ae1428cfa9f4c62438375dc91875ef2bb30c3.jpg)

![c87f13c137fb1d4d51157f64a9255a052a7ae5d95d1a172348a383f51f1b72a4.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/c87f13c137fb1d4d51157f64a9255a052a7ae5d95d1a172348a383f51f1b72a4.jpg)

![dcad0031083364986a47c5ea75611f57c567e5c0743e3ad8cbb9d3e2348e5f43.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/dcad0031083364986a47c5ea75611f57c567e5c0743e3ad8cbb9d3e2348e5f43.jpg)

![f200c328dcee107b5b271e7439384a85fd845da3ee3a78ac59dda2b392dd9035.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/f200c328dcee107b5b271e7439384a85fd845da3ee3a78ac59dda2b392dd9035.jpg)

![ff368aaa1daf38eb183ad37667ac01d463cedb6d89a9f33b9743cadabafe270d.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/images/ff368aaa1daf38eb183ad37667ac01d463cedb6d89a9f33b9743cadabafe270d.jpg)

### Tables

![1cdf3b823c3db04476844c9b2397caf7822897b1849048a3484d5a00db5f56cc.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/1cdf3b823c3db04476844c9b2397caf7822897b1849048a3484d5a00db5f56cc.jpg)

![4a0c18c6dd2548a49e5c73fd6bce879b61e5174c52fd2b21fc625fca82a84928.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/4a0c18c6dd2548a49e5c73fd6bce879b61e5174c52fd2b21fc625fca82a84928.jpg)

![5a4c2460ac7953f5dcc81b7987fb4627e7d6a233d3cd9e692ecf0314adcddc35.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/5a4c2460ac7953f5dcc81b7987fb4627e7d6a233d3cd9e692ecf0314adcddc35.jpg)

![6075ba1c0870967947993577ce7adc130e1a8ebbdffdd1e0cb5c5d7edaa8dfc0.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/6075ba1c0870967947993577ce7adc130e1a8ebbdffdd1e0cb5c5d7edaa8dfc0.jpg)

![96559f5f15c6182663040dcf92e695033e63439ecb7973eb61d13146e70d42df.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/96559f5f15c6182663040dcf92e695033e63439ecb7973eb61d13146e70d42df.jpg)

![9970489cd1396a8d7d572ee3d9323bad3c64f699c27b120307c68c1bec189327.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/9970489cd1396a8d7d572ee3d9323bad3c64f699c27b120307c68c1bec189327.jpg)

![9e92eaec36fd185b26f08974127491540d66f2044b2a7eacc07b799769e531e6.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/9e92eaec36fd185b26f08974127491540d66f2044b2a7eacc07b799769e531e6.jpg)

![a45e0f9e5298eeee241870e91fb36010dd19455f905e3f7cbf8acfa122634f02.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/a45e0f9e5298eeee241870e91fb36010dd19455f905e3f7cbf8acfa122634f02.jpg)

![ad3d8590b884e57b6d0f67a8ce9b14f8e43c9bf14f3f02bb32da37293721428c.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/ad3d8590b884e57b6d0f67a8ce9b14f8e43c9bf14f3f02bb32da37293721428c.jpg)

![ae062ee19d68e84b0bb86c2c15ddefd17707767f441446293cc297690c75fe09.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/ae062ee19d68e84b0bb86c2c15ddefd17707767f441446293cc297690c75fe09.jpg)

![b793445f8a9f784ebeb12d31b60430afacebb8c33033df146b4cda487ceaae68.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/b793445f8a9f784ebeb12d31b60430afacebb8c33033df146b4cda487ceaae68.jpg)

![e3b67d6b550cb06f0a7e48615ab10bb6a6f43acc9eb84c300f8d9f3072e2b120.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/e3b67d6b550cb06f0a7e48615ab10bb6a6f43acc9eb84c300f8d9f3072e2b120.jpg)

![e606caa52be2613fb00c3f5118fbcde4aceab6d9692a97b7bb62b058505a5818.jpg](../icml_results/6021_Controlled%20SDEs%20for%20Long-Horizon%20Motion%20Generation%20under%20Latent%20Decision%20Uncertainty/tables/e606caa52be2613fb00c3f5118fbcde4aceab6d9692a97b7bb62b058505a5818.jpg)

<a id="paper-5973"></a>

## Subspace-Aware Feature Reshaping for Open-Set Graph Class-Incremental Learning

### Images

![1277ebbe25a2d360690d5a89ca1576bae542591ee749c78a0c1f742f4ae48371.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/1277ebbe25a2d360690d5a89ca1576bae542591ee749c78a0c1f742f4ae48371.jpg)

![27ce868e9b690c615b5422a07f679bb6fca2d53041ba8b7139ac90fa9fc947c8.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/27ce868e9b690c615b5422a07f679bb6fca2d53041ba8b7139ac90fa9fc947c8.jpg)

![3b70dad6c56e17b9ba73405cce3767cf1f5440d662e02ebc8f95cb1e241af06b.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/3b70dad6c56e17b9ba73405cce3767cf1f5440d662e02ebc8f95cb1e241af06b.jpg)

![59e2cc1eb849155aa61b5e2d4486298cf79ff35f845c9c069d99f6f24a3f4a7c.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/59e2cc1eb849155aa61b5e2d4486298cf79ff35f845c9c069d99f6f24a3f4a7c.jpg)

![5f1d5ca7513ad84ff6399644b4184219becc9756d8f6df75419fde3deaa321c2.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/5f1d5ca7513ad84ff6399644b4184219becc9756d8f6df75419fde3deaa321c2.jpg)

![a369c2b1bb8cf9a04d098a325e95da1f3d16b5ba5546beef6de44bd9831a9a4b.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/a369c2b1bb8cf9a04d098a325e95da1f3d16b5ba5546beef6de44bd9831a9a4b.jpg)

![e818a511227e3026c85720ea93a39420d02f0695a3b70245c5fa44836cbecff0.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/e818a511227e3026c85720ea93a39420d02f0695a3b70245c5fa44836cbecff0.jpg)

![ec2e381c6db33690cf049ca1736937a92947688710eb872ba8e81a12852d1285.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/ec2e381c6db33690cf049ca1736937a92947688710eb872ba8e81a12852d1285.jpg)

![f38373df7e10b259789d25e5b5b7d358a2eb3920e4d804836cffd6afe6a9f64f.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/f38373df7e10b259789d25e5b5b7d358a2eb3920e4d804836cffd6afe6a9f64f.jpg)

![f9fbe15420ec4e07f1b1c670452db14c129853450f3d569364855f801dc79a50.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/images/f9fbe15420ec4e07f1b1c670452db14c129853450f3d569364855f801dc79a50.jpg)

### Tables

![148d16c4a1cc1f62d4d3a634de783d461c6c582d88afa1b60c643021f0483f7e.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/148d16c4a1cc1f62d4d3a634de783d461c6c582d88afa1b60c643021f0483f7e.jpg)

![1953703f3afa2ae79d889448b5dbc44a0399243cffec8242b311c0f032001b71.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/1953703f3afa2ae79d889448b5dbc44a0399243cffec8242b311c0f032001b71.jpg)

![1a0569586097bff84f8d8687dae001e9abd6c60f48bee5a4d2ce2799b0cb8f75.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/1a0569586097bff84f8d8687dae001e9abd6c60f48bee5a4d2ce2799b0cb8f75.jpg)

![3eb1c8cb9107c272bb3f7c1eea60df34d6844e534fce10b6d27e063bbb46f0c6.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/3eb1c8cb9107c272bb3f7c1eea60df34d6844e534fce10b6d27e063bbb46f0c6.jpg)

![66ead100288996a57cd05bd8205bf892ecbc89761cc1a44d29722be4aaaf4cfa.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/66ead100288996a57cd05bd8205bf892ecbc89761cc1a44d29722be4aaaf4cfa.jpg)

![becc5a993a5514bca34d2374c7d340a98a56082e5fba0487442d84d42080f18d.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/becc5a993a5514bca34d2374c7d340a98a56082e5fba0487442d84d42080f18d.jpg)

![cfbcefb0a86569b834c3d02b0f03e9056bc7d8d56cb67b164095eaeddab3d9f1.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/cfbcefb0a86569b834c3d02b0f03e9056bc7d8d56cb67b164095eaeddab3d9f1.jpg)

![d33f374008de6a1c1c7b3a7cfe6d60a07cba817b24e5f1013b802b18fb0195a4.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/d33f374008de6a1c1c7b3a7cfe6d60a07cba817b24e5f1013b802b18fb0195a4.jpg)

![e75382a31f500460afd3b0462dbec9983cc9f567ed5ea9fcb3d89aa6a0dcf8ce.jpg](../icml_results/6022_Subspace-Aware%20Feature%20Reshaping%20for%20Open-Set%20Graph%20Class-Incremental%20Learning/tables/e75382a31f500460afd3b0462dbec9983cc9f567ed5ea9fcb3d89aa6a0dcf8ce.jpg)
