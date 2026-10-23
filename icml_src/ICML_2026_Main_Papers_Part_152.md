# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 152 of 197

## 目录 (Table of Contents)

1. [SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models](#paper-4984)
2. [NavOL: Navigation Policy with Online Imitation Learning](#paper-4985)
3. [Variational Adapter for Cross-modal Similarity Representation](#paper-4986)
4. [RAT+: Train Dense, Infer Sparse - Recurrence Augmented Attention for Dilated Inference](#paper-4987)
5. [When to Think, When to Speak: Learning Disclosure Policies for LLM Reasoning](#paper-4988)
6. [Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception](#paper-4989)
7. [Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring](#paper-4990)
8. [Topological Active Inference for Task Disambiguation](#paper-4991)
9. [EvoMAS: Heuristics in the Loop—Evolving Smarter Agentic Workflows](#paper-4992)
10. [Pix2Key: Controllable Open-Vocabulary Retrieval with Semantic Decomposition and Self-Supervised Visual Dictionary Learning](#paper-4993)
11. [Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack Efficiently](#paper-4994)
12. [EvoC2F: Compiling Tool Orchestration for Efficient and Evolvable LLM Agents](#paper-4995)
13. [TEAM: Temporal–Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](#paper-4996)
14. [Deep neural networks divide and conquer dihedral multiplication](#paper-4997)
15. [The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search](#paper-4998)
16. [PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception](#paper-4999)
17. [Off-Policy Evaluation for Missingness-Aware Policies in MDPs with Rewards Missing Not at Random](#paper-5000)
18. [SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](#paper-5001)
19. [Toward Training Superintelligent Software Agents through Self-Play SWE-RL](#paper-5002)
20. [Return of Frustratingly Easy Unsupervised Video Domain Adaptation](#paper-5003)
21. [ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics](#paper-5004)
22. [Words & Weights: Streamlining Multi-Turn Interactions via Co-Adaptation](#paper-5005)
23. [TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning](#paper-5006)
24. [A-MemGuard: A Proactive Defense Framework For LLM-Based Agent Memory](#paper-5007)
25. [PSMix: Robust Point Cloud Recognition through Spectral Domain Mixing](#paper-5008)
26. [Token-Free Hierarchical Indexing for RAG beyond LLM-based Summarization](#paper-5009)
27. [Open-Text Aerial Detection: A Unified Framework For Aerial Visual Grounding And Detection](#paper-5010)
28. [GeoLoom: High-quality Geometric Diagram Generation from Textual Input](#paper-5011)
29. [RECOVER: Reliable Detection of Unauthorized Data Usage in Text-to-Image Diffusion Models via Inversion Robustness](#paper-5012)
30. [Exploration-free Algorithms for Multi-group Mean Estimation](#paper-5013)
31. [A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization](#paper-5014)
32. [Scalable Bayesian Inference for Nonlinear Conservation Laws](#paper-5015)
33. [The Role of Target Update Frequencies in Q-Learning](#paper-5016)

---


<a id="paper-4984"></a>

## SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models

### Images

![15121900faef2dc1691441d9f4a892e5584ea49e6262c2116cfba12a6e1f4140.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/15121900faef2dc1691441d9f4a892e5584ea49e6262c2116cfba12a6e1f4140.jpg)

![1e67185740bcde24e4fb8c37a8bd14779eb0f4959ec5e9749055458bb7c3d625.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/1e67185740bcde24e4fb8c37a8bd14779eb0f4959ec5e9749055458bb7c3d625.jpg)

![2a3f26c56b45b21088e73fcb5472b5a6e0991d8766bb74e12494a14276c0ea54.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/2a3f26c56b45b21088e73fcb5472b5a6e0991d8766bb74e12494a14276c0ea54.jpg)

![2ff7309ffcde7c2801f8f3a290a6f9b6408ed578e3a3c5b0233f68405bfd6e3e.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/2ff7309ffcde7c2801f8f3a290a6f9b6408ed578e3a3c5b0233f68405bfd6e3e.jpg)

![45711066f8c00936af9cc61a8d1eb3d1b434e11399276781129154be873c6f63.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/45711066f8c00936af9cc61a8d1eb3d1b434e11399276781129154be873c6f63.jpg)

![56a51c24a6e1096928fdaa2cf8b9de9eb0cc39ba87479aeb20e9864be33f70d8.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/56a51c24a6e1096928fdaa2cf8b9de9eb0cc39ba87479aeb20e9864be33f70d8.jpg)

![5cf0785d3a9114d09cca16f3e8581a6580061f0c4497c43a9380ede7e6e8b78b.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/5cf0785d3a9114d09cca16f3e8581a6580061f0c4497c43a9380ede7e6e8b78b.jpg)

![6b874e73ad21990043d796246572cbe08e3093c0a88993aa1076de56ea96af2c.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/6b874e73ad21990043d796246572cbe08e3093c0a88993aa1076de56ea96af2c.jpg)

![86d33cbc63c32fdeaa87a5fa73806e8a8998acbda643595c2dae5cff15d46457.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/86d33cbc63c32fdeaa87a5fa73806e8a8998acbda643595c2dae5cff15d46457.jpg)

![8ca0b8d9a82b9b517583cdc3ffffdf43859f806c8cf961a889690e2f80b6ccb2.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/8ca0b8d9a82b9b517583cdc3ffffdf43859f806c8cf961a889690e2f80b6ccb2.jpg)

![a266cd744d9ae8c4fd34256cbfab7af0b1eb7232f243ddd845f9f315985caec5.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/a266cd744d9ae8c4fd34256cbfab7af0b1eb7232f243ddd845f9f315985caec5.jpg)

![c036164ca29f36d06f15e4dd5c69a991ffd11f8fecab5c78b51e5bc35465481c.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/c036164ca29f36d06f15e4dd5c69a991ffd11f8fecab5c78b51e5bc35465481c.jpg)

![ca2073c57928cd2fdab84ccd641ff6dcc759a0325f424f1e51635e421b6ff1f9.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/ca2073c57928cd2fdab84ccd641ff6dcc759a0325f424f1e51635e421b6ff1f9.jpg)

![caa58a4aae9d254c49830280b5691e82b0fd1f7dd35379eb742315c969a2869e.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/caa58a4aae9d254c49830280b5691e82b0fd1f7dd35379eb742315c969a2869e.jpg)

![f9b76c2ccc1e461bba2c778290942dd5d642974fda1a80781ab8110bb20baa83.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/images/f9b76c2ccc1e461bba2c778290942dd5d642974fda1a80781ab8110bb20baa83.jpg)

### Tables

![766314e1faea74685475389aae5b1b0394a8ace631723556ded682846bbe09fe.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/tables/766314e1faea74685475389aae5b1b0394a8ace631723556ded682846bbe09fe.jpg)

![82b7846f247d7707a28ebf9b1839eb82f4f4291460db5a4f21828de02ec968cd.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/tables/82b7846f247d7707a28ebf9b1839eb82f4f4291460db5a4f21828de02ec968cd.jpg)

![a3e81c9c8ae9c14bb05a35cdb2125b414bc5af42bc57d6961530b21c0c7cea2f.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/tables/a3e81c9c8ae9c14bb05a35cdb2125b414bc5af42bc57d6961530b21c0c7cea2f.jpg)

![b5e667138a7fb7c294b5978c9d79983c077f9fb641bb4d146d4c1c6a9c99dad7.jpg](../icml_results/5031_Dynamic%20Compression%20Flows%20for%20Neuroscience%20Data/tables/b5e667138a7fb7c294b5978c9d79983c077f9fb641bb4d146d4c1c6a9c99dad7.jpg)

## SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models


### Images

![0f351fe003a83f36c7a03b9b111a225b5be20d403b6f902733fa601102657e53.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/0f351fe003a83f36c7a03b9b111a225b5be20d403b6f902733fa601102657e53.jpg)

![52a8df2af715ee4f564b0039e7989604ddcf4806e774f255a62ac26d4c31861c.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/52a8df2af715ee4f564b0039e7989604ddcf4806e774f255a62ac26d4c31861c.jpg)

![88eda7c4437e212f3be93331ce43ac94bff6728fd16505520c5376fcbf34e3ed.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/88eda7c4437e212f3be93331ce43ac94bff6728fd16505520c5376fcbf34e3ed.jpg)

![98e3b7c2a0b56edf3a7facb67677301b93cc5fa0b0063269fd4c8750149ed70a.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/98e3b7c2a0b56edf3a7facb67677301b93cc5fa0b0063269fd4c8750149ed70a.jpg)

![af3f8746ad9a25396b80b52fe83695a5b60480fbc4ce2f458e9398b9a1d6d9c6.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/af3f8746ad9a25396b80b52fe83695a5b60480fbc4ce2f458e9398b9a1d6d9c6.jpg)

![b2ad24c25a5026040f34ed6b5b296f3dba525dfc97a10fb0a3f6a80fa556a459.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/b2ad24c25a5026040f34ed6b5b296f3dba525dfc97a10fb0a3f6a80fa556a459.jpg)

![b8ac7f2e142d86490a7c8c20de5bdf5435fd73fbf926c783d84baaa13a21ac0a.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/b8ac7f2e142d86490a7c8c20de5bdf5435fd73fbf926c783d84baaa13a21ac0a.jpg)

![b9eb7c42448e770c4836e60aeaf427918d5f6d9243b436a1dc0a3d6a88f25a3f.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/b9eb7c42448e770c4836e60aeaf427918d5f6d9243b436a1dc0a3d6a88f25a3f.jpg)

![dd46bb522a24eaa91ac80a66e96c72e1b413fbf9f4b3b717fb5afcff49212e98.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/images/dd46bb522a24eaa91ac80a66e96c72e1b413fbf9f4b3b717fb5afcff49212e98.jpg)

### Tables

![0525efb59b22d2c07289a42bd6f5f5f1bec0033571a032ac128b7b87e623d777.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/0525efb59b22d2c07289a42bd6f5f5f1bec0033571a032ac128b7b87e623d777.jpg)

![33d0975b68701ca76a4317f28f699f6f03489d4a2a676f5d847190635c6fc3f0.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/33d0975b68701ca76a4317f28f699f6f03489d4a2a676f5d847190635c6fc3f0.jpg)

![3c2c25c1f334dcebe35defdc663da28e37eea32042674d98aa53b4fe683e8775.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/3c2c25c1f334dcebe35defdc663da28e37eea32042674d98aa53b4fe683e8775.jpg)

![6574487fdbe3c2f10af58050f9a3281fdbe48d0f18c113ec445f1b91dfeac8ba.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/6574487fdbe3c2f10af58050f9a3281fdbe48d0f18c113ec445f1b91dfeac8ba.jpg)

![7b02327ce24fb2faa6e4049ee909d9031257b369dddfe9a7b9d74eb1b538e77a.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/7b02327ce24fb2faa6e4049ee909d9031257b369dddfe9a7b9d74eb1b538e77a.jpg)

![987c6e78211ceda8f6b9abf3af4b73094d0702db78b157488cf42c11299d315b.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/987c6e78211ceda8f6b9abf3af4b73094d0702db78b157488cf42c11299d315b.jpg)

![9a028425693482fc87b9dd7ed060363bb53abc33eb064b0335ef89850c904bc8.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/9a028425693482fc87b9dd7ed060363bb53abc33eb064b0335ef89850c904bc8.jpg)

![cf3d3566f5b6b97465b127ff7eb67d848374be6ded6b92407d68b49724366433.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/cf3d3566f5b6b97465b127ff7eb67d848374be6ded6b92407d68b49724366433.jpg)

![d755db2d3025542f00cb1f7b5f8b47c2a597a254f525a717851bd902dbe30e95.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/d755db2d3025542f00cb1f7b5f8b47c2a597a254f525a717851bd902dbe30e95.jpg)

![ebcf2d72ef272b5e10f36316cad34b5defdda3e4d7d9fe6fc65a06e400c3c9c7.jpg](../icml_results/5032_SSR-Merge_%20Subspace%20Signal%20Routing%20for%20Training-Free%20LoRA%20Merging%20in%20Diffusion%20Models/tables/ebcf2d72ef272b5e10f36316cad34b5defdda3e4d7d9fe6fc65a06e400c3c9c7.jpg)

<a id="paper-4985"></a>

## NavOL: Navigation Policy with Online Imitation Learning


### Images

![00d9d2e6508471492dac60d8a2c2fe85cf716d9aa1dc6dbfe9bc544e1cb98cfd.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/00d9d2e6508471492dac60d8a2c2fe85cf716d9aa1dc6dbfe9bc544e1cb98cfd.jpg)

![05a379f60cb9f82ab198b1ba425c882e2ab662296e2a117549587bd03c54fede.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/05a379f60cb9f82ab198b1ba425c882e2ab662296e2a117549587bd03c54fede.jpg)

![22ce51d1457b82afd1657bccb849e874f928677099a03469fa2e7fa2e8d8cb54.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/22ce51d1457b82afd1657bccb849e874f928677099a03469fa2e7fa2e8d8cb54.jpg)

![2bc7838bf0468788990f762516879ca3bf62de8e74af97d86407a0045f5200fc.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/2bc7838bf0468788990f762516879ca3bf62de8e74af97d86407a0045f5200fc.jpg)

![37e7cdf6a731f3608083c1aa23e9e63f90f736e3d450729be222bd831f46e8c2.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/37e7cdf6a731f3608083c1aa23e9e63f90f736e3d450729be222bd831f46e8c2.jpg)

![3fe9500ea77ef45788f45c0e27bc86af8dd4083c0ab070f56da98afdf51c62e0.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/3fe9500ea77ef45788f45c0e27bc86af8dd4083c0ab070f56da98afdf51c62e0.jpg)

![4100ea8a1b79454363b602158719981b119f8135705fdc25c25d10bd1ff605e4.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/4100ea8a1b79454363b602158719981b119f8135705fdc25c25d10bd1ff605e4.jpg)

![55126dbb5e6dd807dc026cc6e2f04a04904e13faa0c9c2933100af5d34879824.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/55126dbb5e6dd807dc026cc6e2f04a04904e13faa0c9c2933100af5d34879824.jpg)

![66be25c465bac3189f4bd999925cc09878dc8f40722c25dad19b6832053b93d6.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/66be25c465bac3189f4bd999925cc09878dc8f40722c25dad19b6832053b93d6.jpg)

![67912fc55a003c26f4c8e6f496c2aa1ffc3d892604a22228b9cb8ebcdf9ab328.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/67912fc55a003c26f4c8e6f496c2aa1ffc3d892604a22228b9cb8ebcdf9ab328.jpg)

![6aab7efb8e751a31e79d472afc52e3bbe67e01a6d1a0de2d5b10073b5d53dff2.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/6aab7efb8e751a31e79d472afc52e3bbe67e01a6d1a0de2d5b10073b5d53dff2.jpg)

![df2207ee0b591e805634bf31dcb54aba7f54e46cfbf9f4da104089b27ea5d299.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/df2207ee0b591e805634bf31dcb54aba7f54e46cfbf9f4da104089b27ea5d299.jpg)

![ee85d0c88741c9a9f352a7d0febb6bdd8faafbdc61d569998cf209e4dd7da949.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/images/ee85d0c88741c9a9f352a7d0febb6bdd8faafbdc61d569998cf209e4dd7da949.jpg)

### Tables

![02104bd2815d2c02c725cdfa70ddedef22d971f09022785591a3fc34296db698.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/tables/02104bd2815d2c02c725cdfa70ddedef22d971f09022785591a3fc34296db698.jpg)

![27cb4290c3b5b85c3e874cc13b577ee75ac86d6b2813987d5f9d05aa0dd37a58.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/tables/27cb4290c3b5b85c3e874cc13b577ee75ac86d6b2813987d5f9d05aa0dd37a58.jpg)

![4909818a173859be5d5439fb790988a7a0fb6db2871a85c98c44110f7a20e999.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/tables/4909818a173859be5d5439fb790988a7a0fb6db2871a85c98c44110f7a20e999.jpg)

![62187e31f43286edefa662d90b07a76a454c07bb2a60e1d64223cb14a8c40631.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/tables/62187e31f43286edefa662d90b07a76a454c07bb2a60e1d64223cb14a8c40631.jpg)

![6d3b4055b5a5f191b3c0fe2c2436a0cdb2b768fe707e4ee6990968d574d5ebc5.jpg](../icml_results/5033_NavOL_%20Navigation%20Policy%20with%20Online%20Imitation%20Learning/tables/6d3b4055b5a5f191b3c0fe2c2436a0cdb2b768fe707e4ee6990968d574d5ebc5.jpg)

<a id="paper-4986"></a>

## Variational Adapter for Cross-modal Similarity Representation


### Images

![1b5ff81c4f7ab4e1a79e0d3dce8ebb1ccdc387253a44124f1805320eaf22764c.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/1b5ff81c4f7ab4e1a79e0d3dce8ebb1ccdc387253a44124f1805320eaf22764c.jpg)

![5708ac4b6a8bffc299ac5ca6f22c7cc0939099895e0ed8916c1ad1b4ff1596e3.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/5708ac4b6a8bffc299ac5ca6f22c7cc0939099895e0ed8916c1ad1b4ff1596e3.jpg)

![610e9bf1cfefb4ced556de0a9651f29b800a88cc933f1d761f0787d4cfa338fc.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/610e9bf1cfefb4ced556de0a9651f29b800a88cc933f1d761f0787d4cfa338fc.jpg)

![72631c01faa8c797f3f25817c6ec4e4767b4540d337cf4f655ff7726b3636792.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/72631c01faa8c797f3f25817c6ec4e4767b4540d337cf4f655ff7726b3636792.jpg)

![8fd6c6cddad8b3ea219a4164d32aa999f52f1c14d71e054f403c0345d349a3c3.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/8fd6c6cddad8b3ea219a4164d32aa999f52f1c14d71e054f403c0345d349a3c3.jpg)

![9a799a95809230c78ef43c17f71dc58066e97824a067c183dec3a5c43d4023c9.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/9a799a95809230c78ef43c17f71dc58066e97824a067c183dec3a5c43d4023c9.jpg)

![b1233466dd76b4719a7cae7c10981daae4f8230ac7d98ad42c8f2f288c9af1ba.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/b1233466dd76b4719a7cae7c10981daae4f8230ac7d98ad42c8f2f288c9af1ba.jpg)

![ca55b0fcb60fe174594932041d6d31fa49f562dae807f4de842966d63aa71a1f.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/ca55b0fcb60fe174594932041d6d31fa49f562dae807f4de842966d63aa71a1f.jpg)

![f04166dacaa8a8da9d7a72cb222044f5c28334de4b9baf276f401b98cb1c7660.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/f04166dacaa8a8da9d7a72cb222044f5c28334de4b9baf276f401b98cb1c7660.jpg)

![f5bff53738775bcd2adb6d6df79c2d90bb89bdd22a6272a98dc959a5e521564e.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/f5bff53738775bcd2adb6d6df79c2d90bb89bdd22a6272a98dc959a5e521564e.jpg)

![f83c340f8a1912a78ac8e2d816ebcc5aa14f0053b7b1b98b1f10cd94799ecf27.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/images/f83c340f8a1912a78ac8e2d816ebcc5aa14f0053b7b1b98b1f10cd94799ecf27.jpg)

### Tables

![0487c0e887ef0526b0fb25be1c28c1dee8c8e69e77cc4719c072ff94d424fe26.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/0487c0e887ef0526b0fb25be1c28c1dee8c8e69e77cc4719c072ff94d424fe26.jpg)

![0e37684f33d57e3820e9848cd274b7551f527d2c87f8c1f7e5a4199cc6ad901e.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/0e37684f33d57e3820e9848cd274b7551f527d2c87f8c1f7e5a4199cc6ad901e.jpg)

![27fa6ce90d2f67a493c4f44b51327bb74c7ecd9a7affb4d42f80ed200e3a59ae.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/27fa6ce90d2f67a493c4f44b51327bb74c7ecd9a7affb4d42f80ed200e3a59ae.jpg)

![4407ecaf60e57f85e5a2b69bb83e6cfa691aecfa2079b5d7fc55555b2e62a891.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/4407ecaf60e57f85e5a2b69bb83e6cfa691aecfa2079b5d7fc55555b2e62a891.jpg)

![5c4256531743de170f88eabc4d95fe0923a1902ff35bc9362f653f9bdea09a06.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/5c4256531743de170f88eabc4d95fe0923a1902ff35bc9362f653f9bdea09a06.jpg)

![7631c090c08aaf7460991079ec9dd52a9fdec399dd577d8b2567399bcfb04d5d.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/7631c090c08aaf7460991079ec9dd52a9fdec399dd577d8b2567399bcfb04d5d.jpg)

![809455020f7fed8eb076a1a77b3873e8e79f87f61ce6f5177181a38b323c654c.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/809455020f7fed8eb076a1a77b3873e8e79f87f61ce6f5177181a38b323c654c.jpg)

![90ef3937bb28c070e94035bada7d3bc3955a5ffdd55a1db0df1903870dc1671b.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/90ef3937bb28c070e94035bada7d3bc3955a5ffdd55a1db0df1903870dc1671b.jpg)

![dc19807361d07be6702c9b5ccf5fdb57b5bbb475f0a2b6107af8a24583c96220.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/dc19807361d07be6702c9b5ccf5fdb57b5bbb475f0a2b6107af8a24583c96220.jpg)

![e1a07a57f277749977542b81b87bb2665880932c10b22c09339a9e17016dc5b0.jpg](../icml_results/5034_Variational%20Adapter%20for%20Cross-modal%20Similarity%20Representation/tables/e1a07a57f277749977542b81b87bb2665880932c10b22c09339a9e17016dc5b0.jpg)

<a id="paper-4987"></a>

## RAT+: Train Dense, Infer Sparse - Recurrence Augmented Attention for Dilated Inference


### Images

![2e254f1967c89ffb0e7b270599421ea30fc09b553bbe8e5e3a7476d3bc9af86f.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/2e254f1967c89ffb0e7b270599421ea30fc09b553bbe8e5e3a7476d3bc9af86f.jpg)

![3e55741ec54c8d59f2e85d134d9c4862a30ad084d55c4881a0e3e349e9a1b499.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/3e55741ec54c8d59f2e85d134d9c4862a30ad084d55c4881a0e3e349e9a1b499.jpg)

![4a7edece84ce9e01c64058adbaba87180042de2f1280270ec5b7f0cebbcac93e.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/4a7edece84ce9e01c64058adbaba87180042de2f1280270ec5b7f0cebbcac93e.jpg)

![5d7997b20864a8016eee3c687c4b5ca8cfdd540d57eb3bb3f03d8321fb92f542.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/5d7997b20864a8016eee3c687c4b5ca8cfdd540d57eb3bb3f03d8321fb92f542.jpg)

![66e3707189e4efcdf7929946c5f660d70d9549cde8534e1f8fff35d0b42a0542.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/66e3707189e4efcdf7929946c5f660d70d9549cde8534e1f8fff35d0b42a0542.jpg)

![737d769ddb98aab98a234350a58bd7a41d3b4d6e61b74deb1110533f75a975e6.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/737d769ddb98aab98a234350a58bd7a41d3b4d6e61b74deb1110533f75a975e6.jpg)

![ba2e6076147d74f750f7e643a1879ffe4922cb500a6bd42849b96a2942c96667.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/ba2e6076147d74f750f7e643a1879ffe4922cb500a6bd42849b96a2942c96667.jpg)

![f1f7b2f90b55e3f4ad37395a4d3f0635d5c7d430ef5e1433249da5a3b9be0a2d.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/f1f7b2f90b55e3f4ad37395a4d3f0635d5c7d430ef5e1433249da5a3b9be0a2d.jpg)

![f801133b0f1d7ac5c3debb59db6082486d22ded8283da2e4a026ad8aee9f514e.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/images/f801133b0f1d7ac5c3debb59db6082486d22ded8283da2e4a026ad8aee9f514e.jpg)

### Tables

![04e0a1e033bdc1372b76e857eaed5bc6a2c4c7fbda591d156294564323bf3224.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/04e0a1e033bdc1372b76e857eaed5bc6a2c4c7fbda591d156294564323bf3224.jpg)

![081400dc23331cd9f08d28079451253fffdcd9d531c9ebdee54135e4a00abc71.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/081400dc23331cd9f08d28079451253fffdcd9d531c9ebdee54135e4a00abc71.jpg)

![18f105ac4fc0fd435012478acb86d1082e142e4008d6a4f7cfd9723092329916.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/18f105ac4fc0fd435012478acb86d1082e142e4008d6a4f7cfd9723092329916.jpg)

![23a76e3d825b3b0c7cc314216cdd747a8b1da0700d6f973ca14511a58c5d5191.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/23a76e3d825b3b0c7cc314216cdd747a8b1da0700d6f973ca14511a58c5d5191.jpg)

![3d7ec192c2548140383156154709b644fbc8a64b16fae635c0c4261355be6b3f.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/3d7ec192c2548140383156154709b644fbc8a64b16fae635c0c4261355be6b3f.jpg)

![3da8d8064dd974580b604293d937b5e41db7943443fc5e147999e54226cb96d8.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/3da8d8064dd974580b604293d937b5e41db7943443fc5e147999e54226cb96d8.jpg)

![45a57ba89eadde7d9a0d67bb3b9e54d1e8023593c6c9a07d4aad4c1daa1c1ce8.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/45a57ba89eadde7d9a0d67bb3b9e54d1e8023593c6c9a07d4aad4c1daa1c1ce8.jpg)

![4c5fddeba62a5b9ccc5b7b566150da0cbbd3ead3f0c6a1f597a4022159081de9.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/4c5fddeba62a5b9ccc5b7b566150da0cbbd3ead3f0c6a1f597a4022159081de9.jpg)

![60ae468a6f2d12b7dd59d2ce4dd93d8401e74453e460eac046db56c95e49da8d.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/60ae468a6f2d12b7dd59d2ce4dd93d8401e74453e460eac046db56c95e49da8d.jpg)

![61804c96c0e43ac54cc8d4a89c35c93e758ca58a6d37fcc2197bfe3c2108e439.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/61804c96c0e43ac54cc8d4a89c35c93e758ca58a6d37fcc2197bfe3c2108e439.jpg)

![6b9777d2526eb72622ec866e92ebbdf52ee1e8b251f010eed96b2bf960ea5478.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/6b9777d2526eb72622ec866e92ebbdf52ee1e8b251f010eed96b2bf960ea5478.jpg)

![7a830f3a488738cf4a1b07e0ba022ef99cde69aa2961f8cc603552bfc017a67c.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/7a830f3a488738cf4a1b07e0ba022ef99cde69aa2961f8cc603552bfc017a67c.jpg)

![7b4c08afcbd1e4c5802530e5a3c6123dca3d44ea2cb6649cb0b57d3b26bdea4f.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/7b4c08afcbd1e4c5802530e5a3c6123dca3d44ea2cb6649cb0b57d3b26bdea4f.jpg)

![842d4761bda673b6437fa7afb7322acf992b545c4a2ef43133c5b8807fbcdba2.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/842d4761bda673b6437fa7afb7322acf992b545c4a2ef43133c5b8807fbcdba2.jpg)

![8f7249978f9f8eecb64eaaf73edba3fd956c8edfbdf887ef3ea4a73ffed14bab.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/8f7249978f9f8eecb64eaaf73edba3fd956c8edfbdf887ef3ea4a73ffed14bab.jpg)

![92046d61ef35646856ddba82b44f480aab64b1e45b8907315e4286b64b4b1aec.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/92046d61ef35646856ddba82b44f480aab64b1e45b8907315e4286b64b4b1aec.jpg)

![951db0fd33250eddd098ca47f4891e2322ff54335f66018c31581228a9bb3666.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/951db0fd33250eddd098ca47f4891e2322ff54335f66018c31581228a9bb3666.jpg)

![a408fde12910f57a3712c149d48984d8eb14deff22ded21312b8466a64739c34.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/a408fde12910f57a3712c149d48984d8eb14deff22ded21312b8466a64739c34.jpg)

![a48573d694d0d28521bf1fafc19a909f4378b148c8c704ea95d4863b936b027b.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/a48573d694d0d28521bf1fafc19a909f4378b148c8c704ea95d4863b936b027b.jpg)

![b161fd76734b12e24c6fb8d921120e64975eb6ae5e90684146995bb603fbceda.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/b161fd76734b12e24c6fb8d921120e64975eb6ae5e90684146995bb603fbceda.jpg)

![c068ab48f7afc8bcf255b20ebc7e6cbd222c97a2500e020e2742a627e9849796.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/c068ab48f7afc8bcf255b20ebc7e6cbd222c97a2500e020e2742a627e9849796.jpg)

![c4af0599658a849ebb788236ba4ca6b46accd5e96e33f0e6d763c88e8036db12.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/c4af0599658a849ebb788236ba4ca6b46accd5e96e33f0e6d763c88e8036db12.jpg)

![c8ee398c7115166d6b3d85db87209dc79889580ab18683bf9feb3ed23bf29ca5.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/c8ee398c7115166d6b3d85db87209dc79889580ab18683bf9feb3ed23bf29ca5.jpg)

![ca7bc60e6666993047dc9b6a2e4b948926bb02bc052ccca848e0c5b710e42ec3.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/ca7bc60e6666993047dc9b6a2e4b948926bb02bc052ccca848e0c5b710e42ec3.jpg)

![e3f32d79da93adbc28382aa82171193d84caf3a85a2686da6155124ef0012e4e.jpg](../icml_results/5035_RAT%2B_%20Train%20Dense%2C%20Infer%20Sparse%20-%20Recurrence%20Augmented%20Attention%20for%20Dilated%20Inference/tables/e3f32d79da93adbc28382aa82171193d84caf3a85a2686da6155124ef0012e4e.jpg)

<a id="paper-4988"></a>

## When to Think, When to Speak: Learning Disclosure Policies for LLM Reasoning


### Images

![25b70b51956b9200dcf8a7395e1f356789c87095c91bf3b757d9a96cb3d4520c.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/images/25b70b51956b9200dcf8a7395e1f356789c87095c91bf3b757d9a96cb3d4520c.jpg)

![caa84a1fff4a309fdfa32200ac4e28d0dc67c30ea74a0fcf19bf6885c47793cf.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/images/caa84a1fff4a309fdfa32200ac4e28d0dc67c30ea74a0fcf19bf6885c47793cf.jpg)

![d8ce7f6d341acc5ea3309b15cbbdbc8f9efb485da008ee8f20d50222022d0354.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/images/d8ce7f6d341acc5ea3309b15cbbdbc8f9efb485da008ee8f20d50222022d0354.jpg)

![f1cd2433d23951ab86bc0866ca2e46716312fcb0181f8fa34b3e0718dbca4cb1.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/images/f1cd2433d23951ab86bc0866ca2e46716312fcb0181f8fa34b3e0718dbca4cb1.jpg)

### Tables

![0ae2b6d3352778e057af77df2097506f6f84ec6592b6f1e635be722877ca023c.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/0ae2b6d3352778e057af77df2097506f6f84ec6592b6f1e635be722877ca023c.jpg)

![628b0c5047176381886f5130cc9b4b7cff994103fb91d585fb30b5242b3ecb7c.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/628b0c5047176381886f5130cc9b4b7cff994103fb91d585fb30b5242b3ecb7c.jpg)

![70c0b2322b4e735a3539e3caf274ec48cd42ffe5ee3250b17bc6c8ea2ceb848c.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/70c0b2322b4e735a3539e3caf274ec48cd42ffe5ee3250b17bc6c8ea2ceb848c.jpg)

![8b5b58978d4a4d10296a948bc5260300d68ab58600f2b520c4a1d96d8663d8f8.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/8b5b58978d4a4d10296a948bc5260300d68ab58600f2b520c4a1d96d8663d8f8.jpg)

![8d355e578e363cf48358c52e7e751c64b9474cd5909990f989072146377a4399.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/8d355e578e363cf48358c52e7e751c64b9474cd5909990f989072146377a4399.jpg)

![9b3624b69337860fdfb77491530cddd0a26a77e60a4cc195e95b9aa5a51c151e.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/9b3624b69337860fdfb77491530cddd0a26a77e60a4cc195e95b9aa5a51c151e.jpg)

![a09a0c90aa570993e880f4226f88b4421c4122da2c048de2820da61732106b4d.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/a09a0c90aa570993e880f4226f88b4421c4122da2c048de2820da61732106b4d.jpg)

![e034ddb598c6be6a371b7ec844744ea0e9490f1e3cda9b9fb379c81433f022b2.jpg](../icml_results/5036_When%20to%20Think%2C%20When%20to%20Speak_%20Learning%20Disclosure%20Policies%20for%20LLM%20Reasoning/tables/e034ddb598c6be6a371b7ec844744ea0e9490f1e3cda9b9fb379c81433f022b2.jpg)

<a id="paper-4989"></a>

## Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception


### Images

![0ab5243230538f6c475c0d1632b364d8766089aa6a0d6c021b78f0585a8756b6.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/0ab5243230538f6c475c0d1632b364d8766089aa6a0d6c021b78f0585a8756b6.jpg)

![0f30c63cd6cade6cb3f49cacc2fb5fc2233e10540e9e978feec5850e25118edf.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/0f30c63cd6cade6cb3f49cacc2fb5fc2233e10540e9e978feec5850e25118edf.jpg)

![3fd8d83770c52e6e4dfe520433d77cddbc0a29c9e71a6b66a48360cbf2a72759.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/3fd8d83770c52e6e4dfe520433d77cddbc0a29c9e71a6b66a48360cbf2a72759.jpg)

![49f0e6d087b4a2f9054c1b36959309ce0d006ce1020f710073e4a7d1ab09c69e.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/49f0e6d087b4a2f9054c1b36959309ce0d006ce1020f710073e4a7d1ab09c69e.jpg)

![4a17ce4ac0c8cce5fcd4343dee538641939acf17b59641976131f7caf79f58c6.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/4a17ce4ac0c8cce5fcd4343dee538641939acf17b59641976131f7caf79f58c6.jpg)

![6f19d0983b70eb2ee654445384be623336d8e5d0f6761c8913c617263dd616a7.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/6f19d0983b70eb2ee654445384be623336d8e5d0f6761c8913c617263dd616a7.jpg)

![99b3a7f6d6f9fb7fe4e0e22515cb075bc1b14546fc0203be926903430afad521.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/99b3a7f6d6f9fb7fe4e0e22515cb075bc1b14546fc0203be926903430afad521.jpg)

![9b1b135def140a0bcbea0004a1fe16c609939f3389ed566a64578862414822e1.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/9b1b135def140a0bcbea0004a1fe16c609939f3389ed566a64578862414822e1.jpg)

![a554ddc2a1e0a5efa6444171b5127e74dc8d45261354d1da94bb8305dfbf558b.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/a554ddc2a1e0a5efa6444171b5127e74dc8d45261354d1da94bb8305dfbf558b.jpg)

![a5892d1e279c674c29029550a463a30ad03170eb343dd5a654f235eec6eae169.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/a5892d1e279c674c29029550a463a30ad03170eb343dd5a654f235eec6eae169.jpg)

![a87f9c8722b27a035661274d053204f52ed2a64bdfecd6fc5795e787e2e881d1.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/a87f9c8722b27a035661274d053204f52ed2a64bdfecd6fc5795e787e2e881d1.jpg)

![ad89296191a2c3342636b3cd6dd371442c58e236b23d43772ec5f3c14f8fd2d9.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/ad89296191a2c3342636b3cd6dd371442c58e236b23d43772ec5f3c14f8fd2d9.jpg)

![b1816dc88aaf3ca34a8ea4260c5edf555bdfb047d72d8251b80f18f2c3b2587e.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/b1816dc88aaf3ca34a8ea4260c5edf555bdfb047d72d8251b80f18f2c3b2587e.jpg)

![b5512d091ddfd4dc38e4539fb937a03e45d04456d4c7abbae8fdb9e9534640ed.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/b5512d091ddfd4dc38e4539fb937a03e45d04456d4c7abbae8fdb9e9534640ed.jpg)

![b6de802c32cd7b438115557c9cc3d826d4a2c9f5aca817fe905117df82ea3898.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/b6de802c32cd7b438115557c9cc3d826d4a2c9f5aca817fe905117df82ea3898.jpg)

![c9e1c38fd5235ceb8fbf93ab3d80a5cf2cc3e7e52f3d4ca682a211be93d53f48.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/c9e1c38fd5235ceb8fbf93ab3d80a5cf2cc3e7e52f3d4ca682a211be93d53f48.jpg)

![f00ab0e1257927eec8a8f21b60f85d46f65bb8b43621ea328f13abc07fe6bba2.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/f00ab0e1257927eec8a8f21b60f85d46f65bb8b43621ea328f13abc07fe6bba2.jpg)

![f0b9dbbe464e29541261491bc66692f237721a1329f296ad2deb7893f3bade72.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/f0b9dbbe464e29541261491bc66692f237721a1329f296ad2deb7893f3bade72.jpg)

![fcb244c2d888f5d51be359e7ae4c2016f9f4ef435040d72b7a166d717f90c79c.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/fcb244c2d888f5d51be359e7ae4c2016f9f4ef435040d72b7a166d717f90c79c.jpg)

![fe5f2c7d4a7119eab67a4e1ab171ea07283f13c8ea7328c340345538a0cc4c3d.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/images/fe5f2c7d4a7119eab67a4e1ab171ea07283f13c8ea7328c340345538a0cc4c3d.jpg)

### Tables

![3a6da56058a73fc4d961a25b65d846178f2bcf901ff1064c716ee6b485088e9c.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/3a6da56058a73fc4d961a25b65d846178f2bcf901ff1064c716ee6b485088e9c.jpg)

![467a49633fb45cb754570631ab8bea684ea06433b313455900c8af90278c1ee0.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/467a49633fb45cb754570631ab8bea684ea06433b313455900c8af90278c1ee0.jpg)

![4de5be093b970edc671b59e1845d9fe2377c29aef50d22cd3dc089fa9f1b0faf.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/4de5be093b970edc671b59e1845d9fe2377c29aef50d22cd3dc089fa9f1b0faf.jpg)

![509256badc94befdd55931490f6c5f0b22982199d9197291a1afbe1f67bc71a9.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/509256badc94befdd55931490f6c5f0b22982199d9197291a1afbe1f67bc71a9.jpg)

![6ac856ac3393a588cbb0b4a0b25368ad14899b7b3562e47cf0431d8a913eaa4a.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/6ac856ac3393a588cbb0b4a0b25368ad14899b7b3562e47cf0431d8a913eaa4a.jpg)

![7d7b4416d9b9f9f66ac85bbce5f3a615531cc7dc9a9dc647f1df4bd388be6e2e.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/7d7b4416d9b9f9f66ac85bbce5f3a615531cc7dc9a9dc647f1df4bd388be6e2e.jpg)

![8afabd8687a82431a9ccc0afc1caf6100cfd31d9419a616cbe933c21c4e44309.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/8afabd8687a82431a9ccc0afc1caf6100cfd31d9419a616cbe933c21c4e44309.jpg)

![a96c2b9ea3696dc2acabcab8b79d1c77e6c9e4e85f9475d7a7636b50897cd627.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/a96c2b9ea3696dc2acabcab8b79d1c77e6c9e4e85f9475d7a7636b50897cd627.jpg)

![ed35da734558188282372936f169ab17796cfe72c4df661c70d806335b0fceb9.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/ed35da734558188282372936f169ab17796cfe72c4df661c70d806335b0fceb9.jpg)

![f35f0c009e6a9634c7405db938b9f3d4faf9f1167389679ad23a060bf395d892.jpg](../icml_results/5037_Zooming%20without%20Zooming_%20Region-to-Image%20Distillation%20for%20Fine-Grained%20Multimodal%20Perception/tables/f35f0c009e6a9634c7405db938b9f3d4faf9f1167389679ad23a060bf395d892.jpg)

<a id="paper-4990"></a>

## Rethinking Federated Prompt Learning for Medical Images: From Textual Tuning to Visual Manifold Anchoring


### Images

![3c7c2ddf2151d9cb73e6f1bb613bb73cc7f93b68941a53b5a43c90ad15a3f4f2.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/3c7c2ddf2151d9cb73e6f1bb613bb73cc7f93b68941a53b5a43c90ad15a3f4f2.jpg)

![3f08cdc6ed379fa8e6cef0489c55bf23c8d9411b7daf75d9bd3e6ec8c4379a40.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/3f08cdc6ed379fa8e6cef0489c55bf23c8d9411b7daf75d9bd3e6ec8c4379a40.jpg)

![7925d40d7edfe16e8eaf3d925148eef2d73ae50d447f607e85bf5f1a1e53e057.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/7925d40d7edfe16e8eaf3d925148eef2d73ae50d447f607e85bf5f1a1e53e057.jpg)

![97013d17e53c62577cc3b4c9092d971a9fea94a24590ec29d8b4b9e0a4e2b6cc.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/97013d17e53c62577cc3b4c9092d971a9fea94a24590ec29d8b4b9e0a4e2b6cc.jpg)

![a07a8c82dbeb4acde5d6d247e04f7020f8e482c984ec9da6f8a31484d268b38b.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/a07a8c82dbeb4acde5d6d247e04f7020f8e482c984ec9da6f8a31484d268b38b.jpg)

![a465b1ac80e7016c768052fac1e358dfcb2859fa4c1c2b33de4a82dcbd37be1e.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/a465b1ac80e7016c768052fac1e358dfcb2859fa4c1c2b33de4a82dcbd37be1e.jpg)

![aec5071b984ca22ffb1a0068ad0ede6e9f96f15eb1805043907c8cfb80fef13d.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/aec5071b984ca22ffb1a0068ad0ede6e9f96f15eb1805043907c8cfb80fef13d.jpg)

![b45c771eb1c6afc4ec28392d04dfff0313ba1f38b5dc355802b4d495a4c262cd.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/b45c771eb1c6afc4ec28392d04dfff0313ba1f38b5dc355802b4d495a4c262cd.jpg)

![d429179d5eb8ac45008e947cfa74b34781494fa3e4678e412fa7d30550ba7bba.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/d429179d5eb8ac45008e947cfa74b34781494fa3e4678e412fa7d30550ba7bba.jpg)

![e15c7ebc4cfb27d943f774482c6d7acc1f89651868b779d9239cac018b31ebc7.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/e15c7ebc4cfb27d943f774482c6d7acc1f89651868b779d9239cac018b31ebc7.jpg)

![efe00e5a6da90e7ac9bf45959b8481c0b794ce20f27ae1481c3754d78651266c.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/efe00e5a6da90e7ac9bf45959b8481c0b794ce20f27ae1481c3754d78651266c.jpg)

![fc50f7b66f7ac41e1a8bc08a6d4f09f37169412900237b7554c63d87728c0844.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/images/fc50f7b66f7ac41e1a8bc08a6d4f09f37169412900237b7554c63d87728c0844.jpg)

### Tables

![03cb535e999f3ec275f7fffbc8f4005f47914b4337ab4fd03ee0b72ad05f8506.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/03cb535e999f3ec275f7fffbc8f4005f47914b4337ab4fd03ee0b72ad05f8506.jpg)

![12c5db77f9fe3695074e8261223f243bfcc0149b04032569754e12b570e34a48.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/12c5db77f9fe3695074e8261223f243bfcc0149b04032569754e12b570e34a48.jpg)

![137c02c248085fd07e9ec7aefc8e4a5f8b855c14de9bb30280d1462a565b1fb1.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/137c02c248085fd07e9ec7aefc8e4a5f8b855c14de9bb30280d1462a565b1fb1.jpg)

![1468b5b783950848c701b08987c729267fc51e0cc479a0ae2ab0ea25ae766347.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/1468b5b783950848c701b08987c729267fc51e0cc479a0ae2ab0ea25ae766347.jpg)

![1ceea99456b42df6c27a64eda42b6c759828108f990400a122d26d0f8fe89eaa.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/1ceea99456b42df6c27a64eda42b6c759828108f990400a122d26d0f8fe89eaa.jpg)

![72bd4a6d5586b8a4f3bbe3dfec3be6578e87dc1f1fd2e329d444ad7e0b8e7818.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/72bd4a6d5586b8a4f3bbe3dfec3be6578e87dc1f1fd2e329d444ad7e0b8e7818.jpg)

![8c53a5598b09164da2bb87302ae838e8f10c7e35d4194012411800f8a3f8060d.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/8c53a5598b09164da2bb87302ae838e8f10c7e35d4194012411800f8a3f8060d.jpg)

![92df475551aa1010100dcfcc29c573adcd8ec387de243e29f91a8ab619a0d6c8.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/92df475551aa1010100dcfcc29c573adcd8ec387de243e29f91a8ab619a0d6c8.jpg)

![b9560ca3fe190c6252bbed3330eb926cfad771ff52470c03fe1cce44a794a7c9.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/b9560ca3fe190c6252bbed3330eb926cfad771ff52470c03fe1cce44a794a7c9.jpg)

![b98ebbeabeb18759f418b662f11130337567f8e063bb28b6d2f571b0c7ed1134.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/b98ebbeabeb18759f418b662f11130337567f8e063bb28b6d2f571b0c7ed1134.jpg)

![f69c9ae72f4e375de2c6fa0a83e7d68e59373196432dd1c680c83c3d2c594774.jpg](../icml_results/5038_Rethinking%20Federated%20Prompt%20Learning%20for%20Medical%20Images_%20From%20Textual%20Tuning%20to%20Visual%20Manifold%20Anch/tables/f69c9ae72f4e375de2c6fa0a83e7d68e59373196432dd1c680c83c3d2c594774.jpg)

<a id="paper-4991"></a>

## Topological Active Inference for Task Disambiguation


### Images

![1f0cb98738440728c57669a3c2d29c6504d44dbe5f7d5875902ee809fb1d6a54.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/1f0cb98738440728c57669a3c2d29c6504d44dbe5f7d5875902ee809fb1d6a54.jpg)

![1f1c2c2278afe5cf0bfdfb853f44599f6acba178629e786032c64d17dcfdbcb3.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/1f1c2c2278afe5cf0bfdfb853f44599f6acba178629e786032c64d17dcfdbcb3.jpg)

![4781942d6584c4d4d327420501f15b1ee4b93666bdd5073c0d98b78c6928bfe2.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/4781942d6584c4d4d327420501f15b1ee4b93666bdd5073c0d98b78c6928bfe2.jpg)

![5f154597c6ba43423fa78b7630751bb2ca7296442144267dc37c7ab0b3e88511.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/5f154597c6ba43423fa78b7630751bb2ca7296442144267dc37c7ab0b3e88511.jpg)

![8aa88552a6e12c3ad0bdb100589abcb5c5c020315a3d91198d5891b162bfe356.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/8aa88552a6e12c3ad0bdb100589abcb5c5c020315a3d91198d5891b162bfe356.jpg)

![93d9de0499b0f1a661adbd2e21bfc01cecdab625ee8cb94eed3c4936841a308a.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/93d9de0499b0f1a661adbd2e21bfc01cecdab625ee8cb94eed3c4936841a308a.jpg)

![a429361d7f2615b27f981f512da18aaeb1fed6e862671628e6e61d1c5e5269f9.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/a429361d7f2615b27f981f512da18aaeb1fed6e862671628e6e61d1c5e5269f9.jpg)

![dfa7e4746e33cfd37f9a554cf40134221893729c454a64b85f0bfd85ec9a4f87.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/dfa7e4746e33cfd37f9a554cf40134221893729c454a64b85f0bfd85ec9a4f87.jpg)

![f49af27bf15f85075de2a3e337e0dd0f742a39baf98fe0e9733d18c98cfd00ac.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/f49af27bf15f85075de2a3e337e0dd0f742a39baf98fe0e9733d18c98cfd00ac.jpg)

![f59ea4980553d04e73e2f99cbc263d78616907807410a789f8e0a1e4b73de24d.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/images/f59ea4980553d04e73e2f99cbc263d78616907807410a789f8e0a1e4b73de24d.jpg)

### Tables

![00388ad9db1fe742c662bf6c5dec4f87c3169a04b3b8e2c209bcbb743d8000cd.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/tables/00388ad9db1fe742c662bf6c5dec4f87c3169a04b3b8e2c209bcbb743d8000cd.jpg)

![27cf41c122a82450e7e10cffeab97dbe1066371dc14c756d39d48842d60c9fc9.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/tables/27cf41c122a82450e7e10cffeab97dbe1066371dc14c756d39d48842d60c9fc9.jpg)

![30853b11b04ab7697e8e6366046614c704d2dce0082f45417dffb7ae74de70aa.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/tables/30853b11b04ab7697e8e6366046614c704d2dce0082f45417dffb7ae74de70aa.jpg)

![3c7c9962caf84894747fd6889a2965b52a875ca1b4a95147aa8153a95456e017.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/tables/3c7c9962caf84894747fd6889a2965b52a875ca1b4a95147aa8153a95456e017.jpg)

![8e2e7abb965192bb3b54acc418eec8e04d63babce6d0057fde29b58073fb872d.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/tables/8e2e7abb965192bb3b54acc418eec8e04d63babce6d0057fde29b58073fb872d.jpg)

![fad7e5bf855d7aafd7872ea0c885949150b4404465060969007d144f70f35836.jpg](../icml_results/5039_Topological%20Active%20Inference%20for%20Task%20Disambiguation/tables/fad7e5bf855d7aafd7872ea0c885949150b4404465060969007d144f70f35836.jpg)

<a id="paper-4992"></a>

## EvoMAS: Heuristics in the Loop—Evolving Smarter Agentic Workflows


### Images

![02d6aaca1421c90424a693507bfceaad758c7baaf5405223cb5253a85559c4a6.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/02d6aaca1421c90424a693507bfceaad758c7baaf5405223cb5253a85559c4a6.jpg)

![1056919ecb36e460eb86f9596aa88c135e6220a31b734745cef4e95452f61276.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/1056919ecb36e460eb86f9596aa88c135e6220a31b734745cef4e95452f61276.jpg)

![22fc039d8f4f106a2624cd1494b4349a0e36cc6e563d75c2a60727c0dc1b849d.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/22fc039d8f4f106a2624cd1494b4349a0e36cc6e563d75c2a60727c0dc1b849d.jpg)

![319e4eed28731d76d040d06721f3e43904fd002bea299219c55e3198173e6bea.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/319e4eed28731d76d040d06721f3e43904fd002bea299219c55e3198173e6bea.jpg)

![345668e45dcd9be4d726c2e5803c0c73939508836f929bb9c8cdfecb9c2060b0.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/345668e45dcd9be4d726c2e5803c0c73939508836f929bb9c8cdfecb9c2060b0.jpg)

![5330667d189b2660123db83aeb6cca8458864fd23588ba3f87ed41752ba3881c.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/5330667d189b2660123db83aeb6cca8458864fd23588ba3f87ed41752ba3881c.jpg)

![c9a087eafbac787f442590af2416fd5551c2f0e34b4162178560c81e20d2d3b9.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/c9a087eafbac787f442590af2416fd5551c2f0e34b4162178560c81e20d2d3b9.jpg)

![f7ace3d808f4e405f7657eab530903dbbd7666518c7b1b28c4b10a3b3a5d9386.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/f7ace3d808f4e405f7657eab530903dbbd7666518c7b1b28c4b10a3b3a5d9386.jpg)

![fba1c65dbb1bc6678016c13d409bcdb6cbc9b3808cb99cd828defc1f9e7ed89c.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/images/fba1c65dbb1bc6678016c13d409bcdb6cbc9b3808cb99cd828defc1f9e7ed89c.jpg)

### Tables

![0ad916f384075880c702ee573e9bd40c965615676f42f341612ab319fc0d899c.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/0ad916f384075880c702ee573e9bd40c965615676f42f341612ab319fc0d899c.jpg)

![15693e230cd8626c92ec58d6dd69b0704eae98fcd67a9e37de85ed79be1bdb65.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/15693e230cd8626c92ec58d6dd69b0704eae98fcd67a9e37de85ed79be1bdb65.jpg)

![266cf345797c135159ca00d21b72d4fdd169dd7f1d55427f6b0f98c84e1d3264.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/266cf345797c135159ca00d21b72d4fdd169dd7f1d55427f6b0f98c84e1d3264.jpg)

![5ca189b47e674a9424c11b4fda7566aa6ccb7941bf2222bbf37c41ddab3a395d.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/5ca189b47e674a9424c11b4fda7566aa6ccb7941bf2222bbf37c41ddab3a395d.jpg)

![630b749a54a2de1a483abee9329b413880117cdfb19cf081a11a79a922ade3d9.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/630b749a54a2de1a483abee9329b413880117cdfb19cf081a11a79a922ade3d9.jpg)

![69c149e44d9afb2a387155631cf486fcb4cb888c139cadc5e4a1741dc8cbf6bb.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/69c149e44d9afb2a387155631cf486fcb4cb888c139cadc5e4a1741dc8cbf6bb.jpg)

![738735cb50eb6e3feff40046ea767fdaab853feb1fe82b69b4afec0655b7ea3b.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/738735cb50eb6e3feff40046ea767fdaab853feb1fe82b69b4afec0655b7ea3b.jpg)

![7fa5abe882df39ef27c1e73f963f49ab920e558143257a31f1d4164d8e30524f.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/7fa5abe882df39ef27c1e73f963f49ab920e558143257a31f1d4164d8e30524f.jpg)

![8b057e339230738c581b5d1a90dcb38cb9e68d46d98d66d3fd1d4b558f1e46f8.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/8b057e339230738c581b5d1a90dcb38cb9e68d46d98d66d3fd1d4b558f1e46f8.jpg)

![9cdc20a37f9c9920350c92179386e6d6748159a3f01f72547b27947a21440c16.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/9cdc20a37f9c9920350c92179386e6d6748159a3f01f72547b27947a21440c16.jpg)

![a677af74fee2aaae8370dc37484427c9cd31320659315c66cab0feb67e3833fd.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/a677af74fee2aaae8370dc37484427c9cd31320659315c66cab0feb67e3833fd.jpg)

![c1744d07ab5ad768f91334ecd34b6c28bc9e56ffa8a379a20e2dae1383d36800.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/c1744d07ab5ad768f91334ecd34b6c28bc9e56ffa8a379a20e2dae1383d36800.jpg)

![cc67334e8404660631941e48433b342e8e3814b65dbd3835520bfb519dd0442d.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/cc67334e8404660631941e48433b342e8e3814b65dbd3835520bfb519dd0442d.jpg)

![f44c0b1c677dd5d7a8f31c7d8b6ad5cdc49ae2e5bd4e92a5fc66663f85e15fed.jpg](../icml_results/5040_EvoMAS_%20Heuristics%20in%20the%20Loop%E2%80%94Evolving%20Smarter%20Agentic%20Workflows/tables/f44c0b1c677dd5d7a8f31c7d8b6ad5cdc49ae2e5bd4e92a5fc66663f85e15fed.jpg)

<a id="paper-4993"></a>

## Pix2Key: Controllable Open-Vocabulary Retrieval with Semantic Decomposition and Self-Supervised Visual Dictionary Learning


### Images

![062537a421ecfb52144c92f3f2cfd2c7bf6e0aeac1df559d404771b8768aa248.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/images/062537a421ecfb52144c92f3f2cfd2c7bf6e0aeac1df559d404771b8768aa248.jpg)

![1a39a613bad902abff2f7e8613c9cc99007219d9a9092ac949a75125ccbbdbcb.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/images/1a39a613bad902abff2f7e8613c9cc99007219d9a9092ac949a75125ccbbdbcb.jpg)

### Tables

![9d9feeb122b2865285348cf38500b9a6d80fae0f8d9e9cfd7a0af42bd343a6d2.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/9d9feeb122b2865285348cf38500b9a6d80fae0f8d9e9cfd7a0af42bd343a6d2.jpg)

![a57927b5ad0710ed69d4694a4a741de9c389fd05eec9ecca7bab26b6eb568809.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/a57927b5ad0710ed69d4694a4a741de9c389fd05eec9ecca7bab26b6eb568809.jpg)

![a99eabbef66a9074fe7ba5f2ad5d3a4c7f1ba128dd8034c47f93c477e3301960.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/a99eabbef66a9074fe7ba5f2ad5d3a4c7f1ba128dd8034c47f93c477e3301960.jpg)

![ca0d8e1d3704ab7e0ba9257d173d367e409f54e419e3e2d30e2078fcbb61a31b.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/ca0d8e1d3704ab7e0ba9257d173d367e409f54e419e3e2d30e2078fcbb61a31b.jpg)

![cfe3996dd7858626f6a08647ed25453241df6414958aa456c3f0628de5ed442e.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/cfe3996dd7858626f6a08647ed25453241df6414958aa456c3f0628de5ed442e.jpg)

![e021f4419275663c560f2c98ead23b11953d08baaf3b3e8d66b6132fbb47e0ed.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/e021f4419275663c560f2c98ead23b11953d08baaf3b3e8d66b6132fbb47e0ed.jpg)

![ebeb5611f7d21a23a5a6ae1aafd6d4f49ca3180927d01fa28d3091fce7564d35.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/ebeb5611f7d21a23a5a6ae1aafd6d4f49ca3180927d01fa28d3091fce7564d35.jpg)

![f200d540457e1494c2005a0771d405b797f5d8ba47ff9950b26b2c77a20bcd16.jpg](../icml_results/5041_Pix2Key_%20Controllable%20Open-Vocabulary%20Retrieval%20with%20Semantic%20Decomposition%20and%20Self-Supervised%20Visu/tables/f200d540457e1494c2005a0771d405b797f5d8ba47ff9950b26b2c77a20bcd16.jpg)

<a id="paper-4994"></a>

## Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack Efficiently


### Images

![09b67a3dfd0df9012c721c143103b37d87edfdd01e1381574ddb60cf4b67592c.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/09b67a3dfd0df9012c721c143103b37d87edfdd01e1381574ddb60cf4b67592c.jpg)

![0a45648aee78158337af98b705b70342250f0b6c6bb6fc0c0b496f96258f381b.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/0a45648aee78158337af98b705b70342250f0b6c6bb6fc0c0b496f96258f381b.jpg)

![1349bf9e38dacc49ecda41b2ddee34670f7e19b94a19900c3ad18a131e693707.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/1349bf9e38dacc49ecda41b2ddee34670f7e19b94a19900c3ad18a131e693707.jpg)

![27cae7d75b36dcf161dd166e0877d505dfd108dcfefedabf068845128451e60d.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/27cae7d75b36dcf161dd166e0877d505dfd108dcfefedabf068845128451e60d.jpg)

![630019d349c9c7e9c25906466e3e6dc6064b0df9c0d1ec1af738a355bf6ce0b2.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/630019d349c9c7e9c25906466e3e6dc6064b0df9c0d1ec1af738a355bf6ce0b2.jpg)

![72818a3060e14205849d61b23f0c2ff21df9fe7b9732c58fc4d1fc1615b0e83a.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/72818a3060e14205849d61b23f0c2ff21df9fe7b9732c58fc4d1fc1615b0e83a.jpg)

![7831d000894abc92ba389e450c3f0878d7baca327af1f4bef129a17d23f61168.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/7831d000894abc92ba389e450c3f0878d7baca327af1f4bef129a17d23f61168.jpg)

![a7ac782a67459225d113e229f1f62a7bb958d3c8f5ff43a3051ccbf211d0270b.jpg](../icml_results/5042_Provable%20Benefits%20of%20RLVR%20over%20SFT%20for%20Reasoning%20Models_%20Learning%20to%20Backtrack%20Efficiently/images/a7ac782a67459225d113e229f1f62a7bb958d3c8f5ff43a3051ccbf211d0270b.jpg)

<a id="paper-4995"></a>

## EvoC2F: Compiling Tool Orchestration for Efficient and Evolvable LLM Agents


### Images

![4f04f9e87d977f3e4285ae32cc6b42995e691cafe5fda64dffdfcd1bf94518bb.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/images/4f04f9e87d977f3e4285ae32cc6b42995e691cafe5fda64dffdfcd1bf94518bb.jpg)

### Tables

![05c3501eeee02950618631731d10dadbe4141fc673fdec4ff1a48e9c08ef5db8.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/05c3501eeee02950618631731d10dadbe4141fc673fdec4ff1a48e9c08ef5db8.jpg)

![075e79510796439330220e092380d25812ca8395caba2edf4fd82f908dd90774.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/075e79510796439330220e092380d25812ca8395caba2edf4fd82f908dd90774.jpg)

![21b6a0c2325856f3b12a9a8e974c5262216ab6780de703b23a8a496927318f22.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/21b6a0c2325856f3b12a9a8e974c5262216ab6780de703b23a8a496927318f22.jpg)

![9b454bd9b0bcafc9de29865424954c219aeb3a3985447101f2d253c8b1db8d80.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/9b454bd9b0bcafc9de29865424954c219aeb3a3985447101f2d253c8b1db8d80.jpg)

![a32db465f99bf54577d8e02bfbbd548845128f7f1f11cd065334b35bd981e1dd.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/a32db465f99bf54577d8e02bfbbd548845128f7f1f11cd065334b35bd981e1dd.jpg)

![a5ed13c0f770cf4ffdda7bd94bf33c396ed544f0ca54b9317c73a98db8571d50.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/a5ed13c0f770cf4ffdda7bd94bf33c396ed544f0ca54b9317c73a98db8571d50.jpg)

![c65223af71c500ab12239942bd6248971dfedfb7864b6220008d626deda300e7.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/c65223af71c500ab12239942bd6248971dfedfb7864b6220008d626deda300e7.jpg)

![d8e24c975c97394cb61b0e48d7048d2560be254647a11b1fd75d4173e3353a42.jpg](../icml_results/5043_EvoC2F_%20Compiling%20Tool%20Orchestration%20for%20Efficient%20and%20Evolvable%20LLM%20Agents/tables/d8e24c975c97394cb61b0e48d7048d2560be254647a11b1fd75d4173e3353a42.jpg)

<a id="paper-4996"></a>

## TEAM: Temporal–Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration


### Images

![1814b80fd837f2ea00a0940d9cbf09ba3ae87c775738198a76ca1fd46f8e82e9.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/1814b80fd837f2ea00a0940d9cbf09ba3ae87c775738198a76ca1fd46f8e82e9.jpg)

![20aa370c8cb35c2cb88447f285f5e6d051607b5c371001bc2fd146417dbeb79b.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/20aa370c8cb35c2cb88447f285f5e6d051607b5c371001bc2fd146417dbeb79b.jpg)

![3153ac45114d0c58b682e76fb9879c9059adf242d1f1b0f1b3e87939f62ee748.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/3153ac45114d0c58b682e76fb9879c9059adf242d1f1b0f1b3e87939f62ee748.jpg)

![5289bf025a6c262c7b5d5ce8bc1b2093104b8cbfa9874dae3b44a3133b2d8c11.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/5289bf025a6c262c7b5d5ce8bc1b2093104b8cbfa9874dae3b44a3133b2d8c11.jpg)

![a11cf3512215190cbd9d46fef1bcb3216bd5791542ee54ad2d4af6db38ce202b.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/a11cf3512215190cbd9d46fef1bcb3216bd5791542ee54ad2d4af6db38ce202b.jpg)

![a9f8b297089c4333ce7ee0ce4f7872e180e658220e18c708835f6dd5b2e556d1.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/a9f8b297089c4333ce7ee0ce4f7872e180e658220e18c708835f6dd5b2e556d1.jpg)

![b2a7ca79ed187a276ff8f37ec1976b3f758e4c0ff5c99d2d0dbdf7fcc4b7f121.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/b2a7ca79ed187a276ff8f37ec1976b3f758e4c0ff5c99d2d0dbdf7fcc4b7f121.jpg)

![b36000d512296a7c4a4652a6cdcd7ece8cd0e397d0be8b9daa5e6a17b2b9c95c.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/b36000d512296a7c4a4652a6cdcd7ece8cd0e397d0be8b9daa5e6a17b2b9c95c.jpg)

![c54e8d4ce3c211fb50ba80ef42a498c95d9eaf735587d8b4d120da5770cbf73e.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/c54e8d4ce3c211fb50ba80ef42a498c95d9eaf735587d8b4d120da5770cbf73e.jpg)

![cf96a45b0448388d518146538da7d4899f11b1649e1d18adc66891b0a3a5fdb3.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/cf96a45b0448388d518146538da7d4899f11b1649e1d18adc66891b0a3a5fdb3.jpg)

![df511f5be4856592739574879dca66795ef0aba52deeeaa5835f7c13e199f28b.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/images/df511f5be4856592739574879dca66795ef0aba52deeeaa5835f7c13e199f28b.jpg)

### Tables

![1a160d785ee97d09d0cd4dc982fbc3f0b8247c17771c925a3623d76875d8a83a.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/tables/1a160d785ee97d09d0cd4dc982fbc3f0b8247c17771c925a3623d76875d8a83a.jpg)

![4b11f60abdf01b6a78939d492d59b3f13c4337c4ef267410971d63f733b74547.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/tables/4b11f60abdf01b6a78939d492d59b3f13c4337c4ef267410971d63f733b74547.jpg)

![5deadc78631179cbace251f7b59a391dd6f74d01d25e21ac295dd031d36412f6.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/tables/5deadc78631179cbace251f7b59a391dd6f74d01d25e21ac295dd031d36412f6.jpg)

![8eca3e1bf8759ac72ac25a3d28820bf9cc18a6b60b62aa113ce1359db3a2a6a0.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/tables/8eca3e1bf8759ac72ac25a3d28820bf9cc18a6b60b62aa113ce1359db3a2a6a0.jpg)

![ae7265933df7e0c32b284d32c633b714f7ae3d3769dd7681698737f441105ad2.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/tables/ae7265933df7e0c32b284d32c633b714f7ae3d3769dd7681698737f441105ad2.jpg)

![c10efb0c0847cb1f5e72397cf4be078052c7e3603a316d346d4c00e42dd62820.jpg](../icml_results/5044_TEAM_%20Temporal%E2%80%93Spatial%20Consistency%20Guided%20Expert%20Activation%20for%20MoE%20Diffusion%20Language%20Model%20Acceler/tables/c10efb0c0847cb1f5e72397cf4be078052c7e3603a316d346d4c00e42dd62820.jpg)

<a id="paper-4997"></a>

## Deep neural networks divide and conquer dihedral multiplication


### Images

![042a6979ecdbe4c38e7c35e8a442c2a66fd71b189de79734c009f9238b334563.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/042a6979ecdbe4c38e7c35e8a442c2a66fd71b189de79734c009f9238b334563.jpg)

![0765bde3e4ab9e13eb792ac791a9a648eb42ab8d47806e4c7f4710aac5e37944.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/0765bde3e4ab9e13eb792ac791a9a648eb42ab8d47806e4c7f4710aac5e37944.jpg)

![09ef3829ea51a8fb6b3f1be019619fc40785d3776d3d132df8e2e59a2291c336.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/09ef3829ea51a8fb6b3f1be019619fc40785d3776d3d132df8e2e59a2291c336.jpg)

![0b15563c33902210748fc7c5c1a8b245338f459884ead25bd2f44fe163645b78.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/0b15563c33902210748fc7c5c1a8b245338f459884ead25bd2f44fe163645b78.jpg)

![0c104032edf0d51a40455246ec67db2dba5aeae5fb646f9045cd512d578dba0c.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/0c104032edf0d51a40455246ec67db2dba5aeae5fb646f9045cd512d578dba0c.jpg)

![1a35771fa004256ee9adc540b5866ab35a9f4b86694cd331b5c2118757ce3676.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/1a35771fa004256ee9adc540b5866ab35a9f4b86694cd331b5c2118757ce3676.jpg)

![2484d79333b84f42c57c731ab99e6864e675c2293b0b2c75fe2293f9f8dcc0c2.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/2484d79333b84f42c57c731ab99e6864e675c2293b0b2c75fe2293f9f8dcc0c2.jpg)

![25daa1718264e0eddd44ca7dfad5369bf2918d45387aeac00aab2989a447df0d.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/25daa1718264e0eddd44ca7dfad5369bf2918d45387aeac00aab2989a447df0d.jpg)

![297d023af6714796ec92aae70a119576c313090df2ec09c4e34bb4cf02b47cc4.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/297d023af6714796ec92aae70a119576c313090df2ec09c4e34bb4cf02b47cc4.jpg)

![2abd18cbedfa454df3e07413c6702e6724df86ade664ce0b23b3dea997111506.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/2abd18cbedfa454df3e07413c6702e6724df86ade664ce0b23b3dea997111506.jpg)

![2b98549b5ac125941b903000d23517adaef0cdf3ffcdbd730302a3621945dd9d.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/2b98549b5ac125941b903000d23517adaef0cdf3ffcdbd730302a3621945dd9d.jpg)

![31fae625bbd6a656d350dbcbdf8c22b2970a75ee647f0070e5ab0c863362099a.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/31fae625bbd6a656d350dbcbdf8c22b2970a75ee647f0070e5ab0c863362099a.jpg)

![372245613fc3749ae694edb5421a4b39360ab2fd56b18780e22685fddd2b3431.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/372245613fc3749ae694edb5421a4b39360ab2fd56b18780e22685fddd2b3431.jpg)

![3aee3552d96dd9be04561a3d8eb5127e81478dedc423da1b20298bc25f89361a.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/3aee3552d96dd9be04561a3d8eb5127e81478dedc423da1b20298bc25f89361a.jpg)

![3e450dd3552fac902f0368040a3f48d3e1918de63b2e476c6b524fe39fb10602.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/3e450dd3552fac902f0368040a3f48d3e1918de63b2e476c6b524fe39fb10602.jpg)

![3fddaa39426e7e216ac9359d6100185429b5d2e795596f49fc1669d25c0138e2.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/3fddaa39426e7e216ac9359d6100185429b5d2e795596f49fc1669d25c0138e2.jpg)

![4c8abd86a8513fe1a57382209956a4abce29e70447c2d4f39e3c3112f644abcb.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/4c8abd86a8513fe1a57382209956a4abce29e70447c2d4f39e3c3112f644abcb.jpg)

![4d15940b2ba703b93dc6cd90916a29390a6b3ee8cd9a33e0eb247d9e0dc0714a.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/4d15940b2ba703b93dc6cd90916a29390a6b3ee8cd9a33e0eb247d9e0dc0714a.jpg)

![5204b5db7fc9073d8681c5511a00364067825ad15b13461d252ad8665a7daaa4.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/5204b5db7fc9073d8681c5511a00364067825ad15b13461d252ad8665a7daaa4.jpg)

![575fe473d45c21a7aee808caddb2a6366750c7ae5143e7cae09dea14c690b472.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/575fe473d45c21a7aee808caddb2a6366750c7ae5143e7cae09dea14c690b472.jpg)

![583a2cfd9cba512aaf881f3ce56ea11e6679f362659c5a8cfaa3b3389693b619.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/583a2cfd9cba512aaf881f3ce56ea11e6679f362659c5a8cfaa3b3389693b619.jpg)

![584e7b4ed89efd6c2d34422a11d4ce4751e212a1f7dec08a34c5562c352bf544.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/584e7b4ed89efd6c2d34422a11d4ce4751e212a1f7dec08a34c5562c352bf544.jpg)

![5d325f18cb8ef3303dbe08521982feefb89ff60aa591c0ccff28e2b82c3b0a62.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/5d325f18cb8ef3303dbe08521982feefb89ff60aa591c0ccff28e2b82c3b0a62.jpg)

![61c38e02b1768cc415b405ee7ee3cde2ff3376b7b6adf32012631f808bb525d3.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/61c38e02b1768cc415b405ee7ee3cde2ff3376b7b6adf32012631f808bb525d3.jpg)

![66979b37b522481c05696f4853734932a728e4edea98f9448b9195778db21a98.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/66979b37b522481c05696f4853734932a728e4edea98f9448b9195778db21a98.jpg)

![66c52e33525342b1258c1fa6f156e62e79a9342868d59a1215dd47c85471fe65.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/66c52e33525342b1258c1fa6f156e62e79a9342868d59a1215dd47c85471fe65.jpg)

![6bb34a76e6d2f009f5419875ff44efb5b5526a71da6b5977da8d01a8eb03c83f.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/6bb34a76e6d2f009f5419875ff44efb5b5526a71da6b5977da8d01a8eb03c83f.jpg)

![6dc644b557dad302da0ab5d8c396388806394ed2ef5d23baace9e0f333544c5c.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/6dc644b557dad302da0ab5d8c396388806394ed2ef5d23baace9e0f333544c5c.jpg)

![70ce8f3ae2276a8b84816930dcc1a2dd943cfafaadc30fbbe74d9ba16a5adf1b.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/70ce8f3ae2276a8b84816930dcc1a2dd943cfafaadc30fbbe74d9ba16a5adf1b.jpg)

![746b396c812918ba300ce50f83918e0f5557652b1f78b3301931e2cdec83ccd8.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/746b396c812918ba300ce50f83918e0f5557652b1f78b3301931e2cdec83ccd8.jpg)

![782d425d30804c7c90e0e94e509387bf2d874705b6677d3ffcca60aae45fbf6b.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/782d425d30804c7c90e0e94e509387bf2d874705b6677d3ffcca60aae45fbf6b.jpg)

![7b8efc7693246a1a78d082d189337892de24237c4b9d5a997489f2cd4b3cbeef.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/7b8efc7693246a1a78d082d189337892de24237c4b9d5a997489f2cd4b3cbeef.jpg)

![7c4028530a5c97dac94761b0dd605aff072c17d044a1ffefaa984ce61298c0ac.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/7c4028530a5c97dac94761b0dd605aff072c17d044a1ffefaa984ce61298c0ac.jpg)

![8b7f9864672dff8a15badfda132346bc70bd99c04ec7acae5110dbc5f33a7053.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/8b7f9864672dff8a15badfda132346bc70bd99c04ec7acae5110dbc5f33a7053.jpg)

![8e4b05cd6adfbdc66fbe21f62084a9cf1dce6eb6b4c4087ec904ab8715cd04bf.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/8e4b05cd6adfbdc66fbe21f62084a9cf1dce6eb6b4c4087ec904ab8715cd04bf.jpg)

![91a4c272a630b474014fc899b3daac950ece60806f1187b322d6d18668d90945.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/91a4c272a630b474014fc899b3daac950ece60806f1187b322d6d18668d90945.jpg)

![9213788a5dccf9f2ae9ce847fb68d31135f0326f74c1f7bfe0815c74ef060948.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/9213788a5dccf9f2ae9ce847fb68d31135f0326f74c1f7bfe0815c74ef060948.jpg)

![936850e0069d1aa589602298bd1866fb7d1b715b434bc1ed6fbfa3d769634f78.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/936850e0069d1aa589602298bd1866fb7d1b715b434bc1ed6fbfa3d769634f78.jpg)

![93f08e91a6d8dc5d18c2cbfc1785fe14e1235ea96f0fe268de3e7f4b39f6ea75.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/93f08e91a6d8dc5d18c2cbfc1785fe14e1235ea96f0fe268de3e7f4b39f6ea75.jpg)

![98132b9f12a03002c9695000cf56d95da93083a51925b88be82803816df8a396.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/98132b9f12a03002c9695000cf56d95da93083a51925b88be82803816df8a396.jpg)

![9a873ad8129f80d37bf7df66a60d9e8296004b85245b3c06c68004f9a00c5990.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/9a873ad8129f80d37bf7df66a60d9e8296004b85245b3c06c68004f9a00c5990.jpg)

![a8a5adce4c404528ec754605a201fd998ca4b3d4f2b522b799c7cdf8ee9055ff.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/a8a5adce4c404528ec754605a201fd998ca4b3d4f2b522b799c7cdf8ee9055ff.jpg)

![ae335f6f689f8868a351dc02ab5a947a68c773c21cbdbae92c0ffcf18dad42e0.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/ae335f6f689f8868a351dc02ab5a947a68c773c21cbdbae92c0ffcf18dad42e0.jpg)

![bdb4f64d4cca82d6bee76785f039b5666d96778ca0d1a4314719affb2dc1425a.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/bdb4f64d4cca82d6bee76785f039b5666d96778ca0d1a4314719affb2dc1425a.jpg)

![bdc655700d0c7f3e9cad677305c41ab58d68c897b98e96742c95c514cce3dcb8.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/bdc655700d0c7f3e9cad677305c41ab58d68c897b98e96742c95c514cce3dcb8.jpg)

![bff90a45473b97f220ef487b318097b871e6e2cf4c2a3b2e34e887552ac12028.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/bff90a45473b97f220ef487b318097b871e6e2cf4c2a3b2e34e887552ac12028.jpg)

![d071778da7dfa8ef89ef6c6d6b44215184f1a36f07023c6a687d8eba9ab30aad.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/d071778da7dfa8ef89ef6c6d6b44215184f1a36f07023c6a687d8eba9ab30aad.jpg)

![d220ab74adc3bdb74e5fdaa4583cc0d06d690b4c0cce6849030ddd7e768cb93f.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/d220ab74adc3bdb74e5fdaa4583cc0d06d690b4c0cce6849030ddd7e768cb93f.jpg)

![dc5d82252a815feb6f72d7208cf5ef1cdda4c60536463cfb2da328db152bba00.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/dc5d82252a815feb6f72d7208cf5ef1cdda4c60536463cfb2da328db152bba00.jpg)

![e1758694f3d7ef0e073be8ac02ab7a32e3ff4b718f39115d228645d93a0ca184.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/e1758694f3d7ef0e073be8ac02ab7a32e3ff4b718f39115d228645d93a0ca184.jpg)

![ed31d07e74d72b0f184d64f685e01bb9553adfb429636321e0b7503001a746ab.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/ed31d07e74d72b0f184d64f685e01bb9553adfb429636321e0b7503001a746ab.jpg)

![f15a80c3c9b39a10f0ee044c918ce73e8ef812d99cd3286fbb85d756d628e29e.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/f15a80c3c9b39a10f0ee044c918ce73e8ef812d99cd3286fbb85d756d628e29e.jpg)

![f4d787b8563cad5e637836329143e90bd63893c953b6cd47f18b5843f3df445e.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/f4d787b8563cad5e637836329143e90bd63893c953b6cd47f18b5843f3df445e.jpg)

![f641148306ab5700819973fd8ee60c1ed748e204a4a4dd3d2f21a7e71d168149.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/f641148306ab5700819973fd8ee60c1ed748e204a4a4dd3d2f21a7e71d168149.jpg)

![ffa31cd26aa9fea5f957ea41a4a5cc8124be084b0e4a45a908bdc27ebd0caeec.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/images/ffa31cd26aa9fea5f957ea41a4a5cc8124be084b0e4a45a908bdc27ebd0caeec.jpg)

### Tables

![0a8c0c33cfb65463379a14755cf738a66fbd437c74a0de9d6c9d5d99a94b1d47.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/0a8c0c33cfb65463379a14755cf738a66fbd437c74a0de9d6c9d5d99a94b1d47.jpg)

![214f48c6741bf71e4cb1b0123ccdab8c5f002996a11ee6fba71d73969565ce00.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/214f48c6741bf71e4cb1b0123ccdab8c5f002996a11ee6fba71d73969565ce00.jpg)

![37f8ad7943289f3f28372fdc1157cba27b2c0b3411ab91f81449881caece16b0.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/37f8ad7943289f3f28372fdc1157cba27b2c0b3411ab91f81449881caece16b0.jpg)

![3cd71ae7d948fbd2eabd2a9a386661fd52d12af0e111336e26711f582e26fc24.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/3cd71ae7d948fbd2eabd2a9a386661fd52d12af0e111336e26711f582e26fc24.jpg)

![42cfcfce15023e31927efdceb51dc9e98bc6b51ec04413e030ab61d3a3d7f009.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/42cfcfce15023e31927efdceb51dc9e98bc6b51ec04413e030ab61d3a3d7f009.jpg)

![bca80d31630db0675f13ed6afdcaf9e8d757a0e6c7410314b03fa66f0b8da6b6.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/bca80d31630db0675f13ed6afdcaf9e8d757a0e6c7410314b03fa66f0b8da6b6.jpg)

![ceba3a1b4de9ee37d0bbae860ade3d8a0936a5f131601cff568529015e3fa290.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/ceba3a1b4de9ee37d0bbae860ade3d8a0936a5f131601cff568529015e3fa290.jpg)

![e2ba426308d38a7ead46711c881f4f852fafe4992d289c942fc49d402e93be4b.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/e2ba426308d38a7ead46711c881f4f852fafe4992d289c942fc49d402e93be4b.jpg)

![e9e62580b55d33a62945ca8b6988f5ee2b61f6a122fb4221a68b78fce4ad1394.jpg](../icml_results/5045_Deep%20neural%20networks%20divide%20and%20conquer%20dihedral%20multiplication/tables/e9e62580b55d33a62945ca8b6988f5ee2b61f6a122fb4221a68b78fce4ad1394.jpg)

<a id="paper-4998"></a>

## The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search


### Images

![1f5a1f62f9e70843c5b915deccbdd038127a5643b9506a63740759ba758ca34c.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/1f5a1f62f9e70843c5b915deccbdd038127a5643b9506a63740759ba758ca34c.jpg)

![4b183376ef88260c6d0a5758d5d1d6d9dac3209b819d3c45f3259dd9f30f2820.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/4b183376ef88260c6d0a5758d5d1d6d9dac3209b819d3c45f3259dd9f30f2820.jpg)

![51d286a513876cdce4503308c2231f2741c68964136aea866f597f30e3490142.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/51d286a513876cdce4503308c2231f2741c68964136aea866f597f30e3490142.jpg)

![529b05c201147a782e1e1265bafa43fa33ef9051c2689b5eae3279c73e31c202.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/529b05c201147a782e1e1265bafa43fa33ef9051c2689b5eae3279c73e31c202.jpg)

![64a7de1353fd4af6b39f0cfa81f67a9d4a3bd7eea509c45cfc51e15137e0eccd.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/64a7de1353fd4af6b39f0cfa81f67a9d4a3bd7eea509c45cfc51e15137e0eccd.jpg)

![68e9b44a4ca08e87e2c14e5ab84ef4889ea8ff824262139c569d10e3b443da8a.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/68e9b44a4ca08e87e2c14e5ab84ef4889ea8ff824262139c569d10e3b443da8a.jpg)

![91b35db6d921d3175c64b394ea2e6ff09492fb598c2e15aa77751358763dae42.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/91b35db6d921d3175c64b394ea2e6ff09492fb598c2e15aa77751358763dae42.jpg)

![933c10e5a94c1899ade9ecdbf2ae5d5281fbb10bf7e3cfa7c0bf6bae3ec84f2f.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/933c10e5a94c1899ade9ecdbf2ae5d5281fbb10bf7e3cfa7c0bf6bae3ec84f2f.jpg)

![b0923c8ff955dda03ce7be031e7871e413b131fb9b39acc22065c2010b13e69c.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/b0923c8ff955dda03ce7be031e7871e413b131fb9b39acc22065c2010b13e69c.jpg)

![eb7c92a4f989c3cb609f54eab5b55e647eade23e460c37f0bd4aa558812f064b.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/images/eb7c92a4f989c3cb609f54eab5b55e647eade23e460c37f0bd4aa558812f064b.jpg)

### Tables

![0de9ada217ceb26b957de4d666d12bc34e0ee3359b3cf6f5ee0a6ee6e7370ed4.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/0de9ada217ceb26b957de4d666d12bc34e0ee3359b3cf6f5ee0a6ee6e7370ed4.jpg)

![66eed0d3f2264f7241b7525a3552ee4c8681a2eea1a0726c1037108979cbe88f.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/66eed0d3f2264f7241b7525a3552ee4c8681a2eea1a0726c1037108979cbe88f.jpg)

![780d3938e0ad9a1e6ad3d9cff989c796d312dff5b77a1cec965a2bc56aa91809.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/780d3938e0ad9a1e6ad3d9cff989c796d312dff5b77a1cec965a2bc56aa91809.jpg)

![85d23b03b8154cb91b317d070cd579cd7f05a64ff5338009082113106cff5822.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/85d23b03b8154cb91b317d070cd579cd7f05a64ff5338009082113106cff5822.jpg)

![a0c242366e4214496ee7f08e0702de8376dbcce2d4d241d6ec37a6a46d02479c.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/a0c242366e4214496ee7f08e0702de8376dbcce2d4d241d6ec37a6a46d02479c.jpg)

![a8f93acd706f097acf4289b8a22bfe828a4975272af12bf02f720bc1775b81f5.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/a8f93acd706f097acf4289b8a22bfe828a4975272af12bf02f720bc1775b81f5.jpg)

![acb1b81224c2f3cf9104293b9cc8c8d2bb7ceba93c4556768bafff08c6bdece6.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/acb1b81224c2f3cf9104293b9cc8c8d2bb7ceba93c4556768bafff08c6bdece6.jpg)

![cf6bcc29c92ed24169082cf796512cc61d2f33e82b7024655f4283a58c755fa7.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/cf6bcc29c92ed24169082cf796512cc61d2f33e82b7024655f4283a58c755fa7.jpg)

![e9d81673a95601eadacd59c1a8aa9a17a399b996ee8b988c0f68c430c43f9cff.jpg](../icml_results/5046_The%20Trojan%20Knowledge_%20Bypassing%20Commercial%20LLM%20Guardrails%20via%20Harmless%20Prompt%20Weaving%20and%20Adaptive%20T/tables/e9d81673a95601eadacd59c1a8aa9a17a399b996ee8b988c0f68c430c43f9cff.jpg)

<a id="paper-4999"></a>

## PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception


### Images

![0c3ce16cfd2b1a8ce6106377c9c99fa151ec8f82abf70b27cba2400d47a23ccd.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/0c3ce16cfd2b1a8ce6106377c9c99fa151ec8f82abf70b27cba2400d47a23ccd.jpg)

![262208baf86caf1a030b242d6fb01fd03e1545e12993e8e5e661d2abd8244268.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/262208baf86caf1a030b242d6fb01fd03e1545e12993e8e5e661d2abd8244268.jpg)

![3333b2b7dc8656f2d036056068cf90a768ecc5a9b4d30d19d2a7abb73157b23e.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/3333b2b7dc8656f2d036056068cf90a768ecc5a9b4d30d19d2a7abb73157b23e.jpg)

![33379dc8d6ec6f783c18a7c664cb73966990f3a004b2f24a5d8728cf7a289de2.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/33379dc8d6ec6f783c18a7c664cb73966990f3a004b2f24a5d8728cf7a289de2.jpg)

![3da54576318e069884913dabac1d80dbbfbb3e2ea34f3955900732f674e4c93f.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/3da54576318e069884913dabac1d80dbbfbb3e2ea34f3955900732f674e4c93f.jpg)

![3dd5816b78270e376be7a4abe2b0b5b9485389ca3fd32433a613a93591aa0062.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/3dd5816b78270e376be7a4abe2b0b5b9485389ca3fd32433a613a93591aa0062.jpg)

![3ebfe355f32173287382f66e74f8f64832f892643540fdc32d88967e09bd33d4.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/3ebfe355f32173287382f66e74f8f64832f892643540fdc32d88967e09bd33d4.jpg)

![44e70ad2c1b62ac77efd71da767a931d80bea1cd028c0faf87bb5e66d2f372bf.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/44e70ad2c1b62ac77efd71da767a931d80bea1cd028c0faf87bb5e66d2f372bf.jpg)

![47395817e7388e5cb8ccdeb30a3dfb68456a65b94321ec6d2ade7985fb3d712d.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/47395817e7388e5cb8ccdeb30a3dfb68456a65b94321ec6d2ade7985fb3d712d.jpg)

![4ce60fcb1494c9dd353a4bf6c4c3df1d9c410330609de6f97e88a1b5f1bdbadb.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/4ce60fcb1494c9dd353a4bf6c4c3df1d9c410330609de6f97e88a1b5f1bdbadb.jpg)

![55cd4606e65a4997f3806e8a301c362d08be7bdafd3c2d86f2871b34791035a5.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/55cd4606e65a4997f3806e8a301c362d08be7bdafd3c2d86f2871b34791035a5.jpg)

![68f318a857a05f059d84736b37537d5f56cdf2807ad9a6fda859fe2ebaa5c1b8.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/68f318a857a05f059d84736b37537d5f56cdf2807ad9a6fda859fe2ebaa5c1b8.jpg)

![6a70881ae51757ef17c97e9a33affb62a06d7300f7cd5f8d401f312ef7ad7895.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/6a70881ae51757ef17c97e9a33affb62a06d7300f7cd5f8d401f312ef7ad7895.jpg)

![81c1d27254c0b7f1455aa3e7eac4dfb185af220886ab10afa7c5dc6c465af4fa.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/81c1d27254c0b7f1455aa3e7eac4dfb185af220886ab10afa7c5dc6c465af4fa.jpg)

![84f6c7c683a39b49f1798d7d4255c339b74a91b6417868bd3f6e71d7c5d902a0.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/84f6c7c683a39b49f1798d7d4255c339b74a91b6417868bd3f6e71d7c5d902a0.jpg)

![8c18bb3115888aab3e8ccae8d4e3525e65f5ff6b9da8769182d0fd5be1f71bc4.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/8c18bb3115888aab3e8ccae8d4e3525e65f5ff6b9da8769182d0fd5be1f71bc4.jpg)

![98150d44777612e9604e8e344861f62498c50a167f3197e59ff0f14241351c14.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/98150d44777612e9604e8e344861f62498c50a167f3197e59ff0f14241351c14.jpg)

![ab06f0774ede124f6c916dfb838c8803c9ad09a25dba209278e751d3504cfec9.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/ab06f0774ede124f6c916dfb838c8803c9ad09a25dba209278e751d3504cfec9.jpg)

![bb55284799a40b3d16c5cf6f63ff1a339b38011f393b2f788fbb158e4b8e28ba.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/bb55284799a40b3d16c5cf6f63ff1a339b38011f393b2f788fbb158e4b8e28ba.jpg)

![c0a59e8ae902bec83b33fa7769752e93f65db6b804ab84a4843b64e7e9023f1f.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/c0a59e8ae902bec83b33fa7769752e93f65db6b804ab84a4843b64e7e9023f1f.jpg)

![cfc6fc30d869782c5f1bc90842f7a691db9ca90b23e7c5221f36753b0ec4072d.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/cfc6fc30d869782c5f1bc90842f7a691db9ca90b23e7c5221f36753b0ec4072d.jpg)

![e4dd700c7050f5acf795c0653b92b6df8bd778c1ded29ea427c7535544d046fd.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/e4dd700c7050f5acf795c0653b92b6df8bd778c1ded29ea427c7535544d046fd.jpg)

![f653a62a9e3cd62e64427d66d5fefb009eed3559036d57b704b535ef5e7c984a.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/images/f653a62a9e3cd62e64427d66d5fefb009eed3559036d57b704b535ef5e7c984a.jpg)

### Tables

![41d77b150dc6a8121aa3f74cbca6d309250d8d4ac06ee90f09baf0e9b46b6981.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/tables/41d77b150dc6a8121aa3f74cbca6d309250d8d4ac06ee90f09baf0e9b46b6981.jpg)

![7c62f1272357b6eef280933ecfb9c3acc3e47d846291c57c4a3b6701b4a7834d.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/tables/7c62f1272357b6eef280933ecfb9c3acc3e47d846291c57c4a3b6701b4a7834d.jpg)

![93604c5516021bb41c4b39686e2fd7a070b64b31b2fbf1c764224c71e5c024b2.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/tables/93604c5516021bb41c4b39686e2fd7a070b64b31b2fbf1c764224c71e5c024b2.jpg)

![dfabccd380120511db42b289319c4f1173a7c080d242321a2399183c3e1f525c.jpg](../icml_results/5047_PerceptionRubrics_%20Calibrating%20Multimodal%20Evaluation%20to%20Human%20Perception/tables/dfabccd380120511db42b289319c4f1173a7c080d242321a2399183c3e1f525c.jpg)

<a id="paper-5000"></a>

## Off-Policy Evaluation for Missingness-Aware Policies in MDPs with Rewards Missing Not at Random


### Images

![1153650904d5e1afa928b8ef0d40b8fd879aef1f940fcbbd7ba720ba29abe2c8.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/images/1153650904d5e1afa928b8ef0d40b8fd879aef1f940fcbbd7ba720ba29abe2c8.jpg)

![2e73c21104770a72529cf9536ac2770712944d74b7ba32e185789173e8973004.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/images/2e73c21104770a72529cf9536ac2770712944d74b7ba32e185789173e8973004.jpg)

![668ea61f7c6a369ffed4c16612f545b8772046f8574bbfaba28e6ec1a398f114.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/images/668ea61f7c6a369ffed4c16612f545b8772046f8574bbfaba28e6ec1a398f114.jpg)

![b669007ca6bdd866efe7a765b12d81125dc19c309b6e174a2479c51c08be5cf2.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/images/b669007ca6bdd866efe7a765b12d81125dc19c309b6e174a2479c51c08be5cf2.jpg)

![cb40157658d22a15af7a276424684b2af1ff77f588c3a3be66ec3ecb0c7336d5.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/images/cb40157658d22a15af7a276424684b2af1ff77f588c3a3be66ec3ecb0c7336d5.jpg)

![d8da6d093187063adb17f6dd1e80cb8f76b5c73daa5fad5989817b27757bdbcb.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/images/d8da6d093187063adb17f6dd1e80cb8f76b5c73daa5fad5989817b27757bdbcb.jpg)

### Tables

![01374ad924a810bb8d650844e43bb13cac659ad7411dd10e17d5d97705f314cf.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/tables/01374ad924a810bb8d650844e43bb13cac659ad7411dd10e17d5d97705f314cf.jpg)

![237bc8b4154e5fc2455caeae9e356b21e49bc8abcb15a1944f58be7195a509e6.jpg](../icml_results/5048_Off-Policy%20Evaluation%20for%20Missingness-Aware%20Policies%20in%20MDPs%20with%20Rewards%20Missing%20Not%20at%20Random/tables/237bc8b4154e5fc2455caeae9e356b21e49bc8abcb15a1944f58be7195a509e6.jpg)

<a id="paper-5001"></a>

## SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate


### Images

![02bede18b27ba45654234d757ab80c0ad7058b284d052c47ba89062694e7c9c8.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/02bede18b27ba45654234d757ab80c0ad7058b284d052c47ba89062694e7c9c8.jpg)

![0345776b2dd75648833f8ade1a942180c6429adbb8c9769e965a6e5ad0fb4919.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/0345776b2dd75648833f8ade1a942180c6429adbb8c9769e965a6e5ad0fb4919.jpg)

![040f2b6a093d8d96e3ecd57fade34b5449bbf894252bf13ad930ec5002b3bc77.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/040f2b6a093d8d96e3ecd57fade34b5449bbf894252bf13ad930ec5002b3bc77.jpg)

![20e33476a11a948b8aaf70bb85a04f0a8dc85d70fa790a00a8ff3b0d8565c1e7.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/20e33476a11a948b8aaf70bb85a04f0a8dc85d70fa790a00a8ff3b0d8565c1e7.jpg)

![2283b84016c422b4bf0aa37b0f208f6858444067599910f9ba5b50495d16dfd8.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/2283b84016c422b4bf0aa37b0f208f6858444067599910f9ba5b50495d16dfd8.jpg)

![2a76085e0c5c60171f063dd3246ae4b4309cff17faed07a90d602d690bebc9f4.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/2a76085e0c5c60171f063dd3246ae4b4309cff17faed07a90d602d690bebc9f4.jpg)

![44e3a59619fe6c2f202b804079c65406dacc97cc50022191de18ccdabf972eef.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/44e3a59619fe6c2f202b804079c65406dacc97cc50022191de18ccdabf972eef.jpg)

![46f4f1acb8dc48c2fa90c2ab6a92b510114e11a20486498ea138063c09451bc2.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/46f4f1acb8dc48c2fa90c2ab6a92b510114e11a20486498ea138063c09451bc2.jpg)

![53806c184829c4203b5b628ef45da839b6d41c55b3c362fd2e93506173d434f7.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/53806c184829c4203b5b628ef45da839b6d41c55b3c362fd2e93506173d434f7.jpg)

![57830023309c4ecef26049d60b2099e11c4f4dc0daa0c37c9dabfdaecac8e518.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/57830023309c4ecef26049d60b2099e11c4f4dc0daa0c37c9dabfdaecac8e518.jpg)

![5e7d51918c97294bf428b63eca54d792db7d29b3744dd879f0645add8e9febc8.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/5e7d51918c97294bf428b63eca54d792db7d29b3744dd879f0645add8e9febc8.jpg)

![66bc4b727170b952e4851375a977445213f61fb99f1946c60680fa270668c017.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/66bc4b727170b952e4851375a977445213f61fb99f1946c60680fa270668c017.jpg)

![9e9c0de190e3753dd8dd83098a69cfaf525d7753d53167427016c4f6f8b11d57.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/9e9c0de190e3753dd8dd83098a69cfaf525d7753d53167427016c4f6f8b11d57.jpg)

![a269fb94ad5f6562ce2d8bffc36df9ad20d21641583d7708071a1fcd933bdbbd.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/a269fb94ad5f6562ce2d8bffc36df9ad20d21641583d7708071a1fcd933bdbbd.jpg)

![b09fa2e13d13cda22157e96d4ade6614612413e942a360873e7c74f2ec62d67a.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/b09fa2e13d13cda22157e96d4ade6614612413e942a360873e7c74f2ec62d67a.jpg)

![cd06280c70acefa61b287d581b03bfeeed3646fd57a96bd249e21c6af5a8d4be.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/cd06280c70acefa61b287d581b03bfeeed3646fd57a96bd249e21c6af5a8d4be.jpg)

![d389bc8479c05a1c99467d4ae1e46c941d9a1a455fded7f84cf90f73d22d45f7.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/d389bc8479c05a1c99467d4ae1e46c941d9a1a455fded7f84cf90f73d22d45f7.jpg)

![d927ce529935bf1d7a39c33a859cf3d1262914d0eaf2f25899c17e98b01db535.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/d927ce529935bf1d7a39c33a859cf3d1262914d0eaf2f25899c17e98b01db535.jpg)

![f57913669d7db07b0632e9baf3b7aba9331ff37b9d9606545337bedd240f251b.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/f57913669d7db07b0632e9baf3b7aba9331ff37b9d9606545337bedd240f251b.jpg)

![fc0a26dd132798d92c1736380d0286ddcf11efd36b7403f2afececbfd886075a.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/images/fc0a26dd132798d92c1736380d0286ddcf11efd36b7403f2afececbfd886075a.jpg)

### Tables

![273827bef6f48d72f5f5539230977117add330dbc0ee833b99ddc8decf43ed74.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/273827bef6f48d72f5f5539230977117add330dbc0ee833b99ddc8decf43ed74.jpg)

![37b599dd4b1f83741565db8c95445777627cbc55c48ace18a1473bf6200265b5.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/37b599dd4b1f83741565db8c95445777627cbc55c48ace18a1473bf6200265b5.jpg)

![49c81ea70798833975b8c98c2b8de679b6475d1f1085159a4b3a62c2bc4ddcba.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/49c81ea70798833975b8c98c2b8de679b6475d1f1085159a4b3a62c2bc4ddcba.jpg)

![5eeb8aeed91ea6a1ba8da420943f85881608d58629667df53077dee118e130d9.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/5eeb8aeed91ea6a1ba8da420943f85881608d58629667df53077dee118e130d9.jpg)

![bfc2950ea98ea047e1d60d9123132306bd06131c775fda159a011fa0f790ec18.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/bfc2950ea98ea047e1d60d9123132306bd06131c775fda159a011fa0f790ec18.jpg)

![cd2db52de5fe5878e6efadc13664266fdc76b1263426fe247618e9a40d7a2fb4.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/cd2db52de5fe5878e6efadc13664266fdc76b1263426fe247618e9a40d7a2fb4.jpg)

![f046a4ca6df25ed7b697ca1bff7b652c2a00aeb4371031c8b51834b43b88b6d1.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/f046a4ca6df25ed7b697ca1bff7b652c2a00aeb4371031c8b51834b43b88b6d1.jpg)

![ffbb74ae6628f4589d6193e44091ce6bdd114e07d75d7433cf3abd539a3e48f1.jpg](../icml_results/5049_SURGE_%20Approximation%20and%20Training%20Free%20Particle%20Filter%20for%20Diffusion%20Surrogate/tables/ffbb74ae6628f4589d6193e44091ce6bdd114e07d75d7433cf3abd539a3e48f1.jpg)

<a id="paper-5002"></a>

## Toward Training Superintelligent Software Agents through Self-Play SWE-RL


### Images

![1bc7775ad323e7c98f554a50bb1032a83ab3cace114739d8cfb69b10e14025f2.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/1bc7775ad323e7c98f554a50bb1032a83ab3cace114739d8cfb69b10e14025f2.jpg)

![3d54c491fe2d9aa884768256687469dc11d5c128a05366ba3a2f68ef7f3d4963.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/3d54c491fe2d9aa884768256687469dc11d5c128a05366ba3a2f68ef7f3d4963.jpg)

![5d43f0902e1795d8c3ad01d4786c0a1b1374f726144519d497fe78fddd9e55f2.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/5d43f0902e1795d8c3ad01d4786c0a1b1374f726144519d497fe78fddd9e55f2.jpg)

![66890d03879b0380b767c0c6dc9cbb5d8c2d6766ced6c8228deac41c0927d43e.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/66890d03879b0380b767c0c6dc9cbb5d8c2d6766ced6c8228deac41c0927d43e.jpg)

![9035ba9e1fc560d790ee3750dcc8cc7486407f7f6d8eeb60eaaeca1792b53d46.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/9035ba9e1fc560d790ee3750dcc8cc7486407f7f6d8eeb60eaaeca1792b53d46.jpg)

![a90b5874a54608ec60d0a98f07652d5fbce5af737088778452f5baf5bb75281f.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/a90b5874a54608ec60d0a98f07652d5fbce5af737088778452f5baf5bb75281f.jpg)

![e9474cd67ab24ef52d1f617583524c5d7b91e0845bf98d1c90fe868644ca5a3f.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/e9474cd67ab24ef52d1f617583524c5d7b91e0845bf98d1c90fe868644ca5a3f.jpg)

![f2a48b16e8957f2e1ecf37eb45a2c2890ce096c183d9e17acd87a94e425b4cb8.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/f2a48b16e8957f2e1ecf37eb45a2c2890ce096c183d9e17acd87a94e425b4cb8.jpg)

![f4b4cff6c6d8fc4745d9362835a8ffd4bd3f84ed9ba9c88b2fa176a661b0c7a1.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/images/f4b4cff6c6d8fc4745d9362835a8ffd4bd3f84ed9ba9c88b2fa176a661b0c7a1.jpg)

### Tables

![0d225ffa239c9d9902039c732f43da5e040ed2318e39a07a1077006816f7bb4a.jpg](../icml_results/5050_Toward%20Training%20Superintelligent%20Software%20Agents%20through%20Self-Play%20SWE-RL/tables/0d225ffa239c9d9902039c732f43da5e040ed2318e39a07a1077006816f7bb4a.jpg)

<a id="paper-5003"></a>

## Return of Frustratingly Easy Unsupervised Video Domain Adaptation


### Images

![4c52478eac3f3419b3315f8bb0917564fa00f6180d4088a810c4509ba26ad2d4.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/images/4c52478eac3f3419b3315f8bb0917564fa00f6180d4088a810c4509ba26ad2d4.jpg)

![6efbf12c688d0416a91b28b83cc990515f7cd16d9f2d281459a3a59b6b447ed4.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/images/6efbf12c688d0416a91b28b83cc990515f7cd16d9f2d281459a3a59b6b447ed4.jpg)

![fc141a3013c5bf055aa28e981c2ac36ed57808d8e83eba77b86deda3496f4a56.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/images/fc141a3013c5bf055aa28e981c2ac36ed57808d8e83eba77b86deda3496f4a56.jpg)

### Tables

![1661b1313f419b8776db21f40365801609706397165c6eb12d92692d677d9bdc.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/1661b1313f419b8776db21f40365801609706397165c6eb12d92692d677d9bdc.jpg)

![188cd97d58206846a5bc044b2cf58d4034cccf4d29c7f79b0d99ebe726a9cf05.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/188cd97d58206846a5bc044b2cf58d4034cccf4d29c7f79b0d99ebe726a9cf05.jpg)

![31e7958eba73b0fa1f3af06dfbd4a8a8977682b607fa1073a38530d4c33d057f.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/31e7958eba73b0fa1f3af06dfbd4a8a8977682b607fa1073a38530d4c33d057f.jpg)

![699f4e5168d9eaddb8bd437187415da79a28f69e50a92c10794cc60477a0f6ce.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/699f4e5168d9eaddb8bd437187415da79a28f69e50a92c10794cc60477a0f6ce.jpg)

![99e244a1d962654125d8c91cb06f935cde40c7e790aaf749a190d0a683ec2b69.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/99e244a1d962654125d8c91cb06f935cde40c7e790aaf749a190d0a683ec2b69.jpg)

![9d650c0cc3a5baaeebce6bb02a0269cb4512382c1dae6617ec73b94402c4e4be.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/9d650c0cc3a5baaeebce6bb02a0269cb4512382c1dae6617ec73b94402c4e4be.jpg)

![a5945b202147553b59e4f4236a4f60bdce70282e744e7038413adc311accb299.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/a5945b202147553b59e4f4236a4f60bdce70282e744e7038413adc311accb299.jpg)

![c2a0c05e2af0c45f899b36ebc3125674e88162e8850177082388753917bffc24.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/c2a0c05e2af0c45f899b36ebc3125674e88162e8850177082388753917bffc24.jpg)

![cda9bb89bfaa4ee6f35558554962df9654ff3f05a84e4669a4fa0c34cbc72950.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/cda9bb89bfaa4ee6f35558554962df9654ff3f05a84e4669a4fa0c34cbc72950.jpg)

![d733731c4940d5ca9808fa580a93b59e5e50d055cc549542c125a819c1f1eb3a.jpg](../icml_results/5051_Return%20of%20Frustratingly%20Easy%20Unsupervised%20Video%20Domain%20Adaptation/tables/d733731c4940d5ca9808fa580a93b59e5e50d055cc549542c125a819c1f1eb3a.jpg)

<a id="paper-5004"></a>

## ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics


### Images

![00e8c656ce1d14b92724bbc95f75f6ff5b052b90360d12501c1c2ec269b46719.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/00e8c656ce1d14b92724bbc95f75f6ff5b052b90360d12501c1c2ec269b46719.jpg)

![225046ace2b9304a970704bb6190153479775614af00ffa0d3882697f33bdb7f.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/225046ace2b9304a970704bb6190153479775614af00ffa0d3882697f33bdb7f.jpg)

![226b98623b1f4248effdf316716700de4d6bb97a8b800e356d96ba4995a4fc02.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/226b98623b1f4248effdf316716700de4d6bb97a8b800e356d96ba4995a4fc02.jpg)

![4066e72ae1ffcc4ce37c87f58468915ec1d81eb355897d3986a98591b6e5dfcd.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/4066e72ae1ffcc4ce37c87f58468915ec1d81eb355897d3986a98591b6e5dfcd.jpg)

![42f2d5e56eaa32f16e4ca422fdfd440469c729d9cd415a25f03fbd920fda1878.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/42f2d5e56eaa32f16e4ca422fdfd440469c729d9cd415a25f03fbd920fda1878.jpg)

![5287e9c05828fe17e6018d4906259326852c32bc9271bf530b690d3538f4447a.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/5287e9c05828fe17e6018d4906259326852c32bc9271bf530b690d3538f4447a.jpg)

![542477fadf49a57eac8d3080a9dd0505cf94460dfa86fe4d1dcecd7ad7db4852.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/542477fadf49a57eac8d3080a9dd0505cf94460dfa86fe4d1dcecd7ad7db4852.jpg)

![914064a77423427dbfbe1e0401f80523937a0ff22edac8301df22f91fc30004e.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/914064a77423427dbfbe1e0401f80523937a0ff22edac8301df22f91fc30004e.jpg)

![9e0dfd32a281a69c74ab1c45c9452178c4ab11c78f80b28d4a4528f824d2a714.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/9e0dfd32a281a69c74ab1c45c9452178c4ab11c78f80b28d4a4528f824d2a714.jpg)

![9e84f489f034c176e620aec45129bf134078ca2570ba45bb9f23ab291107f439.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/9e84f489f034c176e620aec45129bf134078ca2570ba45bb9f23ab291107f439.jpg)

![a3fdecfde6a4c73125f84109429e4fa658b98c6db20a6892b9c888b556a00054.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/a3fdecfde6a4c73125f84109429e4fa658b98c6db20a6892b9c888b556a00054.jpg)

![a47786719ac164ede8e65674ecf203fadf306b9665a3424f36c2e9242d71dd4b.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/a47786719ac164ede8e65674ecf203fadf306b9665a3424f36c2e9242d71dd4b.jpg)

![af3d9b3baa5d4e15b4eb985bc5a52a175b34d9d2ae8a6ddb683e15341dad1544.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/af3d9b3baa5d4e15b4eb985bc5a52a175b34d9d2ae8a6ddb683e15341dad1544.jpg)

![afcc16d4ec97a811ee040bce059554a4fa23731bf36375babb883a9afed7a1b7.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/afcc16d4ec97a811ee040bce059554a4fa23731bf36375babb883a9afed7a1b7.jpg)

![b4c31428dc3618c67e3e5a560e4fa787a2c9c3a5d20c80179dc6c999a5ad2f97.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/b4c31428dc3618c67e3e5a560e4fa787a2c9c3a5d20c80179dc6c999a5ad2f97.jpg)

![b616b8e5c8a3959d622a169e9d6f81446e554c3905c20ab0ac196827cf698752.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/b616b8e5c8a3959d622a169e9d6f81446e554c3905c20ab0ac196827cf698752.jpg)

![d1f780ad139ac1dc38142a8453c855cd7178cde62f0b6487515a8624ec2473ee.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/d1f780ad139ac1dc38142a8453c855cd7178cde62f0b6487515a8624ec2473ee.jpg)

![de79f4a631d23d2a03342c95b086126c9b6d629bd043a8e17f9050b3a2e2a703.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/de79f4a631d23d2a03342c95b086126c9b6d629bd043a8e17f9050b3a2e2a703.jpg)

![e555b5938c99ab8a12034d239c793b5700d8f0180cf52cc13553fa9960233a9a.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/e555b5938c99ab8a12034d239c793b5700d8f0180cf52cc13553fa9960233a9a.jpg)

![e8d6f5d231dcf07dfc683fd443b9a2622252086793bd9f7704a2bbea2662f294.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/e8d6f5d231dcf07dfc683fd443b9a2622252086793bd9f7704a2bbea2662f294.jpg)

![ebb3d46152c567a8978c90182adafb4d1beb84cfaa2e0ab2f462529cccbf4bf1.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/ebb3d46152c567a8978c90182adafb4d1beb84cfaa2e0ab2f462529cccbf4bf1.jpg)

![f418d946b836964daea7295a58859c49bd37a29592c7de7b781c2607779eb249.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/f418d946b836964daea7295a58859c49bd37a29592c7de7b781c2607779eb249.jpg)

![f5be9e301a39d6f31da84dc1922825f7bed76b6b7fa9005dda9ced3cc142628a.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/f5be9e301a39d6f31da84dc1922825f7bed76b6b7fa9005dda9ced3cc142628a.jpg)

![fa94a1749281d80ba2728f5562b7b12aa91fed4f56144ba7fe9e4c032d4a372b.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/images/fa94a1749281d80ba2728f5562b7b12aa91fed4f56144ba7fe9e4c032d4a372b.jpg)

### Tables

![1495f217d3e367d124ec4cc95ea302dd54b6e9c7441f740078a89a0f1d479735.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/1495f217d3e367d124ec4cc95ea302dd54b6e9c7441f740078a89a0f1d479735.jpg)

![29e27b17c0829451838f6c72a09d56277c040d2abb074173c48fee64af8aec53.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/29e27b17c0829451838f6c72a09d56277c040d2abb074173c48fee64af8aec53.jpg)

![335d25772ad5df1d4d7e3a947d5a86300ae3326bec95990fb70d753650677147.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/335d25772ad5df1d4d7e3a947d5a86300ae3326bec95990fb70d753650677147.jpg)

![6de13a1a4a25ead900f69d449c1d43b3e4272217492393fe6974ed6c0a5412e4.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/6de13a1a4a25ead900f69d449c1d43b3e4272217492393fe6974ed6c0a5412e4.jpg)

![739d9f481b6bf6c035bb5051c7b74f016ea6ba951fbb4284c458d298442aafd5.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/739d9f481b6bf6c035bb5051c7b74f016ea6ba951fbb4284c458d298442aafd5.jpg)

![81982c14416381a05a0b7643e8fb7d05c6c677a3ba411cb12d2afeb37ac38e54.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/81982c14416381a05a0b7643e8fb7d05c6c677a3ba411cb12d2afeb37ac38e54.jpg)

![8fb271f094dc136baacd25a1356725c92f064c8de478b739c1d25682dad89e59.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/8fb271f094dc136baacd25a1356725c92f064c8de478b739c1d25682dad89e59.jpg)

![a807c30dbf63952cffc851e2fa95a6c1721b7b67919bc2c58cc8f9f59f72637f.jpg](../icml_results/5052_ManiSoft_%20Towards%20Vision-Language%20Manipulation%20for%20Soft%20Continuum%20Robotics/tables/a807c30dbf63952cffc851e2fa95a6c1721b7b67919bc2c58cc8f9f59f72637f.jpg)

<a id="paper-5005"></a>

## Words & Weights: Streamlining Multi-Turn Interactions via Co-Adaptation


### Images

![1186c69e250edd409d83cbc80551a6c59edce5863cfba14ead2147325ae8c6cc.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/images/1186c69e250edd409d83cbc80551a6c59edce5863cfba14ead2147325ae8c6cc.jpg)

![43f3b2d0ac06c05093facb09dfddeab7e063a615ad2c7b92e36d84780fc65f7d.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/images/43f3b2d0ac06c05093facb09dfddeab7e063a615ad2c7b92e36d84780fc65f7d.jpg)

![846e6ca04e80af8ea7d5406f093e848b0cdba86918677671aa44fa090584e26c.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/images/846e6ca04e80af8ea7d5406f093e848b0cdba86918677671aa44fa090584e26c.jpg)

![ac34c5ed5a07717084c6cdd7ba0df3d5a72c135c72a705dab0d905cf3beb73eb.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/images/ac34c5ed5a07717084c6cdd7ba0df3d5a72c135c72a705dab0d905cf3beb73eb.jpg)

![c65b87e4a504169bf9a740628e98ff2ec0fcd409c203ded61b7dde9ca2407fa6.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/images/c65b87e4a504169bf9a740628e98ff2ec0fcd409c203ded61b7dde9ca2407fa6.jpg)

### Tables

![0aab99fb036c05273a05b52cab6e0a4363aeff2a5e8db597d9d6de699567626c.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/0aab99fb036c05273a05b52cab6e0a4363aeff2a5e8db597d9d6de699567626c.jpg)

![17ee4dfc24093a2ba6977e19e4c7ce0d3ffbc01489a04664ebaf52ab9065ae80.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/17ee4dfc24093a2ba6977e19e4c7ce0d3ffbc01489a04664ebaf52ab9065ae80.jpg)

![30f6e8c73e0b4b2df5c74076ea4ce924cee5058cb51cc3999fa66db123484836.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/30f6e8c73e0b4b2df5c74076ea4ce924cee5058cb51cc3999fa66db123484836.jpg)

![386ea7c667aea012e120d23508143e7ebd93bdb19dcfe4f74e042ce80e191374.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/386ea7c667aea012e120d23508143e7ebd93bdb19dcfe4f74e042ce80e191374.jpg)

![3cbdd58636cb7d36b9fe4ed0de84d054b024457c109e9968fcadfc944a198641.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/3cbdd58636cb7d36b9fe4ed0de84d054b024457c109e9968fcadfc944a198641.jpg)

![405b11f21ecfde4008f6f58ac9ec6f238319c26bb489aca8790f056677b0ffc1.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/405b11f21ecfde4008f6f58ac9ec6f238319c26bb489aca8790f056677b0ffc1.jpg)

![44a8de6dfcb6cf0bad06817dea359ba74f7f8862c47c77d9354e23abcf7e573d.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/44a8de6dfcb6cf0bad06817dea359ba74f7f8862c47c77d9354e23abcf7e573d.jpg)

![865ee6d8e6c19bc89321ba66e29c141e02b81136a80b30d250eb203049195141.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/865ee6d8e6c19bc89321ba66e29c141e02b81136a80b30d250eb203049195141.jpg)

![a3c056fb9c47f1a6a0f244f3dbb2f039b68bca1f5b97b5c750d8108fa98c215e.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/a3c056fb9c47f1a6a0f244f3dbb2f039b68bca1f5b97b5c750d8108fa98c215e.jpg)

![abb68f6d5eb44d1312cba219ecf288e09556f53bacddac4ed57bdc6279d0c464.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/abb68f6d5eb44d1312cba219ecf288e09556f53bacddac4ed57bdc6279d0c464.jpg)

![b0041071dee0214d2416d41b38867aadf3020d759bda0a4cbd54770fe8fee62a.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/b0041071dee0214d2416d41b38867aadf3020d759bda0a4cbd54770fe8fee62a.jpg)

![bc7c997b9ae66f141e6a2de041cea8178f7377e48f2be22ab47a0da25eac72b4.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/bc7c997b9ae66f141e6a2de041cea8178f7377e48f2be22ab47a0da25eac72b4.jpg)

![fbbb1bb62a0b90b43f2973b82b155058d7ddc74fee03bccc5e1f091640fbf0cc.jpg](../icml_results/5053_Words%20%26%20Weights_%20Streamlining%20Multi-Turn%20Interactions%20via%20Co-Adaptation/tables/fbbb1bb62a0b90b43f2973b82b155058d7ddc74fee03bccc5e1f091640fbf0cc.jpg)

<a id="paper-5006"></a>

## TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning


### Images

![0193ab2d23e297e5a85e3ad8e2aad89ffb2c0e40bf021d2933b31313ef7e7de2.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/0193ab2d23e297e5a85e3ad8e2aad89ffb2c0e40bf021d2933b31313ef7e7de2.jpg)

![0843e5998fa37ead485fad1b84977a2b44a655cf12f5d6485d4cf9d39e19a66a.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/0843e5998fa37ead485fad1b84977a2b44a655cf12f5d6485d4cf9d39e19a66a.jpg)

![0fc26df1dad0423c17d227739d3054077dbbb311ee5dd3df0717b043f7fc5685.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/0fc26df1dad0423c17d227739d3054077dbbb311ee5dd3df0717b043f7fc5685.jpg)

![20b86323ef37224a1ae892e88e6344a0c482a77d21dc83d4d3bd82e59e3d0898.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/20b86323ef37224a1ae892e88e6344a0c482a77d21dc83d4d3bd82e59e3d0898.jpg)

![248738fdaea4fe8edc6db9436feed19697eb7cd7325c44a02cd62510c0f4af8d.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/248738fdaea4fe8edc6db9436feed19697eb7cd7325c44a02cd62510c0f4af8d.jpg)

![2f18bdf947ad5fd0750ccded2595cf81314a6873b469773d6e97f47476ce6f44.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/2f18bdf947ad5fd0750ccded2595cf81314a6873b469773d6e97f47476ce6f44.jpg)

![6607722fcc5f3c9728e0335336e989e5dd4b8122be13c34997bb5d9d6ec8856d.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/images/6607722fcc5f3c9728e0335336e989e5dd4b8122be13c34997bb5d9d6ec8856d.jpg)

### Tables

![0c8bb074bcf012eca8d83e4bdfe812cc9819647e7a1494b74507a672c12b3f4e.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/0c8bb074bcf012eca8d83e4bdfe812cc9819647e7a1494b74507a672c12b3f4e.jpg)

![41f960c024a2078205c4cb00c1d44ddda5ee6eab6b504e0ae74ab02f4cc33ab0.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/41f960c024a2078205c4cb00c1d44ddda5ee6eab6b504e0ae74ab02f4cc33ab0.jpg)

![4279588818a405cc5ac9b6306cabe9010d14ec8e5ff108a416d3e9a026332d72.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/4279588818a405cc5ac9b6306cabe9010d14ec8e5ff108a416d3e9a026332d72.jpg)

![70462dc365bcad002f9e57cdd9576a14bb41b66cd0825a8dd8bc22f23ef9f4f9.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/70462dc365bcad002f9e57cdd9576a14bb41b66cd0825a8dd8bc22f23ef9f4f9.jpg)

![8683f8f8dbddb12d71f3240f2dd14f9bb7f8d6ee1eed879e9afbd5f8691cf545.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/8683f8f8dbddb12d71f3240f2dd14f9bb7f8d6ee1eed879e9afbd5f8691cf545.jpg)

![930516dae2b5038022552a94839613618cb492d3bec6bb8d0a6446838cb987db.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/930516dae2b5038022552a94839613618cb492d3bec6bb8d0a6446838cb987db.jpg)

![977a8efdb66e0728a3bbae94dba10bd67f36ef0ee5eb7afcc0e9eb1da67b0f50.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/977a8efdb66e0728a3bbae94dba10bd67f36ef0ee5eb7afcc0e9eb1da67b0f50.jpg)

![99abbf1fff4aeb67a7904b93b0ca91c64cd6da330011994ffcc8a9d288c64543.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/99abbf1fff4aeb67a7904b93b0ca91c64cd6da330011994ffcc8a9d288c64543.jpg)

![9f89073c47afa036ea03d64f5d022018c35b2ed62da2dda35071c99b82e72a5b.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/9f89073c47afa036ea03d64f5d022018c35b2ed62da2dda35071c99b82e72a5b.jpg)

![bb3681362b88afee33e05710a55e6efb6c618155ae084f249f1e8a84dd1d0dda.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/bb3681362b88afee33e05710a55e6efb6c618155ae084f249f1e8a84dd1d0dda.jpg)

![cb5024ce95b91a24b6cef97524a4978fb48a8efc1d0bd9ad47732beeff8be19a.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/cb5024ce95b91a24b6cef97524a4978fb48a8efc1d0bd9ad47732beeff8be19a.jpg)

![ea2ae593ab37778e74e2321c307f8c6ea7b61b456cb86f2c00b8f55d90b92d93.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/ea2ae593ab37778e74e2321c307f8c6ea7b61b456cb86f2c00b8f55d90b92d93.jpg)

![eb569b314ef5b6415523c6529555281ca6d4763230b2f45f52f334af65f58e63.jpg](../icml_results/5054_TruthRL_%20Incentivizing%20Truthful%20LLMs%20via%20Reinforcement%20Learning/tables/eb569b314ef5b6415523c6529555281ca6d4763230b2f45f52f334af65f58e63.jpg)

<a id="paper-5007"></a>

## A-MemGuard: A Proactive Defense Framework For LLM-Based Agent Memory


### Images

![230760a0a76ebb45f454f21d524cf0388a4213b17f511a6c29ffe6d22020b8f0.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/230760a0a76ebb45f454f21d524cf0388a4213b17f511a6c29ffe6d22020b8f0.jpg)

![36caeaeb4122ab134430d11daca5aac122c337ad91722e82b96d9fa6753ba18d.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/36caeaeb4122ab134430d11daca5aac122c337ad91722e82b96d9fa6753ba18d.jpg)

![415403d3d9f6416d476893ed5dae6cacd2c7782871a1eb4d7a79c60b0018bacc.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/415403d3d9f6416d476893ed5dae6cacd2c7782871a1eb4d7a79c60b0018bacc.jpg)

![5980d3f1641e05472beec0f53101101133e9cde76042d2c094cc13357aeb346c.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/5980d3f1641e05472beec0f53101101133e9cde76042d2c094cc13357aeb346c.jpg)

![7b019c80e10cc062288a3dd33bd199d110d39567ceb5fa23b17a65dc1a5bdd7f.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/7b019c80e10cc062288a3dd33bd199d110d39567ceb5fa23b17a65dc1a5bdd7f.jpg)

![8ac7edd916b52bcb429353b287c5f5b5f1d4251c3155a425d8e88853271f90d1.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/8ac7edd916b52bcb429353b287c5f5b5f1d4251c3155a425d8e88853271f90d1.jpg)

![bdc1414d48f9b6ced63a0837715ca53051092e68e2313f8640cdb0dfddffba2a.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/bdc1414d48f9b6ced63a0837715ca53051092e68e2313f8640cdb0dfddffba2a.jpg)

![be8c64f0060013f4e0b63b578b2617cff4db74ec7eb0c3767fc00ef33f4d2cd6.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/be8c64f0060013f4e0b63b578b2617cff4db74ec7eb0c3767fc00ef33f4d2cd6.jpg)

![c0e3cf14fe01552606e9f025647f602a8c3ad4a11d4b787daa44ee43b36b35bc.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/c0e3cf14fe01552606e9f025647f602a8c3ad4a11d4b787daa44ee43b36b35bc.jpg)

![c4ca5bd3071a647571e6e22dba4436b1b5ff5e453bed21ac9287ddfbb9fb266e.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/c4ca5bd3071a647571e6e22dba4436b1b5ff5e453bed21ac9287ddfbb9fb266e.jpg)

![e38b65de164383c7c9787cce7b2ad42e74393086ea7404847fd7a7c0aeb992f2.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/e38b65de164383c7c9787cce7b2ad42e74393086ea7404847fd7a7c0aeb992f2.jpg)

![e44d22b86ef2311d4dc9c527128f83682f47def5fa36bda1ff25fc30c6218882.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/e44d22b86ef2311d4dc9c527128f83682f47def5fa36bda1ff25fc30c6218882.jpg)

![f00b7204b9bd922f480e0b88b4874412c7034cd501b9ac30160d5ee90ea03ea2.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/images/f00b7204b9bd922f480e0b88b4874412c7034cd501b9ac30160d5ee90ea03ea2.jpg)

### Tables

![13502d77b1859b584f09ee038e096a01cb657499420164106dd18738972f73e8.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/13502d77b1859b584f09ee038e096a01cb657499420164106dd18738972f73e8.jpg)

![4771d9b1e5020596f55c2de488d3b89fdc71873dbd68b2cc855afbf72fd13ddd.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/4771d9b1e5020596f55c2de488d3b89fdc71873dbd68b2cc855afbf72fd13ddd.jpg)

![56bb3f57e8080d90c64cd31ff30a745bbcdb786698f715dc9b22b5901b4dd9d1.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/56bb3f57e8080d90c64cd31ff30a745bbcdb786698f715dc9b22b5901b4dd9d1.jpg)

![5946c1f49c75caf288ab3c4dd90079a5ecfd458592633fb0211d1061252d345f.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/5946c1f49c75caf288ab3c4dd90079a5ecfd458592633fb0211d1061252d345f.jpg)

![6f1ba489c76aa310f8b8ea04019feddb28b25d7607e866d954ee5e16a61ec94a.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/6f1ba489c76aa310f8b8ea04019feddb28b25d7607e866d954ee5e16a61ec94a.jpg)

![7f9bfa2eeb526257a5443eae5c9e353b20b3097f9c964f91d3ff80699b18d833.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/7f9bfa2eeb526257a5443eae5c9e353b20b3097f9c964f91d3ff80699b18d833.jpg)

![88bd952b3049a2c20a6ae9213abb71b481b16939dd1ff384607c8cc9740953bb.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/88bd952b3049a2c20a6ae9213abb71b481b16939dd1ff384607c8cc9740953bb.jpg)

![a332fdb351213ed1d4831aefbbac677afcf87bc1401eb43afbb3a94a0ce60751.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/a332fdb351213ed1d4831aefbbac677afcf87bc1401eb43afbb3a94a0ce60751.jpg)

![b862cf81f2843cc9f99a20c947b2155195bbd5070e02c0f31192d72b84f061d0.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/b862cf81f2843cc9f99a20c947b2155195bbd5070e02c0f31192d72b84f061d0.jpg)

![e4a9c52110643a9e98e459815db426f886a9483e2ce4d0820b5161842ccb56ef.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/e4a9c52110643a9e98e459815db426f886a9483e2ce4d0820b5161842ccb56ef.jpg)

![f5325042b1df1404f11f2fffa0f9a0e21e2dc58adc88292f94754c052d2d95f2.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/f5325042b1df1404f11f2fffa0f9a0e21e2dc58adc88292f94754c052d2d95f2.jpg)

![fe1a0f1f49e381810f5eb7811b0f848d1e28d42f44ed002eab7dcae89538528c.jpg](../icml_results/5055_A-MemGuard_%20A%20Proactive%20Defense%20Framework%20For%20LLM-Based%20Agent%20Memory/tables/fe1a0f1f49e381810f5eb7811b0f848d1e28d42f44ed002eab7dcae89538528c.jpg)

<a id="paper-5008"></a>

## PSMix: Robust Point Cloud Recognition through Spectral Domain Mixing


### Images

![05c21e19545e745a413ce06e78f24560b684a2f23f810df6738f57428bfaceb8.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/images/05c21e19545e745a413ce06e78f24560b684a2f23f810df6738f57428bfaceb8.jpg)

![0749ced4db47467c6df847d31358d6b6ad492eda31c6fe009285d56cafec1bdf.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/images/0749ced4db47467c6df847d31358d6b6ad492eda31c6fe009285d56cafec1bdf.jpg)

![5afd7203c306f5b35d170850f467466de770c4f061a0bb55c5f968814399fedc.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/images/5afd7203c306f5b35d170850f467466de770c4f061a0bb55c5f968814399fedc.jpg)

![8698ba369dc759a8a314876916cccb8c8c09c1b2b0937d1c2b1f67370b31620c.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/images/8698ba369dc759a8a314876916cccb8c8c09c1b2b0937d1c2b1f67370b31620c.jpg)

![dfa32cbc84c3ad3a4c3ad5e43aa82f24fc17efdd3bebffdfa40822c2954cd520.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/images/dfa32cbc84c3ad3a4c3ad5e43aa82f24fc17efdd3bebffdfa40822c2954cd520.jpg)

### Tables

![1fd1de8de40955cefc2ddefe616dd46cbd913ff6016f1aaaccc1733afd96286b.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/1fd1de8de40955cefc2ddefe616dd46cbd913ff6016f1aaaccc1733afd96286b.jpg)

![238817de454cb86e1012108a8b9686946e816a5a5a24910facf478dc997b9ddb.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/238817de454cb86e1012108a8b9686946e816a5a5a24910facf478dc997b9ddb.jpg)

![2581e58527d23bbf86d5190c5e0bef827355343a84b7c727730ae2975b4407ea.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/2581e58527d23bbf86d5190c5e0bef827355343a84b7c727730ae2975b4407ea.jpg)

![995f8c3dabcb544c54699624f68b3d4f4b19c2966b490af3d16137832fd4fa21.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/995f8c3dabcb544c54699624f68b3d4f4b19c2966b490af3d16137832fd4fa21.jpg)

![9be392a6188df2fe4c2370970d4e40f3d1d9607d440410c70b41957caf897d1e.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/9be392a6188df2fe4c2370970d4e40f3d1d9607d440410c70b41957caf897d1e.jpg)

![a96bcbcf5ef500b1f6bc73611a32d9133898f226444f8e5fcf17ca13bb3558aa.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/a96bcbcf5ef500b1f6bc73611a32d9133898f226444f8e5fcf17ca13bb3558aa.jpg)

![d6d334e9d63a02a43aa41a8212b66767da15f416db24ee42b051006214ebb740.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/d6d334e9d63a02a43aa41a8212b66767da15f416db24ee42b051006214ebb740.jpg)

![e0332b330fae6d3e1a506220d5ab4468a47a953bebd40b79816dad14cf1501e7.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/e0332b330fae6d3e1a506220d5ab4468a47a953bebd40b79816dad14cf1501e7.jpg)

![fc7ab7e3522b36c2e07273994e3cedcb63475fbb09a67ce33a9c1310b1740545.jpg](../icml_results/5056_PSMix_%20Robust%20Point%20Cloud%20Recognition%20through%20Spectral%20Domain%20Mixing/tables/fc7ab7e3522b36c2e07273994e3cedcb63475fbb09a67ce33a9c1310b1740545.jpg)

<a id="paper-5009"></a>

## Token-Free Hierarchical Indexing for RAG beyond LLM-based Summarization


### Images

![1c5417d23865dd6e59956358c642df342ca0eabaa79d82d441313d7bbc012849.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/images/1c5417d23865dd6e59956358c642df342ca0eabaa79d82d441313d7bbc012849.jpg)

![5fb22753bbce841aa8765fb9c3a9c81c204a9b81b9a4b43c97b81c46688597a9.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/images/5fb22753bbce841aa8765fb9c3a9c81c204a9b81b9a4b43c97b81c46688597a9.jpg)

![eb6dbe99d995fa4a7b05b3c0d679c5dc56a5bc44640dc6a86cd0024e5b6e4f91.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/images/eb6dbe99d995fa4a7b05b3c0d679c5dc56a5bc44640dc6a86cd0024e5b6e4f91.jpg)

### Tables

![02e6f0c09cd37859aab18344b92284d354b1f29dcb1f1d0ec7d6a6232270f9a3.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/tables/02e6f0c09cd37859aab18344b92284d354b1f29dcb1f1d0ec7d6a6232270f9a3.jpg)

![16a4da4f064f633ac2a0c28e1a696b7ce796b10894a7c3858d6bd3df8f1b991b.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/tables/16a4da4f064f633ac2a0c28e1a696b7ce796b10894a7c3858d6bd3df8f1b991b.jpg)

![41030abca2a94d89c665d886ffe637a1ef8997846ec8e1a5dae5ad69447b2d65.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/tables/41030abca2a94d89c665d886ffe637a1ef8997846ec8e1a5dae5ad69447b2d65.jpg)

![55c40cfe74661b8605011f275edd06e76406745ad522614fdf1415cf2d28ea02.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/tables/55c40cfe74661b8605011f275edd06e76406745ad522614fdf1415cf2d28ea02.jpg)

![ca36283d1affd3a3178e65487c4978a3e9ccc3047021a00cc7ba1fec446e04b7.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/tables/ca36283d1affd3a3178e65487c4978a3e9ccc3047021a00cc7ba1fec446e04b7.jpg)

![e776b3b703ed537befd529c0ceaa26a6007b3750b7ee8faed86071219b0895f2.jpg](../icml_results/5057_Token-Free%20Hierarchical%20Indexing%20for%20RAG%20beyond%20LLM-based%20Summarization/tables/e776b3b703ed537befd529c0ceaa26a6007b3750b7ee8faed86071219b0895f2.jpg)

<a id="paper-5010"></a>

## Open-Text Aerial Detection: A Unified Framework For Aerial Visual Grounding And Detection


### Images

![0540d545792a5301e1bf5512a571d83cff3daf10c83f6e92f5d8230f31d8606d.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/0540d545792a5301e1bf5512a571d83cff3daf10c83f6e92f5d8230f31d8606d.jpg)

![3f88c25fd3988b4619953c3331fbfef0a7b31fb5c23fa1d779610cdeedfa249b.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/3f88c25fd3988b4619953c3331fbfef0a7b31fb5c23fa1d779610cdeedfa249b.jpg)

![5251e6fa37cc2aee90fe7c49d0f2e955ebe1547f6058e3443078787112717be0.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/5251e6fa37cc2aee90fe7c49d0f2e955ebe1547f6058e3443078787112717be0.jpg)

![5f7b8131a1134991ba59579f95f8801aa44074bd4be1fc023c62b09bbfc28e70.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/5f7b8131a1134991ba59579f95f8801aa44074bd4be1fc023c62b09bbfc28e70.jpg)

![66779dabdda2dca9e53c3da9036a89eb2410cf51936a908fc470f4e61103ee42.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/66779dabdda2dca9e53c3da9036a89eb2410cf51936a908fc470f4e61103ee42.jpg)

![80c685766cec10d6fae2d638dd1773d92b242ba4d39030e4b26446e2e9052551.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/80c685766cec10d6fae2d638dd1773d92b242ba4d39030e4b26446e2e9052551.jpg)

![a83c73912375d769a675ad737d20608c55d6b9e044431f2f4a8e88d68c3346de.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/a83c73912375d769a675ad737d20608c55d6b9e044431f2f4a8e88d68c3346de.jpg)

![a8b6708ac45fb302e4940ecd4eb9bf9bc904f958d1d773d18b6ca7d0b1e1dc6e.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/a8b6708ac45fb302e4940ecd4eb9bf9bc904f958d1d773d18b6ca7d0b1e1dc6e.jpg)

![beb56f85e55f45e276439f8a67275ca76b9900568a00c0abbf4f7e105787cb2e.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/beb56f85e55f45e276439f8a67275ca76b9900568a00c0abbf4f7e105787cb2e.jpg)

![d397603c60ad64e10acc561aa1259d7aed75f335946c66dd43dea99cb53f9f39.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/images/d397603c60ad64e10acc561aa1259d7aed75f335946c66dd43dea99cb53f9f39.jpg)

### Tables

![45a4c1498754f4a5ddb32b098f6a631d598b7306d7e2c42359d513a8c68b2cb6.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/45a4c1498754f4a5ddb32b098f6a631d598b7306d7e2c42359d513a8c68b2cb6.jpg)

![466dbe668dad49fc7a79f49f92cf969140f4b89944e18aeb91adc47bc77aae9b.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/466dbe668dad49fc7a79f49f92cf969140f4b89944e18aeb91adc47bc77aae9b.jpg)

![634ed8594510f0195df453fbb85fecab2e0a403eb7315539515c59379eda274c.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/634ed8594510f0195df453fbb85fecab2e0a403eb7315539515c59379eda274c.jpg)

![6d3cb8619024bb9f4732f2746538b36bb2e063cfe79a7435c80e4bc8e6b83747.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/6d3cb8619024bb9f4732f2746538b36bb2e063cfe79a7435c80e4bc8e6b83747.jpg)

![73dabacdf7e3032093cb03ad3100f025183e0ba16f09128cd05da88f81df23c8.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/73dabacdf7e3032093cb03ad3100f025183e0ba16f09128cd05da88f81df23c8.jpg)

![aa049e518ae658ea75546ca9e3fe2c03223cb86f88f1c82b02d0679c703cca6f.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/aa049e518ae658ea75546ca9e3fe2c03223cb86f88f1c82b02d0679c703cca6f.jpg)

![b72c0695b3bd329848002c45b7eec8018638d5434cc46659acd5494bf8547e5e.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/b72c0695b3bd329848002c45b7eec8018638d5434cc46659acd5494bf8547e5e.jpg)

![c5d8c39d2ff200f76e0894b6ad4fcda815e01ef8e454651e2b8c4a6b67f54192.jpg](../icml_results/5058_Open-Text%20Aerial%20Detection_%20A%20Unified%20Framework%20For%20Aerial%20Visual%20Grounding%20And%20Detection/tables/c5d8c39d2ff200f76e0894b6ad4fcda815e01ef8e454651e2b8c4a6b67f54192.jpg)

<a id="paper-5011"></a>

## GeoLoom: High-quality Geometric Diagram Generation from Textual Input


### Images

![0065cf6a814d92abc5dd43064db49c5923d905020306429456c7365d0cd15eee.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/0065cf6a814d92abc5dd43064db49c5923d905020306429456c7365d0cd15eee.jpg)

![189b79b28a24f1b732ada98e46d88c8c29ca49cd41664cf744b70234f29fda80.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/189b79b28a24f1b732ada98e46d88c8c29ca49cd41664cf744b70234f29fda80.jpg)

![1d65c8be3926bf522a3f4ccdac57f9b919db67f0a5a967cb62268110d520a8a8.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/1d65c8be3926bf522a3f4ccdac57f9b919db67f0a5a967cb62268110d520a8a8.jpg)

![4189bf81dea0bd1ddda399260331b3c9e1cfcfce078e921e9d9a8cbf61ecf60a.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/4189bf81dea0bd1ddda399260331b3c9e1cfcfce078e921e9d9a8cbf61ecf60a.jpg)

![57ff85e9cd680de5df31b0c4029a950bd8e15176166c712af21388b884f75271.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/57ff85e9cd680de5df31b0c4029a950bd8e15176166c712af21388b884f75271.jpg)

![896a28ec74e3626e55f82d647549d881abf788d074305ce19db17bc661dcff82.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/896a28ec74e3626e55f82d647549d881abf788d074305ce19db17bc661dcff82.jpg)

![8e3f77d4c6a4f8f0e3dd66f3d3559e5162e868b80b115f498b925c3508d3939d.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/8e3f77d4c6a4f8f0e3dd66f3d3559e5162e868b80b115f498b925c3508d3939d.jpg)

![dce8e777697af7c6d8910f1f4390bcb6609788587c5dbc61a85f26212e23c597.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/dce8e777697af7c6d8910f1f4390bcb6609788587c5dbc61a85f26212e23c597.jpg)

![e48b96587e855ca55c498071b90dd82a28fbddcbd61d98ec36627c8b334cb578.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/e48b96587e855ca55c498071b90dd82a28fbddcbd61d98ec36627c8b334cb578.jpg)

![e50242da7be420849705a2d6eddc9ce491f4d5cfdab569f90a75544305f3e289.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/e50242da7be420849705a2d6eddc9ce491f4d5cfdab569f90a75544305f3e289.jpg)

![ed93277c0a49d8aef3be2f46e09821cb21e02e29eda6116e8a0b844e6686e05d.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/images/ed93277c0a49d8aef3be2f46e09821cb21e02e29eda6116e8a0b844e6686e05d.jpg)

### Tables

![1ca1c544cdb59f3e9fb881f23f1df56316e4ad66045a7437d2fbff6b456b8b3c.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/1ca1c544cdb59f3e9fb881f23f1df56316e4ad66045a7437d2fbff6b456b8b3c.jpg)

![1e00969999c5debf64e371c8cd4353d5b3ad1190fdb1972965b544ed4e36e868.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/1e00969999c5debf64e371c8cd4353d5b3ad1190fdb1972965b544ed4e36e868.jpg)

![1efe5acbb7a08dbf88c3446b74c96a9a41888f4097a265c039e53dd7b50d202a.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/1efe5acbb7a08dbf88c3446b74c96a9a41888f4097a265c039e53dd7b50d202a.jpg)

![2f382450011e07c30c51fdf3a29f1ec90eacbd08b82ac6a44bc57339e129d455.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/2f382450011e07c30c51fdf3a29f1ec90eacbd08b82ac6a44bc57339e129d455.jpg)

![46754e4090add8563f658f61a3537c0bce8293b56c8f85b8bee7e0139d66858a.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/46754e4090add8563f658f61a3537c0bce8293b56c8f85b8bee7e0139d66858a.jpg)

![4894fc1008ee07088e82b38614e83bdd356a416c8c269011e64ab57840ce78c8.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/4894fc1008ee07088e82b38614e83bdd356a416c8c269011e64ab57840ce78c8.jpg)

![4ed534f7e9fcbf35f7277e538b3f2fb96666b47f8f8411a0c2fe051c7fc80516.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/4ed534f7e9fcbf35f7277e538b3f2fb96666b47f8f8411a0c2fe051c7fc80516.jpg)

![6df56cd973f5fac2c177c11c98929e6b3a313176e6b86f60b35d7148f5234787.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/6df56cd973f5fac2c177c11c98929e6b3a313176e6b86f60b35d7148f5234787.jpg)

![784792f64b771912d4b16b19c1be9ec21d8dc50e8e3f238ecc56cb659eb152a1.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/784792f64b771912d4b16b19c1be9ec21d8dc50e8e3f238ecc56cb659eb152a1.jpg)

![88fa2d186a8c0efcaec1c6e54508f208552ec891db9c1e2f49ed8c4dd16daa9d.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/88fa2d186a8c0efcaec1c6e54508f208552ec891db9c1e2f49ed8c4dd16daa9d.jpg)

![92c76b19e2c36d4900f7a2870a8be5d9c1de3b9d6dacb9551840d36e6bdb9c48.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/92c76b19e2c36d4900f7a2870a8be5d9c1de3b9d6dacb9551840d36e6bdb9c48.jpg)

![c47a5d1c11646691a175206f9223ea9b11de84dba20d82730e632f3bc580ade4.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/c47a5d1c11646691a175206f9223ea9b11de84dba20d82730e632f3bc580ade4.jpg)

![e2d350a4dcdba1f5be226f8ab1afc6d031e988cf95e9db6a40454fc4b150c10e.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/e2d350a4dcdba1f5be226f8ab1afc6d031e988cf95e9db6a40454fc4b150c10e.jpg)

![fd3c2b87c0c0ef48da8fbb12fbb6951a7c6b5ea3f5802b4523db777a1ab16d6d.jpg](../icml_results/5059_GeoLoom_%20High-quality%20Geometric%20Diagram%20Generation%20from%20Textual%20Input/tables/fd3c2b87c0c0ef48da8fbb12fbb6951a7c6b5ea3f5802b4523db777a1ab16d6d.jpg)

<a id="paper-5012"></a>

## RECOVER: Reliable Detection of Unauthorized Data Usage in Text-to-Image Diffusion Models via Inversion Robustness


### Images

![0c44589576836865167424492e8e2302d432d8116970249f63f0aefaae625977.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/0c44589576836865167424492e8e2302d432d8116970249f63f0aefaae625977.jpg)

![287a0db6f3683c101b75c37f7571a7f053e1412829ae310158e277890837ab26.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/287a0db6f3683c101b75c37f7571a7f053e1412829ae310158e277890837ab26.jpg)

![2e373f6ffd1d08508c36a490e37c61c6d9f2706163c455cb1c933a8fde4f0928.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/2e373f6ffd1d08508c36a490e37c61c6d9f2706163c455cb1c933a8fde4f0928.jpg)

![31b6498fa083c74a15b8beb76ef8184aaa30df8c7ecf54cb4b590d65d26d01ae.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/31b6498fa083c74a15b8beb76ef8184aaa30df8c7ecf54cb4b590d65d26d01ae.jpg)

![50ced32090dae74fe595bedaf4bc779cf389dcce63c6fade3894ad3b11a10726.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/50ced32090dae74fe595bedaf4bc779cf389dcce63c6fade3894ad3b11a10726.jpg)

![61a4761f086f6a88eabaeb9c54776d1f431d053460a6bab440e26f41e55e2351.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/61a4761f086f6a88eabaeb9c54776d1f431d053460a6bab440e26f41e55e2351.jpg)

![6cd432c3dd8656bcdd66d839c6f7cfdf40cbb6a6ef93572375089ab2c02d074f.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/6cd432c3dd8656bcdd66d839c6f7cfdf40cbb6a6ef93572375089ab2c02d074f.jpg)

![859cf84b9184d449379265255a8b4bda25b85cc7270dea7bdf7e00c8774d9ea9.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/859cf84b9184d449379265255a8b4bda25b85cc7270dea7bdf7e00c8774d9ea9.jpg)

![971f45b09d7ce7dd807111adb28ed554b1466066c3aceca85101007cc6b0ef67.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/971f45b09d7ce7dd807111adb28ed554b1466066c3aceca85101007cc6b0ef67.jpg)

![9b26e445d82e789b843debd55f74069227a34e9f5eb70095d9d493cc623fcca0.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/9b26e445d82e789b843debd55f74069227a34e9f5eb70095d9d493cc623fcca0.jpg)

![a93b98f48ba4b200e76f9d25ea70ba43a3a564051322a43012f97d789c93881c.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/a93b98f48ba4b200e76f9d25ea70ba43a3a564051322a43012f97d789c93881c.jpg)

![bc49d06a3521a87650fc9d62efa0ed2b1e9f05370b717755d1ef3045bcfd3880.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/bc49d06a3521a87650fc9d62efa0ed2b1e9f05370b717755d1ef3045bcfd3880.jpg)

![e0368da13146234d4e0a1ac30212ca076cc5c3b378fa5c62501b03212f5f0072.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/e0368da13146234d4e0a1ac30212ca076cc5c3b378fa5c62501b03212f5f0072.jpg)

![f627401b5ff6e1d8553d9532be136e1ddbfe68f207a405935a71360bf08253b2.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/images/f627401b5ff6e1d8553d9532be136e1ddbfe68f207a405935a71360bf08253b2.jpg)

### Tables

![0b90bfe15487bb943806e53415a9f4a9c35ec79cca6b4d326ce9d9e050a6d917.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/0b90bfe15487bb943806e53415a9f4a9c35ec79cca6b4d326ce9d9e050a6d917.jpg)

![10b5c275b118fb595e7ba50b16346392f6d1132c9bf94e3394c093f9a1b38862.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/10b5c275b118fb595e7ba50b16346392f6d1132c9bf94e3394c093f9a1b38862.jpg)

![14cf2c9ed5d101247ee18f4a7d0793174609e7d5ae8f1e295edd91b0c6ce2882.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/14cf2c9ed5d101247ee18f4a7d0793174609e7d5ae8f1e295edd91b0c6ce2882.jpg)

![4c5797729550b19dd865f7b3074f23965300930c4c74c4403486bd0521a35e4c.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/4c5797729550b19dd865f7b3074f23965300930c4c74c4403486bd0521a35e4c.jpg)

![6c0f91ac7c512ec26898291388aa78410e4838c91e664ad315eb097c2a78084e.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/6c0f91ac7c512ec26898291388aa78410e4838c91e664ad315eb097c2a78084e.jpg)

![70b2d49abb86e6217b46ff4bfe6b6c0ba3e8a8ea79877c8a6702e770a68aa99b.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/70b2d49abb86e6217b46ff4bfe6b6c0ba3e8a8ea79877c8a6702e770a68aa99b.jpg)

![82ff1a49cdde3842e44ae1b757ad17e6d3ad937221dde9423082d24552da40e5.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/82ff1a49cdde3842e44ae1b757ad17e6d3ad937221dde9423082d24552da40e5.jpg)

![9260f867fc5d3c07c4d655962ac6d4f0ca25af9cbadad5b87b9ce1b59480ae1f.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/9260f867fc5d3c07c4d655962ac6d4f0ca25af9cbadad5b87b9ce1b59480ae1f.jpg)

![b25b93acdfc1d38ba57cee2278bb3a61870ca5a4ed6452b773d7c2f38c21449f.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/b25b93acdfc1d38ba57cee2278bb3a61870ca5a4ed6452b773d7c2f38c21449f.jpg)

![c7c538f81d69329b1937db807f75dfb78056208c627642a12b27a22fa1d81e49.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/c7c538f81d69329b1937db807f75dfb78056208c627642a12b27a22fa1d81e49.jpg)

![c9868d2acd9e221e8bf4b751245286599cd75e28dcb99ec68b3ce87ede6fe464.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/c9868d2acd9e221e8bf4b751245286599cd75e28dcb99ec68b3ce87ede6fe464.jpg)

![ce5804c4b8a5144651f95975d917a985cb163444db8f99110ee2f6475bc8b092.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/ce5804c4b8a5144651f95975d917a985cb163444db8f99110ee2f6475bc8b092.jpg)

![d10346a3ea37461a907bf609d88fe38904f5b150e0bfb15a1b149780ee647949.jpg](../icml_results/5060_RECOVER_%20Reliable%20Detection%20of%20Unauthorized%20Data%20Usage%20in%20Text-to-Image%20Diffusion%20Models%20via%20Inversi/tables/d10346a3ea37461a907bf609d88fe38904f5b150e0bfb15a1b149780ee647949.jpg)

<a id="paper-5013"></a>

## Exploration-free Algorithms for Multi-group Mean Estimation


### Images

![41536ec974eda092e93a6fee03625e717f1b04ec468112faa842fa8767d575c2.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/images/41536ec974eda092e93a6fee03625e717f1b04ec468112faa842fa8767d575c2.jpg)

![452a737404290fd921356dc4f99c8ead5441e4de22d3d9a0a8a749a93f15fba5.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/images/452a737404290fd921356dc4f99c8ead5441e4de22d3d9a0a8a749a93f15fba5.jpg)

![533caa14e8abeb145f420c4fd66d9a7c262d8f1379f4c1eb4839fe72da6e42ed.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/images/533caa14e8abeb145f420c4fd66d9a7c262d8f1379f4c1eb4839fe72da6e42ed.jpg)

![96d19838d667b7c85b8b5a9723eb4e15fc00cbb8f860eded03cf2cce9bfe828f.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/images/96d19838d667b7c85b8b5a9723eb4e15fc00cbb8f860eded03cf2cce9bfe828f.jpg)

![b6bfd2d395d515a8540bf8d28dfeb0bbe08dbc4d1b0617fb054fcea5861ae56d.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/images/b6bfd2d395d515a8540bf8d28dfeb0bbe08dbc4d1b0617fb054fcea5861ae56d.jpg)

![f7aa54794fc5c82a31078913cd3b29f3695a0f5dddd25724fbd8727c83248aae.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/images/f7aa54794fc5c82a31078913cd3b29f3695a0f5dddd25724fbd8727c83248aae.jpg)

### Tables

![c19e9558223a8405b875adfc243808f6aae1c3b3bd9a81a99d2464150bfd86ba.jpg](../icml_results/5061_Exploration-free%20Algorithms%20for%20Multi-group%20Mean%20Estimation/tables/c19e9558223a8405b875adfc243808f6aae1c3b3bd9a81a99d2464150bfd86ba.jpg)

<a id="paper-5014"></a>

## A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization


### Images

![2719f42371321541771725a02fcf789aea3d90044cd3b1b3db53d487b3c509ec.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/2719f42371321541771725a02fcf789aea3d90044cd3b1b3db53d487b3c509ec.jpg)

![552b8156c5e62756d55bbd2d240eb8ea48e025e8c7470496ce259e4aadc0acfa.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/552b8156c5e62756d55bbd2d240eb8ea48e025e8c7470496ce259e4aadc0acfa.jpg)

![7e580239d862e52a594a5eb04800dc1fb3b20af74c10a697858ffdede5d38fec.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/7e580239d862e52a594a5eb04800dc1fb3b20af74c10a697858ffdede5d38fec.jpg)

![8d5eda7c4130e60a3ae755fa15043cd5dd5c502fa20f3ef2c28ce33018782ed1.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/8d5eda7c4130e60a3ae755fa15043cd5dd5c502fa20f3ef2c28ce33018782ed1.jpg)

![9577f25e3d66100340acf877de0ddd7d038dcd841691dc0337431697313f0004.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/9577f25e3d66100340acf877de0ddd7d038dcd841691dc0337431697313f0004.jpg)

![e9bc438ae24c027272069eb7d03695ef5e5ea80a10c3762cc14e1ecb4519fec0.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/e9bc438ae24c027272069eb7d03695ef5e5ea80a10c3762cc14e1ecb4519fec0.jpg)

![fe18fe8816313c9fecf90bdfba6d0bf5dd075951f1a85263eea0de95f93ab9c0.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/images/fe18fe8816313c9fecf90bdfba6d0bf5dd075951f1a85263eea0de95f93ab9c0.jpg)

### Tables

![02c9ad4ec08f0d1e6de4e8fed48da5df3520e87fff921a11cf7c4d7d2cef341a.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/tables/02c9ad4ec08f0d1e6de4e8fed48da5df3520e87fff921a11cf7c4d7d2cef341a.jpg)

![0be4f2a97513891207c2c12e2782333efca74788b287eae253e646d1de174be5.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/tables/0be4f2a97513891207c2c12e2782333efca74788b287eae253e646d1de174be5.jpg)

![100f78d0632f039fe9d84d981cfd07cb5c3183c8013011163dacda285e4f3fbe.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/tables/100f78d0632f039fe9d84d981cfd07cb5c3183c8013011163dacda285e4f3fbe.jpg)

![16a32fd610753d0f4a45403e738c28fd8d30fe206f31c3d842fe9243d86c3ba4.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/tables/16a32fd610753d0f4a45403e738c28fd8d30fe206f31c3d842fe9243d86c3ba4.jpg)

![7dc7a214f75fec368ec60489cc4fda92668bb14a693a4334044920c3ceb11447.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/tables/7dc7a214f75fec368ec60489cc4fda92668bb14a693a4334044920c3ceb11447.jpg)

![990a203ab823aa2c18181ac3d62a055fa57f8cb5de8f38bcba9f5dd667343908.jpg](../icml_results/5062_A%20Geometry-Aware%20Efficient%20Algorithm%20for%20Compositional%20Entropic%20Risk%20Minimization/tables/990a203ab823aa2c18181ac3d62a055fa57f8cb5de8f38bcba9f5dd667343908.jpg)

<a id="paper-5015"></a>

## Scalable Bayesian Inference for Nonlinear Conservation Laws


### Images

![01a3392fd9092eaa827393a25d7e3bc3f8123f53286feb759383cf77efbf56ae.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/01a3392fd9092eaa827393a25d7e3bc3f8123f53286feb759383cf77efbf56ae.jpg)

![253df75d362787c2a6222142e6a26479bf6b1cf63dca4a350a7ece5d31eedda2.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/253df75d362787c2a6222142e6a26479bf6b1cf63dca4a350a7ece5d31eedda2.jpg)

![286e8911bab5083afb2aa4a98ca7df22d907673a5e596d481a51dd238a690ce0.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/286e8911bab5083afb2aa4a98ca7df22d907673a5e596d481a51dd238a690ce0.jpg)

![2f6f775244690d73c798a1a3a0443450c60d87fef8cfcc2181b9ed42a30a223c.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/2f6f775244690d73c798a1a3a0443450c60d87fef8cfcc2181b9ed42a30a223c.jpg)

![3d84c52c7608bced13e6909d4b3a4ae6ba97ca2000193804fd74b5e0651159b8.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/3d84c52c7608bced13e6909d4b3a4ae6ba97ca2000193804fd74b5e0651159b8.jpg)

![7c05e3cc9df904f6c4d72574f5134d5f65e2699dcf6cdda39d94c96abd48a92a.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/7c05e3cc9df904f6c4d72574f5134d5f65e2699dcf6cdda39d94c96abd48a92a.jpg)

![83ab0b7dc5e451b2e2a60bbde07fa43fc7ff5e02c2646f17cce612ff561b52aa.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/83ab0b7dc5e451b2e2a60bbde07fa43fc7ff5e02c2646f17cce612ff561b52aa.jpg)

![a5aff101c6090efb252165bd948dba50229d7cf2270c3f0ccdd912c15c3d25ff.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/a5aff101c6090efb252165bd948dba50229d7cf2270c3f0ccdd912c15c3d25ff.jpg)

![c324194de4694fc59391ce396c2d0e5771bdb9a4e8aafaee52c8bfd0d9db3245.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/c324194de4694fc59391ce396c2d0e5771bdb9a4e8aafaee52c8bfd0d9db3245.jpg)

![ed734352386b4f9bbedacb4af1922ca33a7761cdd1873517b18459344f3ae0c3.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/ed734352386b4f9bbedacb4af1922ca33a7761cdd1873517b18459344f3ae0c3.jpg)

![f1021402b497070a5b5932875a6a72b29ced4350c06a6c7802e47bc47e111523.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/f1021402b497070a5b5932875a6a72b29ced4350c06a6c7802e47bc47e111523.jpg)

![f55be475c0c6dd3d9106c7adf9319e9f5f318735591291f23c3f0355d493d8ac.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/f55be475c0c6dd3d9106c7adf9319e9f5f318735591291f23c3f0355d493d8ac.jpg)

![ff6817932d23e39d6f43b829969ca2933aca053a97fcbecddc0eb53c83b2ef6f.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/images/ff6817932d23e39d6f43b829969ca2933aca053a97fcbecddc0eb53c83b2ef6f.jpg)

### Tables

![2de3a5f070b3dc8a6ee0e90845e8ded5d2d8ba6ff90988b13c3d57944c2b1513.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/tables/2de3a5f070b3dc8a6ee0e90845e8ded5d2d8ba6ff90988b13c3d57944c2b1513.jpg)

![3c25903a7db86b0239f877b8ab0e2de64c1831f14250fa940aa330238b233aa4.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/tables/3c25903a7db86b0239f877b8ab0e2de64c1831f14250fa940aa330238b233aa4.jpg)

![731dfdcef80dc30a2ca8cbdd0d8691d323be3cd9de5a7c13f008df386118e7d0.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/tables/731dfdcef80dc30a2ca8cbdd0d8691d323be3cd9de5a7c13f008df386118e7d0.jpg)

![8b3e58c95c93fee0c9ca8200781a91d044eb93fc918d631b62561139f29406a4.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/tables/8b3e58c95c93fee0c9ca8200781a91d044eb93fc918d631b62561139f29406a4.jpg)

![8dccae0a62c6be61c53dffef19871ebeeff803b642bcaec7e3384e13713514e9.jpg](../icml_results/5063_Scalable%20Bayesian%20Inference%20for%20Nonlinear%20Conservation%20Laws/tables/8dccae0a62c6be61c53dffef19871ebeeff803b642bcaec7e3384e13713514e9.jpg)

<a id="paper-5016"></a>

## The Role of Target Update Frequencies in Q-Learning

### Images

![05e7db3fe2934bde8c58756b9836cbff251927361e20be808ef78798a49190a3.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/05e7db3fe2934bde8c58756b9836cbff251927361e20be808ef78798a49190a3.jpg)

![05fd92dd4665892cb8001a4216e90783eb82d50ea453b1d4b644edbb722f7eb7.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/05fd92dd4665892cb8001a4216e90783eb82d50ea453b1d4b644edbb722f7eb7.jpg)

![5ff56977d804f15e45d5cd0fff85d80e74011ff235c92ce49cde13becd63070b.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/5ff56977d804f15e45d5cd0fff85d80e74011ff235c92ce49cde13becd63070b.jpg)

![60b8f064a694121e1d1726bd3f793a3c4db90bef26f14bf39a9834d5b4e14aa6.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/60b8f064a694121e1d1726bd3f793a3c4db90bef26f14bf39a9834d5b4e14aa6.jpg)

![87a2a7d706c5e428909a4a116b10c4c4a837e2bcc629b20300a28f0311c19dbb.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/87a2a7d706c5e428909a4a116b10c4c4a837e2bcc629b20300a28f0311c19dbb.jpg)

![8fa7811db244e48ff6e04b35aeb0ca9fbe30c197275c7df0375a9374c56f4e3e.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/8fa7811db244e48ff6e04b35aeb0ca9fbe30c197275c7df0375a9374c56f4e3e.jpg)

![9aa68ade7a904d1fbc617b17052825ca9efa7eaf11b6897aa90e341dba89fdba.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/9aa68ade7a904d1fbc617b17052825ca9efa7eaf11b6897aa90e341dba89fdba.jpg)

![a888e28b71f87db7827c1915df887af2cc80509bdef6b96118716a5706bd5dd2.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/images/a888e28b71f87db7827c1915df887af2cc80509bdef6b96118716a5706bd5dd2.jpg)

### Tables

![104d63ebdaabbc55814f2a531781d51782b58e298e6a114a79a693120db97d86.jpg](../icml_results/5064_The%20Role%20of%20Target%20Update%20Frequencies%20in%20Q-Learning/tables/104d63ebdaabbc55814f2a531781d51782b58e298e6a114a79a693120db97d86.jpg)
