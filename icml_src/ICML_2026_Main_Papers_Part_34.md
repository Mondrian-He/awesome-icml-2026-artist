# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 34 of 197

## 目录 (Table of Contents)

1. [Show, Don’t Tell: Morphing Latent Reasoning into Image Generation](#paper-1090)
2. [Generalizable and Actionable Parts Pose Estimation with Symmetry Annotation-Free Learning Strategy](#paper-1091)
3. [LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation](#paper-1092)
4. [Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](#paper-1093)
5. [JANUS-LORA: A Balanced Low-Rank Adaptation for Continual Learning](#paper-1094)
6. [Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Spectral Super-Resolution for Snapshot Compressive Imaging](#paper-1095)
7. [CAPTCHA Solving for Native GUI Agents: Automated Reasoning-Action Data Generation and Self-Corrective Training](#paper-1096)
8. [Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs](#paper-1097)
9. [Transport or Discard: Robust Unbalanced Optimal Transport for Cross-Domain Policy Adaptation](#paper-1098)
10. [CGSVD: Cascaded Granular Singular Value Decomposition for Large Language Model Compression](#paper-1099)
11. [Leaderboard Incentives: Model Rankings under Strategic Post-Training](#paper-1100)
12. [Distributionally Robust Set Representation Learning Under Inference-Time Element Corruption](#paper-1101)
13. [JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG](#paper-1102)
14. [From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment](#paper-1103)
15. [DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use](#paper-1104)
16. [Offline Two-Player Zero-Sum Markov Games with KL Regularization](#paper-1105)
17. [Weaving Graph over Tokens: Contextualizing Structured Sequences for LLMs](#paper-1106)
18. [Nonlinear Covariate Balance in Experimental Design](#paper-1107)
19. [Any2Any: Unified Arbitrary Modality Translation for Remote Sensing](#paper-1108)
20. [DECOR: Learning to Decompose and Collaborate in Deep Search via Multi-Agent Reinforcement Learning](#paper-1109)
21. [Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging](#paper-1110)
22. [Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions](#paper-1111)
23. [Coarse-Grained Boltzmann Generators](#paper-1112)
24. [GraphP-FL: Personalized Federated Graph Learning via Dynamic Structure Awareness and Fisher Information Elastic Alignment](#paper-1113)
25. [Token-Sparse Medical Multimodal Reasoning via Dual-Stream Reinforcement Learning](#paper-1114)
26. [Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets](#paper-1115)
27. [ST-TGExplainer: Disentangling Stability and Transition Patterns for Temporal GNN Interpretability](#paper-1116)
28. [ImgCoT: Compressing Long Chain of Thought into Compact Visual Tokens for Efficient Reasoning of Large Language Model](#paper-1117)
29. [HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing](#paper-1118)
30. [RBCBF: Decoding Time Safety Alignment via Risk Guided Rollback and Barrier Control](#paper-1119)
31. [$A_2$DEPT: Large Language Model–Driven Automated Algorithm Design via Evolutionary Program Trees](#paper-1120)
32. [$\textitS$-SPPO: Semantic-Calibrated Self-Play Preference Optimization](#paper-1121)
33. [Rethinking the Flow-based Gradual Domain Adaptation: A Semi-Dual Optimal Transport Perspective](#paper-1122)

---


<a id="paper-1090"></a>

## Show, Don’t Tell: Morphing Latent Reasoning into Image Generation

### Images

![07b3a8ed71c9499e1d42c60123404b50ee881cd1c3e0c6f6832896ea408f89a6.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/07b3a8ed71c9499e1d42c60123404b50ee881cd1c3e0c6f6832896ea408f89a6.jpg)

![1b7f17e7426fd0106c3be3810476bbf1ae789b8309ffaf55324fe90e55bf3ed7.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/1b7f17e7426fd0106c3be3810476bbf1ae789b8309ffaf55324fe90e55bf3ed7.jpg)

![2a4c8c89985bf2d3433971414e4e89866934f968a54ff6bbf20e354149e9a47c.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/2a4c8c89985bf2d3433971414e4e89866934f968a54ff6bbf20e354149e9a47c.jpg)

![45b33fcf4effd12d26a4228fd453350fd4206555f74d4dc2ecb2cb04a2afdef7.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/45b33fcf4effd12d26a4228fd453350fd4206555f74d4dc2ecb2cb04a2afdef7.jpg)

![904441bab8d20172450bf148c67f1982ecc21e4159dda7f645746a207985b96a.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/904441bab8d20172450bf148c67f1982ecc21e4159dda7f645746a207985b96a.jpg)

![a8e3fbfa21157948b58d35cee70fbf1db1d1e5370fa002792927c38d9f6955f1.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/a8e3fbfa21157948b58d35cee70fbf1db1d1e5370fa002792927c38d9f6955f1.jpg)

![b13c2be15725a7097a9b1e3c9eac7c5d859797ba89b7920d0f51e632d817febd.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/b13c2be15725a7097a9b1e3c9eac7c5d859797ba89b7920d0f51e632d817febd.jpg)

![d7368a70612214ff8e699aa591dbc9f9cdae824a4ba0b5ee3f3bd3d5a37a748a.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/d7368a70612214ff8e699aa591dbc9f9cdae824a4ba0b5ee3f3bd3d5a37a748a.jpg)

![ff6f8026c9ceeb565b2b9fd409376c7082701f05cfd07eb69e315f90b1c10b12.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/ff6f8026c9ceeb565b2b9fd409376c7082701f05cfd07eb69e315f90b1c10b12.jpg)

### Tables

![13bcaebfa0a4211f75c0e47f49944cb9797415a2465fc316a8b62fd04e965452.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/13bcaebfa0a4211f75c0e47f49944cb9797415a2465fc316a8b62fd04e965452.jpg)

![27cecf789872bc7c4a13daaaf46f917c009aba2c002ba201a56ecdb0719788eb.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/27cecf789872bc7c4a13daaaf46f917c009aba2c002ba201a56ecdb0719788eb.jpg)

![2d139ab3d9065f8c62fe930b1b4873bbb1ac72a5fa89e05ab2b506518124bff7.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/2d139ab3d9065f8c62fe930b1b4873bbb1ac72a5fa89e05ab2b506518124bff7.jpg)

![3c09445bb3aed24b8af9c20b0cb62573b62db69e1660a99dbf60f260c5191459.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/3c09445bb3aed24b8af9c20b0cb62573b62db69e1660a99dbf60f260c5191459.jpg)

![4f82ec39673e620bcb6ec67d17a6cf26b590071eca8cfc1681fe45c019a0bfb5.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/4f82ec39673e620bcb6ec67d17a6cf26b590071eca8cfc1681fe45c019a0bfb5.jpg)

![4f9795d3a4139e0d6919b7fe0560ac0f0c75c4919d1e7ac6da03bd14227abb82.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/4f9795d3a4139e0d6919b7fe0560ac0f0c75c4919d1e7ac6da03bd14227abb82.jpg)

![5d8181f0ea9b8562b947adb6d8bccf492f013bb8db50d5cb035c1fd95675a93e.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/5d8181f0ea9b8562b947adb6d8bccf492f013bb8db50d5cb035c1fd95675a93e.jpg)

![6b28b34170f9821b45d5a1e7c0b3c84c684e997e6276e168dacad28aa52bd9a0.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/6b28b34170f9821b45d5a1e7c0b3c84c684e997e6276e168dacad28aa52bd9a0.jpg)

![7adc443c4ddcd577a8d5217d7f6dffe6286afec5eb072bc5bcf75830578254bb.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/7adc443c4ddcd577a8d5217d7f6dffe6286afec5eb072bc5bcf75830578254bb.jpg)

![811aa3569fb99ffcf3105613eb26c356fdb8ff6c5c077084d07dd1e6b2f3b456.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/811aa3569fb99ffcf3105613eb26c356fdb8ff6c5c077084d07dd1e6b2f3b456.jpg)

![9462ffc5b73326cd69a589157ac366786fe13c7a4681815b4110ae49bb48c5b6.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/9462ffc5b73326cd69a589157ac366786fe13c7a4681815b4110ae49bb48c5b6.jpg)

![9e9486f9c14d625d09505be940a8a1b97575c767d7b56490f0ac85e1e63beced.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/9e9486f9c14d625d09505be940a8a1b97575c767d7b56490f0ac85e1e63beced.jpg)

![bff31d29fbc049d09796bd1ab80256cc9b8e4635fd5cb58cb88aacd857fe1a4b.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/bff31d29fbc049d09796bd1ab80256cc9b8e4635fd5cb58cb88aacd857fe1a4b.jpg)

![c1c59da82ba66bf13fc3495b4e0f6511916c34b06da022795dd11959f6f0b4d2.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/c1c59da82ba66bf13fc3495b4e0f6511916c34b06da022795dd11959f6f0b4d2.jpg)

![c345c89a2e4bd9c1026c9820b5aa906e2aba7954b26a031ed51b6651beafe18b.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/c345c89a2e4bd9c1026c9820b5aa906e2aba7954b26a031ed51b6651beafe18b.jpg)

![c7d3ed16564d0a9dce04e3252a3be9a01f76af500493b615900f2c22d2f466e0.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/c7d3ed16564d0a9dce04e3252a3be9a01f76af500493b615900f2c22d2f466e0.jpg)

![ebfb57f7103d0747d4edbcdc4f0ea4e74aeb31e1ffa44dd331e17e3654843f83.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/ebfb57f7103d0747d4edbcdc4f0ea4e74aeb31e1ffa44dd331e17e3654843f83.jpg)

## Show, Don’t Tell: Morphing Latent Reasoning into Image Generation


### Images

![133e5f1953854aa9d13e17615fb79c5cb5d5aee8fdcae1a739898cb595108262.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/133e5f1953854aa9d13e17615fb79c5cb5d5aee8fdcae1a739898cb595108262.jpg)

![43791877cbb4c56b0eee02bb2cb9926085ee6d3e617f90037431804e0d54182d.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/43791877cbb4c56b0eee02bb2cb9926085ee6d3e617f90037431804e0d54182d.jpg)

![596f73b531bb2831282cd872b21e73432b7589314f2f63035454522af7fb0e69.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/596f73b531bb2831282cd872b21e73432b7589314f2f63035454522af7fb0e69.jpg)

![9b29e424e15b0a5b50fc7e042cb0e5fc04065820dc1f73d7cb5c02c40ba6b8e8.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/9b29e424e15b0a5b50fc7e042cb0e5fc04065820dc1f73d7cb5c02c40ba6b8e8.jpg)

![ca1aad95761ef3d9f5de24078b3f4ef5c60efe07dea9f82fb5b4099b7bf9af64.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/ca1aad95761ef3d9f5de24078b3f4ef5c60efe07dea9f82fb5b4099b7bf9af64.jpg)

![d74092329e437a6109a3da60538743653acb0482497e4a2ab0d6c525868c559e.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/d74092329e437a6109a3da60538743653acb0482497e4a2ab0d6c525868c559e.jpg)

![d74722cc704d533317ce795c6b1533bf78a4b2a6db68eebff80739f5ab5d0a3b.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/d74722cc704d533317ce795c6b1533bf78a4b2a6db68eebff80739f5ab5d0a3b.jpg)

![e7138b760a2642f11671d14981b42f9c614affda871f81f1319d7aa3bebb2d08.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/e7138b760a2642f11671d14981b42f9c614affda871f81f1319d7aa3bebb2d08.jpg)

![e9d88a4403f5b04f946be9d65c67d41e17bd9ea318348f94092806e70d8b5b9e.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/images/e9d88a4403f5b04f946be9d65c67d41e17bd9ea318348f94092806e70d8b5b9e.jpg)

### Tables

![1824d554466ecd98bc65b8664218f00f734ff8f55d858cef709a8f40100b281c.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/1824d554466ecd98bc65b8664218f00f734ff8f55d858cef709a8f40100b281c.jpg)

![5d64db55c588e1283c554dc125cf1af5a0ff78a2f8132e36b6752f9dca659ce8.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/5d64db55c588e1283c554dc125cf1af5a0ff78a2f8132e36b6752f9dca659ce8.jpg)

![81662cff86ff4c809d09ff109eb220e3cd7f754f48e2dcc9baddec75309d3162.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/81662cff86ff4c809d09ff109eb220e3cd7f754f48e2dcc9baddec75309d3162.jpg)

![818017444f323439a3acb56a7dfd1d192cbe85be2040d68293f7e211ac9de5be.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/818017444f323439a3acb56a7dfd1d192cbe85be2040d68293f7e211ac9de5be.jpg)

![a3bce4dbc0a280e1dac793f7e21a195afc6f68f6e0dd12face462a9017b49631.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/a3bce4dbc0a280e1dac793f7e21a195afc6f68f6e0dd12face462a9017b49631.jpg)

![cfda30cb1257e4db3a46415f327acb2d26ef7a4e5d5c1c3410e925adf2953d86.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/cfda30cb1257e4db3a46415f327acb2d26ef7a4e5d5c1c3410e925adf2953d86.jpg)

![f7693b7245c8513f57bfd1db8bc4fefd5ad56adc23a05f6e81ae3f72bc994bea.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/f7693b7245c8513f57bfd1db8bc4fefd5ad56adc23a05f6e81ae3f72bc994bea.jpg)

![f8dee3bc9c1e9a7dcd3a9c9f8149875ff3c2b1996fdb898703ed0d249b014391.jpg](../icml_results/1108_Show%2C%20Don%E2%80%99t%20Tell_%20Morphing%20Latent%20Reasoning%20into%20Image%20Generation/tables/f8dee3bc9c1e9a7dcd3a9c9f8149875ff3c2b1996fdb898703ed0d249b014391.jpg)

<a id="paper-1091"></a>

## Generalizable and Actionable Parts Pose Estimation with Symmetry Annotation-Free Learning Strategy


### Images

![23bfb6b42428d8bf8cebfdede7ade99fd87eb62bb7ee77473703fdc6a5780510.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/23bfb6b42428d8bf8cebfdede7ade99fd87eb62bb7ee77473703fdc6a5780510.jpg)

![2cc9e33babcf3858231201980b548404a6a3ef187af4e5db15cb825ed11d8aab.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/2cc9e33babcf3858231201980b548404a6a3ef187af4e5db15cb825ed11d8aab.jpg)

![3943c3d8d1a06e768ccd6683bccd0841822765a6b63e21a86aff6f3d0972c641.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/3943c3d8d1a06e768ccd6683bccd0841822765a6b63e21a86aff6f3d0972c641.jpg)

![738571069f6cdd054e3eaaf8667e9752f572699be506806d72ef27a8596eb0a2.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/738571069f6cdd054e3eaaf8667e9752f572699be506806d72ef27a8596eb0a2.jpg)

![77f2602c924ab0f1ba62ec1796488e0d00c2ded2e43fc95f8e20faf6964e4e05.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/77f2602c924ab0f1ba62ec1796488e0d00c2ded2e43fc95f8e20faf6964e4e05.jpg)

![875af3a93e8bfdc865c998930fece96bad409ec9d0a97daca9015f275295f0be.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/875af3a93e8bfdc865c998930fece96bad409ec9d0a97daca9015f275295f0be.jpg)

![af944ece2a00db7751684d327b0f18f14aaf58329e2d347f4d39b670b3fcac57.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/af944ece2a00db7751684d327b0f18f14aaf58329e2d347f4d39b670b3fcac57.jpg)

![ba108802cf5e55f81dfeeb96a3259ce7b41bc3e2ffe87784edbad4a5799a8774.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/ba108802cf5e55f81dfeeb96a3259ce7b41bc3e2ffe87784edbad4a5799a8774.jpg)

![e7158a8f7c1a7ad3c9bb9068ed1dedd3af044b34ba9afc92fd5d1c71e6705619.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/e7158a8f7c1a7ad3c9bb9068ed1dedd3af044b34ba9afc92fd5d1c71e6705619.jpg)

![fa4a2296e91870674d89f7a091e9864912a7d72128dfb549aca5bcc51d78fffa.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/images/fa4a2296e91870674d89f7a091e9864912a7d72128dfb549aca5bcc51d78fffa.jpg)

### Tables

![079921f18efe9bb3e4dedd5c47b883bcb3be8e0f0dcfd59d344f6efff5463786.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/079921f18efe9bb3e4dedd5c47b883bcb3be8e0f0dcfd59d344f6efff5463786.jpg)

![08186a8d6de17554e74c1d6d88390d2285122902dc65a74228643ef6fd525aa5.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/08186a8d6de17554e74c1d6d88390d2285122902dc65a74228643ef6fd525aa5.jpg)

![234f3e32318b4d4d758ae7301a8de6b8f2aafc4972eb8080d454dca8c87a51b3.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/234f3e32318b4d4d758ae7301a8de6b8f2aafc4972eb8080d454dca8c87a51b3.jpg)

![3bd24b33e6ca81131857bed2ce14eb4b8cf7798d927c0b4be2f8f689f3dc42d8.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/3bd24b33e6ca81131857bed2ce14eb4b8cf7798d927c0b4be2f8f689f3dc42d8.jpg)

![405ea775004be19af91958d0c9a83d387651c7464981da0b02494d6e890c39d5.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/405ea775004be19af91958d0c9a83d387651c7464981da0b02494d6e890c39d5.jpg)

![6fffe0ecdbf95496244a260c9972a35264103d5111ec6a1a1fe2278889cefcf9.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/6fffe0ecdbf95496244a260c9972a35264103d5111ec6a1a1fe2278889cefcf9.jpg)

![76e32ee989db73ea15e119bc71ccf174fdd4631b8536868ad29ecbabb8c4f5e0.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/76e32ee989db73ea15e119bc71ccf174fdd4631b8536868ad29ecbabb8c4f5e0.jpg)

![fd28060d365a6bb7a59c504dcfd4cf16d09db3b992583a1f1d7547b3e4b74b9b.jpg](../icml_results/1109_Generalizable%20and%20Actionable%20Parts%20Pose%20Estimation%20with%20Symmetry%20Annotation-Free%20Learning%20Strategy/tables/fd28060d365a6bb7a59c504dcfd4cf16d09db3b992583a1f1d7547b3e4b74b9b.jpg)

<a id="paper-1092"></a>

## LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation


### Images

![074d7b8ee3b745d6c85f436341330e1f8ce329fb4b9a514f515b3ca2d6142dd9.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/074d7b8ee3b745d6c85f436341330e1f8ce329fb4b9a514f515b3ca2d6142dd9.jpg)

![1beb7da30a23e12dccda777f3ec25af0af857d492d1021a8e96775e6ff4f53f6.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/1beb7da30a23e12dccda777f3ec25af0af857d492d1021a8e96775e6ff4f53f6.jpg)

![57f9f286059670ec4ba5de06a2a76fa1c54bbc679ec8233b1837ebb473935dc4.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/57f9f286059670ec4ba5de06a2a76fa1c54bbc679ec8233b1837ebb473935dc4.jpg)

![5c1f0cfc51eae2826476e65cbe1df4856c97f7b978166f2ed85f71ef29fed5ed.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/5c1f0cfc51eae2826476e65cbe1df4856c97f7b978166f2ed85f71ef29fed5ed.jpg)

![78d626bbda9bd47f7ff395858fc0ca1b8ba05d5d762455781a55ec081d62e39c.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/78d626bbda9bd47f7ff395858fc0ca1b8ba05d5d762455781a55ec081d62e39c.jpg)

![d8752149e89d37239e85315ed0d6ec74734e896d9e4722a3c6b2d2b181205180.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/d8752149e89d37239e85315ed0d6ec74734e896d9e4722a3c6b2d2b181205180.jpg)

![e78ce13f9c046b65a393d2b6e526b6a5dd1e8fb052444726c300c1849869bef4.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/e78ce13f9c046b65a393d2b6e526b6a5dd1e8fb052444726c300c1849869bef4.jpg)

![fcf6350dc60b349c5424a321fbcae1e2bf2fb1c263d20be30555b3d0b157c0fe.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/images/fcf6350dc60b349c5424a321fbcae1e2bf2fb1c263d20be30555b3d0b157c0fe.jpg)

### Tables

![08ec7dc231a5a04e3d7fce19f9dde21ee02252efc1837a91fe62766acfe1a796.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/08ec7dc231a5a04e3d7fce19f9dde21ee02252efc1837a91fe62766acfe1a796.jpg)

![1b49f6a898704bc397a46420d722cdacbfb5a02502bbd7385a25ac9f89675838.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/1b49f6a898704bc397a46420d722cdacbfb5a02502bbd7385a25ac9f89675838.jpg)

![1e29d3fc89dba78a5f9f84c800ace3e0844017d2fb30a0790a02b6310876092a.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/1e29d3fc89dba78a5f9f84c800ace3e0844017d2fb30a0790a02b6310876092a.jpg)

![2b54fd6399823bc76dacf0ab7a2981b8c986d24dc5085472367e7e8827090e38.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/2b54fd6399823bc76dacf0ab7a2981b8c986d24dc5085472367e7e8827090e38.jpg)

![7e9fdbae5426b36a2a9b0ad0136676c0bf96a1ef0a5bc3626090b32d4da0b9b6.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/7e9fdbae5426b36a2a9b0ad0136676c0bf96a1ef0a5bc3626090b32d4da0b9b6.jpg)

![8ac20b20f2ef7645bffd4210da72bff095e4acb58e543ee2375c5e8789997e6f.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/8ac20b20f2ef7645bffd4210da72bff095e4acb58e543ee2375c5e8789997e6f.jpg)

![99a43253f17a46d9904ada7605fd1974dec7ef5264561b1a55032e541ad03130.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/99a43253f17a46d9904ada7605fd1974dec7ef5264561b1a55032e541ad03130.jpg)

![9c7f6ed7849bc07fed2236293e173b2b7561cb6558d47097d7768009e1151d1c.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/9c7f6ed7849bc07fed2236293e173b2b7561cb6558d47097d7768009e1151d1c.jpg)

![b06da45f821920df34b4c017c6515ba521f51d7cd4106ac4bc5576acb9b292a9.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/b06da45f821920df34b4c017c6515ba521f51d7cd4106ac4bc5576acb9b292a9.jpg)

![e11dbf07133318d613931e0eb50ecd38b534fda5cdd00285d5eb1c1107d2e140.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/e11dbf07133318d613931e0eb50ecd38b534fda5cdd00285d5eb1c1107d2e140.jpg)

![f85b562d83be583e3a8f101a15e0cc850f108e44e0cfe17e2f55c9d3df01dce6.jpg](../icml_results/1110_LABO_%20LLM-Accelerated%20Bayesian%20Optimization%20through%20Broad%20Exploration%20and%20Selective%20Experimentation/tables/f85b562d83be583e3a8f101a15e0cc850f108e44e0cfe17e2f55c9d3df01dce6.jpg)

<a id="paper-1093"></a>

## Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills


### Images

![10d3f7d34bcbb01d0412772259b6da8e8390d46df0de7af13f6726383387fa70.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/10d3f7d34bcbb01d0412772259b6da8e8390d46df0de7af13f6726383387fa70.jpg)

![38e3f6ea079ea2f653c440d6e936697312d57712ee1abea473448978d41764c3.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/38e3f6ea079ea2f653c440d6e936697312d57712ee1abea473448978d41764c3.jpg)

![5a3e0a3f683c6ed4f67ba75551f46a8ee633b71c0876118afbe3f48633f196ee.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/5a3e0a3f683c6ed4f67ba75551f46a8ee633b71c0876118afbe3f48633f196ee.jpg)

![70e36f1d34883b24af4d175ef4fa6a036d07f15d916fcbf370719964c3eb9df7.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/70e36f1d34883b24af4d175ef4fa6a036d07f15d916fcbf370719964c3eb9df7.jpg)

![7e22f82b9019c8a18fae76e31697ed3d8cdbee3b9911776e7aaefd45171dbcac.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/7e22f82b9019c8a18fae76e31697ed3d8cdbee3b9911776e7aaefd45171dbcac.jpg)

![c5c4abd8dff858c4e176a5bcaa338ab47ecfae7f3dc60bcfd2c9f6f0835e0622.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/c5c4abd8dff858c4e176a5bcaa338ab47ecfae7f3dc60bcfd2c9f6f0835e0622.jpg)

![fad375df7a52d21357855e5930a4a17d365a58e3106089d43ada1249bb114a17.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/images/fad375df7a52d21357855e5930a4a17d365a58e3106089d43ada1249bb114a17.jpg)

### Tables

![00cb3f58dc60d9a0284ca3574237daea7c1042e45b5a623a756e436253e84994.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/00cb3f58dc60d9a0284ca3574237daea7c1042e45b5a623a756e436253e84994.jpg)

![2d21beb63c36374e4b373baaf3ef1ef52f4ecfa31343413d132c14f264a1cc0f.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/2d21beb63c36374e4b373baaf3ef1ef52f4ecfa31343413d132c14f264a1cc0f.jpg)

![39c4c44df10f1206befce8373c86561dbd0448a1d6ab78f38c27b23f5cfcffc5.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/39c4c44df10f1206befce8373c86561dbd0448a1d6ab78f38c27b23f5cfcffc5.jpg)

![415782872e72d9b0422886b42d73811e0c8cac9f471d7cc007c119cf517afd1d.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/415782872e72d9b0422886b42d73811e0c8cac9f471d7cc007c119cf517afd1d.jpg)

![4a0cbd3739cdef0d1b16ba278d1e30ea873ab4bad474f4de8dc95f59a5b6b7f4.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/4a0cbd3739cdef0d1b16ba278d1e30ea873ab4bad474f4de8dc95f59a5b6b7f4.jpg)

![4e8affcc37678c056ce68c768509536941ac013b4279dee183c89aa2429ccd93.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/4e8affcc37678c056ce68c768509536941ac013b4279dee183c89aa2429ccd93.jpg)

![5cf743a11862128b7500121b8077fa1a7ee2e4b69f507d28f60b974df03f1be6.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/5cf743a11862128b7500121b8077fa1a7ee2e4b69f507d28f60b974df03f1be6.jpg)

![60a35cd09a33bdac10725ff17799d383f23d5ce450ba99646d0bc1ce75050bd5.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/60a35cd09a33bdac10725ff17799d383f23d5ce450ba99646d0bc1ce75050bd5.jpg)

![6796778a98a4a3ebdd811ebcb6cec97d2568f724803c73c301863c53c91f9550.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/6796778a98a4a3ebdd811ebcb6cec97d2568f724803c73c301863c53c91f9550.jpg)

![873c267ddda1928ef3a69a4cf8977497b6f450932d919480091e7ca61613b687.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/873c267ddda1928ef3a69a4cf8977497b6f450932d919480091e7ca61613b687.jpg)

![a1f7ce7ea9438d04a9ebe3bc5a8924b98ceea15e9b42d4d06578ded92ca3e3f4.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/a1f7ce7ea9438d04a9ebe3bc5a8924b98ceea15e9b42d4d06578ded92ca3e3f4.jpg)

![b45c07f57eb28f0e024f4faf9687aa7e662d6377b230637eb7521e0af451b265.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/b45c07f57eb28f0e024f4faf9687aa7e662d6377b230637eb7521e0af451b265.jpg)

![beafaf63a86e6f260cee8c74393af1cf70bb719e025e92b468c973fbc9112aba.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/beafaf63a86e6f260cee8c74393af1cf70bb719e025e92b468c973fbc9112aba.jpg)

![cdbe2d14506830da0eb51027f69c475f4358d21e3eef3470fd3d24eb759fd78e.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/cdbe2d14506830da0eb51027f69c475f4358d21e3eef3470fd3d24eb759fd78e.jpg)

![d6c90b2d6f1d3a8d25db3865055feaff02d16ef57847b2ee1c14877d76ec7c2c.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/d6c90b2d6f1d3a8d25db3865055feaff02d16ef57847b2ee1c14877d76ec7c2c.jpg)

![de90f0d29421724dba070d1e48225facb875c523da2d11c52f3dd0a4c3aa8a6c.jpg](../icml_results/1111_Skill-Based%20Mixture-of-Experts_%20Adaptive%20Routing%20for%20Heterogeneous%20Reasoning%20via%20Inferred%20Skills/tables/de90f0d29421724dba070d1e48225facb875c523da2d11c52f3dd0a4c3aa8a6c.jpg)

<a id="paper-1094"></a>

## JANUS-LORA: A Balanced Low-Rank Adaptation for Continual Learning


### Images

![16442b7fefb0d1c5706b57a7a06796ce5ed31e5a3e6519cf20c35965d6cdcdb8.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/16442b7fefb0d1c5706b57a7a06796ce5ed31e5a3e6519cf20c35965d6cdcdb8.jpg)

![54509407bc61bcdc0cfaf50f90fae3b55c6c19274485975dad404a88fefb459f.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/54509407bc61bcdc0cfaf50f90fae3b55c6c19274485975dad404a88fefb459f.jpg)

![5c43104da62ab636b704c0bb4fe67b2dc498aa2426df6d11f82d472ca97859fd.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/5c43104da62ab636b704c0bb4fe67b2dc498aa2426df6d11f82d472ca97859fd.jpg)

![67256023d016926dedf29194fbcebd7076485b93425b8ce524f546fb924e0d8b.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/67256023d016926dedf29194fbcebd7076485b93425b8ce524f546fb924e0d8b.jpg)

![c679c2c07001d92c03ad18430d1ef315c9c8af8590fbb97b756360b4923b31eb.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/c679c2c07001d92c03ad18430d1ef315c9c8af8590fbb97b756360b4923b31eb.jpg)

![e33217fc1b5f1074f5cb36a9cc35eba4382547daff5a49fec812ec56ff0ac0fc.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/e33217fc1b5f1074f5cb36a9cc35eba4382547daff5a49fec812ec56ff0ac0fc.jpg)

![e4288fbf5a8dc0437ada5a194775138d7e65f19669a2e0a25c4c081e3623200f.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/e4288fbf5a8dc0437ada5a194775138d7e65f19669a2e0a25c4c081e3623200f.jpg)

![f9844d288e853e74f28567fc50305cfd7a58de846fc22f0143d0bc7592318b0c.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/f9844d288e853e74f28567fc50305cfd7a58de846fc22f0143d0bc7592318b0c.jpg)

![f9c436105d5f6f26c49e62700c242775e7e5dc212760318965a3eb84ffc02f88.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/f9c436105d5f6f26c49e62700c242775e7e5dc212760318965a3eb84ffc02f88.jpg)

![fdaa7e2dc245718c305e016890cddedd60caf878bddf344be601df81b69b2f3e.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/fdaa7e2dc245718c305e016890cddedd60caf878bddf344be601df81b69b2f3e.jpg)

![ffbd20b7bac0dfee94989ad9fcdefe8fe2c654340304bf9f280e8d8c5287e97f.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/images/ffbd20b7bac0dfee94989ad9fcdefe8fe2c654340304bf9f280e8d8c5287e97f.jpg)

### Tables

![30ffa79ba3a40336ded5222814c8247b64e5931415331535aedc75001d735685.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/30ffa79ba3a40336ded5222814c8247b64e5931415331535aedc75001d735685.jpg)

![71c70b0dd3565aa733a0e22003b91e85bca81ca1324aea9ba60b7879d8b8d6aa.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/71c70b0dd3565aa733a0e22003b91e85bca81ca1324aea9ba60b7879d8b8d6aa.jpg)

![97d46bd4f584aa0cfe83ef0b885b6ceaffe268755ebe9e7bda10b321a8df5756.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/97d46bd4f584aa0cfe83ef0b885b6ceaffe268755ebe9e7bda10b321a8df5756.jpg)

![add59822be570de37c20e1e045710fe41002458d15d03d77c87140c3592ecd4d.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/add59822be570de37c20e1e045710fe41002458d15d03d77c87140c3592ecd4d.jpg)

![dd288f642950d4853bccbec657fe10583b1d55f3adceac74000de0b85fb5488c.jpg](../icml_results/1112_JANUS-LORA_%20A%20Balanced%20Low-Rank%20Adaptation%20for%20Continual%20Learning/tables/dd288f642950d4853bccbec657fe10583b1d55f3adceac74000de0b85fb5488c.jpg)

<a id="paper-1095"></a>

## Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Spectral Super-Resolution for Snapshot Compressive Imaging


### Images

![036d7e522251ef112f1b2eff410156992365e6c65e1eff2a2d5e36246df1c652.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/036d7e522251ef112f1b2eff410156992365e6c65e1eff2a2d5e36246df1c652.jpg)

![26783611a13fb60823f9efca3b6c59cd48925f9ed113d4830f17f7c8713b99ab.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/26783611a13fb60823f9efca3b6c59cd48925f9ed113d4830f17f7c8713b99ab.jpg)

![313b7a004d7522fc5229594b033dc3725693d25450a7d2ea83c64c57cac8d054.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/313b7a004d7522fc5229594b033dc3725693d25450a7d2ea83c64c57cac8d054.jpg)

![415954e8105629cfd4d8cb976c2fbd0fefd9e4d8ded0f608056f9d7c7f1cc668.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/415954e8105629cfd4d8cb976c2fbd0fefd9e4d8ded0f608056f9d7c7f1cc668.jpg)

![9ac6f998daa3b6b31cc6a526d900ae86d5807abf53f052704fa0b5b9e665ceaa.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/9ac6f998daa3b6b31cc6a526d900ae86d5807abf53f052704fa0b5b9e665ceaa.jpg)

![aba4396dc3c0281704f856f0cf4b9043085ca4b3f6766aa6fe1b3cdd71a57c48.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/aba4396dc3c0281704f856f0cf4b9043085ca4b3f6766aa6fe1b3cdd71a57c48.jpg)

![b7f27c5e33f51e127575c5e42efb0af9f1088a5553957ca8f37c59d7556f17af.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/b7f27c5e33f51e127575c5e42efb0af9f1088a5553957ca8f37c59d7556f17af.jpg)

![bb43307d58ecc2a5a205604d86ceb62faa04b6175d92a4f34a171504eb1ccbc9.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/bb43307d58ecc2a5a205604d86ceb62faa04b6175d92a4f34a171504eb1ccbc9.jpg)

![bed5e336633534e97d6f37a1b81a9091f498f0e88f385aa35d34b7342072bacf.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/bed5e336633534e97d6f37a1b81a9091f498f0e88f385aa35d34b7342072bacf.jpg)

![e7e5060fb8265535d716c190aeb32efed768a2d45fd4916d8a80213ce0d2b5a0.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/images/e7e5060fb8265535d716c190aeb32efed768a2d45fd4916d8a80213ce0d2b5a0.jpg)

### Tables

![017ea61f602d543a54a830c00ead3c78cd8fcb6142fde3e44d34f310193b653a.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/017ea61f602d543a54a830c00ead3c78cd8fcb6142fde3e44d34f310193b653a.jpg)

![218c15f8c3ad239eeb29c9cc52fb90c284cdfffba1284b87024522da189e5b9e.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/218c15f8c3ad239eeb29c9cc52fb90c284cdfffba1284b87024522da189e5b9e.jpg)

![32905eada12bf5765b6f625866e05715ee85548dfa6bafbad0eaab3437cf2de4.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/32905eada12bf5765b6f625866e05715ee85548dfa6bafbad0eaab3437cf2de4.jpg)

![5c85bf52bade13bcbf03bd030445d8342ece8c696463cb0abd30c5032c1074a6.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/5c85bf52bade13bcbf03bd030445d8342ece8c696463cb0abd30c5032c1074a6.jpg)

![88421d60c31963ee462c6af176533ebf0ba969d5ad967f0cec15b5af4de7e4d0.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/88421d60c31963ee462c6af176533ebf0ba969d5ad967f0cec15b5af4de7e4d0.jpg)

![8a7df1953f5bd5501049ddd6bf19292dbef878abaf8fc17009843387dd432ce5.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/8a7df1953f5bd5501049ddd6bf19292dbef878abaf8fc17009843387dd432ce5.jpg)

![a855ad125d1e0216add8a56f27a94cfc65361ea60beee701882265dfd25c6163.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/a855ad125d1e0216add8a56f27a94cfc65361ea60beee701882265dfd25c6163.jpg)

![fcf0251fd09d5d75d5fa7a9aa8ee853474c492b9e4b66c3a98780cdf3001576d.jpg](../icml_results/1113_Phy-CoSF_%20Physics-Guided%20Continuous%20Spectral%20Fields%20Reconstruction%20and%20Spectral%20Super-Resolution%20for/tables/fcf0251fd09d5d75d5fa7a9aa8ee853474c492b9e4b66c3a98780cdf3001576d.jpg)

<a id="paper-1096"></a>

## CAPTCHA Solving for Native GUI Agents: Automated Reasoning-Action Data Generation and Self-Corrective Training


### Images

![0e78a7f628e61f39567f687617480e71395111ab99e60bb3efd89c988be3ce70.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/0e78a7f628e61f39567f687617480e71395111ab99e60bb3efd89c988be3ce70.jpg)

![32f3a570f2ee4c0dda358d5bfc23487e351e029dd6cb7f337bea618e39a1f544.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/32f3a570f2ee4c0dda358d5bfc23487e351e029dd6cb7f337bea618e39a1f544.jpg)

![38ddd0b3ffd29b17083410a33dc610c5c4fc691fcbb5f2f859312037e4523f88.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/38ddd0b3ffd29b17083410a33dc610c5c4fc691fcbb5f2f859312037e4523f88.jpg)

![3921da2b338461bbb2f6a553bd22780175e927e3f0bf1930c84bd5c7ada6f376.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/3921da2b338461bbb2f6a553bd22780175e927e3f0bf1930c84bd5c7ada6f376.jpg)

![458eed56d22d51b1195a8b48af136f9642050c75e029ad46653719f442532ee0.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/458eed56d22d51b1195a8b48af136f9642050c75e029ad46653719f442532ee0.jpg)

![562fd2634f1f2935d601ada87381f030e101138b6c7ed44f1c828919e6587d69.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/562fd2634f1f2935d601ada87381f030e101138b6c7ed44f1c828919e6587d69.jpg)

![5cd294d1d4cb7afd3d5c87e71858a9c4489e5e8843ff08057203baa39f86741d.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/5cd294d1d4cb7afd3d5c87e71858a9c4489e5e8843ff08057203baa39f86741d.jpg)

![6b270e83abdd129adb726f06ab0238a48743309c320314bd0de5dfac72d6bcd8.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/6b270e83abdd129adb726f06ab0238a48743309c320314bd0de5dfac72d6bcd8.jpg)

![93a8582c301e6d0c1383340aced5d3ac082751e2e965e25e8a85796e7983f1ee.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/93a8582c301e6d0c1383340aced5d3ac082751e2e965e25e8a85796e7983f1ee.jpg)

![b4d4dad943b45456f63ac645fcab55b9cdb29fb98ec28d93bfffef39b8cb72e6.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/b4d4dad943b45456f63ac645fcab55b9cdb29fb98ec28d93bfffef39b8cb72e6.jpg)

![c717437b7ee3757bdbab4f452210adf0c6d30400c4ef4755142acd0968ce0022.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/c717437b7ee3757bdbab4f452210adf0c6d30400c4ef4755142acd0968ce0022.jpg)

![eb8f08e204d20ec5b83eeb71889ff6b83ae3496dcc710b96d9c9f4f9588dd3fe.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/eb8f08e204d20ec5b83eeb71889ff6b83ae3496dcc710b96d9c9f4f9588dd3fe.jpg)

![ee1cc0d38b543350f92eb918e3150d272afd53a39a0bd5b8754d9d5fb4f16e07.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/images/ee1cc0d38b543350f92eb918e3150d272afd53a39a0bd5b8754d9d5fb4f16e07.jpg)

### Tables

![325c4555d0cc135ff0546a57b58eaa154d548e03280379e3bea0f98b5bc563df.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/325c4555d0cc135ff0546a57b58eaa154d548e03280379e3bea0f98b5bc563df.jpg)

![4da2e5354300c48a1ef37f40a3c53f636f3ffdf6d3dabf6080766ac375e9647d.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/4da2e5354300c48a1ef37f40a3c53f636f3ffdf6d3dabf6080766ac375e9647d.jpg)

![658b1e80a762317a7a9f4ce6188f70a45d8081cd0de50f0c5e8b78ff476b6392.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/658b1e80a762317a7a9f4ce6188f70a45d8081cd0de50f0c5e8b78ff476b6392.jpg)

![87440c0b47983668a399e50d2c029c0db92e674d789aaa14533544e857d358ec.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/87440c0b47983668a399e50d2c029c0db92e674d789aaa14533544e857d358ec.jpg)

![8df0ce0b3e36269bda0e3058ea03eb317d85a98cdf2ad62e06a6fcb89c3fc3dd.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/8df0ce0b3e36269bda0e3058ea03eb317d85a98cdf2ad62e06a6fcb89c3fc3dd.jpg)

![91499abe859166b744d72d0f72eea5d486a838548803d4df960ffd291d14bd5c.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/91499abe859166b744d72d0f72eea5d486a838548803d4df960ffd291d14bd5c.jpg)

![aa1c14eb9f7dd11a535d6bec8760bbcc7bae486a6db532b955327bdceb1275b4.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/aa1c14eb9f7dd11a535d6bec8760bbcc7bae486a6db532b955327bdceb1275b4.jpg)

![baad0e7693103c29eed0e730e78121519c3cc137f9685795faea0ff57fc98da7.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/baad0e7693103c29eed0e730e78121519c3cc137f9685795faea0ff57fc98da7.jpg)

![fc443fb9e1da01ff3e373cb75a19d766fd93855ad6d42f69f422e8024246b69b.jpg](../icml_results/1114_CAPTCHA%20Solving%20for%20Native%20GUI%20Agents_%20Automated%20Reasoning-Action%20Data%20Generation%20and%20Self-Correctiv/tables/fc443fb9e1da01ff3e373cb75a19d766fd93855ad6d42f69f422e8024246b69b.jpg)

<a id="paper-1097"></a>

## Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs


### Images

![3394082e2c6f68c1fea870366e61cbcc5795fb911d9039b7bf802801e7a1af20.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/images/3394082e2c6f68c1fea870366e61cbcc5795fb911d9039b7bf802801e7a1af20.jpg)

![43a6e96b942037314bc15240c72b93b9cd30f3db68552c5acca3ca52a3542b27.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/images/43a6e96b942037314bc15240c72b93b9cd30f3db68552c5acca3ca52a3542b27.jpg)

![76c91940c50e2cdbbd086fa7879d97e00f1dce9c8c8d61da9d9a878e72a0a227.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/images/76c91940c50e2cdbbd086fa7879d97e00f1dce9c8c8d61da9d9a878e72a0a227.jpg)

![af8f0c11b31ab189816c96dcbeb69fbdbc5c47e478787129fc6f6b682ab101ea.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/images/af8f0c11b31ab189816c96dcbeb69fbdbc5c47e478787129fc6f6b682ab101ea.jpg)

### Tables

![00a3ebd2848a4134a25b71d57a78fb4f01fa42ee84019a7f450c35f9358f6d77.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/00a3ebd2848a4134a25b71d57a78fb4f01fa42ee84019a7f450c35f9358f6d77.jpg)

![0bda74d28ee13ff1d5c154f9c5a4597f685a647e5f0e1884da8f559e014ef81f.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/0bda74d28ee13ff1d5c154f9c5a4597f685a647e5f0e1884da8f559e014ef81f.jpg)

![2895a291d17020de6eeb6ab79834f97a68907021395f3893b4652752b17c09a5.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/2895a291d17020de6eeb6ab79834f97a68907021395f3893b4652752b17c09a5.jpg)

![32410d36ee056ed414db714d4da8260acf25f1eb867c517c2acf0e9b97c9583f.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/32410d36ee056ed414db714d4da8260acf25f1eb867c517c2acf0e9b97c9583f.jpg)

![43783b33fcd086a256915d435c6ef194b9e337340cb41fe13d6434bf9b8a8842.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/43783b33fcd086a256915d435c6ef194b9e337340cb41fe13d6434bf9b8a8842.jpg)

![5478200660865774216828818a99137c7e52c198b87581b8948e42a5dd014d0f.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/5478200660865774216828818a99137c7e52c198b87581b8948e42a5dd014d0f.jpg)

![703c15b6d54b56737bca352f38d529a491f28903f4870a7c3f5b7d0afbf4d7c0.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/703c15b6d54b56737bca352f38d529a491f28903f4870a7c3f5b7d0afbf4d7c0.jpg)

![8ef486a53f1ca84cf6828bce6435571999c98c815489c96ec806d2e3065c5adc.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/8ef486a53f1ca84cf6828bce6435571999c98c815489c96ec806d2e3065c5adc.jpg)

![bdb11311abe56d576c2de18280afc7671f8f9d69a3d16e55178aa558916dd015.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/bdb11311abe56d576c2de18280afc7671f8f9d69a3d16e55178aa558916dd015.jpg)

![e48227f2763f8d3b43b0e5620f3cfa51fece8107080c0830b76f77ae09dc4ea5.jpg](../icml_results/1115_Learn%20from%20Your%20Mistakes_%20Tree-like%20Self-Play%20for%20Secure%20Code%20LLMs/tables/e48227f2763f8d3b43b0e5620f3cfa51fece8107080c0830b76f77ae09dc4ea5.jpg)

<a id="paper-1098"></a>

## Transport or Discard: Robust Unbalanced Optimal Transport for Cross-Domain Policy Adaptation


### Images

![0059b3dc9faa012bb8e1476d64ef8529c84684a10c9b971809f2a2482817d42f.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/images/0059b3dc9faa012bb8e1476d64ef8529c84684a10c9b971809f2a2482817d42f.jpg)

![168755682bcb23d5fb3328487165aa4c5942dcd3dbce27e8eee6ea0e7e00b085.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/images/168755682bcb23d5fb3328487165aa4c5942dcd3dbce27e8eee6ea0e7e00b085.jpg)

![41de079ce2afd31dd7ebe165755731db019c4ca42436272a90065a540a99a4da.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/images/41de079ce2afd31dd7ebe165755731db019c4ca42436272a90065a540a99a4da.jpg)

![43b8f26b061c43cab4e531e1ff4d303de4db9c84eaa0c490fb6ebc9a3c56ca06.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/images/43b8f26b061c43cab4e531e1ff4d303de4db9c84eaa0c490fb6ebc9a3c56ca06.jpg)

### Tables

![03fe6a39f96a0b0ba0f44742aca189799eb4a4d0490de0f13bfb00676241df11.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/03fe6a39f96a0b0ba0f44742aca189799eb4a4d0490de0f13bfb00676241df11.jpg)

![22767d1786e16c17ed23c09386b646f591d8e3601be544f4d2e3f4641b532a89.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/22767d1786e16c17ed23c09386b646f591d8e3601be544f4d2e3f4641b532a89.jpg)

![3dedc5e9b2b923e50bce054ed9bcf4a79671df3092a68085296648e3b9fc7119.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/3dedc5e9b2b923e50bce054ed9bcf4a79671df3092a68085296648e3b9fc7119.jpg)

![3ecd22d7abcd4c1709a9206a34c43ce519df4940dafc3a61def63b604ca62deb.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/3ecd22d7abcd4c1709a9206a34c43ce519df4940dafc3a61def63b604ca62deb.jpg)

![6664188f14cc9fc0f6829b59aa0153635bfe215bf3204e49ec3857400ff5ec87.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/6664188f14cc9fc0f6829b59aa0153635bfe215bf3204e49ec3857400ff5ec87.jpg)

![68fc266cb6dfdcab4cbacada207275647d1550ed111eb52991c6336eb30d1b26.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/68fc266cb6dfdcab4cbacada207275647d1550ed111eb52991c6336eb30d1b26.jpg)

![81b36931a98b2f5b04d5cef9c0a8a8143b2035891babf20fdafd071cb718290b.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/81b36931a98b2f5b04d5cef9c0a8a8143b2035891babf20fdafd071cb718290b.jpg)

![92522f16df9d42ca2cd9623dc4f47eb8661264793f9d67bae1e74e79a83ab3ef.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/92522f16df9d42ca2cd9623dc4f47eb8661264793f9d67bae1e74e79a83ab3ef.jpg)

![b0ac949acce0a65d4e105c865a25661cab5cbbf1d3267f8be7cf558a0dc30ed6.jpg](../icml_results/1116_Transport%20or%20Discard_%20Robust%20Unbalanced%20Optimal%20Transport%20for%20Cross-Domain%20Policy%20Adaptation/tables/b0ac949acce0a65d4e105c865a25661cab5cbbf1d3267f8be7cf558a0dc30ed6.jpg)

<a id="paper-1099"></a>

## CGSVD: Cascaded Granular Singular Value Decomposition for Large Language Model Compression


### Images

![0564f56bdf6f999c0f58dc3cb3ac917d2ac399523211dc3495b7f46a4012e850.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/0564f56bdf6f999c0f58dc3cb3ac917d2ac399523211dc3495b7f46a4012e850.jpg)

![b51e215320b310e70033dd16d27f15949a5a585fab921f3f2643df950bc48151.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/b51e215320b310e70033dd16d27f15949a5a585fab921f3f2643df950bc48151.jpg)

![b7863432e91d0daf0e780ff251349457abd370f5bb4f59c73f3a0d1977c5c1fa.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/b7863432e91d0daf0e780ff251349457abd370f5bb4f59c73f3a0d1977c5c1fa.jpg)

![bb1063d1e518e87dc7110178a1d5d7f16a62b509aefa6f7b72b05f29b7bb62db.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/bb1063d1e518e87dc7110178a1d5d7f16a62b509aefa6f7b72b05f29b7bb62db.jpg)

![e016431136289fc8e1f9e259c660a0c58a79a3df6fcd94e78488cb85437c6ca4.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/e016431136289fc8e1f9e259c660a0c58a79a3df6fcd94e78488cb85437c6ca4.jpg)

![febdee3d7caa96932e115958bfd3d6a2dc654adb179853136785b6c79fa6a1a3.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/febdee3d7caa96932e115958bfd3d6a2dc654adb179853136785b6c79fa6a1a3.jpg)

### Tables

![0996257d76645b357e3d7f00dc217dabe55c652272ff4d884088863f3a6add99.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/0996257d76645b357e3d7f00dc217dabe55c652272ff4d884088863f3a6add99.jpg)

![1b3857d1a2d4a3c5ed91980c52c463cca64ae2d1f546d6095cd2a5242169a155.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/1b3857d1a2d4a3c5ed91980c52c463cca64ae2d1f546d6095cd2a5242169a155.jpg)

![28167a89f0b3e20f6ff1ae9e990a92f901798f6c1ec5a2715921f3848c978a74.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/28167a89f0b3e20f6ff1ae9e990a92f901798f6c1ec5a2715921f3848c978a74.jpg)

![2ccf9ef15986e727d014912239b463d893b377f993cf22164283d6e67b6e1c81.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/2ccf9ef15986e727d014912239b463d893b377f993cf22164283d6e67b6e1c81.jpg)

![586cb793e73c7d38a27beeff43d66c031ffcd05733a2c7a423cff1ea0c916b59.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/586cb793e73c7d38a27beeff43d66c031ffcd05733a2c7a423cff1ea0c916b59.jpg)

![70b897a9a7f7a2c9cb99443a0f6a57fb136dc3440a2c338b1b97ffe712f63f95.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/70b897a9a7f7a2c9cb99443a0f6a57fb136dc3440a2c338b1b97ffe712f63f95.jpg)

![7f371a237688ce4216cdaa9c126a68504761f4a8a718ea2551058b005e6891c5.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/7f371a237688ce4216cdaa9c126a68504761f4a8a718ea2551058b005e6891c5.jpg)

![a65566f9ac80df2ccd56b982c72c9f0123b5c12e444616f78916950c9bd56545.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/a65566f9ac80df2ccd56b982c72c9f0123b5c12e444616f78916950c9bd56545.jpg)

![d98a15c062cd46cbca5dd9e8ca56819ea3b2d3851231d4f72ce244406e2100fe.jpg](../icml_results/1117_CGSVD_%20Cascaded%20Granular%20Singular%20Value%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/d98a15c062cd46cbca5dd9e8ca56819ea3b2d3851231d4f72ce244406e2100fe.jpg)

<a id="paper-1100"></a>

## Leaderboard Incentives: Model Rankings under Strategic Post-Training


### Images

![33da82837856b40fac161a6f5d7903558af6ed872721555c6b15bbe2ecbd5b83.jpg](../icml_results/1118_Leaderboard%20Incentives_%20Model%20Rankings%20under%20Strategic%20Post-Training/images/33da82837856b40fac161a6f5d7903558af6ed872721555c6b15bbe2ecbd5b83.jpg)

![738808a9f9df4e405f7698eb2fca6f9f5b414c091b32a6422ed28ef23b9d6ff0.jpg](../icml_results/1118_Leaderboard%20Incentives_%20Model%20Rankings%20under%20Strategic%20Post-Training/images/738808a9f9df4e405f7698eb2fca6f9f5b414c091b32a6422ed28ef23b9d6ff0.jpg)

![891b397ded116b499877fb353a25fb6e92181a176dc355a47322af510a854d30.jpg](../icml_results/1118_Leaderboard%20Incentives_%20Model%20Rankings%20under%20Strategic%20Post-Training/images/891b397ded116b499877fb353a25fb6e92181a176dc355a47322af510a854d30.jpg)

![b440e25d4d0e8f2c5a0f4847bf7e1b0bad186bdaa07238d55c03e2118554b559.jpg](../icml_results/1118_Leaderboard%20Incentives_%20Model%20Rankings%20under%20Strategic%20Post-Training/images/b440e25d4d0e8f2c5a0f4847bf7e1b0bad186bdaa07238d55c03e2118554b559.jpg)

![fa5d9acb909fe53567b39e57dc4c72674feb82b9da6c34d9508d1fce440eb724.jpg](../icml_results/1118_Leaderboard%20Incentives_%20Model%20Rankings%20under%20Strategic%20Post-Training/images/fa5d9acb909fe53567b39e57dc4c72674feb82b9da6c34d9508d1fce440eb724.jpg)

### Tables

![4df5bfa996c8bbee498364ddc4e3d6b0c1e11c813b18766d601abc4731802141.jpg](../icml_results/1118_Leaderboard%20Incentives_%20Model%20Rankings%20under%20Strategic%20Post-Training/tables/4df5bfa996c8bbee498364ddc4e3d6b0c1e11c813b18766d601abc4731802141.jpg)

<a id="paper-1101"></a>

## Distributionally Robust Set Representation Learning Under Inference-Time Element Corruption


### Images

![0ba12c5dd52db4541af3dc9480d2194e68d43a1b52b74cc42714a2e6cbce5112.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/images/0ba12c5dd52db4541af3dc9480d2194e68d43a1b52b74cc42714a2e6cbce5112.jpg)

![0c5959575f395ad9d5033d980287d6f0c1ad9bdf29c2a07b4381806e2f0a16e7.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/images/0c5959575f395ad9d5033d980287d6f0c1ad9bdf29c2a07b4381806e2f0a16e7.jpg)

![1b5964ba618779fe82dbc15cc324708a4ebcb6f0d878ca4cf5c78e9dfb2a1958.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/images/1b5964ba618779fe82dbc15cc324708a4ebcb6f0d878ca4cf5c78e9dfb2a1958.jpg)

![d01d1d7f702bd54f527bced4768fb397de4b01f55a64ca143c6c7b51b3cb8c2b.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/images/d01d1d7f702bd54f527bced4768fb397de4b01f55a64ca143c6c7b51b3cb8c2b.jpg)

### Tables

![05a8674a00de98329ee7752b247e1f07dccc5e657322771bf45c2a866d10981f.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/05a8674a00de98329ee7752b247e1f07dccc5e657322771bf45c2a866d10981f.jpg)

![0c2428a0d75269444f2cc237414d9c24f508e85d7d55c6c32d95e93689e5e3e8.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/0c2428a0d75269444f2cc237414d9c24f508e85d7d55c6c32d95e93689e5e3e8.jpg)

![149cd2c36fe2fd1d688f75c8755ae7b0b1331a1234a5b164dffdf59c5d3e5042.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/149cd2c36fe2fd1d688f75c8755ae7b0b1331a1234a5b164dffdf59c5d3e5042.jpg)

![19bdd66f74e0a2e940d00b550755aff559dc4930da4bb969b9a457104fc65787.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/19bdd66f74e0a2e940d00b550755aff559dc4930da4bb969b9a457104fc65787.jpg)

![24fc5403553e11b4dcd380217a8994a2a1d6362549f9333df925a21d4b7985df.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/24fc5403553e11b4dcd380217a8994a2a1d6362549f9333df925a21d4b7985df.jpg)

![38f336e9b7df99e6c3b907502f3913a56f365f66dd9cc2218fb837527261896b.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/38f336e9b7df99e6c3b907502f3913a56f365f66dd9cc2218fb837527261896b.jpg)

![4083d6ed5c753f59c1d77deba5e388771b5bb4dca6b02e3647f61818526ec49b.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/4083d6ed5c753f59c1d77deba5e388771b5bb4dca6b02e3647f61818526ec49b.jpg)

![65f78b2ca59fdcd2eaaf28f1a65aacbe6d63eb2fa48dc3dbdc1ba9befd8b9146.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/65f78b2ca59fdcd2eaaf28f1a65aacbe6d63eb2fa48dc3dbdc1ba9befd8b9146.jpg)

![784284b73c459a14b56701157f9aabb34d33dcf888d944c2ac00528c0ece9b7a.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/784284b73c459a14b56701157f9aabb34d33dcf888d944c2ac00528c0ece9b7a.jpg)

![7e0c7ded72bba629da692f3d277062ab4ceb15858fed38029198b1d8df24fb55.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/7e0c7ded72bba629da692f3d277062ab4ceb15858fed38029198b1d8df24fb55.jpg)

![904390e05914b7b1c2924332b3229de82bf64fd311c578817584f4f8d5646665.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/904390e05914b7b1c2924332b3229de82bf64fd311c578817584f4f8d5646665.jpg)

![a66433655de9100752781d8c4865bf037bdccae6de863fdd5b3ce100c38e3495.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/a66433655de9100752781d8c4865bf037bdccae6de863fdd5b3ce100c38e3495.jpg)

![ac1c5fac57046e5b840c2b08855e840bb45f58c8291a4ec77da2c5999a505187.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/ac1c5fac57046e5b840c2b08855e840bb45f58c8291a4ec77da2c5999a505187.jpg)

![af7a657b956ff19dc230c2c570caa80312852dd1f27bf62438454dba58f07c3b.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/af7a657b956ff19dc230c2c570caa80312852dd1f27bf62438454dba58f07c3b.jpg)

![c125095a981c8b0e0a7f0214beca28def4055e3928db40cfd6230a084e1d430f.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/c125095a981c8b0e0a7f0214beca28def4055e3928db40cfd6230a084e1d430f.jpg)

![c40c464d9f351a522f84a969592c10eca6584650eac053b867b032aba4cdee63.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/c40c464d9f351a522f84a969592c10eca6584650eac053b867b032aba4cdee63.jpg)

![ded179bab15e14c33b38b14d11ce06ddda00895ded1fd75a95d5b187a11e9e33.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/ded179bab15e14c33b38b14d11ce06ddda00895ded1fd75a95d5b187a11e9e33.jpg)

![f8247caa5f7816589551f2d95a603efd770d69067b4dc0695a485c95ae60e5c5.jpg](../icml_results/1119_Distributionally%20Robust%20Set%20Representation%20Learning%20Under%20Inference-Time%20Element%20Corruption/tables/f8247caa5f7816589551f2d95a603efd770d69067b4dc0695a485c95ae60e5c5.jpg)

<a id="paper-1102"></a>

## JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG


### Images

![1a5c99135b76dd718ecb3da1b83ae12ebfaa8b360f540686003c8d19faa71c0b.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/1a5c99135b76dd718ecb3da1b83ae12ebfaa8b360f540686003c8d19faa71c0b.jpg)

![2aba383f59012e55728f3a4f3e82e69035930347f2ba5876e27c34b27f24daa8.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/2aba383f59012e55728f3a4f3e82e69035930347f2ba5876e27c34b27f24daa8.jpg)

![34c0966a4bebb912bf6e8eb4d5c74dd783fe5c86a4fab0a6abba5074379b1e96.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/34c0966a4bebb912bf6e8eb4d5c74dd783fe5c86a4fab0a6abba5074379b1e96.jpg)

![87d7f2a97e8214a7d760fe71d7978f89fb07f0b8bf4d62450f78f095a6f8b33c.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/87d7f2a97e8214a7d760fe71d7978f89fb07f0b8bf4d62450f78f095a6f8b33c.jpg)

![ba17feb707542dc7fd9cb676eb0d00f51e1c2e5a6bf48e8b1d0c4420017e5bd0.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/ba17feb707542dc7fd9cb676eb0d00f51e1c2e5a6bf48e8b1d0c4420017e5bd0.jpg)

![d85dfec0f95dd07caac945487525062b211176461912f838c80922b15359526f.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/d85dfec0f95dd07caac945487525062b211176461912f838c80922b15359526f.jpg)

![ecc9eb181715af8796dc5f726b2d1ea3a977f25b73cf3b24215f0d54fc8a2ce3.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/images/ecc9eb181715af8796dc5f726b2d1ea3a977f25b73cf3b24215f0d54fc8a2ce3.jpg)

### Tables

![06834663ba38d690d52f4130af33011efb000e8b2616b7f79fb494a4bc0335e7.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/tables/06834663ba38d690d52f4130af33011efb000e8b2616b7f79fb494a4bc0335e7.jpg)

![93731cb8f5a7c96e039e97129a0e5b256d141be3b8ae57f69535a6917b31e129.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/tables/93731cb8f5a7c96e039e97129a0e5b256d141be3b8ae57f69535a6917b31e129.jpg)

![9dfcf4f62467c72e1b6948755e6a904cf01a0f71fb41620a8bb3926b9bee27e1.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/tables/9dfcf4f62467c72e1b6948755e6a904cf01a0f71fb41620a8bb3926b9bee27e1.jpg)

![a7757602137db7348988a4f59812679675344ebcb8834cbbee51b7191cffd9cf.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/tables/a7757602137db7348988a4f59812679675344ebcb8834cbbee51b7191cffd9cf.jpg)

![b388cdd9dc27ef32023ede1594c7046df00d380507bfb818edf4c451f210f9da.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/tables/b388cdd9dc27ef32023ede1594c7046df00d380507bfb818edf4c451f210f9da.jpg)

![f78623f79bf2c5b94a6137464afccccfecc854520900e505cfc7aa329a14fabb.jpg](../icml_results/1120_JADE_%20Bridging%20the%20Strategic-Operational%20Gap%20in%20Dynamic%20Agentic%20RAG/tables/f78623f79bf2c5b94a6137464afccccfecc854520900e505cfc7aa329a14fabb.jpg)

<a id="paper-1103"></a>

## From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment


### Images

![0e1f7979f7c326e7e9968a61580a203741d5ad3e4f083c2b06de2879725293c0.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/0e1f7979f7c326e7e9968a61580a203741d5ad3e4f083c2b06de2879725293c0.jpg)

![18572e116f168a361b9ecf9839af0426229c224dbe12fb11f69dc8194ac84c28.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/18572e116f168a361b9ecf9839af0426229c224dbe12fb11f69dc8194ac84c28.jpg)

![22b0296bbe09fc00c085b169ec7e34f3225e4343e925b616a71aeafbd19ba010.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/22b0296bbe09fc00c085b169ec7e34f3225e4343e925b616a71aeafbd19ba010.jpg)

![2955b32f3f0c0dc85afcf2e7ce918abf044b5b3c33feed8de6c60239fe575d50.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/2955b32f3f0c0dc85afcf2e7ce918abf044b5b3c33feed8de6c60239fe575d50.jpg)

![2edcf6b9dfa748ec76e32e7f64a5e64332c9f0a58cc16302c7dff8b3a4683581.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/2edcf6b9dfa748ec76e32e7f64a5e64332c9f0a58cc16302c7dff8b3a4683581.jpg)

![48fc4ecfc1379881b2eef76b8e2e655a64067f784a09580d1a90bfe028cdaf1f.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/48fc4ecfc1379881b2eef76b8e2e655a64067f784a09580d1a90bfe028cdaf1f.jpg)

![4d56e3760d6cef45fb6cd53a0b40087eeb9691587c03814c5f9e72e66a04b11c.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/4d56e3760d6cef45fb6cd53a0b40087eeb9691587c03814c5f9e72e66a04b11c.jpg)

![502ec731a6f195a3d6094faafa851038fc4b44f7cfa072ce692ffbfcc8910f99.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/502ec731a6f195a3d6094faafa851038fc4b44f7cfa072ce692ffbfcc8910f99.jpg)

![94ea4191be356d6bcd25a2ba045a287cc892ffe5a131ec8ab46be66421319bfe.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/94ea4191be356d6bcd25a2ba045a287cc892ffe5a131ec8ab46be66421319bfe.jpg)

![b0fbdb79f7814235b1560832e82d4022a4a409c0b6b2033d6e964e5780370433.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/images/b0fbdb79f7814235b1560832e82d4022a4a409c0b6b2033d6e964e5780370433.jpg)

### Tables

![07e6b1e21358749da27bbdbf85d1c46d579becd3a8d2e1e938ee254206397130.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/07e6b1e21358749da27bbdbf85d1c46d579becd3a8d2e1e938ee254206397130.jpg)

![19ec1bd9da06fd1531c2f8cedc4cf0204d1ce320b5c8a5dd49564a1994a55499.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/19ec1bd9da06fd1531c2f8cedc4cf0204d1ce320b5c8a5dd49564a1994a55499.jpg)

![1ca0763271a82de8eb95a1602d640f68a9a93dd767563fb6574980e602a2c381.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/1ca0763271a82de8eb95a1602d640f68a9a93dd767563fb6574980e602a2c381.jpg)

![22e80e33dbfb2d79c0ee24f9601c6ae815f239c551347dba281adefb2601a5fe.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/22e80e33dbfb2d79c0ee24f9601c6ae815f239c551347dba281adefb2601a5fe.jpg)

![2fd000cf18c02228e5cd5dbb702f8127f50306e65e6357542ba39e64bf8147c8.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/2fd000cf18c02228e5cd5dbb702f8127f50306e65e6357542ba39e64bf8147c8.jpg)

![31f881be60205c5a0eb41f7cd1a32f50d9e0312fe2ca3355c1bdf98bc73efcf8.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/31f881be60205c5a0eb41f7cd1a32f50d9e0312fe2ca3355c1bdf98bc73efcf8.jpg)

![4a647aaa4a60c2be0858c4e2a5dbc47ffe9a5a9751ae093bcb85fb11673fc18f.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/4a647aaa4a60c2be0858c4e2a5dbc47ffe9a5a9751ae093bcb85fb11673fc18f.jpg)

![4c2f7a97d751bb4803dd4aede7408176ab9c2b70f1f7498250b76419de77cbc1.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/4c2f7a97d751bb4803dd4aede7408176ab9c2b70f1f7498250b76419de77cbc1.jpg)

![55fc4585791f88655d868149aa0c472ebfb3dbee1e1d73e0dd7a638f1798bfb9.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/55fc4585791f88655d868149aa0c472ebfb3dbee1e1d73e0dd7a638f1798bfb9.jpg)

![5edb20d444959028289480654212a8585a28479af3a8af78a738c2ceeaaf7bda.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/5edb20d444959028289480654212a8585a28479af3a8af78a738c2ceeaaf7bda.jpg)

![6b19f68a0b90351a2fa8be60e7851d724ba24c7506bf62134dc431278678174d.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/6b19f68a0b90351a2fa8be60e7851d724ba24c7506bf62134dc431278678174d.jpg)

![76ab65cc8635bf0ddce4979524ff18b3e1d96599612a61268161f6e255734882.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/76ab65cc8635bf0ddce4979524ff18b3e1d96599612a61268161f6e255734882.jpg)

![77bbe1be37e189c2c711f7ad26b09ad77e9005b6f6dd54c33c6015fc6559b164.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/77bbe1be37e189c2c711f7ad26b09ad77e9005b6f6dd54c33c6015fc6559b164.jpg)

![79ef000c0e81bd6bcb9fbefef6d2231c7eb6fb3fdf590f113fafd7d40494f62c.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/79ef000c0e81bd6bcb9fbefef6d2231c7eb6fb3fdf590f113fafd7d40494f62c.jpg)

![a2507e442389ddbb729bee5cda0a7510f76e8e631b94077be17dc8a7b378b1dc.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/a2507e442389ddbb729bee5cda0a7510f76e8e631b94077be17dc8a7b378b1dc.jpg)

![a458f905674266976cea08d74b1dae8d05d99ec91e3fefe5e4f9383b4c9fa1a0.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/a458f905674266976cea08d74b1dae8d05d99ec91e3fefe5e4f9383b4c9fa1a0.jpg)

![ac7cf5bb8eb76fd18ee5211e3f5bef9c002d2636e8bedd5f2f436b21c49cf82c.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/ac7cf5bb8eb76fd18ee5211e3f5bef9c002d2636e8bedd5f2f436b21c49cf82c.jpg)

![b3f8fa737b37d0fec7c28c912e5cf36a60d221e2dcbd7c37b371c8ab0218a429.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/b3f8fa737b37d0fec7c28c912e5cf36a60d221e2dcbd7c37b371c8ab0218a429.jpg)

![d83e07e75f64cb3aa0b3c93d60713f5319d9d142c3d327019cfe68f0ec8683d9.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/d83e07e75f64cb3aa0b3c93d60713f5319d9d142c3d327019cfe68f0ec8683d9.jpg)

![db12e5b19475fadccb553462d835888226d6f4547336a59b2088017b6011aa17.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/db12e5b19475fadccb553462d835888226d6f4547336a59b2088017b6011aa17.jpg)

![dc1f5fd5530561753812aa9d48c9afdb3f6ad9506666b9406b0e3bf02e87c0de.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/dc1f5fd5530561753812aa9d48c9afdb3f6ad9506666b9406b0e3bf02e87c0de.jpg)

![deeeee4bbdb07d431c5e8c448e1ec645223bbba1165f36002701a5bcaaaa68b8.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/deeeee4bbdb07d431c5e8c448e1ec645223bbba1165f36002701a5bcaaaa68b8.jpg)

![e72cee418d711db06caf4d0e37be3cf718311a397613d2501cfb3524caf8f418.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/e72cee418d711db06caf4d0e37be3cf718311a397613d2501cfb3524caf8f418.jpg)

![faa4dd15955e723c67c6b333763234862735b5d24a16a4080d3a39b773a04127.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/faa4dd15955e723c67c6b333763234862735b5d24a16a4080d3a39b773a04127.jpg)

![fbca2dca020f9359e4fb67fc94b94be28d78cf0c6b85b761d525231326145109.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/fbca2dca020f9359e4fb67fc94b94be28d78cf0c6b85b761d525231326145109.jpg)

![fe290ddae11aca213b432b3d938b1ff22999ff637adc7e8e5215f7f2d06fbf52.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/fe290ddae11aca213b432b3d938b1ff22999ff637adc7e8e5215f7f2d06fbf52.jpg)

![fe9d09210048bb04e764b8b68c8276faa33194fd0a8be2eade706c0ad0979713.jpg](../icml_results/1121_From%20Parameters%20to%20Data_%20A%20Task-Parameter-Guided%20Fine-Tuning%20Pipeline%20for%20Efficient%20LLM%20Alignment/tables/fe9d09210048bb04e764b8b68c8276faa33194fd0a8be2eade706c0ad0979713.jpg)

<a id="paper-1104"></a>

## DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use


### Images

![227b3c49cad8a5a3f5a41228accd6d4b3a099974951c044c8beb0cc88abd07d9.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/227b3c49cad8a5a3f5a41228accd6d4b3a099974951c044c8beb0cc88abd07d9.jpg)

![4c7b546f131531003d00f0f3f44747c346132ff11842a20ad1fe79e4d73ecc9b.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/4c7b546f131531003d00f0f3f44747c346132ff11842a20ad1fe79e4d73ecc9b.jpg)

![560c2cb856b47e09ac878c398395b213f039529e4478261f2712972e31a7351d.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/560c2cb856b47e09ac878c398395b213f039529e4478261f2712972e31a7351d.jpg)

![5c4253e17c0c8d84df704c1b933eb1b9a39a867223433a9322504cd57a13928c.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/5c4253e17c0c8d84df704c1b933eb1b9a39a867223433a9322504cd57a13928c.jpg)

![838753ebdb0cfc0cb78f15fed8b03dfb906ed3f9fd0c1bf9ec5000de614c10fa.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/838753ebdb0cfc0cb78f15fed8b03dfb906ed3f9fd0c1bf9ec5000de614c10fa.jpg)

![921776cbee276a93cdcd5ea7480aa72a25e6fcf6f614d1aca72129122f65c47e.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/921776cbee276a93cdcd5ea7480aa72a25e6fcf6f614d1aca72129122f65c47e.jpg)

![d57107ffd67151411d56e542429a4b2c7d50ffd29f3d0f996ad57afd2bf77b86.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/images/d57107ffd67151411d56e542429a4b2c7d50ffd29f3d0f996ad57afd2bf77b86.jpg)

### Tables

![08bc720e930322bf9555de2b435a084f404f0e4103c80804c8beff2a65a841c9.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/08bc720e930322bf9555de2b435a084f404f0e4103c80804c8beff2a65a841c9.jpg)

![1f2659251dba85534e285bc95bb6ba055143856b2e65c5e32600b38f7d52b3ac.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/1f2659251dba85534e285bc95bb6ba055143856b2e65c5e32600b38f7d52b3ac.jpg)

![25e9f8f5d91eae8bf2f02960376e5ec6fd331b0d36b9ad532936759bf27a9262.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/25e9f8f5d91eae8bf2f02960376e5ec6fd331b0d36b9ad532936759bf27a9262.jpg)

![29fbd71e4abb54caef02179c7242fa939f9eaf6df96fabbd7b70338f64519936.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/29fbd71e4abb54caef02179c7242fa939f9eaf6df96fabbd7b70338f64519936.jpg)

![2a0978d4409bb5e179cc6317765840c9baf7197795da53ed2363d95eede77628.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/2a0978d4409bb5e179cc6317765840c9baf7197795da53ed2363d95eede77628.jpg)

![3dc3d7fab0e561e19f930832306045fa1773cd06bd7831891e5d20eebc85fa78.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/3dc3d7fab0e561e19f930832306045fa1773cd06bd7831891e5d20eebc85fa78.jpg)

![41a83cf3d71124d512363510a6a97069b0e4bdfb2a9fbd5b1d8d6fc04afd44e6.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/41a83cf3d71124d512363510a6a97069b0e4bdfb2a9fbd5b1d8d6fc04afd44e6.jpg)

![4a5eeee3dcae363168e72da33a64c5df71f7bacd07690b29ac77a1e64dc3bef5.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/4a5eeee3dcae363168e72da33a64c5df71f7bacd07690b29ac77a1e64dc3bef5.jpg)

![83dbc85738ca41c38bfd92830bc6ccdfd4b91a820a5fbf5400b5b422987cab14.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/83dbc85738ca41c38bfd92830bc6ccdfd4b91a820a5fbf5400b5b422987cab14.jpg)

![8abc42d4d78bbf92d13a446bbb6e8ca4a25976a9240d23798d7efe9d51e02e71.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/8abc42d4d78bbf92d13a446bbb6e8ca4a25976a9240d23798d7efe9d51e02e71.jpg)

![92b9e3e1e917e46f7498fa9289311021c814d3e295c0f9dcd62ddbfba0d16471.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/92b9e3e1e917e46f7498fa9289311021c814d3e295c0f9dcd62ddbfba0d16471.jpg)

![95981825d6e3d90b87cdb3db0de59d20a5bd63a358457cc7421f7938ceaf7e9c.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/95981825d6e3d90b87cdb3db0de59d20a5bd63a358457cc7421f7938ceaf7e9c.jpg)

![a5e5c07cfefe91fd6961f637906cf802f719c7eea69dd1ba16bf202924b54e18.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/a5e5c07cfefe91fd6961f637906cf802f719c7eea69dd1ba16bf202924b54e18.jpg)

![a9f150d12b36a16503c0a72f33f6cf601fc4d5436b27d3bd69c8f9fc07f56903.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/a9f150d12b36a16503c0a72f33f6cf601fc4d5436b27d3bd69c8f9fc07f56903.jpg)

![b7ae2ed9c40ddcf692b7fb53ea4d0b4bb786278b96a41ded01e4de53b5773b2a.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/b7ae2ed9c40ddcf692b7fb53ea4d0b4bb786278b96a41ded01e4de53b5773b2a.jpg)

![c3e230302022f13634c537a6cccdc9bc287682db8598ae47d83913c37dec0e84.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/c3e230302022f13634c537a6cccdc9bc287682db8598ae47d83913c37dec0e84.jpg)

![cedd78ef00decad4739705065d8e535d49e60d5705634819b7e95b0ffc25dc7c.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/cedd78ef00decad4739705065d8e535d49e60d5705634819b7e95b0ffc25dc7c.jpg)

![e369f9c2d0d4a2bdd6553066ff8589e38994e98a3b766b93fb847b0031adae55.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/e369f9c2d0d4a2bdd6553066ff8589e38994e98a3b766b93fb847b0031adae55.jpg)

![f5266e3b0d45ee53cf568aa8f176ff483db2850074104dbc9f230f356b216a23.jpg](../icml_results/1122_DIVE_%20Scaling%20Diversity%20in%20Agentic%20Task%20Synthesis%20for%20Generalizable%20Tool%20Use/tables/f5266e3b0d45ee53cf568aa8f176ff483db2850074104dbc9f230f356b216a23.jpg)

<a id="paper-1105"></a>

## Offline Two-Player Zero-Sum Markov Games with KL Regularization


### Tables

![13c246e15ffe73486da77dafdadeffca6fc4b4f65bbca4eaf4eecaea43d7f819.jpg](../icml_results/1123_Offline%20Two-Player%20Zero-Sum%20Markov%20Games%20with%20KL%20Regularization/tables/13c246e15ffe73486da77dafdadeffca6fc4b4f65bbca4eaf4eecaea43d7f819.jpg)

![93bf26a84bb0b002d33d88d9862a32036e80ee66cc10a4984b6a54b2261f9515.jpg](../icml_results/1123_Offline%20Two-Player%20Zero-Sum%20Markov%20Games%20with%20KL%20Regularization/tables/93bf26a84bb0b002d33d88d9862a32036e80ee66cc10a4984b6a54b2261f9515.jpg)

![95a282b7abf849623f75908de6f1a058d204c405814741b9ad242049d0b7d4f9.jpg](../icml_results/1123_Offline%20Two-Player%20Zero-Sum%20Markov%20Games%20with%20KL%20Regularization/tables/95a282b7abf849623f75908de6f1a058d204c405814741b9ad242049d0b7d4f9.jpg)

<a id="paper-1106"></a>

## Weaving Graph over Tokens: Contextualizing Structured Sequences for LLMs


### Images

![09625a544b707e831d850b777b2b128edc721368da32d10fddaf963bdf7ff46d.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/09625a544b707e831d850b777b2b128edc721368da32d10fddaf963bdf7ff46d.jpg)

![17c9e54fa4690e287ecdfcb65667e879863eb9a03b8f85123859ff89b4254a8f.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/17c9e54fa4690e287ecdfcb65667e879863eb9a03b8f85123859ff89b4254a8f.jpg)

![362e09af5f1306e35fd84c7fd5f4039389a35c9e5b591a0556585248350fb533.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/362e09af5f1306e35fd84c7fd5f4039389a35c9e5b591a0556585248350fb533.jpg)

![59284e42a622125536fdb6e73c648704ac2d6e3d9b924ec08f3be60790fb499b.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/59284e42a622125536fdb6e73c648704ac2d6e3d9b924ec08f3be60790fb499b.jpg)

![9e44b1d3b5d08cb6cefde99bea1de8affb31a52950ad48cc93ca6039e1cea38c.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/9e44b1d3b5d08cb6cefde99bea1de8affb31a52950ad48cc93ca6039e1cea38c.jpg)

![b538ec662f9a16f9fdc1da575aecbcba8a173727d85406ca5d18fed11560e2c4.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/b538ec662f9a16f9fdc1da575aecbcba8a173727d85406ca5d18fed11560e2c4.jpg)

![b77cce636ae6663ac91d1f4119e5d64ec19fa7c8863a911563ba25b7952ee86a.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/b77cce636ae6663ac91d1f4119e5d64ec19fa7c8863a911563ba25b7952ee86a.jpg)

![c7a691a6bd75dd7ad34434eb7a583e73c49b28ec0025f53a4e6726f5b3e24bc0.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/c7a691a6bd75dd7ad34434eb7a583e73c49b28ec0025f53a4e6726f5b3e24bc0.jpg)

![da11bfe1d8c1ef86eee58d471033d45e2f6630810892b3fdae8532aa77c9e75d.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/da11bfe1d8c1ef86eee58d471033d45e2f6630810892b3fdae8532aa77c9e75d.jpg)

![e04efaa46358b1fb1fe86eb76ea0fc20afebd3b105b56211512a4947d2a318de.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/e04efaa46358b1fb1fe86eb76ea0fc20afebd3b105b56211512a4947d2a318de.jpg)

![e7bd3aaae0d8a334840b7b360ff34a23a08100c5a9b9258ad187ed8006c898b0.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/e7bd3aaae0d8a334840b7b360ff34a23a08100c5a9b9258ad187ed8006c898b0.jpg)

![ed8f813af1b3488a0a05ab7e094e0bab7801d0d931a75c87b0c9c724788f05e8.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/ed8f813af1b3488a0a05ab7e094e0bab7801d0d931a75c87b0c9c724788f05e8.jpg)

![edf56ef7a0ecfe14977598caeb3053916c412f2b81f58f6cace56259e2c8174b.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/edf56ef7a0ecfe14977598caeb3053916c412f2b81f58f6cace56259e2c8174b.jpg)

![f2907cca862c4c7f01b81891769b3ecbb56d0c5f4fefacf62f743f8f67036996.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/f2907cca862c4c7f01b81891769b3ecbb56d0c5f4fefacf62f743f8f67036996.jpg)

![fde2c5157395afe50858f78b66c6a0a462a30aa666bc69550e1dc11688ba1454.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/images/fde2c5157395afe50858f78b66c6a0a462a30aa666bc69550e1dc11688ba1454.jpg)

### Tables

![0b1380e7de54e0cce2dac9cce678caa185a6014de2ff84f9090d59c76da7ce4e.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/0b1380e7de54e0cce2dac9cce678caa185a6014de2ff84f9090d59c76da7ce4e.jpg)

![2ccbb4c6198e4179e6729dfa76e446f32f57d75a6ebcd35e90b2a20a8b0e105c.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/2ccbb4c6198e4179e6729dfa76e446f32f57d75a6ebcd35e90b2a20a8b0e105c.jpg)

![5e6b333399c45cb224e5e85f11c9bec76799810fba18c5f5f7c167431e119630.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/5e6b333399c45cb224e5e85f11c9bec76799810fba18c5f5f7c167431e119630.jpg)

![6329a6204e0460598f20b796ccfc6c413c3e9caa54d7dbfde5937f084f006c77.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/6329a6204e0460598f20b796ccfc6c413c3e9caa54d7dbfde5937f084f006c77.jpg)

![647311474ea7886145a8492e08ce51c0aea05d1e287a403cd37e82856da04377.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/647311474ea7886145a8492e08ce51c0aea05d1e287a403cd37e82856da04377.jpg)

![7a4fbfd558c6549694294f8605d7122d7e824073699fc435bb8c70c96f705b75.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/7a4fbfd558c6549694294f8605d7122d7e824073699fc435bb8c70c96f705b75.jpg)

![840902288aa5adeaa27c431a9b40f092fc433b8d501c4e93605dca24c2278ef3.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/840902288aa5adeaa27c431a9b40f092fc433b8d501c4e93605dca24c2278ef3.jpg)

![d806c1aa475088ffc7744666029e7bfdcd12bb493787eca479389571d4114d55.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/d806c1aa475088ffc7744666029e7bfdcd12bb493787eca479389571d4114d55.jpg)

![e7381f8cad775ec8cbc5fa08a261053188eb326db998d5dc77ab87110ef39581.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/e7381f8cad775ec8cbc5fa08a261053188eb326db998d5dc77ab87110ef39581.jpg)

![ed0a04cb2ab6ba982ea3926c63d2f5dcc1a434ba8fa21e1a1682444eeded3880.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/ed0a04cb2ab6ba982ea3926c63d2f5dcc1a434ba8fa21e1a1682444eeded3880.jpg)

![f7b825f028c474ac5a411a10540f3018e7d2187065e4ff7d619e5720245e7399.jpg](../icml_results/1124_Weaving%20Graph%20over%20Tokens_%20Contextualizing%20Structured%20Sequences%20for%20LLMs/tables/f7b825f028c474ac5a411a10540f3018e7d2187065e4ff7d619e5720245e7399.jpg)

<a id="paper-1107"></a>

## Nonlinear Covariate Balance in Experimental Design


### Images

![806affb7aa6f672eb14b35b29d4b6740d2f9f424c33eed02ff8bb88a3f06c4cb.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/images/806affb7aa6f672eb14b35b29d4b6740d2f9f424c33eed02ff8bb88a3f06c4cb.jpg)

![81dac7be2d802e4a88e104aa3f158b638e4bc93111fe442824aefd871b40bcfe.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/images/81dac7be2d802e4a88e104aa3f158b638e4bc93111fe442824aefd871b40bcfe.jpg)

![ca08dd1ea9d41dd8e7f6ad29ea6a63d7ba59227969da47edee94e887752b0092.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/images/ca08dd1ea9d41dd8e7f6ad29ea6a63d7ba59227969da47edee94e887752b0092.jpg)

![daa9eb6700369d0517853dd319178362608d09930f4abca8c25d79ea2a2d294b.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/images/daa9eb6700369d0517853dd319178362608d09930f4abca8c25d79ea2a2d294b.jpg)

### Tables

![2914044737db53f395e8c082215c50e4fcf6d2288227484323d23f993d138df6.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/2914044737db53f395e8c082215c50e4fcf6d2288227484323d23f993d138df6.jpg)

![57b006153ae3ae96931f79d58119ac6aa48953774c3a362093006baac0e374f4.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/57b006153ae3ae96931f79d58119ac6aa48953774c3a362093006baac0e374f4.jpg)

![761d2da4e5df2093312c95e544c6e444fcf92840b59222f9c7b5e6cb1cb3dd2a.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/761d2da4e5df2093312c95e544c6e444fcf92840b59222f9c7b5e6cb1cb3dd2a.jpg)

![7a8be49d877238ac806d586725f246d61e4a792b150d68df51d7c7d651152c41.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/7a8be49d877238ac806d586725f246d61e4a792b150d68df51d7c7d651152c41.jpg)

![7fcd53871a931f2b74c012a567eabf6dd7c5db0b7086bf4f3926ca45c41c6ebb.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/7fcd53871a931f2b74c012a567eabf6dd7c5db0b7086bf4f3926ca45c41c6ebb.jpg)

![87d6b836b6759e61519deeda83c4257bdeb117795d2ee01ae7b0bbcd69fae76a.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/87d6b836b6759e61519deeda83c4257bdeb117795d2ee01ae7b0bbcd69fae76a.jpg)

![9e732006795358270f50ae2e59d11ef9fd0c0826d08d7d52e18c657d648aa84f.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/9e732006795358270f50ae2e59d11ef9fd0c0826d08d7d52e18c657d648aa84f.jpg)

![af8918e88aaffd1a6a05b0226465f4556dde8bf0f5d4c671324692cfc8de410e.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/af8918e88aaffd1a6a05b0226465f4556dde8bf0f5d4c671324692cfc8de410e.jpg)

![b1d4c7433336ceb3a2f9a86cfb5e70e585eea04bc14e4745b598bbb6ff036d6a.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/b1d4c7433336ceb3a2f9a86cfb5e70e585eea04bc14e4745b598bbb6ff036d6a.jpg)

![bf662abeb18a71fab11858fda4bb42f9d91e5d1028186f9bb36bc9e7242f1315.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/bf662abeb18a71fab11858fda4bb42f9d91e5d1028186f9bb36bc9e7242f1315.jpg)

![e2c9f97602e07172156db0821079dc5d5506d27ec8952c1521303ec560d8bd78.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/e2c9f97602e07172156db0821079dc5d5506d27ec8952c1521303ec560d8bd78.jpg)

![e3a3d2b9f0263eb5e55fbb5d357904e0640bd3f3e99a80638d83dc6707d47172.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/e3a3d2b9f0263eb5e55fbb5d357904e0640bd3f3e99a80638d83dc6707d47172.jpg)

![f60e1e6d5abca579c058afac1fc32e83e03970344f928782c9d945366d0b6526.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/f60e1e6d5abca579c058afac1fc32e83e03970344f928782c9d945366d0b6526.jpg)

![fc4a1915acaa2c9abd673f76e1698539430617a125ac2a059df031c991ecd012.jpg](../icml_results/1125_Nonlinear%20Covariate%20Balance%20in%20Experimental%20Design/tables/fc4a1915acaa2c9abd673f76e1698539430617a125ac2a059df031c991ecd012.jpg)

<a id="paper-1108"></a>

## Any2Any: Unified Arbitrary Modality Translation for Remote Sensing


### Images

![0e6b0ce20ec6822447fd84b8912a01da531d9724c8d102923526b073ea7668e2.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/0e6b0ce20ec6822447fd84b8912a01da531d9724c8d102923526b073ea7668e2.jpg)

![3c032da1b423e0eb51117110d214b6e9746a9220dfac877034c4211f7031ae2a.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/3c032da1b423e0eb51117110d214b6e9746a9220dfac877034c4211f7031ae2a.jpg)

![45571d3f1ff303f5c1caec1cc3ca411817a5540c87749af13e85fdeec9e1edcc.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/45571d3f1ff303f5c1caec1cc3ca411817a5540c87749af13e85fdeec9e1edcc.jpg)

![4735feb7c0e15fa93619dbbd2d4eea5c2edb2c84b19de7fb10feab8274274c40.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/4735feb7c0e15fa93619dbbd2d4eea5c2edb2c84b19de7fb10feab8274274c40.jpg)

![8f8dd4fcf8cbaa5857338be633b6a04e4cfe4e48b1cc685521ef0dadd0585f12.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/8f8dd4fcf8cbaa5857338be633b6a04e4cfe4e48b1cc685521ef0dadd0585f12.jpg)

![92a741824358d6ac76d5b28d589060b967de14360bcf2e889521a0f8da974956.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/92a741824358d6ac76d5b28d589060b967de14360bcf2e889521a0f8da974956.jpg)

![c88ce2f47ff7bbcc94d101fcfcb858829acc21269b5905c41ca93012f458a29d.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/images/c88ce2f47ff7bbcc94d101fcfcb858829acc21269b5905c41ca93012f458a29d.jpg)

### Tables

![0127953f902503712a28c00f4d515015884a63433fbcf5fcc15e531f6c9da6a4.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/tables/0127953f902503712a28c00f4d515015884a63433fbcf5fcc15e531f6c9da6a4.jpg)

![6fb9d544d9589d7d238fd6d73dc6f81d52b51b1e9241a5b554d9b98591fca870.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/tables/6fb9d544d9589d7d238fd6d73dc6f81d52b51b1e9241a5b554d9b98591fca870.jpg)

![98685e7c585bc941a8b582f950d0a27c55c43c9fcfe561ded711ea122fe7e789.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/tables/98685e7c585bc941a8b582f950d0a27c55c43c9fcfe561ded711ea122fe7e789.jpg)

![fbba21673f0284e5491629312daee56f6a97e6e1d0893134e9783143f2e5ccb8.jpg](../icml_results/1126_Any2Any_%20Unified%20Arbitrary%20Modality%20Translation%20for%20Remote%20Sensing/tables/fbba21673f0284e5491629312daee56f6a97e6e1d0893134e9783143f2e5ccb8.jpg)

<a id="paper-1109"></a>

## DECOR: Learning to Decompose and Collaborate in Deep Search via Multi-Agent Reinforcement Learning


### Images

![7e4074c2e099b2d97c4591b910dae8810f7cfa14c10d25bbc57a67ded5adf2d3.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/images/7e4074c2e099b2d97c4591b910dae8810f7cfa14c10d25bbc57a67ded5adf2d3.jpg)

![9d19123653fd8ff27789c187a5a17716f7afa691e4c5ececa30568cbac94cd93.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/images/9d19123653fd8ff27789c187a5a17716f7afa691e4c5ececa30568cbac94cd93.jpg)

![a847204c1f717acb4e205119a055a9d2437954fdcf1cc8f7637fae88ca69c41e.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/images/a847204c1f717acb4e205119a055a9d2437954fdcf1cc8f7637fae88ca69c41e.jpg)

![b232eae6c6cf9ee96a3916cf763a2adfd0104d44588d2e98363942a0abfc68e9.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/images/b232eae6c6cf9ee96a3916cf763a2adfd0104d44588d2e98363942a0abfc68e9.jpg)

### Tables

![427dd046991cfa34bcd605723400adcd66c4d52228499026ba8c3c6a51861a8f.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/tables/427dd046991cfa34bcd605723400adcd66c4d52228499026ba8c3c6a51861a8f.jpg)

![429234e7aae3049c9a23b64b2314c673f781ad4bd50bf6dada9d402a09164a28.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/tables/429234e7aae3049c9a23b64b2314c673f781ad4bd50bf6dada9d402a09164a28.jpg)

![77e62a8d1025eb97ac7c24712426e6817f4eb592dc09e384e35cbeed3dd3a4e9.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/tables/77e62a8d1025eb97ac7c24712426e6817f4eb592dc09e384e35cbeed3dd3a4e9.jpg)

![f2299413fb4338edf847e1d62d7e352da636bd9fa13c9360c94c931e8cc0ec16.jpg](../icml_results/1127_DECOR_%20Learning%20to%20Decompose%20and%20Collaborate%20in%20Deep%20Search%20via%20Multi-Agent%20Reinforcement%20Learning/tables/f2299413fb4338edf847e1d62d7e352da636bd9fa13c9360c94c931e8cc0ec16.jpg)

<a id="paper-1110"></a>

## Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging


### Images

![15c242c3f324a5dc18a8aedab02aed6431ed14869488a1ca11b71ab9e4b8211c.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/15c242c3f324a5dc18a8aedab02aed6431ed14869488a1ca11b71ab9e4b8211c.jpg)

![6213861dd7475bd13662a171057079161b0bead4dd4b6eadf7a3d22e73cd3249.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/6213861dd7475bd13662a171057079161b0bead4dd4b6eadf7a3d22e73cd3249.jpg)

![89d29e010381b2876718b7a9d6f35192a375ccfecfe26924248f1fae1e5b47d3.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/89d29e010381b2876718b7a9d6f35192a375ccfecfe26924248f1fae1e5b47d3.jpg)

![963d68b709405e1f6e77311372aeda10c96aa595077722ef3a9dc31d3055de36.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/963d68b709405e1f6e77311372aeda10c96aa595077722ef3a9dc31d3055de36.jpg)

![b43fc1e28cdf8c74967004da7175f494b249383af72fe61634dc8bf1fa432f89.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/b43fc1e28cdf8c74967004da7175f494b249383af72fe61634dc8bf1fa432f89.jpg)

![cfcb23cd92513a50015bb1dcae7e1c1b6eefe4e6c9292a2d839403e4693a58d7.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/cfcb23cd92513a50015bb1dcae7e1c1b6eefe4e6c9292a2d839403e4693a58d7.jpg)

![e460b0f1fd93f046f373d02db7709289db1c1ea087e4542d604977de20819316.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/images/e460b0f1fd93f046f373d02db7709289db1c1ea087e4542d604977de20819316.jpg)

### Tables

![2f076b92978c4c1b546d827afc0e343d7cb3526992b047d1affa9d3689e97b83.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/2f076b92978c4c1b546d827afc0e343d7cb3526992b047d1affa9d3689e97b83.jpg)

![3a7c2521d6df7e10bd2a98703c01c9150fc13349bc1433ff339b4dbd7692fc35.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/3a7c2521d6df7e10bd2a98703c01c9150fc13349bc1433ff339b4dbd7692fc35.jpg)

![48f78b5fd8edb2498f57bb1ff3d2f0bc674e51b8777cf122e767650a27fee001.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/48f78b5fd8edb2498f57bb1ff3d2f0bc674e51b8777cf122e767650a27fee001.jpg)

![681590e66c31468659e4cd2cb20aecd1cf0a2a6f09e4190f782256e8a1ace114.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/681590e66c31468659e4cd2cb20aecd1cf0a2a6f09e4190f782256e8a1ace114.jpg)

![69cf148d017f434c2571d0954b206182cb13e0abc54343591fdf350b500b1c6a.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/69cf148d017f434c2571d0954b206182cb13e0abc54343591fdf350b500b1c6a.jpg)

![9a5c83c9470e88a319072c3627367b00b3f012f3bced70032328006f94b94b87.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/9a5c83c9470e88a319072c3627367b00b3f012f3bced70032328006f94b94b87.jpg)

![c120c551a7663c52a996870037e4145c0b6405fda6c42fac2fa2fc6e32ee42f2.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/c120c551a7663c52a996870037e4145c0b6405fda6c42fac2fa2fc6e32ee42f2.jpg)

![c2797179211955ff9b48f456f0a8b5e52d7cdec9c297270489a207bc294f6276.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/c2797179211955ff9b48f456f0a8b5e52d7cdec9c297270489a207bc294f6276.jpg)

![c4e55bf1aea401f4758bb2d316d80f453e63525d3dc8754d548cb6f73f145dd6.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/c4e55bf1aea401f4758bb2d316d80f453e63525d3dc8754d548cb6f73f145dd6.jpg)

![d534161a27c17d6a2e6fac5f81c839437896afcfb0193880594dd9ff79281b7b.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/d534161a27c17d6a2e6fac5f81c839437896afcfb0193880594dd9ff79281b7b.jpg)

![ec28981f4e67b8f2c9f2ee9375393c427497614848fd824ed4f84df20384b214.jpg](../icml_results/1128_Learn%20to%20Merge_%20Meta-Learning%20for%20Adaptive%20Multi-Task%20Model%20Merging/tables/ec28981f4e67b8f2c9f2ee9375393c427497614848fd824ed4f84df20384b214.jpg)

<a id="paper-1111"></a>

## Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions


### Images

![181a6780a996a1cd0de5600d50ca94a5641b3f57c6f62ca0c319fd1bacba6b41.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/181a6780a996a1cd0de5600d50ca94a5641b3f57c6f62ca0c319fd1bacba6b41.jpg)

![23a79ebdf791297996e143b64a8bc27cdc3e450da552c6c30373cb7cd2992144.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/23a79ebdf791297996e143b64a8bc27cdc3e450da552c6c30373cb7cd2992144.jpg)

![30f1d9652e501fabcac25531f906b5b4ca4f2d4d7778eb3dd0df0969aeefc312.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/30f1d9652e501fabcac25531f906b5b4ca4f2d4d7778eb3dd0df0969aeefc312.jpg)

![598c47062cb8887fb260cd5db83cc4cee8b0db3fdba1774435315c7e49bbc8e8.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/598c47062cb8887fb260cd5db83cc4cee8b0db3fdba1774435315c7e49bbc8e8.jpg)

![6608cf41d6459587e92c44fa460c598e76e3f3f9a6c9c1e3028a98adc277f286.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/6608cf41d6459587e92c44fa460c598e76e3f3f9a6c9c1e3028a98adc277f286.jpg)

![6723e08dc39e2965902aa5084a75e8fccf720a88318a16a51ec63fdfdf25b151.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/6723e08dc39e2965902aa5084a75e8fccf720a88318a16a51ec63fdfdf25b151.jpg)

![75fcafe55e4a2129d66e7310b6e1ade05d649f9a45bbdf3ddd8add1de6569c67.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/75fcafe55e4a2129d66e7310b6e1ade05d649f9a45bbdf3ddd8add1de6569c67.jpg)

![7a1f0d6eb321ed8df7c0f090dfe95e47cb2a3b312ddbc8427e2388fea5a4010e.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/7a1f0d6eb321ed8df7c0f090dfe95e47cb2a3b312ddbc8427e2388fea5a4010e.jpg)

![7d192bd3101c4409160f3fa55c96cf2172b3740f3648e579263d745d60a3ea05.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/7d192bd3101c4409160f3fa55c96cf2172b3740f3648e579263d745d60a3ea05.jpg)

![7db7552aee9664ea44bace8eaabf90c727927188267e45a94de5f2ed234020d8.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/7db7552aee9664ea44bace8eaabf90c727927188267e45a94de5f2ed234020d8.jpg)

![9fe8323ffe23fd56127cc942922e60754ca0a3ce27c05cde4747337a3b840ab3.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/9fe8323ffe23fd56127cc942922e60754ca0a3ce27c05cde4747337a3b840ab3.jpg)

![c222622a3d15cd32dfaf55af6ad76117374eeb2903c76b055d82abfe1a993b49.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/c222622a3d15cd32dfaf55af6ad76117374eeb2903c76b055d82abfe1a993b49.jpg)

![c3aeec29719bcdb2edc71a9e7ec1f860a0af500e581ac0b791d02ba1337c0398.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/c3aeec29719bcdb2edc71a9e7ec1f860a0af500e581ac0b791d02ba1337c0398.jpg)

![c85c970b95732ce5554f708a9ee1ae5b926a4a1a036404fe136c5b05b792fb05.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/c85c970b95732ce5554f708a9ee1ae5b926a4a1a036404fe136c5b05b792fb05.jpg)

![cb0790341bf2704c83a72c3c9d1ed87920424b8bd2b95ad91826f001f10e4f27.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/cb0790341bf2704c83a72c3c9d1ed87920424b8bd2b95ad91826f001f10e4f27.jpg)

![d2e4a6f055b74000ecb7cc19a9580edc1bb0952367b068a748f3383fd0acde06.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/d2e4a6f055b74000ecb7cc19a9580edc1bb0952367b068a748f3383fd0acde06.jpg)

![f080d36c4f3e125fbf92e30c315a973bcd33e8760b1ff23c2090085f2d41902b.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/f080d36c4f3e125fbf92e30c315a973bcd33e8760b1ff23c2090085f2d41902b.jpg)

![ff503f27eafd9b73e0cf0261f00d84f3bf74136c42e1cb9c91a2787089f738be.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/images/ff503f27eafd9b73e0cf0261f00d84f3bf74136c42e1cb9c91a2787089f738be.jpg)

### Tables

![0c47cf9ec062bc2d127543f2c086d73d6177168d18a6e94e7ff633e87b77cc50.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/0c47cf9ec062bc2d127543f2c086d73d6177168d18a6e94e7ff633e87b77cc50.jpg)

![0fa70c7b4872f202e2b0c6e0119e36848033ee8eada0426c4a7a58942a8d7511.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/0fa70c7b4872f202e2b0c6e0119e36848033ee8eada0426c4a7a58942a8d7511.jpg)

![128ee9855751ee4be4faa2bf3caa65cf7083809b29e6ca0f313e5d9073ba6a7b.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/128ee9855751ee4be4faa2bf3caa65cf7083809b29e6ca0f313e5d9073ba6a7b.jpg)

![161750f12a4ed4a5d0c3294295ef43ef0777791faf68cb3fef47913ad3fcf456.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/161750f12a4ed4a5d0c3294295ef43ef0777791faf68cb3fef47913ad3fcf456.jpg)

![21f69b4ef5ee21e73966c5453e0187134123680d9d601ccc5e09750b0c4fc95f.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/21f69b4ef5ee21e73966c5453e0187134123680d9d601ccc5e09750b0c4fc95f.jpg)

![24bed985aa183dbec2e5c68b0fc7b332d637ed658c456242ab660fdaeb0fbf99.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/24bed985aa183dbec2e5c68b0fc7b332d637ed658c456242ab660fdaeb0fbf99.jpg)

![26438dd9da9c4885c148101356988ebedb0f8c74acfd0f33a19b52626372ead7.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/26438dd9da9c4885c148101356988ebedb0f8c74acfd0f33a19b52626372ead7.jpg)

![34af7da3c4fa558a310201f8da3a6623e4c0c9fc53d0f4064554f4d84e761979.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/34af7da3c4fa558a310201f8da3a6623e4c0c9fc53d0f4064554f4d84e761979.jpg)

![45b785e277f76463b30dfb5e8b9542b4cb25192396a376f73f958b58cd11df3f.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/45b785e277f76463b30dfb5e8b9542b4cb25192396a376f73f958b58cd11df3f.jpg)

![498a62365a06b54cc76567099767128a3e68280aff92aea350e93f9c3fe908f9.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/498a62365a06b54cc76567099767128a3e68280aff92aea350e93f9c3fe908f9.jpg)

![518d94330bcc3051efeb7f42a269278fe6c1a451e726f59037806853fca64513.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/518d94330bcc3051efeb7f42a269278fe6c1a451e726f59037806853fca64513.jpg)

![575278c57ce89de36a4c4ecbaaf7d490835f347787bfd7f87029d93ff855f48a.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/575278c57ce89de36a4c4ecbaaf7d490835f347787bfd7f87029d93ff855f48a.jpg)

![59aa44013737ca2f17510de4eba8236378d49e4ca825e4ebf8a1f29dbeb017e3.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/59aa44013737ca2f17510de4eba8236378d49e4ca825e4ebf8a1f29dbeb017e3.jpg)

![7735f5a7a4cd3c1420241959e67adb3d1e02c68b84cb8f59454201463a8f9fe4.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/7735f5a7a4cd3c1420241959e67adb3d1e02c68b84cb8f59454201463a8f9fe4.jpg)

![9981c2234bd4304eff539ef951ee8a1421d056603260ac355d739240465fccc1.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/9981c2234bd4304eff539ef951ee8a1421d056603260ac355d739240465fccc1.jpg)

![bfd4b38a70758008aa5e5daf5474f0d875cd9ee0a691542079c49cf0811ac981.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/bfd4b38a70758008aa5e5daf5474f0d875cd9ee0a691542079c49cf0811ac981.jpg)

![c9a2566c92b98e317ca0d90f30b7cf6daaac1cbd38b9b4b48f07eaf0ed863b35.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/c9a2566c92b98e317ca0d90f30b7cf6daaac1cbd38b9b4b48f07eaf0ed863b35.jpg)

![cf44f2bddee2006e8674e80b63c3af534a9aa6d4e776ca5f38e1975104e88e97.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/cf44f2bddee2006e8674e80b63c3af534a9aa6d4e776ca5f38e1975104e88e97.jpg)

![d7e796a9de0349c06547fd5b3b36436dd419ae73f5d9d8fcb5bf5ebd8cbc4aca.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/d7e796a9de0349c06547fd5b3b36436dd419ae73f5d9d8fcb5bf5ebd8cbc4aca.jpg)

![f4999ed12641691da4aed3d4318cb89c41ef258c4a17e9f882eb7a2a60ff2fa2.jpg](../icml_results/1129_Beyond%20Next-Token%20Alignment_%20Distilling%20Multimodal%20Large%20Language%20Models%20via%20Token%20Interactions/tables/f4999ed12641691da4aed3d4318cb89c41ef258c4a17e9f882eb7a2a60ff2fa2.jpg)

<a id="paper-1112"></a>

## Coarse-Grained Boltzmann Generators


### Images

![12a80871c367d72d59e5cc9d05f4168213fdda295350cbec75587882242da35d.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/12a80871c367d72d59e5cc9d05f4168213fdda295350cbec75587882242da35d.jpg)

![2ea0d2024675f9b3b860881ede2ad8248c86ee7cc3e147f30bbdcb2bc80be68b.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/2ea0d2024675f9b3b860881ede2ad8248c86ee7cc3e147f30bbdcb2bc80be68b.jpg)

![3eccc53e28b94617930183b65885eb6ad141d0e6c4ccf7b4e84fe7b2bedba826.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/3eccc53e28b94617930183b65885eb6ad141d0e6c4ccf7b4e84fe7b2bedba826.jpg)

![3f99ba5f31e62192e20dd91908d88ce0bc06f9bb9972fc6f03f1d231301061c8.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/3f99ba5f31e62192e20dd91908d88ce0bc06f9bb9972fc6f03f1d231301061c8.jpg)

![6a9d439b47fda60190555d16b64344b478b7fd5d47ac83901d087831956d3880.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/6a9d439b47fda60190555d16b64344b478b7fd5d47ac83901d087831956d3880.jpg)

![6cc5ebb80b3f376ae5c199e31e2be495181170625c08e6f7397509982cac4400.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/6cc5ebb80b3f376ae5c199e31e2be495181170625c08e6f7397509982cac4400.jpg)

![8f664ae598fb1aa5c3dd4fb57d05efa99d2c17cc8d6f617b436c400dae48827b.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/8f664ae598fb1aa5c3dd4fb57d05efa99d2c17cc8d6f617b436c400dae48827b.jpg)

![a59205ced43ae5ec61a4b398657b7d34b584685acf51ab3d8000e2656cc0042a.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/a59205ced43ae5ec61a4b398657b7d34b584685acf51ab3d8000e2656cc0042a.jpg)

![acbb40ee6dae55056a94eabb816dab74f08e76659024cc51b833ee642459a5a6.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/acbb40ee6dae55056a94eabb816dab74f08e76659024cc51b833ee642459a5a6.jpg)

![aedb59a039e591a1d890cec4340c64689ad25a015794286b8cc36f24454e6e67.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/aedb59a039e591a1d890cec4340c64689ad25a015794286b8cc36f24454e6e67.jpg)

![af0a04fdba9f5ffde2e84592d38a043555d9ac103d872e1292342990d9305da1.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/af0a04fdba9f5ffde2e84592d38a043555d9ac103d872e1292342990d9305da1.jpg)

![ebc92470cbf01996dea691c28f866a445000dd5d92057eaf7d3d345e5a8abc90.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/images/ebc92470cbf01996dea691c28f866a445000dd5d92057eaf7d3d345e5a8abc90.jpg)

### Tables

![08f6108a2e84d11454cbb2e83104874330fde52305c684093303d617a5c87e1b.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/08f6108a2e84d11454cbb2e83104874330fde52305c684093303d617a5c87e1b.jpg)

![0a6518e127df174afc96f489440d6d96f168ec1d0d04c5552de5404058439c69.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/0a6518e127df174afc96f489440d6d96f168ec1d0d04c5552de5404058439c69.jpg)

![0ddbe836d598099e3604537a8793f2482b2627666682df66bc3e666840b996b1.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/0ddbe836d598099e3604537a8793f2482b2627666682df66bc3e666840b996b1.jpg)

![184620b0f6798c84ecb5fae7dc089a373e1d7cae01bb1b41968ac7bc2769a51a.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/184620b0f6798c84ecb5fae7dc089a373e1d7cae01bb1b41968ac7bc2769a51a.jpg)

![3f063f2e4eceef8e41509696c2f80923ecf3d61717f90d75857ec8298306d17d.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/3f063f2e4eceef8e41509696c2f80923ecf3d61717f90d75857ec8298306d17d.jpg)

![423658dfe4df6bc847dc921991b8d66158ffe8e6b2aa14e99a81a3a4e139c82e.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/423658dfe4df6bc847dc921991b8d66158ffe8e6b2aa14e99a81a3a4e139c82e.jpg)

![47db8ce34cec8c45d2fc6e492a472b56346015375c376cf5cb0fb82f1772a6d4.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/47db8ce34cec8c45d2fc6e492a472b56346015375c376cf5cb0fb82f1772a6d4.jpg)

![522ea9792ba3f2b7027eb5c48143ea7e0d392d818bba790d27cfd5887664f806.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/522ea9792ba3f2b7027eb5c48143ea7e0d392d818bba790d27cfd5887664f806.jpg)

![87c53cb8024bd2d3fb98294823d14262299855e2dff06d3136acbeeb8915548c.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/87c53cb8024bd2d3fb98294823d14262299855e2dff06d3136acbeeb8915548c.jpg)

![c27e8824293d208db78e5c96a1cfd49638d90f99b59598d2db3f4ca685051f87.jpg](../icml_results/1130_Coarse-Grained%20Boltzmann%20Generators/tables/c27e8824293d208db78e5c96a1cfd49638d90f99b59598d2db3f4ca685051f87.jpg)

<a id="paper-1113"></a>

## GraphP-FL: Personalized Federated Graph Learning via Dynamic Structure Awareness and Fisher Information Elastic Alignment


### Images

![1737d985e9904f61f661bbcda7e006f469766573f598ca25590fc13263623645.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/images/1737d985e9904f61f661bbcda7e006f469766573f598ca25590fc13263623645.jpg)

![91823997fec7704b8f7d5139b4b590ae8e19544e0cf54444c40c92381dee58c1.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/images/91823997fec7704b8f7d5139b4b590ae8e19544e0cf54444c40c92381dee58c1.jpg)

![ddc439d21bf99a3bdc6bd8f64164ed69084566079a9e8769400b0b0ef6cc54b0.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/images/ddc439d21bf99a3bdc6bd8f64164ed69084566079a9e8769400b0b0ef6cc54b0.jpg)

![e93b047a446e74da8b9c5a9c85e11f732e75ecd2ed413d58187752fb5d948f69.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/images/e93b047a446e74da8b9c5a9c85e11f732e75ecd2ed413d58187752fb5d948f69.jpg)

![f3045e73d3ced13eaea896a61815fbfebff72df3d464fdbc010ad46244d10330.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/images/f3045e73d3ced13eaea896a61815fbfebff72df3d464fdbc010ad46244d10330.jpg)

### Tables

![004c3b14bc80a5bb5ccdc290d465f5d632f9b9676f87c7e4b2ee7e294d49f7c1.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/004c3b14bc80a5bb5ccdc290d465f5d632f9b9676f87c7e4b2ee7e294d49f7c1.jpg)

![4b4a1ca9461608813edba86db06afe4ede0a932ac9a8966080fadca039fd9b37.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/4b4a1ca9461608813edba86db06afe4ede0a932ac9a8966080fadca039fd9b37.jpg)

![4bc0669f02d5a15ff3ec1a7eb0d2d119fd4e7dfcdc5ea891c9e742d9830929ef.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/4bc0669f02d5a15ff3ec1a7eb0d2d119fd4e7dfcdc5ea891c9e742d9830929ef.jpg)

![72193172821627484ef3235780cbbb5b2a1e50e034163e5852358c44c5054ad1.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/72193172821627484ef3235780cbbb5b2a1e50e034163e5852358c44c5054ad1.jpg)

![a03858d82fa093efb56af26d91524f5df815057ff66cb1cfca72c9e9e2d9aca9.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/a03858d82fa093efb56af26d91524f5df815057ff66cb1cfca72c9e9e2d9aca9.jpg)

![bf1feb22753dfa8e2e31ad15c79f8eaffca7f753926b532ffc0fc032a728c081.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/bf1feb22753dfa8e2e31ad15c79f8eaffca7f753926b532ffc0fc032a728c081.jpg)

![cb4803800dbf0a7d3570c57f9c24565a753739f33ade4fd3f95546cc83b25b66.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/cb4803800dbf0a7d3570c57f9c24565a753739f33ade4fd3f95546cc83b25b66.jpg)

![d2d8f2c01459dd2af29a3f7c190037a9961b77451bed6450b795509389645258.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/d2d8f2c01459dd2af29a3f7c190037a9961b77451bed6450b795509389645258.jpg)

![d906a0c93c21ae1e9e3cf2f1ebb823105a39671e335a600fc14b45fd03993a76.jpg](../icml_results/1131_GraphP-FL_%20Personalized%20Federated%20Graph%20Learning%20via%20Dynamic%20Structure%20Awareness%20and%20Fisher%20Informat/tables/d906a0c93c21ae1e9e3cf2f1ebb823105a39671e335a600fc14b45fd03993a76.jpg)

<a id="paper-1114"></a>

## Token-Sparse Medical Multimodal Reasoning via Dual-Stream Reinforcement Learning


### Images

![02039cf8860dbd2c66f6ce9d2ed44d503aa493ec0e0a5ea2b05581e939ddd56e.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/02039cf8860dbd2c66f6ce9d2ed44d503aa493ec0e0a5ea2b05581e939ddd56e.jpg)

![1d3f0a4d68e974e9f5c12b354deed2501df7f05aef30b615f445bd4b8564cda4.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/1d3f0a4d68e974e9f5c12b354deed2501df7f05aef30b615f445bd4b8564cda4.jpg)

![291aae8c06a38693b1444c1e22f08ae5198aaf9971e54d1b7552d8b32d0f1a7e.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/291aae8c06a38693b1444c1e22f08ae5198aaf9971e54d1b7552d8b32d0f1a7e.jpg)

![3142d2831a5d868793d2cc71ea223754b66376969365592a6fec2f8bc0b6eb3d.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/3142d2831a5d868793d2cc71ea223754b66376969365592a6fec2f8bc0b6eb3d.jpg)

![44d444a0feadc4772f174e0887d036a4e883548524c7387af0cc6efba7f6028d.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/44d444a0feadc4772f174e0887d036a4e883548524c7387af0cc6efba7f6028d.jpg)

![4af8863a02715802cba5379bac53b9e4ec0623b80b06c695b16491af539eb952.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/4af8863a02715802cba5379bac53b9e4ec0623b80b06c695b16491af539eb952.jpg)

![51d718b11c47ea3fd8163918b9473f6b1ccb95bcc3e4df9f96050df6663e8503.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/51d718b11c47ea3fd8163918b9473f6b1ccb95bcc3e4df9f96050df6663e8503.jpg)

![6226e4813ffd6f44562807f3bafe0c35b83be6dce1a341762d89e444ebde7db0.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/6226e4813ffd6f44562807f3bafe0c35b83be6dce1a341762d89e444ebde7db0.jpg)

![7cd42dd7ae7d743ca29da7f3675b38656931dc6f5cdfe2f07f0ac8e36de28915.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/7cd42dd7ae7d743ca29da7f3675b38656931dc6f5cdfe2f07f0ac8e36de28915.jpg)

![7f58e271ee8285538cf22838ff01882c1799015a7918ff246e8fa7be431c6f7f.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/7f58e271ee8285538cf22838ff01882c1799015a7918ff246e8fa7be431c6f7f.jpg)

![8942d568eb705bae6882d3b8f88bf8b846bc90b2586647bde020f128266b1357.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/8942d568eb705bae6882d3b8f88bf8b846bc90b2586647bde020f128266b1357.jpg)

![89762be6f7b7f4f2c2f1cbf6c2c1d03b995c5b4442b2bd99f58e41b8089751aa.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/89762be6f7b7f4f2c2f1cbf6c2c1d03b995c5b4442b2bd99f58e41b8089751aa.jpg)

![8c5983d266e7880a330df3bb30773e320403bf48bbaf612b2888e6f71b3d944d.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/8c5983d266e7880a330df3bb30773e320403bf48bbaf612b2888e6f71b3d944d.jpg)

![8c5ce8976e455824b6afb4dee03cbc7fa1e56e91a297fbc69f6bf640da81ea20.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/8c5ce8976e455824b6afb4dee03cbc7fa1e56e91a297fbc69f6bf640da81ea20.jpg)

![af086391e1c88717acf162e06c89eb483fef28701ddd3c238198f708f7222627.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/af086391e1c88717acf162e06c89eb483fef28701ddd3c238198f708f7222627.jpg)

![af8de38010507cab87cb79996f2552498ccc7afb3b85f91a58445625a8145cca.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/af8de38010507cab87cb79996f2552498ccc7afb3b85f91a58445625a8145cca.jpg)

![b339e3781a836f99746edf132dbbffebd37332094ab15730ea9301e445522c14.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/b339e3781a836f99746edf132dbbffebd37332094ab15730ea9301e445522c14.jpg)

![b7aa2ccca99d728af1cdbf2b564941f7d2d8590ab7876529ebfcab20aed415de.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/b7aa2ccca99d728af1cdbf2b564941f7d2d8590ab7876529ebfcab20aed415de.jpg)

![e3f06a3c6c2426cffead5b983453f784a55c74cfb60b30477415e3e12aeeaeac.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/e3f06a3c6c2426cffead5b983453f784a55c74cfb60b30477415e3e12aeeaeac.jpg)

![fd959e67c30c97763a8d2280a91f2be124ef38c51a9c5dce19722799ebb37ec0.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/images/fd959e67c30c97763a8d2280a91f2be124ef38c51a9c5dce19722799ebb37ec0.jpg)

### Tables

![1b3f730fded8f7deffa3ad28c4bbb2c5325eb6fe9b93244bcfb9dff12d264481.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/1b3f730fded8f7deffa3ad28c4bbb2c5325eb6fe9b93244bcfb9dff12d264481.jpg)

![4bcb394c1bd39e76ee21203abd41e01f8cd7dac426e913d6c107db4da28843d8.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/4bcb394c1bd39e76ee21203abd41e01f8cd7dac426e913d6c107db4da28843d8.jpg)

![50f4f54b7e544eb7478f87d757918ac73664ad186811b06b77752dcfe460a570.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/50f4f54b7e544eb7478f87d757918ac73664ad186811b06b77752dcfe460a570.jpg)

![9f7b2f7a58ff6c305f0c5dba53c64ad589f848640270ed2953d60233460b2d8e.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/9f7b2f7a58ff6c305f0c5dba53c64ad589f848640270ed2953d60233460b2d8e.jpg)

![b523c50d2c0b9a673788109897bd43c060b6955a1cfb9decfd6bd50b51fb253c.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/b523c50d2c0b9a673788109897bd43c060b6955a1cfb9decfd6bd50b51fb253c.jpg)

![f231e962fd38a9004de1cddbacfc332bb2870c6c5a56e7818bb289bf5c3bc53a.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/f231e962fd38a9004de1cddbacfc332bb2870c6c5a56e7818bb289bf5c3bc53a.jpg)

![ff4def1979c72e20dc4319cd8379d9a21ad0df543d6f7d1b7d18697c83897b3e.jpg](../icml_results/1132_Token-Sparse%20Medical%20Multimodal%20Reasoning%20via%20Dual-Stream%20Reinforcement%20Learning/tables/ff4def1979c72e20dc4319cd8379d9a21ad0df543d6f7d1b7d18697c83897b3e.jpg)

<a id="paper-1115"></a>

## Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets


### Images

![053688463ad0f59decfc7b22e270896573b9c58492713ff593f83b1b6ceee738.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/images/053688463ad0f59decfc7b22e270896573b9c58492713ff593f83b1b6ceee738.jpg)

![1882672d055cf294bb6502dd639227eb8203cf4c6df90b699cddb9164c4fec21.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/images/1882672d055cf294bb6502dd639227eb8203cf4c6df90b699cddb9164c4fec21.jpg)

![3ed4f8555164ecde798a2b8e05f6864dbe7da99dd9fd7768202fad7d588951b5.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/images/3ed4f8555164ecde798a2b8e05f6864dbe7da99dd9fd7768202fad7d588951b5.jpg)

![61f2ddc1b582ac55a0b3f8ecd7bcb12de0b957d74bdae12172e8c82332860386.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/images/61f2ddc1b582ac55a0b3f8ecd7bcb12de0b957d74bdae12172e8c82332860386.jpg)

![a74e01ea87fea0fe1d07f30960b0a66a915098f0b90b358f56348ccdb7d99b61.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/images/a74e01ea87fea0fe1d07f30960b0a66a915098f0b90b358f56348ccdb7d99b61.jpg)

![d53bc8377326ea9de362e4f2f221d3e436f0cfeae893d9bcf296f11b433d3946.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/images/d53bc8377326ea9de362e4f2f221d3e436f0cfeae893d9bcf296f11b433d3946.jpg)

### Tables

![9693d71a846f31637e214328c4d02499b97a5d574ed6cab629d98f3e991722d1.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/tables/9693d71a846f31637e214328c4d02499b97a5d574ed6cab629d98f3e991722d1.jpg)

![a9511d08de19d35fe4c607f6d698c03d967a8324d6a8963823c87d18e36a7682.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/tables/a9511d08de19d35fe4c607f6d698c03d967a8324d6a8963823c87d18e36a7682.jpg)

![ab0545f69cb8140a21e19e1984c76f55f173dc86eb33f4bd78d54c76a303a500.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/tables/ab0545f69cb8140a21e19e1984c76f55f173dc86eb33f4bd78d54c76a303a500.jpg)

![b27e51209ca7793bd5d3bfbf625c207f62131e1155249821b84b48f688c21955.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/tables/b27e51209ca7793bd5d3bfbf625c207f62131e1155249821b84b48f688c21955.jpg)

![e3d5dd7edd3e5b0455c92418594a1cfb2aecfde0a3e06bb18a32d06bc8807a08.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/tables/e3d5dd7edd3e5b0455c92418594a1cfb2aecfde0a3e06bb18a32d06bc8807a08.jpg)

![e878b70e9c5a0648514099ea1a1f49ce282e05de0cf9fe25588f55efc75da86a.jpg](../icml_results/1133_Efficient%2C%20Validation-Free%20Intrinsic%20Quality%20Estimation%20for%20Large-Scale%20Face%20Recognition%20Datasets/tables/e878b70e9c5a0648514099ea1a1f49ce282e05de0cf9fe25588f55efc75da86a.jpg)

<a id="paper-1116"></a>

## ST-TGExplainer: Disentangling Stability and Transition Patterns for Temporal GNN Interpretability


### Images

![2bd502e88ce6834ac50c7f0d60926aedc11e95a508dd300f4c1030054cc751cb.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/2bd502e88ce6834ac50c7f0d60926aedc11e95a508dd300f4c1030054cc751cb.jpg)

![4a8c545746743205df878a90071cdd7c4ea26ec1d2a25edfa435ee4a774e840b.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/4a8c545746743205df878a90071cdd7c4ea26ec1d2a25edfa435ee4a774e840b.jpg)

![4dfa1a5e458492fb975b34940601c3650d3c38e4db14a6528cd935b852864d8f.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/4dfa1a5e458492fb975b34940601c3650d3c38e4db14a6528cd935b852864d8f.jpg)

![605aa342fa0d0e646a354aa8e22c85311f17860c26e9a1cea5773e45331f9d67.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/605aa342fa0d0e646a354aa8e22c85311f17860c26e9a1cea5773e45331f9d67.jpg)

![8178eee2bd0c11a9f01a8ae31b56cf8636bf895ba553a6c28b96ca438f2e06b7.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/8178eee2bd0c11a9f01a8ae31b56cf8636bf895ba553a6c28b96ca438f2e06b7.jpg)

![87530c3ea62c0690c1d8978f9ae3cdd89e064f556284fef2d661cd0415f8b0ce.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/87530c3ea62c0690c1d8978f9ae3cdd89e064f556284fef2d661cd0415f8b0ce.jpg)

![8807d8fb4a47c59b5623909ff3de5f75379f4ff2b1807708b78f8cb35fe70a74.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/8807d8fb4a47c59b5623909ff3de5f75379f4ff2b1807708b78f8cb35fe70a74.jpg)

![a9478bee10a052ea54156720ad6876ddae5d4c5046eaeb300d1d620143c8f2fd.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/a9478bee10a052ea54156720ad6876ddae5d4c5046eaeb300d1d620143c8f2fd.jpg)

![b5ab31c83c08548c78f62d44a962de9ba02b85c8e53b4938ca8e9b2c0e580cb2.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/b5ab31c83c08548c78f62d44a962de9ba02b85c8e53b4938ca8e9b2c0e580cb2.jpg)

![caeb21b4030987c9ceb61281871662a6fe46cd32f7af0c1a1e3ae87bcc607330.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/images/caeb21b4030987c9ceb61281871662a6fe46cd32f7af0c1a1e3ae87bcc607330.jpg)

### Tables

![04e99ddd22f9ab1dfaa94900a83e026797ccec0c37844af082e89eecd6254ae4.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/04e99ddd22f9ab1dfaa94900a83e026797ccec0c37844af082e89eecd6254ae4.jpg)

![4458b7f62572250697657de04e9e62831aad85a29f42e1df8cf2ae80c92b4cbb.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/4458b7f62572250697657de04e9e62831aad85a29f42e1df8cf2ae80c92b4cbb.jpg)

![7d6ebc0bd33393aee713482873c7ee883a8c5a6baac6e9cf8e838410faf0e9cc.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/7d6ebc0bd33393aee713482873c7ee883a8c5a6baac6e9cf8e838410faf0e9cc.jpg)

![b03b8e2638f47bbe024b5cfd4e0f7037cb4dc5622ad8c350b02760c97759cd12.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/b03b8e2638f47bbe024b5cfd4e0f7037cb4dc5622ad8c350b02760c97759cd12.jpg)

![c6cc8ba24c8135195930c0faa7dc7a123e1bd9240f4d66e89c18b6f4a70fea95.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/c6cc8ba24c8135195930c0faa7dc7a123e1bd9240f4d66e89c18b6f4a70fea95.jpg)

![e5000052ad7f964861280b4552e099e3ac6653006c3d7a5a11b223a3c127de25.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/e5000052ad7f964861280b4552e099e3ac6653006c3d7a5a11b223a3c127de25.jpg)

![ef02093f2ddcbf70b89373e98cc1f8d79eaeba2e086ff3ddc5ce5eec8e8386fd.jpg](../icml_results/1134_ST-TGExplainer_%20Disentangling%20Stability%20and%20Transition%20Patterns%20for%20Temporal%20GNN%20Interpretability/tables/ef02093f2ddcbf70b89373e98cc1f8d79eaeba2e086ff3ddc5ce5eec8e8386fd.jpg)

<a id="paper-1117"></a>

## ImgCoT: Compressing Long Chain of Thought into Compact Visual Tokens for Efficient Reasoning of Large Language Model


### Images

![053238da28e40b155954b4298d0f44355deedcf8592759b9439674d5bda46d9f.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/images/053238da28e40b155954b4298d0f44355deedcf8592759b9439674d5bda46d9f.jpg)

![10b6b83090240bf2ddee4487f2d8140be9ed3f1bde189947b9e05eaf6e4105ae.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/images/10b6b83090240bf2ddee4487f2d8140be9ed3f1bde189947b9e05eaf6e4105ae.jpg)

![698539d457a696e745c5aba36cc90e0e7547fa811e378c8325f83147dd73c115.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/images/698539d457a696e745c5aba36cc90e0e7547fa811e378c8325f83147dd73c115.jpg)

![732a7097399c6a673aa93bc735c65ba8aeb3d075f164208f1681535bec55a281.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/images/732a7097399c6a673aa93bc735c65ba8aeb3d075f164208f1681535bec55a281.jpg)

![ca7ba09b3d1ee49b0523d4c8dca645b50dc8ffa0b16d805e66b5b535971120c4.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/images/ca7ba09b3d1ee49b0523d4c8dca645b50dc8ffa0b16d805e66b5b535971120c4.jpg)

![edfd4af4dab20c0bb49ad6c32b7db4416da92eadd04612d2b3730bc3031afdbd.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/images/edfd4af4dab20c0bb49ad6c32b7db4416da92eadd04612d2b3730bc3031afdbd.jpg)

### Tables

![09ab113f1f2e662ef5bdf329d215d82adfde34383e8d0d4d0f7554a4329b128a.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/09ab113f1f2e662ef5bdf329d215d82adfde34383e8d0d4d0f7554a4329b128a.jpg)

![0acc917ba0d7f4ab42bd12da6f976e09db0a8ffccde37c24d658dc1261e9171b.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/0acc917ba0d7f4ab42bd12da6f976e09db0a8ffccde37c24d658dc1261e9171b.jpg)

![0fee16893660b2a4a0c6c1b912d7309490164d675428da0a95bae9ce7af1a487.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/0fee16893660b2a4a0c6c1b912d7309490164d675428da0a95bae9ce7af1a487.jpg)

![13115a5a23d5568bf40ce536f898064009ceaec3e2736d5cb8ea8e118cf7f441.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/13115a5a23d5568bf40ce536f898064009ceaec3e2736d5cb8ea8e118cf7f441.jpg)

![1341821fc3f3a0041c241744a040433a1b01a351cc3a610cd016e1aa7123fbc4.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/1341821fc3f3a0041c241744a040433a1b01a351cc3a610cd016e1aa7123fbc4.jpg)

![2528712ad759e2ab3dcce1102176749c45fc4618771a5de22098a352b0d27845.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/2528712ad759e2ab3dcce1102176749c45fc4618771a5de22098a352b0d27845.jpg)

![436108ab4823b039ea6753dcb92de110088aa6dd89807e1fe91746ffe7d7e18f.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/436108ab4823b039ea6753dcb92de110088aa6dd89807e1fe91746ffe7d7e18f.jpg)

![5f895c9851437c6a7a91ffb090bb6b1930d0a598653b9c4163a88d6559866ccf.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/5f895c9851437c6a7a91ffb090bb6b1930d0a598653b9c4163a88d6559866ccf.jpg)

![baafae25cab9a68529e4ead978943953c08d6551ac8a737b46eb07157b0dcf6e.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/baafae25cab9a68529e4ead978943953c08d6551ac8a737b46eb07157b0dcf6e.jpg)

![caae3867dab655d3c245d03728dad2b1fcf3aaed69f70070177c6ee142ecf305.jpg](../icml_results/1135_ImgCoT_%20Compressing%20Long%20Chain%20of%20Thought%20into%20Compact%20Visual%20Tokens%20for%20Efficient%20Reasoning%20of%20Larg/tables/caae3867dab655d3c245d03728dad2b1fcf3aaed69f70070177c6ee142ecf305.jpg)

<a id="paper-1118"></a>

## HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing


### Images

![0345dbf4ac7fb15df0188d6aea13005e380cb33bf69de498a2df82b130380436.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/0345dbf4ac7fb15df0188d6aea13005e380cb33bf69de498a2df82b130380436.jpg)

![04bc5a23a59134ddc7ab8e51e3be583a39d6bd080f6afb45b9f6a67371689df2.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/04bc5a23a59134ddc7ab8e51e3be583a39d6bd080f6afb45b9f6a67371689df2.jpg)

![0c08062e1dc56a8a720d4a1711736c04c7b2467e08d545b6bc91c510cfdc41fd.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/0c08062e1dc56a8a720d4a1711736c04c7b2467e08d545b6bc91c510cfdc41fd.jpg)

![1fb57f3744d8db345833c94a911da94ecb1205e6f1362eb24d057857031dd5b4.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/1fb57f3744d8db345833c94a911da94ecb1205e6f1362eb24d057857031dd5b4.jpg)

![204d86f62d3e089e3d138fc2574d47be863097b623e82d4423d277f740f4aee1.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/204d86f62d3e089e3d138fc2574d47be863097b623e82d4423d277f740f4aee1.jpg)

![22d7868eae0ae55fb45cd04802455694af3ced18809fc040cec79de3486944f9.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/22d7868eae0ae55fb45cd04802455694af3ced18809fc040cec79de3486944f9.jpg)

![23beac0119ca14fb2730ad937f3a7585b1601067e9c1114486116883b16b02f6.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/23beac0119ca14fb2730ad937f3a7585b1601067e9c1114486116883b16b02f6.jpg)

![42f3ff426643c1d42664be54f54b3f3cb3935eaf39e0a0e6f225d58b16728ecb.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/42f3ff426643c1d42664be54f54b3f3cb3935eaf39e0a0e6f225d58b16728ecb.jpg)

![47dabfe5b37067d6318a22d950de1ae0a2080173d57880719a041160cb73cc7e.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/47dabfe5b37067d6318a22d950de1ae0a2080173d57880719a041160cb73cc7e.jpg)

![55fdc9d860fe804ebcb8ef98466e7a4065f0a2c28644d052f3968d22d5e3230a.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/55fdc9d860fe804ebcb8ef98466e7a4065f0a2c28644d052f3968d22d5e3230a.jpg)

![56493b34c2b8980ba74b58e1eee06cf60eb4ace86bfa4404854e27311e2247a1.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/56493b34c2b8980ba74b58e1eee06cf60eb4ace86bfa4404854e27311e2247a1.jpg)

![5ffafc85dabba07bd6d719fdacbb0bbe66a87a5d71ed79a32e99a6cbf315d6df.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/5ffafc85dabba07bd6d719fdacbb0bbe66a87a5d71ed79a32e99a6cbf315d6df.jpg)

![6b71160923ac8e4bf689c2db21a34ae875100dee68bbfd256f0a685fcfefed2a.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/6b71160923ac8e4bf689c2db21a34ae875100dee68bbfd256f0a685fcfefed2a.jpg)

![7641be6475e48a87100fdcc6105f5d253fab3645fb9cb54e918fdcdc653c0b4e.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/7641be6475e48a87100fdcc6105f5d253fab3645fb9cb54e918fdcdc653c0b4e.jpg)

![a5ff9cf7db3328d26425679234cf1fb4516894111aef597d6d94860a2ab3bd22.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/a5ff9cf7db3328d26425679234cf1fb4516894111aef597d6d94860a2ab3bd22.jpg)

![ad5e1076b6d515293698ae79c9fd5afca8dd619bb3b1f0a96aad34eaf999dae1.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/ad5e1076b6d515293698ae79c9fd5afca8dd619bb3b1f0a96aad34eaf999dae1.jpg)

![b7e327b0f5fb9297a8aa199f1167cc08e08969f9f8ee3ff527c1a3f6b463666d.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/b7e327b0f5fb9297a8aa199f1167cc08e08969f9f8ee3ff527c1a3f6b463666d.jpg)

![c1686f94af5f7ff00034157cb22fa15082f76ee292e50c7f131ba35f4ca98e0e.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/c1686f94af5f7ff00034157cb22fa15082f76ee292e50c7f131ba35f4ca98e0e.jpg)

![cde78125ab436fd75fd18a8ad7a98c30c4dc7887d08a340419af2116891013cd.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/cde78125ab436fd75fd18a8ad7a98c30c4dc7887d08a340419af2116891013cd.jpg)

![ede2a339aa31467a8eada8fa899f255b92d3d5ebb182dc5f52a3a2953192d6b2.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/ede2a339aa31467a8eada8fa899f255b92d3d5ebb182dc5f52a3a2953192d6b2.jpg)

![f91362d3d5fd669a5eda538e69f961e5d0f29efe25b682e769825f4d8acd4797.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/f91362d3d5fd669a5eda538e69f961e5d0f29efe25b682e769825f4d8acd4797.jpg)

![fd9f95605ebfccc1feea9cba6c2a7fd077066fb4b85ad33efe75747baeef798e.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/images/fd9f95605ebfccc1feea9cba6c2a7fd077066fb4b85ad33efe75747baeef798e.jpg)

### Tables

![0cb6971445c6673e6e30263ea8901e3a247f5f0e9996b0c8ac35405904590121.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/0cb6971445c6673e6e30263ea8901e3a247f5f0e9996b0c8ac35405904590121.jpg)

![26c7186bdea595cfccc1c601dd6f62fc976b7d214d5a6407fffba55a9ac2929e.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/26c7186bdea595cfccc1c601dd6f62fc976b7d214d5a6407fffba55a9ac2929e.jpg)

![29fc73bfb5137a5681e3b1fa2b526c149d943deb8a26f8bc4a787976e1459f49.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/29fc73bfb5137a5681e3b1fa2b526c149d943deb8a26f8bc4a787976e1459f49.jpg)

![2ed5d9d77d6f310703b0861e5f316d7d92ddb26aea45b5c827fa7bcf959a55b3.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/2ed5d9d77d6f310703b0861e5f316d7d92ddb26aea45b5c827fa7bcf959a55b3.jpg)

![3564173d36089d478ff8fa234135c7e962d3097f67132328ca9e19ed02c532e5.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/3564173d36089d478ff8fa234135c7e962d3097f67132328ca9e19ed02c532e5.jpg)

![45362d8d2eea383598f6af0b6575b9b6329a9441f2c54bf3e64435569046dc2c.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/45362d8d2eea383598f6af0b6575b9b6329a9441f2c54bf3e64435569046dc2c.jpg)

![4f85d45abaedf9e9def042d5c8b7ea095b9be9f5662eb66423c31b50b1baf9a4.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/4f85d45abaedf9e9def042d5c8b7ea095b9be9f5662eb66423c31b50b1baf9a4.jpg)

![85959d508063ddb5000c68eea943d3debef2453723b643fcfa326d214bbc0e74.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/85959d508063ddb5000c68eea943d3debef2453723b643fcfa326d214bbc0e74.jpg)

![8aa3cab60c29592ae7bfa10de05d1b2b005792b072d3c2281d4c0169fada6c9b.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/8aa3cab60c29592ae7bfa10de05d1b2b005792b072d3c2281d4c0169fada6c9b.jpg)

![9b13c63b3ea590e93a2513385f57c0bcf379a30100e4df198bb8ca622a2fee8f.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/9b13c63b3ea590e93a2513385f57c0bcf379a30100e4df198bb8ca622a2fee8f.jpg)

![9c9295909f47949c93b08706405c3b72fe730c4215cb21d74789b41789f202b5.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/9c9295909f47949c93b08706405c3b72fe730c4215cb21d74789b41789f202b5.jpg)

![a48fff68514400e6d29db90c33aaa5907506b1bbc68e29a5f626ec7c33936cac.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/a48fff68514400e6d29db90c33aaa5907506b1bbc68e29a5f626ec7c33936cac.jpg)

![bf9c8d0f7ef49034ef9db5de0c339da898ccae9c1b2c69b6b61de36bab2e6c9f.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/bf9c8d0f7ef49034ef9db5de0c339da898ccae9c1b2c69b6b61de36bab2e6c9f.jpg)

![c87821b54015c11e1b4dff48c21b159f10f629656d25f2c6dfa4c6233ce2f0ca.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/c87821b54015c11e1b4dff48c21b159f10f629656d25f2c6dfa4c6233ce2f0ca.jpg)

![ca6d4f96bd5a7c38a690c73f475474db9ba93a5584573ea51a688ef332df90e7.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/ca6d4f96bd5a7c38a690c73f475474db9ba93a5584573ea51a688ef332df90e7.jpg)

![e74c7d4b4211606d3a1984b27e14dbd3414afad98193b2df50f240fd2a284f5c.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/e74c7d4b4211606d3a1984b27e14dbd3414afad98193b2df50f240fd2a284f5c.jpg)

![f9d6b3a9dc7a5aa1904048e478f02935c69ca9e9adf09a4691d1a4f44c3e3cad.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/f9d6b3a9dc7a5aa1904048e478f02935c69ca9e9adf09a4691d1a4f44c3e3cad.jpg)

![fbdf5cb2080a91d65dc467cdc4a5386831c9464853c2cf5d3ef7ce6358087187.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/fbdf5cb2080a91d65dc467cdc4a5386831c9464853c2cf5d3ef7ce6358087187.jpg)

![fd903bede3e55641398374a4de4df5b8cd6e6080cf05854f0f29b72026348e06.jpg](../icml_results/1136_HoloFair_%20Unified%20T2I%20Fairness%20Evaluation%20and%20Fair-GRPO%20Debiasing/tables/fd903bede3e55641398374a4de4df5b8cd6e6080cf05854f0f29b72026348e06.jpg)

<a id="paper-1119"></a>

## RBCBF: Decoding Time Safety Alignment via Risk Guided Rollback and Barrier Control


### Images

![04bd8ee5dd6674e0730da92c52897d0fd02b145b521116011325db3d18d4fec7.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/04bd8ee5dd6674e0730da92c52897d0fd02b145b521116011325db3d18d4fec7.jpg)

![0967252c943ba491669982302eb2176f9ef7455b79010eccd50cc9e56f1374fe.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/0967252c943ba491669982302eb2176f9ef7455b79010eccd50cc9e56f1374fe.jpg)

![437bd62123d2d282f808e0b5228e7e1fde1ee1f91ccf2e1fc44528f6b3ce11ea.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/437bd62123d2d282f808e0b5228e7e1fde1ee1f91ccf2e1fc44528f6b3ce11ea.jpg)

![45c7612d89fc1c865c6e25c9b5137ae73f57c6c25a51ae0a9e1d9df977b0bb6e.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/45c7612d89fc1c865c6e25c9b5137ae73f57c6c25a51ae0a9e1d9df977b0bb6e.jpg)

![48e10573b81c9cc4bfbc74bd1e955c9ee270c9e6dd210d87c9a41964dee89b1b.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/48e10573b81c9cc4bfbc74bd1e955c9ee270c9e6dd210d87c9a41964dee89b1b.jpg)

![6d4931c569474b58ffa472a33e8185ecd19e1d2b390df1942c787fb4124bceb4.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/6d4931c569474b58ffa472a33e8185ecd19e1d2b390df1942c787fb4124bceb4.jpg)

![9e0ae8f7a992cf6e995e2ede02d9d81ced7335c9d7e7e651a2403701dc656df8.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/9e0ae8f7a992cf6e995e2ede02d9d81ced7335c9d7e7e651a2403701dc656df8.jpg)

![de0b80f1a35156dbe6ed5d089d75d1e54372d20061d1c860822e434fed8e2917.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/images/de0b80f1a35156dbe6ed5d089d75d1e54372d20061d1c860822e434fed8e2917.jpg)

### Tables

![4e437ba975bc05150f9f6050c29404a1b945616481ae988985bf3a15165e089c.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/4e437ba975bc05150f9f6050c29404a1b945616481ae988985bf3a15165e089c.jpg)

![98a1926f03d88bd8e2b320a87a9ed5f3bcc7c4be99d7f725a78fc128f7690c75.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/98a1926f03d88bd8e2b320a87a9ed5f3bcc7c4be99d7f725a78fc128f7690c75.jpg)

![9948cd0376451a684a23c733690efa753906c4330054d3a8765c7be42804f5eb.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/9948cd0376451a684a23c733690efa753906c4330054d3a8765c7be42804f5eb.jpg)

![b242920c1a50f78ccefca40ac87433485325b1333e2a3296d2af4047e7e44a9d.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/b242920c1a50f78ccefca40ac87433485325b1333e2a3296d2af4047e7e44a9d.jpg)

![b4b636ecea989f2fa4b2bd5c0b316628b89eee87a054b7f910859382d6d583e2.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/b4b636ecea989f2fa4b2bd5c0b316628b89eee87a054b7f910859382d6d583e2.jpg)

![edf0a65fd042e8430268774efc8ad229721bd94ad421b52312403661bca142d9.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/edf0a65fd042e8430268774efc8ad229721bd94ad421b52312403661bca142d9.jpg)

![f674b259005628fa4275da2711471ec0dbc2675ea002d7e3a51aa3067eb7fb19.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/f674b259005628fa4275da2711471ec0dbc2675ea002d7e3a51aa3067eb7fb19.jpg)

![f7127d5c04041263a32ada73c26749bce048d19a8afa8c37368f07c2db89a008.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/f7127d5c04041263a32ada73c26749bce048d19a8afa8c37368f07c2db89a008.jpg)

![fff3c5d2938eaa4e8410472700ccd5cd7a55dc79e7b1e3c4a207fb8229bd064d.jpg](../icml_results/1137_RBCBF_%20Decoding%20Time%20Safety%20Alignment%20via%20Risk%20Guided%20Rollback%20and%20Barrier%20Control/tables/fff3c5d2938eaa4e8410472700ccd5cd7a55dc79e7b1e3c4a207fb8229bd064d.jpg)

<a id="paper-1120"></a>

## $A_2$DEPT: Large Language Model–Driven Automated Algorithm Design via Evolutionary Program Trees


### Images

![0f9b6943294fe4baee7833eea7627ec34c2d7ccc4450f58c5e9dad72d07e580e.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/0f9b6943294fe4baee7833eea7627ec34c2d7ccc4450f58c5e9dad72d07e580e.jpg)

![62b5fc39fd522b0e73a0cd1bbf2d0b3438c95cca7441ea7f441d24fda76aee7a.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/62b5fc39fd522b0e73a0cd1bbf2d0b3438c95cca7441ea7f441d24fda76aee7a.jpg)

![7ea6f403e85ba0cc0b5023583f4271d3e8fe877ffd8517b8221d2d22244cd5d0.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/7ea6f403e85ba0cc0b5023583f4271d3e8fe877ffd8517b8221d2d22244cd5d0.jpg)

![93e752c961973a9e7cd1d57404de8a70b0b81b448d65aabb441b5fa1f13cd787.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/93e752c961973a9e7cd1d57404de8a70b0b81b448d65aabb441b5fa1f13cd787.jpg)

![eb27020cb3a5397a5b3496c3b13eda48de944d15e4904bafe5176c4526046cbe.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/eb27020cb3a5397a5b3496c3b13eda48de944d15e4904bafe5176c4526046cbe.jpg)

![ec95e54d9ff6bb7308837ab8820ccc2b2b2533e317c4e1383995c23cd6d635a1.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/ec95e54d9ff6bb7308837ab8820ccc2b2b2533e317c4e1383995c23cd6d635a1.jpg)

![f5c07d26b7ac19424c630f22a2ccddf1dbfee34dfed235d87e3fad97d3ea2e6e.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/f5c07d26b7ac19424c630f22a2ccddf1dbfee34dfed235d87e3fad97d3ea2e6e.jpg)

![feed5bc6101dbf62eec21c09942b474c4ffcf32c0e31fc8f78f5f3ab27240873.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/images/feed5bc6101dbf62eec21c09942b474c4ffcf32c0e31fc8f78f5f3ab27240873.jpg)

### Tables

![198ed2412244396065585fa546d6cae687d85db2f9d1e1faf3a2c73c25a4e07a.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/198ed2412244396065585fa546d6cae687d85db2f9d1e1faf3a2c73c25a4e07a.jpg)

![20591dfaaba516250bef941e4ed4872673a1f8b429a0ec570bcdc8384d2730f9.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/20591dfaaba516250bef941e4ed4872673a1f8b429a0ec570bcdc8384d2730f9.jpg)

![3b90e98675eb5c7e044734d0679fa6691430e183bce0af46e9061fe91bbda92a.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/3b90e98675eb5c7e044734d0679fa6691430e183bce0af46e9061fe91bbda92a.jpg)

![456ffc0c5b61ace19e0235e72089022651b553c55ba3a73f330c81dbbd6d06dc.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/456ffc0c5b61ace19e0235e72089022651b553c55ba3a73f330c81dbbd6d06dc.jpg)

![4dfbd5000fd5904e7bb7fd19186ffe1be66f9693caf40032bc93665156724972.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/4dfbd5000fd5904e7bb7fd19186ffe1be66f9693caf40032bc93665156724972.jpg)

![6748ec4dd7d3da22673231cdc5d210d524cd0b8fe963d43fbee41a2f1eb8f2b0.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/6748ec4dd7d3da22673231cdc5d210d524cd0b8fe963d43fbee41a2f1eb8f2b0.jpg)

![682e171a4f582d8be6a4fd22426448b8a08c9a449a772baeeae2a8baad93e041.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/682e171a4f582d8be6a4fd22426448b8a08c9a449a772baeeae2a8baad93e041.jpg)

![6ed632b8485b8519d94a9a9b091e65e46715ba4851132c9e78e10582c36fd595.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/6ed632b8485b8519d94a9a9b091e65e46715ba4851132c9e78e10582c36fd595.jpg)

![7c9649b2df6184e41d29859822fc95788f6378ea8e45397c989e9ae5fdb3783d.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/7c9649b2df6184e41d29859822fc95788f6378ea8e45397c989e9ae5fdb3783d.jpg)

![7ee5d3189f659bba4ace080a056b903c57c423eda6f88069f8ca221b6bbf4d66.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/7ee5d3189f659bba4ace080a056b903c57c423eda6f88069f8ca221b6bbf4d66.jpg)

![84cf7130569de8fc50ec22ad9d1571ae45e437dd3b0c6bd5e01d234f533c386b.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/84cf7130569de8fc50ec22ad9d1571ae45e437dd3b0c6bd5e01d234f533c386b.jpg)

![9d3f77c245a712c42b7bab4af546944f3876911d801545a2a8577c1c715a67f2.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/9d3f77c245a712c42b7bab4af546944f3876911d801545a2a8577c1c715a67f2.jpg)

![9e2eabb2600f345aaac1dd8509d5d8027c6fac2a14711383159eff72467f92d6.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/9e2eabb2600f345aaac1dd8509d5d8027c6fac2a14711383159eff72467f92d6.jpg)

![a7abb7f6bde6f138513313e4cbae377fec2764dd33225b1a5f817ae71ecb8b91.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/a7abb7f6bde6f138513313e4cbae377fec2764dd33225b1a5f817ae71ecb8b91.jpg)

![d66d047492191187083c68abd181aa4f09022e41c1d7e1495abe25a0b7cd7efb.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/d66d047492191187083c68abd181aa4f09022e41c1d7e1495abe25a0b7cd7efb.jpg)

![d91a45f347a8dbe7302c162f11f7931de338cbe2bc52493d14ad435db55cb5a2.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/d91a45f347a8dbe7302c162f11f7931de338cbe2bc52493d14ad435db55cb5a2.jpg)

![df19b340e84967c4cfccee60115069d3f184c518091e2b077a55c135217ad3c5.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/df19b340e84967c4cfccee60115069d3f184c518091e2b077a55c135217ad3c5.jpg)

![df3229cc505a516578f0f9406417823b8e13b5fd4ae1b68a576ea3eef254c5d3.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/df3229cc505a516578f0f9406417823b8e13b5fd4ae1b68a576ea3eef254c5d3.jpg)

![e90e14462ef3683b12cdcb59a66edd1ec0b8e39895540bfe5f4ad616a451ed14.jpg](../icml_results/1138_%24A_2%24DEPT_%20Large%20Language%20Model%E2%80%93Driven%20Automated%20Algorithm%20Design%20via%20Evolutionary%20Program%20Trees/tables/e90e14462ef3683b12cdcb59a66edd1ec0b8e39895540bfe5f4ad616a451ed14.jpg)

<a id="paper-1121"></a>

## $\textitS$-SPPO: Semantic-Calibrated Self-Play Preference Optimization


### Images

![092e08f03286bb6b437971ecc21bd92da47ceaa6050579bb21e8414f7492ecc8.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/images/092e08f03286bb6b437971ecc21bd92da47ceaa6050579bb21e8414f7492ecc8.jpg)

![0c2bf5156dafda1066653be9825e01632f72391dcb99098d75808ae91ca7f3c3.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/images/0c2bf5156dafda1066653be9825e01632f72391dcb99098d75808ae91ca7f3c3.jpg)

![112eaf987f4f8034ab274c19244a81fb785f5f2cc320eb0e25acaf8fd77e61cc.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/images/112eaf987f4f8034ab274c19244a81fb785f5f2cc320eb0e25acaf8fd77e61cc.jpg)

![86179e6761011cb4e931961b545b70a2d4ddf5bf89a2e952394d3a6fb3aac9e3.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/images/86179e6761011cb4e931961b545b70a2d4ddf5bf89a2e952394d3a6fb3aac9e3.jpg)

### Tables

![0911718bf04793d36e0da4fc7c7d95d52b506e22ecca120a161b69aa37bbaaf6.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/0911718bf04793d36e0da4fc7c7d95d52b506e22ecca120a161b69aa37bbaaf6.jpg)

![5bcc06e05e61bea6f1c6569bc1d6b1ed9cddf966eda5ab278ff36a0b1bf56f17.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/5bcc06e05e61bea6f1c6569bc1d6b1ed9cddf966eda5ab278ff36a0b1bf56f17.jpg)

![661df59aaf28c8942092fd3c0691f7f2ae597085ecd711848d9962cf63b8df03.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/661df59aaf28c8942092fd3c0691f7f2ae597085ecd711848d9962cf63b8df03.jpg)

![681a45bb2d86942a86c67522ff960c61ef7180d07d772d69d8828f7be4e9b28f.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/681a45bb2d86942a86c67522ff960c61ef7180d07d772d69d8828f7be4e9b28f.jpg)

![7079117980c6eeb557c45a919f4dc33b4179a5293fb663c00b9a73788ba16ded.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/7079117980c6eeb557c45a919f4dc33b4179a5293fb663c00b9a73788ba16ded.jpg)

![7392548640ea4155f5f4966154022ba10d6edc55483fe8060bf7de5ebfe19dee.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/7392548640ea4155f5f4966154022ba10d6edc55483fe8060bf7de5ebfe19dee.jpg)

![87cfe84d951c30df803fdc6aead608a696000937b90d0fe14cd15c54407d38d4.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/87cfe84d951c30df803fdc6aead608a696000937b90d0fe14cd15c54407d38d4.jpg)

![abc4fa6d72b762065f39217b8e5cbe57969f12440fa6ff9ff6cf9ac5cb1814e9.jpg](../icml_results/1139_%24_textitS%24-SPPO_%20Semantic-Calibrated%20Self-Play%20Preference%20Optimization/tables/abc4fa6d72b762065f39217b8e5cbe57969f12440fa6ff9ff6cf9ac5cb1814e9.jpg)

<a id="paper-1122"></a>

## Rethinking the Flow-based Gradual Domain Adaptation: A Semi-Dual Optimal Transport Perspective

### Images

![2e7053fa3bb28e7017aa9a7d24f9e3d40ebfb8a9e88f25c959a62471f5d14c0a.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/2e7053fa3bb28e7017aa9a7d24f9e3d40ebfb8a9e88f25c959a62471f5d14c0a.jpg)

![43d08410b60754b7e9ab0ce18d4ed184d3799f9b9afb601e03634e288637ed3e.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/43d08410b60754b7e9ab0ce18d4ed184d3799f9b9afb601e03634e288637ed3e.jpg)

![6143687678a167ba50e92d5953de72b85753b3f814ba6895e8105abbddad73a8.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/6143687678a167ba50e92d5953de72b85753b3f814ba6895e8105abbddad73a8.jpg)

![71e074b2421bb662ddd5f8ecb63b41c712123b2059967f420945821b39afedb4.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/71e074b2421bb662ddd5f8ecb63b41c712123b2059967f420945821b39afedb4.jpg)

![9308f9a06593f0cdc3fa6e466ffedc1661b94dc45d02db0f4713be61e042f94c.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/9308f9a06593f0cdc3fa6e466ffedc1661b94dc45d02db0f4713be61e042f94c.jpg)

![a4184ba99efd89975330ff8c791565f4f328935b52db0812dbe54ea1a932f75b.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/a4184ba99efd89975330ff8c791565f4f328935b52db0812dbe54ea1a932f75b.jpg)

![bde68ecc359e5636269ce523b6da99b9c4d07d3f1698b1e65cb6044bf75d3934.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/bde68ecc359e5636269ce523b6da99b9c4d07d3f1698b1e65cb6044bf75d3934.jpg)

![da225a8d25d4a05b052e2b16a4cfebee661058ce2f843e56d1b46c3431e2b215.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/da225a8d25d4a05b052e2b16a4cfebee661058ce2f843e56d1b46c3431e2b215.jpg)

![db402a7002f998490ebc2b62c4cb1810e431651ef92da6cc9891b48eaa65aa00.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/db402a7002f998490ebc2b62c4cb1810e431651ef92da6cc9891b48eaa65aa00.jpg)

![de93ff327ba5a2fd3eca6457ac31c601fb9dde7aaeaa6fcb8516d5ffd331f355.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/de93ff327ba5a2fd3eca6457ac31c601fb9dde7aaeaa6fcb8516d5ffd331f355.jpg)

![e508fdd2f310e5b2d6771f8007556d7e0a6870cec197836b7e9c7ae3b5afc0f9.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/images/e508fdd2f310e5b2d6771f8007556d7e0a6870cec197836b7e9c7ae3b5afc0f9.jpg)

### Tables

![05222b92b9c898e83d81111547017172b2dcacbccf6f0674dac56e7e3a6c4e27.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/05222b92b9c898e83d81111547017172b2dcacbccf6f0674dac56e7e3a6c4e27.jpg)

![0a28ab9a027e4b5e5929f88e04492216e5b00d06d9a173b49f79284c445dec37.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/0a28ab9a027e4b5e5929f88e04492216e5b00d06d9a173b49f79284c445dec37.jpg)

![1ffaecd6012ceb34186b6d78d0fe7e42373fa49425ece40d152db6204e1feb45.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/1ffaecd6012ceb34186b6d78d0fe7e42373fa49425ece40d152db6204e1feb45.jpg)

![273365c8c2324eedabe54864559933e06a45337a64bb22699b5bd95a728512dd.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/273365c8c2324eedabe54864559933e06a45337a64bb22699b5bd95a728512dd.jpg)

![2e7bd73347ae2436bcc73d19bc7f749295fd3c7f8a9a4fd70392ca1c5a9656b4.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/2e7bd73347ae2436bcc73d19bc7f749295fd3c7f8a9a4fd70392ca1c5a9656b4.jpg)

![4e81c7d845dab81f7dae9c6d5a62fe6af27b1f1374bab23c2b5997d665648e1b.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/4e81c7d845dab81f7dae9c6d5a62fe6af27b1f1374bab23c2b5997d665648e1b.jpg)

![5f68b37c958fbe36428688180725bd8b7322f13c7d28ae73963eb51c93e4174d.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/5f68b37c958fbe36428688180725bd8b7322f13c7d28ae73963eb51c93e4174d.jpg)

![8d6d2acecb5105038f91b51ee86dffe8b066cfb6fee12c2b029e045f31cdc255.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/8d6d2acecb5105038f91b51ee86dffe8b066cfb6fee12c2b029e045f31cdc255.jpg)

![b6404c14719447320f8d6eacc08e720b29bee13e89132db98a3f49d5f5dfac71.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/b6404c14719447320f8d6eacc08e720b29bee13e89132db98a3f49d5f5dfac71.jpg)

![bd42b27b293d709b38456bcd234e26b1141de933888026f27835d23c109799e1.jpg](../icml_results/1140_Rethinking%20the%20Flow-based%20Gradual%20Domain%20Adaptation_%20A%20Semi-Dual%20Optimal%20Transport%20Perspective/tables/bd42b27b293d709b38456bcd234e26b1141de933888026f27835d23c109799e1.jpg)
