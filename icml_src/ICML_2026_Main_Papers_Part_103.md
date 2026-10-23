# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 103 of 197

## 目录 (Table of Contents)

1. [Learning Generalized Trackers with Elastic Token Budgets](#paper-3367)
2. [PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Update](#paper-3368)
3. [What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom](#paper-3369)
4. [Reflector: Internalizing Step-wise Reflection against Indirect Jailbreaks](#paper-3370)
5. [CodeMamba: Shifting from Target Semantics to Self-Supervised Background Manifold Learning for Singularity Detection in Infrared Sequences](#paper-3371)
6. [Scaling Agentic Verifier for Competitive Coding](#paper-3372)
7. [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](#paper-3373)
8. [Trifuse: Enhancing Attention-Based GUI Grounding via Multimodal Fusion](#paper-3374)
9. [Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](#paper-3375)
10. [Scalable Traffic Signal Control with Shared Policy Framework](#paper-3376)
11. [Mechanisms of Introspective Awareness](#paper-3377)
12. [Characterizing the Predictive Impact of Modalities with Supervised Latent-Variable Modeling](#paper-3378)
13. [Rethinking Thinking Tokens: LLMs as Improvement Operators](#paper-3379)
14. [Inference-time optimization for experiment-grounded protein ensemble generation](#paper-3380)
15. [HypRAG: Hyperbolic Dense Retrieval for Retrieval Augmented Generation](#paper-3381)
16. [Collaborative and Efficient Fine-tuning: Leveraging Task Similarity](#paper-3382)
17. [Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](#paper-3383)
18. [Scaling Generative Verifiers For Natural Language Mathematical Proof Verification And Selection](#paper-3384)
19. [Joint-Embedding Predictive Learning of Latent Market States in U.S. Equities](#paper-3385)
20. [Correcting Overparameterization Effects in Fair Empirical Risk Minimization](#paper-3386)
21. [Revisiting the Bertrand Paradox via Equilibrium Analysis of No-regret Learners](#paper-3387)
22. [Corruption-Tolerant Asynchronous Q-Learning with Near-Optimal Rates](#paper-3388)
23. [Reward Shaping Control Variates for Off-Policy Evaluation Under Sparse Rewards](#paper-3389)
24. [TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins](#paper-3390)
25. [Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings](#paper-3391)
26. [MacroGuide: Topological Guidance for Macrocycle Generation](#paper-3392)
27. [EnterpriseOps-Gym: Environments and Evaluations for Stateful Agentic Planning and Tool Use in Enterprise Settings](#paper-3393)
28. [MADE: Benchmark Environments for Closed-Loop Materials Discovery](#paper-3394)
29. [Directly Optimizing Natural Language Explanations for Behavioral Faithfulness: Simulatability and Recoverability](#paper-3395)
30. [Variational Flow Maps: Make Some Noise for One-Step Conditional Generation](#paper-3396)
31. [Turbo4DGen: Ultra-Fast Acceleration for 4D Generation](#paper-3397)
32. [The Consistency Dilemma in LLMs: Generator-Evaluator Agreement and Vulnerability to Mistakes](#paper-3398)
33. [Distributionally Robust Reinforcement Learning from Human Feedback](#paper-3399)

---


<a id="paper-3367"></a>

## Learning Generalized Trackers with Elastic Token Budgets

### Images

![75b7c121aa0802db8a428c2e5dbeef1f4b9d3ecd1fddb04e10774e2ae2f72701.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/images/75b7c121aa0802db8a428c2e5dbeef1f4b9d3ecd1fddb04e10774e2ae2f72701.jpg)

![7a7ffc9780ad116d40fc9b221da9da51b804cede59d16d255440dfa1b4226ae3.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/images/7a7ffc9780ad116d40fc9b221da9da51b804cede59d16d255440dfa1b4226ae3.jpg)

![7d3bc161b8878b1262b35dbf12a0b3fef9d64642ec34b3ce16551ea690a6c6b6.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/images/7d3bc161b8878b1262b35dbf12a0b3fef9d64642ec34b3ce16551ea690a6c6b6.jpg)

![862023610f1153a7d8ae064a61a3a1132b2e14971e04ce6100200c53da61db21.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/images/862023610f1153a7d8ae064a61a3a1132b2e14971e04ce6100200c53da61db21.jpg)

![8ab14490a7d349ca1704494f2a6fc25f5788b28f48a709c5ba2ba5389990ace1.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/images/8ab14490a7d349ca1704494f2a6fc25f5788b28f48a709c5ba2ba5389990ace1.jpg)

![ddc8ebc375d59823fba2f914f0eff2374030c9fa979ec2c73a6505ac2ed40fcb.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/images/ddc8ebc375d59823fba2f914f0eff2374030c9fa979ec2c73a6505ac2ed40fcb.jpg)

### Tables

![07bf31340c7616adcd47df7626d8e58dae5897e84a068e05bae531013f1bb94a.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/07bf31340c7616adcd47df7626d8e58dae5897e84a068e05bae531013f1bb94a.jpg)

![2b24f541840276105034bcecb3d1c0563f58226abb9c13f4e31abf4f2caf1a2f.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/2b24f541840276105034bcecb3d1c0563f58226abb9c13f4e31abf4f2caf1a2f.jpg)

![4e408d967c75489c42a0f031e53384172ea89eb28d1d5cc878bc1ab4a1fd0c80.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/4e408d967c75489c42a0f031e53384172ea89eb28d1d5cc878bc1ab4a1fd0c80.jpg)

![6ff770324387ab7c9ded62530939af9f2bbca0f14a1f28fc44d21d7040c85cf4.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/6ff770324387ab7c9ded62530939af9f2bbca0f14a1f28fc44d21d7040c85cf4.jpg)

![ad2be1437771ffca77720788f4b2c4b37ee4b185617e2976071d9a99f90a160f.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/ad2be1437771ffca77720788f4b2c4b37ee4b185617e2976071d9a99f90a160f.jpg)

![befae738ddd84a1e61293fbb183a5c19d7d326f6ae2a7b845546b1811f05eb50.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/befae738ddd84a1e61293fbb183a5c19d7d326f6ae2a7b845546b1811f05eb50.jpg)

![f5768ef673905292c579656664d7af996b6852ab36989e7a4b6ca81139845984.jpg](../icml_results/3406_Can%20LLM%20Agents%20Stick%20to%20the%20Script_%20A%20Benchmark%20for%20Long-Horizon%20Consistency%20in%20Interactive%20Narrativ/tables/f5768ef673905292c579656664d7af996b6852ab36989e7a4b6ca81139845984.jpg)

## Learning Generalized Trackers with Elastic Token Budgets


### Images

![273b80bedf215a2adf9cc45b11b6fe7d8d058d930494917a82e72086a6cf01ce.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/273b80bedf215a2adf9cc45b11b6fe7d8d058d930494917a82e72086a6cf01ce.jpg)

![3f29e282833d06a06a4e42180d8ff1c7bec81a2be8f73a4813c5a4a721a2f658.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/3f29e282833d06a06a4e42180d8ff1c7bec81a2be8f73a4813c5a4a721a2f658.jpg)

![5bfb6f4ac24e042e07bf6b8f47250c0d719229c8cf628471b21e2c50a8d3d416.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/5bfb6f4ac24e042e07bf6b8f47250c0d719229c8cf628471b21e2c50a8d3d416.jpg)

![f20a60d96640459cec4061e912ebc28202951298d0898b5c2c3d251f86a9ebdc.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/f20a60d96640459cec4061e912ebc28202951298d0898b5c2c3d251f86a9ebdc.jpg)

![f32478476e1806202089cbc1db7bff1ba0ab6cb179e61339e7f09a28357ce565.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/f32478476e1806202089cbc1db7bff1ba0ab6cb179e61339e7f09a28357ce565.jpg)

![f3b4e2c2a043ca4d90068df911dff08691bca70d066466a2a99559ff634f8620.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/f3b4e2c2a043ca4d90068df911dff08691bca70d066466a2a99559ff634f8620.jpg)

![fa1b40aab438b293f4c9f5a460d0e47335874f9379772e01127b06b5958443c6.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/images/fa1b40aab438b293f4c9f5a460d0e47335874f9379772e01127b06b5958443c6.jpg)

### Tables

![0b8d5799a0791b2bc635f701428ba4c42b3ab33b32c18b94226d6ccee187f1cc.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/0b8d5799a0791b2bc635f701428ba4c42b3ab33b32c18b94226d6ccee187f1cc.jpg)

![0b91d69c9fce6c57a0b9c58f245e1da1655303da33ec67cda387113c1be8ad6a.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/0b91d69c9fce6c57a0b9c58f245e1da1655303da33ec67cda387113c1be8ad6a.jpg)

![16d3f0ff50105556a50e1cb3faf6b30dda6686dd8a24e57b25b6a1909de90562.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/16d3f0ff50105556a50e1cb3faf6b30dda6686dd8a24e57b25b6a1909de90562.jpg)

![314f313d71c66399199916966fee98ca89090e4f56aa78335f93737c1ddfc293.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/314f313d71c66399199916966fee98ca89090e4f56aa78335f93737c1ddfc293.jpg)

![360c1b5b058a5caec75f07830a7385fa55c978393882ce8fc477a44c247628af.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/360c1b5b058a5caec75f07830a7385fa55c978393882ce8fc477a44c247628af.jpg)

![40fa7218160df9f6169c3efba1ccc4bf01d06e4685dca9939f697a80ab642962.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/40fa7218160df9f6169c3efba1ccc4bf01d06e4685dca9939f697a80ab642962.jpg)

![456678fbf8d8b6ebe204aae15a4f64e9d3e8450e8a3ed74499b5c87a6324b5ca.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/456678fbf8d8b6ebe204aae15a4f64e9d3e8450e8a3ed74499b5c87a6324b5ca.jpg)

![458c274d767abd1bcdf68eddacb99f93209e590f2dab5f7dedf38662c9e1628a.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/458c274d767abd1bcdf68eddacb99f93209e590f2dab5f7dedf38662c9e1628a.jpg)

![597858c6e42644b88fb1fe04300d41cedf702514818b19ff4530d59551b1dcb2.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/597858c6e42644b88fb1fe04300d41cedf702514818b19ff4530d59551b1dcb2.jpg)

![69139d1f2ca1420021ec50e71b84267a72c0ed4b40a29e900345919a5cc6d493.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/69139d1f2ca1420021ec50e71b84267a72c0ed4b40a29e900345919a5cc6d493.jpg)

![97d07eb2f9473fe22c418ac5c2e620035ae305a97a4d5ad65dcba8eca68af32e.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/97d07eb2f9473fe22c418ac5c2e620035ae305a97a4d5ad65dcba8eca68af32e.jpg)

![b2217f696863a160e9aa09559c6916479d68127d13267f1e0f17da5b7587cdbb.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/b2217f696863a160e9aa09559c6916479d68127d13267f1e0f17da5b7587cdbb.jpg)

![cb56af6ec6d5e9e95715422b0cd53d966ef6068f7a0f34c418dc3e717d28451a.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/cb56af6ec6d5e9e95715422b0cd53d966ef6068f7a0f34c418dc3e717d28451a.jpg)

![d6c210b753e31c3092b8033260b87f7af1f1a639321d2d67eded89738748e884.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/d6c210b753e31c3092b8033260b87f7af1f1a639321d2d67eded89738748e884.jpg)

![fa0116173d2923ad1797e92e065b181e3342536fc55cec0aee2ff1893eabb0b6.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/fa0116173d2923ad1797e92e065b181e3342536fc55cec0aee2ff1893eabb0b6.jpg)

![fee253a7ee53f42ad5811b4407c2f3bcf1bb57b064fa3a7e95afe48e2aab8228.jpg](../icml_results/3407_Learning%20Generalized%20Trackers%20with%20Elastic%20Token%20Budgets/tables/fee253a7ee53f42ad5811b4407c2f3bcf1bb57b064fa3a7e95afe48e2aab8228.jpg)

<a id="paper-3368"></a>

## PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Update


### Images

![228806016123bb56812e8816fff90eb06a11fcfc2d497898ddbb2132a06db797.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/228806016123bb56812e8816fff90eb06a11fcfc2d497898ddbb2132a06db797.jpg)

![26372da6c216fc36f2e7805de5e89aa06ea92a85f77a373110f1e690d89547a5.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/26372da6c216fc36f2e7805de5e89aa06ea92a85f77a373110f1e690d89547a5.jpg)

![4431b8ddb2435af34b0fc97dd8e3adb1cb10fd670518ace1c10519b49d0dd85b.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/4431b8ddb2435af34b0fc97dd8e3adb1cb10fd670518ace1c10519b49d0dd85b.jpg)

![5a57589485fed0c09ecbba83540e03d81ff7e1f7f677f4e8bf44f9996e95c969.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/5a57589485fed0c09ecbba83540e03d81ff7e1f7f677f4e8bf44f9996e95c969.jpg)

![7cb0579e698213a89e13ef370e98ca0086d29f4e5e3afdd78cec77e04179835a.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/7cb0579e698213a89e13ef370e98ca0086d29f4e5e3afdd78cec77e04179835a.jpg)

![7d4ee708e56cbffb8ee67f368ce3bb2c6f37d6a002948d172fd309cf8ddf0042.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/7d4ee708e56cbffb8ee67f368ce3bb2c6f37d6a002948d172fd309cf8ddf0042.jpg)

![8e5c77b78967d8510fd4e9d3b7bcdb2161d7f18281d3b22abb87126116273e13.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/8e5c77b78967d8510fd4e9d3b7bcdb2161d7f18281d3b22abb87126116273e13.jpg)

![b10f27a54a7641316700adf1166c6aebcf8815c90dd4a9447f1f6f5eac6eda96.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/b10f27a54a7641316700adf1166c6aebcf8815c90dd4a9447f1f6f5eac6eda96.jpg)

![c32d73cae5627c83eb9d1b1873350003539e958e3e22a01fe4243f9b6391df95.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/c32d73cae5627c83eb9d1b1873350003539e958e3e22a01fe4243f9b6391df95.jpg)

![e0dfe7829292d2ba3312089e5f69e7dc8214ec697b23bd1fcfd9006e9a4edc54.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/e0dfe7829292d2ba3312089e5f69e7dc8214ec697b23bd1fcfd9006e9a4edc54.jpg)

![e704b5eb3cf48879829cd9af60d3c0172ae4f63f3a3e2e96a797cd64fa36b813.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/e704b5eb3cf48879829cd9af60d3c0172ae4f63f3a3e2e96a797cd64fa36b813.jpg)

![f16ae7875b584d8be5f1248c97ad18d8bf15f9bf8ddd5b0a663ad1cb8dec929e.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/images/f16ae7875b584d8be5f1248c97ad18d8bf15f9bf8ddd5b0a663ad1cb8dec929e.jpg)

### Tables

![043d8dbede9834f1b58c8ed2f316fb7c359f22141da3ac0d84b4dee41dd884ff.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/043d8dbede9834f1b58c8ed2f316fb7c359f22141da3ac0d84b4dee41dd884ff.jpg)

![1bcc3c1b735484a17ae4ef00bd290161f076aa8886cd6aecc3562245deade1a1.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/1bcc3c1b735484a17ae4ef00bd290161f076aa8886cd6aecc3562245deade1a1.jpg)

![1be034e931addebdbc2e3c62dbbc2e1a68de2b2d6919df6e4ec9e0f1114e62f0.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/1be034e931addebdbc2e3c62dbbc2e1a68de2b2d6919df6e4ec9e0f1114e62f0.jpg)

![1e9b0f123144bd3a83e65f8d8dca8c6eb5b26284c246b8629443925adb18343a.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/1e9b0f123144bd3a83e65f8d8dca8c6eb5b26284c246b8629443925adb18343a.jpg)

![25a84b54c233c0e8c11b5b1b74dd21311ce8684b4e73707b92697938b4455e36.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/25a84b54c233c0e8c11b5b1b74dd21311ce8684b4e73707b92697938b4455e36.jpg)

![2ac4d6b556267c06dd70bb0d71e8ea9c928b37d644f3eb088f62dacd1ee1c899.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/2ac4d6b556267c06dd70bb0d71e8ea9c928b37d644f3eb088f62dacd1ee1c899.jpg)

![2ba45f57874dbc0290a9f26fec6f3be8e337c074ca8204abe19d4d3892056d06.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/2ba45f57874dbc0290a9f26fec6f3be8e337c074ca8204abe19d4d3892056d06.jpg)

![32f8a7996c6763617bad1f44e8f2d3846126011ab8cb272ff6c84501f4cea684.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/32f8a7996c6763617bad1f44e8f2d3846126011ab8cb272ff6c84501f4cea684.jpg)

![4dd3adb2bad8f0a6962bba68c1bfa34f97b8b2d748e899385e40b2d30b5040ff.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/4dd3adb2bad8f0a6962bba68c1bfa34f97b8b2d748e899385e40b2d30b5040ff.jpg)

![619f3d63b74c7df16ebf92c0db6c86a5350beb498c6178b176a0dc94d69bdd7e.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/619f3d63b74c7df16ebf92c0db6c86a5350beb498c6178b176a0dc94d69bdd7e.jpg)

![739607dfcf3a6bb73130a99041373e679d6aa3c6f92b4de5c983df1215734348.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/739607dfcf3a6bb73130a99041373e679d6aa3c6f92b4de5c983df1215734348.jpg)

![7d6d7ac95b7d409a0b46966db49b197a71c8d4f09c2932e52d7713071412d1ab.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/7d6d7ac95b7d409a0b46966db49b197a71c8d4f09c2932e52d7713071412d1ab.jpg)

![83cf131a0ff7200795bd386397a3f4d1ed189aaafb6b33a6862b86bd0e55fca3.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/83cf131a0ff7200795bd386397a3f4d1ed189aaafb6b33a6862b86bd0e55fca3.jpg)

![91732622409949a79b5b131154f4083428fd8e367edd3fbbed670ef7ee17bd0c.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/91732622409949a79b5b131154f4083428fd8e367edd3fbbed670ef7ee17bd0c.jpg)

![93454aec7a861f257d982e9d2641b7bac9893b5f279895e512e669c835741638.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/93454aec7a861f257d982e9d2641b7bac9893b5f279895e512e669c835741638.jpg)

![9732e3caa058f2985f13d42c05cea5618e60abaa11deca3a0d984c922841e6fb.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/9732e3caa058f2985f13d42c05cea5618e60abaa11deca3a0d984c922841e6fb.jpg)

![989344545494538b3331f07bfd46764cfb603114b5388e3307b6ea2c51c44ce9.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/989344545494538b3331f07bfd46764cfb603114b5388e3307b6ea2c51c44ce9.jpg)

![9b6c018f95b1537d6d7593e5d54c70e7d1591e2ced3e29afcd80def58d984be2.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/9b6c018f95b1537d6d7593e5d54c70e7d1591e2ced3e29afcd80def58d984be2.jpg)

![9bb22457ef5d828f77822f65c30dfbf41418e981bdccc30c9e65cab50c63178b.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/9bb22457ef5d828f77822f65c30dfbf41418e981bdccc30c9e65cab50c63178b.jpg)

![9df497f61f50de4b78522f9b63a25ea5f5d7ef77530e41392f10925ae2515c6f.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/9df497f61f50de4b78522f9b63a25ea5f5d7ef77530e41392f10925ae2515c6f.jpg)

![ace3d2c8dae3218104f23d4b2e725c47bd8ca2adf1100692dd11164f6ec90f5d.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/ace3d2c8dae3218104f23d4b2e725c47bd8ca2adf1100692dd11164f6ec90f5d.jpg)

![af6d25c539adc8f3dd6d5ad55da888d5299a023cab2947242d48b6bf7be750a0.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/af6d25c539adc8f3dd6d5ad55da888d5299a023cab2947242d48b6bf7be750a0.jpg)

![dbeddc2b580fe8023368b90f17f5f4d00469dad1ac58d1858fac137367849143.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/dbeddc2b580fe8023368b90f17f5f4d00469dad1ac58d1858fac137367849143.jpg)

![dd0b13f22eb2a2d40c55bd289fdfb01fabb73ab2b03f58906d52c85d647b9cdf.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/dd0b13f22eb2a2d40c55bd289fdfb01fabb73ab2b03f58906d52c85d647b9cdf.jpg)

![ddc22df7984d6407ca449e86847a7dac145bfb25407792ecd077d66cbdcdca76.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/ddc22df7984d6407ca449e86847a7dac145bfb25407792ecd077d66cbdcdca76.jpg)

![e561b22b654fc73779a6720ee9eba2e762b2c9a4749e1ce89438afc4e6a2ea2e.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/e561b22b654fc73779a6720ee9eba2e762b2c9a4749e1ce89438afc4e6a2ea2e.jpg)

![ee01a85412d89ccd2f2ad6608a5f706a87e56d9f4d1edf57971bce00743eca92.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/ee01a85412d89ccd2f2ad6608a5f706a87e56d9f4d1edf57971bce00743eca92.jpg)

![f06571559b410462bcfab0f9bcd729d52ee63da44227e89e6b3e33569b67922b.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/f06571559b410462bcfab0f9bcd729d52ee63da44227e89e6b3e33569b67922b.jpg)

![f8c1b53415b80eccc64952897f41c2392a4fbcb1697b87d3835554c93db4f31a.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/f8c1b53415b80eccc64952897f41c2392a4fbcb1697b87d3835554c93db4f31a.jpg)

![fe0e09d0745e532ed6cc21825d7336bc3b0be7dd8e1ec7cfff8914ae41137155.jpg](../icml_results/3408_PolyFlow_%20Safe%20and%20Efficient%20Polytope-Constrained%20Flow%20Matching%20with%20Constraint%20Embedding%20and%20Projec/tables/fe0e09d0745e532ed6cc21825d7336bc3b0be7dd8e1ec7cfff8914ae41137155.jpg)

<a id="paper-3369"></a>

## What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom


### Images

![11ded7d88331ae5b5798a5bbb0c74ee45f291b7baf741bc6eff24e1a5a3adf20.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/11ded7d88331ae5b5798a5bbb0c74ee45f291b7baf741bc6eff24e1a5a3adf20.jpg)

![50d36c5b396d68994ae04027e5d3effdf8a3ccd3f86b82f9c7c207a5cfad20db.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/50d36c5b396d68994ae04027e5d3effdf8a3ccd3f86b82f9c7c207a5cfad20db.jpg)

![566f3c585f0b90a50b741452d7fd9c233b14f39ad7a0c4ddf48a28fc0f20004b.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/566f3c585f0b90a50b741452d7fd9c233b14f39ad7a0c4ddf48a28fc0f20004b.jpg)

![610ff6549e5d078b361101df2feab2159e784e1b9df630785cbe08da21d32aca.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/610ff6549e5d078b361101df2feab2159e784e1b9df630785cbe08da21d32aca.jpg)

![7b7a6ae34b162a2703d5c86b21d4c59220652a11b1798f9b0aa1c5a8c4e0358d.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/7b7a6ae34b162a2703d5c86b21d4c59220652a11b1798f9b0aa1c5a8c4e0358d.jpg)

![9d6732a39a15810771ad88afa07d23ace8aafc159583a42f3584e4f8a953f890.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/9d6732a39a15810771ad88afa07d23ace8aafc159583a42f3584e4f8a953f890.jpg)

![c7ec1a96740fde40a3d10baf288d18013a32831a4d9d539e7c552046b510f7d5.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/c7ec1a96740fde40a3d10baf288d18013a32831a4d9d539e7c552046b510f7d5.jpg)

![ce4fd5ac47eeea2cd0cec3f4bb53f37686f3fe6138f7fed9cd4c573d83769ede.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/ce4fd5ac47eeea2cd0cec3f4bb53f37686f3fe6138f7fed9cd4c573d83769ede.jpg)

![ce5fd42a8be79ba1dcefedcbf904686dc58b1f5cee89ee90149be37734a4360a.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/ce5fd42a8be79ba1dcefedcbf904686dc58b1f5cee89ee90149be37734a4360a.jpg)

![e2d6602a09c11eefb42542712cfe71801186d36e52cb099b25ed9f7327af6c13.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/images/e2d6602a09c11eefb42542712cfe71801186d36e52cb099b25ed9f7327af6c13.jpg)

### Tables

![15bff80939b107783affcef8808f8f0ab24d2ccbfd371ef021093fdf0a76076f.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/15bff80939b107783affcef8808f8f0ab24d2ccbfd371ef021093fdf0a76076f.jpg)

![3ea8f810bbdeba22483914bd9d6b7ba62f115667d8e098c45bf48eadae3963e5.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/3ea8f810bbdeba22483914bd9d6b7ba62f115667d8e098c45bf48eadae3963e5.jpg)

![5ee38e81fe2c43560661a3a4bcd5978673140eb250f1da38cd581f8236ae8ad4.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/5ee38e81fe2c43560661a3a4bcd5978673140eb250f1da38cd581f8236ae8ad4.jpg)

![9fd6f370f570ac169050623d24bdcffd8a5fb8fb9fd3be2794fb3b3540d79aa0.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/9fd6f370f570ac169050623d24bdcffd8a5fb8fb9fd3be2794fb3b3540d79aa0.jpg)

![b318e4086b92307b3f4c9c920c44a06b72137a94bf371c7c7edba2b2ebd4cf91.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/b318e4086b92307b3f4c9c920c44a06b72137a94bf371c7c7edba2b2ebd4cf91.jpg)

![d3ab7abb47fe689f44c97bd68f2fbadf2d06e8d063bcb359570e7ac3aff1b7d2.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/d3ab7abb47fe689f44c97bd68f2fbadf2d06e8d063bcb359570e7ac3aff1b7d2.jpg)

![ed4dfca2ab72a33bdaa53612e40e5f5db5a3f7e96731fef2f2dfc9aa3f3b9e01.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/ed4dfca2ab72a33bdaa53612e40e5f5db5a3f7e96731fef2f2dfc9aa3f3b9e01.jpg)

![f5888955731577d2d760f244e92667068462290d88abc61de698b75dad7b3a0f.jpg](../icml_results/3409_What%20Does%20Vision%20Tool-Use%20Reinforcement%20Learning%20Really%20Learn_%20Disentangling%20Tool-Induced%20and%20Intrin/tables/f5888955731577d2d760f244e92667068462290d88abc61de698b75dad7b3a0f.jpg)

<a id="paper-3370"></a>

## Reflector: Internalizing Step-wise Reflection against Indirect Jailbreaks


### Images

![197b2ddd1b729bee30c83db11d54138fe88e1b0f62557ae5ba92d20db5feb1a1.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/images/197b2ddd1b729bee30c83db11d54138fe88e1b0f62557ae5ba92d20db5feb1a1.jpg)

![4d9746bf3fb8d71c99784abe62d13334f358e923b9f562b0b562f25cc09a6393.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/images/4d9746bf3fb8d71c99784abe62d13334f358e923b9f562b0b562f25cc09a6393.jpg)

![e8cbee581707db424fc4eb45d016639a12ed8155cd58943d3562c18d3341fd03.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/images/e8cbee581707db424fc4eb45d016639a12ed8155cd58943d3562c18d3341fd03.jpg)

### Tables

![008051140579c039fe53131c1548ce28dddcc6f51783646e837919660e130f7d.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/008051140579c039fe53131c1548ce28dddcc6f51783646e837919660e130f7d.jpg)

![162200ee62ef33e6ff880a5e223f05632ee640d35455423d9ec0fe555447eb81.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/162200ee62ef33e6ff880a5e223f05632ee640d35455423d9ec0fe555447eb81.jpg)

![306143733453a3d5f0e7b25b2f57bc96746babc7ccaa4db95dfc51789929e5a0.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/306143733453a3d5f0e7b25b2f57bc96746babc7ccaa4db95dfc51789929e5a0.jpg)

![350855223b31bf2fce72b9a8ca9495b2e405fc388d8844e6b30e90f88737cd50.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/350855223b31bf2fce72b9a8ca9495b2e405fc388d8844e6b30e90f88737cd50.jpg)

![39b3482527742192e7644baf7396a5a5af9ef0b982655e7e5405fd5eff6e8dce.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/39b3482527742192e7644baf7396a5a5af9ef0b982655e7e5405fd5eff6e8dce.jpg)

![4ff434e4d7f34143555064731dbe43466a17e0784d1017a42caf165d96e33c48.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/4ff434e4d7f34143555064731dbe43466a17e0784d1017a42caf165d96e33c48.jpg)

![528f730041aee5199a36ae93a4d5b82ae08a942ec69b6baa4629af7e9ebecb25.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/528f730041aee5199a36ae93a4d5b82ae08a942ec69b6baa4629af7e9ebecb25.jpg)

![67a3df232e3d6284f812095b6f92a5288c8bb16ecc0f0c9a9ffd838286152230.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/67a3df232e3d6284f812095b6f92a5288c8bb16ecc0f0c9a9ffd838286152230.jpg)

![7e028b2bc02882fa5d40b17b5c3501663ab939a55c41c25ddb7b094ce7543aa4.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/7e028b2bc02882fa5d40b17b5c3501663ab939a55c41c25ddb7b094ce7543aa4.jpg)

![a5051d759464d91bbfab73838d38c95990db9d52dc8f136b71fc4bfa1eb73be5.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/a5051d759464d91bbfab73838d38c95990db9d52dc8f136b71fc4bfa1eb73be5.jpg)

![cef77f8dbd02312da081cfac92866cb58967a4bcc31a0f01524149497c5c0c64.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/cef77f8dbd02312da081cfac92866cb58967a4bcc31a0f01524149497c5c0c64.jpg)

![d54afefce2914685c816417d7e216961ed9810472a26ef2633949ff1a65cf690.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/d54afefce2914685c816417d7e216961ed9810472a26ef2633949ff1a65cf690.jpg)

![fc81c3c11778cc34905ed0eefd5dab4df590467fadfbf9e18c3da402f2e5b7f3.jpg](../icml_results/3410_Reflector_%20Internalizing%20Step-wise%20Reflection%20against%20Indirect%20Jailbreaks/tables/fc81c3c11778cc34905ed0eefd5dab4df590467fadfbf9e18c3da402f2e5b7f3.jpg)

<a id="paper-3371"></a>

## CodeMamba: Shifting from Target Semantics to Self-Supervised Background Manifold Learning for Singularity Detection in Infrared Sequences


### Images

![3750190c71c2bc76e43ec0524195ceb0c7abdf433a9aa63d58768fdb3cbd58e6.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/images/3750190c71c2bc76e43ec0524195ceb0c7abdf433a9aa63d58768fdb3cbd58e6.jpg)

![666895b1674eb8d634c73164f735efd267b9e55598dd5396e9999a16d5d32a5e.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/images/666895b1674eb8d634c73164f735efd267b9e55598dd5396e9999a16d5d32a5e.jpg)

![669ce58b527472ae3bf4d2ab5901b02c970342641095b725c8c9512cfadc9262.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/images/669ce58b527472ae3bf4d2ab5901b02c970342641095b725c8c9512cfadc9262.jpg)

![689e13a88ac356f7cc748c22641e40ee078256bdbac47b766ce825f14da052ec.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/images/689e13a88ac356f7cc748c22641e40ee078256bdbac47b766ce825f14da052ec.jpg)

![d67cdbee7d8670d41bd568b31b88dbeef738d6d6ca80dbce0962492fbd133bb8.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/images/d67cdbee7d8670d41bd568b31b88dbeef738d6d6ca80dbce0962492fbd133bb8.jpg)

![f9626c2a9607e84b599ceecbd41c0fe277219705168f8e1a270a11f3934c72e4.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/images/f9626c2a9607e84b599ceecbd41c0fe277219705168f8e1a270a11f3934c72e4.jpg)

### Tables

![06437c4b0e1dbed5dfc6b5380fc8e36e29ec1b656bad001891d8440c180c6337.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/06437c4b0e1dbed5dfc6b5380fc8e36e29ec1b656bad001891d8440c180c6337.jpg)

![11c97172c6a4ed2549cdb494acd1ad37c7f600a74dabae25959c35bcaadfdd40.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/11c97172c6a4ed2549cdb494acd1ad37c7f600a74dabae25959c35bcaadfdd40.jpg)

![1ab8dde1b84d98f1715fdf81aaf0e4f952214245e6a611f07c683535d39e7b74.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/1ab8dde1b84d98f1715fdf81aaf0e4f952214245e6a611f07c683535d39e7b74.jpg)

![46bb3dece8f97d241bfa8bc62e727fc34dd9ecb6962f8e7d9c24a4dc7664b2ac.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/46bb3dece8f97d241bfa8bc62e727fc34dd9ecb6962f8e7d9c24a4dc7664b2ac.jpg)

![4989e3ad062827bf77f322f8323670b8260f41196a85d3d92ffb11131a5f0cd6.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/4989e3ad062827bf77f322f8323670b8260f41196a85d3d92ffb11131a5f0cd6.jpg)

![5212598939424280f31caa758c12ff6329bfc00ea754fd30809b8ef3eabb6393.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/5212598939424280f31caa758c12ff6329bfc00ea754fd30809b8ef3eabb6393.jpg)

![53e46afed96dbf0e02f27cd582324778dfb8966c51d7ca645a3fc48020d86589.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/53e46afed96dbf0e02f27cd582324778dfb8966c51d7ca645a3fc48020d86589.jpg)

![91459b38105963c4f1b6134b50223be571f3761b9d078d44ac6ad535e1cb159f.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/91459b38105963c4f1b6134b50223be571f3761b9d078d44ac6ad535e1cb159f.jpg)

![9e3629205865e8fce1a3ef3c230dc21737e4e8eaa15557942d4f97d9f306dfef.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/9e3629205865e8fce1a3ef3c230dc21737e4e8eaa15557942d4f97d9f306dfef.jpg)

![ccbf6fcbf8c9f65c056e7de3debeb9de74224115740c1a56ef0e2ff205b210c2.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/ccbf6fcbf8c9f65c056e7de3debeb9de74224115740c1a56ef0e2ff205b210c2.jpg)

![cd79d511703caf715499d9587fdd6ab85bb9c0dd92c3b2e9e6bedb151307d409.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/cd79d511703caf715499d9587fdd6ab85bb9c0dd92c3b2e9e6bedb151307d409.jpg)

![d8ec0eb553a8d2f902c1c35c7834140d41a0ad843119af82586a2c85833969de.jpg](../icml_results/3411_CodeMamba_%20Shifting%20from%20Target%20Semantics%20to%20Self-Supervised%20Background%20Manifold%20Learning%20for%20Singul/tables/d8ec0eb553a8d2f902c1c35c7834140d41a0ad843119af82586a2c85833969de.jpg)

<a id="paper-3372"></a>

## Scaling Agentic Verifier for Competitive Coding


### Images

![578838ca2e26c7332ab0c0f78b0d7b1b56772b1e27e2591cbea610d7d73a1c35.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/images/578838ca2e26c7332ab0c0f78b0d7b1b56772b1e27e2591cbea610d7d73a1c35.jpg)

![65bbfea63076097ccfca7509456d9ef07d244bfeb7c81bada0d2cdcdcdcdb9e8.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/images/65bbfea63076097ccfca7509456d9ef07d244bfeb7c81bada0d2cdcdcdcdb9e8.jpg)

![9e0584c5e038bf1c3e695e13b258cf19494a00919b63aa9a9e3b4d4f2ec59950.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/images/9e0584c5e038bf1c3e695e13b258cf19494a00919b63aa9a9e3b4d4f2ec59950.jpg)

![c0474d113e4b9b7e38fe54b5daf581f31efc65f053cc01f96d009c32e1d740f7.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/images/c0474d113e4b9b7e38fe54b5daf581f31efc65f053cc01f96d009c32e1d740f7.jpg)

![df6dcaf7a76b6d6ac45e67c20ea8c4169b6fd8859113a6d5588de8f7fa6210fc.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/images/df6dcaf7a76b6d6ac45e67c20ea8c4169b6fd8859113a6d5588de8f7fa6210fc.jpg)

![eb0cc4578cda778a63a28657c25f4dada0cba6622b7d6b40386d1bc6e1f67bcf.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/images/eb0cc4578cda778a63a28657c25f4dada0cba6622b7d6b40386d1bc6e1f67bcf.jpg)

### Tables

![09027ca09c91c628a80cd8785de7696f5a476a375ca3021b26eb1b5e598c070b.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/09027ca09c91c628a80cd8785de7696f5a476a375ca3021b26eb1b5e598c070b.jpg)

![6018df8f616f9fca72d0588e216c97cc55a893b24d92511a31ce125fa84d0cc9.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/6018df8f616f9fca72d0588e216c97cc55a893b24d92511a31ce125fa84d0cc9.jpg)

![65d2b11897f07f54bea4745516f39a24f53dfb80775170050f801e38e32c0264.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/65d2b11897f07f54bea4745516f39a24f53dfb80775170050f801e38e32c0264.jpg)

![8c02aa7bb91a0499f19d652577bc77bb0dde4f02848c8be9f76842ed52b2a0f8.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/8c02aa7bb91a0499f19d652577bc77bb0dde4f02848c8be9f76842ed52b2a0f8.jpg)

![d1eda6bc06d1b647877c4f8151c9e1b1fa318d19cbf17aa5aee50f8635619bcf.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/d1eda6bc06d1b647877c4f8151c9e1b1fa318d19cbf17aa5aee50f8635619bcf.jpg)

![d51216bd3a81d4bb72391abdd66c3a3f6fef023696b91427fb44da94ef5db0ee.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/d51216bd3a81d4bb72391abdd66c3a3f6fef023696b91427fb44da94ef5db0ee.jpg)

![eb1b53ca1b25939444e033aa34fb0a4b5cdfa0a2c7e6063c05897c483017895b.jpg](../icml_results/3412_Scaling%20Agentic%20Verifier%20for%20Competitive%20Coding/tables/eb1b53ca1b25939444e033aa34fb0a4b5cdfa0a2c7e6063c05897c483017895b.jpg)

<a id="paper-3373"></a>

## Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers


### Images

![1dbd607c3b50ea6129dcd3dfcdefb6d6244371c38f485c31d3b6a2db2c082169.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/1dbd607c3b50ea6129dcd3dfcdefb6d6244371c38f485c31d3b6a2db2c082169.jpg)

![305d7526f50d7c37ef313d5f70ecfea38c0a218bb6502255bf5c884608baffe3.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/305d7526f50d7c37ef313d5f70ecfea38c0a218bb6502255bf5c884608baffe3.jpg)

![4b00afa8419f82352cf1d193bb708906d762e439baed4030e53fa3fb5f5c4bc0.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/4b00afa8419f82352cf1d193bb708906d762e439baed4030e53fa3fb5f5c4bc0.jpg)

![539b83526c929155ca887e5dd383433fb4cb25da068c92d9a8e475aa70e43b45.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/539b83526c929155ca887e5dd383433fb4cb25da068c92d9a8e475aa70e43b45.jpg)

![5cc64fea9c821dd841d34b961a581d388da72005360e657aac4db91b0df5bda3.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/5cc64fea9c821dd841d34b961a581d388da72005360e657aac4db91b0df5bda3.jpg)

![61ddfe1cf3d3fa566b1bfc5506729bb511d2075ec15c2ff89b566396aba14ffa.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/61ddfe1cf3d3fa566b1bfc5506729bb511d2075ec15c2ff89b566396aba14ffa.jpg)

![aa952051e9f7bdcf20af9b2c35e60e7fe8d6cbbb8b3f03d8e0cfcd6cacab604e.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/aa952051e9f7bdcf20af9b2c35e60e7fe8d6cbbb8b3f03d8e0cfcd6cacab604e.jpg)

![c0f1c5e75028914e211686c8244d758402c0e6fd54654042e128cb97a64ef3d6.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/c0f1c5e75028914e211686c8244d758402c0e6fd54654042e128cb97a64ef3d6.jpg)

![d898a71d2bf0de11aac5386eac00c2e217914f4b53d95de42c8cccb1c3681911.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/d898a71d2bf0de11aac5386eac00c2e217914f4b53d95de42c8cccb1c3681911.jpg)

![e4fdf591189db27d5d02c8e4cb9af10b11ca9ccd375179218bded3252460cd71.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/e4fdf591189db27d5d02c8e4cb9af10b11ca9ccd375179218bded3252460cd71.jpg)

![e656e56db0632bf59ad44076eb851b35ead51773015e27a71f2c9ebda2efb3e9.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/images/e656e56db0632bf59ad44076eb851b35ead51773015e27a71f2c9ebda2efb3e9.jpg)

### Tables

![10dfb05be3c7ffa5e1f0cbc648348e5eb8baf1591731dcbbb0d8d26d65ea9345.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/10dfb05be3c7ffa5e1f0cbc648348e5eb8baf1591731dcbbb0d8d26d65ea9345.jpg)

![12a940699bf790c1ded654c4f1a7de9fc0279a55ffde705bf34efd09ef4efac9.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/12a940699bf790c1ded654c4f1a7de9fc0279a55ffde705bf34efd09ef4efac9.jpg)

![2baf3162a9b07be45f6b4b2c422021494b94bd2b16baf441f1e4dd6bca4a0a66.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/2baf3162a9b07be45f6b4b2c422021494b94bd2b16baf441f1e4dd6bca4a0a66.jpg)

![322530b634e788608e1b4a271eba41ac8d153776e9b50c8ed9a1118680045193.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/322530b634e788608e1b4a271eba41ac8d153776e9b50c8ed9a1118680045193.jpg)

![51498363e7287cf35edcd056f752d9bf0db337ba0826d2747b60e1230305f3f9.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/51498363e7287cf35edcd056f752d9bf0db337ba0826d2747b60e1230305f3f9.jpg)

![594cd7183516f70657806d8c2227aeba494c44cff0dd6f49e5860e70d92f4629.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/594cd7183516f70657806d8c2227aeba494c44cff0dd6f49e5860e70d92f4629.jpg)

![70a54ee8e860e829afe7101d6c60bf7b6fe05099334b4b8e651d8a86fddc17a4.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/70a54ee8e860e829afe7101d6c60bf7b6fe05099334b4b8e651d8a86fddc17a4.jpg)

![83c9ac4d5130ce9d2d470c4eb5917877ed12ec74b5dca3cd927a1560d0990cba.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/83c9ac4d5130ce9d2d470c4eb5917877ed12ec74b5dca3cd927a1560d0990cba.jpg)

![ac50260c3016dd1c5a6483fc34b99241bb5e76379c7f3a96d06c56e0d52c82c7.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/ac50260c3016dd1c5a6483fc34b99241bb5e76379c7f3a96d06c56e0d52c82c7.jpg)

![ae96f0d4c0a99eb25da84695e21efbffee64ec29fe7c5d11c5c44936f0b26858.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/ae96f0d4c0a99eb25da84695e21efbffee64ec29fe7c5d11c5c44936f0b26858.jpg)

![af97d6ec89828bcd2320d8ca42b2c6b34927a3d0a5761291c51c80330f03863a.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/af97d6ec89828bcd2320d8ca42b2c6b34927a3d0a5761291c51c80330f03863a.jpg)

![b7136763a367eb27df34a698573719afc9b1e13839f448c2777fe79f92a5fa52.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/b7136763a367eb27df34a698573719afc9b1e13839f448c2777fe79f92a5fa52.jpg)

![c37710de76d4b2d7af02fce92d4585c62d3ed844bcb326b4d00a4f606f219bb3.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/c37710de76d4b2d7af02fce92d4585c62d3ed844bcb326b4d00a4f606f219bb3.jpg)

![d05e876d91ee173ac4d4c5319ad91675c453ed9f001dc9cfe1219151a294cf6c.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/d05e876d91ee173ac4d4c5319ad91675c453ed9f001dc9cfe1219151a294cf6c.jpg)

![d771d643e4d9813e25446a4e05ab3a17afab8b1682dc42a59a18f80ffcc9f6a3.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/d771d643e4d9813e25446a4e05ab3a17afab8b1682dc42a59a18f80ffcc9f6a3.jpg)

![fae09accfa59199fd4d1f77264dbf4dbbdbbfac20cc63d36ba3178a59afa3456.jpg](../icml_results/3413_Stabilizing%20MoE%20Reinforcement%20Learning%20by%20Aligning%20Training%20and%20Inference%20Routers/tables/fae09accfa59199fd4d1f77264dbf4dbbdbbfac20cc63d36ba3178a59afa3456.jpg)

<a id="paper-3374"></a>

## Trifuse: Enhancing Attention-Based GUI Grounding via Multimodal Fusion


### Images

![039837193e2ed636eda28efafeb99d3dd50fba2d883989cce3471fd774b49f99.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/039837193e2ed636eda28efafeb99d3dd50fba2d883989cce3471fd774b49f99.jpg)

![4312180faa01c66c350074e2fac7a7b55891235a3eddbe6d7a4348bff7a1fce8.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/4312180faa01c66c350074e2fac7a7b55891235a3eddbe6d7a4348bff7a1fce8.jpg)

![7caee2499a232c033d924c1071db93eb7f25784ca861dfb7051ed4c0a29e579b.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/7caee2499a232c033d924c1071db93eb7f25784ca861dfb7051ed4c0a29e579b.jpg)

![bf88b21719370e8a42ffe21392e65c4f5e2e0616f915c07248e8f304f27034df.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/bf88b21719370e8a42ffe21392e65c4f5e2e0616f915c07248e8f304f27034df.jpg)

![cfb8638e95824667c4f68dbfb0b021a5b14325e8c745980d9f5c2347bd467cf7.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/cfb8638e95824667c4f68dbfb0b021a5b14325e8c745980d9f5c2347bd467cf7.jpg)

![e96b8792e053779c37f6a39c062ca35b155e9c9fd6219731cf6427e743c5a23d.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/e96b8792e053779c37f6a39c062ca35b155e9c9fd6219731cf6427e743c5a23d.jpg)

![eed3d3fdadb4a4a4fd73d904939b23584b5871e3d3c8b606afd979e62c9ce103.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/images/eed3d3fdadb4a4a4fd73d904939b23584b5871e3d3c8b606afd979e62c9ce103.jpg)

### Tables

![064909fad60e1ffaf96e2030a415048306ecf2f445000e1f753f5d210ee0022d.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/064909fad60e1ffaf96e2030a415048306ecf2f445000e1f753f5d210ee0022d.jpg)

![2adbc56ea2ffdf184f92b685af39d23fd931c27ab3d1f339076f54a4e74c5bd6.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/2adbc56ea2ffdf184f92b685af39d23fd931c27ab3d1f339076f54a4e74c5bd6.jpg)

![3ca5e25e5ccad83cb3908e6aeba80fdca47ada088044a256c61132c09744ce88.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/3ca5e25e5ccad83cb3908e6aeba80fdca47ada088044a256c61132c09744ce88.jpg)

![467d338501585280b26aca2995c31a322dc4953dec5282537e0b83cb0ca54059.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/467d338501585280b26aca2995c31a322dc4953dec5282537e0b83cb0ca54059.jpg)

![51be6c2b318d9df3579c7d383b92cac02307fc7a2ae5f4110d931cd78a32b258.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/51be6c2b318d9df3579c7d383b92cac02307fc7a2ae5f4110d931cd78a32b258.jpg)

![52ebb32debf39bc1c3094f971d26ad5cdedee99185ad53418d15f757e038e6fb.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/52ebb32debf39bc1c3094f971d26ad5cdedee99185ad53418d15f757e038e6fb.jpg)

![56363e595a218dd6ef70c77a201acac158d31d01dba56589f7bad35126271669.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/56363e595a218dd6ef70c77a201acac158d31d01dba56589f7bad35126271669.jpg)

![5cb06c003f40210a67e060f992bd691e4bcca1dab05117be839f3eba0bc5c0e0.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/5cb06c003f40210a67e060f992bd691e4bcca1dab05117be839f3eba0bc5c0e0.jpg)

![6412f3e3b2f784682464088dc6e4fb0c1154254543f957b566e3659c311ecc20.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/6412f3e3b2f784682464088dc6e4fb0c1154254543f957b566e3659c311ecc20.jpg)

![71778e6af3537fa8847b23c92c058d5fbb113e104aa0dbcd9a87bc725c7dd33c.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/71778e6af3537fa8847b23c92c058d5fbb113e104aa0dbcd9a87bc725c7dd33c.jpg)

![81b9a8be033974a93530b33c3fa34743277a27ef4936ad3314bc5688b6be9fc9.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/81b9a8be033974a93530b33c3fa34743277a27ef4936ad3314bc5688b6be9fc9.jpg)

![8d18b4ae1e9ef861a37a90d69593b404ccf48915579b794f6dcf0ef7fa59329f.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/8d18b4ae1e9ef861a37a90d69593b404ccf48915579b794f6dcf0ef7fa59329f.jpg)

![94bb2dd16ccdc867aab6d08c057b7bd935428ab76357d29cb140fa62e2e7e21c.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/94bb2dd16ccdc867aab6d08c057b7bd935428ab76357d29cb140fa62e2e7e21c.jpg)

![9d34b4a022615e6adb7a3fa14b48a2428046150a53bf48e77f6ca438b988cf1a.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/9d34b4a022615e6adb7a3fa14b48a2428046150a53bf48e77f6ca438b988cf1a.jpg)

![afc0ea41109196f1ff1c2f2348cf8ce326eada675f04f98a08e9d9fbcbb4212b.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/afc0ea41109196f1ff1c2f2348cf8ce326eada675f04f98a08e9d9fbcbb4212b.jpg)

![b37c9a5bf232b4fbe5df304ceaa945edb18fd32be2ffab7bd6f18de18164497f.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/b37c9a5bf232b4fbe5df304ceaa945edb18fd32be2ffab7bd6f18de18164497f.jpg)

![b408ecf4dbdb577141bea834e4abcc45f9ff628d4d773376ac24d67b3498274f.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/b408ecf4dbdb577141bea834e4abcc45f9ff628d4d773376ac24d67b3498274f.jpg)

![c9d44f587f46032c13ef1ebaf72b45ad5168593aa02d0e76a89140d486b5be90.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/c9d44f587f46032c13ef1ebaf72b45ad5168593aa02d0e76a89140d486b5be90.jpg)

![cf86bba2f666ea7dd95b663003d0351a0437c64cdc3b7651a2d4682c695f1db5.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/cf86bba2f666ea7dd95b663003d0351a0437c64cdc3b7651a2d4682c695f1db5.jpg)

![d133266f8c1cea6f0090bf03cb6cf16878355d2e6094b832b9069d865ac93343.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/d133266f8c1cea6f0090bf03cb6cf16878355d2e6094b832b9069d865ac93343.jpg)

![da16d04f03e9c63a862c6f9201c3b3730bc3cf3d659ab4ba678b423997658f75.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/da16d04f03e9c63a862c6f9201c3b3730bc3cf3d659ab4ba678b423997658f75.jpg)

![e1416630b919ffacd60305f35476fe9f2727e37bea5b227889c4569bd96df1bf.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/e1416630b919ffacd60305f35476fe9f2727e37bea5b227889c4569bd96df1bf.jpg)

![f7467ef8e761f0e1c87c92ec0b8a92506e69dbe6ad3025422b2419ee56698f6c.jpg](../icml_results/3414_Trifuse_%20Enhancing%20Attention-Based%20GUI%20Grounding%20via%20Multimodal%20Fusion/tables/f7467ef8e761f0e1c87c92ec0b8a92506e69dbe6ad3025422b2419ee56698f6c.jpg)

<a id="paper-3375"></a>

## Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning


### Images

![0bc4fa428d9043f89051a8df15b73473defcfe82cb41f6f357df7377bce31196.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/0bc4fa428d9043f89051a8df15b73473defcfe82cb41f6f357df7377bce31196.jpg)

![2135afdc09452bed13e9e7b82d3815d9f160a3c9ad49d7e344206c2a8e1a56ab.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/2135afdc09452bed13e9e7b82d3815d9f160a3c9ad49d7e344206c2a8e1a56ab.jpg)

![253aec5645f35df9e1c09cb5e67055e1c8953c87b876975512afb4649e42cebd.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/253aec5645f35df9e1c09cb5e67055e1c8953c87b876975512afb4649e42cebd.jpg)

![265bc9c29f7f12dd7dbd0e32c94e821634614fa4eb510ffa07b80a0a76fd033f.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/265bc9c29f7f12dd7dbd0e32c94e821634614fa4eb510ffa07b80a0a76fd033f.jpg)

![5e440a5176996c368518f86601f905734ce5316fd5b2668448ab384c4a3c67a0.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/5e440a5176996c368518f86601f905734ce5316fd5b2668448ab384c4a3c67a0.jpg)

![5e5241f1418841da5310f4f7d177dd7ef16383c6d72bca755c46869cf7a6f978.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/5e5241f1418841da5310f4f7d177dd7ef16383c6d72bca755c46869cf7a6f978.jpg)

![6ac943fef8fc378f35508b44e45ee0de5bf7fc4f0f8a7c4de80c6867d9c39a6a.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/6ac943fef8fc378f35508b44e45ee0de5bf7fc4f0f8a7c4de80c6867d9c39a6a.jpg)

![75ddcf1625b445183751c0f49ec32cfbd28f7742302f57da02ad1805eaa3a291.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/75ddcf1625b445183751c0f49ec32cfbd28f7742302f57da02ad1805eaa3a291.jpg)

![792fe7e118154f750ab6867e2db9ba27b9cd9d56357e1a384431e3edc19b36a6.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/792fe7e118154f750ab6867e2db9ba27b9cd9d56357e1a384431e3edc19b36a6.jpg)

![7cd6603a10d7f7f7dbd438b1a2f16cff2b7587be4e0190349c935b38e3fb1ba4.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/7cd6603a10d7f7f7dbd438b1a2f16cff2b7587be4e0190349c935b38e3fb1ba4.jpg)

![803d939955aabf9bc5af22268a25af03625716dc967cf7532178fe76f3982a85.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/803d939955aabf9bc5af22268a25af03625716dc967cf7532178fe76f3982a85.jpg)

![84f44ee036379b13b3f2285c930f5c1b8ba8e1ba8e96a66c41b90ab163160659.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/84f44ee036379b13b3f2285c930f5c1b8ba8e1ba8e96a66c41b90ab163160659.jpg)

![9661cad1cc38ef36f9298839ca94cdf2ab756a55970bebeb7dd1407cd174ab70.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/9661cad1cc38ef36f9298839ca94cdf2ab756a55970bebeb7dd1407cd174ab70.jpg)

![9835e011c1e4acf75a2ec99fa603d97a5a49ae21df3e116c37f8b3418affdeed.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/9835e011c1e4acf75a2ec99fa603d97a5a49ae21df3e116c37f8b3418affdeed.jpg)

![a0e633b32aa930c8612571be556bea30e72da1367d6499d66359fc059b1cceb4.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/a0e633b32aa930c8612571be556bea30e72da1367d6499d66359fc059b1cceb4.jpg)

![a1002d024b8736165255e9d2cf86e3f865d7e9216a7f68122066eaf5b760d671.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/a1002d024b8736165255e9d2cf86e3f865d7e9216a7f68122066eaf5b760d671.jpg)

![ae641d2fc3c01fe52b651b8cded074eacde559463d551edaa06fb329d227dee2.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/ae641d2fc3c01fe52b651b8cded074eacde559463d551edaa06fb329d227dee2.jpg)

![bbc9ef4ab2f5b0716ae5c4ade71bb3e72544194d7a16f87dab52b20cae154f0d.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/bbc9ef4ab2f5b0716ae5c4ade71bb3e72544194d7a16f87dab52b20cae154f0d.jpg)

![d80fcb7fd1be3c8e4f5ad37e9aae1827b7d4cefe51d3aa829fc40d6070c48d6e.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/d80fcb7fd1be3c8e4f5ad37e9aae1827b7d4cefe51d3aa829fc40d6070c48d6e.jpg)

![d933424f2c9dba8e6af3af9d2b5bb28bcdd6d9ac7cc2ab4f2b105ea7279211b4.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/d933424f2c9dba8e6af3af9d2b5bb28bcdd6d9ac7cc2ab4f2b105ea7279211b4.jpg)

![e64de77b2f438aa9e796c040995853b506d2f53b5b2b3b05d34bc1720d576af8.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/images/e64de77b2f438aa9e796c040995853b506d2f53b5b2b3b05d34bc1720d576af8.jpg)

### Tables

![00680ebcce13ac3465a824e7594b32d6a9b82700d984a9654909b72390909049.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/00680ebcce13ac3465a824e7594b32d6a9b82700d984a9654909b72390909049.jpg)

![3223385cb96b90d19c1b6b941fbbefe628048eb180688cdce0766af62cff29ea.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/3223385cb96b90d19c1b6b941fbbefe628048eb180688cdce0766af62cff29ea.jpg)

![39a7838e27c2c45b09dd677d9abb03869535ad5f6dd92612f1bc6a3eae25b44a.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/39a7838e27c2c45b09dd677d9abb03869535ad5f6dd92612f1bc6a3eae25b44a.jpg)

![45bbc23939ca1dd941d0067fc1d94e4c9a4fd6774ff0e39c90a6532d65388dfb.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/45bbc23939ca1dd941d0067fc1d94e4c9a4fd6774ff0e39c90a6532d65388dfb.jpg)

![5c53a1b362d9acb1fca7a7c4a026e421d473d7dd31b4e8890342a894f2f0e654.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/5c53a1b362d9acb1fca7a7c4a026e421d473d7dd31b4e8890342a894f2f0e654.jpg)

![5fe3df17bc328d1670d0e96e91256f7c97f4f1e42620f1c6874d0daf8d2c4437.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/5fe3df17bc328d1670d0e96e91256f7c97f4f1e42620f1c6874d0daf8d2c4437.jpg)

![78c3e09712e9b66b2971d3a7e214982ebc24f302f6144b97810ece8c43d5f8e8.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/78c3e09712e9b66b2971d3a7e214982ebc24f302f6144b97810ece8c43d5f8e8.jpg)

![838ee043f1161cf3ee7d466c719720f39efa65174fb0786270c0c591ffbd0ea9.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/838ee043f1161cf3ee7d466c719720f39efa65174fb0786270c0c591ffbd0ea9.jpg)

![8c1aab16d34f894d13155f48231a8769e4de4926b186c326617858f650f3e879.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/8c1aab16d34f894d13155f48231a8769e4de4926b186c326617858f650f3e879.jpg)

![92b7a7321bdfc014f07821bec4eb202ce4d661f5abb0d899a320c8c21e9196bd.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/92b7a7321bdfc014f07821bec4eb202ce4d661f5abb0d899a320c8c21e9196bd.jpg)

![9f6a9ecb6d2878ee6bf05c2a2a3523f30eaf5d8fee263961c08ff7ff6bf77344.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/9f6a9ecb6d2878ee6bf05c2a2a3523f30eaf5d8fee263961c08ff7ff6bf77344.jpg)

![de0941708dde22c4b36e1f054489b2a4384c6635ea3c44297a4daa7bcbad4136.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/de0941708dde22c4b36e1f054489b2a4384c6635ea3c44297a4daa7bcbad4136.jpg)

![ebf2a1fc41c286a6f5b474fa7c7cf7dc73d8ab591496e968e4a584a4ad4b5f5e.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/ebf2a1fc41c286a6f5b474fa7c7cf7dc73d8ab591496e968e4a584a4ad4b5f5e.jpg)

![f30f50abfedebf1a06988650b2a3a5470c5e9bede45754533f1eea505659582c.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/f30f50abfedebf1a06988650b2a3a5470c5e9bede45754533f1eea505659582c.jpg)

![f84d95a54b4de8d00752809f34dec93eabc57a89d0415b7571c019b53d6cf7b8.jpg](../icml_results/3415_Are%20Tools%20Always%20Beneficial_%20Learning%20to%20Invoke%20Tools%20Adaptively%20for%20Dual-Mode%20Multimodal%20LLM%20Reason/tables/f84d95a54b4de8d00752809f34dec93eabc57a89d0415b7571c019b53d6cf7b8.jpg)

<a id="paper-3376"></a>

## Scalable Traffic Signal Control with Shared Policy Framework


### Images

![1a6c87d438bb909d4ab34ba6f2f0341bcddec25652d875988d5cb1373ece58ae.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/images/1a6c87d438bb909d4ab34ba6f2f0341bcddec25652d875988d5cb1373ece58ae.jpg)

![3e73815a3be3028d2d23ea7dea45653a715b604abaaa122e32a18436beff1c0a.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/images/3e73815a3be3028d2d23ea7dea45653a715b604abaaa122e32a18436beff1c0a.jpg)

![48edf2ac3329a95a1ea25320c235f7bdad2373a796d07a62568b50cb87934cc8.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/images/48edf2ac3329a95a1ea25320c235f7bdad2373a796d07a62568b50cb87934cc8.jpg)

![86cc7f9cb55c5cb83beb1d972961f025a154fc53b3848897bda3de2f9bd3f4b3.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/images/86cc7f9cb55c5cb83beb1d972961f025a154fc53b3848897bda3de2f9bd3f4b3.jpg)

![91f96f882a7cd772dae189791b4a86f918d94120b9cc5c5d693ee0f848735ee3.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/images/91f96f882a7cd772dae189791b4a86f918d94120b9cc5c5d693ee0f848735ee3.jpg)

![9cf6693d13dc3c1cb6a77432e7aaf3aae30fbeeae3ed43f78ea36cd04e1ef0e3.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/images/9cf6693d13dc3c1cb6a77432e7aaf3aae30fbeeae3ed43f78ea36cd04e1ef0e3.jpg)

### Tables

![012be93fc12ff4ffdaebfef60b6419698febbf010956a4b0855303885359cc0d.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/tables/012be93fc12ff4ffdaebfef60b6419698febbf010956a4b0855303885359cc0d.jpg)

![0f233f14b77249bb05cc83dacdd0e994d4b1e80c64d2c5c11c8ca940b2070e16.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/tables/0f233f14b77249bb05cc83dacdd0e994d4b1e80c64d2c5c11c8ca940b2070e16.jpg)

![37bc7f7ce4a0ad380f8a33ae002b31042e913c1d03a9c2719a795823729551e2.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/tables/37bc7f7ce4a0ad380f8a33ae002b31042e913c1d03a9c2719a795823729551e2.jpg)

![4f5c3ddea31c410a5e6e8674ababd5ed2f75a27b5a26321f2469314c053f9d54.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/tables/4f5c3ddea31c410a5e6e8674ababd5ed2f75a27b5a26321f2469314c053f9d54.jpg)

![dc6e5a971a051fb5010de42c540aa1fbf0133b3a1954c3f50e7d59d98b75e2e5.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/tables/dc6e5a971a051fb5010de42c540aa1fbf0133b3a1954c3f50e7d59d98b75e2e5.jpg)

![e13b9b9226cffae7e973e58973707b2873a95c6474d7b07c01b6112204ffd539.jpg](../icml_results/3416_Scalable%20Traffic%20Signal%20Control%20with%20Shared%20Policy%20Framework/tables/e13b9b9226cffae7e973e58973707b2873a95c6474d7b07c01b6112204ffd539.jpg)

<a id="paper-3377"></a>

## Mechanisms of Introspective Awareness


### Images

![04696d5d28de13b7fc47e035ce5f30381f87aff77c93e5359851b4ee2b495c6c.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/04696d5d28de13b7fc47e035ce5f30381f87aff77c93e5359851b4ee2b495c6c.jpg)

![0ca32f4a47bdeb7413132eedd0e153fdc9e8a28062e8387127050c0129cd1c22.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/0ca32f4a47bdeb7413132eedd0e153fdc9e8a28062e8387127050c0129cd1c22.jpg)

![120518241b2ffd2f5ee1a475d79951e0bb6d1597419152e665274ff96a79d464.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/120518241b2ffd2f5ee1a475d79951e0bb6d1597419152e665274ff96a79d464.jpg)

![147d6e41aaef37d97029f7ec5d704c7de0f6cd823906ce7b5cef83acfdcd5a9e.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/147d6e41aaef37d97029f7ec5d704c7de0f6cd823906ce7b5cef83acfdcd5a9e.jpg)

![22260b93b415857e1bdae46a4913216330ecc1d39d6672c8f0b3669efbb3d7d7.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/22260b93b415857e1bdae46a4913216330ecc1d39d6672c8f0b3669efbb3d7d7.jpg)

![231ac88317c525b0b371dc9e6885513bbf8f795b90e9b464dfabfe2f4757ba8d.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/231ac88317c525b0b371dc9e6885513bbf8f795b90e9b464dfabfe2f4757ba8d.jpg)

![233d18af81c5fa9d12d96fff29d9a8bd17c4322e592e33ad01fac119d11f6701.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/233d18af81c5fa9d12d96fff29d9a8bd17c4322e592e33ad01fac119d11f6701.jpg)

![3cb7502453cf13fe8acf1ef724fd2f84f6709e92e01a6d49c0ec19350f8c59cc.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/3cb7502453cf13fe8acf1ef724fd2f84f6709e92e01a6d49c0ec19350f8c59cc.jpg)

![44ec0fdd9ae60914544fab250255350ec949789abc04135eebbaf95a228dfc9a.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/44ec0fdd9ae60914544fab250255350ec949789abc04135eebbaf95a228dfc9a.jpg)

![5109c14266992494eccc340ab4e6a1f9bcec303786cff1d58871dea1b44a7239.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/5109c14266992494eccc340ab4e6a1f9bcec303786cff1d58871dea1b44a7239.jpg)

![6baf9887ee0c65f8083e95948890b6156cd35886016f7f8737f2b41aa3bb7c9f.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/6baf9887ee0c65f8083e95948890b6156cd35886016f7f8737f2b41aa3bb7c9f.jpg)

![6e302a6dc1b0d881ef9d1f180b18c48c513e2b32a2f2ad8fbbe2d6bcb17486a6.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/6e302a6dc1b0d881ef9d1f180b18c48c513e2b32a2f2ad8fbbe2d6bcb17486a6.jpg)

![74327822d744c1f58959554ce4187ba766e79a87a985476d168594c92e544435.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/74327822d744c1f58959554ce4187ba766e79a87a985476d168594c92e544435.jpg)

![74fe0d287cac539da7e882ba89db7ca58f3770583cdf3af2d5208e7f414c235f.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/74fe0d287cac539da7e882ba89db7ca58f3770583cdf3af2d5208e7f414c235f.jpg)

![7f0aaad30451cdd95ff7477cdf4dfd8325ce247af6bbb74a7b8b1045eb6251c9.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/7f0aaad30451cdd95ff7477cdf4dfd8325ce247af6bbb74a7b8b1045eb6251c9.jpg)

![8e49cb93ba2ee05957af39670ef1841761412a5f6b04e9e0e2dc8ac1a1bedf31.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/8e49cb93ba2ee05957af39670ef1841761412a5f6b04e9e0e2dc8ac1a1bedf31.jpg)

![9ad42031332ea82e0a94bc500ae5f3a22c800b049fdf185f26902c2343380ab4.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/9ad42031332ea82e0a94bc500ae5f3a22c800b049fdf185f26902c2343380ab4.jpg)

![a51d5dec7a4dd6ba495f15ef255582320b74a294406c4e1aff1fc77f4d47390f.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/a51d5dec7a4dd6ba495f15ef255582320b74a294406c4e1aff1fc77f4d47390f.jpg)

![a8b5a7a211f71e0c86f4885dd6accc055239a25615be81980a719ee244cc046f.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/a8b5a7a211f71e0c86f4885dd6accc055239a25615be81980a719ee244cc046f.jpg)

![aa5a462c33c7d8dd91475baa10425096190f03be1129b0fe5eb77b72dbac7637.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/aa5a462c33c7d8dd91475baa10425096190f03be1129b0fe5eb77b72dbac7637.jpg)

![aa617fb98fcbc354b6633b0c5a9d267acb0c6c635e1b810a807815dd9f5120b0.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/aa617fb98fcbc354b6633b0c5a9d267acb0c6c635e1b810a807815dd9f5120b0.jpg)

![b2eb8d2fa7a3e4087b18b10a80099d0cf18ebc1a60409d5432db4733deae0f44.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/b2eb8d2fa7a3e4087b18b10a80099d0cf18ebc1a60409d5432db4733deae0f44.jpg)

![c83a54b1af01b9526a7e01a921f109c57de4e1f646dfb4858c0516f5677fa3dd.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/c83a54b1af01b9526a7e01a921f109c57de4e1f646dfb4858c0516f5677fa3dd.jpg)

![d0336f25240095e26f9e942f7f27b09a473046875c9ac5a8ab82d3bc88da3b6d.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/d0336f25240095e26f9e942f7f27b09a473046875c9ac5a8ab82d3bc88da3b6d.jpg)

![dabccf17dba40af5c9c9afcaf5109914aa3849bc1c427c227f05590b1869b990.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/dabccf17dba40af5c9c9afcaf5109914aa3849bc1c427c227f05590b1869b990.jpg)

![e9411193a1176947656ced7a0ee74388627a62deb64eadb181fb5db478f589c7.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/e9411193a1176947656ced7a0ee74388627a62deb64eadb181fb5db478f589c7.jpg)

![ed648a38ca74faba62f327a9225bc94334d8fe3093602c3e0b65bb1172f029e9.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/ed648a38ca74faba62f327a9225bc94334d8fe3093602c3e0b65bb1172f029e9.jpg)

![f0f098f96577ebe378a976a8f1cb1c6eb1bbdd28a247481b5ab103d424ccc537.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/f0f098f96577ebe378a976a8f1cb1c6eb1bbdd28a247481b5ab103d424ccc537.jpg)

![f35906cf8d0bfe3ae5508898e5dbd9e870d1a8ef84698822ed26c642d7fbc3c0.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/images/f35906cf8d0bfe3ae5508898e5dbd9e870d1a8ef84698822ed26c642d7fbc3c0.jpg)

### Tables

![092edcf71cb5394713ee738985d8c20fec214a41767d3e45a1bcf7601cefcefb.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/092edcf71cb5394713ee738985d8c20fec214a41767d3e45a1bcf7601cefcefb.jpg)

![370ee800ad019745064206363b4138aebca22dcdafa4d56b5dcc5e78094aa28b.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/370ee800ad019745064206363b4138aebca22dcdafa4d56b5dcc5e78094aa28b.jpg)

![81a648ecee09e54bdf45dd8caa7734bec83a143ae8490696365d44c6cd8190f8.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/81a648ecee09e54bdf45dd8caa7734bec83a143ae8490696365d44c6cd8190f8.jpg)

![8350b5c8e02e61c9825bf15625472740319d9ab213afd6fbea936bf116583fd4.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/8350b5c8e02e61c9825bf15625472740319d9ab213afd6fbea936bf116583fd4.jpg)

![ab233d4cded85637e2057e1005826c3f4161d0fa6006a8ff7de5b4440ada0cbf.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/ab233d4cded85637e2057e1005826c3f4161d0fa6006a8ff7de5b4440ada0cbf.jpg)

![f4be1093a00e9a66aa5b1eb553071b66588acf70fecc919d628b20dcc9b36e88.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/f4be1093a00e9a66aa5b1eb553071b66588acf70fecc919d628b20dcc9b36e88.jpg)

![f88dcac294f49e04739bbc06bbb838809c1ba4c618f8f2f3d4e08fd578a3a975.jpg](../icml_results/3417_Mechanisms%20of%20Introspective%20Awareness/tables/f88dcac294f49e04739bbc06bbb838809c1ba4c618f8f2f3d4e08fd578a3a975.jpg)

<a id="paper-3378"></a>

## Characterizing the Predictive Impact of Modalities with Supervised Latent-Variable Modeling


### Images

![0d5a330955a585c2a56ad07ff0af540348fe8b7d1d2a8580d91352a0b75005e6.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/0d5a330955a585c2a56ad07ff0af540348fe8b7d1d2a8580d91352a0b75005e6.jpg)

![12c0b005264d6166bdf14a3b77c52145e94eac60bbddb67a6030851c42b623bd.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/12c0b005264d6166bdf14a3b77c52145e94eac60bbddb67a6030851c42b623bd.jpg)

![2539c9c2b17d3bcdda737bc8c316428ebe4ffaccbdda50230c9c8e705c067325.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/2539c9c2b17d3bcdda737bc8c316428ebe4ffaccbdda50230c9c8e705c067325.jpg)

![2ef54ecc5278c43a934cc389626874cf5d0537e09fde5f412a1dc3587540a7b2.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/2ef54ecc5278c43a934cc389626874cf5d0537e09fde5f412a1dc3587540a7b2.jpg)

![3694cdb0fe81a46be2826169642f8d2ca9716fb33f3e8f6f89416a1c59e19c40.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/3694cdb0fe81a46be2826169642f8d2ca9716fb33f3e8f6f89416a1c59e19c40.jpg)

![56fc2f7ae799f597a4f1de3f85313701afeac94c77f7811ed5fbd3044447c6ad.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/56fc2f7ae799f597a4f1de3f85313701afeac94c77f7811ed5fbd3044447c6ad.jpg)

![69e7c812c95dca5137cae13eb9e3ba216866049a3e870e2c438e7bd5d731f013.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/69e7c812c95dca5137cae13eb9e3ba216866049a3e870e2c438e7bd5d731f013.jpg)

![6f1269d08192a497c5134be067e76eee0ee4d43000533370c27f0527968fddb0.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/6f1269d08192a497c5134be067e76eee0ee4d43000533370c27f0527968fddb0.jpg)

![98cb5e6f06cc1815085c77e02bc02bc268b8d58a0f81f746523c1fb188473ed2.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/98cb5e6f06cc1815085c77e02bc02bc268b8d58a0f81f746523c1fb188473ed2.jpg)

![99530a1aac2813d3955aafbcfe4f5cdf64bdabde0cc41f83f58a75e16f82ddf0.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/99530a1aac2813d3955aafbcfe4f5cdf64bdabde0cc41f83f58a75e16f82ddf0.jpg)

![9abe95855281571f425e34d23e75d2d26537f96a8cd11217716131e8f7ac200f.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/9abe95855281571f425e34d23e75d2d26537f96a8cd11217716131e8f7ac200f.jpg)

![9df4be51f34313fca97e27c1dbff7b28e22f7fca38d91a76afccaafefe5ac5e5.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/9df4be51f34313fca97e27c1dbff7b28e22f7fca38d91a76afccaafefe5ac5e5.jpg)

![b174ab4db6ae6756a5dc7cfe77977a19dc40835b78d4e9529d53a48e0eb1f4c8.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/b174ab4db6ae6756a5dc7cfe77977a19dc40835b78d4e9529d53a48e0eb1f4c8.jpg)

![b76dbaeca37c94d718953b4af3a2ea8efb2b1a7785a7ef239ba4dcc9332191ba.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/b76dbaeca37c94d718953b4af3a2ea8efb2b1a7785a7ef239ba4dcc9332191ba.jpg)

![ba25fa4290a5e06d8345eeb83ea4fc1ef1a9c81d464c48c86c14aa7e5de7ad69.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/ba25fa4290a5e06d8345eeb83ea4fc1ef1a9c81d464c48c86c14aa7e5de7ad69.jpg)

![bb89042afaab37476708c4585ab78b1ae1bcb9af1e534da2f47e9fda2b9c64d4.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/bb89042afaab37476708c4585ab78b1ae1bcb9af1e534da2f47e9fda2b9c64d4.jpg)

![c0083426bcc1a7d2e4886cd91783d727de5173c8282a8fb36b10cff0ddd8f8a5.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/c0083426bcc1a7d2e4886cd91783d727de5173c8282a8fb36b10cff0ddd8f8a5.jpg)

![ca0d23b40e050f7e93f517455b46153b37ba6206bca0dfcec58ecbc034858662.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/ca0d23b40e050f7e93f517455b46153b37ba6206bca0dfcec58ecbc034858662.jpg)

![de0277220c729d30573028d6075b96373ae882d8f930720ffe5d3d08dd7ef223.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/de0277220c729d30573028d6075b96373ae882d8f930720ffe5d3d08dd7ef223.jpg)

![e6a565edfd321906782920c7e28d126b7b7cbd60cfca5a2c1305557c340f0aa8.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/e6a565edfd321906782920c7e28d126b7b7cbd60cfca5a2c1305557c340f0aa8.jpg)

![f4deec520cadc68234703706529910056a6ef3376ae55ebece68fc3e9489ce00.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/images/f4deec520cadc68234703706529910056a6ef3376ae55ebece68fc3e9489ce00.jpg)

### Tables

![6bdb2c0306232554b8d59272c5cd18979fc22e053b1b054e01621fc964966a14.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/tables/6bdb2c0306232554b8d59272c5cd18979fc22e053b1b054e01621fc964966a14.jpg)

![b34ee1e722096f19dce25abf9a90c31aa4514eb11eaa2057021c09be554675f5.jpg](../icml_results/3418_Characterizing%20the%20Predictive%20Impact%20of%20Modalities%20with%20Supervised%20Latent-Variable%20Modeling/tables/b34ee1e722096f19dce25abf9a90c31aa4514eb11eaa2057021c09be554675f5.jpg)

<a id="paper-3379"></a>

## Rethinking Thinking Tokens: LLMs as Improvement Operators


### Images

![45b769093b14dd3f2aa51f96130daa58e01dcd2ecdbf3496ba8095fe1239831a.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/45b769093b14dd3f2aa51f96130daa58e01dcd2ecdbf3496ba8095fe1239831a.jpg)

![4958a69d4a698ae2f10073e4e6fe378f2da590d97b71f06c03879a5f7a13cc4e.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/4958a69d4a698ae2f10073e4e6fe378f2da590d97b71f06c03879a5f7a13cc4e.jpg)

![6509eee1082d2df548fc830046f054cb1b68c736933e0bff44d45ef62176f601.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/6509eee1082d2df548fc830046f054cb1b68c736933e0bff44d45ef62176f601.jpg)

![6b55964a7185b1f0396f053857ac8262d783969fecb9b28f99231af828299360.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/6b55964a7185b1f0396f053857ac8262d783969fecb9b28f99231af828299360.jpg)

![88a5f013ef7df38a9d56bcdc0dc99bfcd2ad1e95627554d2afbd2a2889705650.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/88a5f013ef7df38a9d56bcdc0dc99bfcd2ad1e95627554d2afbd2a2889705650.jpg)

![899757593589fa0265d076e7cff694ddc313c6bc35f7f38b4d42e3e0d3a63035.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/899757593589fa0265d076e7cff694ddc313c6bc35f7f38b4d42e3e0d3a63035.jpg)

![b5941682253d5b7c6e5c2959f0d5311409d1b10e1a7449ae8846e7ffda53e269.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/b5941682253d5b7c6e5c2959f0d5311409d1b10e1a7449ae8846e7ffda53e269.jpg)

![da9fa4026b320e17374b596ee335b5b41ed0495759dc704e3352bea1f953fa0c.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/da9fa4026b320e17374b596ee335b5b41ed0495759dc704e3352bea1f953fa0c.jpg)

![e787edbad5efed152fb009d8f54fa84582ea6649e9f6ff708eac3811d0e173bf.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/images/e787edbad5efed152fb009d8f54fa84582ea6649e9f6ff708eac3811d0e173bf.jpg)

### Tables

![038d91efbf7b6ba60e31c6c2c34c914c128d4deed329d93f683193bebce6af9a.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/038d91efbf7b6ba60e31c6c2c34c914c128d4deed329d93f683193bebce6af9a.jpg)

![231dc36da164286f62be76307d6ca1fde186cb0049d3fe31499fa54666099531.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/231dc36da164286f62be76307d6ca1fde186cb0049d3fe31499fa54666099531.jpg)

![3b8606424c8fe59416b86d2ba6675f7440390287f777e52ffdcc2c9156fb4d32.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/3b8606424c8fe59416b86d2ba6675f7440390287f777e52ffdcc2c9156fb4d32.jpg)

![40931bbf07f070503aa4006a43d556c6d1d7d021b2a89ba6cf888f5e4be0cb7d.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/40931bbf07f070503aa4006a43d556c6d1d7d021b2a89ba6cf888f5e4be0cb7d.jpg)

![734552f0a5465e1f990ea767864d5392ad746bff22922dbbbbca243de46b13de.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/734552f0a5465e1f990ea767864d5392ad746bff22922dbbbbca243de46b13de.jpg)

![aa572cb99e103865c344881f2bd2b84749ec8223bd0c91ea19fe7dab9051e365.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/aa572cb99e103865c344881f2bd2b84749ec8223bd0c91ea19fe7dab9051e365.jpg)

![b5fd8e810191660d38808343dbb1bf0560366e35cc51af9bea14d1aa456ca50d.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/b5fd8e810191660d38808343dbb1bf0560366e35cc51af9bea14d1aa456ca50d.jpg)

![b82d288773020e9e0e7bff21a1cb9d06d53c03faacc5d60f63b48715c3dba4d0.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/b82d288773020e9e0e7bff21a1cb9d06d53c03faacc5d60f63b48715c3dba4d0.jpg)

![ba3571156aaeeb1f74a9c725e6bce982c40e0d3175b6c041036c46d605e88ec0.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/ba3571156aaeeb1f74a9c725e6bce982c40e0d3175b6c041036c46d605e88ec0.jpg)

![de369d0c658ba5715ba0dd6c970a28e307f07353fd43a9dbab591b4fee8a55d4.jpg](../icml_results/3419_Rethinking%20Thinking%20Tokens_%20LLMs%20as%20Improvement%20Operators/tables/de369d0c658ba5715ba0dd6c970a28e307f07353fd43a9dbab591b4fee8a55d4.jpg)

<a id="paper-3380"></a>

## Inference-time optimization for experiment-grounded protein ensemble generation


### Images

![26b81e08c12158a9bb96c88858913079cdd0cb97a675db63ba8cfc1d79bafcff.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/26b81e08c12158a9bb96c88858913079cdd0cb97a675db63ba8cfc1d79bafcff.jpg)

![290ffd5291e535f4eb709039096aeddcac0ada893cd2f42483a78d7b105442ea.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/290ffd5291e535f4eb709039096aeddcac0ada893cd2f42483a78d7b105442ea.jpg)

![2ab382f250221ae0f0b49a31cf56c53537be0415d649d23ac12baf2c934d3da7.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/2ab382f250221ae0f0b49a31cf56c53537be0415d649d23ac12baf2c934d3da7.jpg)

![4635931e63fa0d16343ca283b511475a587369e4cc207b8716f21201d137285f.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/4635931e63fa0d16343ca283b511475a587369e4cc207b8716f21201d137285f.jpg)

![5d44457b0633eb5c421215dbafbb95b8bdf662d7d02057dae757659ed3fb8d41.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/5d44457b0633eb5c421215dbafbb95b8bdf662d7d02057dae757659ed3fb8d41.jpg)

![60150362e5be0d5b0c59aa2651a0fe72b2918e9b6aba654baad2f4ca71502585.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/60150362e5be0d5b0c59aa2651a0fe72b2918e9b6aba654baad2f4ca71502585.jpg)

![637c8d9c99720082d963cc54a17d97baac24f6433ef032c09c505a6d5e50d52c.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/637c8d9c99720082d963cc54a17d97baac24f6433ef032c09c505a6d5e50d52c.jpg)

![8744bd93161714f1382d1e644a387eb07cdb55b3629c6f9c705e5e75b6ba3f85.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/8744bd93161714f1382d1e644a387eb07cdb55b3629c6f9c705e5e75b6ba3f85.jpg)

![91a4ac89e89c3daa7c00f39e31407a3be9a7a0810b1c3ef2d586635e316c4904.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/91a4ac89e89c3daa7c00f39e31407a3be9a7a0810b1c3ef2d586635e316c4904.jpg)

![bae7877c55a461dd401b06bf2961696e9d4412eb192c0fc90f230eaeb8753a1c.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/bae7877c55a461dd401b06bf2961696e9d4412eb192c0fc90f230eaeb8753a1c.jpg)

![c6b079f248b0f48435d809a8275ca670e94111e86303da838795173b1b5ceac7.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/c6b079f248b0f48435d809a8275ca670e94111e86303da838795173b1b5ceac7.jpg)

![d202928edadafdf13bc010fca912c319f49d6ee6ca7b62771f3e8d563a8a1810.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/d202928edadafdf13bc010fca912c319f49d6ee6ca7b62771f3e8d563a8a1810.jpg)

![e42c961d9fa2f26b62c5e5098ad06d002c7c3fcac964240c2fecd51b91ab247a.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/e42c961d9fa2f26b62c5e5098ad06d002c7c3fcac964240c2fecd51b91ab247a.jpg)

![e6730ade27d899076e5f33785ae93f2578f747ea9595080805216f531d590e3a.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/e6730ade27d899076e5f33785ae93f2578f747ea9595080805216f531d590e3a.jpg)

![f4f1f38102668ee4d3adff57446b58fd51897270a61ad784b8d3de00080ce04b.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/images/f4f1f38102668ee4d3adff57446b58fd51897270a61ad784b8d3de00080ce04b.jpg)

### Tables

![04f14d225a5aa64f78cf868dc028a59b86c8ce80bf30a237863fe446e7e62207.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/04f14d225a5aa64f78cf868dc028a59b86c8ce80bf30a237863fe446e7e62207.jpg)

![0f7ae3555e106b659113aca6f35a7ed93cfbe28bbe01cf0e326054c6e519fb5d.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/0f7ae3555e106b659113aca6f35a7ed93cfbe28bbe01cf0e326054c6e519fb5d.jpg)

![116099f19e50b48d0a47e3cacc23f6bd08a23b4f2700dd5b41d6c9562aa15341.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/116099f19e50b48d0a47e3cacc23f6bd08a23b4f2700dd5b41d6c9562aa15341.jpg)

![35f03c037417a2691ca3eb4ad7ccd9851e940850dd5f556e531856bc15cca48e.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/35f03c037417a2691ca3eb4ad7ccd9851e940850dd5f556e531856bc15cca48e.jpg)

![36e6a501874e4d7986b7d80192d0c2497ef0e694aefbfe7df0c21eabd03307ed.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/36e6a501874e4d7986b7d80192d0c2497ef0e694aefbfe7df0c21eabd03307ed.jpg)

![4842be152f40bb232b8ee836468b6c80d84fcf4b07df0807864b1135b8695afd.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/4842be152f40bb232b8ee836468b6c80d84fcf4b07df0807864b1135b8695afd.jpg)

![4950090c3cb3b9b1acd6d6e3df378ebb3f27560eed0bd9d7f0d756a44e6daa0e.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/4950090c3cb3b9b1acd6d6e3df378ebb3f27560eed0bd9d7f0d756a44e6daa0e.jpg)

![5dd3633c76bc3fbbb02c5f91a7d57b5262abea5a4f2e9787c57d049fe7fcf0fb.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/5dd3633c76bc3fbbb02c5f91a7d57b5262abea5a4f2e9787c57d049fe7fcf0fb.jpg)

![6ac29e40a43aa5117b2aa8eb354626d86a55e7aa04dbdc555d543821d4291c5e.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/6ac29e40a43aa5117b2aa8eb354626d86a55e7aa04dbdc555d543821d4291c5e.jpg)

![6f8fea1dd9623e3470a49696cf2719ae54c684db152e7d681306908a1627cedc.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/6f8fea1dd9623e3470a49696cf2719ae54c684db152e7d681306908a1627cedc.jpg)

![7f7da7237431e541cf2df48bb3c8d1144c840d59f1cd392cdc1b897e003e8df4.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/7f7da7237431e541cf2df48bb3c8d1144c840d59f1cd392cdc1b897e003e8df4.jpg)

![814d3aaae780a98ab788134258e81df0f582e0cbf4a1d0e416949305b84c0718.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/814d3aaae780a98ab788134258e81df0f582e0cbf4a1d0e416949305b84c0718.jpg)

![8259d931f81d409cc4c8c4c6fe80ca291a19b71e4d18025f291a78380d25f5b3.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/8259d931f81d409cc4c8c4c6fe80ca291a19b71e4d18025f291a78380d25f5b3.jpg)

![a71b07575215f2cf3bc52c2d9bef101a37f93f2a595b9c174ece0b24184b2865.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/a71b07575215f2cf3bc52c2d9bef101a37f93f2a595b9c174ece0b24184b2865.jpg)

![aed60a7aa0c80b01b7e6ffd54be48769e4d47bc5247b568cd0151df5821c2b2f.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/aed60a7aa0c80b01b7e6ffd54be48769e4d47bc5247b568cd0151df5821c2b2f.jpg)

![aef3ce8cf1dc694da5556d3b9f54782e01bf822c1864bccb5bade8106e01dbea.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/aef3ce8cf1dc694da5556d3b9f54782e01bf822c1864bccb5bade8106e01dbea.jpg)

![b355ae2c281bff11cef2df2dd01e359e9fe82ffa2c157aba9aa8a7bee320d066.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/b355ae2c281bff11cef2df2dd01e359e9fe82ffa2c157aba9aa8a7bee320d066.jpg)

![b63352e41736f2a0309f9ad1aeb781b37756b9fca7f66736c1a4200d0e6ea09d.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/b63352e41736f2a0309f9ad1aeb781b37756b9fca7f66736c1a4200d0e6ea09d.jpg)

![c2e8a8062a613c1d2981c48d722c9582812035c7110b9e07b59a6fa610cbb975.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/c2e8a8062a613c1d2981c48d722c9582812035c7110b9e07b59a6fa610cbb975.jpg)

![c53d3375583619a46f7a5dcfa0db9a19ee8ad0596903ab3da3bff7a5a4c25f18.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/c53d3375583619a46f7a5dcfa0db9a19ee8ad0596903ab3da3bff7a5a4c25f18.jpg)

![d499d6a59661ca308197e741fc3a273557f0379fc23d52737b92d8115cbf5ee9.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/d499d6a59661ca308197e741fc3a273557f0379fc23d52737b92d8115cbf5ee9.jpg)

![f1600fcb3585cd91f40f5fcf707519b90f1b16bf3ff72e57514aa5f11400b9d8.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/f1600fcb3585cd91f40f5fcf707519b90f1b16bf3ff72e57514aa5f11400b9d8.jpg)

![fb6215df153601181433c10542b7195b30a51a718b61416967bdeaa2b93962cf.jpg](../icml_results/3420_Inference-time%20optimization%20for%20experiment-grounded%20protein%20ensemble%20generation/tables/fb6215df153601181433c10542b7195b30a51a718b61416967bdeaa2b93962cf.jpg)

<a id="paper-3381"></a>

## HypRAG: Hyperbolic Dense Retrieval for Retrieval Augmented Generation


### Images

![0a38204bc4e5c41f734c09a1266b9714163bed44e397b87150ef2ae89f3ab289.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/images/0a38204bc4e5c41f734c09a1266b9714163bed44e397b87150ef2ae89f3ab289.jpg)

![389422b084485061785a98c1053b25a93be89e73718c0e58e9101b16e53692ec.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/images/389422b084485061785a98c1053b25a93be89e73718c0e58e9101b16e53692ec.jpg)

![65ccbef4d79c8b13b50f4713007837157e75d127a0aed021bec17b71e44a8e7d.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/images/65ccbef4d79c8b13b50f4713007837157e75d127a0aed021bec17b71e44a8e7d.jpg)

![7bc7dfa6ccf439fa011ccd3f13ffae39149ab4cffdde9a7406ffc0aa9699e0bb.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/images/7bc7dfa6ccf439fa011ccd3f13ffae39149ab4cffdde9a7406ffc0aa9699e0bb.jpg)

### Tables

![100248fb9c49e061ee0f3b1becfd0b6c731620f0459cd9e7b082db94edee07ad.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/100248fb9c49e061ee0f3b1becfd0b6c731620f0459cd9e7b082db94edee07ad.jpg)

![1cf4810073a071e997ff5059b787e80f305f985fce10c05f3e760d87b6519990.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/1cf4810073a071e997ff5059b787e80f305f985fce10c05f3e760d87b6519990.jpg)

![20ff85251549a516c0e4225d3ed3fdcf4c8df31466171bf6f5684a60daf87cd8.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/20ff85251549a516c0e4225d3ed3fdcf4c8df31466171bf6f5684a60daf87cd8.jpg)

![52c67fe64463170e01c444ea56dd741396c55dbf4a603a3add2204911149ed90.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/52c67fe64463170e01c444ea56dd741396c55dbf4a603a3add2204911149ed90.jpg)

![52de67ba04135575715c95b996d01d2e851a6a4b8f250f01b4dd4e5d866e9aef.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/52de67ba04135575715c95b996d01d2e851a6a4b8f250f01b4dd4e5d866e9aef.jpg)

![69384355648c8595be7992c250c8a42ae5220139c466b0597014e20ab7811bb3.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/69384355648c8595be7992c250c8a42ae5220139c466b0597014e20ab7811bb3.jpg)

![74e2fb12d48fce31fd115c3c4bd6d4a28ed8d81204ab4968062236fe6bd268ef.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/74e2fb12d48fce31fd115c3c4bd6d4a28ed8d81204ab4968062236fe6bd268ef.jpg)

![835c8099eaef890e0258e6a1cf0a72ece1603483095404c95d08cfd01bdd1170.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/835c8099eaef890e0258e6a1cf0a72ece1603483095404c95d08cfd01bdd1170.jpg)

![840b6ba2547b98eba7070b47e7f2f77a525ecf4b0caccf6622f2e7e531346236.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/840b6ba2547b98eba7070b47e7f2f77a525ecf4b0caccf6622f2e7e531346236.jpg)

![899752afcfa796a2405bb499f82f25a00e291b6f7dd04642a74bd02cc377605b.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/899752afcfa796a2405bb499f82f25a00e291b6f7dd04642a74bd02cc377605b.jpg)

![973c67342c0a180e8a4a876c00df6bad78ed9b55fe863a67d25650a8cdc38ac1.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/973c67342c0a180e8a4a876c00df6bad78ed9b55fe863a67d25650a8cdc38ac1.jpg)

![9c14cdb04dfb40caedee1d7f64451003f09dfbd305512c3e5851d66f8d13bb91.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/9c14cdb04dfb40caedee1d7f64451003f09dfbd305512c3e5851d66f8d13bb91.jpg)

![b368994203e902547f869c0a3e4c9763d03cf3525d90133c105327bdf3b88806.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/b368994203e902547f869c0a3e4c9763d03cf3525d90133c105327bdf3b88806.jpg)

![d341b54fd89e9b304203b7eca89390f22f3f0c2d7a93c735bc6c8a753b5c28bb.jpg](../icml_results/3421_HypRAG_%20Hyperbolic%20Dense%20Retrieval%20for%20Retrieval%20Augmented%20Generation/tables/d341b54fd89e9b304203b7eca89390f22f3f0c2d7a93c735bc6c8a753b5c28bb.jpg)

<a id="paper-3382"></a>

## Collaborative and Efficient Fine-tuning: Leveraging Task Similarity


### Images

![13c9e3a0bc5d23abef272b7857795d9ed7c1b3e3ff9e36236f365802a00aa287.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/13c9e3a0bc5d23abef272b7857795d9ed7c1b3e3ff9e36236f365802a00aa287.jpg)

![15adb063372642f006efcacca3891bc8aebe65a398762f0b73e4a229e16e020a.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/15adb063372642f006efcacca3891bc8aebe65a398762f0b73e4a229e16e020a.jpg)

![22e24daf1f50d43b89adbeb5af64ac6052524fe2a4eac2ecb9b7091c1b62d809.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/22e24daf1f50d43b89adbeb5af64ac6052524fe2a4eac2ecb9b7091c1b62d809.jpg)

![8304c388918910794c7345667330ce77760a34790a92d3a2b895b0114dd17f19.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/8304c388918910794c7345667330ce77760a34790a92d3a2b895b0114dd17f19.jpg)

![c96232295872ff6b5ba59a64ffc1ff7d9a80b861c5087e6ff44413d344900e19.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/c96232295872ff6b5ba59a64ffc1ff7d9a80b861c5087e6ff44413d344900e19.jpg)

![cbfa551f10895885936a7a6a504443221f6a3cf84b39a30dbcd6194651d61626.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/cbfa551f10895885936a7a6a504443221f6a3cf84b39a30dbcd6194651d61626.jpg)

![ead6143b5cc4ee018ff268741240a5ade8e391a986f71e7d9e939babc4eeee36.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/ead6143b5cc4ee018ff268741240a5ade8e391a986f71e7d9e939babc4eeee36.jpg)

![f2ae323f43e5782fe3ebbb08f9a2696001fc46931f7f1d557cd7d4e606b44193.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/f2ae323f43e5782fe3ebbb08f9a2696001fc46931f7f1d557cd7d4e606b44193.jpg)

![fe2823657897e6eda26f10290290200f35c0945ed9339530c6aac44dac9c097c.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/images/fe2823657897e6eda26f10290290200f35c0945ed9339530c6aac44dac9c097c.jpg)

### Tables

![217c5127efcf6ce97dbd0eb38e29a7ff9f0f6c9ffd1855b91e3bda43f2e52b53.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/217c5127efcf6ce97dbd0eb38e29a7ff9f0f6c9ffd1855b91e3bda43f2e52b53.jpg)

![38e2e58eedb314a5661d8a5f38a197a2f837e12e58fa96dd087a91d6c1e8caf9.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/38e2e58eedb314a5661d8a5f38a197a2f837e12e58fa96dd087a91d6c1e8caf9.jpg)

![3ea29445242bef8f8acc9b02adce27c1d0bb3cac84ba6a170adff3ea376f45f4.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/3ea29445242bef8f8acc9b02adce27c1d0bb3cac84ba6a170adff3ea376f45f4.jpg)

![75271dd9fff84c46556b10c587c2a4fdeb0a576a816f94a80b254bf9798308c4.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/75271dd9fff84c46556b10c587c2a4fdeb0a576a816f94a80b254bf9798308c4.jpg)

![7a480b6fef5176f56b6b015e308671714a09c8a846acbb64824c143f919bc5bb.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/7a480b6fef5176f56b6b015e308671714a09c8a846acbb64824c143f919bc5bb.jpg)

![7a7ef43d6b1d7ab5d3729d00dd6c5f70d4c2364a28dc1789f76882ec8952161c.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/7a7ef43d6b1d7ab5d3729d00dd6c5f70d4c2364a28dc1789f76882ec8952161c.jpg)

![97e69367f052d717550344014d9815936197166cf7c8a54b30e9ee12ddb93e6e.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/97e69367f052d717550344014d9815936197166cf7c8a54b30e9ee12ddb93e6e.jpg)

![dc6b6b5a90f4aa82ab24070705b0986bd97bc8499badc6efcee550e09acd7c9b.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/dc6b6b5a90f4aa82ab24070705b0986bd97bc8499badc6efcee550e09acd7c9b.jpg)

![e94cb87d2d7a38dbe85a6feb26b8c25fd683c1d7767e6e942c79097df07a298b.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/e94cb87d2d7a38dbe85a6feb26b8c25fd683c1d7767e6e942c79097df07a298b.jpg)

![f7597e8b312685263bf641b5a1ecc17190ba15962c48e7cbfb3e0794179ba3db.jpg](../icml_results/3422_Collaborative%20and%20Efficient%20Fine-tuning_%20Leveraging%20Task%20Similarity/tables/f7597e8b312685263bf641b5a1ecc17190ba15962c48e7cbfb3e0794179ba3db.jpg)

<a id="paper-3383"></a>

## Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning


### Images

![bbebabc346e78e7a3e9448edd6ec3fcd31fab9acd6badf375cf175aa4f310134.jpg](../icml_results/3423_Convergence%20of%20Two-Timescale%20Markovian%20Stochastic%20Approximations%20with%20Applications%20in%20Reinforcement%20/images/bbebabc346e78e7a3e9448edd6ec3fcd31fab9acd6badf375cf175aa4f310134.jpg)

### Tables

![f993cd2e95279bc66f0fcc36597b396d4151951b9975ae86f81089529f6b1759.jpg](../icml_results/3423_Convergence%20of%20Two-Timescale%20Markovian%20Stochastic%20Approximations%20with%20Applications%20in%20Reinforcement%20/tables/f993cd2e95279bc66f0fcc36597b396d4151951b9975ae86f81089529f6b1759.jpg)

<a id="paper-3384"></a>

## Scaling Generative Verifiers For Natural Language Mathematical Proof Verification And Selection


### Images

![0d893b3ff8eff8d8d2a3a0f23bb74632ab94396dba6f788dd87832e54ddacf15.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/images/0d893b3ff8eff8d8d2a3a0f23bb74632ab94396dba6f788dd87832e54ddacf15.jpg)

![894074aa7a11336dd5b1ae438f57fe8c1e23be347ff829df6f9ce6730aa0f7d6.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/images/894074aa7a11336dd5b1ae438f57fe8c1e23be347ff829df6f9ce6730aa0f7d6.jpg)

![978bf2febeb9424e40d273344907fc9c13f3dbf0495dc839cee61feb524e01cb.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/images/978bf2febeb9424e40d273344907fc9c13f3dbf0495dc839cee61feb524e01cb.jpg)

![b2d6c43c5fa740ec4596048e5d674f2ae3d83dcebb4df3fd642f9329c83da6e3.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/images/b2d6c43c5fa740ec4596048e5d674f2ae3d83dcebb4df3fd642f9329c83da6e3.jpg)

![dd2f313afd36aad62bda3f46e0d1563a63ec0561106810d5d01aaca5f0b0bef8.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/images/dd2f313afd36aad62bda3f46e0d1563a63ec0561106810d5d01aaca5f0b0bef8.jpg)

### Tables

![0112b1b918af9920126636db5e24151847706845a6b19c34a28a1393cd36ddba.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/0112b1b918af9920126636db5e24151847706845a6b19c34a28a1393cd36ddba.jpg)

![04f8366cc0bbfa4f0b39c63898caf721115a120f57f8dbd0322da28b40fc08bc.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/04f8366cc0bbfa4f0b39c63898caf721115a120f57f8dbd0322da28b40fc08bc.jpg)

![3d68cff884ffa5c00ab4d9abb483de693bc0cba153414759167dc6fdc68cd381.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/3d68cff884ffa5c00ab4d9abb483de693bc0cba153414759167dc6fdc68cd381.jpg)

![42d79b5c9895f409b0f0391841a68823d19ec08184d9f461b754c03c96f37005.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/42d79b5c9895f409b0f0391841a68823d19ec08184d9f461b754c03c96f37005.jpg)

![a89c536a9d0005fb02b5d7efafca5d3db444031b6bcc2e141004cabd5cc43cc6.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/a89c536a9d0005fb02b5d7efafca5d3db444031b6bcc2e141004cabd5cc43cc6.jpg)

![ada2b47d43bc490e9c986d9d8b6097b92afd7164d14aae12aba9d2b4c682ba35.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/ada2b47d43bc490e9c986d9d8b6097b92afd7164d14aae12aba9d2b4c682ba35.jpg)

![bbbe79404ebdf33e2cb1ccdf11e16bd83d2baf65a38d5a37a95651ccb5c6d5d4.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/bbbe79404ebdf33e2cb1ccdf11e16bd83d2baf65a38d5a37a95651ccb5c6d5d4.jpg)

![c858b171cffdbaa85f6a7735e947d088ea156bf9b187bc7129892df373561da1.jpg](../icml_results/3424_Scaling%20Generative%20Verifiers%20For%20Natural%20Language%20Mathematical%20Proof%20Verification%20And%20Selection/tables/c858b171cffdbaa85f6a7735e947d088ea156bf9b187bc7129892df373561da1.jpg)

<a id="paper-3385"></a>

## Joint-Embedding Predictive Learning of Latent Market States in U.S. Equities


### Images

![1e02a12285b8a7e532a888dbf74bcb6149fc2174087b9b2fcd636f51aa4d0331.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/images/1e02a12285b8a7e532a888dbf74bcb6149fc2174087b9b2fcd636f51aa4d0331.jpg)

![323fd48dd20d15df4ebda6dd889939c408267f0a3d3141417f230f84338020e0.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/images/323fd48dd20d15df4ebda6dd889939c408267f0a3d3141417f230f84338020e0.jpg)

![4ff8888880a80f23cdf3f39d9a1fe278f7f2354e6c39e39ebbb7af26d18de295.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/images/4ff8888880a80f23cdf3f39d9a1fe278f7f2354e6c39e39ebbb7af26d18de295.jpg)

![75a4d60a0c3d4b6efc8e6f0da5ea164a660ac82843ff0acd56ccc450f95ec891.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/images/75a4d60a0c3d4b6efc8e6f0da5ea164a660ac82843ff0acd56ccc450f95ec891.jpg)

![d9452c325cd204f2a12d14debc2b7aa59987d30de420f4f61cb160d02e2c3f47.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/images/d9452c325cd204f2a12d14debc2b7aa59987d30de420f4f61cb160d02e2c3f47.jpg)

### Tables

![0a75dabf38d139911d3760461bdfd8483ec2c558546e03c4d9283693ef266abc.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/0a75dabf38d139911d3760461bdfd8483ec2c558546e03c4d9283693ef266abc.jpg)

![12450e7e1bec9522073205db3af3c6257d961791b2b4d9622960da017742c6aa.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/12450e7e1bec9522073205db3af3c6257d961791b2b4d9622960da017742c6aa.jpg)

![1d17e7477b677d15fa536e02cba2a5f1a44f60a0257c402a620189894bcdc328.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/1d17e7477b677d15fa536e02cba2a5f1a44f60a0257c402a620189894bcdc328.jpg)

![21b1fb2664a5ae67211be357ecdf8a50b7fe0ffea1477fc927ff1765aed146ad.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/21b1fb2664a5ae67211be357ecdf8a50b7fe0ffea1477fc927ff1765aed146ad.jpg)

![3b6953003d95f8f615f7e025b168f8d8a99cb4ef42c860656c81e61d22badee5.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/3b6953003d95f8f615f7e025b168f8d8a99cb4ef42c860656c81e61d22badee5.jpg)

![41b33a3f36f9a791555ee0467015ca285ed1a4d9a2d22c09fe749e0f947cd370.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/41b33a3f36f9a791555ee0467015ca285ed1a4d9a2d22c09fe749e0f947cd370.jpg)

![43be1d0c78c5ccf6bfb4a46eb632d5dd58464fa9306817196218aa031783a5d1.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/43be1d0c78c5ccf6bfb4a46eb632d5dd58464fa9306817196218aa031783a5d1.jpg)

![488711125ec04cfa2e8160eefdfbe71116f60bb767b73c1a2a1d71a597a564c8.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/488711125ec04cfa2e8160eefdfbe71116f60bb767b73c1a2a1d71a597a564c8.jpg)

![5f76a13eeb6281b790109a27c45855915ab3256b93f374797c8de4bff6d1dff4.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/5f76a13eeb6281b790109a27c45855915ab3256b93f374797c8de4bff6d1dff4.jpg)

![6680fe5dc34f4084e1c204c6f71b4fb01810f1fa8a459d6ebcec6aa668dfd3d8.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/6680fe5dc34f4084e1c204c6f71b4fb01810f1fa8a459d6ebcec6aa668dfd3d8.jpg)

![68c1b3d222fc9ed2a20dd980a7de92af1301998bf0afd7f0d4b77a253a58ccc0.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/68c1b3d222fc9ed2a20dd980a7de92af1301998bf0afd7f0d4b77a253a58ccc0.jpg)

![6e4f727d73d5e2a411a7ce5d672698dd9e339312133fbc321946518e26e22f0f.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/6e4f727d73d5e2a411a7ce5d672698dd9e339312133fbc321946518e26e22f0f.jpg)

![8e3d99052e0ac84acb03e0a6607411ea5ea8264a40ba320c2b5df385f4d49c0d.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/8e3d99052e0ac84acb03e0a6607411ea5ea8264a40ba320c2b5df385f4d49c0d.jpg)

![a6c20773a0172210e070a84c6794012e8e61b5d3275be39bf0d857d12e837995.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/a6c20773a0172210e070a84c6794012e8e61b5d3275be39bf0d857d12e837995.jpg)

![abc0077bbb354d61de5da703bd508021957f742a2405aa018bc5c01f1ed71cb8.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/abc0077bbb354d61de5da703bd508021957f742a2405aa018bc5c01f1ed71cb8.jpg)

![b9943fbcf9d0d9ba6135e9c943be6391836bff510c5aed21176c71d177218bb4.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/b9943fbcf9d0d9ba6135e9c943be6391836bff510c5aed21176c71d177218bb4.jpg)

![bb54d688b0ab4d5ded67f0af5d64c60b2928d515b1d82630b83d1abdd45e6ead.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/bb54d688b0ab4d5ded67f0af5d64c60b2928d515b1d82630b83d1abdd45e6ead.jpg)

![c8917e53f61a948d1d39ec87b907b600195e7f8b19d2aad99588317913b3d17c.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/c8917e53f61a948d1d39ec87b907b600195e7f8b19d2aad99588317913b3d17c.jpg)

![cdbcf52a9903ffdceac7871538e935b476cf8510a429836460544bd323c281f5.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/cdbcf52a9903ffdceac7871538e935b476cf8510a429836460544bd323c281f5.jpg)

![d5a408c423cb376769a9b366d772e7da6fe32744f55f2661f970cc9294e98d18.jpg](../icml_results/3425_Joint-Embedding%20Predictive%20Learning%20of%20Latent%20Market%20States%20in%20U.S.%20Equities/tables/d5a408c423cb376769a9b366d772e7da6fe32744f55f2661f970cc9294e98d18.jpg)

<a id="paper-3386"></a>

## Correcting Overparameterization Effects in Fair Empirical Risk Minimization


### Images

![3beb3bc575786978fd895a71f29c5e932b970d42ad98bf9fe9090ee24d612151.jpg](../icml_results/3426_Correcting%20Overparameterization%20Effects%20in%20Fair%20Empirical%20Risk%20Minimization/images/3beb3bc575786978fd895a71f29c5e932b970d42ad98bf9fe9090ee24d612151.jpg)

![ac2468ddf9cde1fea57d03d93c42e3659c5322ec32b64f9da307d562e2c3d561.jpg](../icml_results/3426_Correcting%20Overparameterization%20Effects%20in%20Fair%20Empirical%20Risk%20Minimization/images/ac2468ddf9cde1fea57d03d93c42e3659c5322ec32b64f9da307d562e2c3d561.jpg)

![bad7e26c618ab1daaa06e03a7d9ec3f9312554d4a931bd1a130b8c441d43d731.jpg](../icml_results/3426_Correcting%20Overparameterization%20Effects%20in%20Fair%20Empirical%20Risk%20Minimization/images/bad7e26c618ab1daaa06e03a7d9ec3f9312554d4a931bd1a130b8c441d43d731.jpg)

![f9c34c6bed86f5b923fb03edd1cc6e994d2a902904893c0f7600eb3f8970c8ca.jpg](../icml_results/3426_Correcting%20Overparameterization%20Effects%20in%20Fair%20Empirical%20Risk%20Minimization/images/f9c34c6bed86f5b923fb03edd1cc6e994d2a902904893c0f7600eb3f8970c8ca.jpg)

![fd69e024cee5f1ce196ea439d21cd0dcd1966f8e68a2be72ab7e5803bff33d4a.jpg](../icml_results/3426_Correcting%20Overparameterization%20Effects%20in%20Fair%20Empirical%20Risk%20Minimization/images/fd69e024cee5f1ce196ea439d21cd0dcd1966f8e68a2be72ab7e5803bff33d4a.jpg)

<a id="paper-3387"></a>

## Revisiting the Bertrand Paradox via Equilibrium Analysis of No-regret Learners


### Images

![0542b42ee05c47a1636535fa90df620fbe5e26ecbef28d93226f0e8e953861d1.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/0542b42ee05c47a1636535fa90df620fbe5e26ecbef28d93226f0e8e953861d1.jpg)

![05f2e4bdaf872b279448f23a50f9394cd1d02354d48479817bb35039b2a4fe75.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/05f2e4bdaf872b279448f23a50f9394cd1d02354d48479817bb35039b2a4fe75.jpg)

![0713b29d4981a92688b118df78f853bdb40db5089183c8cfacb9631f67a6c29e.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/0713b29d4981a92688b118df78f853bdb40db5089183c8cfacb9631f67a6c29e.jpg)

![082f098702f587a578dd90ad889c1491173e8e3b03d9d0cf2ef0865fb5a19bf6.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/082f098702f587a578dd90ad889c1491173e8e3b03d9d0cf2ef0865fb5a19bf6.jpg)

![09e601813982d8bbdfd80d533464e67ae6900a120d4d3e506c3923e0fe64420c.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/09e601813982d8bbdfd80d533464e67ae6900a120d4d3e506c3923e0fe64420c.jpg)

![0c6f774dac80c19a60b3ef9722063edb73dd0e0a197d029b40f262ac76dba93f.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/0c6f774dac80c19a60b3ef9722063edb73dd0e0a197d029b40f262ac76dba93f.jpg)

![127782e8dc35a226fc9bd313d0f959ee1e16edecd1b63424680f893d044a6fe2.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/127782e8dc35a226fc9bd313d0f959ee1e16edecd1b63424680f893d044a6fe2.jpg)

![146a7b529044898aa7d3cc583ea43d81a6e9bfccee5cf8400c515210092dad72.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/146a7b529044898aa7d3cc583ea43d81a6e9bfccee5cf8400c515210092dad72.jpg)

![17e456a4c5837e9c69353f67c96716fe03afb381daefea8f4c25f85bc2297050.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/17e456a4c5837e9c69353f67c96716fe03afb381daefea8f4c25f85bc2297050.jpg)

![1834989ffe859c57b4d6b8bfddf73ef758a02cfd7fd524ab806c6456c63807ed.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/1834989ffe859c57b4d6b8bfddf73ef758a02cfd7fd524ab806c6456c63807ed.jpg)

![1cddeb9f63ca34c079eb1303c4f1d625d78b411bb9358625606a2b75838b4cfc.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/1cddeb9f63ca34c079eb1303c4f1d625d78b411bb9358625606a2b75838b4cfc.jpg)

![2a224dfa47ac74df65142a28597b8b6fa4bcc77c8161022bf16de2b25803cd6d.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/2a224dfa47ac74df65142a28597b8b6fa4bcc77c8161022bf16de2b25803cd6d.jpg)

![2e9295fe21edbb04f33c0cd066ff3932029fb29820bcb6aaaf00edbac10f585c.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/2e9295fe21edbb04f33c0cd066ff3932029fb29820bcb6aaaf00edbac10f585c.jpg)

![345351ae37a32b198cea4b5f6cab21320af1c79327a9d058876101193f98ec52.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/345351ae37a32b198cea4b5f6cab21320af1c79327a9d058876101193f98ec52.jpg)

![37b0445c5db0ca9be14155112a0fdf9e5b0dd02f29fe02f60d31011aa42fcdc6.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/37b0445c5db0ca9be14155112a0fdf9e5b0dd02f29fe02f60d31011aa42fcdc6.jpg)

![3e2f69fe2cd27489f3a737fe432e1298988332175f07e96c2bf35fb5b2d4368f.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/3e2f69fe2cd27489f3a737fe432e1298988332175f07e96c2bf35fb5b2d4368f.jpg)

![4092f0b219f7a5f9cff6824ec016b15bc5ae0cf2ed96e97963fd20bc29353b43.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/4092f0b219f7a5f9cff6824ec016b15bc5ae0cf2ed96e97963fd20bc29353b43.jpg)

![4188cd1be2a660f9d47671218836f2a231cf125ad0ee1a657265dc1757f3acc2.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/4188cd1be2a660f9d47671218836f2a231cf125ad0ee1a657265dc1757f3acc2.jpg)

![494581b64d22d2ebf2e6c2ef5d505ce819ff546f380fc86c97820871e5c9b6bd.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/494581b64d22d2ebf2e6c2ef5d505ce819ff546f380fc86c97820871e5c9b6bd.jpg)

![4dc9e6a0cba963c722dc2131b0de49c8dbe3bcef87cb2b39feb08ba70d24d87e.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/4dc9e6a0cba963c722dc2131b0de49c8dbe3bcef87cb2b39feb08ba70d24d87e.jpg)

![5d29445ab2ee58d47800261e620b1475c19942750d3a1eb436db7672fe64aaa5.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/5d29445ab2ee58d47800261e620b1475c19942750d3a1eb436db7672fe64aaa5.jpg)

![63ab72fc810155860744610acdb0b12660452211a7b8c69dd16cc93864496c78.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/63ab72fc810155860744610acdb0b12660452211a7b8c69dd16cc93864496c78.jpg)

![67bb62bf97f50791951b5590246d0e69465a2575e836abb6502717e873b40df3.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/67bb62bf97f50791951b5590246d0e69465a2575e836abb6502717e873b40df3.jpg)

![68db16ecaa07fd2039d689af075fdf0a821089a3a05f625e8fd9c54963767618.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/68db16ecaa07fd2039d689af075fdf0a821089a3a05f625e8fd9c54963767618.jpg)

![6cd34a070a825c3c464d6fac35358c5e8d15cb655ccbecf46c130085e2d54d03.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/6cd34a070a825c3c464d6fac35358c5e8d15cb655ccbecf46c130085e2d54d03.jpg)

![70d3c37733c3ac4d34dcefd7dc954c7a842426b177f2a0291fc9e47d28ca749b.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/70d3c37733c3ac4d34dcefd7dc954c7a842426b177f2a0291fc9e47d28ca749b.jpg)

![75467ea59e1604008251e6bef3d1ee40013b15771d688bd5b357e4853942d156.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/75467ea59e1604008251e6bef3d1ee40013b15771d688bd5b357e4853942d156.jpg)

![8912fc131e27aba741dc5323275acf81f0974795d3e498b3b7d0c9baa025eb76.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/8912fc131e27aba741dc5323275acf81f0974795d3e498b3b7d0c9baa025eb76.jpg)

![8fca6ad194496ca70266804b9462dbc1aa267caa4c77f440b9da767af78bc16a.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/8fca6ad194496ca70266804b9462dbc1aa267caa4c77f440b9da767af78bc16a.jpg)

![935f33bf5ac13d9f60aef981bf62103300e9bd4fe071122a627db05cfec99ab0.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/935f33bf5ac13d9f60aef981bf62103300e9bd4fe071122a627db05cfec99ab0.jpg)

![985f6c890629e0b8050f223483c73dd312e220be17fd0fcf1aca04f69088e154.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/985f6c890629e0b8050f223483c73dd312e220be17fd0fcf1aca04f69088e154.jpg)

![9afe88d8e3f0358ecfc97b386fdeada5e177f9a63f4cfbfb37949cdb796d01a7.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/9afe88d8e3f0358ecfc97b386fdeada5e177f9a63f4cfbfb37949cdb796d01a7.jpg)

![9d52dbef2d6b195594587d045b997d6bb4f8f7cac49783235b841f379a3358f1.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/9d52dbef2d6b195594587d045b997d6bb4f8f7cac49783235b841f379a3358f1.jpg)

![9e30a42741e2099103525554fc7be7f350f5f0dc987a60d9f0e1759e56ed508e.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/9e30a42741e2099103525554fc7be7f350f5f0dc987a60d9f0e1759e56ed508e.jpg)

![a5e63be8d402bfc510de5c3f2e79360e6dabbb659da54b0be28508efec8ca440.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/a5e63be8d402bfc510de5c3f2e79360e6dabbb659da54b0be28508efec8ca440.jpg)

![a7cbed69025abf50a1925d81d71752e508d703aadeebc17cc74fae155da03002.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/a7cbed69025abf50a1925d81d71752e508d703aadeebc17cc74fae155da03002.jpg)

![a8d1840a3bc16c1836d2b0592b0ec5ef27936560cb378b486a61a44d5c6359d3.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/a8d1840a3bc16c1836d2b0592b0ec5ef27936560cb378b486a61a44d5c6359d3.jpg)

![bb0b7bf8e0b67260fab9fe6c98081943e0a8d3ea5e11167b5f834005802f6560.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/bb0b7bf8e0b67260fab9fe6c98081943e0a8d3ea5e11167b5f834005802f6560.jpg)

![c111099a2315f87ca62b873c794dca52af84b6229151f4cd6f7f65fd0e2e3f45.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/c111099a2315f87ca62b873c794dca52af84b6229151f4cd6f7f65fd0e2e3f45.jpg)

![da06da3e4f64308232018953a285bb9c57c3efc88bde9b16ffc0747d2ea4b33c.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/da06da3e4f64308232018953a285bb9c57c3efc88bde9b16ffc0747d2ea4b33c.jpg)

![deb5c6c1ff216217f5c59eee748211006327741209efa5e3c88acdd0458f9299.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/deb5c6c1ff216217f5c59eee748211006327741209efa5e3c88acdd0458f9299.jpg)

![e8c5dcf7b01dfbccb2ddbd1dbf1e4642192f2e7e6e05b91e25bb2bc8304d0ceb.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/e8c5dcf7b01dfbccb2ddbd1dbf1e4642192f2e7e6e05b91e25bb2bc8304d0ceb.jpg)

![e96777c7bfa5c63c9655bd8fd2715b0fc389a7fcebf70fa057c694e045dc028d.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/e96777c7bfa5c63c9655bd8fd2715b0fc389a7fcebf70fa057c694e045dc028d.jpg)

![ea3dfd08891434e8c5a598160646eec9476dc84f0c315476e7872edf937f7527.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/ea3dfd08891434e8c5a598160646eec9476dc84f0c315476e7872edf937f7527.jpg)

![edc1ea1eedd8638ea76412c46a8bf9fa8398cccb867d0ca8fe1471a337230b37.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/edc1ea1eedd8638ea76412c46a8bf9fa8398cccb867d0ca8fe1471a337230b37.jpg)

![efcca55b027449841e1fbc780b6541308a7d12a05094628294f19cc2095d9c50.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/efcca55b027449841e1fbc780b6541308a7d12a05094628294f19cc2095d9c50.jpg)

![f2b5ec38ada7fd1c349e266b552fe7be35f0843daddf2931d34b67ca2bd06084.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f2b5ec38ada7fd1c349e266b552fe7be35f0843daddf2931d34b67ca2bd06084.jpg)

![f3ea9a28ce025c771b87ca41ba6a0d5ce1fc685f05ed49f4cc1db9b9940296d4.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f3ea9a28ce025c771b87ca41ba6a0d5ce1fc685f05ed49f4cc1db9b9940296d4.jpg)

![f4fd128f80519d0e71738dc37bd4c0486c0db1a3ddb5dfba05e5dd77fe5d0734.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f4fd128f80519d0e71738dc37bd4c0486c0db1a3ddb5dfba05e5dd77fe5d0734.jpg)

![f648148bdd10fb36c76f9e672392a5917a85e929138a73ada53f74342f08d5ba.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f648148bdd10fb36c76f9e672392a5917a85e929138a73ada53f74342f08d5ba.jpg)

![f715d8f4a31d2d9a41612c47dca8725fbdc02fec3768b35bf381c4ed58deca06.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f715d8f4a31d2d9a41612c47dca8725fbdc02fec3768b35bf381c4ed58deca06.jpg)

![f81b605a02b3f4e8a76f13d617d264c2fa8ee8720a2330bf029b00ecb0d23155.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f81b605a02b3f4e8a76f13d617d264c2fa8ee8720a2330bf029b00ecb0d23155.jpg)

![f8f9658a43badd01847d07bb2cbf4af11045c88bab9c13b779e84fbac987a691.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/f8f9658a43badd01847d07bb2cbf4af11045c88bab9c13b779e84fbac987a691.jpg)

![fb2cef76080892a8600ca2df78c88cfc08e1901b59039efd74d11de1c5ee559b.jpg](../icml_results/3427_Revisiting%20the%20Bertrand%20Paradox%20via%20Equilibrium%20Analysis%20of%20No-regret%20Learners/images/fb2cef76080892a8600ca2df78c88cfc08e1901b59039efd74d11de1c5ee559b.jpg)

<a id="paper-3388"></a>

## Corruption-Tolerant Asynchronous Q-Learning with Near-Optimal Rates


### Images

![0fa94d34e5f044f7305bc8a29af9cfc9ed1f673dc8305b72cb9e094d32b5d012.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/0fa94d34e5f044f7305bc8a29af9cfc9ed1f673dc8305b72cb9e094d32b5d012.jpg)

![2003bb60f68bf590b3a95a23a78221ffb32a000896737c000f2bb899da82fd39.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/2003bb60f68bf590b3a95a23a78221ffb32a000896737c000f2bb899da82fd39.jpg)

![3df1270b5eafd5b6d2bdec34651cfb50520b470a30ef94a0200f28a5ec08ad56.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/3df1270b5eafd5b6d2bdec34651cfb50520b470a30ef94a0200f28a5ec08ad56.jpg)

![5b6e7eced186546853f7280bfac65ed20e372f1e8960bd36e734cc2461d1c03a.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/5b6e7eced186546853f7280bfac65ed20e372f1e8960bd36e734cc2461d1c03a.jpg)

![5fb0223b4ddfe4b78e0af2e871745c4a9057e9a14daf4c59fcb20ba39258181e.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/5fb0223b4ddfe4b78e0af2e871745c4a9057e9a14daf4c59fcb20ba39258181e.jpg)

![67970bb432255c5748c16c19a0c801bd0550e3d2c7a46325b96c9ec8ea6d3b01.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/67970bb432255c5748c16c19a0c801bd0550e3d2c7a46325b96c9ec8ea6d3b01.jpg)

![71d93518a84ec01d6f719cb6e97e32958df3fbbc3e34b6cff233b64865727420.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/71d93518a84ec01d6f719cb6e97e32958df3fbbc3e34b6cff233b64865727420.jpg)

![9b861be083542d68d44951af2f96f22c7a30255d35615f08b2695c825b48eaf3.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/9b861be083542d68d44951af2f96f22c7a30255d35615f08b2695c825b48eaf3.jpg)

![9fe09378497f5673bacb2a2d497357a5503b9c17721e9e95ca6763622eddc2f1.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/9fe09378497f5673bacb2a2d497357a5503b9c17721e9e95ca6763622eddc2f1.jpg)

![b9883bfa5260c4f26173215033d3bd68f35aef6dda4c48245682a1351845d494.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/b9883bfa5260c4f26173215033d3bd68f35aef6dda4c48245682a1351845d494.jpg)

![e0b8ec90214f87cc007c9db8a9b88e53362edfb6853aa5a84578abdf0dd01c77.jpg](../icml_results/3428_Corruption-Tolerant%20Asynchronous%20Q-Learning%20with%20Near-Optimal%20Rates/images/e0b8ec90214f87cc007c9db8a9b88e53362edfb6853aa5a84578abdf0dd01c77.jpg)

<a id="paper-3389"></a>

## Reward Shaping Control Variates for Off-Policy Evaluation Under Sparse Rewards


### Images

![1b81782c4b6121eb7d60b4de42cadaf1b993acd9de4351377f4daa5594e1fc7e.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/1b81782c4b6121eb7d60b4de42cadaf1b993acd9de4351377f4daa5594e1fc7e.jpg)

![3502ad5ff7807e9e841a9ef21bad76e411706a93745ccfbe126efa2e33f4f50d.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/3502ad5ff7807e9e841a9ef21bad76e411706a93745ccfbe126efa2e33f4f50d.jpg)

![395ac78f7f5c389d5de80bb5a1b30df0cb3b3d80f1abfcbce4282230afa58d1d.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/395ac78f7f5c389d5de80bb5a1b30df0cb3b3d80f1abfcbce4282230afa58d1d.jpg)

![468605f6e474b4aebc2e08bc100d753eba3a32a02b240d954bb27bbcd967cd6b.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/468605f6e474b4aebc2e08bc100d753eba3a32a02b240d954bb27bbcd967cd6b.jpg)

![635d5b36f32f29e334e2fe9282786ea695eca3da77c7a43c73f2399bef8b27bb.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/635d5b36f32f29e334e2fe9282786ea695eca3da77c7a43c73f2399bef8b27bb.jpg)

![6f423cea44d8567297086fe5743eb821c42f1909eada0fbf91b3ce1666c4566d.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/6f423cea44d8567297086fe5743eb821c42f1909eada0fbf91b3ce1666c4566d.jpg)

![7a30c27d237c8dab7976cde8ba578f2e5823ce519f6b69810102011e84984d7a.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/7a30c27d237c8dab7976cde8ba578f2e5823ce519f6b69810102011e84984d7a.jpg)

![939e421a6db67d1b3d830e84848c77139ab20a71cd2bb8d3f85114355c89558d.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/939e421a6db67d1b3d830e84848c77139ab20a71cd2bb8d3f85114355c89558d.jpg)

![a875e51b5161c27a9eb8f86fcf393db747401ce0251aea644a3d07c09c69e78e.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/a875e51b5161c27a9eb8f86fcf393db747401ce0251aea644a3d07c09c69e78e.jpg)

![b44a4e96f846e5a0f29eed70e42c538089f8a13660675a9d5c044fbc3763d308.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/b44a4e96f846e5a0f29eed70e42c538089f8a13660675a9d5c044fbc3763d308.jpg)

![ce98eae7a1eba3926d58c188f15880d4993be2c1a72c8878ad1ebb82013e6f3f.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/ce98eae7a1eba3926d58c188f15880d4993be2c1a72c8878ad1ebb82013e6f3f.jpg)

![d087f2ca57f0d1ede5d01e1a9242214fbe432656a61d99673b1929ee067de3cf.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/d087f2ca57f0d1ede5d01e1a9242214fbe432656a61d99673b1929ee067de3cf.jpg)

![d47ac53be664a49954394cb8684cbd24146e2c73948367be592d041d395d7c1a.jpg](../icml_results/3429_Reward%20Shaping%20Control%20Variates%20for%20Off-Policy%20Evaluation%20Under%20Sparse%20Rewards/images/d47ac53be664a49954394cb8684cbd24146e2c73948367be592d041d395d7c1a.jpg)

<a id="paper-3390"></a>

## TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins


### Images

![0181f5d524b45154544b91f4bcf7b1422bc3f19782bb0620fa58bee497765355.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/0181f5d524b45154544b91f4bcf7b1422bc3f19782bb0620fa58bee497765355.jpg)

![118e24a478604976c86153376bb8ef0cc79bde739146f5aae9d2a72d9e821e49.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/118e24a478604976c86153376bb8ef0cc79bde739146f5aae9d2a72d9e821e49.jpg)

![1764803c1038c923d0bf6c89830fdeae5c291884892f7634790d41ac124a1355.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/1764803c1038c923d0bf6c89830fdeae5c291884892f7634790d41ac124a1355.jpg)

![1aacb517d2c98ca3e8f5437030d60ba6e8b6c1f6d83e1947386331a876281106.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/1aacb517d2c98ca3e8f5437030d60ba6e8b6c1f6d83e1947386331a876281106.jpg)

![1e6d12701dd1de48aad9f49bfc173a4bf3c9bb951d9e0147398ee926b611abc5.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/1e6d12701dd1de48aad9f49bfc173a4bf3c9bb951d9e0147398ee926b611abc5.jpg)

![274925d40d266e2b2eadb102016890436dc1164ece5e066f8a664d97e0a9d54f.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/274925d40d266e2b2eadb102016890436dc1164ece5e066f8a664d97e0a9d54f.jpg)

![2d29d709480f92f2a27d41635aced1d346094d07d646d88ac6090c6a83344ca8.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/2d29d709480f92f2a27d41635aced1d346094d07d646d88ac6090c6a83344ca8.jpg)

![407c0b8c4f4957a4a2654bd1a475987573d09e2678eaccd10b72c2aa75708208.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/407c0b8c4f4957a4a2654bd1a475987573d09e2678eaccd10b72c2aa75708208.jpg)

![4227ef8b3bb516bac3872e2dabdaa3cac5af005e3820118ea8146dab7b70a401.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/4227ef8b3bb516bac3872e2dabdaa3cac5af005e3820118ea8146dab7b70a401.jpg)

![5e345ebd1ebdc7d28bcbffd63db081cb1aa76b31d2de7fe8030540ea23ba7bb9.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/5e345ebd1ebdc7d28bcbffd63db081cb1aa76b31d2de7fe8030540ea23ba7bb9.jpg)

![60c8af1516fbecbbb61f42ff3df3f481115ce673e7db6ed039cdea4f57729b6f.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/60c8af1516fbecbbb61f42ff3df3f481115ce673e7db6ed039cdea4f57729b6f.jpg)

![7e75aab7ffa542c4821e4300c4e272aea50a531146d0c4da4689db8bb46712e7.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/7e75aab7ffa542c4821e4300c4e272aea50a531146d0c4da4689db8bb46712e7.jpg)

![877d4b01594659453a733e84434ec6e0df50ffd56a26484e56dee6cf72b60458.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/877d4b01594659453a733e84434ec6e0df50ffd56a26484e56dee6cf72b60458.jpg)

![cab6d5007b089f7b6ae348c3fbb096c3aa5ad6a8c071723bb1daf5a92f5b953b.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/cab6d5007b089f7b6ae348c3fbb096c3aa5ad6a8c071723bb1daf5a92f5b953b.jpg)

![cd07f233cc5834f2d0e51397995510af8fb0e527d6546c1bab4d1da7e8606804.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/cd07f233cc5834f2d0e51397995510af8fb0e527d6546c1bab4d1da7e8606804.jpg)

![ed94a76806666fc06523bc7ebcca3859d2676bef7cda4e646fa3a3aef0fa5622.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/ed94a76806666fc06523bc7ebcca3859d2676bef7cda4e646fa3a3aef0fa5622.jpg)

![fa0044ded378a07e9cc08271844c4187e6a644065e432ccffb538a38ccb25ca3.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/fa0044ded378a07e9cc08271844c4187e6a644065e432ccffb538a38ccb25ca3.jpg)

![fb8edd4777abccb0aafac7b604684aa0ecce870a3a6adc67138b7bbd781dc293.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/images/fb8edd4777abccb0aafac7b604684aa0ecce870a3a6adc67138b7bbd781dc293.jpg)

### Tables

![03b6f5f99716e1a11f7f71fe02a9deb1c09c112c73eb4fb1a7667c5d119aabe0.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/03b6f5f99716e1a11f7f71fe02a9deb1c09c112c73eb4fb1a7667c5d119aabe0.jpg)

![643bf375ac9c6e71df3035adee336461d6872032e5b5f9e69e423a7736821a2c.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/643bf375ac9c6e71df3035adee336461d6872032e5b5f9e69e423a7736821a2c.jpg)

![74b622bb403e5c7ec610dc21b7e9861de35f04f6382a654c008b53f73b11e54b.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/74b622bb403e5c7ec610dc21b7e9861de35f04f6382a654c008b53f73b11e54b.jpg)

![96c287b0552e2512270c8815f57a445ce70d11c06596ca6c6137e8dc484c0671.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/96c287b0552e2512270c8815f57a445ce70d11c06596ca6c6137e8dc484c0671.jpg)

![a6ec806c2ffa0577ebf6460a6949782508dc9dbd996ade06c5808229f8e932d0.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/a6ec806c2ffa0577ebf6460a6949782508dc9dbd996ade06c5808229f8e932d0.jpg)

![bc4e1c25b8a23265812ca7dc4884646578201797729a3f99f2d7c8d90f4a8c43.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/bc4e1c25b8a23265812ca7dc4884646578201797729a3f99f2d7c8d90f4a8c43.jpg)

![c5f8b5de8ff5d64f51d35c1ff6be7fa761a825d226f70d95834cd2fc683da245.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/c5f8b5de8ff5d64f51d35c1ff6be7fa761a825d226f70d95834cd2fc683da245.jpg)

![e59cea64cbdf5180e2403e916b2eec9f665940749f4496aafc198b96b5ee6ff0.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/e59cea64cbdf5180e2403e916b2eec9f665940749f4496aafc198b96b5ee6ff0.jpg)

![e6498735b5bca2fb5b306206e22109a1a86f5cad13370bfb490a5cadc1b685e2.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/e6498735b5bca2fb5b306206e22109a1a86f5cad13370bfb490a5cadc1b685e2.jpg)

![fc44ffecd83878d9644e08faafb2b837d18c97de2a72530275d9b599f68ab9fa.jpg](../icml_results/3430_TwinWeaver_%20An%20LLM-Based%20Foundation%20Model%20Framework%20for%20Pan-Cancer%20Digital%20Twins/tables/fc44ffecd83878d9644e08faafb2b837d18c97de2a72530275d9b599f68ab9fa.jpg)

<a id="paper-3391"></a>

## Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings


### Images

![0fba54f8644f63261fb36acb0a5909d64c4a9c550b73de7a3e8691e5c85c0a8d.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/0fba54f8644f63261fb36acb0a5909d64c4a9c550b73de7a3e8691e5c85c0a8d.jpg)

![46cf09931a0b644fb2d056fefb1097b2c092df0988f71c451afbdd9e95b195b0.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/46cf09931a0b644fb2d056fefb1097b2c092df0988f71c451afbdd9e95b195b0.jpg)

![57cf4fafc4ca8be6a6ebc6beb4915b800ccef858ae913bacb335bd23959f0e25.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/57cf4fafc4ca8be6a6ebc6beb4915b800ccef858ae913bacb335bd23959f0e25.jpg)

![74536be126a7d07d1ef7ca1e3b4b81e80aebb9fc90babfa9e8ada4376dea1e10.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/74536be126a7d07d1ef7ca1e3b4b81e80aebb9fc90babfa9e8ada4376dea1e10.jpg)

![785dc835e9e92fd3ca03e0d40e5a6dee54f3e1ac94d48b58ac5cddb89f8e9300.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/785dc835e9e92fd3ca03e0d40e5a6dee54f3e1ac94d48b58ac5cddb89f8e9300.jpg)

![7e541afdcb71702e7bf44d36ebc08e55e8dbc8f33cc773909b960b3d9dbb37ed.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/7e541afdcb71702e7bf44d36ebc08e55e8dbc8f33cc773909b960b3d9dbb37ed.jpg)

![86eb4d739d7d00e8bce4d42ad4acf3057ca159663710abff6e04c9d51980f1f0.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/86eb4d739d7d00e8bce4d42ad4acf3057ca159663710abff6e04c9d51980f1f0.jpg)

![9733074ad3b9492136ce15a20ee3a5b3b7d74942e0b6d7e83be9ae354256a2fb.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/9733074ad3b9492136ce15a20ee3a5b3b7d74942e0b6d7e83be9ae354256a2fb.jpg)

![a3b7a64e44b9b62c75c6fa0389f7364e1f02f782e565147666edd08a96b1c86e.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/a3b7a64e44b9b62c75c6fa0389f7364e1f02f782e565147666edd08a96b1c86e.jpg)

![c2db286b573a9f1ecfd3b88e519f4f0b22c04b31802f72a46267bbe138ec4fb6.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/c2db286b573a9f1ecfd3b88e519f4f0b22c04b31802f72a46267bbe138ec4fb6.jpg)

![e347f0323a2c13fedaac03ad421fab21fa5b9be9c4cd0945bde2724c173cc25d.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/e347f0323a2c13fedaac03ad421fab21fa5b9be9c4cd0945bde2724c173cc25d.jpg)

![e8e5276114b1a663efbc2dc8f7611e42a312cda992fe1505414acfeabbed9011.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/images/e8e5276114b1a663efbc2dc8f7611e42a312cda992fe1505414acfeabbed9011.jpg)

### Tables

![1cf02fd5e5b94695b7bfad9e91f470cc80fb7b39ab85c05404a15d57066b3fd7.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/1cf02fd5e5b94695b7bfad9e91f470cc80fb7b39ab85c05404a15d57066b3fd7.jpg)

![2573a746be01d9f1d7bf9531b5ff92e2ec19ab83d67aa292e2d24e102d47133c.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/2573a746be01d9f1d7bf9531b5ff92e2ec19ab83d67aa292e2d24e102d47133c.jpg)

![371b8faab01f734a2feb2079cd0b36f1f0b386d9e650abc4844f1aa30b792cb5.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/371b8faab01f734a2feb2079cd0b36f1f0b386d9e650abc4844f1aa30b792cb5.jpg)

![529fc2432dd40c2a6813c765ee50f1939c08e9953c58388a3919cc440f2e8fb7.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/529fc2432dd40c2a6813c765ee50f1939c08e9953c58388a3919cc440f2e8fb7.jpg)

![935b92ca83d08129b52b7934caaf746099dad66699cd0d48203663ff331d8761.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/935b92ca83d08129b52b7934caaf746099dad66699cd0d48203663ff331d8761.jpg)

![9905cf773bdab78e09e1b7a78cd4a7c82c1a915650c56fbb93541ba1969d31a6.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/9905cf773bdab78e09e1b7a78cd4a7c82c1a915650c56fbb93541ba1969d31a6.jpg)

![a8a6cf914d0d258373bdfd50b1efafaca6e1b7dd7e7a49a2b24a0d5d4d20933f.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/a8a6cf914d0d258373bdfd50b1efafaca6e1b7dd7e7a49a2b24a0d5d4d20933f.jpg)

![ec01da030424a6ab7193ee1be8714e4c4731bd6f59001508aac20dcc418fc44b.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/ec01da030424a6ab7193ee1be8714e4c4731bd6f59001508aac20dcc418fc44b.jpg)

![f130f86fa6fdb36f8d27993722bb5da92efe73aff1517465b0086ac56292749e.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/f130f86fa6fdb36f8d27993722bb5da92efe73aff1517465b0086ac56292749e.jpg)

![f633bf1de57b2d1009b76eacf75b85517648eca14dd9280675847b244ffa4589.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/f633bf1de57b2d1009b76eacf75b85517648eca14dd9280675847b244ffa4589.jpg)

![f6aecc8630600f8007ad0fc2e48b3b1d256fa0f008645046b0c5b088b862c965.jpg](../icml_results/3431_Bayesian%20Meta-Learning%20with%20Expert%20Feedback%20for%20Task-Shift%20Adaptation%20through%20Causal%20Embeddings/tables/f6aecc8630600f8007ad0fc2e48b3b1d256fa0f008645046b0c5b088b862c965.jpg)

<a id="paper-3392"></a>

## MacroGuide: Topological Guidance for Macrocycle Generation


### Images

![062893aea4a71a05d63598c6f0384293e1047a3eeb61d355dc46623acc8d5765.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/062893aea4a71a05d63598c6f0384293e1047a3eeb61d355dc46623acc8d5765.jpg)

![06e023ff93a33497a02a61e78f248a3546cf94bf161bea915248e6fbae62e3e6.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/06e023ff93a33497a02a61e78f248a3546cf94bf161bea915248e6fbae62e3e6.jpg)

![173ea36c3d2ad1ce8c9d3f142c2c26c491a0bb42e0e99f6d947b7ba1284220cd.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/173ea36c3d2ad1ce8c9d3f142c2c26c491a0bb42e0e99f6d947b7ba1284220cd.jpg)

![1c08cede09dc0be3382748bf902f63c6b66833d5c6eb6dcfb8cb0b54fd62fa31.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/1c08cede09dc0be3382748bf902f63c6b66833d5c6eb6dcfb8cb0b54fd62fa31.jpg)

![3021774e6783c35c4a2fa6970470460af8324341da056081875c719dbf1756cb.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/3021774e6783c35c4a2fa6970470460af8324341da056081875c719dbf1756cb.jpg)

![3a55513af31bb7988463bb1241c2d4ee23a0844cfb3c6898da8271b242934f4b.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/3a55513af31bb7988463bb1241c2d4ee23a0844cfb3c6898da8271b242934f4b.jpg)

![3ab8a20702942aac6059716333c081aacc5515435fc6bce1b4691933fb95357d.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/3ab8a20702942aac6059716333c081aacc5515435fc6bce1b4691933fb95357d.jpg)

![3b528e03385e4795bdcf24600cbdb7ffb1f0b6b5d9e44ad73cae5de7cea3e281.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/3b528e03385e4795bdcf24600cbdb7ffb1f0b6b5d9e44ad73cae5de7cea3e281.jpg)

![43279758edc7a2542a4e674c8cc4332f43a62c00bd34b4a0476f0f44438d0f96.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/43279758edc7a2542a4e674c8cc4332f43a62c00bd34b4a0476f0f44438d0f96.jpg)

![52c7b75ae198ea66252d9c1108f8be790c311d7597850bdda179d54c9e2b953e.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/52c7b75ae198ea66252d9c1108f8be790c311d7597850bdda179d54c9e2b953e.jpg)

![6812d51a5a8669212b9e93984e1451c05ae529e0ff07cff28d48b1281f67c09b.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/6812d51a5a8669212b9e93984e1451c05ae529e0ff07cff28d48b1281f67c09b.jpg)

![75ca0fed24eb7f6a33af58e4a48166c666478a4a1ecced49dc1e74a35b34170a.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/75ca0fed24eb7f6a33af58e4a48166c666478a4a1ecced49dc1e74a35b34170a.jpg)

![7c68f7d5848a643631f3e5bf138456042d92c9c2e158c85373d1f9e1d93fbb4d.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/7c68f7d5848a643631f3e5bf138456042d92c9c2e158c85373d1f9e1d93fbb4d.jpg)

![8385cd8d6469f09a24f9ede9feb4566be4ce160ded3b5a4f24181359379b1998.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/8385cd8d6469f09a24f9ede9feb4566be4ce160ded3b5a4f24181359379b1998.jpg)

![874782f579dc8b361657a71fcacfef702a565beedc6b1805a7e717c9f0f2adc5.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/874782f579dc8b361657a71fcacfef702a565beedc6b1805a7e717c9f0f2adc5.jpg)

![89a31b9df02fd6e2a43059ba347b2f3d80b7d31e5da62b3b8bba1c716c728d0e.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/89a31b9df02fd6e2a43059ba347b2f3d80b7d31e5da62b3b8bba1c716c728d0e.jpg)

![89de3fb3330651212e7c6868cb3afcd507675bb68450a643dc811468dda3fba6.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/89de3fb3330651212e7c6868cb3afcd507675bb68450a643dc811468dda3fba6.jpg)

![983729d1e47d099ddc46e9b61c1cf943427a89371d3e621c76d7b2fffdd85ca5.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/983729d1e47d099ddc46e9b61c1cf943427a89371d3e621c76d7b2fffdd85ca5.jpg)

![af057af6e1570436c903f856b4d1ba52c12c8db65a2312c981409f09d8e16fae.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/af057af6e1570436c903f856b4d1ba52c12c8db65a2312c981409f09d8e16fae.jpg)

![bcc2320bcd7c11dea320e1c6a24d5030d1a8b75da146cb7ad4d228596505cc21.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/bcc2320bcd7c11dea320e1c6a24d5030d1a8b75da146cb7ad4d228596505cc21.jpg)

![bf34e0277441204dc96b8c2aa657cb7ec3c6c462b447005108d80b9e0952d153.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/bf34e0277441204dc96b8c2aa657cb7ec3c6c462b447005108d80b9e0952d153.jpg)

![c8c371acca01c9cf3d8115f55b30d1fc3bb719c113acc44012ce34b72ff2f6ac.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/c8c371acca01c9cf3d8115f55b30d1fc3bb719c113acc44012ce34b72ff2f6ac.jpg)

![cb23c384213deb3eacdf637ba139941799e4297c1450cbe8362a9c7ff79b542f.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/cb23c384213deb3eacdf637ba139941799e4297c1450cbe8362a9c7ff79b542f.jpg)

![dc45da94e98ad26fc527849d9523e8a38556323b9e8439345be614b6dd780e0a.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/dc45da94e98ad26fc527849d9523e8a38556323b9e8439345be614b6dd780e0a.jpg)

![f4ff6bc75d58a086c4195e22c58a612b91c9ba301654632dcde7944d27b57b52.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/f4ff6bc75d58a086c4195e22c58a612b91c9ba301654632dcde7944d27b57b52.jpg)

![f7519f9ef634b08a987b4a62e98d25533e3f17b12466170e9304dc3fc4f6f96e.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/images/f7519f9ef634b08a987b4a62e98d25533e3f17b12466170e9304dc3fc4f6f96e.jpg)

### Tables

![001cd4e19a0edc0c73effb005e9c36e7666397b25f4b4c8aa8b8718bd6283906.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/001cd4e19a0edc0c73effb005e9c36e7666397b25f4b4c8aa8b8718bd6283906.jpg)

![02e01a6695c00b9f030da7e8312a61fd80f93f29fc1c29979f99c51d286f087d.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/02e01a6695c00b9f030da7e8312a61fd80f93f29fc1c29979f99c51d286f087d.jpg)

![12bc28d1258adb8fb2782251894dc67700a3d423754ec6b3bfdc0178d582ff9e.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/12bc28d1258adb8fb2782251894dc67700a3d423754ec6b3bfdc0178d582ff9e.jpg)

![20d8d499be678b9602e32a06e2ce59bd07492d6bddb0cf8eb1a15b8aeb3a3db4.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/20d8d499be678b9602e32a06e2ce59bd07492d6bddb0cf8eb1a15b8aeb3a3db4.jpg)

![293b83b4c6a46e6443ffaca4937c4fd8b05d87d2662f35f26e11bf6b8a007b88.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/293b83b4c6a46e6443ffaca4937c4fd8b05d87d2662f35f26e11bf6b8a007b88.jpg)

![640fd6f6b3bb14a851304b372f7dda733b68bfb8cd04fcdefe7efa2ce285e41e.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/640fd6f6b3bb14a851304b372f7dda733b68bfb8cd04fcdefe7efa2ce285e41e.jpg)

![71fc020f685164b329a341cf37ffa201dd6d321f7d0c894cf87a10ddb9652b0a.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/71fc020f685164b329a341cf37ffa201dd6d321f7d0c894cf87a10ddb9652b0a.jpg)

![8458e68030d163595f2be9416ff1098c0f5783e58a0e6fbf6ee97263f64ffb46.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/8458e68030d163595f2be9416ff1098c0f5783e58a0e6fbf6ee97263f64ffb46.jpg)

![87ec82a36812c8daf9b22168219e805d452d2c0cd33481dd161f25ee9cfa21bf.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/87ec82a36812c8daf9b22168219e805d452d2c0cd33481dd161f25ee9cfa21bf.jpg)

![907496e5a390bb2b67eedf6310285b959b477745faf5de7ef481d624895758b1.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/907496e5a390bb2b67eedf6310285b959b477745faf5de7ef481d624895758b1.jpg)

![91bf6c2fa5821a1f16bb2656155dfbe5f111c7a28e43997e9631bd1c317e62b8.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/91bf6c2fa5821a1f16bb2656155dfbe5f111c7a28e43997e9631bd1c317e62b8.jpg)

![9dae2f7c9e0177c620699a826400872ee58f8e85f08ab58433d1c432fe0f93e6.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/9dae2f7c9e0177c620699a826400872ee58f8e85f08ab58433d1c432fe0f93e6.jpg)

![9f31d6cd449e03f72ee6ff229c5435b7049a8cdff6dc3aa2f716ed22d94bf599.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/9f31d6cd449e03f72ee6ff229c5435b7049a8cdff6dc3aa2f716ed22d94bf599.jpg)

![abcaf3b6f2bc1157475de9f76800cb5c49a712208c6b3236a7ff51217d55f350.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/abcaf3b6f2bc1157475de9f76800cb5c49a712208c6b3236a7ff51217d55f350.jpg)

![ae2702f31d8044b6b39b74ff3537fd63b39f0fa9a23aec1a9bf6587d9aee4f36.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/ae2702f31d8044b6b39b74ff3537fd63b39f0fa9a23aec1a9bf6587d9aee4f36.jpg)

![b2635c25b1dfd4df6f79c4213cd0317812a1aba1e8284d1fc683be7e0ab73758.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/b2635c25b1dfd4df6f79c4213cd0317812a1aba1e8284d1fc683be7e0ab73758.jpg)

![c2fa4356b3e6a420c6e0e3d3788c530fa6b345f8e7ea16b71e1277ac8857a4b0.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/c2fa4356b3e6a420c6e0e3d3788c530fa6b345f8e7ea16b71e1277ac8857a4b0.jpg)

![c69988fa144197580787587f2113c969cd6b8be4c2ef4d3d914dccfe93edf175.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/c69988fa144197580787587f2113c969cd6b8be4c2ef4d3d914dccfe93edf175.jpg)

![c8c1c1274a640240105df92908e40bed1f4a81ebf6f708d18b5072e0430e9dee.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/c8c1c1274a640240105df92908e40bed1f4a81ebf6f708d18b5072e0430e9dee.jpg)

![cea74b75000f770b8eb97121326508bd2f4775a50cd9fe3bed14669649b85003.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/cea74b75000f770b8eb97121326508bd2f4775a50cd9fe3bed14669649b85003.jpg)

![f0c0d200e0b3451f4ed18b35dbbda368c2cf48b1566401d45b1fe1ebe9faa872.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/f0c0d200e0b3451f4ed18b35dbbda368c2cf48b1566401d45b1fe1ebe9faa872.jpg)

![ff1fc560d673a3ec512ae36f5929159eb4e8894f5bae022593f2cfc66e43a45b.jpg](../icml_results/3432_MacroGuide_%20Topological%20Guidance%20for%20Macrocycle%20Generation/tables/ff1fc560d673a3ec512ae36f5929159eb4e8894f5bae022593f2cfc66e43a45b.jpg)

<a id="paper-3393"></a>

## EnterpriseOps-Gym: Environments and Evaluations for Stateful Agentic Planning and Tool Use in Enterprise Settings


### Images

![00e9ae9d7bf942f730dee42e9c8f59c5e5b911f8fb628036cff3e3f62329685e.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/00e9ae9d7bf942f730dee42e9c8f59c5e5b911f8fb628036cff3e3f62329685e.jpg)

![03f02925c879de05b598ab83d0493215e0e79776fbcb43fbf854293761d61204.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/03f02925c879de05b598ab83d0493215e0e79776fbcb43fbf854293761d61204.jpg)

![532c3504e1508b33bbe262bed481ffcd2bc7874682e59fba7db27d67b58635d5.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/532c3504e1508b33bbe262bed481ffcd2bc7874682e59fba7db27d67b58635d5.jpg)

![5461bb806456da617b97418ffd31bf8fe9052baa9d1ffbf9be36cf546d83a826.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/5461bb806456da617b97418ffd31bf8fe9052baa9d1ffbf9be36cf546d83a826.jpg)

![60e50ab57d173ddffd7ce6e9fcfa103626b9d5bc199132aa6fb065f362874f83.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/60e50ab57d173ddffd7ce6e9fcfa103626b9d5bc199132aa6fb065f362874f83.jpg)

![73af69775675ff9715ca4465760a84e657ac1dbf424ed1273e5ff7d15a9cda32.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/73af69775675ff9715ca4465760a84e657ac1dbf424ed1273e5ff7d15a9cda32.jpg)

![8b43d0e40ffe5bf264afed7c0e14e6fe1833322b264e918eaa4d764a169c71c5.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/images/8b43d0e40ffe5bf264afed7c0e14e6fe1833322b264e918eaa4d764a169c71c5.jpg)

### Tables

![07f1e8a1ea9c69a613fbfe09a52102597d3314a071f168887cf9fbd58c17a9ab.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/07f1e8a1ea9c69a613fbfe09a52102597d3314a071f168887cf9fbd58c17a9ab.jpg)

![0b2b1aa8b9e0dbb0bec689223dfbdc1b8a32c97d905509d1797814437d9d53fc.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/0b2b1aa8b9e0dbb0bec689223dfbdc1b8a32c97d905509d1797814437d9d53fc.jpg)

![0c94e10cfb66043c93328291c6da7cf6e898322672f43407258a058c7ca5285d.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/0c94e10cfb66043c93328291c6da7cf6e898322672f43407258a058c7ca5285d.jpg)

![10f11c23f464fb2d72ad1a9a2153ede83001041112c6472cbb74f82291a2dc16.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/10f11c23f464fb2d72ad1a9a2153ede83001041112c6472cbb74f82291a2dc16.jpg)

![220f19389b50f25e7e4a3aa7a96ab2655b869dbfd4d5067fcbcd6d6c1d33b415.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/220f19389b50f25e7e4a3aa7a96ab2655b869dbfd4d5067fcbcd6d6c1d33b415.jpg)

![3b9974d72c5eda46b86ea7145cdefb112bb6220017763c335d0fadfcb1f9b8ac.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/3b9974d72c5eda46b86ea7145cdefb112bb6220017763c335d0fadfcb1f9b8ac.jpg)

![3edabc47125ec83f2e0ec03d6fea42d0df28ec3ced5bb44064271f9fe87c9f51.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/3edabc47125ec83f2e0ec03d6fea42d0df28ec3ced5bb44064271f9fe87c9f51.jpg)

![79d2b1adf5413034256e8b90f7f587c4bac911ebaeb95c043b1cbd33e9b38764.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/79d2b1adf5413034256e8b90f7f587c4bac911ebaeb95c043b1cbd33e9b38764.jpg)

![8d43d66bd24afe9bd35c9ac5f8d2aeb134e45d376f012f3724701df3bed0c533.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/8d43d66bd24afe9bd35c9ac5f8d2aeb134e45d376f012f3724701df3bed0c533.jpg)

![cd227d8d49d136eb0b52bdc287bf7f78213de5b43fc8d3b665db4fbb1fc36aae.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/cd227d8d49d136eb0b52bdc287bf7f78213de5b43fc8d3b665db4fbb1fc36aae.jpg)

![ff00e2cd18ac1bb8545ed9c427aeed7aa9fc6709b2f418802fb625ffd863e1c9.jpg](../icml_results/3433_EnterpriseOps-Gym_%20Environments%20and%20Evaluations%20for%20Stateful%20Agentic%20Planning%20and%20Tool%20Use%20in%20Enterp/tables/ff00e2cd18ac1bb8545ed9c427aeed7aa9fc6709b2f418802fb625ffd863e1c9.jpg)

<a id="paper-3394"></a>

## MADE: Benchmark Environments for Closed-Loop Materials Discovery


### Images

![123d4e4c7bc7c6ada240382077bcf259741d2e17777fc55f8ad8426847c1fd08.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/123d4e4c7bc7c6ada240382077bcf259741d2e17777fc55f8ad8426847c1fd08.jpg)

![167f02ae33d8bb69c7f2ff587005627014b9f006429b877d7a8b3efed12bdf50.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/167f02ae33d8bb69c7f2ff587005627014b9f006429b877d7a8b3efed12bdf50.jpg)

![18acf43c3e46e0d96c4b909a754b9c9f6a7f5c363374ebd65d420ce177616e11.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/18acf43c3e46e0d96c4b909a754b9c9f6a7f5c363374ebd65d420ce177616e11.jpg)

![19b0046e2811a92291b07b93ad4a01903187044bac238245a5af4fb2f651cc30.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/19b0046e2811a92291b07b93ad4a01903187044bac238245a5af4fb2f651cc30.jpg)

![225154ef68667678c6ee7542d791d3393a5730582e2b47c0b600f73c7e954f35.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/225154ef68667678c6ee7542d791d3393a5730582e2b47c0b600f73c7e954f35.jpg)

![36acfcf9561051c6cfbe7002cdd93d6e03c2518b5fda75e220b79afa3581f579.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/36acfcf9561051c6cfbe7002cdd93d6e03c2518b5fda75e220b79afa3581f579.jpg)

![46c9f2cc7fd0f9ae372c8c2c31fae7909238cc2750666716d1754a37d0a9dab1.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/46c9f2cc7fd0f9ae372c8c2c31fae7909238cc2750666716d1754a37d0a9dab1.jpg)

![5f9595a5e23625ded8343c7a909e26c09fc4dc253430732248656a4b75d86cdc.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/5f9595a5e23625ded8343c7a909e26c09fc4dc253430732248656a4b75d86cdc.jpg)

![6be2028c052ea33a18fff9114359c35c23dfbc6e11c984bed5ef85b18e7be723.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/6be2028c052ea33a18fff9114359c35c23dfbc6e11c984bed5ef85b18e7be723.jpg)

![6e6a8ae8c0fc496d813e9ac69228441d92da534d64cb0a69541d5ee8b254edfe.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/6e6a8ae8c0fc496d813e9ac69228441d92da534d64cb0a69541d5ee8b254edfe.jpg)

![7df094195a02948bbcd5fcd1f8bdf1fd59f8ba1166253fb18f49d33d9c0b8e34.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/7df094195a02948bbcd5fcd1f8bdf1fd59f8ba1166253fb18f49d33d9c0b8e34.jpg)

![9ca588905c70346f8185fd71d74a5a1cd90327f0022c6fd2c85e3fe95cdd72cd.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/9ca588905c70346f8185fd71d74a5a1cd90327f0022c6fd2c85e3fe95cdd72cd.jpg)

![a0936c566467bc9f8097422573aac199f138ee264e4b7c8502424aca0276331c.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/a0936c566467bc9f8097422573aac199f138ee264e4b7c8502424aca0276331c.jpg)

![a935e88a89aa107cb1a875b70799bca94ba7179d34c29718134871e35c8be8be.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/a935e88a89aa107cb1a875b70799bca94ba7179d34c29718134871e35c8be8be.jpg)

![c463bf1df221b859c7e0c461bc8053f1a940cf1160415de1f088a336211563af.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/c463bf1df221b859c7e0c461bc8053f1a940cf1160415de1f088a336211563af.jpg)

![ef738881768848a0c12af656de1a6bc766642f387b2e61efea7a05f47afe8cf3.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/images/ef738881768848a0c12af656de1a6bc766642f387b2e61efea7a05f47afe8cf3.jpg)

### Tables

![0fbe786209ceddb975ff5087d4ddd023588d77736f6dab233927e1987dd7862a.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/0fbe786209ceddb975ff5087d4ddd023588d77736f6dab233927e1987dd7862a.jpg)

![24734af60c5d19253711c6c8539593bd77f94a2cc132f070fc597696948e3470.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/24734af60c5d19253711c6c8539593bd77f94a2cc132f070fc597696948e3470.jpg)

![26311032a79b58d8967d7f848c9911b541d8c0a102937164004e66634c9bfb2f.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/26311032a79b58d8967d7f848c9911b541d8c0a102937164004e66634c9bfb2f.jpg)

![bf174e604c04e4216a96df47c6ae956b20610df4d26bc9013a51054ea67dde17.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/bf174e604c04e4216a96df47c6ae956b20610df4d26bc9013a51054ea67dde17.jpg)

![e3e28b9860fd8a7dda87d70e9634bd7864a886434db1e62d00c76b623a6ac2c6.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/e3e28b9860fd8a7dda87d70e9634bd7864a886434db1e62d00c76b623a6ac2c6.jpg)

![fbb92cdb2535f43a428a4bc79595e9f7d765a4de151ab6025fd015b243c04abf.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/fbb92cdb2535f43a428a4bc79595e9f7d765a4de151ab6025fd015b243c04abf.jpg)

![fed02daef1887b3c3dc1250f1774ab95568f30d8b16422175edfe69a66189dd3.jpg](../icml_results/3434_MADE_%20Benchmark%20Environments%20for%20Closed-Loop%20Materials%20Discovery/tables/fed02daef1887b3c3dc1250f1774ab95568f30d8b16422175edfe69a66189dd3.jpg)

<a id="paper-3395"></a>

## Directly Optimizing Natural Language Explanations for Behavioral Faithfulness: Simulatability and Recoverability


### Images

![2b20428717f197fb67d9ffb9c0f34cd637d77823b803357061f9a7a60bbb9c41.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/images/2b20428717f197fb67d9ffb9c0f34cd637d77823b803357061f9a7a60bbb9c41.jpg)

![7f04dad8194ddc08c4a3e7bb32d5513539f924fa1ab330aae111687b44372759.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/images/7f04dad8194ddc08c4a3e7bb32d5513539f924fa1ab330aae111687b44372759.jpg)

![ab0b7fd4e6a613da37cc65700c13186e1e79769350bf0a5dde079dca6eec07b5.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/images/ab0b7fd4e6a613da37cc65700c13186e1e79769350bf0a5dde079dca6eec07b5.jpg)

### Tables

![03538adb7a45090fb131a79c29835468bc95c875b494f740636b88d91ffe3d68.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/03538adb7a45090fb131a79c29835468bc95c875b494f740636b88d91ffe3d68.jpg)

![18f0d02b981dc40d558d4fc0203307552c4ff8ababd8765384991c26d8e56d35.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/18f0d02b981dc40d558d4fc0203307552c4ff8ababd8765384991c26d8e56d35.jpg)

![47262a8c69ead88a2c127df7aa863a6d4d46f1f739dccbde29ca0e6f15755c07.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/47262a8c69ead88a2c127df7aa863a6d4d46f1f739dccbde29ca0e6f15755c07.jpg)

![590ec454858910467e55343b8ba592451fae4b3340c94ac8a51b7b9f35e75bee.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/590ec454858910467e55343b8ba592451fae4b3340c94ac8a51b7b9f35e75bee.jpg)

![73fbd79304387bd67bd68dbd6bf47348a940b822861143c4b5a24ac6df7a2e01.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/73fbd79304387bd67bd68dbd6bf47348a940b822861143c4b5a24ac6df7a2e01.jpg)

![7e9d2db3c555a685d70f538f88b4a12afd9f719efb400cdf397596e24e6193ff.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/7e9d2db3c555a685d70f538f88b4a12afd9f719efb400cdf397596e24e6193ff.jpg)

![e16413b92493cf474c56226411552c2775ef054cf6a842e9e0a06072bc8fc82e.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/e16413b92493cf474c56226411552c2775ef054cf6a842e9e0a06072bc8fc82e.jpg)

![e96ec7d8d81df3ac3735457a7b4586552d8b972a03808fa696f959f3b4e7e7d6.jpg](../icml_results/3435_Directly%20Optimizing%20Natural%20Language%20Explanations%20for%20Behavioral%20Faithfulness_%20Simulatability%20and%20Re/tables/e96ec7d8d81df3ac3735457a7b4586552d8b972a03808fa696f959f3b4e7e7d6.jpg)

<a id="paper-3396"></a>

## Variational Flow Maps: Make Some Noise for One-Step Conditional Generation


### Images

![021c7b0299af3537d8a94e6e86a013dfa10f812309013ea3bd53cd5d9e1e5898.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/021c7b0299af3537d8a94e6e86a013dfa10f812309013ea3bd53cd5d9e1e5898.jpg)

![03564079912d3e96be30281b53dbffe9bb2e7def7f9465e39fa7c54116a60a28.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/03564079912d3e96be30281b53dbffe9bb2e7def7f9465e39fa7c54116a60a28.jpg)

![33668cff8e5c2d5183b0da25b036950725fe31f2b029d2f7d7d580f8af7755d5.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/33668cff8e5c2d5183b0da25b036950725fe31f2b029d2f7d7d580f8af7755d5.jpg)

![380faab7a97276fde31d27fc32dace17b7eaa1c49de9d2e00771023a04ae31b3.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/380faab7a97276fde31d27fc32dace17b7eaa1c49de9d2e00771023a04ae31b3.jpg)

![3a79e63401b28d7d0bdc45bc9b727f9c768d026dadf5ba7f9e039f24bfa996af.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/3a79e63401b28d7d0bdc45bc9b727f9c768d026dadf5ba7f9e039f24bfa996af.jpg)

![3c7eaa87e1e79663010b00002dd549f565ea8f457123aa62447d1f3f7be5f2b0.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/3c7eaa87e1e79663010b00002dd549f565ea8f457123aa62447d1f3f7be5f2b0.jpg)

![53c959d0f6eea5ecfca03c9c7eba66379ce6a100bc163dd3a945618dd1eb9569.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/53c959d0f6eea5ecfca03c9c7eba66379ce6a100bc163dd3a945618dd1eb9569.jpg)

![5479029d76b376518f09012a1e9c834615f493ea40cbdb6f9b1da23c87796124.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/5479029d76b376518f09012a1e9c834615f493ea40cbdb6f9b1da23c87796124.jpg)

![6b5bead724b359fd911b59a66b993153746a147c25ad1f621e2bee800a32cc86.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/6b5bead724b359fd911b59a66b993153746a147c25ad1f621e2bee800a32cc86.jpg)

![78b1cd87f119f5ccf02b24cadf9c37d27b6fa10281d82d90dea0689f5bb09ff2.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/78b1cd87f119f5ccf02b24cadf9c37d27b6fa10281d82d90dea0689f5bb09ff2.jpg)

![8292a4211e1db259f147ae3eb76f470045257f05c51dd40fa7e0602db143e276.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/8292a4211e1db259f147ae3eb76f470045257f05c51dd40fa7e0602db143e276.jpg)

![85d618a591d32660634aff2928de646f6e51429341ee5ae4adcd35bc6df7c4ca.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/85d618a591d32660634aff2928de646f6e51429341ee5ae4adcd35bc6df7c4ca.jpg)

![90f0e12a8de94838ba3df10b336475d0b050cfb43ce74fc7e6154c201f8503e5.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/90f0e12a8de94838ba3df10b336475d0b050cfb43ce74fc7e6154c201f8503e5.jpg)

![95448a1661506afeaeb0f1891f6b3d3088395c06bbb745315175180c91d6cf61.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/95448a1661506afeaeb0f1891f6b3d3088395c06bbb745315175180c91d6cf61.jpg)

![a1586264ab5b57000352d2ffe5565eb065f4ca9a85aeab2f8d31843e765c18ea.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/a1586264ab5b57000352d2ffe5565eb065f4ca9a85aeab2f8d31843e765c18ea.jpg)

![a9cd84009be72e41adc133bbddf83028491cc946c865bac9d6621ff55e9a406f.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/a9cd84009be72e41adc133bbddf83028491cc946c865bac9d6621ff55e9a406f.jpg)

![b1c3411de5dfa3aff5382e1d39d48693fa32aae735f7cfcf5da2670da94db6f9.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/b1c3411de5dfa3aff5382e1d39d48693fa32aae735f7cfcf5da2670da94db6f9.jpg)

![b58a3d83474d49656485dd8829385a4e0ad28aeadfb339a1361f2556dde61d56.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/b58a3d83474d49656485dd8829385a4e0ad28aeadfb339a1361f2556dde61d56.jpg)

![c6f8d6d50e7e203a3f0a17d40bcbba3b73587a4fd6852841a4194d4a0f3f2fd3.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/c6f8d6d50e7e203a3f0a17d40bcbba3b73587a4fd6852841a4194d4a0f3f2fd3.jpg)

![e585240679d234b00d0db8e02ea753717171b32d9b2980454b36288ab87fed1d.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/e585240679d234b00d0db8e02ea753717171b32d9b2980454b36288ab87fed1d.jpg)

![e73783381813e4ffdb35c7b46a2f7cc9cf1430182a64d38fb4a718199abb97f9.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/e73783381813e4ffdb35c7b46a2f7cc9cf1430182a64d38fb4a718199abb97f9.jpg)

![fc20805cf2ba93bfdb1f5d8ccf02dd5c3a794ee98d14b4b96349316e1944fc48.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/images/fc20805cf2ba93bfdb1f5d8ccf02dd5c3a794ee98d14b4b96349316e1944fc48.jpg)

### Tables

![5f1356ce77715a2767603dc3e0f0c11ba700aef833c8ba9edcec6f242ee163bb.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/tables/5f1356ce77715a2767603dc3e0f0c11ba700aef833c8ba9edcec6f242ee163bb.jpg)

![8a980bcb3d78db203bccb866b8e35aed4ae766c06dfb283564c46850c72e21a6.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/tables/8a980bcb3d78db203bccb866b8e35aed4ae766c06dfb283564c46850c72e21a6.jpg)

![b1ba400d2555ce03d5084947564751b6d4630586aca570dc668c2c416a70d3e4.jpg](../icml_results/3436_Variational%20Flow%20Maps_%20Make%20Some%20Noise%20for%20One-Step%20Conditional%20Generation/tables/b1ba400d2555ce03d5084947564751b6d4630586aca570dc668c2c416a70d3e4.jpg)

<a id="paper-3397"></a>

## Turbo4DGen: Ultra-Fast Acceleration for 4D Generation


### Images

![0389a6dba4516817b561ec290071bd3d373235c2d08ed64b5988e0e70696da4d.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/0389a6dba4516817b561ec290071bd3d373235c2d08ed64b5988e0e70696da4d.jpg)

![08f21b583163971724d640d074b3109cd238a57bc16c32fd1e3e0600ff4d69ff.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/08f21b583163971724d640d074b3109cd238a57bc16c32fd1e3e0600ff4d69ff.jpg)

![107cfd5b8903ca9613b60447ae70ecd825dc744283f3f7c4583ba531db609a0a.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/107cfd5b8903ca9613b60447ae70ecd825dc744283f3f7c4583ba531db609a0a.jpg)

![14bb4344d45db9b1373a10c7dbf4ab8d2e8e13f5d52f0133747cb5f4bb61cda1.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/14bb4344d45db9b1373a10c7dbf4ab8d2e8e13f5d52f0133747cb5f4bb61cda1.jpg)

![23822562095515bf98a71113a78806ec8bb75ecf5d2e37e3a6ec13f23f1cfdbe.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/23822562095515bf98a71113a78806ec8bb75ecf5d2e37e3a6ec13f23f1cfdbe.jpg)

![29a1222cd7f29a70a846daa0ff2759bb1e80cc7680527bb7f14614e6d76d8339.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/29a1222cd7f29a70a846daa0ff2759bb1e80cc7680527bb7f14614e6d76d8339.jpg)

![3dc2c847b5b85320e1b6d05fea98c54ee4c502abc5aa4e95bbdbde44d6e5df9e.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/3dc2c847b5b85320e1b6d05fea98c54ee4c502abc5aa4e95bbdbde44d6e5df9e.jpg)

![653efe3adfc24086d48290c249b5e517a0014ac13fdad49e6f441849d2279a3d.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/653efe3adfc24086d48290c249b5e517a0014ac13fdad49e6f441849d2279a3d.jpg)

![ae4687f3c987b060cf3871f085e860b1168cdd6b668bfcc0171ec42b7fb9b8a5.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/ae4687f3c987b060cf3871f085e860b1168cdd6b668bfcc0171ec42b7fb9b8a5.jpg)

![d4528b8e0ec1851316e10ec2ff3f963c99aace51bebff0b54dfc8105829a9a09.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/d4528b8e0ec1851316e10ec2ff3f963c99aace51bebff0b54dfc8105829a9a09.jpg)

![dcd2c313776ff40f8240d3856df6d29e4f524da967a013554e9a2827675b09fc.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/dcd2c313776ff40f8240d3856df6d29e4f524da967a013554e9a2827675b09fc.jpg)

![ef00c8599bc97e5a80c6135a5b7d4c3717347113c078c359add122a1103bc876.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/ef00c8599bc97e5a80c6135a5b7d4c3717347113c078c359add122a1103bc876.jpg)

![fac72d2db257aba45e1189223ae5603c4ef56e555c0864c173842f4e66683a21.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/images/fac72d2db257aba45e1189223ae5603c4ef56e555c0864c173842f4e66683a21.jpg)

### Tables

![156c0adf4103ed145c67819bb8baf158b34fa6be9ddd66d76f2599dca694b872.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/tables/156c0adf4103ed145c67819bb8baf158b34fa6be9ddd66d76f2599dca694b872.jpg)

![15b56b20bbee824b0f0f8557a4b4436ab1214af384acaf1a16eda504b2291ba0.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/tables/15b56b20bbee824b0f0f8557a4b4436ab1214af384acaf1a16eda504b2291ba0.jpg)

![326e6534dc862b894ca59b0b5baa88e4996451f733acaf5675e681c8c2b712e5.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/tables/326e6534dc862b894ca59b0b5baa88e4996451f733acaf5675e681c8c2b712e5.jpg)

![4b1a7200c06b553a8d5822d425f312d41e7b4198069596376aa57a91508195b8.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/tables/4b1a7200c06b553a8d5822d425f312d41e7b4198069596376aa57a91508195b8.jpg)

![769c4c653032b0498e1001f16fc7d03c8c2dcc8d7fa66d7a48e55d80089135f6.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/tables/769c4c653032b0498e1001f16fc7d03c8c2dcc8d7fa66d7a48e55d80089135f6.jpg)

![90f1186de0aa5b21aea8073aecf8471fc86a694a02f294538b22107f4661ee29.jpg](../icml_results/3437_Turbo4DGen_%20Ultra-Fast%20Acceleration%20for%204D%20Generation/tables/90f1186de0aa5b21aea8073aecf8471fc86a694a02f294538b22107f4661ee29.jpg)

<a id="paper-3398"></a>

## The Consistency Dilemma in LLMs: Generator-Evaluator Agreement and Vulnerability to Mistakes


### Images

![16f82cff209d700edfb7375d45c5b8519062730f3e7de08db37db9e60da5b321.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/images/16f82cff209d700edfb7375d45c5b8519062730f3e7de08db37db9e60da5b321.jpg)

![4544ee17a1286a5871da2ab45af60520de7693f19c3ccd4a166880f09150adf9.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/images/4544ee17a1286a5871da2ab45af60520de7693f19c3ccd4a166880f09150adf9.jpg)

![4c840c625cb97507c9a57bbd07632c813913f0fc2559711a7257d888a7550842.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/images/4c840c625cb97507c9a57bbd07632c813913f0fc2559711a7257d888a7550842.jpg)

![cfeb01d1c1b2bbbf1af01c5eca3375656e2ea6937ced89ddba08fe48584f0d9a.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/images/cfeb01d1c1b2bbbf1af01c5eca3375656e2ea6937ced89ddba08fe48584f0d9a.jpg)

![f32f931e2d82b8300a31d0cac81d229e60d2acf01e46e2338d4247c562eab4d6.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/images/f32f931e2d82b8300a31d0cac81d229e60d2acf01e46e2338d4247c562eab4d6.jpg)

### Tables

![0c713db679a58a68b983fb195b73e9cbadfdf42dc7049be06edddea56155c646.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/0c713db679a58a68b983fb195b73e9cbadfdf42dc7049be06edddea56155c646.jpg)

![0d60812df0fd04fe3b7cfc067fdf6dc563a71b1183015d27dd0f5fc34a903408.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/0d60812df0fd04fe3b7cfc067fdf6dc563a71b1183015d27dd0f5fc34a903408.jpg)

![0d7d28511211689e143d14849bad22a7b3fe34d2d10614c6f4a2993e97b6fc15.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/0d7d28511211689e143d14849bad22a7b3fe34d2d10614c6f4a2993e97b6fc15.jpg)

![5c453719afe9c13e20a3dcb20e8db2907343b34b10d135a456e842213be996e2.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/5c453719afe9c13e20a3dcb20e8db2907343b34b10d135a456e842213be996e2.jpg)

![5c703b22bb3b3d17faffb199f0de067ed02f84d871a11e4c14ee64afdc9d8853.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/5c703b22bb3b3d17faffb199f0de067ed02f84d871a11e4c14ee64afdc9d8853.jpg)

![93cedcde247967cbdaf8b0891fcebd9e5d1563b4c7e3b0d71c1b806c8235e332.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/93cedcde247967cbdaf8b0891fcebd9e5d1563b4c7e3b0d71c1b806c8235e332.jpg)

![9458aa61e5f836bb6f38b2a7d3b3977646abc6351b8c695ab11a181c2d3c10bf.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/9458aa61e5f836bb6f38b2a7d3b3977646abc6351b8c695ab11a181c2d3c10bf.jpg)

![9949231669b20d84273c2aafb97c7047a7e7667b028cfe98fc318b353e15b5ae.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/9949231669b20d84273c2aafb97c7047a7e7667b028cfe98fc318b353e15b5ae.jpg)

![c32f3a09edc23dfd0d6b8dcf04bd87debc6d9c90831c3ce068a8bb18ff6a0fe3.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/c32f3a09edc23dfd0d6b8dcf04bd87debc6d9c90831c3ce068a8bb18ff6a0fe3.jpg)

![c5bf94ed6982072c8573dfbcf8f6910386e97867601c211f4cae59bdc8672f51.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/c5bf94ed6982072c8573dfbcf8f6910386e97867601c211f4cae59bdc8672f51.jpg)

![cc93cf315fb7569e3b35d2443004e7b58247185a587a053b224b3c949f7f793f.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/cc93cf315fb7569e3b35d2443004e7b58247185a587a053b224b3c949f7f793f.jpg)

![d42fc8578941afcbdcf646455f64f8a45857b05049fd5f612b383198b772028e.jpg](../icml_results/3438_The%20Consistency%20Dilemma%20in%20LLMs_%20Generator-Evaluator%20Agreement%20and%20Vulnerability%20to%20Mistakes/tables/d42fc8578941afcbdcf646455f64f8a45857b05049fd5f612b383198b772028e.jpg)

<a id="paper-3399"></a>

## Distributionally Robust Reinforcement Learning from Human Feedback

### Images

![5e2ea7d9ccffd6c91dfb9230e3263a453bf4092be183b5edd2b4c967b1ffa478.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/images/5e2ea7d9ccffd6c91dfb9230e3263a453bf4092be183b5edd2b4c967b1ffa478.jpg)

![9054a9d460768ec2b514c9c1cad667dd5ce276ecb5a0386c8b0a6593dac188b8.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/images/9054a9d460768ec2b514c9c1cad667dd5ce276ecb5a0386c8b0a6593dac188b8.jpg)

### Tables

![0d57cbca1ba1e34b637c8f0ebeaeb6aaeb138240b6786a7f4151a02e0fbf4ce8.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/0d57cbca1ba1e34b637c8f0ebeaeb6aaeb138240b6786a7f4151a02e0fbf4ce8.jpg)

![446a122fb05944f95eb562bd980fb92984d89ef7bd8fc944fc2d69e394c96d30.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/446a122fb05944f95eb562bd980fb92984d89ef7bd8fc944fc2d69e394c96d30.jpg)

![5525db329cc482930ba4d7796112cd9131dfdbe4261aaa0174c17293eff880b7.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/5525db329cc482930ba4d7796112cd9131dfdbe4261aaa0174c17293eff880b7.jpg)

![56ef48e0c24852a225f556ecefeb2b6b4145d80618b9bf8de665835c0ccbb3b3.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/56ef48e0c24852a225f556ecefeb2b6b4145d80618b9bf8de665835c0ccbb3b3.jpg)

![7548afd210d733d5c368e329bc59d8271d8451c77a3f72e5e9df9795cf80c2fb.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/7548afd210d733d5c368e329bc59d8271d8451c77a3f72e5e9df9795cf80c2fb.jpg)

![77cbbce604fbab175f04b4dd4463e20c94ebb3c4812f0db7a6beaf6123dd01da.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/77cbbce604fbab175f04b4dd4463e20c94ebb3c4812f0db7a6beaf6123dd01da.jpg)

![7924779d49a75e79def31b7ae9ccba4c8b44c05f0443343b0b6a9284bd75619a.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/7924779d49a75e79def31b7ae9ccba4c8b44c05f0443343b0b6a9284bd75619a.jpg)

![a54eb44eb528d26f10d922ab8c0b1a7e942425a9431150c14eeaeb40be764ea8.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/a54eb44eb528d26f10d922ab8c0b1a7e942425a9431150c14eeaeb40be764ea8.jpg)

![b6f0ea1d433ab11a737efdfc38d70bc4b578327b77b8d2285563f9358d5150cf.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/b6f0ea1d433ab11a737efdfc38d70bc4b578327b77b8d2285563f9358d5150cf.jpg)

![b77fd62c3d0579b4dba488cfdf2523be9fa4223f560a0e7946469ac641ff4e70.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/b77fd62c3d0579b4dba488cfdf2523be9fa4223f560a0e7946469ac641ff4e70.jpg)

![c327fd5fd13d54124ba49fbe71f502ea46ae1d899a164e13393ad56dc0d4092d.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/c327fd5fd13d54124ba49fbe71f502ea46ae1d899a164e13393ad56dc0d4092d.jpg)

![c36104ad94bfe91cc3032e0914fa9c6848d8be9a79bd135b48816fd45df77308.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/c36104ad94bfe91cc3032e0914fa9c6848d8be9a79bd135b48816fd45df77308.jpg)

![c3997f33d3a254486c6bd6304b65e3f79afa5095eac2adcbffe5ec6b9f3e8b1d.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/c3997f33d3a254486c6bd6304b65e3f79afa5095eac2adcbffe5ec6b9f3e8b1d.jpg)

![d1d8765a2932cf3b6cb1be80025f3f72cbd643dd2ced6e79dd1870db2b63c50c.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/d1d8765a2932cf3b6cb1be80025f3f72cbd643dd2ced6e79dd1870db2b63c50c.jpg)

![d79f1a82708c35489b35a243e7949891110fbe8e9717cd6d9ac1dc7a7c018e84.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/d79f1a82708c35489b35a243e7949891110fbe8e9717cd6d9ac1dc7a7c018e84.jpg)

![e0765ea3fa5bbd5218284b10db7e23f7ab39768106513ac6e30c4f9ed949ba1a.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/e0765ea3fa5bbd5218284b10db7e23f7ab39768106513ac6e30c4f9ed949ba1a.jpg)

![e495cf9d245b5bc512c006f4585d8fc7266117859ff3c3ff498590b9253a3cb2.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/e495cf9d245b5bc512c006f4585d8fc7266117859ff3c3ff498590b9253a3cb2.jpg)

![f5f9add28f5eb57fce94aa50547091644a523cbf8120bd0419f44b936c417ea8.jpg](../icml_results/3439_Distributionally%20Robust%20Reinforcement%20Learning%20from%20Human%20Feedback/tables/f5f9add28f5eb57fce94aa50547091644a523cbf8120bd0419f44b936c417ea8.jpg)
