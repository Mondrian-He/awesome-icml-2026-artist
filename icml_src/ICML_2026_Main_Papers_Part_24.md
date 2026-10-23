# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 24 of 197

## 目录 (Table of Contents)

1. [Smoothness Errors in Dynamics Models and How to Avoid Them](#paper-0760)
2. [Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics](#paper-0761)
3. [Approximate Equivariance via Projection-Based Regularisation](#paper-0762)
4. [M-Star: Markovian Projection of Star-Shaped Diffusion for Exponential Family Distributions](#paper-0763)
5. [Motion Planning in Compressed Representation Spaces](#paper-0764)
6. [RSA-CP: Efficient Conformal Prediction in Small-Sample Regimes via Random Score Alignment](#paper-0765)
7. [Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long-Form Text Generation](#paper-0766)
8. [CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings](#paper-0767)
9. [Dynamic High-Dimensional Facility Location with Low Recourse](#paper-0768)
10. [EchoRL: Reinforcement Learning via Rollout Echoing](#paper-0769)
11. [FedFit: Federated Dynamic Sparse Training via Fisher Information scoring](#paper-0770)
12. [Does Reasoning Improve Seeing? Understanding When Vision-Language Models Benefit from Thinking](#paper-0771)
13. [The Geometry of Reasoning: Self-Evaluation via Layerwise Trajectory Evolution](#paper-0772)
14. [RN-D: Discretized Categorical Actors for On-Policy Reinforcement Learning](#paper-0773)
15. [Future-Gain Guided Test-Time Learning for Large Language Models](#paper-0774)
16. [AgentVocab: Structure-Aware Vocabulary Adaptation for Efficient LLM Agents](#paper-0775)
17. [CURE: Consistency-under-Unified Semantic Regularization for Generalized Category Discovery](#paper-0776)
18. [FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](#paper-0777)
19. [Retrieval-Aware Distillation for Transformer-SSM Hybrids](#paper-0778)
20. [Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering](#paper-0779)
21. [What If We Allocate Test-Time Compute Adaptively?](#paper-0780)
22. [Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models](#paper-0781)
23. [CORAL: Uncertainty-Aware Regulation of Exposure Concentration in Recommender Systems](#paper-0782)
24. [CARE: Adaptive Calibration for Reliable Recommendations](#paper-0783)
25. [FOVI: A biologically-inspired foveated interface for deep vision models](#paper-0784)
26. [Bridge Matching Sampler: Scalable Sampling via Generalized Fixed-Point Diffusion Matching](#paper-0785)
27. [Batched First-Order Methods for Parallel LP Solving in MIP](#paper-0786)
28. [Temper-Then-Tilt: Principled Unlearning for Generative Models through Tempering and Classifier Guidance](#paper-0787)
29. [Noise-Guided Transport: Imitation Learning from Random Priors](#paper-0788)
30. [Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction](#paper-0789)
31. [Benchmarking Dense and Indiscernible Object Counting with Blueberries](#paper-0790)
32. [Efficient Multi-modal Dataset Distillation via Analytic Parameter Matching](#paper-0791)
33. [Counterfactual Bootstrap for Robust Meta-Reinforcement Learning](#paper-0792)

---


<a id="paper-0760"></a>

## Smoothness Errors in Dynamics Models and How to Avoid Them

### Images

![183e88e50a34a48552504d9347ee2696b5f91ffc5559df79a21f027c54b23417.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/183e88e50a34a48552504d9347ee2696b5f91ffc5559df79a21f027c54b23417.jpg)

![1c583e3f061fad7aac1461d8b7dd79f8df034a3cac72568d90c1ae38691d8e5a.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/1c583e3f061fad7aac1461d8b7dd79f8df034a3cac72568d90c1ae38691d8e5a.jpg)

![53fc51c90f9c6ee9c3d0e10e4206e1d7f5349be218e6921da844d0b7971f5cc4.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/53fc51c90f9c6ee9c3d0e10e4206e1d7f5349be218e6921da844d0b7971f5cc4.jpg)

![5433a7b5562f2a9556eccc77754f53a40ef4fec489d56328b5a321cd8670a6ed.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/5433a7b5562f2a9556eccc77754f53a40ef4fec489d56328b5a321cd8670a6ed.jpg)

![5c66e33cdcea619483983035f8dc28a175b978b61530b0a7c1ea19290dda3d8e.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/5c66e33cdcea619483983035f8dc28a175b978b61530b0a7c1ea19290dda3d8e.jpg)

![69610cfe03f155f24bff6d7fd0b9a15d756893a338910c431500add7aee89af3.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/69610cfe03f155f24bff6d7fd0b9a15d756893a338910c431500add7aee89af3.jpg)

![9465315b6215674a3172832d69318c518121e0790f8d826704b5927045921b1d.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/9465315b6215674a3172832d69318c518121e0790f8d826704b5927045921b1d.jpg)

![aeef3d5c2ed26a85ce7bda469b776689fdf7b83dc6b799f852cf6870000d09c0.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/aeef3d5c2ed26a85ce7bda469b776689fdf7b83dc6b799f852cf6870000d09c0.jpg)

![e1444796b5b5c8910ff3346489d9522ed42bd791498157f7ce49487049266a13.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/images/e1444796b5b5c8910ff3346489d9522ed42bd791498157f7ce49487049266a13.jpg)

### Tables

![05338ac53580b247ff7cd16f34fa77bc70ff914218f4f8d8a848df83b0378744.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/05338ac53580b247ff7cd16f34fa77bc70ff914218f4f8d8a848df83b0378744.jpg)

![187c4909dfad9853b06e3e9d8c137cf0f451f7c87f9e44eb15e9ddf0df71bd70.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/187c4909dfad9853b06e3e9d8c137cf0f451f7c87f9e44eb15e9ddf0df71bd70.jpg)

![196bb3ca865180f2104c841de5d7a64016781cc4b1cec6167544f443f8944ab6.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/196bb3ca865180f2104c841de5d7a64016781cc4b1cec6167544f443f8944ab6.jpg)

![2bb4c91e793334ffa0b722d2dd8fe23e7f7abbcb8ee337cadaa1a9d736cd71df.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/2bb4c91e793334ffa0b722d2dd8fe23e7f7abbcb8ee337cadaa1a9d736cd71df.jpg)

![41299d1eb0fcff48e2a689d21d9f4b8b8fd7c9982129137354ea0fa9931ef5a3.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/41299d1eb0fcff48e2a689d21d9f4b8b8fd7c9982129137354ea0fa9931ef5a3.jpg)

![9020f16abee409f913e7434dd2971d49420fd4b9922c9f37a409610007e88be4.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/9020f16abee409f913e7434dd2971d49420fd4b9922c9f37a409610007e88be4.jpg)

![98d57da016f844b5435b2b6e6d64668d648b952522fdfee8a69fb73fe9f87ebd.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/98d57da016f844b5435b2b6e6d64668d648b952522fdfee8a69fb73fe9f87ebd.jpg)

![9f54129f1ee11643e08ddff5b96a13f05d1102e0ae3dbb53b30c8a478e0ab469.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/9f54129f1ee11643e08ddff5b96a13f05d1102e0ae3dbb53b30c8a478e0ab469.jpg)

![9f70b52ada134829278dacfe073e7b60ae5bceb7839a354c99ba65097c8f1164.jpg](../icml_results/771_Who%20Said%20Neural%20Networks%20Aren%E2%80%99t%20Linear_/tables/9f70b52ada134829278dacfe073e7b60ae5bceb7839a354c99ba65097c8f1164.jpg)

## Smoothness Errors in Dynamics Models and How to Avoid Them


### Images

![0077d91e31c9503b6c4e5011cf80a23c4a740d01ee5223734fb10711715273e1.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/0077d91e31c9503b6c4e5011cf80a23c4a740d01ee5223734fb10711715273e1.jpg)

![10771816101f5abfd7c267663d759b6f334e90e9f1158e6808fa9c98994ff95f.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/10771816101f5abfd7c267663d759b6f334e90e9f1158e6808fa9c98994ff95f.jpg)

![19eeed91c5ce768f012ddd7814258461dea0e2ccac9046014941e4ed02baf8c7.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/19eeed91c5ce768f012ddd7814258461dea0e2ccac9046014941e4ed02baf8c7.jpg)

![21e55b1bbc31743a7dc077c5be4693fdea5fb4245a5f737d9269b72acfb13bcc.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/21e55b1bbc31743a7dc077c5be4693fdea5fb4245a5f737d9269b72acfb13bcc.jpg)

![2f615af49b53476d2667c4a415e03ba2603e45db540e9fbab955a1c2a87bb7d4.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/2f615af49b53476d2667c4a415e03ba2603e45db540e9fbab955a1c2a87bb7d4.jpg)

![379a6cf129e53ae9c2bbc4f7928ef8883cdbd324a0ab383d6c93a4a3b2179414.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/379a6cf129e53ae9c2bbc4f7928ef8883cdbd324a0ab383d6c93a4a3b2179414.jpg)

![5439bb60a55b942e3ffe59a2e122582ecb02dc83c44a217415c48b3db87d3aeb.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/5439bb60a55b942e3ffe59a2e122582ecb02dc83c44a217415c48b3db87d3aeb.jpg)

![64a473b05b2dd308c8b9779a3a82086409eb0b970aa2e77582779e4d2cdb6051.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/64a473b05b2dd308c8b9779a3a82086409eb0b970aa2e77582779e4d2cdb6051.jpg)

![71c3dd365e5bbaa26ff7cd4157f78ba84943debf7f34e27456f26eaa393f87ef.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/71c3dd365e5bbaa26ff7cd4157f78ba84943debf7f34e27456f26eaa393f87ef.jpg)

![76b8c80109d27b960ffebbd4dfea783449d4b648dab8954122c3e1bb98572a67.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/76b8c80109d27b960ffebbd4dfea783449d4b648dab8954122c3e1bb98572a67.jpg)

![991ec032f529c3eb3a01aa3b0fd0b5d832deced161c1687cff8940e790d36eb5.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/991ec032f529c3eb3a01aa3b0fd0b5d832deced161c1687cff8940e790d36eb5.jpg)

![b20bb44f1eacb93670812d9d263324a598640eb88a8545e415eb4cdf533379bb.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/b20bb44f1eacb93670812d9d263324a598640eb88a8545e415eb4cdf533379bb.jpg)

![dab4386eb4f91410cfa1ccf7558f8a14d2ef426a681d5896a31691c427fb0f26.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/dab4386eb4f91410cfa1ccf7558f8a14d2ef426a681d5896a31691c427fb0f26.jpg)

![f68eaa4fad2eb8458cc9c632cdc1e9855d9a50342565eeeddf263f7af3e60b7c.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/f68eaa4fad2eb8458cc9c632cdc1e9855d9a50342565eeeddf263f7af3e60b7c.jpg)

![f9c18fc84c2715c8cc2e0382ba49a990e0683de0ef6261f103c3adf5bd635bed.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/images/f9c18fc84c2715c8cc2e0382ba49a990e0683de0ef6261f103c3adf5bd635bed.jpg)

### Tables

![0e25ca70778f97a881183d77660ce4799599479e14bcc3cbc14cf5ff99831745.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/0e25ca70778f97a881183d77660ce4799599479e14bcc3cbc14cf5ff99831745.jpg)

![230c7c91bb2d439587041f1b0c17e1d8d7bec8eba6c0461bb58d48b39967b6f6.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/230c7c91bb2d439587041f1b0c17e1d8d7bec8eba6c0461bb58d48b39967b6f6.jpg)

![269f4fb292b2f4ce4395d5418d0b7cecdf5cc4a5ae64f336a4e69641e1726b70.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/269f4fb292b2f4ce4395d5418d0b7cecdf5cc4a5ae64f336a4e69641e1726b70.jpg)

![533b41c6ffc70033ea0161eb2981f4aec60aa406d8769a630f262cfff059eb25.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/533b41c6ffc70033ea0161eb2981f4aec60aa406d8769a630f262cfff059eb25.jpg)

![73c81f5a935160250efb5fdb68020e624bde8b62aff2767a8e42b546583cd0f9.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/73c81f5a935160250efb5fdb68020e624bde8b62aff2767a8e42b546583cd0f9.jpg)

![7ef6c20c287e4d82a00fc08705f7ff342f794921d499b6ca5d5550b0879a9695.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/7ef6c20c287e4d82a00fc08705f7ff342f794921d499b6ca5d5550b0879a9695.jpg)

![8882a0824b9280aa66b1e56cebf4d0495a212c627d39181d300a193a8a3722f0.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/8882a0824b9280aa66b1e56cebf4d0495a212c627d39181d300a193a8a3722f0.jpg)

![a4e3a9e1478f3b0be6a476400a150ca8da324e0e8dd844a9058a0d75eef2c426.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/a4e3a9e1478f3b0be6a476400a150ca8da324e0e8dd844a9058a0d75eef2c426.jpg)

![b355dc4627c5d50610b468556c59aa12996c9f188a932647c3f1a0dbdab53aed.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/b355dc4627c5d50610b468556c59aa12996c9f188a932647c3f1a0dbdab53aed.jpg)

![e4c96ec3b3ccc6d8a467f22f3de8f633636105e9821b5e355102cd2e02fababc.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/e4c96ec3b3ccc6d8a467f22f3de8f633636105e9821b5e355102cd2e02fababc.jpg)

![ed633e0cd8df271ffdb9ccd6591931f640fa1f43609da88de4da247684bb244b.jpg](../icml_results/772_Smoothness%20Errors%20in%20Dynamics%20Models%20and%20How%20to%20Avoid%20Them/tables/ed633e0cd8df271ffdb9ccd6591931f640fa1f43609da88de4da247684bb244b.jpg)

<a id="paper-0761"></a>

## Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics


### Images

![032d9c4f318429fe2dc9280ae0051833c727624ac96d74b4b48c6c9a014841eb.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/032d9c4f318429fe2dc9280ae0051833c727624ac96d74b4b48c6c9a014841eb.jpg)

![1ddc231494d4ed1fb74d094383b781ab08d57c0fbfa42ea21f8b32696b4db74c.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/1ddc231494d4ed1fb74d094383b781ab08d57c0fbfa42ea21f8b32696b4db74c.jpg)

![3a2d0741a708dfcb752001a5310c669baa22f27b187186f146d9d515b8f2c27b.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/3a2d0741a708dfcb752001a5310c669baa22f27b187186f146d9d515b8f2c27b.jpg)

![681bfac5e570fe6a29a5522d9d0a9fe33e177720faacf039937a128293d384f9.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/681bfac5e570fe6a29a5522d9d0a9fe33e177720faacf039937a128293d384f9.jpg)

![b7257eb573a3a57d1e6ca22d64faee258340f2f2ec17908d8231250f0266983f.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/b7257eb573a3a57d1e6ca22d64faee258340f2f2ec17908d8231250f0266983f.jpg)

![c1da3d7e0df15809b428282ca30fe4fef3da227bec1f49b959b3f188534c26c9.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/c1da3d7e0df15809b428282ca30fe4fef3da227bec1f49b959b3f188534c26c9.jpg)

![c9646d7f497565d63057bb897fcc6a6d126b62152c38bfae6e6ff07fba5ee43d.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/images/c9646d7f497565d63057bb897fcc6a6d126b62152c38bfae6e6ff07fba5ee43d.jpg)

### Tables

![088e60c9cdc25720213bdeb3ee20a79ef2b8a84681613bbbc8e56dbb0b812e66.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/tables/088e60c9cdc25720213bdeb3ee20a79ef2b8a84681613bbbc8e56dbb0b812e66.jpg)

![452b9dca9c7951fa29dd57b727aa84bf07702ec59aa3c16fdecb566e18e1bdea.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/tables/452b9dca9c7951fa29dd57b727aa84bf07702ec59aa3c16fdecb566e18e1bdea.jpg)

![642580b0d626f84b08035188328218aa6a10c1ee57a268db9843ae83f31f6f3a.jpg](../icml_results/773_Revisiting%20Anisotropy%20in%20Language%20Transformers_%20The%20Geometry%20of%20Learning%20Dynamics/tables/642580b0d626f84b08035188328218aa6a10c1ee57a268db9843ae83f31f6f3a.jpg)

<a id="paper-0762"></a>

## Approximate Equivariance via Projection-Based Regularisation


### Images

![0078ab9a16a532ae8a5eec29097b8cf5bfc4e22f35680375b4519fb6e1704c0d.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/0078ab9a16a532ae8a5eec29097b8cf5bfc4e22f35680375b4519fb6e1704c0d.jpg)

![1c1c2aa359d47be039ee59474f99be0f3f07dd7affe80ad12daccecafc2021e6.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/1c1c2aa359d47be039ee59474f99be0f3f07dd7affe80ad12daccecafc2021e6.jpg)

![30be539f1bafe522996c73ec6beaf165ea65a202bce891c6f07cb73bf0616245.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/30be539f1bafe522996c73ec6beaf165ea65a202bce891c6f07cb73bf0616245.jpg)

![3599ffdf3d2529aa3d479f4eae6175a03877e8848b801ae0587ddb4d287f0909.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/3599ffdf3d2529aa3d479f4eae6175a03877e8848b801ae0587ddb4d287f0909.jpg)

![3d5d842ca538f155a6b2dfa4bc4aabe135b27528a4e471d8a6d93e678dd1b9c4.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/3d5d842ca538f155a6b2dfa4bc4aabe135b27528a4e471d8a6d93e678dd1b9c4.jpg)

![4d3cd8e19fcae60397b9275741746e4d9a7fdf90c72318661abd1032155357eb.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/4d3cd8e19fcae60397b9275741746e4d9a7fdf90c72318661abd1032155357eb.jpg)

![7fd063e2a4cf6530b0c4c4f47078aa623cdc9fa6c4861901119a506031663ab9.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/7fd063e2a4cf6530b0c4c4f47078aa623cdc9fa6c4861901119a506031663ab9.jpg)

![945ecbff144dbaefec2d11e9ae720f23e0aa51c5d990914592758de733ccdfa4.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/945ecbff144dbaefec2d11e9ae720f23e0aa51c5d990914592758de733ccdfa4.jpg)

![a796714c36a24436792b8993063522ab5f4fde4b9fd429ca671edb16d3fe28a0.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/a796714c36a24436792b8993063522ab5f4fde4b9fd429ca671edb16d3fe28a0.jpg)

![f4f9a1242de8cffcf0e38f4d274eef681470a7c6cb7767dc2a3f267d4ec66ac2.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/images/f4f9a1242de8cffcf0e38f4d274eef681470a7c6cb7767dc2a3f267d4ec66ac2.jpg)

### Tables

![2c3d9503583ee31488805d598fd18466395d1d1a3b0328ac6c7b20698b3b4bee.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/2c3d9503583ee31488805d598fd18466395d1d1a3b0328ac6c7b20698b3b4bee.jpg)

![3baf4ac17bc7064f0fffe1a15576cf4259c4b74a4779914f6ce1b29bf17e322b.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/3baf4ac17bc7064f0fffe1a15576cf4259c4b74a4779914f6ce1b29bf17e322b.jpg)

![3d72b55ea12130729b873d5b2208068e602f14eb886b2232e5a09a11e95f5b9e.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/3d72b55ea12130729b873d5b2208068e602f14eb886b2232e5a09a11e95f5b9e.jpg)

![63b14b62a17b50b65c507aa0d9bdd307c56254715f352798ee39ff6082de3aac.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/63b14b62a17b50b65c507aa0d9bdd307c56254715f352798ee39ff6082de3aac.jpg)

![766ec5fe557bf37e4fec7469a6c6753fb9b715a858148b5d0e6d786455c00bd9.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/766ec5fe557bf37e4fec7469a6c6753fb9b715a858148b5d0e6d786455c00bd9.jpg)

![842aece3cb2ef68432cee1d38645cdee9d5a0b03036c820b86e8089c5f5a9ed0.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/842aece3cb2ef68432cee1d38645cdee9d5a0b03036c820b86e8089c5f5a9ed0.jpg)

![a6f769dc29f9b8c6a0e3493a59de9285ceb3ac0124325bdd488e888ad324cea3.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/a6f769dc29f9b8c6a0e3493a59de9285ceb3ac0124325bdd488e888ad324cea3.jpg)

![b24c4944d743935c9c49b44e2b9366b851b8929f76ed0766f16a7251d35e395e.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/b24c4944d743935c9c49b44e2b9366b851b8929f76ed0766f16a7251d35e395e.jpg)

![bc72400a1d4feb0dafa0f13e7ba743776944a774109be651b49eb2422ea62d72.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/bc72400a1d4feb0dafa0f13e7ba743776944a774109be651b49eb2422ea62d72.jpg)

![d415e7c427801ec3c5ccbbcfb8e38e79b7fb66baedd909e10ba38568aafa6e9d.jpg](../icml_results/774_Approximate%20Equivariance%20via%20Projection-Based%20Regularisation/tables/d415e7c427801ec3c5ccbbcfb8e38e79b7fb66baedd909e10ba38568aafa6e9d.jpg)

<a id="paper-0763"></a>

## M-Star: Markovian Projection of Star-Shaped Diffusion for Exponential Family Distributions


### Images

![0be869668f1ec63befd963934e11cff47b5e691e6809d3802f3f67648d74a29d.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/0be869668f1ec63befd963934e11cff47b5e691e6809d3802f3f67648d74a29d.jpg)

![4fd8664b4ef09bf04f364b2d65ef2167bac88f481e6f8f63794dd5a9a6f61da2.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/4fd8664b4ef09bf04f364b2d65ef2167bac88f481e6f8f63794dd5a9a6f61da2.jpg)

![62efb44d39061cd7b68691ff1f7f5b142daaf9207a905ea646b73595f5233963.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/62efb44d39061cd7b68691ff1f7f5b142daaf9207a905ea646b73595f5233963.jpg)

![6ccabb24176dd1be30738907f48641e84c824a08b7b4f2ecbf7b2839a65a1f78.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/6ccabb24176dd1be30738907f48641e84c824a08b7b4f2ecbf7b2839a65a1f78.jpg)

![7f5e671303f1274150eca50923b9c16a7bf3c6c5360ad9f4b078339dd1aa14c4.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/7f5e671303f1274150eca50923b9c16a7bf3c6c5360ad9f4b078339dd1aa14c4.jpg)

![97d1c379b3dbd8d604fc1deff2fcb56feb24cbef1a7b9d20fb201dbc8a5df30d.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/97d1c379b3dbd8d604fc1deff2fcb56feb24cbef1a7b9d20fb201dbc8a5df30d.jpg)

![aaceb1381c7fa8fb98b39bcc84aabbdb7c7b785e676b33a2b3bad0fb84de41f1.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/aaceb1381c7fa8fb98b39bcc84aabbdb7c7b785e676b33a2b3bad0fb84de41f1.jpg)

![c33007108abb828f6bcbf9381d3d63d9360d7ecd910c2d25b9568eec543cf389.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/c33007108abb828f6bcbf9381d3d63d9360d7ecd910c2d25b9568eec543cf389.jpg)

![df34192625665cbaa2a0e2fbcc42977d28f9da8a0567b8ee6556227343f2d7cd.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/df34192625665cbaa2a0e2fbcc42977d28f9da8a0567b8ee6556227343f2d7cd.jpg)

![f232e330e7240abc6ae49ec882df0a68c82f4e6e3809b9d15da9b98305f435d5.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/images/f232e330e7240abc6ae49ec882df0a68c82f4e6e3809b9d15da9b98305f435d5.jpg)

### Tables

![05853614a123905b2485a7902bcad5017829efb1d936fdd86b2cc37d7f64493c.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/05853614a123905b2485a7902bcad5017829efb1d936fdd86b2cc37d7f64493c.jpg)

![08977a069b335f542b69fbe929e221b60b41663110434e62eb34e6afe98f3885.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/08977a069b335f542b69fbe929e221b60b41663110434e62eb34e6afe98f3885.jpg)

![23ff69556f592eb3d71f3d726324876c989e071798bae284025f130b27e58c02.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/23ff69556f592eb3d71f3d726324876c989e071798bae284025f130b27e58c02.jpg)

![411cde5d61eb9a4f3ee949a6ad762fdfad4dfd29e60cdd494ed8df551be61255.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/411cde5d61eb9a4f3ee949a6ad762fdfad4dfd29e60cdd494ed8df551be61255.jpg)

![4a44c9afa5c7555b9f9d41984d33e45e4d57079ac369be0fe55369627ecb9570.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/4a44c9afa5c7555b9f9d41984d33e45e4d57079ac369be0fe55369627ecb9570.jpg)

![5c93c864de545020e478a1ee76c641e738d3ed158034034614f4deeaf3cbd38e.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/5c93c864de545020e478a1ee76c641e738d3ed158034034614f4deeaf3cbd38e.jpg)

![5f907606ae273ed9d993269879755a122d53ccb7ef3fde04c738b33494bca5b7.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/5f907606ae273ed9d993269879755a122d53ccb7ef3fde04c738b33494bca5b7.jpg)

![a563aa7b16a2b8e7e5f29fb219635028b100631e9b997305081f18d0a2c2d86d.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/a563aa7b16a2b8e7e5f29fb219635028b100631e9b997305081f18d0a2c2d86d.jpg)

![a6e51889237571b4116db80147412f1f7d9287daef00e8e85e0ef8d6e6ec5e55.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/a6e51889237571b4116db80147412f1f7d9287daef00e8e85e0ef8d6e6ec5e55.jpg)

![aec95905034c0e4d267587dbe4d6c35992035fbe719bf00626f0efcd9a8d3046.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/aec95905034c0e4d267587dbe4d6c35992035fbe719bf00626f0efcd9a8d3046.jpg)

![af15503be3bb46601d43605d0c408ea165ba7e4717d3ed9bdba3ccadb881d5db.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/af15503be3bb46601d43605d0c408ea165ba7e4717d3ed9bdba3ccadb881d5db.jpg)

![c4d8757273ea728927d2cc073899e5417fb9970c4d32c66ea7da001852240f78.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/c4d8757273ea728927d2cc073899e5417fb9970c4d32c66ea7da001852240f78.jpg)

![d3157b9f7212a4ba10e406fc383ce9e91d747dfe8d78b6a94203d1b056352f2c.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/d3157b9f7212a4ba10e406fc383ce9e91d747dfe8d78b6a94203d1b056352f2c.jpg)

![e1bf511d1215cdc1db20bc801e8f98e134b11bee21193959a998f7e8a02104c3.jpg](../icml_results/775_M-Star_%20Markovian%20Projection%20of%20Star-Shaped%20Diffusion%20for%20Exponential%20Family%20Distributions/tables/e1bf511d1215cdc1db20bc801e8f98e134b11bee21193959a998f7e8a02104c3.jpg)

<a id="paper-0764"></a>

## Motion Planning in Compressed Representation Spaces


### Images

![02213fed4a2d7fad27907acc61c1552d3f5bd1ca9446a473821bbf892b64894a.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/02213fed4a2d7fad27907acc61c1552d3f5bd1ca9446a473821bbf892b64894a.jpg)

![030736554e11bb10643102543f573addb1e3117e9f4915f292d9c954a0a735b6.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/030736554e11bb10643102543f573addb1e3117e9f4915f292d9c954a0a735b6.jpg)

![12f433b59d204102fbf7879f74e351565a02f4eaddc4c3417b3fbb34d140889d.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/12f433b59d204102fbf7879f74e351565a02f4eaddc4c3417b3fbb34d140889d.jpg)

![297afcfe9255d6f4d349963f79ac38caac79b20158024fda17856ae59e678713.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/297afcfe9255d6f4d349963f79ac38caac79b20158024fda17856ae59e678713.jpg)

![3e741709f616ac2a74e8a62294497c59d13e023aca11a7bf90c425458f4eccad.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/3e741709f616ac2a74e8a62294497c59d13e023aca11a7bf90c425458f4eccad.jpg)

![4742031ed2eb396c7fb587ea43d129d9acb424ddf5def9a5dd142a969b16d92e.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/4742031ed2eb396c7fb587ea43d129d9acb424ddf5def9a5dd142a969b16d92e.jpg)

![50b2cc48046f02d20e66d285191f5056d8111f1399ddf6886fbd7d3e1bd9744b.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/50b2cc48046f02d20e66d285191f5056d8111f1399ddf6886fbd7d3e1bd9744b.jpg)

![53c2e5c93445c4c160e2b4c582109504eef3791dd8b42dedbb5aa92d4628838f.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/53c2e5c93445c4c160e2b4c582109504eef3791dd8b42dedbb5aa92d4628838f.jpg)

![61f9061e3858a8709c728a4fd39617f4ce9dc8f7509525fd4548fb11a633090d.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/61f9061e3858a8709c728a4fd39617f4ce9dc8f7509525fd4548fb11a633090d.jpg)

![707d222d4948ca6f3e4e35144eec35ef375ee1c35ac2bba93faa939888ada4da.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/707d222d4948ca6f3e4e35144eec35ef375ee1c35ac2bba93faa939888ada4da.jpg)

![79830d23993d0c6197d49a38254bcaf92d48c91236e30a2d8a495012c63104d6.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/79830d23993d0c6197d49a38254bcaf92d48c91236e30a2d8a495012c63104d6.jpg)

![80eb74c0aa39f5a8fa7c62c66a9dc57398eb68cce44ed2c6a395d3fe260d584c.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/80eb74c0aa39f5a8fa7c62c66a9dc57398eb68cce44ed2c6a395d3fe260d584c.jpg)

![a3294277eaab0ee3cae27b72a4e7bff437eb907de89325010551bc035701accb.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/a3294277eaab0ee3cae27b72a4e7bff437eb907de89325010551bc035701accb.jpg)

![bf65243010dfb6986530fac1ea87e529fc18f20eb966f5f2e26a183e45a8d443.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/bf65243010dfb6986530fac1ea87e529fc18f20eb966f5f2e26a183e45a8d443.jpg)

![cb24cf9868ef69ebb62991399859bdccdad65cbd3516607301ee6cac08cc1cd6.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/cb24cf9868ef69ebb62991399859bdccdad65cbd3516607301ee6cac08cc1cd6.jpg)

![cb6bd1111f2b759f8cd722e2135740bad741b9c74cbf9b3df73e2d49be7a52e3.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/cb6bd1111f2b759f8cd722e2135740bad741b9c74cbf9b3df73e2d49be7a52e3.jpg)

![dc052688a7feb5e430268116544f58bb5110f54d4b4730a644a3c66c9aa13811.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/dc052688a7feb5e430268116544f58bb5110f54d4b4730a644a3c66c9aa13811.jpg)

![fea418a2c9cf503aacd80aba8e58d251bf287df1546b0cca1563ab7b37a929c6.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/images/fea418a2c9cf503aacd80aba8e58d251bf287df1546b0cca1563ab7b37a929c6.jpg)

### Tables

![208df47258a5bbb5bf56588f348c56ef7fc59cf64b37602837e33e18ed1bd492.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/208df47258a5bbb5bf56588f348c56ef7fc59cf64b37602837e33e18ed1bd492.jpg)

![22c027031f88cdfb9ad40874a7b8c76d770242ce2e1fcdacb6493da98ff03f3a.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/22c027031f88cdfb9ad40874a7b8c76d770242ce2e1fcdacb6493da98ff03f3a.jpg)

![427ae1c9f7d197688b81ca769d79e6e7a3109fc70cd88848d9464b1fcd059fa5.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/427ae1c9f7d197688b81ca769d79e6e7a3109fc70cd88848d9464b1fcd059fa5.jpg)

![6054402353d8cacae77f2e1a7fa02311e81895ee9d25ccdd434c1bcf291a2faa.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/6054402353d8cacae77f2e1a7fa02311e81895ee9d25ccdd434c1bcf291a2faa.jpg)

![9f83f0adb380a42eb3cdb5528479fe7ada2d50bb96b64a24a95264c6de0e9743.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/9f83f0adb380a42eb3cdb5528479fe7ada2d50bb96b64a24a95264c6de0e9743.jpg)

![b884e00061b062bcaaebcefed8799d5120eb0c52bad612bfd6c280195b0fbaa4.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/b884e00061b062bcaaebcefed8799d5120eb0c52bad612bfd6c280195b0fbaa4.jpg)

![c6eb033178237c951e831d091bfc708cb52f31acae0382b8c046f670edd93df3.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/c6eb033178237c951e831d091bfc708cb52f31acae0382b8c046f670edd93df3.jpg)

![c9a00dbdc74fc6c99f8d455000b44fd0bd788947292a446f4c3d42550eef5453.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/c9a00dbdc74fc6c99f8d455000b44fd0bd788947292a446f4c3d42550eef5453.jpg)

![d59df499b7da5abc2193808e96255190c65fdedb0088c0396b6ef6c0e71298ae.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/d59df499b7da5abc2193808e96255190c65fdedb0088c0396b6ef6c0e71298ae.jpg)

![ef2f36c0b45e137c77fc83af870dceca9681f9d1b1bdd41940c2c68f37b08c98.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/ef2f36c0b45e137c77fc83af870dceca9681f9d1b1bdd41940c2c68f37b08c98.jpg)

![fe84bde74c41f778aaa9adeaaf263fbb02031840d09a4bd04c069ccf73c4f94f.jpg](../icml_results/776_Motion%20Planning%20in%20Compressed%20Representation%20Spaces/tables/fe84bde74c41f778aaa9adeaaf263fbb02031840d09a4bd04c069ccf73c4f94f.jpg)

<a id="paper-0765"></a>

## RSA-CP: Efficient Conformal Prediction in Small-Sample Regimes via Random Score Alignment


### Images

![013fadee55839474a48066fbbbfaee46c6c61185411a2b016fe0edc71f533afa.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/013fadee55839474a48066fbbbfaee46c6c61185411a2b016fe0edc71f533afa.jpg)

![3c80f955ce6264f078c1e0a53a992010f2e9bace7e1dae72bf2ff9872b33011e.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/3c80f955ce6264f078c1e0a53a992010f2e9bace7e1dae72bf2ff9872b33011e.jpg)

![5a1b75591290302be03ba399ebf2e588b564d07e672aa22cac0679d4ab5c850e.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/5a1b75591290302be03ba399ebf2e588b564d07e672aa22cac0679d4ab5c850e.jpg)

![789a6fa220eb345ea58a5bd52cbf5e7c59d5ea8eb6b5872755b7f9a5b4815531.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/789a6fa220eb345ea58a5bd52cbf5e7c59d5ea8eb6b5872755b7f9a5b4815531.jpg)

![912c7f5bbf075d8b64816a7ca120df98ec100dc7331d35a8b968ebc6f15cec3e.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/912c7f5bbf075d8b64816a7ca120df98ec100dc7331d35a8b968ebc6f15cec3e.jpg)

![91d35e821cc620095ba8b6e074e3e995f691c1dae53aaeccaea87cf00d792d54.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/91d35e821cc620095ba8b6e074e3e995f691c1dae53aaeccaea87cf00d792d54.jpg)

![b13b711580f5fc743796fefe25c3e955db9762e21ee0cb34ff9f462b50ce7b08.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/b13b711580f5fc743796fefe25c3e955db9762e21ee0cb34ff9f462b50ce7b08.jpg)

![f3ca911e26a359c3f5334552ae245cb3ce30cdf125d76c0a1ecd13080f80c46a.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/f3ca911e26a359c3f5334552ae245cb3ce30cdf125d76c0a1ecd13080f80c46a.jpg)

![fb4974feffdb4bba778af6f222b4aed69859438049427f5b96948e0717e21973.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/fb4974feffdb4bba778af6f222b4aed69859438049427f5b96948e0717e21973.jpg)

![ff0ec2063bc7ec3d8b72c57e5de3b0a632af89c51e4fad29e97e36f2f28c3512.jpg](../icml_results/777_RSA-CP_%20Efficient%20Conformal%20Prediction%20in%20Small-Sample%20Regimes%20via%20Random%20Score%20Alignment/images/ff0ec2063bc7ec3d8b72c57e5de3b0a632af89c51e4fad29e97e36f2f28c3512.jpg)

<a id="paper-0766"></a>

## Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long-Form Text Generation


### Images

![07b2bf6a812e9638eb4305f433db043686e6c12049eb9c5910325c5f6c3a40d4.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/07b2bf6a812e9638eb4305f433db043686e6c12049eb9c5910325c5f6c3a40d4.jpg)

![0d86da76e003e0727dd3f927ae57140dacd3d47e33c67b49128d5fb7951b0f13.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/0d86da76e003e0727dd3f927ae57140dacd3d47e33c67b49128d5fb7951b0f13.jpg)

![1ab3e7966a5b35b2ea7ca0308a586c9593a06eb3bca11db6b3c9fc303b4cf67b.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/1ab3e7966a5b35b2ea7ca0308a586c9593a06eb3bca11db6b3c9fc303b4cf67b.jpg)

![1dcd0db9f869837bb4b884f438e6a99080cf6c56f9e44ef6c21f28152371d391.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/1dcd0db9f869837bb4b884f438e6a99080cf6c56f9e44ef6c21f28152371d391.jpg)

![4f50dfb60e4141f1a189548ae6131a0acb9d249c5ea52ea14a986f52ab6f4f04.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/4f50dfb60e4141f1a189548ae6131a0acb9d249c5ea52ea14a986f52ab6f4f04.jpg)

![5449dc61971fc8a38365c76ee6bbf725ac20b39ca29b2c404824036ee1688863.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/5449dc61971fc8a38365c76ee6bbf725ac20b39ca29b2c404824036ee1688863.jpg)

![5b41188de6b41311db44505fe7cfb7d9ba0f6242df8b0f355bc5ab13f66ef5d6.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/5b41188de6b41311db44505fe7cfb7d9ba0f6242df8b0f355bc5ab13f66ef5d6.jpg)

![6cc2e41becf7f09bf6140b42cead8847122621f35458f29ce1d8fd8198018f0b.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/6cc2e41becf7f09bf6140b42cead8847122621f35458f29ce1d8fd8198018f0b.jpg)

![6f23d7059e8e2f5ac68011c67d66796ac5007386bb6877c92d84ed49ae8c64d3.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/6f23d7059e8e2f5ac68011c67d66796ac5007386bb6877c92d84ed49ae8c64d3.jpg)

![9c194e715558fbca69a75eeaae1c6afd83e0a8731da023989b06667ced02cc40.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/9c194e715558fbca69a75eeaae1c6afd83e0a8731da023989b06667ced02cc40.jpg)

![c3ac42ad512a29d3c108c8381bba9d398bcbf1ee7151ef2f5e5cfe98ff674ec3.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/c3ac42ad512a29d3c108c8381bba9d398bcbf1ee7151ef2f5e5cfe98ff674ec3.jpg)

![eeaf2b6d116b9a0389d69221006b90fd9015b68627509d52e32b8dfb1d050265.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/images/eeaf2b6d116b9a0389d69221006b90fd9015b68627509d52e32b8dfb1d050265.jpg)

### Tables

![351697fd9af1bbfca97af60670f8741c6d91fff59ebaac2997c07003c03c0154.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/tables/351697fd9af1bbfca97af60670f8741c6d91fff59ebaac2997c07003c03c0154.jpg)

![cc79ada39f1277bf17bf67e2364d4af5f9e3174052a0d57430b3eb0fbf8de608.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/tables/cc79ada39f1277bf17bf67e2364d4af5f9e3174052a0d57430b3eb0fbf8de608.jpg)

![d205dd57e138e4c171a940f3ad5ba3641a2a1ad572a739aca5085280c0753050.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/tables/d205dd57e138e4c171a940f3ad5ba3641a2a1ad572a739aca5085280c0753050.jpg)

![d5dcf5cca3f67188615a2490fd4f4cdf2c69500aa8b12f1013b2bfc9b48c88a9.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/tables/d5dcf5cca3f67188615a2490fd4f4cdf2c69500aa8b12f1013b2bfc9b48c88a9.jpg)

![edde95d758b6a95606fa33ddc5919539411c66656d791b6f46d33c5ef9476910.jpg](../icml_results/778_Embedding%20Trust_%20Semantic%20Isotropy%20Predicts%20Nonfactuality%20in%20Long-Form%20Text%20Generation/tables/edde95d758b6a95606fa33ddc5919539411c66656d791b6f46d33c5ef9476910.jpg)

<a id="paper-0767"></a>

## CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings


### Images

![01c58380154929269453a97fdd0c91d852cc8c0266b7a5f237e5d378bad1f73c.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/01c58380154929269453a97fdd0c91d852cc8c0266b7a5f237e5d378bad1f73c.jpg)

![0906f4c7a088319dbdee89a9e39532ec36303ff938d1c7114ff6bfd59e230e10.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/0906f4c7a088319dbdee89a9e39532ec36303ff938d1c7114ff6bfd59e230e10.jpg)

![13506378fe1b1c25164de827c26382fe9303a3c1b4a83b182e5c94c42bf8d661.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/13506378fe1b1c25164de827c26382fe9303a3c1b4a83b182e5c94c42bf8d661.jpg)

![24795c7eab6a6ebf7c0fcdc4e56ba5a45fbf985a5516313cac0f22dbdb80108a.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/24795c7eab6a6ebf7c0fcdc4e56ba5a45fbf985a5516313cac0f22dbdb80108a.jpg)

![555e30c7b3ae03b036d7c8bd1edc9bcff91dfa9a91d8aafa08759f23180cbbbb.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/555e30c7b3ae03b036d7c8bd1edc9bcff91dfa9a91d8aafa08759f23180cbbbb.jpg)

![87b27c51e28fedb5ad6e7cfd9dce5bc562537d0536489aec90d3687f0f8928e7.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/87b27c51e28fedb5ad6e7cfd9dce5bc562537d0536489aec90d3687f0f8928e7.jpg)

![aec4a40453216a3589e9ea1b028297b8a0fb180aaad4adbdb974235052089fc4.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/aec4a40453216a3589e9ea1b028297b8a0fb180aaad4adbdb974235052089fc4.jpg)

![d08daae72607853979875f9a2b5ec4c30a98a8cb0a618508ca283e354476e144.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/images/d08daae72607853979875f9a2b5ec4c30a98a8cb0a618508ca283e354476e144.jpg)

### Tables

![0755cdfc64065069a6bee9af194f24a016728f1166385916285a651edb4405fa.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/0755cdfc64065069a6bee9af194f24a016728f1166385916285a651edb4405fa.jpg)

![07cc9628e4473098d30de35cc479be0fc82522d6fb4725a007de13a7903a01dc.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/07cc9628e4473098d30de35cc479be0fc82522d6fb4725a007de13a7903a01dc.jpg)

![1339aa388a0999e74f87d6d9a3a5d5ac7f7b0a4cc5e878c7e5db0f891aeb6b04.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/1339aa388a0999e74f87d6d9a3a5d5ac7f7b0a4cc5e878c7e5db0f891aeb6b04.jpg)

![15848231fd4b18a533ff97c37275520395ef7224eee74574ad8408a4ed751931.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/15848231fd4b18a533ff97c37275520395ef7224eee74574ad8408a4ed751931.jpg)

![18bf0812efbadbe809239c262db7239b6d5030f4eef75e5101ece3a50e1842df.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/18bf0812efbadbe809239c262db7239b6d5030f4eef75e5101ece3a50e1842df.jpg)

![25c7cb15dac4b49794a446af12abe193d0849930d1b61386d17fbe2c9e8016bf.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/25c7cb15dac4b49794a446af12abe193d0849930d1b61386d17fbe2c9e8016bf.jpg)

![4f7205b246849030df3b232d908346fd64bf24f26e47c6d25328aaefb48c3c9e.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/4f7205b246849030df3b232d908346fd64bf24f26e47c6d25328aaefb48c3c9e.jpg)

![53cee2ddf2738ab943feba78611ff1de46364e6e947d1c4e11649809f7b5c5ee.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/53cee2ddf2738ab943feba78611ff1de46364e6e947d1c4e11649809f7b5c5ee.jpg)

![60121b05b879bbe8d5599858a728ef0c09f1952c3913a60b20ea2e523b45ec10.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/60121b05b879bbe8d5599858a728ef0c09f1952c3913a60b20ea2e523b45ec10.jpg)

![80e19a23ebc6829ba4ceedc2c350a7b37031da493d8ed5de9b9d0334b6fac884.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/80e19a23ebc6829ba4ceedc2c350a7b37031da493d8ed5de9b9d0334b6fac884.jpg)

![94b08dba7f42f0f4de91c1c830fbb676ffed980ab43a641135c26213a3fddc00.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/94b08dba7f42f0f4de91c1c830fbb676ffed980ab43a641135c26213a3fddc00.jpg)

![94d51ebe64cfd861252ad398ceafc512954231bdeb251401d5454ef39726896a.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/94d51ebe64cfd861252ad398ceafc512954231bdeb251401d5454ef39726896a.jpg)

![a62d8e93c9eb58630eb3d1d6eaf467597c1924125386b3b210e8a01cce29bf0e.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/a62d8e93c9eb58630eb3d1d6eaf467597c1924125386b3b210e8a01cce29bf0e.jpg)

![a71110ee02da320db596aff9600c14d94555da1d138ec79bd31086ba2db4bff9.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/a71110ee02da320db596aff9600c14d94555da1d138ec79bd31086ba2db4bff9.jpg)

![cec770c5ce2400b1bf0de6572df2d44a189095c72eeba9a6a32b181979c7feab.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/cec770c5ce2400b1bf0de6572df2d44a189095c72eeba9a6a32b181979c7feab.jpg)

![cf42db9d3f43d85d7f8cf01ba0916c8a3aa05c83fad9be4e211306d6ef6b67e2.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/cf42db9d3f43d85d7f8cf01ba0916c8a3aa05c83fad9be4e211306d6ef6b67e2.jpg)

![ec77e3db511e4ed5e51b6773007f2fa47d3dd1e76693dcff4b4d96e509062ee5.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/ec77e3db511e4ed5e51b6773007f2fa47d3dd1e76693dcff4b4d96e509062ee5.jpg)

![ef83f8be5e2a74c57d8f1bdea105af2619c0aca93cfc7b16da8ef37bcbe5c793.jpg](../icml_results/779_CLoVE_%20Personalized%20Federated%20Learning%20through%20Clustering%20of%20Loss%20Vector%20Embeddings/tables/ef83f8be5e2a74c57d8f1bdea105af2619c0aca93cfc7b16da8ef37bcbe5c793.jpg)

<a id="paper-0768"></a>

## Dynamic High-Dimensional Facility Location with Low Recourse


### Tables

![932828d95e142f71d4e6b9097b581d7232e72a466b1e36bf916efc6a8491ea6e.jpg](../icml_results/780_Dynamic%20High-Dimensional%20Facility%20Location%20with%20Low%20Recourse/tables/932828d95e142f71d4e6b9097b581d7232e72a466b1e36bf916efc6a8491ea6e.jpg)

![9d11e4e2c07e5c22c97ce82f0158ab390e88dab7808522837bff7c1ff1ce74b9.jpg](../icml_results/780_Dynamic%20High-Dimensional%20Facility%20Location%20with%20Low%20Recourse/tables/9d11e4e2c07e5c22c97ce82f0158ab390e88dab7808522837bff7c1ff1ce74b9.jpg)

![c297bb86de0fddcaa5d6e948e6f38fd8f5410b8e1a024edc53a5c9e080efaf18.jpg](../icml_results/780_Dynamic%20High-Dimensional%20Facility%20Location%20with%20Low%20Recourse/tables/c297bb86de0fddcaa5d6e948e6f38fd8f5410b8e1a024edc53a5c9e080efaf18.jpg)

![f37e9ecc5aaf50f581486baa49ffa74dbd380b0b77dd03d283c9f549aa68c64c.jpg](../icml_results/780_Dynamic%20High-Dimensional%20Facility%20Location%20with%20Low%20Recourse/tables/f37e9ecc5aaf50f581486baa49ffa74dbd380b0b77dd03d283c9f549aa68c64c.jpg)

<a id="paper-0769"></a>

## EchoRL: Reinforcement Learning via Rollout Echoing


### Images

![1c10328e8f7f360844dd0969837b137be260af2844e2e769c58549d736e4205c.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/1c10328e8f7f360844dd0969837b137be260af2844e2e769c58549d736e4205c.jpg)

![2348908dd60613d7a295b51b07d915e2c16314dbfcf36819b08de65a94bd3114.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/2348908dd60613d7a295b51b07d915e2c16314dbfcf36819b08de65a94bd3114.jpg)

![2840c7441b8a60ea1ea7e6ac47de4b479b1d49c894c7923618b7b2c922c6de38.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/2840c7441b8a60ea1ea7e6ac47de4b479b1d49c894c7923618b7b2c922c6de38.jpg)

![2e4089619b44fea73f869966dd0ee0520a6261251ca53e7da63cfc0efecc3d7b.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/2e4089619b44fea73f869966dd0ee0520a6261251ca53e7da63cfc0efecc3d7b.jpg)

![73f280bb47454ee72d80c605aaa5d91d0d01d842353d3898e696a1da8bd62040.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/73f280bb47454ee72d80c605aaa5d91d0d01d842353d3898e696a1da8bd62040.jpg)

![83974cd0fa106d7ed7a8fb58c3d6654339deaa4b9a7e4df89bf9457ef01a0b09.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/83974cd0fa106d7ed7a8fb58c3d6654339deaa4b9a7e4df89bf9457ef01a0b09.jpg)

![8984e14971567496a1fab96425dc8b1c7e4d19ac50cb8ce95b487cf496f64b1b.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/8984e14971567496a1fab96425dc8b1c7e4d19ac50cb8ce95b487cf496f64b1b.jpg)

![a775573de370c097b3e3daf67d5d1cccb6f88bd843bc05e38694a7e20555e931.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/a775573de370c097b3e3daf67d5d1cccb6f88bd843bc05e38694a7e20555e931.jpg)

![b55df803aa376667c4ac2bdc3ec65f421c7241f3d69ab75b13fb58f348263053.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/b55df803aa376667c4ac2bdc3ec65f421c7241f3d69ab75b13fb58f348263053.jpg)

![b9cc149a2efbc2b03a53224de030a3b17d48737195c69642c7b0056919626dde.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/b9cc149a2efbc2b03a53224de030a3b17d48737195c69642c7b0056919626dde.jpg)

![c9aae802d938e0b623fb6374553206900129afeb5afe0e8e1179b52711c41cb8.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/c9aae802d938e0b623fb6374553206900129afeb5afe0e8e1179b52711c41cb8.jpg)

![d39201ebb6baeaf42c09d21f433a7c7f7532265504546e74c81c5ca0d7fa0800.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/images/d39201ebb6baeaf42c09d21f433a7c7f7532265504546e74c81c5ca0d7fa0800.jpg)

### Tables

![189f9f6c47f138b6e1f71e00ce5e687a27e2299893bf7a57d748a004e4e46a14.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/189f9f6c47f138b6e1f71e00ce5e687a27e2299893bf7a57d748a004e4e46a14.jpg)

![494733119744b18a879e49fd3c36dd582dc169762cce3b3d53158318779fcb0b.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/494733119744b18a879e49fd3c36dd582dc169762cce3b3d53158318779fcb0b.jpg)

![657293229f0c62e4b15871a7aa2532fb3cd14b2bbd14cb3e4a9b18c575608aed.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/657293229f0c62e4b15871a7aa2532fb3cd14b2bbd14cb3e4a9b18c575608aed.jpg)

![686d3ed6afbeff80e9b67fed785743181a93cb0513108f303ff73b4d87717ce7.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/686d3ed6afbeff80e9b67fed785743181a93cb0513108f303ff73b4d87717ce7.jpg)

![6c8a9071ae82993efb07b69a45548e3301314b58db247c73f8d8d09ee977a6cf.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/6c8a9071ae82993efb07b69a45548e3301314b58db247c73f8d8d09ee977a6cf.jpg)

![8d1544a0e9aefc8b38e9fbb7ef271cd5a6afbf55e889d0c73cc3c24e194a0bc1.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/8d1544a0e9aefc8b38e9fbb7ef271cd5a6afbf55e889d0c73cc3c24e194a0bc1.jpg)

![a8939d916b23732b28cf2cdacde4364420c8f1e2010926067a3d7fd51b48e7a2.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/a8939d916b23732b28cf2cdacde4364420c8f1e2010926067a3d7fd51b48e7a2.jpg)

![eee7e1732b7d8306379401b19bdc1436931728b28a3c13fc2644135a80094cf0.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/eee7e1732b7d8306379401b19bdc1436931728b28a3c13fc2644135a80094cf0.jpg)

![faa832574aa9e5af3168dc7b6692412098716474c645687b3e245172bb1dc20f.jpg](../icml_results/782_EchoRL_%20Reinforcement%20Learning%20via%20Rollout%20Echoing/tables/faa832574aa9e5af3168dc7b6692412098716474c645687b3e245172bb1dc20f.jpg)

<a id="paper-0770"></a>

## FedFit: Federated Dynamic Sparse Training via Fisher Information scoring


### Images

![01fe72aa174a9b085f9155b21b8b37cb4b8ea255e4341e512af614527c73f414.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/images/01fe72aa174a9b085f9155b21b8b37cb4b8ea255e4341e512af614527c73f414.jpg)

![06c2b1780da4b414a8f982b206b8778211d2e6dd758e9183d4b50c2572c22e92.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/images/06c2b1780da4b414a8f982b206b8778211d2e6dd758e9183d4b50c2572c22e92.jpg)

![14dadf3095a144b209f2a12d7cc22b09fdb03d9a728ebfcf923a1d094eba475c.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/images/14dadf3095a144b209f2a12d7cc22b09fdb03d9a728ebfcf923a1d094eba475c.jpg)

![3e108a093a85a2dbdc64cb5385c25dcac9e215c6989a85551c6571b261e26bf3.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/images/3e108a093a85a2dbdc64cb5385c25dcac9e215c6989a85551c6571b261e26bf3.jpg)

![5e1aee829ae7996dc8609e86f6c8126672c47ffcfebed5b36a9b6351d5a345dc.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/images/5e1aee829ae7996dc8609e86f6c8126672c47ffcfebed5b36a9b6351d5a345dc.jpg)

![617b2427e56589a48a68674eba05538eed9243c8a2c22e734a1c7d7bbf19e83f.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/images/617b2427e56589a48a68674eba05538eed9243c8a2c22e734a1c7d7bbf19e83f.jpg)

### Tables

![0a7cd8585335ab48410117d7a45b70a432979ef1388ee30c388f287a700a53b0.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/tables/0a7cd8585335ab48410117d7a45b70a432979ef1388ee30c388f287a700a53b0.jpg)

![212af251413285960413cf72509699e6236758bda8a4c1b4d6adab6cc1737191.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/tables/212af251413285960413cf72509699e6236758bda8a4c1b4d6adab6cc1737191.jpg)

![4b04413018817cd6b48d6accc06fc7017cd9b73fe0f83fe21185e450d3b6959a.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/tables/4b04413018817cd6b48d6accc06fc7017cd9b73fe0f83fe21185e450d3b6959a.jpg)

![8450077901297ee54f1dedc38d1937945334344fb7ef3450a1afe516c097d4b3.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/tables/8450077901297ee54f1dedc38d1937945334344fb7ef3450a1afe516c097d4b3.jpg)

![94fdf938e28b929e726168452127bb4acd195c48ed2f4b6d64162bc778648233.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/tables/94fdf938e28b929e726168452127bb4acd195c48ed2f4b6d64162bc778648233.jpg)

![f4fc2f85f26231eaae37681c9ea5b166c7ed90cceb2970d9eaf639b5adf56694.jpg](../icml_results/783_FedFit_%20Federated%20Dynamic%20Sparse%20Training%20via%20Fisher%20Information%20scoring/tables/f4fc2f85f26231eaae37681c9ea5b166c7ed90cceb2970d9eaf639b5adf56694.jpg)

<a id="paper-0771"></a>

## Does Reasoning Improve Seeing? Understanding When Vision-Language Models Benefit from Thinking


### Images

![12631e8b85893f47af097c165bd6f4d503134b2119a2140f65b831ae8c2c171f.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/images/12631e8b85893f47af097c165bd6f4d503134b2119a2140f65b831ae8c2c171f.jpg)

![25eb1ef24933b85f983637c4ff665e4a2feb622d7a346b0cd530c778a667e158.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/images/25eb1ef24933b85f983637c4ff665e4a2feb622d7a346b0cd530c778a667e158.jpg)

![2ef04f9ffaa6dbee71833b1ee739ef34dce9fd31451b71272efd0828fbac7595.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/images/2ef04f9ffaa6dbee71833b1ee739ef34dce9fd31451b71272efd0828fbac7595.jpg)

![9f7ca9189ff4fe005c2c89d6593c22ce4364f28df5dfd3d1b646c76747053d4b.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/images/9f7ca9189ff4fe005c2c89d6593c22ce4364f28df5dfd3d1b646c76747053d4b.jpg)

![abd57eea37a39a7b17feb1242d76bc7f2142074331e6304826ff434ce6331706.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/images/abd57eea37a39a7b17feb1242d76bc7f2142074331e6304826ff434ce6331706.jpg)

![b388a68bc63e696eda310c913a63d6ab1d47d14e3bbd8e5497843046ef16b57a.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/images/b388a68bc63e696eda310c913a63d6ab1d47d14e3bbd8e5497843046ef16b57a.jpg)

### Tables

![2e779ede798ac978a769fcf47d82295101b8eb219d9b3aadb24a6b5c5d443fb2.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/2e779ede798ac978a769fcf47d82295101b8eb219d9b3aadb24a6b5c5d443fb2.jpg)

![2fe3139de5bc19fbb1d2210d82af13a8b61be4cebb5e36f270cf51aa1a4231b1.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/2fe3139de5bc19fbb1d2210d82af13a8b61be4cebb5e36f270cf51aa1a4231b1.jpg)

![3cf30ea4303aef7d52f70330bcc1548d6345112a0b9e782de0fbbd28e8624897.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/3cf30ea4303aef7d52f70330bcc1548d6345112a0b9e782de0fbbd28e8624897.jpg)

![47ce385b302507624535a2d7723584f1a2fbc5d7acc5ddedb55877bc858e9335.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/47ce385b302507624535a2d7723584f1a2fbc5d7acc5ddedb55877bc858e9335.jpg)

![6e2ac7ea1bbf9c1f829581cd916f561819dd81e9ce0b2fbf77bb4cf330a27c50.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/6e2ac7ea1bbf9c1f829581cd916f561819dd81e9ce0b2fbf77bb4cf330a27c50.jpg)

![862db8f28ecd39e9c4f99fdfd52fbe4a60f419c2895adcb6203d71735996dedf.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/862db8f28ecd39e9c4f99fdfd52fbe4a60f419c2895adcb6203d71735996dedf.jpg)

![fcb35ba2e2e97be66287bb9d8aa27f22fef6c6865272306639220f77dbfadd59.jpg](../icml_results/784_Does%20Reasoning%20Improve%20Seeing_%20Understanding%20When%20Vision-Language%20Models%20Benefit%20from%20Thinking/tables/fcb35ba2e2e97be66287bb9d8aa27f22fef6c6865272306639220f77dbfadd59.jpg)

<a id="paper-0772"></a>

## The Geometry of Reasoning: Self-Evaluation via Layerwise Trajectory Evolution


### Images

![125d0269b379a6b0894c4fe9140bea83bd1611f0aea03d3ed717e8086320f753.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/125d0269b379a6b0894c4fe9140bea83bd1611f0aea03d3ed717e8086320f753.jpg)

![37aff0c6aa910800459d98be005689e9e6e1de8bb05287fc185b88edb9319326.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/37aff0c6aa910800459d98be005689e9e6e1de8bb05287fc185b88edb9319326.jpg)

![6428511d1025f4a0b4504790f9bcd81e25c93d6d81e82e66bab2e541eb454c7b.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/6428511d1025f4a0b4504790f9bcd81e25c93d6d81e82e66bab2e541eb454c7b.jpg)

![83663dd8088ad284bddb86d0de145483af5266247abc9f20af2e94bfbd8a2ee3.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/83663dd8088ad284bddb86d0de145483af5266247abc9f20af2e94bfbd8a2ee3.jpg)

![ae33967c4aeb47df16943c58a1ff7d402c17e076ef65025f0ad4b49c1d861158.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/ae33967c4aeb47df16943c58a1ff7d402c17e076ef65025f0ad4b49c1d861158.jpg)

![e6e81a8a6e20c9d9feaa58b9e2a61cd036ed53b73276e42c570e5d0c413feeec.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/e6e81a8a6e20c9d9feaa58b9e2a61cd036ed53b73276e42c570e5d0c413feeec.jpg)

![fa74299b9d67f67f3e1e791753f9d3042f31af4942025474da6393606c765b8f.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/images/fa74299b9d67f67f3e1e791753f9d3042f31af4942025474da6393606c765b8f.jpg)

### Tables

![046ddcf5bf5d7e4647b04a57c3625ee448749dcd6d07cf38836414757f29f178.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/046ddcf5bf5d7e4647b04a57c3625ee448749dcd6d07cf38836414757f29f178.jpg)

![15c997334c9c6c5ad73530db328b9cc57ea0ca603562992c07c040bc5708f7d3.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/15c997334c9c6c5ad73530db328b9cc57ea0ca603562992c07c040bc5708f7d3.jpg)

![1a7ece94cde88c3156544afb72b30eafcfc721dee670aa5dd70537deefff4e36.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/1a7ece94cde88c3156544afb72b30eafcfc721dee670aa5dd70537deefff4e36.jpg)

![1e243ea89f8d5b026574c5a58a943b952eee4f97acb0aede0429747b34f5fe97.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/1e243ea89f8d5b026574c5a58a943b952eee4f97acb0aede0429747b34f5fe97.jpg)

![2a6a161f0b8263c9031753ff113939a93d987ee6abea362d930e87c84fbd34ce.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/2a6a161f0b8263c9031753ff113939a93d987ee6abea362d930e87c84fbd34ce.jpg)

![3741ebc38e5a6e4a1e10a3e66cb504e76cbe48b98ee10d0a555e2c790c138574.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/3741ebc38e5a6e4a1e10a3e66cb504e76cbe48b98ee10d0a555e2c790c138574.jpg)

![3df7aef0abb4810c171149be4a0af51eb1f8a8f1f0073eba53077bafe62d8912.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/3df7aef0abb4810c171149be4a0af51eb1f8a8f1f0073eba53077bafe62d8912.jpg)

![9c46fb0d52a3cd60aaa437e6641126530312642cb137460cef3640a5676c1ab0.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/9c46fb0d52a3cd60aaa437e6641126530312642cb137460cef3640a5676c1ab0.jpg)

![abf51eac7b9d2dc4221132929e5726015bda332de95b257243d2a17ef619c06b.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/abf51eac7b9d2dc4221132929e5726015bda332de95b257243d2a17ef619c06b.jpg)

![b51b5ba69351987bf234497df8f6f7ab94e54ead6a4a407d5accda20643095cb.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/b51b5ba69351987bf234497df8f6f7ab94e54ead6a4a407d5accda20643095cb.jpg)

![bd3260f4dda58991c20cc9c9632bb318a52d7667b6e77e3e79b3fcd92f3ba9ec.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/bd3260f4dda58991c20cc9c9632bb318a52d7667b6e77e3e79b3fcd92f3ba9ec.jpg)

![c18a3ac754cd2b8d8af11d7a79dd86be05112b61a9b9ba0b37a4d17cd68d22ca.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/c18a3ac754cd2b8d8af11d7a79dd86be05112b61a9b9ba0b37a4d17cd68d22ca.jpg)

![f36fd73d319979b310615ca82685ff327db6ae85c762d5cbbf08351644a7ba4a.jpg](../icml_results/785_The%20Geometry%20of%20Reasoning_%20Self-Evaluation%20via%20Layerwise%20Trajectory%20Evolution/tables/f36fd73d319979b310615ca82685ff327db6ae85c762d5cbbf08351644a7ba4a.jpg)

<a id="paper-0773"></a>

## RN-D: Discretized Categorical Actors for On-Policy Reinforcement Learning


### Images

![036c094a6fe0b2497df48b892a95447a05b03424eea92f2b24884656838adbfc.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/036c094a6fe0b2497df48b892a95447a05b03424eea92f2b24884656838adbfc.jpg)

![0f0cf5fa26ee0b418c989dae04b1a9bc2258f8f591cba2cfeeb04142430ba845.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/0f0cf5fa26ee0b418c989dae04b1a9bc2258f8f591cba2cfeeb04142430ba845.jpg)

![10824c83289665a1a5b606f5c4221f9dc097a9b2c510e92eac89efc17ab75383.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/10824c83289665a1a5b606f5c4221f9dc097a9b2c510e92eac89efc17ab75383.jpg)

![1dd143f48ac89ed79c4898e68c4b404a99b74fa2689c1760ca841f7282b00dda.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/1dd143f48ac89ed79c4898e68c4b404a99b74fa2689c1760ca841f7282b00dda.jpg)

![45d501b0ff5e5f1e4ce9f1ba5d033e5fb2773eba69ef03c84ac1c5ff88169ded.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/45d501b0ff5e5f1e4ce9f1ba5d033e5fb2773eba69ef03c84ac1c5ff88169ded.jpg)

![74bf4432f0c20141151e1b300a5b5c1537a77bdc68c21e81cd9a7047631c57be.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/74bf4432f0c20141151e1b300a5b5c1537a77bdc68c21e81cd9a7047631c57be.jpg)

![762357452681041aa621a69c35692b8e0bf616f99d23f462977612b8edefdd2f.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/762357452681041aa621a69c35692b8e0bf616f99d23f462977612b8edefdd2f.jpg)

![9173e3f73532f941f48b6aad1271ee11510f946a748fc256cf04df65dce36e58.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/9173e3f73532f941f48b6aad1271ee11510f946a748fc256cf04df65dce36e58.jpg)

![a1f3411202ed2e2590fd43a615be8b3ce060fad14e2f04d4171d14b2017db28d.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/a1f3411202ed2e2590fd43a615be8b3ce060fad14e2f04d4171d14b2017db28d.jpg)

![d8b9a23d95457a45780d14ec71586151adf25bcc4893604d6baed3791ce89713.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/d8b9a23d95457a45780d14ec71586151adf25bcc4893604d6baed3791ce89713.jpg)

![eb0dfb7e2a93e13ba43e6a9c49e136345a787cc7a6a7476d973ae10e56113e33.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/images/eb0dfb7e2a93e13ba43e6a9c49e136345a787cc7a6a7476d973ae10e56113e33.jpg)

### Tables

![16e4dd04054da6c23cbea67c0fa8826023e1b9fe46f82b09de5ebd3ee881830e.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/tables/16e4dd04054da6c23cbea67c0fa8826023e1b9fe46f82b09de5ebd3ee881830e.jpg)

![2c6a9fe6fff50463c219f6586cb1a6bfb556b9b1249123461329a1bfe994eb12.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/tables/2c6a9fe6fff50463c219f6586cb1a6bfb556b9b1249123461329a1bfe994eb12.jpg)

![30048f20be721fa13b1f1c20d99a218537a126bec39d54f85c7aa6d6d19d51f1.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/tables/30048f20be721fa13b1f1c20d99a218537a126bec39d54f85c7aa6d6d19d51f1.jpg)

![976aeed66f395b4bd3916f007607e6b51f34b418512be1d4b01fed9667068d96.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/tables/976aeed66f395b4bd3916f007607e6b51f34b418512be1d4b01fed9667068d96.jpg)

![d7ccaba1f03e656458dde7f8c9736c37adb3abbb38fc48328928381bb593a034.jpg](../icml_results/786_RN-D_%20Discretized%20Categorical%20Actors%20for%20On-Policy%20Reinforcement%20Learning/tables/d7ccaba1f03e656458dde7f8c9736c37adb3abbb38fc48328928381bb593a034.jpg)

<a id="paper-0774"></a>

## Future-Gain Guided Test-Time Learning for Large Language Models


### Images

![1b608a36e80de4618d7355c5f6ac92ac29b7bec2463c4522e8baa0d60e9b750a.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/images/1b608a36e80de4618d7355c5f6ac92ac29b7bec2463c4522e8baa0d60e9b750a.jpg)

![33690f9e31b3d308156981af13105e931faaf1f8b6b48f53ad5036a96587474f.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/images/33690f9e31b3d308156981af13105e931faaf1f8b6b48f53ad5036a96587474f.jpg)

![645ee6e269821ffa60f044fc86c125c237cfd2255acf32ee4217b14d8b0e67d3.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/images/645ee6e269821ffa60f044fc86c125c237cfd2255acf32ee4217b14d8b0e67d3.jpg)

![65928942c519ead4cc7556993f60b97b1f1b9dca0b648c6cfac48c8b21d97fbb.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/images/65928942c519ead4cc7556993f60b97b1f1b9dca0b648c6cfac48c8b21d97fbb.jpg)

![c3919b599f66509890c23365c56ff074bb96b4c2c921ad4f5ba4e33be5439dc6.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/images/c3919b599f66509890c23365c56ff074bb96b4c2c921ad4f5ba4e33be5439dc6.jpg)

![f3aea374c3c36396685bd460529e8408f1ab483b77be9bd6b4a91c3b0e3674fa.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/images/f3aea374c3c36396685bd460529e8408f1ab483b77be9bd6b4a91c3b0e3674fa.jpg)

### Tables

![477e41926422f2093ee1d7f8df40f065fc679daa09fbfd9a1451bd4b8ed8b7a6.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/477e41926422f2093ee1d7f8df40f065fc679daa09fbfd9a1451bd4b8ed8b7a6.jpg)

![52dc5cfbc1722eded6491cddefc605d392a77384f503212a8ffd3eaf7bd65cdb.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/52dc5cfbc1722eded6491cddefc605d392a77384f503212a8ffd3eaf7bd65cdb.jpg)

![540ff22f547d11dd20284087cf6e3894a1aeef972c85070ac1ed23cd21d7bd59.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/540ff22f547d11dd20284087cf6e3894a1aeef972c85070ac1ed23cd21d7bd59.jpg)

![542c4e5373163a770699129ea567b89fe91f45269861f03fe987ee28f6895618.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/542c4e5373163a770699129ea567b89fe91f45269861f03fe987ee28f6895618.jpg)

![6fe879f24b455b7b47f77adce10bf29979827a2a0f49dfbba5a20df175f587ae.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/6fe879f24b455b7b47f77adce10bf29979827a2a0f49dfbba5a20df175f587ae.jpg)

![7aec1d5225dc0388846131311f1488a29e570dd87628162d9848a6458f37e0d8.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/7aec1d5225dc0388846131311f1488a29e570dd87628162d9848a6458f37e0d8.jpg)

![819d4538e09da35c3bcff93f1dc3347483956923001d9b7bd49895414ed66082.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/819d4538e09da35c3bcff93f1dc3347483956923001d9b7bd49895414ed66082.jpg)

![98352ade1b5fbd068163623badeb898d4600665c8d3cb923907ae89cabdf62b9.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/98352ade1b5fbd068163623badeb898d4600665c8d3cb923907ae89cabdf62b9.jpg)

![9c42facce4256bcb604f7d69a956e051d8e4975a79de845816fd01cae59799df.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/9c42facce4256bcb604f7d69a956e051d8e4975a79de845816fd01cae59799df.jpg)

![9e632dfe9d91ac9cb8948df54ae7c7fb7a16eb8238aadbe7553becb62b7c9e91.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/9e632dfe9d91ac9cb8948df54ae7c7fb7a16eb8238aadbe7553becb62b7c9e91.jpg)

![9fcb07872cd31a48028bef93447381198859039a6d363ebceceb5fe09d2682d6.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/9fcb07872cd31a48028bef93447381198859039a6d363ebceceb5fe09d2682d6.jpg)

![a7ce25713b570edca0d58f355f1e432197ba8d74ec8525b27ed6c2ab4ed010af.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/a7ce25713b570edca0d58f355f1e432197ba8d74ec8525b27ed6c2ab4ed010af.jpg)

![a9252dff13bb9f33bd44d54b5f5853daaad48471fa2004bb6d1ee54fc840f5c6.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/a9252dff13bb9f33bd44d54b5f5853daaad48471fa2004bb6d1ee54fc840f5c6.jpg)

![c11a07d08151e8bac9bd3095d463ab0f5a8e71ba546b041477bcd241d737f53e.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/c11a07d08151e8bac9bd3095d463ab0f5a8e71ba546b041477bcd241d737f53e.jpg)

![dde7895d6744f66eaf45378c9978024d39825ab5147478c7987dd5e956c359e1.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/dde7895d6744f66eaf45378c9978024d39825ab5147478c7987dd5e956c359e1.jpg)

![e16be367c643e493480bd73daf4dc394ea67f6d31b41ce96e50f9338e2277807.jpg](../icml_results/787_Future-Gain%20Guided%20Test-Time%20Learning%20for%20Large%20Language%20Models/tables/e16be367c643e493480bd73daf4dc394ea67f6d31b41ce96e50f9338e2277807.jpg)

<a id="paper-0775"></a>

## AgentVocab: Structure-Aware Vocabulary Adaptation for Efficient LLM Agents


### Images

![4209b05baee1c23f42a33deec52162538e3b6b98fb919cfd797b3b5f53434416.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/images/4209b05baee1c23f42a33deec52162538e3b6b98fb919cfd797b3b5f53434416.jpg)

![4b81f61fb15f3e3a3ee0fe98d398c5026cdb329c8e467f59fe9474f328a647c6.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/images/4b81f61fb15f3e3a3ee0fe98d398c5026cdb329c8e467f59fe9474f328a647c6.jpg)

![4f7312b39a22134d0c2dfbee9186dfb3e989686d9cec729b0759fe0759995144.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/images/4f7312b39a22134d0c2dfbee9186dfb3e989686d9cec729b0759fe0759995144.jpg)

![d346458b73e7dec668c13050fd4fb694f366eaa06002914a872c07bc9a62cfa3.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/images/d346458b73e7dec668c13050fd4fb694f366eaa06002914a872c07bc9a62cfa3.jpg)

### Tables

![6e22b26221522dcd7789ecdd34e8a85b32ed144e269c02363091d11af5a978f3.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/tables/6e22b26221522dcd7789ecdd34e8a85b32ed144e269c02363091d11af5a978f3.jpg)

![794224ba58c0b211b6cb3400e98151a8e1a1e2417fd6220f921e77e6792c31ff.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/tables/794224ba58c0b211b6cb3400e98151a8e1a1e2417fd6220f921e77e6792c31ff.jpg)

![d8a54d945ac96b68b2a81dfbc7a50ec6a04613dc88d4ae61f945b16d7cfdebe6.jpg](../icml_results/788_AgentVocab_%20Structure-Aware%20Vocabulary%20Adaptation%20for%20Efficient%20LLM%20Agents/tables/d8a54d945ac96b68b2a81dfbc7a50ec6a04613dc88d4ae61f945b16d7cfdebe6.jpg)

<a id="paper-0776"></a>

## CURE: Consistency-under-Unified Semantic Regularization for Generalized Category Discovery


### Images

![121f270850833db1cb7528ed647468ca1876ab9ae0d381fd749cc6ce6a8d9b78.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/121f270850833db1cb7528ed647468ca1876ab9ae0d381fd749cc6ce6a8d9b78.jpg)

![3016b11ed0034637bbbadb9397584121908150b141feb6847d00ab3277481833.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/3016b11ed0034637bbbadb9397584121908150b141feb6847d00ab3277481833.jpg)

![32862f5d67d687d25b11c98e19dd7eaf88cbd86b90723a71a77328401ea06593.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/32862f5d67d687d25b11c98e19dd7eaf88cbd86b90723a71a77328401ea06593.jpg)

![81a0f4cc1c8a642bcfc1fa53735331d0d8a1940f84f3e3ea88049477b95b24e6.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/81a0f4cc1c8a642bcfc1fa53735331d0d8a1940f84f3e3ea88049477b95b24e6.jpg)

![d545327bf69071fc128bd3035f8a7d3f656b54497b53ac3e053fa2e0c921d5ac.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/d545327bf69071fc128bd3035f8a7d3f656b54497b53ac3e053fa2e0c921d5ac.jpg)

![e17d374d45368fdae8b1eb0192c98fcc631e6b56b75ce02f288e2909b3703cd7.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/e17d374d45368fdae8b1eb0192c98fcc631e6b56b75ce02f288e2909b3703cd7.jpg)

![e6a4755aa16047eb60161df302aedb93deec614709fc10c7c73c0ccd4f05e48e.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/images/e6a4755aa16047eb60161df302aedb93deec614709fc10c7c73c0ccd4f05e48e.jpg)

### Tables

![00490b7aa75e3d4395cce545bde9e819e9f89dd64db97a2ba5f9fd7f1c5acc4d.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/00490b7aa75e3d4395cce545bde9e819e9f89dd64db97a2ba5f9fd7f1c5acc4d.jpg)

![0c7ee3140b8351bf303822c17055ed3db449b894b0fc12090b8293c53bd316fa.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/0c7ee3140b8351bf303822c17055ed3db449b894b0fc12090b8293c53bd316fa.jpg)

![108c1d2e25c03e46080178c03636f98c9904110d20b53566df0e35fd91d75a5c.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/108c1d2e25c03e46080178c03636f98c9904110d20b53566df0e35fd91d75a5c.jpg)

![1e06664739fcc15cecb0dff89741a0aed4cf31e52160e816902874dd2a778106.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/1e06664739fcc15cecb0dff89741a0aed4cf31e52160e816902874dd2a778106.jpg)

![4ebd95452738bd6edad3fd040b0827e18ac75b98a9a313d73b56dd725e1f20fa.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/4ebd95452738bd6edad3fd040b0827e18ac75b98a9a313d73b56dd725e1f20fa.jpg)

![56176aa818efe0cc9113494e44c243968b5517bc2fe5d2780f58439182dcf6f3.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/56176aa818efe0cc9113494e44c243968b5517bc2fe5d2780f58439182dcf6f3.jpg)

![5cf0b0a12efd78108cb5a4f70ab54dc77909bcbef62ea89e7b5312f5c37f7fb2.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/5cf0b0a12efd78108cb5a4f70ab54dc77909bcbef62ea89e7b5312f5c37f7fb2.jpg)

![635aead71f207ef74f613a68ef7eb921653dc91c3fc5e66e1a750c3d3eab813d.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/635aead71f207ef74f613a68ef7eb921653dc91c3fc5e66e1a750c3d3eab813d.jpg)

![6c967a11f7c9b59f114dccfabe6068202e1c4925fad305238d7df9ec6744b143.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/6c967a11f7c9b59f114dccfabe6068202e1c4925fad305238d7df9ec6744b143.jpg)

![80b03b8bf9ac85b68e75c7fc31b9cc0ec6ff1b0e60fa3a69f9db6f1225e2ea75.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/80b03b8bf9ac85b68e75c7fc31b9cc0ec6ff1b0e60fa3a69f9db6f1225e2ea75.jpg)

![91af731f4bed5d9740ff3a5669fd73e84ee3863ca42c5809bfb2e3e5b4ac4d52.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/91af731f4bed5d9740ff3a5669fd73e84ee3863ca42c5809bfb2e3e5b4ac4d52.jpg)

![9b8789ceb99ad10219cb7db214a6b34947c6ba4f23b1119760b749e223a45ea2.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/9b8789ceb99ad10219cb7db214a6b34947c6ba4f23b1119760b749e223a45ea2.jpg)

![9c425e8dbcaa3690f1433a752f67e10b22ebf906b82f53f93ecc63a3e868048e.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/9c425e8dbcaa3690f1433a752f67e10b22ebf906b82f53f93ecc63a3e868048e.jpg)

![9fb7eb96f0816a62e3fdac2f7ef7b7b3b23dca4879a0c6a8f9e01ed5c7bd5435.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/9fb7eb96f0816a62e3fdac2f7ef7b7b3b23dca4879a0c6a8f9e01ed5c7bd5435.jpg)

![a63aadb7d95119b632d67d334ee8b7c8e67af611708668bae7ce9592125309b5.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/a63aadb7d95119b632d67d334ee8b7c8e67af611708668bae7ce9592125309b5.jpg)

![b15b8c465734c44d5a219105f588da03d1aea1fc9aa9280ee516186e1f032cc0.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/b15b8c465734c44d5a219105f588da03d1aea1fc9aa9280ee516186e1f032cc0.jpg)

![c66f9b10d799d4d9284a9c0ae56758e17a2e0648e3280c04b93e73a9b1879fa7.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/c66f9b10d799d4d9284a9c0ae56758e17a2e0648e3280c04b93e73a9b1879fa7.jpg)

![c6f0361c4d9743ec39747270dde7b246ad11231b27ce8e1ed3527e42b7e8fc82.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/c6f0361c4d9743ec39747270dde7b246ad11231b27ce8e1ed3527e42b7e8fc82.jpg)

![c881b1c9bb1e4b5c8ebb207068de4710599d366c53659fa02d166d9540be1c09.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/c881b1c9bb1e4b5c8ebb207068de4710599d366c53659fa02d166d9540be1c09.jpg)

![cc60dd6edac5139a24b0e1948d589faeee7bdfa135722948f23d18bd7292991e.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/cc60dd6edac5139a24b0e1948d589faeee7bdfa135722948f23d18bd7292991e.jpg)

![ceb189cfa3e6e6b7861262a8e2c3972844ff5f7a8456a255f632bacef22789cf.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/ceb189cfa3e6e6b7861262a8e2c3972844ff5f7a8456a255f632bacef22789cf.jpg)

![dc8ef1992aef9e5d3f49a01813063543e5b8efd7325849eda53422cd10dfcab4.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/dc8ef1992aef9e5d3f49a01813063543e5b8efd7325849eda53422cd10dfcab4.jpg)

![f496f1410b411d627ec3187b9bea826755a890231b6b2dde7a0acf716820744c.jpg](../icml_results/789_CURE_%20Consistency-under-Unified%20Semantic%20Regularization%20for%20Generalized%20Category%20Discovery/tables/f496f1410b411d627ec3187b9bea826755a890231b6b2dde7a0acf716820744c.jpg)

<a id="paper-0777"></a>

## FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning


### Images

![3ade76b6e6a49f3499f532319dcf15372813ebebb938d2e47d7a50b99a43c71d.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/3ade76b6e6a49f3499f532319dcf15372813ebebb938d2e47d7a50b99a43c71d.jpg)

![4124cb2606877c4ebd283c86225bf4b2e0f3142c883f56daca474878cc7ce86c.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/4124cb2606877c4ebd283c86225bf4b2e0f3142c883f56daca474878cc7ce86c.jpg)

![767cf7141a376dbcd7d094a59ade032e9de28b1cdcdbee65f067c04d959e651a.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/767cf7141a376dbcd7d094a59ade032e9de28b1cdcdbee65f067c04d959e651a.jpg)

![8a1c5fa19182c4374724a58f42e28c321d7ec1e118d8d14471ace616b02e4096.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/8a1c5fa19182c4374724a58f42e28c321d7ec1e118d8d14471ace616b02e4096.jpg)

![dbbf8679047477c26a655df526888059ee2042e502fa65526d16f9ca9b7e4e53.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/dbbf8679047477c26a655df526888059ee2042e502fa65526d16f9ca9b7e4e53.jpg)

![de917dc2532471efa7acbd46721ef05b6cb6f026db4786d253a38792d5d7f122.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/de917dc2532471efa7acbd46721ef05b6cb6f026db4786d253a38792d5d7f122.jpg)

![f30064d64dc0696b557f8a8e8e33a1b0ab90ba3d6153efaa6ddbaa593c4a5b31.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/f30064d64dc0696b557f8a8e8e33a1b0ab90ba3d6153efaa6ddbaa593c4a5b31.jpg)

![f7ff955e6ad50e7cf2c5951ff0ab9bd4b95341bf44b3dd9de1ef21f8ffadcd65.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/images/f7ff955e6ad50e7cf2c5951ff0ab9bd4b95341bf44b3dd9de1ef21f8ffadcd65.jpg)

### Tables

![0d22951e4c3aee6603f36730ce0a64869ac6bddca2eba64d9a0f0560c40224a0.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/0d22951e4c3aee6603f36730ce0a64869ac6bddca2eba64d9a0f0560c40224a0.jpg)

![1ef3cf8b60c03997110771a69abb720721c8cc7d61b5cfd6ac4113a631dce804.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/1ef3cf8b60c03997110771a69abb720721c8cc7d61b5cfd6ac4113a631dce804.jpg)

![34033607f1ac98a733cc82f73a8a81fb20ce9511127729b518d9522c4cde246a.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/34033607f1ac98a733cc82f73a8a81fb20ce9511127729b518d9522c4cde246a.jpg)

![35f5a4b04037dad43abe65aa6f223e3080f164ac9e769ba882054d09bd48c3be.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/35f5a4b04037dad43abe65aa6f223e3080f164ac9e769ba882054d09bd48c3be.jpg)

![512dcc29ef0cf9c4532e627a0c9901d9aac8b9aed3401f031d90e0453b67de32.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/512dcc29ef0cf9c4532e627a0c9901d9aac8b9aed3401f031d90e0453b67de32.jpg)

![51798e3ad041d788fa2af805ae45621f8859a6a772e1766d30346b6538a7b8ce.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/51798e3ad041d788fa2af805ae45621f8859a6a772e1766d30346b6538a7b8ce.jpg)

![5c3b672054c387992be8c790ac3a1de46a1e11fb99340c36696958c3a6f039e3.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/5c3b672054c387992be8c790ac3a1de46a1e11fb99340c36696958c3a6f039e3.jpg)

![605dac69af00c2f63114e2b8e525b915a63ea258342d95e01fe2e4f490dd6c29.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/605dac69af00c2f63114e2b8e525b915a63ea258342d95e01fe2e4f490dd6c29.jpg)

![72ce2a524a7ba925f4c816bb08f0faf1c1b41f904720999de1967c7ff3c49a11.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/72ce2a524a7ba925f4c816bb08f0faf1c1b41f904720999de1967c7ff3c49a11.jpg)

![846b079a6a0e620ddb5f3fce3830acbc2c8dd469174adc0e4b97f8efc698aca8.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/846b079a6a0e620ddb5f3fce3830acbc2c8dd469174adc0e4b97f8efc698aca8.jpg)

![89c5475d38cf0f9cc1a31fb11b7952216885fbcc37f82f4fb9e0da39a8c9be7c.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/89c5475d38cf0f9cc1a31fb11b7952216885fbcc37f82f4fb9e0da39a8c9be7c.jpg)

![8e6302e03a52c661139f9ba235d20f279155459058b7a072b579917e1cd546ac.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/8e6302e03a52c661139f9ba235d20f279155459058b7a072b579917e1cd546ac.jpg)

![98d4ed8c81ae2047752ad47f0bc2d64a273112a745b7a995a5fdf0b659688917.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/98d4ed8c81ae2047752ad47f0bc2d64a273112a745b7a995a5fdf0b659688917.jpg)

![a1951307e4f1371950edf2e22e5d30c4041f425e8e5787f842cce094e33e5e91.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/a1951307e4f1371950edf2e22e5d30c4041f425e8e5787f842cce094e33e5e91.jpg)

![a2bd79c5b3cf4acf6de314a314134da08c17e94b25f654d1b67001bfd27b7f71.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/a2bd79c5b3cf4acf6de314a314134da08c17e94b25f654d1b67001bfd27b7f71.jpg)

![acfab38bb3eb29364310d2749789276c8ba99f319e8fe0ff2fcc742d07f13463.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/acfab38bb3eb29364310d2749789276c8ba99f319e8fe0ff2fcc742d07f13463.jpg)

![c4b259b093ad2b67b24c973eb5b2afb67f1c83d24b7032d580386dc860b85918.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/c4b259b093ad2b67b24c973eb5b2afb67f1c83d24b7032d580386dc860b85918.jpg)

![db7c63205dbb07e8e07198a9286ca0b5b4207ea7f2088d3a19922997f59d2226.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/db7c63205dbb07e8e07198a9286ca0b5b4207ea7f2088d3a19922997f59d2226.jpg)

![eb5303e9d92128822eafccf65c5bb6cbb032040430608a7caf033faacf45a81c.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/eb5303e9d92128822eafccf65c5bb6cbb032040430608a7caf033faacf45a81c.jpg)

![ec3f6cc5928f123b046cb33e9b214426780d63f924d015e800bb031f93dec882.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/ec3f6cc5928f123b046cb33e9b214426780d63f924d015e800bb031f93dec882.jpg)

![efe75353a25e9b95f81c2990c24efc61fe9d52e12d606a06c5afa94993967fa7.jpg](../icml_results/790_FedTreeLoRA_%20Reconciling%20Statistical%20and%20Functional%20Heterogeneity%20in%20Federated%20LoRA%20Fine-Tuning/tables/efe75353a25e9b95f81c2990c24efc61fe9d52e12d606a06c5afa94993967fa7.jpg)

<a id="paper-0778"></a>

## Retrieval-Aware Distillation for Transformer-SSM Hybrids


### Images

![39afa0816dcb0bb3fb6e2fd1f17ff961514c5e57764655765db6526aa5f4d18d.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/images/39afa0816dcb0bb3fb6e2fd1f17ff961514c5e57764655765db6526aa5f4d18d.jpg)

![c6f8cdc3efad84a264a2822fbb6d105e530deffd520372c9a18c00e4da352012.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/images/c6f8cdc3efad84a264a2822fbb6d105e530deffd520372c9a18c00e4da352012.jpg)

![e01d9f8bf15380c8166b6734ea6356966a2bff4095654310eed3e67f0637ce64.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/images/e01d9f8bf15380c8166b6734ea6356966a2bff4095654310eed3e67f0637ce64.jpg)

![e5bb2f3f92a4019aecfdae8b6b7938da0ee342f1949d35edc5a8423b1c8dcf19.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/images/e5bb2f3f92a4019aecfdae8b6b7938da0ee342f1949d35edc5a8423b1c8dcf19.jpg)

### Tables

![12e0a03a3d7763ebe29b2bcdb4aa532429ccc8889d9e1ce8eea1bad4f74568b9.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/12e0a03a3d7763ebe29b2bcdb4aa532429ccc8889d9e1ce8eea1bad4f74568b9.jpg)

![13ccc11bb6aabb28258e09d72616f2ac63667b5e3dcfbc9462869d742a56bce7.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/13ccc11bb6aabb28258e09d72616f2ac63667b5e3dcfbc9462869d742a56bce7.jpg)

![16d23a6ac2a65df3d4b582e5334811a04ef055b4f6e23d56c5cc57bcdcfb481e.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/16d23a6ac2a65df3d4b582e5334811a04ef055b4f6e23d56c5cc57bcdcfb481e.jpg)

![26dd1da003e3cc819acc569a97a7e9c67d4d813d0ed9f2d764c11376e39c624a.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/26dd1da003e3cc819acc569a97a7e9c67d4d813d0ed9f2d764c11376e39c624a.jpg)

![2f1b12e11a6968843bd9fbe7bb27cfcd3dca2425b66c8b176f643d8d6ee2f011.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/2f1b12e11a6968843bd9fbe7bb27cfcd3dca2425b66c8b176f643d8d6ee2f011.jpg)

![5ebf8a2922b131bfc44ae642544d594952e34ed035150c61acbec9141526e567.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/5ebf8a2922b131bfc44ae642544d594952e34ed035150c61acbec9141526e567.jpg)

![bb821c6a64c51aaf067ac5e8a9526069f21aafe06343e5d91328237c26f8d81b.jpg](../icml_results/791_Retrieval-Aware%20Distillation%20for%20Transformer-SSM%20Hybrids/tables/bb821c6a64c51aaf067ac5e8a9526069f21aafe06343e5d91328237c26f8d81b.jpg)

<a id="paper-0779"></a>

## Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering


### Images

![035f8a3c923bfa75ea9f035ac281908307bcd44dfecb4447321c545a07cbb105.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/035f8a3c923bfa75ea9f035ac281908307bcd44dfecb4447321c545a07cbb105.jpg)

![068745a46ce34f46698ce166a5b995bc3f0ae8baf73e117d82f2b10d5a9a6fed.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/068745a46ce34f46698ce166a5b995bc3f0ae8baf73e117d82f2b10d5a9a6fed.jpg)

![08cb43aae8f965673c89c44258d0604d03e45f235723da6832b252f935304bed.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/08cb43aae8f965673c89c44258d0604d03e45f235723da6832b252f935304bed.jpg)

![1b3eb313db92c822c9f0cc0e7bb3c30489ea28db98e22fa2bd2bf3932830822e.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/1b3eb313db92c822c9f0cc0e7bb3c30489ea28db98e22fa2bd2bf3932830822e.jpg)

![2d7fcce78b6959a1074a6b142586dd482b19807795ffdab7478cda202e152a75.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/2d7fcce78b6959a1074a6b142586dd482b19807795ffdab7478cda202e152a75.jpg)

![3ae45aba8bf7ea7d4fabb80b88e00a344c22bfc49bd326d0a16ee6d5f9aca800.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/3ae45aba8bf7ea7d4fabb80b88e00a344c22bfc49bd326d0a16ee6d5f9aca800.jpg)

![3c9279386d093fc68d383c2890c13a5c6a9c0c2faa6042b7fcf4fa32e4d06f99.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/3c9279386d093fc68d383c2890c13a5c6a9c0c2faa6042b7fcf4fa32e4d06f99.jpg)

![406aa5cce27e4b234c71836a1caaec41990e41d0905a2d5c82d0a22716b1d6f4.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/406aa5cce27e4b234c71836a1caaec41990e41d0905a2d5c82d0a22716b1d6f4.jpg)

![4184a51d34a47fdf657d3d1834a21f325c99ed3ad1e0cfd53fdb89f7f3d029f9.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/4184a51d34a47fdf657d3d1834a21f325c99ed3ad1e0cfd53fdb89f7f3d029f9.jpg)

![71d3d02d48fcc02d2bf5d800d68b5fbb8b99d97cd3ccb4d0156a56b4cad6cca9.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/71d3d02d48fcc02d2bf5d800d68b5fbb8b99d97cd3ccb4d0156a56b4cad6cca9.jpg)

![812127881e44ac77dfb9bc6ddcc20edec10fdec84f8e77715b834765f3da54db.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/812127881e44ac77dfb9bc6ddcc20edec10fdec84f8e77715b834765f3da54db.jpg)

![858ddeebdba49117f7e1b7f72ef18574860efa25a0f0eadd4c5e3c1c91038265.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/858ddeebdba49117f7e1b7f72ef18574860efa25a0f0eadd4c5e3c1c91038265.jpg)

![99cb120f3f545507e629539e5c121e4529533f7b0d714b0bdebe55d2b0b63472.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/99cb120f3f545507e629539e5c121e4529533f7b0d714b0bdebe55d2b0b63472.jpg)

![9b74134cac07de6ada58123d3f8ac1455a0d45a90851d18be698db2c9d124ed4.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/9b74134cac07de6ada58123d3f8ac1455a0d45a90851d18be698db2c9d124ed4.jpg)

![a461fed3d606ca47c1942f7e69568654c494135c9be857be7b0849ba2b3c7d87.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/a461fed3d606ca47c1942f7e69568654c494135c9be857be7b0849ba2b3c7d87.jpg)

![c070c134726468365e40b93496d9d850b616a1257e7eea3f35d26597a8a214d7.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/c070c134726468365e40b93496d9d850b616a1257e7eea3f35d26597a8a214d7.jpg)

![c1879d2846afe860c5f529a850c6c5a18b0a380218b5c3eb748834e30a66cae9.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/c1879d2846afe860c5f529a850c6c5a18b0a380218b5c3eb748834e30a66cae9.jpg)

![c6883e795e8c07e4e67fd3554d7d6a01be4e37dd2174bddc9dbb524e26a3c687.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/c6883e795e8c07e4e67fd3554d7d6a01be4e37dd2174bddc9dbb524e26a3c687.jpg)

![ce9019682562c28c24900625199184cede2d1a22a49b78c35761c34079fcca8c.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/ce9019682562c28c24900625199184cede2d1a22a49b78c35761c34079fcca8c.jpg)

![e8240d88fc21c9ae81d493aa19b06892e824e1cbde082f24cc0ff0b07517fa6c.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/e8240d88fc21c9ae81d493aa19b06892e824e1cbde082f24cc0ff0b07517fa6c.jpg)

![ecc8423a61477aea3b28a97b9662400d3d25ea01509bc5e8ec8c418185000eb8.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/ecc8423a61477aea3b28a97b9662400d3d25ea01509bc5e8ec8c418185000eb8.jpg)

![fe1cb42ba24f535d9906a3fee833dbd20467aa85e66d27ecac4910f0e7650ebf.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/fe1cb42ba24f535d9906a3fee833dbd20467aa85e66d27ecac4910f0e7650ebf.jpg)

![ff076112db77d964db9fd109b259f8dc74d33d76f0eb4e148499e25dae0d88c2.jpg](../icml_results/792_Belief%20Dynamics%20Reveal%20the%20Dual%20Nature%20of%20In-Context%20Learning%20and%20Activation%20Steering/images/ff076112db77d964db9fd109b259f8dc74d33d76f0eb4e148499e25dae0d88c2.jpg)

<a id="paper-0780"></a>

## What If We Allocate Test-Time Compute Adaptively?


### Images

![0854c9f20822bd27967f6c2fc4fe69ca09ad29aadeadc2944a5aa325fcac20e9.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/images/0854c9f20822bd27967f6c2fc4fe69ca09ad29aadeadc2944a5aa325fcac20e9.jpg)

![30457834231e034c5ccd713303dacd6ddf6a7be34372b90ab718c1c538c44516.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/images/30457834231e034c5ccd713303dacd6ddf6a7be34372b90ab718c1c538c44516.jpg)

![7b144e5b164d05664309480a66ecf21dbcb976f379b027c636f35cb7ba5fe237.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/images/7b144e5b164d05664309480a66ecf21dbcb976f379b027c636f35cb7ba5fe237.jpg)

![84ef4670d81790a5c77f1d4407623198a34fdc4345ce9048fb62532251dbb2db.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/images/84ef4670d81790a5c77f1d4407623198a34fdc4345ce9048fb62532251dbb2db.jpg)

![eb3781e0712c144107d5da05cd422454c2430f9c974441e3e5d85a15d968c145.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/images/eb3781e0712c144107d5da05cd422454c2430f9c974441e3e5d85a15d968c145.jpg)

### Tables

![01e9354ba8a6a626c4a5f8902ebd7295220d474441d648399c6271d474fb7b01.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/01e9354ba8a6a626c4a5f8902ebd7295220d474441d648399c6271d474fb7b01.jpg)

![56c75ceb5d5797544e21f9769b777ce22a639230a32646f400e8db43ebde8131.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/56c75ceb5d5797544e21f9769b777ce22a639230a32646f400e8db43ebde8131.jpg)

![5723f7ba1fe1a3c40e2f8882e7c97c3020b02b77c972c3996f0bb73219fc8195.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/5723f7ba1fe1a3c40e2f8882e7c97c3020b02b77c972c3996f0bb73219fc8195.jpg)

![602cd92bd5632125275a2a14cd2b47974ea1586988dda1cd5e975277dc41f51b.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/602cd92bd5632125275a2a14cd2b47974ea1586988dda1cd5e975277dc41f51b.jpg)

![6858d12e6eb375d1d14e743bf06c79eec99d4a176314d822163a8f0646860ab5.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/6858d12e6eb375d1d14e743bf06c79eec99d4a176314d822163a8f0646860ab5.jpg)

![a3ae1bb117037a1fdb1cd9d15e5f0182e2e2c048a8583744cb54632b32cdc48f.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/a3ae1bb117037a1fdb1cd9d15e5f0182e2e2c048a8583744cb54632b32cdc48f.jpg)

![a5367048a3709920e97be97e9ca799e64d2ea40b1f9c48b2af2a13a7cffe903c.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/a5367048a3709920e97be97e9ca799e64d2ea40b1f9c48b2af2a13a7cffe903c.jpg)

![ab018ca8af65b76c289169e2e9458766d1f0df4d501023ef9f7fc4eb90451ea6.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/ab018ca8af65b76c289169e2e9458766d1f0df4d501023ef9f7fc4eb90451ea6.jpg)

![b730178b5644688ad55bdeef63b8378e88a8885d19d10efd1cefade57b5d5c3e.jpg](../icml_results/793_What%20If%20We%20Allocate%20Test-Time%20Compute%20Adaptively_/tables/b730178b5644688ad55bdeef63b8378e88a8885d19d10efd1cefade57b5d5c3e.jpg)

<a id="paper-0781"></a>

## Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models


### Images

![3d7dd5b5f3a06857614ea0fac42753fe6a0360abd1ce4d8d117e021cf6be3828.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/3d7dd5b5f3a06857614ea0fac42753fe6a0360abd1ce4d8d117e021cf6be3828.jpg)

![5421fd28b304fff9772699c790c50ac5cfd78746ba0e1b593d16e0d088b86867.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/5421fd28b304fff9772699c790c50ac5cfd78746ba0e1b593d16e0d088b86867.jpg)

![676987cc3c4ba9e4534bdc1979968311b180ed74ec54f119219a04a81506dcf7.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/676987cc3c4ba9e4534bdc1979968311b180ed74ec54f119219a04a81506dcf7.jpg)

![6aac1df23c185cb4f707362acc9e5e59dde333fed6d95650a58ca0d70fd2cb6d.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/6aac1df23c185cb4f707362acc9e5e59dde333fed6d95650a58ca0d70fd2cb6d.jpg)

![70a7e531241eab02c415ccbb95b609574b8d6e7f460a860009e5c45938ba4ea1.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/70a7e531241eab02c415ccbb95b609574b8d6e7f460a860009e5c45938ba4ea1.jpg)

![77b92b93d9c38cf68777a811548e7deff1cfb0f966eb96bb8111b73e63d6bbc3.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/77b92b93d9c38cf68777a811548e7deff1cfb0f966eb96bb8111b73e63d6bbc3.jpg)

![78f36f4191dc655a64b2a2490f3f7bc9085443e4cca8f4659bf0a3c341645eeb.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/78f36f4191dc655a64b2a2490f3f7bc9085443e4cca8f4659bf0a3c341645eeb.jpg)

![a599226d9d608455b8a80910ec16afde5a9da4b5c80de62187dcaf58bb6a3989.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/a599226d9d608455b8a80910ec16afde5a9da4b5c80de62187dcaf58bb6a3989.jpg)

![ab783c0a870bc196437ed78a42eddadd479d3e5cd3a2c9a3f8b6d2a723115659.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/ab783c0a870bc196437ed78a42eddadd479d3e5cd3a2c9a3f8b6d2a723115659.jpg)

![e6c22fe841d6dae434db3c6f1a86140852b0f76603b77d135834f75e5f7ad071.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/e6c22fe841d6dae434db3c6f1a86140852b0f76603b77d135834f75e5f7ad071.jpg)

![fbf76b5fd112b20f4819c8a2bd908d112d015cb4e9fa510071a81964a5732364.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/fbf76b5fd112b20f4819c8a2bd908d112d015cb4e9fa510071a81964a5732364.jpg)

![feb503a670c8a245aa574a6b0fb165919ee24e7c81fda7f83cbe71c63563b59a.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/images/feb503a670c8a245aa574a6b0fb165919ee24e7c81fda7f83cbe71c63563b59a.jpg)

### Tables

![264e0662f96c8a954fed0ef4547414eeb2b4ee1d7c34fbce017b4767c10fa856.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/tables/264e0662f96c8a954fed0ef4547414eeb2b4ee1d7c34fbce017b4767c10fa856.jpg)

![3e5d1662cd42565c9074a6465094f74847ae8eefbad1d0db28253799d8292380.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/tables/3e5d1662cd42565c9074a6465094f74847ae8eefbad1d0db28253799d8292380.jpg)

![42b45d0735bf3466b6ba80f6c3db910d3dfd68cbc2f6c1106779d9be2e03cf44.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/tables/42b45d0735bf3466b6ba80f6c3db910d3dfd68cbc2f6c1106779d9be2e03cf44.jpg)

![94f21a1910a22f597ca6007f5aa81280934107251132f73e747b84053e7e60e6.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/tables/94f21a1910a22f597ca6007f5aa81280934107251132f73e747b84053e7e60e6.jpg)

![abc2ab5b9e9c0012f452fd55e6b602a86a1fde255ca6a34b4271608ea6a1dc70.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/tables/abc2ab5b9e9c0012f452fd55e6b602a86a1fde255ca6a34b4271608ea6a1dc70.jpg)

![ff085b5c197bb2055c42fefb5a7349b9a53472141df57448cc5a7e4a0317ef54.jpg](../icml_results/794_Attention%20Sinks%20as%20Internal%20Signals%20for%20Hallucination%20Detection%20in%20Large%20Language%20Models/tables/ff085b5c197bb2055c42fefb5a7349b9a53472141df57448cc5a7e4a0317ef54.jpg)

<a id="paper-0782"></a>

## CORAL: Uncertainty-Aware Regulation of Exposure Concentration in Recommender Systems


### Images

![3361436c09bd9cce2c7ee6eaeb78a37fc369d705b835a2acb57383ec6da7dbab.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/images/3361436c09bd9cce2c7ee6eaeb78a37fc369d705b835a2acb57383ec6da7dbab.jpg)

![35984d7fd8f9fbf507bed7ab01c7139a28348a79b02b1e54fd34826e59ea8227.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/images/35984d7fd8f9fbf507bed7ab01c7139a28348a79b02b1e54fd34826e59ea8227.jpg)

![54e106f11946501b428aba1c847c7add49467db7d56225077bdf930f16d0c38f.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/images/54e106f11946501b428aba1c847c7add49467db7d56225077bdf930f16d0c38f.jpg)

![729c5114e837bb1164ec38eed4e81a611fc5e65f9c3fc5aa48c8c92a0d1e76fd.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/images/729c5114e837bb1164ec38eed4e81a611fc5e65f9c3fc5aa48c8c92a0d1e76fd.jpg)

![fbb9bfd5169bb35f74b9be7af72ef3de7160402e8bffe1be003a9c7ff87ed338.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/images/fbb9bfd5169bb35f74b9be7af72ef3de7160402e8bffe1be003a9c7ff87ed338.jpg)

### Tables

![088e0ba87519980d942a26dbe59b640619415cee496d0546489eb4410d9bece8.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/088e0ba87519980d942a26dbe59b640619415cee496d0546489eb4410d9bece8.jpg)

![276a8e2959a9f46b0a0d45674124cd4eb0363ae47c4b02bcf14d7a861bf8e240.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/276a8e2959a9f46b0a0d45674124cd4eb0363ae47c4b02bcf14d7a861bf8e240.jpg)

![3cf2465bb966fcdd55c489907790fd0d994a19e85646d229167046c2d07ea76f.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/3cf2465bb966fcdd55c489907790fd0d994a19e85646d229167046c2d07ea76f.jpg)

![4900e700b7b0915a8c4c585afb5d504a57cad5dfb399871762e6668ae78b9101.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/4900e700b7b0915a8c4c585afb5d504a57cad5dfb399871762e6668ae78b9101.jpg)

![8b04025bf4b6739f205f86ae1611cf9042237b091d3b9a95814713c589ac0200.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/8b04025bf4b6739f205f86ae1611cf9042237b091d3b9a95814713c589ac0200.jpg)

![b9bb190770e2eb62c34e53ad6e1ebcf0f5ae18dea142b396b1bc79643b9ce9be.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/b9bb190770e2eb62c34e53ad6e1ebcf0f5ae18dea142b396b1bc79643b9ce9be.jpg)

![d1ec9eeef456f2f2cc34061e05ada813386a78684a9d3f389e24c07d3c7f5e24.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/d1ec9eeef456f2f2cc34061e05ada813386a78684a9d3f389e24c07d3c7f5e24.jpg)

![e9450e810a1c2315eac5b9d41023862cb9fbaf7f3c7a18ccac3729023cc77fb5.jpg](../icml_results/795_CORAL_%20Uncertainty-Aware%20Regulation%20of%20Exposure%20Concentration%20in%20Recommender%20Systems/tables/e9450e810a1c2315eac5b9d41023862cb9fbaf7f3c7a18ccac3729023cc77fb5.jpg)

<a id="paper-0783"></a>

## CARE: Adaptive Calibration for Reliable Recommendations


### Images

![3f45904ff6ee97726361f59530c4555ffb0e88caf83facd29440692925967946.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/images/3f45904ff6ee97726361f59530c4555ffb0e88caf83facd29440692925967946.jpg)

![7bbebf3f783b1462619b1e79100304721b8914cefe8ab0975499a15954638293.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/images/7bbebf3f783b1462619b1e79100304721b8914cefe8ab0975499a15954638293.jpg)

![990d1b672e5a3ceb8a9669b841f34fc4ea72168b58001c553e803ddd919ed3f5.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/images/990d1b672e5a3ceb8a9669b841f34fc4ea72168b58001c553e803ddd919ed3f5.jpg)

![a77b27fe04843bca6474a346d795d629d2c5d3958ec57c87ebde0779dbcb9ed1.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/images/a77b27fe04843bca6474a346d795d629d2c5d3958ec57c87ebde0779dbcb9ed1.jpg)

![b8b687932393bb83aa10b28ad1ecc9f6eb287501c3e3fd3ad7b537ca755aa17b.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/images/b8b687932393bb83aa10b28ad1ecc9f6eb287501c3e3fd3ad7b537ca755aa17b.jpg)

### Tables

![0fbb63abdf137cf2fbf9871f68e5d4cec35693b1dea4d2c778bb1daae58f1162.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/0fbb63abdf137cf2fbf9871f68e5d4cec35693b1dea4d2c778bb1daae58f1162.jpg)

![16db86ab00603c21a7fb1e0b2a53c0e3c71e66361e4628102e9552ed3fd93044.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/16db86ab00603c21a7fb1e0b2a53c0e3c71e66361e4628102e9552ed3fd93044.jpg)

![2b9e92914cbe595ba6cc548a1ec4c391d35614cb384b936e941a319edd313bab.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/2b9e92914cbe595ba6cc548a1ec4c391d35614cb384b936e941a319edd313bab.jpg)

![3718295b0d6427dc7851cec0c931e49af6134bae01d44bfbb083544f5dd34dfe.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/3718295b0d6427dc7851cec0c931e49af6134bae01d44bfbb083544f5dd34dfe.jpg)

![381a9dd19089eef048bc3f36d974eea3851c48bf722ce7410f1f244c01d5976a.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/381a9dd19089eef048bc3f36d974eea3851c48bf722ce7410f1f244c01d5976a.jpg)

![71218bc47b92ce88177c14867ec0de390bf5ea7fa9181096a1fda1b01aa5f43d.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/71218bc47b92ce88177c14867ec0de390bf5ea7fa9181096a1fda1b01aa5f43d.jpg)

![a9c1167a9f8485736bd95a880ce661ef6af9cc08c2e72c7b708477f34b7df899.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/a9c1167a9f8485736bd95a880ce661ef6af9cc08c2e72c7b708477f34b7df899.jpg)

![b1a65bb46c9c95e3454a087b0f4db2fe3196321eba5e6b6d5005d2d2097b07c9.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/b1a65bb46c9c95e3454a087b0f4db2fe3196321eba5e6b6d5005d2d2097b07c9.jpg)

![b3cd4efa0f364453d476dcc0a8ecd49d6fd21fe3dd908338340fbb288c590ff9.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/b3cd4efa0f364453d476dcc0a8ecd49d6fd21fe3dd908338340fbb288c590ff9.jpg)

![baddd81a72c16b82d1c3443bbfe66eebe4a144d49a963d7ff42e4c1930e4102b.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/baddd81a72c16b82d1c3443bbfe66eebe4a144d49a963d7ff42e4c1930e4102b.jpg)

![c6db64bb2187fc5645758a7ac9f1f6c7cb9951e25d99b76e7de0b43327340fed.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/c6db64bb2187fc5645758a7ac9f1f6c7cb9951e25d99b76e7de0b43327340fed.jpg)

![c9b4d60ef7c67104732c042aa75d9e91662fc41c14ad257cca784fa14adb02e3.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/c9b4d60ef7c67104732c042aa75d9e91662fc41c14ad257cca784fa14adb02e3.jpg)

![dc5bfcdb5400759bafc1f99baac5b36182ae609be9919c8ad74799ed74dcadc3.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/dc5bfcdb5400759bafc1f99baac5b36182ae609be9919c8ad74799ed74dcadc3.jpg)

![f1e04d9f4f8f733bc440465e2904be8c1b9c300d1da133edd89a03a2f37608e1.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/f1e04d9f4f8f733bc440465e2904be8c1b9c300d1da133edd89a03a2f37608e1.jpg)

![fa0a47bf415736fa526dda4c57eff1e638c883ea8a03a5f85bfaf6ee229c7ccd.jpg](../icml_results/796_CARE_%20Adaptive%20Calibration%20for%20Reliable%20Recommendations/tables/fa0a47bf415736fa526dda4c57eff1e638c883ea8a03a5f85bfaf6ee229c7ccd.jpg)

<a id="paper-0784"></a>

## FOVI: A biologically-inspired foveated interface for deep vision models


### Images

![202d6f0aa5f22f54308e5f8437afe1fa74bf1c3561fa53da00162cba573acb6b.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/202d6f0aa5f22f54308e5f8437afe1fa74bf1c3561fa53da00162cba573acb6b.jpg)

![4c5a971c76007528f837c281c191cf5e17240e706422536bc1bb0f6c65c9234c.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/4c5a971c76007528f837c281c191cf5e17240e706422536bc1bb0f6c65c9234c.jpg)

![618a877c3150ca39a19cb1695eb50975fa1ddaec337796edec2b266f2e89a2d5.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/618a877c3150ca39a19cb1695eb50975fa1ddaec337796edec2b266f2e89a2d5.jpg)

![7f90ecf2062bc40a21c0089b291c2d7d2ade9aabeb8baf34d9d4eeda5df8e1c0.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/7f90ecf2062bc40a21c0089b291c2d7d2ade9aabeb8baf34d9d4eeda5df8e1c0.jpg)

![8228d2bc6d2324a796d97169d10d8a532b3a575990fb672e12a9eea7855d4034.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/8228d2bc6d2324a796d97169d10d8a532b3a575990fb672e12a9eea7855d4034.jpg)

![855747c6bd7c9bf81fb9a9f07d345d86e5f973faacc4075a482c67008c5deec7.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/855747c6bd7c9bf81fb9a9f07d345d86e5f973faacc4075a482c67008c5deec7.jpg)

![9318843716475cd791264b2f76dc3da436f18835f57e0f1947b1273f3e0026c1.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/9318843716475cd791264b2f76dc3da436f18835f57e0f1947b1273f3e0026c1.jpg)

![9386b929407158dfe534d591d9f51f0cf5b15362807804adee1740b33e501fd7.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/9386b929407158dfe534d591d9f51f0cf5b15362807804adee1740b33e501fd7.jpg)

![a45bda2b3294919087fba2f42ef787785d70ce48b489ae9fbca3154c810d52df.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/a45bda2b3294919087fba2f42ef787785d70ce48b489ae9fbca3154c810d52df.jpg)

![a648d1a21d1dffcb21d1f2dbfcf4f8738abfab18e92235c50c0238d7f5b41867.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/a648d1a21d1dffcb21d1f2dbfcf4f8738abfab18e92235c50c0238d7f5b41867.jpg)

![b1d1049bf6f96d71a3425c4bf36f03a1943f1fc37eb53e66140d83c323fec541.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/b1d1049bf6f96d71a3425c4bf36f03a1943f1fc37eb53e66140d83c323fec541.jpg)

![bbd2b8a61b12ab5e03497e4244fbc0a8fe5328b64850912ee8a631635f31c638.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/bbd2b8a61b12ab5e03497e4244fbc0a8fe5328b64850912ee8a631635f31c638.jpg)

![c84fb1c2e9b8f3fbf9e7703f80c0851fc6eae7e557cf2e2e5765e8e5ac2f30fd.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/c84fb1c2e9b8f3fbf9e7703f80c0851fc6eae7e557cf2e2e5765e8e5ac2f30fd.jpg)

![d28495321f925b32338eb79c5f397f1aed11ec9517ff0feea821e8f86699ff5f.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/d28495321f925b32338eb79c5f397f1aed11ec9517ff0feea821e8f86699ff5f.jpg)

![ea11ff98a2b37e4efe847b21211fd73abc1564903d56fdb584dcad44140bd4c8.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/ea11ff98a2b37e4efe847b21211fd73abc1564903d56fdb584dcad44140bd4c8.jpg)

![ea14d59a9cf70b404a109aaea092bb8b96f4272a9fd90a84700ffbaa0641267d.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/ea14d59a9cf70b404a109aaea092bb8b96f4272a9fd90a84700ffbaa0641267d.jpg)

![f3cb4a71dbb080ff49f40929843953dd8350c1de20e64ec2a0e25d57b7da8d6f.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/images/f3cb4a71dbb080ff49f40929843953dd8350c1de20e64ec2a0e25d57b7da8d6f.jpg)

### Tables

![9d94a12b587a6d5e30313503d9dd56bcaa5439ad65c2a9a1a8a7536992f88d68.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/tables/9d94a12b587a6d5e30313503d9dd56bcaa5439ad65c2a9a1a8a7536992f88d68.jpg)

![cce8c5192217da657ce68a3bc065f151d10db210852577ba4345c12d8cb765b8.jpg](../icml_results/797_FOVI_%20A%20biologically-inspired%20foveated%20interface%20for%20deep%20vision%20models/tables/cce8c5192217da657ce68a3bc065f151d10db210852577ba4345c12d8cb765b8.jpg)

<a id="paper-0785"></a>

## Bridge Matching Sampler: Scalable Sampling via Generalized Fixed-Point Diffusion Matching


### Images

![43c84752494c7462ab1247e40ef20130805477c2fd3f41f95e188d3fc520eb9d.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/43c84752494c7462ab1247e40ef20130805477c2fd3f41f95e188d3fc520eb9d.jpg)

![655d61c87e0b56d099dbd10d14694ffcaa34776e8ce3c66a06b91a25ba210489.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/655d61c87e0b56d099dbd10d14694ffcaa34776e8ce3c66a06b91a25ba210489.jpg)

![82595c68b21f9eab5acc53ccb51a82bb544b7135abd98f398b379087fae57fd9.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/82595c68b21f9eab5acc53ccb51a82bb544b7135abd98f398b379087fae57fd9.jpg)

![829155b3a682c13d48d45d72cdb81f4caf5da4b5c04253923131b347471d5955.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/829155b3a682c13d48d45d72cdb81f4caf5da4b5c04253923131b347471d5955.jpg)

![8540bbf3383ecc63456d2e80097846fbed404e8b798463325d69c9871d690b7b.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/8540bbf3383ecc63456d2e80097846fbed404e8b798463325d69c9871d690b7b.jpg)

![859024a3755627a42cff58d9666e5afc6fa11d8484992fe273de71ce1185bcf7.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/859024a3755627a42cff58d9666e5afc6fa11d8484992fe273de71ce1185bcf7.jpg)

![900213defb6fa030877a255d5d42b7bfaeeda9119271cf6a61ef2ef10dac7718.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/900213defb6fa030877a255d5d42b7bfaeeda9119271cf6a61ef2ef10dac7718.jpg)

![90a04b253dfd8af8714f851cb4cf626dae7365b942cc6044f063d7ca9542b702.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/90a04b253dfd8af8714f851cb4cf626dae7365b942cc6044f063d7ca9542b702.jpg)

![9f34ad80b4dbf841f296b75f60dba352cfbd0724b6d66aef8cbe63b4dcb55dce.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/9f34ad80b4dbf841f296b75f60dba352cfbd0724b6d66aef8cbe63b4dcb55dce.jpg)

![a8028615df85a925bcf198a5a2cf3f5cbad707bdce19f4a86527818b45eea29a.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/a8028615df85a925bcf198a5a2cf3f5cbad707bdce19f4a86527818b45eea29a.jpg)

![ab5cea8bd48c1a09f298916859f2d4cf486555009f8df4d75ac217a81217b33b.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/ab5cea8bd48c1a09f298916859f2d4cf486555009f8df4d75ac217a81217b33b.jpg)

![ac9b0e94284ade5c5f5e97e854c52fcd72e15a88da287a1af63bd151ec4e3396.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/ac9b0e94284ade5c5f5e97e854c52fcd72e15a88da287a1af63bd151ec4e3396.jpg)

![eae5b6b907af7d43220e06d19cb30000c04f15b7feb4c86acdd3b231d460ecc2.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/images/eae5b6b907af7d43220e06d19cb30000c04f15b7feb4c86acdd3b231d460ecc2.jpg)

### Tables

![1a5cd02a93df4c41e1410950cf7c32828c02ba6951bc8bf9376a932505b7f143.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/1a5cd02a93df4c41e1410950cf7c32828c02ba6951bc8bf9376a932505b7f143.jpg)

![446c146264a3c54be3f12158ff9c814d4e17d95e5d5f01571376a3be5384d1ae.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/446c146264a3c54be3f12158ff9c814d4e17d95e5d5f01571376a3be5384d1ae.jpg)

![5e65df5d2e925acd3f987c2c62e097dab2f94acd45499de3db114d8a8b6e2931.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/5e65df5d2e925acd3f987c2c62e097dab2f94acd45499de3db114d8a8b6e2931.jpg)

![5ff4a01a0e24601bb8c898d2cf91edb8f8a764fd83fe16f9de3c0690d98b10aa.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/5ff4a01a0e24601bb8c898d2cf91edb8f8a764fd83fe16f9de3c0690d98b10aa.jpg)

![bfef8c5f65d217a34ba298a51601986673e6af1091a6228a4ad9a3928f3d03e0.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/bfef8c5f65d217a34ba298a51601986673e6af1091a6228a4ad9a3928f3d03e0.jpg)

![c2df5c49fab8614a31c1b0e70fd56563d5d037bb4103c0b7c534d27e46141800.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/c2df5c49fab8614a31c1b0e70fd56563d5d037bb4103c0b7c534d27e46141800.jpg)

![ec5caf95e25c224c5cfe36657764c8d6796b1f98258ebdccfe3e90f8840779b0.jpg](../icml_results/798_Bridge%20Matching%20Sampler_%20Scalable%20Sampling%20via%20Generalized%20Fixed-Point%20Diffusion%20Matching/tables/ec5caf95e25c224c5cfe36657764c8d6796b1f98258ebdccfe3e90f8840779b0.jpg)

<a id="paper-0786"></a>

## Batched First-Order Methods for Parallel LP Solving in MIP


### Images

![0022693adba8b0740096f87a9df0447eff4c38add4b18945bdaf513683aa1ce3.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/images/0022693adba8b0740096f87a9df0447eff4c38add4b18945bdaf513683aa1ce3.jpg)

![611a9aeb1612fc628c02b923299995e759bf5caac174229caba94c634c0dca48.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/images/611a9aeb1612fc628c02b923299995e759bf5caac174229caba94c634c0dca48.jpg)

![65f5055296453ffaf8a4cdb46e7e7256d28ae1ddaf70802d2be7d02e6ed4b2f6.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/images/65f5055296453ffaf8a4cdb46e7e7256d28ae1ddaf70802d2be7d02e6ed4b2f6.jpg)

![e354c5d454e5f538f2682213a48225ac17d4e7feded1d86c3370e44c02bb0f28.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/images/e354c5d454e5f538f2682213a48225ac17d4e7feded1d86c3370e44c02bb0f28.jpg)

### Tables

![1159dce2f78f9cf43f9b6b0cf045bfc89b2277cda4362c9fc5061c92306a4102.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/1159dce2f78f9cf43f9b6b0cf045bfc89b2277cda4362c9fc5061c92306a4102.jpg)

![1226200307b37cbd6e5ab3b46f64a27b45cdef0cb266ba460b281b8f66a4dc1d.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/1226200307b37cbd6e5ab3b46f64a27b45cdef0cb266ba460b281b8f66a4dc1d.jpg)

![37151daa9d102aa34993892ad0b310d3440da061dafbe31bf19a63167885b016.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/37151daa9d102aa34993892ad0b310d3440da061dafbe31bf19a63167885b016.jpg)

![6d18808eaea4fa16491027cfb5c1ab459852c5b8e8b23644f7f46542aeac50fc.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/6d18808eaea4fa16491027cfb5c1ab459852c5b8e8b23644f7f46542aeac50fc.jpg)

![7fa64af65813390a290fd58853fdf4e269f3f3b48686c0f8987585981ac3a1d2.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/7fa64af65813390a290fd58853fdf4e269f3f3b48686c0f8987585981ac3a1d2.jpg)

![ac74d7a87dfbd5cef44e3bfe424153962fb0c2b46ca3dd723b8b8f0dad178771.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/ac74d7a87dfbd5cef44e3bfe424153962fb0c2b46ca3dd723b8b8f0dad178771.jpg)

![d7210dff67fea7e3401b80e6e7f046e01c0cd9f5e41ad198d3bcbf790a5f36a5.jpg](../icml_results/799_Batched%20First-Order%20Methods%20for%20Parallel%20LP%20Solving%20in%20MIP/tables/d7210dff67fea7e3401b80e6e7f046e01c0cd9f5e41ad198d3bcbf790a5f36a5.jpg)

<a id="paper-0787"></a>

## Temper-Then-Tilt: Principled Unlearning for Generative Models through Tempering and Classifier Guidance


### Images

![9c791e9ae232b6ca8494bc00fff4098e06869b562a649efa90d9b839ab0756e1.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/images/9c791e9ae232b6ca8494bc00fff4098e06869b562a649efa90d9b839ab0756e1.jpg)

![a1b261c87f83e2c07adb1096ccb44dc73e87006abaa2a63c254af23e9bdd567d.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/images/a1b261c87f83e2c07adb1096ccb44dc73e87006abaa2a63c254af23e9bdd567d.jpg)

![a8b3a32cea1d518f8f545865cb560d306f85a722900bd21e950a6f39aae7819c.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/images/a8b3a32cea1d518f8f545865cb560d306f85a722900bd21e950a6f39aae7819c.jpg)

![e2d5df4ca8cb77c1a408620ba27c8ecf95078b366b26e71a315ec59b6964d4f3.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/images/e2d5df4ca8cb77c1a408620ba27c8ecf95078b366b26e71a315ec59b6964d4f3.jpg)

![f537762c9d87f4510cc604f91ed0469b5a2126782a026eac831a022226dcfefe.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/images/f537762c9d87f4510cc604f91ed0469b5a2126782a026eac831a022226dcfefe.jpg)

### Tables

![1afe16c3eb1120e78b4304d55645f02aee087f15b1bbd579c405ed0b29bbe7e6.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/tables/1afe16c3eb1120e78b4304d55645f02aee087f15b1bbd579c405ed0b29bbe7e6.jpg)

![47fc64e4ee2d254a0e4c22507ceb9bc2d36e87fe5175a908fa4eb0586a4e6a3a.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/tables/47fc64e4ee2d254a0e4c22507ceb9bc2d36e87fe5175a908fa4eb0586a4e6a3a.jpg)

![9e2f91355157becc114c0521c51eeaf6811e550c1a0ab48b6aa5fd8683ab0242.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/tables/9e2f91355157becc114c0521c51eeaf6811e550c1a0ab48b6aa5fd8683ab0242.jpg)

![c013778d877acd78d3e13fd8bc8975c95230d13d8e215c267b08f30548ac9451.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/tables/c013778d877acd78d3e13fd8bc8975c95230d13d8e215c267b08f30548ac9451.jpg)

![f04523798fea31423d6f27d387021074bfd69b5b6e5cebaa533c1c7a95054387.jpg](../icml_results/800_Temper-Then-Tilt_%20Principled%20Unlearning%20for%20Generative%20Models%20through%20Tempering%20and%20Classifier%20Guida/tables/f04523798fea31423d6f27d387021074bfd69b5b6e5cebaa533c1c7a95054387.jpg)

<a id="paper-0788"></a>

## Noise-Guided Transport: Imitation Learning from Random Priors


### Images

![0783a1e0420b2da7ff0eb658c59a9207fd6499c3247c85b1570668fc723f3b87.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/0783a1e0420b2da7ff0eb658c59a9207fd6499c3247c85b1570668fc723f3b87.jpg)

![079bb515e42934632441d89b9a5fcd3402077a998d79085bb7354137126127f4.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/079bb515e42934632441d89b9a5fcd3402077a998d79085bb7354137126127f4.jpg)

![0c36986e0776f40d4bed0753e4e8957cc64b204bbe1b72ba80135cefd539bb74.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/0c36986e0776f40d4bed0753e4e8957cc64b204bbe1b72ba80135cefd539bb74.jpg)

![218b0c4b850c53b5c708c645d5fe17504c1970821cf58cb07be5a639ae44f03b.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/218b0c4b850c53b5c708c645d5fe17504c1970821cf58cb07be5a639ae44f03b.jpg)

![66136216c9fc33a94ab045c610b2ce0a4f449b35a142f3199a6eae37e602c9d8.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/66136216c9fc33a94ab045c610b2ce0a4f449b35a142f3199a6eae37e602c9d8.jpg)

![6685b5f25aa7689c6f8b6d14aa9c58d9a2cdde8d50156acb8fb41fa7b9fb395f.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/6685b5f25aa7689c6f8b6d14aa9c58d9a2cdde8d50156acb8fb41fa7b9fb395f.jpg)

![89d66f9de8c3afba0fcba8a4ac1042f6ea581867ac008ca5e13056abbfd28888.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/89d66f9de8c3afba0fcba8a4ac1042f6ea581867ac008ca5e13056abbfd28888.jpg)

![98d0868ef0910456daabc4fbd291ee2fc44e0948d7f228f33a885dd4d86810e7.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/98d0868ef0910456daabc4fbd291ee2fc44e0948d7f228f33a885dd4d86810e7.jpg)

![d133fdd6b331bf980e6f49ffb923d39239b3245a7056e85d6d129ff6fb995dd8.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/d133fdd6b331bf980e6f49ffb923d39239b3245a7056e85d6d129ff6fb995dd8.jpg)

![fe81c88ae96331619d5d450417bf496493c1365c311ded632ddc701760386f1a.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/images/fe81c88ae96331619d5d450417bf496493c1365c311ded632ddc701760386f1a.jpg)

### Tables

![354eb10081dc9ba705f3fcdf54f4e51c4b22d4cfb73c10c6cc6d3aeb9d89e79b.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/tables/354eb10081dc9ba705f3fcdf54f4e51c4b22d4cfb73c10c6cc6d3aeb9d89e79b.jpg)

![73e52aa40dd7d859d98db3a2a40955ed7a24875a87c5b26888479bdeb641eb7e.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/tables/73e52aa40dd7d859d98db3a2a40955ed7a24875a87c5b26888479bdeb641eb7e.jpg)

![d1c008fcedd070476445c6d986c1b1c0c6e216c99bde6107ec3fd2ce1a6bdcbf.jpg](../icml_results/802_Noise-Guided%20Transport_%20Imitation%20Learning%20from%20Random%20Priors/tables/d1c008fcedd070476445c6d986c1b1c0c6e216c99bde6107ec3fd2ce1a6bdcbf.jpg)

<a id="paper-0789"></a>

## Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction


### Images

![1d716b1d24746120dfea016fea3da00679229d9b822540411443ace3d497e86b.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/1d716b1d24746120dfea016fea3da00679229d9b822540411443ace3d497e86b.jpg)

![2f732ec52ab42bd2187cecbd6034a546020ff1ecfd14f3522df3bfc1f0e8d50b.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/2f732ec52ab42bd2187cecbd6034a546020ff1ecfd14f3522df3bfc1f0e8d50b.jpg)

![555951d000beb38746b30b5411bea39195503891e501560de0dd86f8d77c378c.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/555951d000beb38746b30b5411bea39195503891e501560de0dd86f8d77c378c.jpg)

![727072369d581fa08bfbdad23134869d0f70fcf3635347ac06622781d26564c7.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/727072369d581fa08bfbdad23134869d0f70fcf3635347ac06622781d26564c7.jpg)

![72be0f0bf0b686d5e378fff2a7dcd0184cfdef04c5aa394a8567a40352810a28.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/72be0f0bf0b686d5e378fff2a7dcd0184cfdef04c5aa394a8567a40352810a28.jpg)

![8f17182e32abdeb28a8e90fbb688ce8e6d8149bb93e233c6891b764ce9bdb022.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/8f17182e32abdeb28a8e90fbb688ce8e6d8149bb93e233c6891b764ce9bdb022.jpg)

![928a12cbe8f021af9a3e32d8f37472e09f9a2a855df22b421ac412a9ffe68882.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/928a12cbe8f021af9a3e32d8f37472e09f9a2a855df22b421ac412a9ffe68882.jpg)

![c9c14aca9a75d61dcebb5e6f5c79a03242f248038d72c82dd1e5c4f663987a90.jpg](../icml_results/803_Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models_%20Insights%20into%20the%20Lookahead%20Capabil/images/c9c14aca9a75d61dcebb5e6f5c79a03242f248038d72c82dd1e5c4f663987a90.jpg)

<a id="paper-0790"></a>

## Benchmarking Dense and Indiscernible Object Counting with Blueberries


### Images

![18266030b4e54b136be2fc678850303c2479207f040c5d5905135ae5da223ed5.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/18266030b4e54b136be2fc678850303c2479207f040c5d5905135ae5da223ed5.jpg)

![1a1e4d881b850f49d904aab918e3f7dd8a4390cab07c14d12e16ae966e3bdf05.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/1a1e4d881b850f49d904aab918e3f7dd8a4390cab07c14d12e16ae966e3bdf05.jpg)

![1e161a17e5d525115b0cc8e2acfba18b8a5d8920759a0d782529e2b36ac5509c.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/1e161a17e5d525115b0cc8e2acfba18b8a5d8920759a0d782529e2b36ac5509c.jpg)

![2f3f08e3c7fe10a9e596b6ee118c654796eb86c9b1e7aad16117d6d8a3ea2c92.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/2f3f08e3c7fe10a9e596b6ee118c654796eb86c9b1e7aad16117d6d8a3ea2c92.jpg)

![4c5bbfba088ec92ce0cea94829dd02773e2db35e6ff329a6518283c1c2a45f5d.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/4c5bbfba088ec92ce0cea94829dd02773e2db35e6ff329a6518283c1c2a45f5d.jpg)

![746ce32de381d37f10c54607460ec89a0b5c8c5f99f348614e5eed1b00473841.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/746ce32de381d37f10c54607460ec89a0b5c8c5f99f348614e5eed1b00473841.jpg)

![8a8e87ad8c470fbc56f52cd93f8a616cd1e17dd40c919bcf33e57970b069dace.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/8a8e87ad8c470fbc56f52cd93f8a616cd1e17dd40c919bcf33e57970b069dace.jpg)

![911e5a722da343f3d89c7c58c739d57dc41405496bd4ecfc55b81d9625a09c0f.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/911e5a722da343f3d89c7c58c739d57dc41405496bd4ecfc55b81d9625a09c0f.jpg)

![a8643148616641ba079bca260d4c231d162ad05fd75d8b75ef1a24b032e78c2d.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/a8643148616641ba079bca260d4c231d162ad05fd75d8b75ef1a24b032e78c2d.jpg)

![d85cb237e2095756ae8a6da3ae6629bf292328f4fc55b01d4b18f42172e0e2a5.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/d85cb237e2095756ae8a6da3ae6629bf292328f4fc55b01d4b18f42172e0e2a5.jpg)

![e2221387f022bd774409095efd93164a730c5d94f4a2b72738e333df32aa596a.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/e2221387f022bd774409095efd93164a730c5d94f4a2b72738e333df32aa596a.jpg)

![e2bb27b2383df322cc4389ae336e6b1bb3068d0eae7e77467b0d2b3c70f81747.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/e2bb27b2383df322cc4389ae336e6b1bb3068d0eae7e77467b0d2b3c70f81747.jpg)

![f239e8a63f71690f65fe5a1480aa90b5f5504e5b51fb0b5851dc0c91d89b4690.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/images/f239e8a63f71690f65fe5a1480aa90b5f5504e5b51fb0b5851dc0c91d89b4690.jpg)

### Tables

![369904d61de7d3e76d563f6a324bd839b1ccf98df5a6a781bcbbeed0a0a038aa.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/369904d61de7d3e76d563f6a324bd839b1ccf98df5a6a781bcbbeed0a0a038aa.jpg)

![3ea47eac7a73ee4403e92b58224eb433090b41dab0166aa89e5b3be14ab3fd19.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/3ea47eac7a73ee4403e92b58224eb433090b41dab0166aa89e5b3be14ab3fd19.jpg)

![538c82786083a10db32548d40a275a28190127e0db169de83b1b115a81b5f500.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/538c82786083a10db32548d40a275a28190127e0db169de83b1b115a81b5f500.jpg)

![6d665a3b29ce7704872c25db8177acc5fb820e8ceb4edc22f59e207087f50ed9.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/6d665a3b29ce7704872c25db8177acc5fb820e8ceb4edc22f59e207087f50ed9.jpg)

![81518aec2b1f2e589dd55b77aab1d8fca7ae66f26cc0e789512a55277622c50e.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/81518aec2b1f2e589dd55b77aab1d8fca7ae66f26cc0e789512a55277622c50e.jpg)

![c1e51727bfc9beb4ad01ad8450967b408e324ebfa2c2c91da756a52738b71e26.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/c1e51727bfc9beb4ad01ad8450967b408e324ebfa2c2c91da756a52738b71e26.jpg)

![e66fa3f1dc1dfe63501674d3439a9417812b98a47c886cb16e9dfc4fee03c8a1.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/e66fa3f1dc1dfe63501674d3439a9417812b98a47c886cb16e9dfc4fee03c8a1.jpg)

![e7db9d9ac35d0a4e67fbaeb65a9698092b13b4256128dd89a4ec912a95252bbe.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/e7db9d9ac35d0a4e67fbaeb65a9698092b13b4256128dd89a4ec912a95252bbe.jpg)

![f1abef6772595de1ab7f7f01f48c782086863b533993c7324681d06658307c21.jpg](../icml_results/804_Benchmarking%20Dense%20and%20Indiscernible%20Object%20Counting%20with%20Blueberries/tables/f1abef6772595de1ab7f7f01f48c782086863b533993c7324681d06658307c21.jpg)

<a id="paper-0791"></a>

## Efficient Multi-modal Dataset Distillation via Analytic Parameter Matching


### Images

![052e12e5c878bbd136037b659804468541470af60a4904ec1e6ecfb9e6678b0a.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/images/052e12e5c878bbd136037b659804468541470af60a4904ec1e6ecfb9e6678b0a.jpg)

![8222d342358a6ad31e66a757246a9664d4d989f9604ec034110853bf2566c883.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/images/8222d342358a6ad31e66a757246a9664d4d989f9604ec034110853bf2566c883.jpg)

![9dbe7bf44d5866722cc662450d40260e94fc16dd809356e47c67a138969659af.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/images/9dbe7bf44d5866722cc662450d40260e94fc16dd809356e47c67a138969659af.jpg)

![cb45c53b9ce5a5fe548b9c777eb16c129c25b842ee5c78d39128f4a8c972b85c.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/images/cb45c53b9ce5a5fe548b9c777eb16c129c25b842ee5c78d39128f4a8c972b85c.jpg)

![f6adfb0a72b9f19c75bb81978cfae12f92dea4965e5243cbd1426ffd23d45281.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/images/f6adfb0a72b9f19c75bb81978cfae12f92dea4965e5243cbd1426ffd23d45281.jpg)

### Tables

![09f1cfd0850426410754ad418efe4c0c44e6e89b332346d3d930e8be6514f73c.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/09f1cfd0850426410754ad418efe4c0c44e6e89b332346d3d930e8be6514f73c.jpg)

![163844b2ec03743103614e6506e3a49502fc8ba541c7c8088ef46b96db1a989e.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/163844b2ec03743103614e6506e3a49502fc8ba541c7c8088ef46b96db1a989e.jpg)

![1ea3fd74724331e412f5e6923d44314c456e5d8397986c6001d90069758d9ed3.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/1ea3fd74724331e412f5e6923d44314c456e5d8397986c6001d90069758d9ed3.jpg)

![34671fd5502e2635b849b2a55005609298ca0f0346785bd89e3c39ec039f219c.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/34671fd5502e2635b849b2a55005609298ca0f0346785bd89e3c39ec039f219c.jpg)

![411f3c3d41281ad0d0675fe1a924ec04268ec525d23f1a9e3740a88f2f6e7289.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/411f3c3d41281ad0d0675fe1a924ec04268ec525d23f1a9e3740a88f2f6e7289.jpg)

![7a46bee8e6edd2b15de641eb84a7ab19a962c539aed1fb04ce43309c07c0b0b3.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/7a46bee8e6edd2b15de641eb84a7ab19a962c539aed1fb04ce43309c07c0b0b3.jpg)

![7e516b3940fb7f4a8dc4c6f89e6382263156184e49c2aa47ee7a8f11b16f48ea.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/7e516b3940fb7f4a8dc4c6f89e6382263156184e49c2aa47ee7a8f11b16f48ea.jpg)

![8e98e0b4fd41a9e8b38590e8eaa42cb8f610f455a875aee9f6c7045a0b4acb6c.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/8e98e0b4fd41a9e8b38590e8eaa42cb8f610f455a875aee9f6c7045a0b4acb6c.jpg)

![a0a0033932578bb29fd3cd61a32c93aa4879779537f8f2ef5252a6bd7de54efd.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/a0a0033932578bb29fd3cd61a32c93aa4879779537f8f2ef5252a6bd7de54efd.jpg)

![a76eb0956e783e76297a8eebd9c570fc24013dfe06fcbfa92cde2ed36f7114ce.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/a76eb0956e783e76297a8eebd9c570fc24013dfe06fcbfa92cde2ed36f7114ce.jpg)

![b211a107cb79434223f0d0e1a897c32f6da765e2b1c2591de7121fc090d5f020.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/b211a107cb79434223f0d0e1a897c32f6da765e2b1c2591de7121fc090d5f020.jpg)

![b311288823d12449823d5382ea86b91c7aa48978a12bbeeedc32a4118434ec7e.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/b311288823d12449823d5382ea86b91c7aa48978a12bbeeedc32a4118434ec7e.jpg)

![b5fe9a90aaac53651ec2d2ee31a77fed20ce80691b97e594e01c436c061f55fb.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/b5fe9a90aaac53651ec2d2ee31a77fed20ce80691b97e594e01c436c061f55fb.jpg)

![d7cac0c14d053936ffee0edec2f0fe15513c65f4c2d8dc556500adbe0da6e9e2.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/d7cac0c14d053936ffee0edec2f0fe15513c65f4c2d8dc556500adbe0da6e9e2.jpg)

![d8c1e38cbcad9c21177f9936d4a71c5a048a2c22b711389dff9aff9cf607106a.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/d8c1e38cbcad9c21177f9936d4a71c5a048a2c22b711389dff9aff9cf607106a.jpg)

![e8b2cdf37bd714c5250dacc644ae6ecaca7a84f314109c980fdee682cf11705d.jpg](../icml_results/805_Efficient%20Multi-modal%20Dataset%20Distillation%20via%20Analytic%20Parameter%20Matching/tables/e8b2cdf37bd714c5250dacc644ae6ecaca7a84f314109c980fdee682cf11705d.jpg)

<a id="paper-0792"></a>

## Counterfactual Bootstrap for Robust Meta-Reinforcement Learning

### Images

![2f0bd3e4d4aa52ec4ba5e120343f353e8e1c80f74b1fde539b953d6a9a8e6be5.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/2f0bd3e4d4aa52ec4ba5e120343f353e8e1c80f74b1fde539b953d6a9a8e6be5.jpg)

![4bd5c508229c17e822c57343d1a07b37cddde5202f379be05ea79c083def10b9.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/4bd5c508229c17e822c57343d1a07b37cddde5202f379be05ea79c083def10b9.jpg)

![5dd53c7ccc4bec2d00c7ef33bad99493f4e8ad00dae719908e1cc799d3536e54.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/5dd53c7ccc4bec2d00c7ef33bad99493f4e8ad00dae719908e1cc799d3536e54.jpg)

![89caf925f0826c31cd63091cca51d41ad923d263557b3e65deadfa779fe10529.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/89caf925f0826c31cd63091cca51d41ad923d263557b3e65deadfa779fe10529.jpg)

![940049322044fe558f75d0866352bc4acceaa0ba5aa2c655a935e12180702081.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/940049322044fe558f75d0866352bc4acceaa0ba5aa2c655a935e12180702081.jpg)

![b435dd076b3311b491e33fc8418f6c7bfd3dd997fa8eac1dacbd22b423646ead.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/b435dd076b3311b491e33fc8418f6c7bfd3dd997fa8eac1dacbd22b423646ead.jpg)

![d4fab2fb13cf892e3242818a670012451180cd9adc35fa363c83db3a3a37f3f7.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/d4fab2fb13cf892e3242818a670012451180cd9adc35fa363c83db3a3a37f3f7.jpg)

![ebe151ddb5c25d74ffce78c3c8e43609ea04070ee8d53af7c4ce507e883da3ce.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/images/ebe151ddb5c25d74ffce78c3c8e43609ea04070ee8d53af7c4ce507e883da3ce.jpg)

### Tables

![0d2be609689f170d9f58058616720470b845cc79e499062edb8776c2f4e7f4f6.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/tables/0d2be609689f170d9f58058616720470b845cc79e499062edb8776c2f4e7f4f6.jpg)

![79a921e3e4612609e8bfe91f8f1117769c4fd8a2b5104c044400d039704250b7.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/tables/79a921e3e4612609e8bfe91f8f1117769c4fd8a2b5104c044400d039704250b7.jpg)

![da6a6ae6e3269ad0cd3656199fa0218d0f0d86d905085341b579fc34dd54161d.jpg](../icml_results/806_Counterfactual%20Bootstrap%20for%20Robust%20Meta-Reinforcement%20Learning/tables/da6a6ae6e3269ad0cd3656199fa0218d0f0d86d905085341b579fc34dd54161d.jpg)
