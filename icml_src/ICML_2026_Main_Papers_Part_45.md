# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 45 of 197

## 目录 (Table of Contents)

1. [Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](#paper-1453)
2. [Graph-Preference Learning: Debiasing Network-Sampled Human Feedback for Target Welfare Estimation](#paper-1454)
3. [ZeroDiff: Zero-Shot Time Series Reconstruction via Informed-Prior Diffusion](#paper-1455)
4. [Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation](#paper-1456)
5. [De4D-SLAM: Gradient-Isolated Static-Dynamic Decoupling for Monocular SLAM in Dynamic Environments](#paper-1457)
6. [Rethinking Time-Series Imputation as Conditional Inference along Temporal Evolution](#paper-1458)
7. [TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning](#paper-1459)
8. [Navigating the Pareto Frontier of Alignment: Spectrum-Adaptive Fine-Tuning for LLMs](#paper-1460)
9. [NExT-Guard: Streaming Safeguard without Token-Level Labels](#paper-1461)
10. [Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness](#paper-1462)
11. [SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling](#paper-1463)
12. [Bridging On-Device and Cloud LLMs for Collaborative Reasoning: A Unified Methodology for Local Routing and Post-Training](#paper-1464)
13. [Federated Sketching LoRA: A Flexible Framework for Heterogeneous Collaborative Fine-Tuning of LLMs](#paper-1465)
14. [Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models](#paper-1466)
15. [MC-HNN: Learning Latent Structural Semantics and High-Rank Representations for Hypergraph Neural Networks](#paper-1467)
16. [OMP: One-step Meanflow Policy with Directional Alignment](#paper-1468)
17. [MINT: Minimal Information Neuro-Symbolic Tree for Objective-Driven Knowledge-Gap Reasoning and Active Elicitation](#paper-1469)
18. [PRM-PBE: Process Reward Model for Reinforcement Learning in Programming-by-Example](#paper-1470)
19. [LOZO+: Provably Efficient Zeroth-Order Fine-Tuning via Greedy Low-Rank Subspace Selection](#paper-1471)
20. [iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](#paper-1472)
21. [Information Geometry Loss for Time Series Forecasting](#paper-1473)
22. [Trajectory-Aware Spiking DiTs Conversion via Membrane Potential Error-Feedback](#paper-1474)
23. [Event2Vec: Processing neuromorphic events directly by representations in vector space](#paper-1475)
24. [Dynamic Fractal Mamba: A Neural Renormalization Group Flow for Scale-Invariant Sequence Modeling](#paper-1476)
25. [VBA: Vector Bundle Attention for Intrinsically Geometric Representation Learning](#paper-1477)
26. [Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization](#paper-1478)
27. [Real Data Lies: Unveiling and Closing the Quality Shortcut in Generalizable AI-Generated Video Detection](#paper-1479)
28. [ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](#paper-1480)
29. [BIOARC: Discovering Optimal Neural Architectures for Biological Foundation Models](#paper-1481)
30. [Decoupled Low-Rank Adaptation for Robust Federated Fine-Tuning](#paper-1482)
31. [Artificial Hippocampus Networks for Efficient Long-Context Modeling](#paper-1483)
32. [Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](#paper-1484)
33. [StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation](#paper-1485)

---


<a id="paper-1453"></a>

## Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds

### Images

![05fb4ff9005f06b46e06bdcf88960722e301719ee1cd92b96f31048441fd445a.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/05fb4ff9005f06b46e06bdcf88960722e301719ee1cd92b96f31048441fd445a.jpg)

![09664c1401ef942af56d93063221de330fa2059687eea684f050516027296b26.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/09664c1401ef942af56d93063221de330fa2059687eea684f050516027296b26.jpg)

![0ce47809a9ee474c3225aaf15fac30fd4b95750cfa9ac618d575d5343f61fc11.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/0ce47809a9ee474c3225aaf15fac30fd4b95750cfa9ac618d575d5343f61fc11.jpg)

![1d9401bcb7e1c48b4322c8385501ff3d560ed9c1560143c385b07ecda4af93c9.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/1d9401bcb7e1c48b4322c8385501ff3d560ed9c1560143c385b07ecda4af93c9.jpg)

![1de0a1d63151d49c6fabe1ac5efb71ad36f79e87b5b00a3d241620194a89ce9c.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/1de0a1d63151d49c6fabe1ac5efb71ad36f79e87b5b00a3d241620194a89ce9c.jpg)

![211744b4fb56b0cfada0779199cdaf360636d62b1cf374c119c2d9cd5c813d85.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/211744b4fb56b0cfada0779199cdaf360636d62b1cf374c119c2d9cd5c813d85.jpg)

![240687eb77f85400ef78e818fbad6f32b8e24138617c59aad3b7ed94e97dd66b.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/240687eb77f85400ef78e818fbad6f32b8e24138617c59aad3b7ed94e97dd66b.jpg)

![27165a0b271403ea23802d66ed51e1eaf49e7cb4fc52410104017b412bd928b0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/27165a0b271403ea23802d66ed51e1eaf49e7cb4fc52410104017b412bd928b0.jpg)

![2b478e33fc3152e68f1748891b4663697dd1acb75a129585157009cb6a072e26.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/2b478e33fc3152e68f1748891b4663697dd1acb75a129585157009cb6a072e26.jpg)

![3a710d444071bb21f28be681ffcd9d78d85b09a35c792182b473b6b9b0c8ae61.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/3a710d444071bb21f28be681ffcd9d78d85b09a35c792182b473b6b9b0c8ae61.jpg)

![45e8daae3517522848394e108b78f5e27a234d75fa0a208978b4906e97cfc8e0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/45e8daae3517522848394e108b78f5e27a234d75fa0a208978b4906e97cfc8e0.jpg)

![478774a8ad32c4e14768040dd4523e6a6ab1d484f97d3e509f53f93c1b4405d2.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/478774a8ad32c4e14768040dd4523e6a6ab1d484f97d3e509f53f93c1b4405d2.jpg)

![551a3c8cb5e60a994549aced2129b05971c07eed7d52bbbb8f90ca32962fa77e.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/551a3c8cb5e60a994549aced2129b05971c07eed7d52bbbb8f90ca32962fa77e.jpg)

![5c6f7fd8270a0b3836d8b21b4dd817904c9e926b8d1b8d674e8c4c4618c44d57.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5c6f7fd8270a0b3836d8b21b4dd817904c9e926b8d1b8d674e8c4c4618c44d57.jpg)

![5cde67a52649b3c27377a4697ea54b7bd5dc9f8ee00a6b3d2ceae20eca5bcbd7.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5cde67a52649b3c27377a4697ea54b7bd5dc9f8ee00a6b3d2ceae20eca5bcbd7.jpg)

![5d6b551eac2b0553ddfe10854b8fb59513c6b64e081f01295cdb93c53f75dde0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5d6b551eac2b0553ddfe10854b8fb59513c6b64e081f01295cdb93c53f75dde0.jpg)

![5e65f54114a43365ef27e979e02252af0b327746a233ddd1caf7533db736a9d3.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5e65f54114a43365ef27e979e02252af0b327746a233ddd1caf7533db736a9d3.jpg)

![5f445b58b36192474e0f593ee766ac3d9faa386ed162dcc3518a6bb21ac528f9.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/5f445b58b36192474e0f593ee766ac3d9faa386ed162dcc3518a6bb21ac528f9.jpg)

![7316ae64e2805b229ee4f4e34a84887753a70ff2e3c5d2cf7820e63ca5ddf9f6.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/7316ae64e2805b229ee4f4e34a84887753a70ff2e3c5d2cf7820e63ca5ddf9f6.jpg)

![7c6a4824b9fa4b625a5a88275d3adff1d18a497fdc623ff19ce79bf933478b79.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/7c6a4824b9fa4b625a5a88275d3adff1d18a497fdc623ff19ce79bf933478b79.jpg)

![a206107934e09e42777ca22b6fd5cb7541d1f22b8dcf111be55301ff2d6549dd.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/a206107934e09e42777ca22b6fd5cb7541d1f22b8dcf111be55301ff2d6549dd.jpg)

![a3748a6a889ceb4040e974fcca706865a06de47781cc45394060f5f533f10e4d.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/a3748a6a889ceb4040e974fcca706865a06de47781cc45394060f5f533f10e4d.jpg)

![a744ff389ad855fbecc9dd939b2177a71ddd0e487d50007da7d8fb2e3e84a0b9.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/a744ff389ad855fbecc9dd939b2177a71ddd0e487d50007da7d8fb2e3e84a0b9.jpg)

![c0d857b162dacdc20f4bc0c9d978a0034f628a1a7532730fe9029b35b2bdd5a0.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/c0d857b162dacdc20f4bc0c9d978a0034f628a1a7532730fe9029b35b2bdd5a0.jpg)

![c28146104a452ad4ac7ba9ed76cecafc613fe4a53310951ce72f6d78198c43ed.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/c28146104a452ad4ac7ba9ed76cecafc613fe4a53310951ce72f6d78198c43ed.jpg)

![c7b7306bd04bdd79a497a391b2ef6291977bb86d7c5668b8be59c3804bf31749.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/c7b7306bd04bdd79a497a391b2ef6291977bb86d7c5668b8be59c3804bf31749.jpg)

![d41f39854c1279b700d8010bf9e39c720b6654dc0cbd92eb8adcaec3eff5d909.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/d41f39854c1279b700d8010bf9e39c720b6654dc0cbd92eb8adcaec3eff5d909.jpg)

![e41909636de20c0932ffe773f4d42e5054de159b838431aaa73f27e15c3ab178.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/e41909636de20c0932ffe773f4d42e5054de159b838431aaa73f27e15c3ab178.jpg)

![ea7ef55d82302a521912240366e351d3bb55951d0d185998f6de2c9dde9865b4.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/ea7ef55d82302a521912240366e351d3bb55951d0d185998f6de2c9dde9865b4.jpg)

![ef8062053eb17c5baf0b8a8b7e936780af1a52d0303697db01e95c960bffdb2b.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/ef8062053eb17c5baf0b8a8b7e936780af1a52d0303697db01e95c960bffdb2b.jpg)

![efe71c6ce8af2d20cac32786d9d7f44b366c878d262c12af27a3545b1b10669a.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/efe71c6ce8af2d20cac32786d9d7f44b366c878d262c12af27a3545b1b10669a.jpg)

![f922a11762efdd25a66adc0a79450c1a5dfba7b0d90753c2f08bba4b0ac0c625.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/images/f922a11762efdd25a66adc0a79450c1a5dfba7b0d90753c2f08bba4b0ac0c625.jpg)

### Tables

![0a8e9e20b17bbe6a14777dd7e0b1a5df41448d4dac529052b705e133adc56a4f.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/tables/0a8e9e20b17bbe6a14777dd7e0b1a5df41448d4dac529052b705e133adc56a4f.jpg)

![84c629591a833029561fa0104e470182f8d261fd8b4eac027cf31980546ddbca.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/tables/84c629591a833029561fa0104e470182f8d261fd8b4eac027cf31980546ddbca.jpg)

![a4aab0c2be438c817614063ffefd86403c36ca86fed598668c29bd4b81a7da5c.jpg](../icml_results/1473_NAVIGATE_%20Evaluating%20Visual-Guided%20Search%20Decision-Making%20on%20the%20Open%20Web/tables/a4aab0c2be438c817614063ffefd86403c36ca86fed598668c29bd4b81a7da5c.jpg)

## Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds


### Images

![1188e8561c131796c2071bfff28ec1cbbaf30078a43799b225bb6315a15d5610.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/1188e8561c131796c2071bfff28ec1cbbaf30078a43799b225bb6315a15d5610.jpg)

![41a8551e0522689ba104198f3e36250d5a236a86bb4f7441f07924642dc2507d.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/41a8551e0522689ba104198f3e36250d5a236a86bb4f7441f07924642dc2507d.jpg)

![4c216145ed1815a113ae327e99ce32039b4ace874ac715055c199df6e8938ce8.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/4c216145ed1815a113ae327e99ce32039b4ace874ac715055c199df6e8938ce8.jpg)

![53f332eb1ace0d982eda46bd13b2e72db61ce8086373a7ce8b7f37d67affa735.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/53f332eb1ace0d982eda46bd13b2e72db61ce8086373a7ce8b7f37d67affa735.jpg)

![9099354b3dba3a0bc7a0e6c90e223b0e2b185a3fdaa3b94c9f6e2e7291c56745.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/9099354b3dba3a0bc7a0e6c90e223b0e2b185a3fdaa3b94c9f6e2e7291c56745.jpg)

![962ab8f879b4a8d903d69855248f2ca1aa3ed739f8415dd2e5dd410d60b8568a.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/962ab8f879b4a8d903d69855248f2ca1aa3ed739f8415dd2e5dd410d60b8568a.jpg)

![ac44038d0accef48a277e1579d43c2a67e5f1f5ac8b4e25a0a4387e80404a131.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/ac44038d0accef48a277e1579d43c2a67e5f1f5ac8b4e25a0a4387e80404a131.jpg)

![f46c5f61e02e656238a63ed414d97fd586d609478e2902dc2c47688532733964.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/images/f46c5f61e02e656238a63ed414d97fd586d609478e2902dc2c47688532733964.jpg)

### Tables

![02e002c907b9498021bd56154af810312af04d7dd870d06279fa3f3643c49451.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/02e002c907b9498021bd56154af810312af04d7dd870d06279fa3f3643c49451.jpg)

![02fc1040bd99230f118c9d1d312c00330ac68d8f017704701d14a55cd5a34cdc.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/02fc1040bd99230f118c9d1d312c00330ac68d8f017704701d14a55cd5a34cdc.jpg)

![11a961ce5fe283c0f8b052529e6901f0ce07561a197205516a1ed4e4f139a54d.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/11a961ce5fe283c0f8b052529e6901f0ce07561a197205516a1ed4e4f139a54d.jpg)

![14d416b6003937d76ead861da12bb16ee70aa898640e583476c01c28a89a7847.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/14d416b6003937d76ead861da12bb16ee70aa898640e583476c01c28a89a7847.jpg)

![361b2c395451e361fa6d50597f38f63222e5d486a7f3e60e6f859b0957a3cf6f.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/361b2c395451e361fa6d50597f38f63222e5d486a7f3e60e6f859b0957a3cf6f.jpg)

![3e261692fbbfd3ad8621fc4768671d379f54466c4018c59a3a9141720feb1138.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/3e261692fbbfd3ad8621fc4768671d379f54466c4018c59a3a9141720feb1138.jpg)

![7787df3e83d3a65043a2e977522538fb38a7273a3a8b085514fc9bc6f837a803.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/7787df3e83d3a65043a2e977522538fb38a7273a3a8b085514fc9bc6f837a803.jpg)

![97ba39b53d98005a289610046443049c9f547dfaae078f73280d6325efdd4604.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/97ba39b53d98005a289610046443049c9f547dfaae078f73280d6325efdd4604.jpg)

![b96e3e9cccddc998dff7bcaf73709214c96fe8d9325e0364ce3acf402cc14126.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/b96e3e9cccddc998dff7bcaf73709214c96fe8d9325e0364ce3acf402cc14126.jpg)

![de886453f6b78029e9bdc0147b69e00344ee2697c78daf972154e947b91a5372.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/de886453f6b78029e9bdc0147b69e00344ee2697c78daf972154e947b91a5372.jpg)

![e3a1da4703ac784869631e6a3bf5fac904c244a93d5eadc89906fd5e03f4c08d.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/e3a1da4703ac784869631e6a3bf5fac904c244a93d5eadc89906fd5e03f4c08d.jpg)

![f908468f5bafa3269f53d4c2fbc2ccd63c1c1a71f9cad4e9d2649b22abadca6c.jpg](../icml_results/1474_Any3D-VLA_%20Enhancing%20VLA%20Robustness%20via%20Diverse%20Point%20Clouds/tables/f908468f5bafa3269f53d4c2fbc2ccd63c1c1a71f9cad4e9d2649b22abadca6c.jpg)

<a id="paper-1454"></a>

## Graph-Preference Learning: Debiasing Network-Sampled Human Feedback for Target Welfare Estimation


### Images

![06af5c2b79f2d84002bfeb3d23aa3508806880c82d4e0cb4864f63c546fdf1e7.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/images/06af5c2b79f2d84002bfeb3d23aa3508806880c82d4e0cb4864f63c546fdf1e7.jpg)

![3c1470b04f193487e83f598d5799d524ee27dbf06ae7844e0a8d26d4fbf12dd7.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/images/3c1470b04f193487e83f598d5799d524ee27dbf06ae7844e0a8d26d4fbf12dd7.jpg)

![40179bda122c7cadd7a3e1699ceaf36f66034ad0afccb472c27a42a70c34370b.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/images/40179bda122c7cadd7a3e1699ceaf36f66034ad0afccb472c27a42a70c34370b.jpg)

### Tables

![016fb83a3d349b3eb1f857b13526cc17b07d481af753832119bc2d837c6fdf23.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/016fb83a3d349b3eb1f857b13526cc17b07d481af753832119bc2d837c6fdf23.jpg)

![050b9589632422362338e9b929ca00de8afd3a906ca1a95e90b88f415a231e70.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/050b9589632422362338e9b929ca00de8afd3a906ca1a95e90b88f415a231e70.jpg)

![2b61250978f4cae1ccf2f5284aa92de49210b1d0c6367b0e03b3ffd720ab4fa9.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/2b61250978f4cae1ccf2f5284aa92de49210b1d0c6367b0e03b3ffd720ab4fa9.jpg)

![2fe51be742a37e1a83907e2c7d994fa1fe151606cd7a20253440ec0d2fd420fe.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/2fe51be742a37e1a83907e2c7d994fa1fe151606cd7a20253440ec0d2fd420fe.jpg)

![4917802cb22efba62224d3d13edfee6b1455f9c0ef8cba33979c208252c55456.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/4917802cb22efba62224d3d13edfee6b1455f9c0ef8cba33979c208252c55456.jpg)

![5beba80e5898e621373ca82a150d3a31c792159b4f016056eee58b7dc2df6fa6.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/5beba80e5898e621373ca82a150d3a31c792159b4f016056eee58b7dc2df6fa6.jpg)

![736ef90481f18f34972bdaf1a7ad5891a658181c0309eff45abc3a58a789c6db.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/736ef90481f18f34972bdaf1a7ad5891a658181c0309eff45abc3a58a789c6db.jpg)

![9108481beb43f557197c96fac32d665fd7dc1c4afa5c0f8a99da9894b07e45ae.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/9108481beb43f557197c96fac32d665fd7dc1c4afa5c0f8a99da9894b07e45ae.jpg)

![bf70834ebbee2bd51473496cb047a9a00765c875f1ccc382c3f35098ecd9319e.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/bf70834ebbee2bd51473496cb047a9a00765c875f1ccc382c3f35098ecd9319e.jpg)

![e66ccca10b008bf7c6057d91570ce691988892c07c66e6a1164453501a7c8657.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/e66ccca10b008bf7c6057d91570ce691988892c07c66e6a1164453501a7c8657.jpg)

![f5dfa07d0d6b5bd33796e97618dbd15cd2a86e382d31396371b0f995ff54995e.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/f5dfa07d0d6b5bd33796e97618dbd15cd2a86e382d31396371b0f995ff54995e.jpg)

![f6dc0eba9ad9f25917180c14f18aa1b20e74c159ec9041288b2b4266c642cc20.jpg](../icml_results/1475_Graph-Preference%20Learning_%20Debiasing%20Network-Sampled%20Human%20Feedback%20for%20Target%20Welfare%20Estimation/tables/f6dc0eba9ad9f25917180c14f18aa1b20e74c159ec9041288b2b4266c642cc20.jpg)

<a id="paper-1455"></a>

## ZeroDiff: Zero-Shot Time Series Reconstruction via Informed-Prior Diffusion


### Images

![0753b2eb2e94a53a714ac2bcd487ed1990a1e5c7a1139be7b70c4c33a62cd7c6.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/images/0753b2eb2e94a53a714ac2bcd487ed1990a1e5c7a1139be7b70c4c33a62cd7c6.jpg)

![0de18c800ba362e7ddea54517f241870776eae3ac6b2c570bd7287e5e1263411.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/images/0de18c800ba362e7ddea54517f241870776eae3ac6b2c570bd7287e5e1263411.jpg)

![1dc8b0262a827f18fa63f96e86b290bb112bc55e99c88e512efc9c89cfbc4f1e.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/images/1dc8b0262a827f18fa63f96e86b290bb112bc55e99c88e512efc9c89cfbc4f1e.jpg)

![308acf8d1a590a01be27fc947b68944ea171102c842272d1f3355cdcd77488aa.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/images/308acf8d1a590a01be27fc947b68944ea171102c842272d1f3355cdcd77488aa.jpg)

![7a5bb2781fff1ed39eecf3863b7f5c251f300710e7d0ea78458c60329bf9bbf0.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/images/7a5bb2781fff1ed39eecf3863b7f5c251f300710e7d0ea78458c60329bf9bbf0.jpg)

### Tables

![0c7cc9992e4ed60d9f0c457a8f26d17571d12fad84660383ad228d3617253a3a.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/0c7cc9992e4ed60d9f0c457a8f26d17571d12fad84660383ad228d3617253a3a.jpg)

![1a579bc71c446464c6756691325ff725be99a6047439948efbe4b557114ba079.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/1a579bc71c446464c6756691325ff725be99a6047439948efbe4b557114ba079.jpg)

![3e5e58b51d4581e4be456f08437e09c08a9d5ec0a5db135fe53a3e3eab4aeacb.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/3e5e58b51d4581e4be456f08437e09c08a9d5ec0a5db135fe53a3e3eab4aeacb.jpg)

![4632efe875ea87321b9bc346af3fe5ba7b1587e94ad07419850feeca75839804.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/4632efe875ea87321b9bc346af3fe5ba7b1587e94ad07419850feeca75839804.jpg)

![74c66a3aaee073c26d9fa76ccc88ebaa1184d62844e5851ea9be079e70c5a032.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/74c66a3aaee073c26d9fa76ccc88ebaa1184d62844e5851ea9be079e70c5a032.jpg)

![92a6bee02d588e8f8a976ae755454cd01c94164f933bff3c025f4e3bcae46864.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/92a6bee02d588e8f8a976ae755454cd01c94164f933bff3c025f4e3bcae46864.jpg)

![a074d191325b0900c9b68372bc227b02d2758c027583a3dab0c5a9c2b5a26b9c.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/a074d191325b0900c9b68372bc227b02d2758c027583a3dab0c5a9c2b5a26b9c.jpg)

![bb375727017dbfaa3bcba45e38b3d25b863fad564f97c6da2c1b11b40978d410.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/bb375727017dbfaa3bcba45e38b3d25b863fad564f97c6da2c1b11b40978d410.jpg)

![bd14f9a9b69ac6bb90c73ba14edd87a8b9eafce6e084352922c70ce3255e8722.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/bd14f9a9b69ac6bb90c73ba14edd87a8b9eafce6e084352922c70ce3255e8722.jpg)

![ec26e4486da3409cee05acbb14d8a8494b05811bf0538311ed47c7031ad8b1b0.jpg](../icml_results/1476_ZeroDiff_%20Zero-Shot%20Time%20Series%20Reconstruction%20via%20Informed-Prior%20Diffusion/tables/ec26e4486da3409cee05acbb14d8a8494b05811bf0538311ed47c7031ad8b1b0.jpg)

<a id="paper-1456"></a>

## Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation


### Images

![0d81e493b71e8cb27a8f7bceec1b57389f534437f5b087ba98e02590ba3764e5.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/images/0d81e493b71e8cb27a8f7bceec1b57389f534437f5b087ba98e02590ba3764e5.jpg)

![128fce2fbe6a2a68049e616c6728376ffc3b3c30fafc0293e42a11796ace341d.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/images/128fce2fbe6a2a68049e616c6728376ffc3b3c30fafc0293e42a11796ace341d.jpg)

![38e8ba26477ee874ca6151b407edbd3585a2096a722e7a29c0189dc30c2b1cf7.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/images/38e8ba26477ee874ca6151b407edbd3585a2096a722e7a29c0189dc30c2b1cf7.jpg)

![7588e484e24bab87d9182de0c4adc86173c35f0f5f58e2e6bc6b2ad910893b68.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/images/7588e484e24bab87d9182de0c4adc86173c35f0f5f58e2e6bc6b2ad910893b68.jpg)

![acb757e59658db4b1f790e99553e38e70e3ad08029013ee5f337c12a65f6a3b7.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/images/acb757e59658db4b1f790e99553e38e70e3ad08029013ee5f337c12a65f6a3b7.jpg)

### Tables

![1ee912423c01d1713871d4dc1da551590bd08468a80d719c6f1097bf61fd27f5.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/tables/1ee912423c01d1713871d4dc1da551590bd08468a80d719c6f1097bf61fd27f5.jpg)

![63dc9dad0a65104c49b853f6b553cd7676111809571d1e0556ea68051d876d00.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/tables/63dc9dad0a65104c49b853f6b553cd7676111809571d1e0556ea68051d876d00.jpg)

![e6e9d63b010654f76895bcd070b4da1074989bd0bb026a6017279d5665b14897.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/tables/e6e9d63b010654f76895bcd070b4da1074989bd0bb026a6017279d5665b14897.jpg)

![fa0786a5b42a324624b43d1d0c05e7956ebaa3513ffc3e112f1650d6bc6ada9c.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/tables/fa0786a5b42a324624b43d1d0c05e7956ebaa3513ffc3e112f1650d6bc6ada9c.jpg)

![fb12dab7e724e6159d4c3d6ec263657ff94d778200757b144f028ad369eb7f2e.jpg](../icml_results/1477_Uncovering%20the%20Gradient%20Geometry%20of%20Long%20CoT_%20A%20Spectral-guided%20Approach%20to%20Reasoning%20Distillation/tables/fb12dab7e724e6159d4c3d6ec263657ff94d778200757b144f028ad369eb7f2e.jpg)

<a id="paper-1457"></a>

## De4D-SLAM: Gradient-Isolated Static-Dynamic Decoupling for Monocular SLAM in Dynamic Environments


### Images

![04ab61e5f8ab22740c20e2badd5fcb32f25dc8bc325816cbcfa09a49ef891eb7.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/04ab61e5f8ab22740c20e2badd5fcb32f25dc8bc325816cbcfa09a49ef891eb7.jpg)

![04fc4018633b28bcafd5b7016d654f146ccd741f4a90121e1c60448656ae737f.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/04fc4018633b28bcafd5b7016d654f146ccd741f4a90121e1c60448656ae737f.jpg)

![1d115d26ad06f3e635b72d9941031fbaac193136ad9980253ca505c2508e84b8.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/1d115d26ad06f3e635b72d9941031fbaac193136ad9980253ca505c2508e84b8.jpg)

![870e98c4387c8b21ee75cc6fc7d1a353c50074da8b46e2db5f000b038093e34c.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/870e98c4387c8b21ee75cc6fc7d1a353c50074da8b46e2db5f000b038093e34c.jpg)

![ab91eab35b5b6e561d1e8abf7ae7f8dc06921d4e914b00dc6cd65b31a4ebe136.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/ab91eab35b5b6e561d1e8abf7ae7f8dc06921d4e914b00dc6cd65b31a4ebe136.jpg)

![b619aeb89b59ebbd566aa028c46ba8f2d25b2db6d26c803a52a27d66b07830a2.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/b619aeb89b59ebbd566aa028c46ba8f2d25b2db6d26c803a52a27d66b07830a2.jpg)

![cbd819cc7b9b8b11bc62950848caeb15d2aeaca50ab120272113310ce639953e.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/cbd819cc7b9b8b11bc62950848caeb15d2aeaca50ab120272113310ce639953e.jpg)

![dda9a2fc220b643882040781e36329c6121ac8690d45a836e44b7cfe1ba86b56.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/dda9a2fc220b643882040781e36329c6121ac8690d45a836e44b7cfe1ba86b56.jpg)

![f9ba88ea7f5f7a8c1cad3f3523549ccfbed334854877da26daf539b5c9ea176f.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/images/f9ba88ea7f5f7a8c1cad3f3523549ccfbed334854877da26daf539b5c9ea176f.jpg)

### Tables

![0d4575e4f9515ea0e4fc816cdb9317983ff1e002315ce80059f2d9a54d2432f5.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/0d4575e4f9515ea0e4fc816cdb9317983ff1e002315ce80059f2d9a54d2432f5.jpg)

![168d7c8f552c956738f469ccfdbb73764b9dd62456285fc23501e536872af046.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/168d7c8f552c956738f469ccfdbb73764b9dd62456285fc23501e536872af046.jpg)

![1fc8ea15119302970108e3209f973ecf5c0aeab642d281bf1396bc60af8d5f65.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/1fc8ea15119302970108e3209f973ecf5c0aeab642d281bf1396bc60af8d5f65.jpg)

![447dd003a7c7b2b2137625ee8872808f2fcca7ee53dc5a07b6942a5d3a31c37c.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/447dd003a7c7b2b2137625ee8872808f2fcca7ee53dc5a07b6942a5d3a31c37c.jpg)

![45ff4f27bb3fe0c189c5b101dfb5798e2ffcb20385ee07ed32adfc30cda62b15.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/45ff4f27bb3fe0c189c5b101dfb5798e2ffcb20385ee07ed32adfc30cda62b15.jpg)

![4e51c855c74586a6ce9f5b890f4c7990abf3dc80b797e6fca3209db17219e520.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/4e51c855c74586a6ce9f5b890f4c7990abf3dc80b797e6fca3209db17219e520.jpg)

![6688016c066dfac627e7ddb587b9e2b3edf9ccad2849df85d079919b7d9f618e.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/6688016c066dfac627e7ddb587b9e2b3edf9ccad2849df85d079919b7d9f618e.jpg)

![6bd8d32dff130f255f9c13a4977f9919e4fe9c003be5680e449bbe0de085181b.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/6bd8d32dff130f255f9c13a4977f9919e4fe9c003be5680e449bbe0de085181b.jpg)

![7598951d813995b9b0701e1bf6c7bdd81b6ce19309dbb0d5ac34b1553d1a04dc.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/7598951d813995b9b0701e1bf6c7bdd81b6ce19309dbb0d5ac34b1553d1a04dc.jpg)

![8527f3bb4f76dae0109b333e205ab1a0a3059894f1c585b0ec5e8d75daba4aaa.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/8527f3bb4f76dae0109b333e205ab1a0a3059894f1c585b0ec5e8d75daba4aaa.jpg)

![93aed7b67e5497c8448513b144aa46272a6ca1d672c4cabcce8c4614428f2935.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/93aed7b67e5497c8448513b144aa46272a6ca1d672c4cabcce8c4614428f2935.jpg)

![a2d6205dfb2ddbb8fadd8d3e2faf30631d1c49397069296e5c66a653cc2fbd74.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/a2d6205dfb2ddbb8fadd8d3e2faf30631d1c49397069296e5c66a653cc2fbd74.jpg)

![a462f26e071926dc12b7afdedbf291bae13d8d34f733c7766aad1af04620cb87.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/a462f26e071926dc12b7afdedbf291bae13d8d34f733c7766aad1af04620cb87.jpg)

![b281dfd0b4e9cd3984591da92e384ace560ef889d6654bad335524093a4b54e0.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/b281dfd0b4e9cd3984591da92e384ace560ef889d6654bad335524093a4b54e0.jpg)

![c25c6e0efd1fef88a1729b2fd00f4f8e19b5c6ee162755b7440f35f1c27b758e.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/c25c6e0efd1fef88a1729b2fd00f4f8e19b5c6ee162755b7440f35f1c27b758e.jpg)

![cf6018459623bd670b866cc1a587250e01552ccd317afad18ca4de3bdec1847e.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/cf6018459623bd670b866cc1a587250e01552ccd317afad18ca4de3bdec1847e.jpg)

![e017b3751448afcb3f3867ba3bc9a74997cd251022648a58df22219f40fb6f40.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/e017b3751448afcb3f3867ba3bc9a74997cd251022648a58df22219f40fb6f40.jpg)

![e37048be3654a2e1254cc1ca7bafc5feb6ca864a4a089d1aa8f56bf9e374f7b8.jpg](../icml_results/1478_De4D-SLAM_%20Gradient-Isolated%20Static-Dynamic%20Decoupling%20for%20Monocular%20SLAM%20in%20Dynamic%20Environments/tables/e37048be3654a2e1254cc1ca7bafc5feb6ca864a4a089d1aa8f56bf9e374f7b8.jpg)

<a id="paper-1458"></a>

## Rethinking Time-Series Imputation as Conditional Inference along Temporal Evolution


### Images

![2084898bf2a63e0a1f0c51ba162bd499991b844c9517ea4b95980fe0b637bd3a.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/2084898bf2a63e0a1f0c51ba162bd499991b844c9517ea4b95980fe0b637bd3a.jpg)

![7428c4d45d5cec0c7d98f391487986b1b05e8412b5567c3a64cb5cf125c4786a.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/7428c4d45d5cec0c7d98f391487986b1b05e8412b5567c3a64cb5cf125c4786a.jpg)

![a5fb83fb28b02b3efe41227e6fad26d4503f2b3fc471fdd438eb7eecfb7f579a.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/a5fb83fb28b02b3efe41227e6fad26d4503f2b3fc471fdd438eb7eecfb7f579a.jpg)

![be1bffb3200c939ad700f08bf9cc5be328ea7701540a1765cc825a2ec59a2b5f.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/be1bffb3200c939ad700f08bf9cc5be328ea7701540a1765cc825a2ec59a2b5f.jpg)

![d23c6743db69b79df5cbcfdfd215dc23b914c95e6af44b4cbc99f69c0afc4e7d.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/d23c6743db69b79df5cbcfdfd215dc23b914c95e6af44b4cbc99f69c0afc4e7d.jpg)

![efd39ce6925db16d7bb2dd7476507ce57246d78e09dbd516e5978ae0381209a9.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/efd39ce6925db16d7bb2dd7476507ce57246d78e09dbd516e5978ae0381209a9.jpg)

![fa69ca6a9989b7b430407ecc517e6f70510a28be312ca7955719afd8a1e69e58.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/images/fa69ca6a9989b7b430407ecc517e6f70510a28be312ca7955719afd8a1e69e58.jpg)

### Tables

![21ce9eaf41a2fdf5bfc9372eb14c00cc5f35dd0282e9dc2f1f9b05fdb5bebbb6.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/21ce9eaf41a2fdf5bfc9372eb14c00cc5f35dd0282e9dc2f1f9b05fdb5bebbb6.jpg)

![2a71e33529b9e0a0fecdaf64c2ba4547895a97a118c018da79d3ce0cc468a13d.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/2a71e33529b9e0a0fecdaf64c2ba4547895a97a118c018da79d3ce0cc468a13d.jpg)

![2d166e1bb31e07415726b8ebb337afdb44869d9eb861c709656fd41350db7f2a.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/2d166e1bb31e07415726b8ebb337afdb44869d9eb861c709656fd41350db7f2a.jpg)

![2ebfd8400b03610cbe991ece92c80eb6cdf1766414b5552f851a535e92715afe.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/2ebfd8400b03610cbe991ece92c80eb6cdf1766414b5552f851a535e92715afe.jpg)

![3baebd06dd64904388eff81b1adf19ea798721f6e3367abb6100e5002b8cb580.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/3baebd06dd64904388eff81b1adf19ea798721f6e3367abb6100e5002b8cb580.jpg)

![6ee8c15aa643ef8ab0731140804a5b84e164be61d55596f0a648b126f549b9cf.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/6ee8c15aa643ef8ab0731140804a5b84e164be61d55596f0a648b126f549b9cf.jpg)

![81ebf8e8dc76340dcd3a230ad2fa89b678599cb1028ee8521196bb5df4bdf55c.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/81ebf8e8dc76340dcd3a230ad2fa89b678599cb1028ee8521196bb5df4bdf55c.jpg)

![8a32dfeb15c8b4e4a0b9eaf52d13f55d9f71a23998993307ee40e56966e3199e.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/8a32dfeb15c8b4e4a0b9eaf52d13f55d9f71a23998993307ee40e56966e3199e.jpg)

![a06de6f75cf40fa8c10ef1f25dd5c3a8d6415c0e3f3c36ebf68dc57cabe49f64.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/a06de6f75cf40fa8c10ef1f25dd5c3a8d6415c0e3f3c36ebf68dc57cabe49f64.jpg)

![b045cdc0b351e7186fb5fe71cc31502eff7c2df6cd90686145b55fd06975de83.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/b045cdc0b351e7186fb5fe71cc31502eff7c2df6cd90686145b55fd06975de83.jpg)

![b4c341bf61c6e39b5d1401a80fe2b22b7fb0e4d193ef7c0f452a1674f545cecc.jpg](../icml_results/1479_Rethinking%20Time-Series%20Imputation%20as%20Conditional%20Inference%20along%20Temporal%20Evolution/tables/b4c341bf61c6e39b5d1401a80fe2b22b7fb0e4d193ef7c0f452a1674f545cecc.jpg)

<a id="paper-1459"></a>

## TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning


### Images

![01ed9fd39060559e75ac77c4fbc52a6df7a693d3acff7c96c032a23bfb04e5a6.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/01ed9fd39060559e75ac77c4fbc52a6df7a693d3acff7c96c032a23bfb04e5a6.jpg)

![143ea97df024de655614febc6b09e49ee9981d2f1cf9bbdcbf03b6795abc0319.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/143ea97df024de655614febc6b09e49ee9981d2f1cf9bbdcbf03b6795abc0319.jpg)

![24870eb00f04805fdbd5b6377a3b9812e6060a4c7f3414bfd807f63d1ac3dae6.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/24870eb00f04805fdbd5b6377a3b9812e6060a4c7f3414bfd807f63d1ac3dae6.jpg)

![2d8e863ae72f772871f0472605de35d9eb3d97b28c0d259953777edc876408e8.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/2d8e863ae72f772871f0472605de35d9eb3d97b28c0d259953777edc876408e8.jpg)

![4cf6f352c09539cd475ab15548d0ff8c9c4851ef088c86082fe69e1eddca04db.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/4cf6f352c09539cd475ab15548d0ff8c9c4851ef088c86082fe69e1eddca04db.jpg)

![6b1cbd6148168d6dbc9780aa61fa7e705477e5d19d3344a4f601fee6d6570561.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/6b1cbd6148168d6dbc9780aa61fa7e705477e5d19d3344a4f601fee6d6570561.jpg)

![829ec06ed945577262a75d2b91165b5b780ed995048d14431e911cc1bfe94add.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/829ec06ed945577262a75d2b91165b5b780ed995048d14431e911cc1bfe94add.jpg)

![8a1e068fdce766d0121c690e282a399f9253df5bcb24b0e197d58b8f3e8b8d65.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/8a1e068fdce766d0121c690e282a399f9253df5bcb24b0e197d58b8f3e8b8d65.jpg)

![daa77cbdfaf7c42a55dbce0273e464f13d3e6d560b9f35dce7a51ee66c7b5573.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/daa77cbdfaf7c42a55dbce0273e464f13d3e6d560b9f35dce7a51ee66c7b5573.jpg)

![e4e17837c4e09b58f4b1b5b0f65725a3ea5a37ea3632fd7ff166ea47dae89e1b.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/images/e4e17837c4e09b58f4b1b5b0f65725a3ea5a37ea3632fd7ff166ea47dae89e1b.jpg)

### Tables

![22b72f96426afb72c3d3c2bd46609db610e09f9289f95d1945021a9476a07d47.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/22b72f96426afb72c3d3c2bd46609db610e09f9289f95d1945021a9476a07d47.jpg)

![269deb9aa73e96b5520f12902bb91eb6d651ea220a45f27bf50848b4037d6c06.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/269deb9aa73e96b5520f12902bb91eb6d651ea220a45f27bf50848b4037d6c06.jpg)

![3160660b9aecaca5c500f296bc73e4d5187832138a3b311dd38c73ea5bab8de6.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/3160660b9aecaca5c500f296bc73e4d5187832138a3b311dd38c73ea5bab8de6.jpg)

![51ab3b8455bb717223baed8a7218c4ada39e8926954b4c5ae86c1b945d8c285e.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/51ab3b8455bb717223baed8a7218c4ada39e8926954b4c5ae86c1b945d8c285e.jpg)

![645c18223b773052679e373fc88de442661121584a84d10a540b1c43ac545e7e.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/645c18223b773052679e373fc88de442661121584a84d10a540b1c43ac545e7e.jpg)

![7341a95733fe23d838c66c60306e5d7c0e9b70656a5e1b70f5d6b55754b2ae56.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/7341a95733fe23d838c66c60306e5d7c0e9b70656a5e1b70f5d6b55754b2ae56.jpg)

![769d8c68bd74f16aaaf1f5ddb99fa6a9d363273b50cb86da6d6f8d26b973c646.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/769d8c68bd74f16aaaf1f5ddb99fa6a9d363273b50cb86da6d6f8d26b973c646.jpg)

![820d16e3366d584c2ce00cc5175b5f781af3fd0a0bc9d45808d8c2b3953935b1.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/820d16e3366d584c2ce00cc5175b5f781af3fd0a0bc9d45808d8c2b3953935b1.jpg)

![85d056b3c6e4e35188f24428bc98d9cb379108e78d1d0cba0c4040590500380f.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/85d056b3c6e4e35188f24428bc98d9cb379108e78d1d0cba0c4040590500380f.jpg)

![86531e3481b64a96319b5c87a63e1964a3415de0b6f89e0c73ca0b78557add67.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/86531e3481b64a96319b5c87a63e1964a3415de0b6f89e0c73ca0b78557add67.jpg)

![95fd7ccbc41fecc2402ed730b3b59cf7ecb5a9f9cdf8db4f83c78b561b969afa.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/95fd7ccbc41fecc2402ed730b3b59cf7ecb5a9f9cdf8db4f83c78b561b969afa.jpg)

![a36f84edeeecfcac2111df16c1c9ed17dd5623561fcfc09fd1e037234448a140.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/a36f84edeeecfcac2111df16c1c9ed17dd5623561fcfc09fd1e037234448a140.jpg)

![a85bae5754fb24488a14f41e34530e29ada4e052651f83c883ef00e058263bba.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/a85bae5754fb24488a14f41e34530e29ada4e052651f83c883ef00e058263bba.jpg)

![cb504966f79f9c3f2450a8ef24df2764c9fc3d70de35e9ef33bae2f171d3356a.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/cb504966f79f9c3f2450a8ef24df2764c9fc3d70de35e9ef33bae2f171d3356a.jpg)

![d40bd747a2de64e84f7fdd5073e3eb4a0896a6072a5855543464bc29d6f71cc7.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/d40bd747a2de64e84f7fdd5073e3eb4a0896a6072a5855543464bc29d6f71cc7.jpg)

![e2a939ac74535051ec3160064896364994fa7fa56584659d23e64ff19f4fa1fc.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/e2a939ac74535051ec3160064896364994fa7fa56584659d23e64ff19f4fa1fc.jpg)

![e659e2a39c2f74478a670cc6aec283901e4e7ec8eea8c9dabd2224015e515819.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/e659e2a39c2f74478a670cc6aec283901e4e7ec8eea8c9dabd2224015e515819.jpg)

![e879d8f1421dc950af16c0061d597974a4c86623b947e6cf49f11ff5aadd56d0.jpg](../icml_results/1480_TRACE_%20Trajectory%20Recovery%20for%20Continuous%20Mechanism%20Evolution%20in%20Causal%20Representation%20Learning/tables/e879d8f1421dc950af16c0061d597974a4c86623b947e6cf49f11ff5aadd56d0.jpg)

<a id="paper-1460"></a>

## Navigating the Pareto Frontier of Alignment: Spectrum-Adaptive Fine-Tuning for LLMs


### Images

![02cf1950b0a19c8c31ee5b54e84c85aefb43e3fea65afff5fbabc3dffa14c0c3.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/images/02cf1950b0a19c8c31ee5b54e84c85aefb43e3fea65afff5fbabc3dffa14c0c3.jpg)

![7cdbcdb98c6e5f7dd428493a318218374b3474fafc0fdaefa2a3ccd39fe1ae82.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/images/7cdbcdb98c6e5f7dd428493a318218374b3474fafc0fdaefa2a3ccd39fe1ae82.jpg)

![d53b1694ae323dfbbfaa8e9abb0aa5f22c3eb7969b964ccb63ac23248b0a02ef.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/images/d53b1694ae323dfbbfaa8e9abb0aa5f22c3eb7969b964ccb63ac23248b0a02ef.jpg)

### Tables

![2449269eaa082a3ca7cb21657068d1876eed86e7fb8d48c48e8863e238db60d0.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/2449269eaa082a3ca7cb21657068d1876eed86e7fb8d48c48e8863e238db60d0.jpg)

![2adb98114f559e013b041f2e56443dba12c9c86f0b30b4c1d7fddff56f98af54.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/2adb98114f559e013b041f2e56443dba12c9c86f0b30b4c1d7fddff56f98af54.jpg)

![449c938809b0ac8792437c4b47e726e500f4c00f24eb67dab1d4b0075cd5d107.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/449c938809b0ac8792437c4b47e726e500f4c00f24eb67dab1d4b0075cd5d107.jpg)

![4d87b0cd6a8cc1488081b7c96777bf04fde1c4453a7c74f3c6b2de22c5719f84.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/4d87b0cd6a8cc1488081b7c96777bf04fde1c4453a7c74f3c6b2de22c5719f84.jpg)

![61dbba7d1c408c7d19b8673f9e4fd7b0ccbeed14ba8de99614223b54c520ff37.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/61dbba7d1c408c7d19b8673f9e4fd7b0ccbeed14ba8de99614223b54c520ff37.jpg)

![73442581443601895db66d4cd4c099a717118bb123cb640c77746b7e56916590.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/73442581443601895db66d4cd4c099a717118bb123cb640c77746b7e56916590.jpg)

![86e26ceb4f8ea82c2e7263d9f7ed74c166bd2095e307f811786d0b5d08c64413.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/86e26ceb4f8ea82c2e7263d9f7ed74c166bd2095e307f811786d0b5d08c64413.jpg)

![886027a115e48d7faeb227c95a001257b89c241e390e2f700e41da96bfe4b012.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/886027a115e48d7faeb227c95a001257b89c241e390e2f700e41da96bfe4b012.jpg)

![8abe2b2804c3a0be5cffc3e4a816e3990b8eb654f6575aaae677d90333872522.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/8abe2b2804c3a0be5cffc3e4a816e3990b8eb654f6575aaae677d90333872522.jpg)

![9e3379d28478f80cd0bf0725a237c00dde8bd6329682514cd0724c5881ed82e6.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/9e3379d28478f80cd0bf0725a237c00dde8bd6329682514cd0724c5881ed82e6.jpg)

![9f348390c6d18d9d02ccb6285610e2b8d689f744e8a3bcb49752fe545c71879a.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/9f348390c6d18d9d02ccb6285610e2b8d689f744e8a3bcb49752fe545c71879a.jpg)

![c74ad0ac81c68385eefc63c7b2e719def5e6cc728de5649e9396c4db1041fd25.jpg](../icml_results/1481_Navigating%20the%20Pareto%20Frontier%20of%20Alignment_%20Spectrum-Adaptive%20Fine-Tuning%20for%20LLMs/tables/c74ad0ac81c68385eefc63c7b2e719def5e6cc728de5649e9396c4db1041fd25.jpg)

<a id="paper-1461"></a>

## NExT-Guard: Streaming Safeguard without Token-Level Labels


### Images

![06d3f01674f8ae5517f236d953e77ebe949c89c19899ade429274164235e2e50.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/images/06d3f01674f8ae5517f236d953e77ebe949c89c19899ade429274164235e2e50.jpg)

![3208fa734f68b530d33616efc6f234160d492b73b7dfbd58c2b443cda88a5f01.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/images/3208fa734f68b530d33616efc6f234160d492b73b7dfbd58c2b443cda88a5f01.jpg)

![39ac0f90238c8b620c51058edb7ebe87b5e6f731617f4403aa172cc6e051ad1e.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/images/39ac0f90238c8b620c51058edb7ebe87b5e6f731617f4403aa172cc6e051ad1e.jpg)

![79ccb87ab4d1ab8db27ae85b39671abaef2f8eaa535380fd28a642cfbc6204b1.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/images/79ccb87ab4d1ab8db27ae85b39671abaef2f8eaa535380fd28a642cfbc6204b1.jpg)

![7bb66372f72cfdbaad0d59f011ef3fca58cc280af61e013e36c39be9c62bc031.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/images/7bb66372f72cfdbaad0d59f011ef3fca58cc280af61e013e36c39be9c62bc031.jpg)

![e03c180171ccf9d2baaafc27f13a2cf75c95f30ef5a438c837b71b2e270be618.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/images/e03c180171ccf9d2baaafc27f13a2cf75c95f30ef5a438c837b71b2e270be618.jpg)

### Tables

![4041262ef3d97c0be0b8bf0e7a266ec8e7453f8815eb52fa8a726463da537e6a.jpg](../icml_results/1482_NExT-Guard_%20Streaming%20Safeguard%20without%20Token-Level%20Labels/tables/4041262ef3d97c0be0b8bf0e7a266ec8e7453f8815eb52fa8a726463da537e6a.jpg)

<a id="paper-1462"></a>

## Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness


### Images

![1ae2e1447524ffef5100459ef5a69b2cd186367a1803be86f7c819a0c6199d51.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/1ae2e1447524ffef5100459ef5a69b2cd186367a1803be86f7c819a0c6199d51.jpg)

![217ff209bb18abee9c892070b00cf4660c09254ea3b1e7fb374c822d1fa344e0.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/217ff209bb18abee9c892070b00cf4660c09254ea3b1e7fb374c822d1fa344e0.jpg)

![44ee557e221fa60fd1f0d540f4acd927922e0c8a0ad243cda414819690e8bb2a.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/44ee557e221fa60fd1f0d540f4acd927922e0c8a0ad243cda414819690e8bb2a.jpg)

![452f250fcdf2aeb0c6a3c9153f3749f0379330dbdf7d899d25cffa677e8a7262.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/452f250fcdf2aeb0c6a3c9153f3749f0379330dbdf7d899d25cffa677e8a7262.jpg)

![54de83dfdb02aaefa019196ac8e97c0d5a85c6a93efb94e6af442c489fa6a777.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/54de83dfdb02aaefa019196ac8e97c0d5a85c6a93efb94e6af442c489fa6a777.jpg)

![af1e3e7631a031f60b9f6b965dda80bc5abf62c84025e50e793ded61127e3c22.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/af1e3e7631a031f60b9f6b965dda80bc5abf62c84025e50e793ded61127e3c22.jpg)

![eaef8dac654cc910c362074cc9b41daf78097ffec8be2914c3867c2e3c51462b.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/images/eaef8dac654cc910c362074cc9b41daf78097ffec8be2914c3867c2e3c51462b.jpg)

### Tables

![443c755003d7e445bf17531beb248ce0bcaa9762a086cbbb0dd1ff6060942e1b.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/tables/443c755003d7e445bf17531beb248ce0bcaa9762a086cbbb0dd1ff6060942e1b.jpg)

![59d7963c61086da6f17177b0085e00eab014a343a50588fd005cd720dda2e28d.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/tables/59d7963c61086da6f17177b0085e00eab014a343a50588fd005cd720dda2e28d.jpg)

![b3c8e1fd4e7a03f71b78bf79f851b9452e7e91641a82cd2e49a021bad3cdb0ac.jpg](../icml_results/1483_Immuno-VLM_%20Immunizing%20Large%20Vision-Language%20Models%20via%20Generative%20Semantic%20Antibodies%20for%20Open-Worl/tables/b3c8e1fd4e7a03f71b78bf79f851b9452e7e91641a82cd2e49a021bad3cdb0ac.jpg)

<a id="paper-1463"></a>

## SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling


### Images

![736459a8ee0cfd92dda000a2ee9f9895c357ffef4cb420dfb09e885ad6f765f2.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/images/736459a8ee0cfd92dda000a2ee9f9895c357ffef4cb420dfb09e885ad6f765f2.jpg)

![858807f06e3134b7c77070c5e440b1f065713d6037fa2b4b3911d392681b31f8.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/images/858807f06e3134b7c77070c5e440b1f065713d6037fa2b4b3911d392681b31f8.jpg)

![8c685cbae9535f8ffd769665b69587765505c066d56fdf5b4d82f5a6400797f8.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/images/8c685cbae9535f8ffd769665b69587765505c066d56fdf5b4d82f5a6400797f8.jpg)

![9385a51d742e262ed92e6e1492e4700b253b761e16962c4888bb3323b624206b.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/images/9385a51d742e262ed92e6e1492e4700b253b761e16962c4888bb3323b624206b.jpg)

![97048b2fc14aac825a83da406fd32c50dc460251b3a0ee1f6eb1efa0dc62e629.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/images/97048b2fc14aac825a83da406fd32c50dc460251b3a0ee1f6eb1efa0dc62e629.jpg)

### Tables

![29ca94b73e8d872a08b003449af3d86618af0c75ac45895981402cb5af2e4219.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/29ca94b73e8d872a08b003449af3d86618af0c75ac45895981402cb5af2e4219.jpg)

![3e9c926140bf6b13bc55e43e8f629e4e1369e9f02789d55cb30f401c16455cb8.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/3e9c926140bf6b13bc55e43e8f629e4e1369e9f02789d55cb30f401c16455cb8.jpg)

![4542fd03d669c7eccb87a01835080e728d4f513eb32829aa5e713216cd687e2a.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/4542fd03d669c7eccb87a01835080e728d4f513eb32829aa5e713216cd687e2a.jpg)

![4c19823495d924c4534374f32cc9031fd62c5a0241b1fd05d7f40fb3e3c3bfc8.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/4c19823495d924c4534374f32cc9031fd62c5a0241b1fd05d7f40fb3e3c3bfc8.jpg)

![7c7d72ec6b016351f01512f228b8bb112faaca1d5cfabdea6e798743692b54a0.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/7c7d72ec6b016351f01512f228b8bb112faaca1d5cfabdea6e798743692b54a0.jpg)

![80f9ad275f0331e4437446e19d8a4212d5d0af5e1afe667cef3acca778b6d075.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/80f9ad275f0331e4437446e19d8a4212d5d0af5e1afe667cef3acca778b6d075.jpg)

![9a894a23028df56c1f80d5f087ae6b23da8f156a7232a7eabd93cff323bd1980.jpg](../icml_results/1484_SLAP_%20The%20Semantic%20Least%20Action%20Principle%20for%20Variational%20Video-Language%20Modeling/tables/9a894a23028df56c1f80d5f087ae6b23da8f156a7232a7eabd93cff323bd1980.jpg)

<a id="paper-1464"></a>

## Bridging On-Device and Cloud LLMs for Collaborative Reasoning: A Unified Methodology for Local Routing and Post-Training


### Images

![083afc1f25cfa03c3a7a6b6787e31c5238d2d8db8543a70b09059929aec0964d.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/083afc1f25cfa03c3a7a6b6787e31c5238d2d8db8543a70b09059929aec0964d.jpg)

![0a0fd558108a1ddc0f37f41de2ef0bb4b4266af832f209be4127dccc39a6a72f.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/0a0fd558108a1ddc0f37f41de2ef0bb4b4266af832f209be4127dccc39a6a72f.jpg)

![1b76088f1524252fa731a225537792b274117ee5090d9ef73acf15ebc5727280.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/1b76088f1524252fa731a225537792b274117ee5090d9ef73acf15ebc5727280.jpg)

![2072e0540cde1cfbebb32f85d4748a7126a717e15fd7f4ffd8ef6f17ccb42583.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/2072e0540cde1cfbebb32f85d4748a7126a717e15fd7f4ffd8ef6f17ccb42583.jpg)

![251ff55c5c59b548799cedae7b566a357f85097b77e805cd393dde8fd5541fb3.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/251ff55c5c59b548799cedae7b566a357f85097b77e805cd393dde8fd5541fb3.jpg)

![254ad33bad9ecb163ce7a075a1fb37a60dc029a2f76e43f7bd2aa4e8202a15d9.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/254ad33bad9ecb163ce7a075a1fb37a60dc029a2f76e43f7bd2aa4e8202a15d9.jpg)

![2d92ad9a18582e6f0cd36ed3a37f4abfc51bacfc28d39fbf6ecf39e1de271bd5.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/2d92ad9a18582e6f0cd36ed3a37f4abfc51bacfc28d39fbf6ecf39e1de271bd5.jpg)

![325e7a980f66c6bf2a2610f45831d478c0319a0d4267b8e45aa44bcdf91749c2.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/325e7a980f66c6bf2a2610f45831d478c0319a0d4267b8e45aa44bcdf91749c2.jpg)

![4a7dae7639dc02a8fa069beacd44cdc66f88ef86d3c31ea9e8b5746d5ae09c48.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/4a7dae7639dc02a8fa069beacd44cdc66f88ef86d3c31ea9e8b5746d5ae09c48.jpg)

![680fb89aa35ecdc74b63bfcd32d09487e434b0c6d25c0f64d351d06c5aaa3160.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/680fb89aa35ecdc74b63bfcd32d09487e434b0c6d25c0f64d351d06c5aaa3160.jpg)

![7a81bf062531b7c059b72ddeaa704d8e094db950daba03a31eb2634f7e49d905.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/7a81bf062531b7c059b72ddeaa704d8e094db950daba03a31eb2634f7e49d905.jpg)

![83fa47e75cc1790cd72d3a3664d037dc5e5e7fafda912de9d941a8ab64654fc5.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/83fa47e75cc1790cd72d3a3664d037dc5e5e7fafda912de9d941a8ab64654fc5.jpg)

![a68c2b2bbcb7f51326810c78b77fcbf65515d8c2d2005f0120bde8fad1785c8b.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/a68c2b2bbcb7f51326810c78b77fcbf65515d8c2d2005f0120bde8fad1785c8b.jpg)

![b31379eb075e75bccba5ce25ce6ae7e8567007959ccdb76150122acf493420b8.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/b31379eb075e75bccba5ce25ce6ae7e8567007959ccdb76150122acf493420b8.jpg)

![e72437aac9a08ac10f88a66af306e000e7782ad04114956e085c1196dac6d4ce.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/images/e72437aac9a08ac10f88a66af306e000e7782ad04114956e085c1196dac6d4ce.jpg)

### Tables

![00881798e94bac3bf1968eb34d2ce825b7d019b2d11e026173decf99394abd64.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/00881798e94bac3bf1968eb34d2ce825b7d019b2d11e026173decf99394abd64.jpg)

![17bf1ca6c329b93e788a2fb72032adb9be876fbe732dd3e34120277bbf58f923.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/17bf1ca6c329b93e788a2fb72032adb9be876fbe732dd3e34120277bbf58f923.jpg)

![3a15cad714835d35fe1395841d91c846225b9d5daa0af512cde47589c8190c0b.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/3a15cad714835d35fe1395841d91c846225b9d5daa0af512cde47589c8190c0b.jpg)

![3a98c80cf58532130bc603f51b0fef5c197bd4202c5786300487205041c62cc0.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/3a98c80cf58532130bc603f51b0fef5c197bd4202c5786300487205041c62cc0.jpg)

![7115ea8a0a79719c4fd11702b66005827af2d391d292bd64319318fdd62ae608.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/7115ea8a0a79719c4fd11702b66005827af2d391d292bd64319318fdd62ae608.jpg)

![75fac1cafaf1e955006baa3f5652937979c278c3eb59fa22c2545e65b883a6d5.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/75fac1cafaf1e955006baa3f5652937979c278c3eb59fa22c2545e65b883a6d5.jpg)

![bc72dadf48ad615a3643595fc2d10e5a26881b5d759980711dfde19a8f8c3880.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/bc72dadf48ad615a3643595fc2d10e5a26881b5d759980711dfde19a8f8c3880.jpg)

![bdd5fd9dfbaa931c28076b1f610101dbb099ddfcb7954645672722d85a7c1611.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/bdd5fd9dfbaa931c28076b1f610101dbb099ddfcb7954645672722d85a7c1611.jpg)

![ddbc4df2898f31eb7220b3cd068e9bd3cdd74f2d13b55925b8c1508aae3baecf.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/ddbc4df2898f31eb7220b3cd068e9bd3cdd74f2d13b55925b8c1508aae3baecf.jpg)

![f39e84e2ce7029af12ab8caa3d48135abf5c98b0ac67e6c8a7361ef82b0e4e47.jpg](../icml_results/1485_Bridging%20On-Device%20and%20Cloud%20LLMs%20for%20Collaborative%20Reasoning_%20A%20Unified%20Methodology%20for%20Local%20Routi/tables/f39e84e2ce7029af12ab8caa3d48135abf5c98b0ac67e6c8a7361ef82b0e4e47.jpg)

<a id="paper-1465"></a>

## Federated Sketching LoRA: A Flexible Framework for Heterogeneous Collaborative Fine-Tuning of LLMs


### Images

![1dd7379f45711270bba7eec82791dd9a6911fa95d0ec758811114f07d44b6c40.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/1dd7379f45711270bba7eec82791dd9a6911fa95d0ec758811114f07d44b6c40.jpg)

![2500671fb9abf403572fc84fb408e504b428bf846fe2aafba726f3bde068b678.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/2500671fb9abf403572fc84fb408e504b428bf846fe2aafba726f3bde068b678.jpg)

![2dd47c4805b520339f959bdac3f9fcffe3553e9b526ce275410863eede3d570c.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/2dd47c4805b520339f959bdac3f9fcffe3553e9b526ce275410863eede3d570c.jpg)

![594487dd2959ddcd6bdd777c8b75b37419a791f9d051205ba90b1175793c2739.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/594487dd2959ddcd6bdd777c8b75b37419a791f9d051205ba90b1175793c2739.jpg)

![65400946c91e3c4f0839fa18e25151f6ba65ffcf37391a5a87c70b7f024c8c78.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/65400946c91e3c4f0839fa18e25151f6ba65ffcf37391a5a87c70b7f024c8c78.jpg)

![b06a157225778e2f7e8f73fd2746cfdead15e5f7b2abc9d181d99b7c05cf7ef6.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/b06a157225778e2f7e8f73fd2746cfdead15e5f7b2abc9d181d99b7c05cf7ef6.jpg)

![d9574c51318420bd79201faafa23f44159db166abaeb60c18bef2b1b66a3c603.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/d9574c51318420bd79201faafa23f44159db166abaeb60c18bef2b1b66a3c603.jpg)

![dc7eb93ae9d7900af40f69dfa6de2d9d1956d8b8594981493a845d528358463b.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/images/dc7eb93ae9d7900af40f69dfa6de2d9d1956d8b8594981493a845d528358463b.jpg)

### Tables

![086b5f09a4a083abdd0b98a48dbc174222108d84a387b7fbd7623b6d5fa3712c.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/086b5f09a4a083abdd0b98a48dbc174222108d84a387b7fbd7623b6d5fa3712c.jpg)

![0fc62025717f093ecb87915f17096c4e42131364e93394c3ce03d5bc0aed118b.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/0fc62025717f093ecb87915f17096c4e42131364e93394c3ce03d5bc0aed118b.jpg)

![13e5ebe6bfbcaea295e11a2dca5093b94a3327914e5572a6eb70876678f179c7.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/13e5ebe6bfbcaea295e11a2dca5093b94a3327914e5572a6eb70876678f179c7.jpg)

![310984f84cbacbd74255568945bfd54645690508725392ea7c41e8f2da697c10.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/310984f84cbacbd74255568945bfd54645690508725392ea7c41e8f2da697c10.jpg)

![3fa0f4d21da7196f3975e9dcc4c7606812391a0cd92250fb435b5fd74b70879a.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/3fa0f4d21da7196f3975e9dcc4c7606812391a0cd92250fb435b5fd74b70879a.jpg)

![54acfb51c432615c1bd25199c466c6dd7ca49de62bd6050dc281df4dcb275ecf.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/54acfb51c432615c1bd25199c466c6dd7ca49de62bd6050dc281df4dcb275ecf.jpg)

![66b12be1cdee7ff49afa2cafcd6b76731f157654a6184787a051ae96be9bf538.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/66b12be1cdee7ff49afa2cafcd6b76731f157654a6184787a051ae96be9bf538.jpg)

![6887e5a28bb17159073ca0a9cde505d272a2f7e0fcd40a3a0646a0ca88d91569.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/6887e5a28bb17159073ca0a9cde505d272a2f7e0fcd40a3a0646a0ca88d91569.jpg)

![6c1c5a2393fe50c9bc2a40f41bd7fecf296cc984e5ec0fa08fc83e685b2e3680.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/6c1c5a2393fe50c9bc2a40f41bd7fecf296cc984e5ec0fa08fc83e685b2e3680.jpg)

![6c85a732b791959b40bfbcafb4f5ddb58b5d7376d493cffc65fc2f2d49e4d672.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/6c85a732b791959b40bfbcafb4f5ddb58b5d7376d493cffc65fc2f2d49e4d672.jpg)

![78f21462486789bacc4c5fa0a8616be302b5cd6c759d55768875295a209e7e8e.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/78f21462486789bacc4c5fa0a8616be302b5cd6c759d55768875295a209e7e8e.jpg)

![7f9323772b87318887d38d80c6d36fe36044cdf7814af3a4b1fc8f6cc29a5bde.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/7f9323772b87318887d38d80c6d36fe36044cdf7814af3a4b1fc8f6cc29a5bde.jpg)

![850487cf84c30e0b29d676b141aec6743ed7cea8d649ac47cf91c8fb218e7e9c.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/850487cf84c30e0b29d676b141aec6743ed7cea8d649ac47cf91c8fb218e7e9c.jpg)

![8aa9d09372bed9720f6ed146139ab0c9f1b467f9f6275553f6f610d338dfe321.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/8aa9d09372bed9720f6ed146139ab0c9f1b467f9f6275553f6f610d338dfe321.jpg)

![90963881220ef69802bb5a94b1975f2aca157d621209ebfd35a196942a4dd2a5.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/90963881220ef69802bb5a94b1975f2aca157d621209ebfd35a196942a4dd2a5.jpg)

![98c88e5a6e1d92a38002f764471956394a49d1f5a9f26c2729b583a26b5b1af2.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/98c88e5a6e1d92a38002f764471956394a49d1f5a9f26c2729b583a26b5b1af2.jpg)

![be2cfffc4f4430cc52d61f73b9906d633b3219e8c5eb9784dc2e079c9993c3ae.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/be2cfffc4f4430cc52d61f73b9906d633b3219e8c5eb9784dc2e079c9993c3ae.jpg)

![d98ef2f643142171151228bc68fe66a86dbe4d933cfbc905c8b8a4cb47912ce0.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/d98ef2f643142171151228bc68fe66a86dbe4d933cfbc905c8b8a4cb47912ce0.jpg)

![f47d32644d97506d996dba7007030c367b49eb3c72ac6391c13a887a182e5301.jpg](../icml_results/1486_Federated%20Sketching%20LoRA_%20A%20Flexible%20Framework%20for%20Heterogeneous%20Collaborative%20Fine-Tuning%20of%20LLMs/tables/f47d32644d97506d996dba7007030c367b49eb3c72ac6391c13a887a182e5301.jpg)

<a id="paper-1466"></a>

## Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models


### Images

![1613bd1c657b5b18a0b1195f7a990269eaf89bc1d2f01778d023e47253b3dd80.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/1613bd1c657b5b18a0b1195f7a990269eaf89bc1d2f01778d023e47253b3dd80.jpg)

![1e578531fe881493c263159740c1fd0fc90573eccfe506a15b0ed73565a8afa1.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/1e578531fe881493c263159740c1fd0fc90573eccfe506a15b0ed73565a8afa1.jpg)

![269fe2807a787a98adf922bd8d40ff875fc24b557c808734a0fb2fd47e563aad.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/269fe2807a787a98adf922bd8d40ff875fc24b557c808734a0fb2fd47e563aad.jpg)

![2e217948d36f21fa65230a96266b06bca2edab93b61f17d4bcb09f82f7e432bd.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/2e217948d36f21fa65230a96266b06bca2edab93b61f17d4bcb09f82f7e432bd.jpg)

![3293374d4aaa816c532ae60a1e2967528bc76525817a140bf158295af1db8af0.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/3293374d4aaa816c532ae60a1e2967528bc76525817a140bf158295af1db8af0.jpg)

![4a991000ce629dd43d198bb7033c8415d7a934a1c20f62e7f1ac36c1b21ba0f2.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/4a991000ce629dd43d198bb7033c8415d7a934a1c20f62e7f1ac36c1b21ba0f2.jpg)

![66c90bb0b2c61723b730dee39832f728bf662926dd8670dde4c363f55745cab9.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/66c90bb0b2c61723b730dee39832f728bf662926dd8670dde4c363f55745cab9.jpg)

![69f63d5e8ad19f9878fa14ba83847a2369754556d0b1e5f49f7716424c1364e6.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/69f63d5e8ad19f9878fa14ba83847a2369754556d0b1e5f49f7716424c1364e6.jpg)

![6e65acacf9d88136870f969a28287705e627f5ea17c0de74d2897db394ca2193.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/6e65acacf9d88136870f969a28287705e627f5ea17c0de74d2897db394ca2193.jpg)

![7569894a05c3a63cd3d29b5077bbbb465cbd0ab4bd58b30f4e49d4e2764f25b9.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/7569894a05c3a63cd3d29b5077bbbb465cbd0ab4bd58b30f4e49d4e2764f25b9.jpg)

![9a435fb8856fc26227cd02a61f7003a71841bc06a15ca62b354596edb9ed6de4.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/9a435fb8856fc26227cd02a61f7003a71841bc06a15ca62b354596edb9ed6de4.jpg)

![9de78765d7d0a6a02dddd6f6fc4587a4d2d99667a69f1500e30490963e74bbd4.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/9de78765d7d0a6a02dddd6f6fc4587a4d2d99667a69f1500e30490963e74bbd4.jpg)

![a4cf94bf36b4c59656f13ef3a53410efa77887799a90c8b88e090ee6a89d7ff4.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/a4cf94bf36b4c59656f13ef3a53410efa77887799a90c8b88e090ee6a89d7ff4.jpg)

![a641ddae37b8cfca95cecd658c774d0c3863c354dc916b3443d84e115fa257e5.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/a641ddae37b8cfca95cecd658c774d0c3863c354dc916b3443d84e115fa257e5.jpg)

![ad030bb0a0d58bc36e44eed1e6091360f6ff778b221f6e950b16f8b376e1fc78.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/ad030bb0a0d58bc36e44eed1e6091360f6ff778b221f6e950b16f8b376e1fc78.jpg)

![b78710f0ccca546baf89a99e06d553973fcb20a53b53166f2efae3396e681b48.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/b78710f0ccca546baf89a99e06d553973fcb20a53b53166f2efae3396e681b48.jpg)

![c639cacf7fea11d1364c1ec715d1a0dc64af90aab5883b96187a6e7935ff0ef7.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/c639cacf7fea11d1364c1ec715d1a0dc64af90aab5883b96187a6e7935ff0ef7.jpg)

![c745802d058040fbc19edf7f054a0a18f4b9ab1085a064c0cc38c78604c57e89.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/c745802d058040fbc19edf7f054a0a18f4b9ab1085a064c0cc38c78604c57e89.jpg)

![e84139254e4bb833ae0175ce44adc2f98ec01d3ac2e92a6c92bbc024682bbbb0.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/images/e84139254e4bb833ae0175ce44adc2f98ec01d3ac2e92a6c92bbc024682bbbb0.jpg)

### Tables

![1d3b77703d457269cc4bf39666643c47e4d15a8077bed617901c5b51df5da0ce.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/1d3b77703d457269cc4bf39666643c47e4d15a8077bed617901c5b51df5da0ce.jpg)

![374458a73e8cf29c167b2099d69ad8a69e838c164ca2feff6f4e1c1b3296334b.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/374458a73e8cf29c167b2099d69ad8a69e838c164ca2feff6f4e1c1b3296334b.jpg)

![3e4e10ec34ccf9b58efd49d62daaf7b2c12380be8ed6bca47d79ffeb9dc6a20e.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/3e4e10ec34ccf9b58efd49d62daaf7b2c12380be8ed6bca47d79ffeb9dc6a20e.jpg)

![46c4dc866bdd15db232af287d1869ed3862e07510c4fa9a6db0b6fb89b3625e0.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/46c4dc866bdd15db232af287d1869ed3862e07510c4fa9a6db0b6fb89b3625e0.jpg)

![810b08e2c5dc0b06375593e49f17cc4ce018dc1a70b507483f2af4bb918afd43.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/810b08e2c5dc0b06375593e49f17cc4ce018dc1a70b507483f2af4bb918afd43.jpg)

![92302a01176e7434205ec5a192144f080a964415baf33619fc7641c5ee43f099.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/92302a01176e7434205ec5a192144f080a964415baf33619fc7641c5ee43f099.jpg)

![fce196cb4e9ec96b37775e426e8c8def4cc5ce3a2dc4f01eae4840b847cc8047.jpg](../icml_results/1487_Debate%20with%20Images_%20Detecting%20Deceptive%20Behaviors%20in%20Multimodal%20Large%20Language%20Models/tables/fce196cb4e9ec96b37775e426e8c8def4cc5ce3a2dc4f01eae4840b847cc8047.jpg)

<a id="paper-1467"></a>

## MC-HNN: Learning Latent Structural Semantics and High-Rank Representations for Hypergraph Neural Networks


### Images

![8728c7f51f7213fefacbfde233a7de1fdc10adde9cf628053206eb72b248406d.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/images/8728c7f51f7213fefacbfde233a7de1fdc10adde9cf628053206eb72b248406d.jpg)

![9a017873091276a948ac7fe5645bd96af4d6605e22eb5fdc6c32740c10ffbe11.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/images/9a017873091276a948ac7fe5645bd96af4d6605e22eb5fdc6c32740c10ffbe11.jpg)

![abeb0fc27dc8112bc620909b1fabc229c5e380f30ebee6e545c45d4f114d430b.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/images/abeb0fc27dc8112bc620909b1fabc229c5e380f30ebee6e545c45d4f114d430b.jpg)

![b5badab2a52d11bc0ac4c63f085cd87c27af998d5f8a586465de13c48529d6b7.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/images/b5badab2a52d11bc0ac4c63f085cd87c27af998d5f8a586465de13c48529d6b7.jpg)

![ee4f3d48c7f51e479a85fe0b333ce0e1f66598cb25de677d66243896e78748e4.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/images/ee4f3d48c7f51e479a85fe0b333ce0e1f66598cb25de677d66243896e78748e4.jpg)

### Tables

![3596c6489865132e3b503fc2a4805741547b7991ced1bdbca8c3f0c5c2cbf571.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/tables/3596c6489865132e3b503fc2a4805741547b7991ced1bdbca8c3f0c5c2cbf571.jpg)

![8a6525c8042a68298f720a80fe9175e2b7b208557ee2d1be0d0944efe59a878e.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/tables/8a6525c8042a68298f720a80fe9175e2b7b208557ee2d1be0d0944efe59a878e.jpg)

![93464ebfaa7a6ba1e87b050962a923511c201384b4f7636d751e904d06528bed.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/tables/93464ebfaa7a6ba1e87b050962a923511c201384b4f7636d751e904d06528bed.jpg)

![a77127753315f02cf051e35acbd9552b634fab074abcab511f13c5675b42d448.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/tables/a77127753315f02cf051e35acbd9552b634fab074abcab511f13c5675b42d448.jpg)

![ab36118bcf60d5bdd628aa4a9a6bcb2c23a1d06c060cb3d82ccf2276b41e3e80.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/tables/ab36118bcf60d5bdd628aa4a9a6bcb2c23a1d06c060cb3d82ccf2276b41e3e80.jpg)

![db849d2aa82ff00dd713bf88277eef97610df3cf0a9fd704f10a89e36ab24814.jpg](../icml_results/1488_MC-HNN_%20Learning%20Latent%20Structural%20Semantics%20and%20High-Rank%20Representations%20for%20Hypergraph%20Neural%20Net/tables/db849d2aa82ff00dd713bf88277eef97610df3cf0a9fd704f10a89e36ab24814.jpg)

<a id="paper-1468"></a>

## OMP: One-step Meanflow Policy with Directional Alignment


### Images

![3ffd810556db306eaa0c15cd83781d8f6806701e450d9c4dea1ee464d70473f0.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/3ffd810556db306eaa0c15cd83781d8f6806701e450d9c4dea1ee464d70473f0.jpg)

![453320d6b0cc0391fbfaab015d00c3fe86044fb0b2a200678b591a4f5a5a2720.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/453320d6b0cc0391fbfaab015d00c3fe86044fb0b2a200678b591a4f5a5a2720.jpg)

![523da5bb260f46693efd15a4ac5dbb271fa0ba1e354c995f6d51bb2e1f324179.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/523da5bb260f46693efd15a4ac5dbb271fa0ba1e354c995f6d51bb2e1f324179.jpg)

![7a12abb457150da352b7768f82e2437fa70338ef78a99e1a10be6de0501c7ee3.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/7a12abb457150da352b7768f82e2437fa70338ef78a99e1a10be6de0501c7ee3.jpg)

![82cbdb21c5079bc749828f329a8ab8d917d58a7bec10abf5189bf76857bb3570.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/82cbdb21c5079bc749828f329a8ab8d917d58a7bec10abf5189bf76857bb3570.jpg)

![c4d9b051a77630a3c53a84e7bef36263de8f07a75f2cb479421bf1ba548b682d.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/c4d9b051a77630a3c53a84e7bef36263de8f07a75f2cb479421bf1ba548b682d.jpg)

![c85658febbe082f9696698c6c72c243c939af3a9c1101201d74eed7fd35e101d.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/c85658febbe082f9696698c6c72c243c939af3a9c1101201d74eed7fd35e101d.jpg)

![d54d3a754192bca6bd24db19a71a78883101c42125d0026ca71ee4b87d8bd072.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/images/d54d3a754192bca6bd24db19a71a78883101c42125d0026ca71ee4b87d8bd072.jpg)

### Tables

![3aab932d0621e9d33cec5d8fc5553e86ab3746e970e1d7698aff4aa9d23c8abe.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/tables/3aab932d0621e9d33cec5d8fc5553e86ab3746e970e1d7698aff4aa9d23c8abe.jpg)

![6bf4b27e3324d3388b511ee8ecec5e94613efca033e8a25a86ece8788de2d764.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/tables/6bf4b27e3324d3388b511ee8ecec5e94613efca033e8a25a86ece8788de2d764.jpg)

![7251e9b86debf3f13ee42349d5232fa5973270195f2db7bfc04b1ba55f9010f3.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/tables/7251e9b86debf3f13ee42349d5232fa5973270195f2db7bfc04b1ba55f9010f3.jpg)

![7fc55e948728870f53d428123d12873d4d68610f9587b6010cf22aedcf1a645a.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/tables/7fc55e948728870f53d428123d12873d4d68610f9587b6010cf22aedcf1a645a.jpg)

![95f49f8ddfefd7a529ad607de6445bbb4c95cfb3633d592ff302435254f70fad.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/tables/95f49f8ddfefd7a529ad607de6445bbb4c95cfb3633d592ff302435254f70fad.jpg)

![c9832ef75a00469bf32ae01e99ec7800841793f5dfe9e467648b98d085025ecd.jpg](../icml_results/1489_OMP_%20One-step%20Meanflow%20Policy%20with%20Directional%20Alignment/tables/c9832ef75a00469bf32ae01e99ec7800841793f5dfe9e467648b98d085025ecd.jpg)

<a id="paper-1469"></a>

## MINT: Minimal Information Neuro-Symbolic Tree for Objective-Driven Knowledge-Gap Reasoning and Active Elicitation


### Images

![7a4facd351d6803475399cbe7a9dabcf14d464a9e5342886b00df26e2ca52297.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/images/7a4facd351d6803475399cbe7a9dabcf14d464a9e5342886b00df26e2ca52297.jpg)

![82533370e56c1fef00484fba25454872cb27d9aa11aec28042aab272797301e5.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/images/82533370e56c1fef00484fba25454872cb27d9aa11aec28042aab272797301e5.jpg)

![f3f8602b331d818e26ca7b8522e40b17b3a28b61d9cbba9fe0e1e9b8b9c2d64f.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/images/f3f8602b331d818e26ca7b8522e40b17b3a28b61d9cbba9fe0e1e9b8b9c2d64f.jpg)

### Tables

![0931edbce68cd51a502a5edd1ff384cea251222c4a42083124458befa6610b3c.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/0931edbce68cd51a502a5edd1ff384cea251222c4a42083124458befa6610b3c.jpg)

![16ca5caccb61ca4672b2461e095500c9d84b941fc7212dd6d3ee62e660ad8137.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/16ca5caccb61ca4672b2461e095500c9d84b941fc7212dd6d3ee62e660ad8137.jpg)

![42a859344e65bb282be6d815bab60517d72d6c5b8bd2543d03d79146fe49cb94.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/42a859344e65bb282be6d815bab60517d72d6c5b8bd2543d03d79146fe49cb94.jpg)

![4333ce0ac4ddd834bdb212e57642f97a0dd0626ea67e001391ea5681d11ef924.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/4333ce0ac4ddd834bdb212e57642f97a0dd0626ea67e001391ea5681d11ef924.jpg)

![4d6069077dd6706fe15f13dc76a7e75be59239510bc970c0b3559c822a6f7125.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/4d6069077dd6706fe15f13dc76a7e75be59239510bc970c0b3559c822a6f7125.jpg)

![76441b6ba097bdd87dc89130d4a4b62ef9f0731b3b088d227cae4e127f5ab697.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/76441b6ba097bdd87dc89130d4a4b62ef9f0731b3b088d227cae4e127f5ab697.jpg)

![8c92ee679c358bb098048cbd52f875630ae8b50be52964639acfb1aea79f5b77.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/8c92ee679c358bb098048cbd52f875630ae8b50be52964639acfb1aea79f5b77.jpg)

![999dd8315014052efe1d6a4bd393724d1dc08f192dd87e31c61961b410dcf18b.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/999dd8315014052efe1d6a4bd393724d1dc08f192dd87e31c61961b410dcf18b.jpg)

![c236457b8512405d245d8ee23d41427b2501b1d91cb9f198ce79933c0b311546.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/c236457b8512405d245d8ee23d41427b2501b1d91cb9f198ce79933c0b311546.jpg)

![c82ce340a158c0afe9d58295090bb72eb1e167dd9cdd9382017e38376a5cd766.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/c82ce340a158c0afe9d58295090bb72eb1e167dd9cdd9382017e38376a5cd766.jpg)

![d8d611e31664d861b04e9faac5fe66732e5ca894e7e51ff4f93f5dba9c9286c0.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/d8d611e31664d861b04e9faac5fe66732e5ca894e7e51ff4f93f5dba9c9286c0.jpg)

![e0f9cc9e3fe88a57f980d03a7b85fda3448677d9b860e02ef43d0635caeb81bf.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/e0f9cc9e3fe88a57f980d03a7b85fda3448677d9b860e02ef43d0635caeb81bf.jpg)

![f3c612d5202f499809da17389bfaabb69acd9f1d8db7445b11daab0de0926064.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/f3c612d5202f499809da17389bfaabb69acd9f1d8db7445b11daab0de0926064.jpg)

![fe7d13f8704615679aa6dca027bf97d8c295b56f88f61ec196cbb2551afa9657.jpg](../icml_results/1490_MINT_%20Minimal%20Information%20Neuro-Symbolic%20Tree%20for%20Objective-Driven%20Knowledge-Gap%20Reasoning%20and%20Activ/tables/fe7d13f8704615679aa6dca027bf97d8c295b56f88f61ec196cbb2551afa9657.jpg)

<a id="paper-1470"></a>

## PRM-PBE: Process Reward Model for Reinforcement Learning in Programming-by-Example


### Images

![040a3f38dd8acc6df002d4a71658118272190ed9b21affb5b2494362bde9b65c.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/images/040a3f38dd8acc6df002d4a71658118272190ed9b21affb5b2494362bde9b65c.jpg)

![14716df6c24a58cedd74606fd9192afd09ca102dba6c75af836bd1ae8641d066.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/images/14716df6c24a58cedd74606fd9192afd09ca102dba6c75af836bd1ae8641d066.jpg)

![3c93a85af279eb414eb48a07ad787d0d52fbb1f4cc83636f3532327cbb16f18d.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/images/3c93a85af279eb414eb48a07ad787d0d52fbb1f4cc83636f3532327cbb16f18d.jpg)

![573d2e07bae0c28c9009a6d0064b0c29a1732633a6fdbc62d5d02cd0004bccfb.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/images/573d2e07bae0c28c9009a6d0064b0c29a1732633a6fdbc62d5d02cd0004bccfb.jpg)

![9037df740c76bcfcf06072020289c58495b8d1a8a1a334b3db67baee94423d0c.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/images/9037df740c76bcfcf06072020289c58495b8d1a8a1a334b3db67baee94423d0c.jpg)

![f9fc91f31aaa8df41f89a26c89f97af28f03573f305bad30de069ece4058a7ef.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/images/f9fc91f31aaa8df41f89a26c89f97af28f03573f305bad30de069ece4058a7ef.jpg)

### Tables

![180f044097eb402d4fafcf0f6af91cb92f4b4a83877f33124a1b855c832736c1.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/tables/180f044097eb402d4fafcf0f6af91cb92f4b4a83877f33124a1b855c832736c1.jpg)

![52b8772d896a8710d6c0ded49aab5860505467eee06a7e65ab3b564622a39339.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/tables/52b8772d896a8710d6c0ded49aab5860505467eee06a7e65ab3b564622a39339.jpg)

![a19a9052058373651e53d1972760a2626dfcaf5fc399240fc1d840982c5b15af.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/tables/a19a9052058373651e53d1972760a2626dfcaf5fc399240fc1d840982c5b15af.jpg)

![acb8a5b4b01fac4d30883c7aac58151536cd09a270801f7428fa457cdcf590f4.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/tables/acb8a5b4b01fac4d30883c7aac58151536cd09a270801f7428fa457cdcf590f4.jpg)

![acfb595b4b924e6ac311d0088ee686ebde2abd57077b2b4b7eea54c0060f5eba.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/tables/acfb595b4b924e6ac311d0088ee686ebde2abd57077b2b4b7eea54c0060f5eba.jpg)

![bedd64488eaeee36b25c113168117cab8d0e0e03252f51126b386fcedbe043e9.jpg](../icml_results/1491_PRM-PBE_%20Process%20Reward%20Model%20for%20Reinforcement%20Learning%20in%20Programming-by-Example/tables/bedd64488eaeee36b25c113168117cab8d0e0e03252f51126b386fcedbe043e9.jpg)

<a id="paper-1471"></a>

## LOZO+: Provably Efficient Zeroth-Order Fine-Tuning via Greedy Low-Rank Subspace Selection


### Images

![125d8211bdcfad2201111cc38c64902e50db9994e7468539b28671b929ea8be9.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/125d8211bdcfad2201111cc38c64902e50db9994e7468539b28671b929ea8be9.jpg)

![48e5b4d8f86555d11f8f3a4d9e1d5ce0536ad3020adf9eadc6ed2e36e80389e1.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/48e5b4d8f86555d11f8f3a4d9e1d5ce0536ad3020adf9eadc6ed2e36e80389e1.jpg)

![695e62ff53c73613c2248a5be6b725eef8c19bd3086ebd5626cb45fecd97b997.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/695e62ff53c73613c2248a5be6b725eef8c19bd3086ebd5626cb45fecd97b997.jpg)

![902cddf3ff4e6eae1b1fc62142c97167fdff2577e5902bca0d9f1ce7ad486d6e.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/902cddf3ff4e6eae1b1fc62142c97167fdff2577e5902bca0d9f1ce7ad486d6e.jpg)

![b72e7955d3abd9fb2beeaa1cbfb43997d0fbe627d3c4f66803491543d44f0531.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/b72e7955d3abd9fb2beeaa1cbfb43997d0fbe627d3c4f66803491543d44f0531.jpg)

![e34fe640cb94e849fa768c37aaa4c815d2887fa0f33c63b192ef82ac7be2fe60.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/e34fe640cb94e849fa768c37aaa4c815d2887fa0f33c63b192ef82ac7be2fe60.jpg)

![f8445bc761769898a2f0d40fff1e6dd818270216a8498efad8540d180a8cdc98.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/images/f8445bc761769898a2f0d40fff1e6dd818270216a8498efad8540d180a8cdc98.jpg)

### Tables

![3f0f6925f984d4e39963c496b26540297c314f4e84178f1bd0008f30ebe8fe60.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/3f0f6925f984d4e39963c496b26540297c314f4e84178f1bd0008f30ebe8fe60.jpg)

![449f343a83b800acdf67efb32e6c84864cb8fb6abdb969ccb04dea339cd8ae75.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/449f343a83b800acdf67efb32e6c84864cb8fb6abdb969ccb04dea339cd8ae75.jpg)

![534b37a7ae66841a4968b2ada11d9153f2de3c4ebb72da45ae3f41bb31bb44ee.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/534b37a7ae66841a4968b2ada11d9153f2de3c4ebb72da45ae3f41bb31bb44ee.jpg)

![5ddc6c6f0fb6caf9cfca14163c21d23303d91886a0f53337b82cf98c48d328d2.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/5ddc6c6f0fb6caf9cfca14163c21d23303d91886a0f53337b82cf98c48d328d2.jpg)

![7853851afbd73b14372094bf671a1db8b4a4a39c2b5bb76aa812b89da2075561.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/7853851afbd73b14372094bf671a1db8b4a4a39c2b5bb76aa812b89da2075561.jpg)

![8c63a3c751c09cf6a9f29cc33ddec55dc94c3722b41991bf6776e228bb1c03a3.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/8c63a3c751c09cf6a9f29cc33ddec55dc94c3722b41991bf6776e228bb1c03a3.jpg)

![8f86ebe917b1f9083f9785583f08b8ffa84b08a7ce9413ef4226f09c6ab8da08.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/8f86ebe917b1f9083f9785583f08b8ffa84b08a7ce9413ef4226f09c6ab8da08.jpg)

![91d5053a757716aea28c6b4b00e70c58cc507fee076bf48145518bd618c20809.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/91d5053a757716aea28c6b4b00e70c58cc507fee076bf48145518bd618c20809.jpg)

![af4cc9281fc28a615a0ba2461978c21bd3baa8a46fab6ae48916e4fe64322950.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/af4cc9281fc28a615a0ba2461978c21bd3baa8a46fab6ae48916e4fe64322950.jpg)

![b77b412396c5d4a4dc61f6f5e8b6772ece64e268a65587350edf4b39bfa42948.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/b77b412396c5d4a4dc61f6f5e8b6772ece64e268a65587350edf4b39bfa42948.jpg)

![d814783f2c6b9b4435a3df5e4b9db41a7815991fa9dfdb29e0e8ae4160585431.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/d814783f2c6b9b4435a3df5e4b9db41a7815991fa9dfdb29e0e8ae4160585431.jpg)

![dd73ce752ee2cef3624f5a1fc6013ba993ee4612cab64311d37aa38648a152a2.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/dd73ce752ee2cef3624f5a1fc6013ba993ee4612cab64311d37aa38648a152a2.jpg)

![f31cf4cb7198ac5711e3d57c5380a1180909a842f16f177e75e87b87ad62e8be.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/f31cf4cb7198ac5711e3d57c5380a1180909a842f16f177e75e87b87ad62e8be.jpg)

![f652c79f83e91392bb81b65bd9c7122faebec7d9db9b55eeb2f77134a01dcf54.jpg](../icml_results/1492_LOZO%2B_%20Provably%20Efficient%20Zeroth-Order%20Fine-Tuning%20via%20Greedy%20Low-Rank%20Subspace%20Selection/tables/f652c79f83e91392bb81b65bd9c7122faebec7d9db9b55eeb2f77134a01dcf54.jpg)

<a id="paper-1472"></a>

## iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework


### Images

![24d281fde6fffdbcb6a6b7807d8aaba6e5e1a48580d767b75e291c0496972f23.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/24d281fde6fffdbcb6a6b7807d8aaba6e5e1a48580d767b75e291c0496972f23.jpg)

![2625f5779ad9b2066460246f3c51fb320d38049af60154f3df4430c8f1ecb1ee.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/2625f5779ad9b2066460246f3c51fb320d38049af60154f3df4430c8f1ecb1ee.jpg)

![2e185d5ccc89daa7ab3a99b09bf382acad70ac97692e144cb8e6f89ac94ae179.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/2e185d5ccc89daa7ab3a99b09bf382acad70ac97692e144cb8e6f89ac94ae179.jpg)

![35b2a9655c3f6792dafdcb37ae5632326c3d3766b81fd844de119df36e7d8a06.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/35b2a9655c3f6792dafdcb37ae5632326c3d3766b81fd844de119df36e7d8a06.jpg)

![5738c9af1abee6ab3d52510253d81c126fcaf9f523e0826a2a5982e5831476e7.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/5738c9af1abee6ab3d52510253d81c126fcaf9f523e0826a2a5982e5831476e7.jpg)

![848037fc3933b89845c67706b799ae68529de99495c9e08de71dd913e3a7398c.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/848037fc3933b89845c67706b799ae68529de99495c9e08de71dd913e3a7398c.jpg)

![9633b963ec988892f5eecec143a445e5d4788707afd65bb34fe9dee14e8c7ca3.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/9633b963ec988892f5eecec143a445e5d4788707afd65bb34fe9dee14e8c7ca3.jpg)

![ba6d55f1dc72ba89f789b43b090d7e4ba179cad96232167ce81521bfd8c3cd53.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/ba6d55f1dc72ba89f789b43b090d7e4ba179cad96232167ce81521bfd8c3cd53.jpg)

![d1ce1f88cae5b8eb7118b06738a2aadb2030c8c4bf6b73f6f4e3672600979588.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/d1ce1f88cae5b8eb7118b06738a2aadb2030c8c4bf6b73f6f4e3672600979588.jpg)

![dfca62ceee2d172e65cebc13bf2a710242616938f42fca084d0992044347c0ba.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/dfca62ceee2d172e65cebc13bf2a710242616938f42fca084d0992044347c0ba.jpg)

![e00e9ecac47be4527ee6b753fe2d89aab1b8caa951cc406051452fdf954a82ab.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/e00e9ecac47be4527ee6b753fe2d89aab1b8caa951cc406051452fdf954a82ab.jpg)

![e22055ba8ae63c9124e5432fb277ea966eeae6b2e7bd14f43757abcaff0b8d32.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/e22055ba8ae63c9124e5432fb277ea966eeae6b2e7bd14f43757abcaff0b8d32.jpg)

![eb215d8396333938a5529afdba59257ecc5a768af1ea0e343eb39b9d46aea13a.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/eb215d8396333938a5529afdba59257ecc5a768af1ea0e343eb39b9d46aea13a.jpg)

![ebcf287518a8b6bb2485a07ed499d4357b339979eade61e8a472ef33309a3d40.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/ebcf287518a8b6bb2485a07ed499d4357b339979eade61e8a472ef33309a3d40.jpg)

![efd7879f04b74a747c1920aa05ea51baded9bcebed665352c6713f2671dc2d1b.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/images/efd7879f04b74a747c1920aa05ea51baded9bcebed665352c6713f2671dc2d1b.jpg)

### Tables

![06645d83ac3c576b9b44a8ff9ecee79a7714c052cd3d9a9968ff610dd2120435.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/06645d83ac3c576b9b44a8ff9ecee79a7714c052cd3d9a9968ff610dd2120435.jpg)

![108c920ef914f0b623bb31ae74cc7a8d64ba83e8d305dd90a8298d0e225f4cb7.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/108c920ef914f0b623bb31ae74cc7a8d64ba83e8d305dd90a8298d0e225f4cb7.jpg)

![111a4a16dded6b346ba982115b80d8003632cdfe296494a392bb29b7277d8f44.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/111a4a16dded6b346ba982115b80d8003632cdfe296494a392bb29b7277d8f44.jpg)

![1ad39760d734090fc5ef786d773ec9410249f9ca737f0a9a7aedcbc4acc66e03.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/1ad39760d734090fc5ef786d773ec9410249f9ca737f0a9a7aedcbc4acc66e03.jpg)

![321406b14b8fa8dd663f1de5cb0f038d2af02cbcaf66a2199dec842a137bd7d3.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/321406b14b8fa8dd663f1de5cb0f038d2af02cbcaf66a2199dec842a137bd7d3.jpg)

![472652613c9ac0146c55a430c23e16c072f175f360614eeb050a8dde9fa44332.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/472652613c9ac0146c55a430c23e16c072f175f360614eeb050a8dde9fa44332.jpg)

![611f57ad1b2c37050463fecf1f840b5dcd65f4fec8fe90db9980404de004a4ef.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/611f57ad1b2c37050463fecf1f840b5dcd65f4fec8fe90db9980404de004a4ef.jpg)

![62d4fb0b03cd15470f329aef7f075c27f5573b0886a139011526b068219a4057.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/62d4fb0b03cd15470f329aef7f075c27f5573b0886a139011526b068219a4057.jpg)

![8061e5d1cbbe9d836e35007a5d6b7df9e36f44d5b5bbfa8fd5d9aa824aacd8b1.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/8061e5d1cbbe9d836e35007a5d6b7df9e36f44d5b5bbfa8fd5d9aa824aacd8b1.jpg)

![8aecaecb1bcba5bba37e56262018d64f4464f1db5433c9e3cd7680ce8d094384.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/8aecaecb1bcba5bba37e56262018d64f4464f1db5433c9e3cd7680ce8d094384.jpg)

![8f36614584184f0fb8d39cf1147991f23907d8f41da5370224ee6bf7dff6ef70.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/8f36614584184f0fb8d39cf1147991f23907d8f41da5370224ee6bf7dff6ef70.jpg)

![a538f07998443f1765b6836b4551586badca6fc76c8863a1428488157f9d6fe0.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/a538f07998443f1765b6836b4551586badca6fc76c8863a1428488157f9d6fe0.jpg)

![a7ae39f13c05979482c4b21fbfe058853feac22cbed01e2f19308ca5656f715e.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/a7ae39f13c05979482c4b21fbfe058853feac22cbed01e2f19308ca5656f715e.jpg)

![c7f10b26f78d8e790ad7f477c777587ca820bf3b76a364d81e53159134212094.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/c7f10b26f78d8e790ad7f477c777587ca820bf3b76a364d81e53159134212094.jpg)

![d643c9a8bbf523dfc84ccf5c9369286283bd221e6aee2206ac1a272bfb643839.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/d643c9a8bbf523dfc84ccf5c9369286283bd221e6aee2206ac1a272bfb643839.jpg)

![dad695d789fab35808a1b881c928d1691916ad3db42406a693ea804102363786.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/dad695d789fab35808a1b881c928d1691916ad3db42406a693ea804102363786.jpg)

![e4416f1681bcacc56e75fcb53a0fe71de6be97790f3587c3c45a21e346a0b722.jpg](../icml_results/1493_iWorld-Bench_%20A%20Benchmark%20for%20Interactive%20World%20Models%20with%20a%20Unified%20Action%20Generation%20Framework/tables/e4416f1681bcacc56e75fcb53a0fe71de6be97790f3587c3c45a21e346a0b722.jpg)

<a id="paper-1473"></a>

## Information Geometry Loss for Time Series Forecasting


### Images

![033b6413c60d9e78f304d695a3bb4cb6d8a518ea23d6774392514120e7e61c92.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/images/033b6413c60d9e78f304d695a3bb4cb6d8a518ea23d6774392514120e7e61c92.jpg)

![117e8ce0c2514085a0a9917f4ca5d766cf7e51c2f878777b75b3c8dcb93f7d20.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/images/117e8ce0c2514085a0a9917f4ca5d766cf7e51c2f878777b75b3c8dcb93f7d20.jpg)

![205183c6ddb5684c6ef757af82296f51e35561caa93fabdbf02d17aba74d8246.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/images/205183c6ddb5684c6ef757af82296f51e35561caa93fabdbf02d17aba74d8246.jpg)

![80677acbc93df48373c186e1e370f8dd57e755adeb3a3a0dca27c2a2e85cca58.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/images/80677acbc93df48373c186e1e370f8dd57e755adeb3a3a0dca27c2a2e85cca58.jpg)

![dc02067f7c2d105f666c5940cff42802e97b2923e55f1a012a0e95af26e7d159.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/images/dc02067f7c2d105f666c5940cff42802e97b2923e55f1a012a0e95af26e7d159.jpg)

### Tables

![16b0e52575eee1f705e502d821522014b0c546f6c4132095783634ee73413f97.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/tables/16b0e52575eee1f705e502d821522014b0c546f6c4132095783634ee73413f97.jpg)

![184811f2f30faf3de19d6df121c7b3b0c52f513c70b9ffeec2d658ccef1d52b7.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/tables/184811f2f30faf3de19d6df121c7b3b0c52f513c70b9ffeec2d658ccef1d52b7.jpg)

![2dce2c9a252f79f3e2f325a3cdf8a139000b108f16db014a151f22a3ea069882.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/tables/2dce2c9a252f79f3e2f325a3cdf8a139000b108f16db014a151f22a3ea069882.jpg)

![64fb43152a0202aa33de695b50dd105ab7102afb19d38d96ca4d4afd573ca9d9.jpg](../icml_results/1494_Information%20Geometry%20Loss%20for%20Time%20Series%20Forecasting/tables/64fb43152a0202aa33de695b50dd105ab7102afb19d38d96ca4d4afd573ca9d9.jpg)

<a id="paper-1474"></a>

## Trajectory-Aware Spiking DiTs Conversion via Membrane Potential Error-Feedback


### Images

![07497aeed34663e64c6540c44fc73dd0b184e8eb8799968c83b450088ff5c28d.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/07497aeed34663e64c6540c44fc73dd0b184e8eb8799968c83b450088ff5c28d.jpg)

![590c36567c20ffaea6a33395d32c6e3331cd8dc47dc729d405841f1bbc795084.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/590c36567c20ffaea6a33395d32c6e3331cd8dc47dc729d405841f1bbc795084.jpg)

![8e7ef606e1a31617e195a0f69be1cc5a29ec7b57854c4d3ad3236ab208151055.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/8e7ef606e1a31617e195a0f69be1cc5a29ec7b57854c4d3ad3236ab208151055.jpg)

![9dbdfe5017000db029077d0b6665abf3392328e2de1110649ee1be36db05937d.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/9dbdfe5017000db029077d0b6665abf3392328e2de1110649ee1be36db05937d.jpg)

![a758ad4517c951c0f7fa27cb935078591cc42fcca0a414996bfda788958000f1.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/a758ad4517c951c0f7fa27cb935078591cc42fcca0a414996bfda788958000f1.jpg)

![c60483b4fac32a0ea3d9cfb8ebb5337b1be503388ea309248e5858caf7876fb2.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/c60483b4fac32a0ea3d9cfb8ebb5337b1be503388ea309248e5858caf7876fb2.jpg)

![d42f7f9a53b9cf65957b0921d3853961198e057ffb774bad9ddc35f99b32dcae.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/images/d42f7f9a53b9cf65957b0921d3853961198e057ffb774bad9ddc35f99b32dcae.jpg)

### Tables

![2adf28d47dd43fa73c80f6e66043981130e3a85078c02b167a51b78f699aff57.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/tables/2adf28d47dd43fa73c80f6e66043981130e3a85078c02b167a51b78f699aff57.jpg)

![96848361b0b58818a411e0865b7bd719d3d762001efb0e5a51349147a7a3f174.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/tables/96848361b0b58818a411e0865b7bd719d3d762001efb0e5a51349147a7a3f174.jpg)

![ccc8ce72456dee1df201e083752cd31dc6d141b95e24748cd4ab55f8eab84e4b.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/tables/ccc8ce72456dee1df201e083752cd31dc6d141b95e24748cd4ab55f8eab84e4b.jpg)

![f1335c38ff83f541abbb7f758dd54c0269db489518f8c14027421231a39b8ee9.jpg](../icml_results/1495_Trajectory-Aware%20Spiking%20DiTs%20Conversion%20via%20Membrane%20Potential%20Error-Feedback/tables/f1335c38ff83f541abbb7f758dd54c0269db489518f8c14027421231a39b8ee9.jpg)

<a id="paper-1475"></a>

## Event2Vec: Processing neuromorphic events directly by representations in vector space


### Images

![277e5a0a6b3b07bbaefd38bf8cd0282e65c23632141fcdf8424243d301ac6ea4.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/277e5a0a6b3b07bbaefd38bf8cd0282e65c23632141fcdf8424243d301ac6ea4.jpg)

![5094afff12ed4de5acb013f982f42910449110aa95df4b0c55d1eb04e4b2b148.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/5094afff12ed4de5acb013f982f42910449110aa95df4b0c55d1eb04e4b2b148.jpg)

![522efe17c5632f113c74e1b6daaca76e493585ab902b2dd2e638082dfa9e8028.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/522efe17c5632f113c74e1b6daaca76e493585ab902b2dd2e638082dfa9e8028.jpg)

![700d3b815e9b407c88d42e84728e94742ec6b356df1ba50729949e4c4fbc2f51.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/700d3b815e9b407c88d42e84728e94742ec6b356df1ba50729949e4c4fbc2f51.jpg)

![80dd9fe5924f8f448f3a0418a3d880820025c0eb07ee180ce31bad6f6c7e944c.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/80dd9fe5924f8f448f3a0418a3d880820025c0eb07ee180ce31bad6f6c7e944c.jpg)

![8232a0d5e8356cc07d561cbc3bff8ca05c33e5e1432db6bf42a92bc777b71db6.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/8232a0d5e8356cc07d561cbc3bff8ca05c33e5e1432db6bf42a92bc777b71db6.jpg)

![90716873d823c43c998594521aefd5ccca2aa1331f534bac67339518baff4156.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/90716873d823c43c998594521aefd5ccca2aa1331f534bac67339518baff4156.jpg)

![b47f416477837cf1128a3dd3a565aa64cf123b7001b235ce067c58304212cb69.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/b47f416477837cf1128a3dd3a565aa64cf123b7001b235ce067c58304212cb69.jpg)

![df3a46c355b4f9fd8f88dd01fbb1890d70edf0d4ce94b00780706525a0d8952e.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/images/df3a46c355b4f9fd8f88dd01fbb1890d70edf0d4ce94b00780706525a0d8952e.jpg)

### Tables

![1c7e8a48e7af8d7c6eae88fb99e66c183800976542e6412e4e2d23740b5d2853.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/1c7e8a48e7af8d7c6eae88fb99e66c183800976542e6412e4e2d23740b5d2853.jpg)

![287f6b0bda3aa01ddb2779c1d13f985dc223b7ce20b9eac44cf0e3cdc6ea31c2.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/287f6b0bda3aa01ddb2779c1d13f985dc223b7ce20b9eac44cf0e3cdc6ea31c2.jpg)

![3dd010ae180fef05ce6865dd14c429b3db3848403a97cc149b0c6a7197d04012.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/3dd010ae180fef05ce6865dd14c429b3db3848403a97cc149b0c6a7197d04012.jpg)

![5c527c08daaccddfb40f232c45aec5c11b236fd9496e8981be901fb67bea3a6f.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/5c527c08daaccddfb40f232c45aec5c11b236fd9496e8981be901fb67bea3a6f.jpg)

![7fb19127c1b6157ef70e8fcc7aff3a845ef7e44cba56886a234e89fcc9fcec12.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/7fb19127c1b6157ef70e8fcc7aff3a845ef7e44cba56886a234e89fcc9fcec12.jpg)

![9823b5c87af28c770c700b768f5b6f55a72a548ceab74f1068c1c99fc885dc6a.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/9823b5c87af28c770c700b768f5b6f55a72a548ceab74f1068c1c99fc885dc6a.jpg)

![994d79c0b115b5bc116a1d3d359d084e9b8d66fd80ebe24196d9b21df4094ae2.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/994d79c0b115b5bc116a1d3d359d084e9b8d66fd80ebe24196d9b21df4094ae2.jpg)

![d599a31f49d9066b56c7f54da054fd730085a97c9274f628eefc5f31ca469c00.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/d599a31f49d9066b56c7f54da054fd730085a97c9274f628eefc5f31ca469c00.jpg)

![f1e14fd77314f3133f43ee3018e8311a338c2ff8a2a51c0764567f01cdb274c3.jpg](../icml_results/1496_Event2Vec_%20Processing%20neuromorphic%20events%20directly%20by%20representations%20in%20vector%20space/tables/f1e14fd77314f3133f43ee3018e8311a338c2ff8a2a51c0764567f01cdb274c3.jpg)

<a id="paper-1476"></a>

## Dynamic Fractal Mamba: A Neural Renormalization Group Flow for Scale-Invariant Sequence Modeling


### Images

![01dc9d26dc3a5dcf2d58cd183399b7e1c44048deee551a0afcfd3d743743904a.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/images/01dc9d26dc3a5dcf2d58cd183399b7e1c44048deee551a0afcfd3d743743904a.jpg)

![42c1928dacd30d86cb63057641d589408a8a58edb264b39b5a3ea4b5b0562483.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/images/42c1928dacd30d86cb63057641d589408a8a58edb264b39b5a3ea4b5b0562483.jpg)

![b760b1358a3fe388f26e7a2cda5b36d6673193805fc416c0a07fcb6cfaf0512e.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/images/b760b1358a3fe388f26e7a2cda5b36d6673193805fc416c0a07fcb6cfaf0512e.jpg)

![c2c76cb5a2e1052048981285c0f69e9b936659b80d5ebb2c81266e6c38ca39dc.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/images/c2c76cb5a2e1052048981285c0f69e9b936659b80d5ebb2c81266e6c38ca39dc.jpg)

![f60b7fb75228cf9eea737fb72ab3475602abee07b9e0773077db17f7da6fdc12.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/images/f60b7fb75228cf9eea737fb72ab3475602abee07b9e0773077db17f7da6fdc12.jpg)

### Tables

![048ba51bea2859cc692ef0aabdfe588754a27de7554d4f68de60be45ac07cb34.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/048ba51bea2859cc692ef0aabdfe588754a27de7554d4f68de60be45ac07cb34.jpg)

![1834f453c5074989df9c4efdf8b14a5763d5aa8f1c548a40fe6847f87f071e52.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/1834f453c5074989df9c4efdf8b14a5763d5aa8f1c548a40fe6847f87f071e52.jpg)

![3596f619e276c332373a8a95b1c96ea326ce1744e349b16d5b8db603daae3cf5.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/3596f619e276c332373a8a95b1c96ea326ce1744e349b16d5b8db603daae3cf5.jpg)

![3de62c9c1559dbf45429585038cbcd6cdece8a20cc1bfd9982c49c59322d6572.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/3de62c9c1559dbf45429585038cbcd6cdece8a20cc1bfd9982c49c59322d6572.jpg)

![4e955059caff3b008bbef966ad7a92d98467a91387888e78cc09d737e66bf705.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/4e955059caff3b008bbef966ad7a92d98467a91387888e78cc09d737e66bf705.jpg)

![56ff293b5257df0ba87663ff5c60bbae51ff9726b38c8effe4e470fb1e9a1094.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/56ff293b5257df0ba87663ff5c60bbae51ff9726b38c8effe4e470fb1e9a1094.jpg)

![6e3c0d2bc03e625e09091f45da98fdd6307cce700887dc6d8b9ef04c2bbc0765.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/6e3c0d2bc03e625e09091f45da98fdd6307cce700887dc6d8b9ef04c2bbc0765.jpg)

![785bbaa4da407d25e0b3bfed26ba5c7fda7908c4d07f41f3f22025225c5e10e0.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/785bbaa4da407d25e0b3bfed26ba5c7fda7908c4d07f41f3f22025225c5e10e0.jpg)

![7afd20696f05b3fc6a92924a746ee044e3dbc937af21c267bba45f4854203209.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/7afd20696f05b3fc6a92924a746ee044e3dbc937af21c267bba45f4854203209.jpg)

![7e82b1708e56daa38b1ee2248edf0c337cac36abff68fa74838021fc0de218f0.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/7e82b1708e56daa38b1ee2248edf0c337cac36abff68fa74838021fc0de218f0.jpg)

![7e8307adb4469278c715b6e8e2e31b79a1d5f53de1d9c0ea63b18b0f1fed314f.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/7e8307adb4469278c715b6e8e2e31b79a1d5f53de1d9c0ea63b18b0f1fed314f.jpg)

![8044d4501bafab34b60756dc439233d73815a1e04d37ac54054ec99531467045.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/8044d4501bafab34b60756dc439233d73815a1e04d37ac54054ec99531467045.jpg)

![8161c7ead3d34336db21b544a0625a8bc52eeed1f756dcd12416795d610bf070.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/8161c7ead3d34336db21b544a0625a8bc52eeed1f756dcd12416795d610bf070.jpg)

![988f0a2f1fd0d7677b112bee8bfa1f5b5cfc4ead44e6895e2ac5d8ed48c61bd5.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/988f0a2f1fd0d7677b112bee8bfa1f5b5cfc4ead44e6895e2ac5d8ed48c61bd5.jpg)

![9d5ae6cc2f5ca674594cf6c66b44fdb8e9e62c698b8e7ca8fb50cf0b25323102.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/9d5ae6cc2f5ca674594cf6c66b44fdb8e9e62c698b8e7ca8fb50cf0b25323102.jpg)

![a480d2c425d727da5b133076862b9440f84781a5cb90a59f7ae27a45f8a9fcc7.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/a480d2c425d727da5b133076862b9440f84781a5cb90a59f7ae27a45f8a9fcc7.jpg)

![d0b411e8f6200e61d9dadcf551a713326f41d4e030943d4dbcafe162a17012c8.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/d0b411e8f6200e61d9dadcf551a713326f41d4e030943d4dbcafe162a17012c8.jpg)

![da516f1a3de20e5e0989e9e534430f6a8084650e7330b66282836a9e0df82174.jpg](../icml_results/1497_Dynamic%20Fractal%20Mamba_%20A%20Neural%20Renormalization%20Group%20Flow%20for%20Scale-Invariant%20Sequence%20Modeling/tables/da516f1a3de20e5e0989e9e534430f6a8084650e7330b66282836a9e0df82174.jpg)

<a id="paper-1477"></a>

## VBA: Vector Bundle Attention for Intrinsically Geometric Representation Learning


### Images

![31f8d6d9bdc8784b3209c34b7c034fd1c68dec61fc165add47f85f533a2b5bf4.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/images/31f8d6d9bdc8784b3209c34b7c034fd1c68dec61fc165add47f85f533a2b5bf4.jpg)

![55f5a21df993c0a0e451663bd40f2edbce4de7b909f38d9875b11a5aad5fa4e0.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/images/55f5a21df993c0a0e451663bd40f2edbce4de7b909f38d9875b11a5aad5fa4e0.jpg)

![5fee7d80bafbeb3b76e85592d6ff4aceacc28c30ba55c69239a7e676a4e66be2.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/images/5fee7d80bafbeb3b76e85592d6ff4aceacc28c30ba55c69239a7e676a4e66be2.jpg)

![ce384ee8289946163b11219deb0c8ea15c495b0434cd34772b28e260ccea48b3.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/images/ce384ee8289946163b11219deb0c8ea15c495b0434cd34772b28e260ccea48b3.jpg)

![ea31a480e5e153d4741dc87daf13fc1b1dcd6bdbc00ecaa32cab96a5cf61cdf8.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/images/ea31a480e5e153d4741dc87daf13fc1b1dcd6bdbc00ecaa32cab96a5cf61cdf8.jpg)

![ff0605b00af9cdfe2e0cc5629b8543fc66c6d0fca1234f3a74f51388fb8d0bd4.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/images/ff0605b00af9cdfe2e0cc5629b8543fc66c6d0fca1234f3a74f51388fb8d0bd4.jpg)

### Tables

![0e74972ce34d869514641668a65a0ed6507d27ef56d9ca143973ec228d4a4ea9.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/0e74972ce34d869514641668a65a0ed6507d27ef56d9ca143973ec228d4a4ea9.jpg)

![1152241bb2e3f8f0d264a6fbc448b942dd50dd50a9e3de459f3f66d81705c889.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/1152241bb2e3f8f0d264a6fbc448b942dd50dd50a9e3de459f3f66d81705c889.jpg)

![16780893a7dcae582ecccffe39d5179edc2a74c0c0f2bba839feabea0849fcd1.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/16780893a7dcae582ecccffe39d5179edc2a74c0c0f2bba839feabea0849fcd1.jpg)

![398f01d79ec1921661d896dadd76f57199649b2cef99326e23d14e4262dd6e00.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/398f01d79ec1921661d896dadd76f57199649b2cef99326e23d14e4262dd6e00.jpg)

![4caf1eef3263cb11b9bca7124b58beda8c0b432e642b3f238eca459ad672eea1.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/4caf1eef3263cb11b9bca7124b58beda8c0b432e642b3f238eca459ad672eea1.jpg)

![50604f6f2a48d39a55d9bbd6bf65ead93985857ed005f766d833d0b09b91884c.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/50604f6f2a48d39a55d9bbd6bf65ead93985857ed005f766d833d0b09b91884c.jpg)

![6d532d6e4f1a2130300fb5817adfc0006fd987b7644c8361b459d01160e95b0c.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/6d532d6e4f1a2130300fb5817adfc0006fd987b7644c8361b459d01160e95b0c.jpg)

![78aa02751d2576b86a76301b06461cd4d20ef26e0e34e62146ffe4cd47300128.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/78aa02751d2576b86a76301b06461cd4d20ef26e0e34e62146ffe4cd47300128.jpg)

![79ec479098f81705a570fcad3615140285e9bd0713b8d97ef14f51d9a784e85b.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/79ec479098f81705a570fcad3615140285e9bd0713b8d97ef14f51d9a784e85b.jpg)

![8121ffbfb0580b4ef2afa7a2f978eb70141fef08c62e4bebd6c8869d81f02866.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/8121ffbfb0580b4ef2afa7a2f978eb70141fef08c62e4bebd6c8869d81f02866.jpg)

![ac72020f52b9710493b74f25f7d63fedd9eacf42cb02f96c21d6e89aadb3f788.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/ac72020f52b9710493b74f25f7d63fedd9eacf42cb02f96c21d6e89aadb3f788.jpg)

![aca31365adfc54a872da3357ab2054535cfbe9d064da76bf99ccdd3c36cda4d1.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/aca31365adfc54a872da3357ab2054535cfbe9d064da76bf99ccdd3c36cda4d1.jpg)

![bed8d6edfd73bb55cb896d34357c7173fd0c6ebf229e53bdfc9755180a8bc037.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/bed8d6edfd73bb55cb896d34357c7173fd0c6ebf229e53bdfc9755180a8bc037.jpg)

![c6f189fb445e14cc2741dda57770e6e58b72c8c6097ecf2a7b9ba9226985ae64.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/c6f189fb445e14cc2741dda57770e6e58b72c8c6097ecf2a7b9ba9226985ae64.jpg)

![cdb6a29e541ce28f373654e0370849f505678d608ad19023d9b2c3d47a2e436d.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/cdb6a29e541ce28f373654e0370849f505678d608ad19023d9b2c3d47a2e436d.jpg)

![ce829591a5015f00d1ccf23054bcfd1c6112ba92f54c9a41dbf7a44af0902e08.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/ce829591a5015f00d1ccf23054bcfd1c6112ba92f54c9a41dbf7a44af0902e08.jpg)

![cf6e5d310d47c0f80a7d54cfcbd116d2783f911bcc16458fb0e96cdb51570ae0.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/cf6e5d310d47c0f80a7d54cfcbd116d2783f911bcc16458fb0e96cdb51570ae0.jpg)

![da18544938681e5788fd4f8e68b88dbdbcb932b87c94dddeeaca68fbe836a085.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/da18544938681e5788fd4f8e68b88dbdbcb932b87c94dddeeaca68fbe836a085.jpg)

![e54fb76c64f08b5b085f38ed63673b0a03c644e88be70116d19f9dbd4c5aed18.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/e54fb76c64f08b5b085f38ed63673b0a03c644e88be70116d19f9dbd4c5aed18.jpg)

![fc195dfa9953329c78fc4f8ffb0d21c1709fac1f76a5b2ab93729c683b50cf6d.jpg](../icml_results/1498_VBA_%20Vector%20Bundle%20Attention%20for%20Intrinsically%20Geometric%20Representation%20Learning/tables/fc195dfa9953329c78fc4f8ffb0d21c1709fac1f76a5b2ab93729c683b50cf6d.jpg)

<a id="paper-1478"></a>

## Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization


### Images

![1cac516eb9336777afefe08755d104c889d80468c90c61e5c2e99666904199a9.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/1cac516eb9336777afefe08755d104c889d80468c90c61e5c2e99666904199a9.jpg)

![2c80ec30830a890e73dd2b78e0794f8059bd6e1d798649cb9a9b060edb77f15b.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/2c80ec30830a890e73dd2b78e0794f8059bd6e1d798649cb9a9b060edb77f15b.jpg)

![793a7aa87f55c31246019148c4c0102b4235726ca6a8213ac7e3f8c6e4adb0af.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/793a7aa87f55c31246019148c4c0102b4235726ca6a8213ac7e3f8c6e4adb0af.jpg)

![92c675e4d71a16f85b0399781aa6105ad9ecaf6182248c02e89dc2e9814e29d6.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/92c675e4d71a16f85b0399781aa6105ad9ecaf6182248c02e89dc2e9814e29d6.jpg)

![bc3d8237b34cb9118c91f07a911372d19cf0e6ee13d19b625cef46af0b7b2eba.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/bc3d8237b34cb9118c91f07a911372d19cf0e6ee13d19b625cef46af0b7b2eba.jpg)

![df447ca50ee768e6203123ae440ccb9bde05ff1672c6d0e4a728934e76605b21.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/df447ca50ee768e6203123ae440ccb9bde05ff1672c6d0e4a728934e76605b21.jpg)

![e94e5c935caa7fab6881fa3a90c5828fd97871fe1adfecd4b04947d3e7a2e544.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/e94e5c935caa7fab6881fa3a90c5828fd97871fe1adfecd4b04947d3e7a2e544.jpg)

![ead4efa4a21294ecb536fda3c5647854f7eeae0ed22c36e73a190a689b3f893f.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/ead4efa4a21294ecb536fda3c5647854f7eeae0ed22c36e73a190a689b3f893f.jpg)

![f75bf799de9471146b7e84937a93d7ef7fe7c94f49272c7c89fbe678f71d6e0e.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/images/f75bf799de9471146b7e84937a93d7ef7fe7c94f49272c7c89fbe678f71d6e0e.jpg)

### Tables

![12bc4234aacbedf5e6b5e087107ef9644e2d9b7fafdda747e4af5be51c9dd65f.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/12bc4234aacbedf5e6b5e087107ef9644e2d9b7fafdda747e4af5be51c9dd65f.jpg)

![222b89e9b2564adcc4dfcf2e426e685ba7883d9fa171906c93ffaea2d5749ecf.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/222b89e9b2564adcc4dfcf2e426e685ba7883d9fa171906c93ffaea2d5749ecf.jpg)

![2f1bafc918772dcd6415aef99cc9d005091493986009bcef61a645538b2cc43d.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/2f1bafc918772dcd6415aef99cc9d005091493986009bcef61a645538b2cc43d.jpg)

![5ba36be72a217734e3552de449f4f5d2d381c6075cc301dc3d959c47649da559.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/5ba36be72a217734e3552de449f4f5d2d381c6075cc301dc3d959c47649da559.jpg)

![5ea4258c7b9a5831913e63b6146de66242d3c91445e88726eed6a47635479687.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/5ea4258c7b9a5831913e63b6146de66242d3c91445e88726eed6a47635479687.jpg)

![80c9bbd0097be14771473d27795644e52e306e305d1c494e3d1d20cdf0e34cbe.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/80c9bbd0097be14771473d27795644e52e306e305d1c494e3d1d20cdf0e34cbe.jpg)

![8e2a22d3e70f3d1cf4f99bf8b1a4e5c76aacb2b91ab8055c40cdebfe0cf90047.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/8e2a22d3e70f3d1cf4f99bf8b1a4e5c76aacb2b91ab8055c40cdebfe0cf90047.jpg)

![b058c09034024326c5cd42f85c33b4ee97523f0c9586d6098e43e68fd6e389c2.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/b058c09034024326c5cd42f85c33b4ee97523f0c9586d6098e43e68fd6e389c2.jpg)

![be7281bbb520f976cccb91cc768d39c365f587cb9b5cd20e2153ffa7533a8a2c.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/be7281bbb520f976cccb91cc768d39c365f587cb9b5cd20e2153ffa7533a8a2c.jpg)

![d5796eaab7a6245588c66406476ac75e303d91634eee3cb8c517489fc79cc837.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/d5796eaab7a6245588c66406476ac75e303d91634eee3cb8c517489fc79cc837.jpg)

![fca4652fe33c9954396d44c4d711844097ee00ee647f8bbefe9606c2d512bb23.jpg](../icml_results/1499_Sparse%20Tokens%20Suffice_%20Jailbreaking%20Audio%20Language%20Models%20via%20Token-Aware%20Gradient%20Optimization/tables/fca4652fe33c9954396d44c4d711844097ee00ee647f8bbefe9606c2d512bb23.jpg)

<a id="paper-1479"></a>

## Real Data Lies: Unveiling and Closing the Quality Shortcut in Generalizable AI-Generated Video Detection


### Images

![0c554bfe983caf5117f1d38f9eb4530de10846bd61fd091b3de957215e738deb.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/0c554bfe983caf5117f1d38f9eb4530de10846bd61fd091b3de957215e738deb.jpg)

![0cb61f00220738b00c723099118c655ae865bc1b2d424521a259857c3df0f341.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/0cb61f00220738b00c723099118c655ae865bc1b2d424521a259857c3df0f341.jpg)

![0ff039a4b13a42d92fa6dd91a718304d229fdf0661d0653ace549d252954939a.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/0ff039a4b13a42d92fa6dd91a718304d229fdf0661d0653ace549d252954939a.jpg)

![1a6594961745ddc13872273378e10e9c7e30177e0e8d25708435add0d24b81c5.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/1a6594961745ddc13872273378e10e9c7e30177e0e8d25708435add0d24b81c5.jpg)

![38a7adf132dbd3892d62368d0e3795e1215ac09f318b13f253e80353d2973393.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/38a7adf132dbd3892d62368d0e3795e1215ac09f318b13f253e80353d2973393.jpg)

![42763f833e6d9add7158215d05d6e5c38203d407207341b26c6db2afc66dfcd3.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/42763f833e6d9add7158215d05d6e5c38203d407207341b26c6db2afc66dfcd3.jpg)

![443fe4758161cb2379e1aecb7a03e08720acc6d3d6e5fe8cfefb7e634a5f0dbc.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/443fe4758161cb2379e1aecb7a03e08720acc6d3d6e5fe8cfefb7e634a5f0dbc.jpg)

![5e2e1e8edd5ce7ebfa8fe849c663eb2489447f5afaadca434e08a558be0df0e8.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/5e2e1e8edd5ce7ebfa8fe849c663eb2489447f5afaadca434e08a558be0df0e8.jpg)

![6cfc794fa2193d735c74ccca12315187546d4ce80cb056e1aa72c43a94fa7d7b.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/6cfc794fa2193d735c74ccca12315187546d4ce80cb056e1aa72c43a94fa7d7b.jpg)

![8991379b4d94063e9df67360bdb9494da1438499cc393f6f7531f4e9c4d1d8ba.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/8991379b4d94063e9df67360bdb9494da1438499cc393f6f7531f4e9c4d1d8ba.jpg)

![b466d3dc4edcee98c052403b66e03d98e3ef1ff8444e1f1e0a010bdf92f4f0e5.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/b466d3dc4edcee98c052403b66e03d98e3ef1ff8444e1f1e0a010bdf92f4f0e5.jpg)

![bc4342098541ffa3010648c7565332443b26e5a70c4002f9ea7d1aba479aa4e7.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/bc4342098541ffa3010648c7565332443b26e5a70c4002f9ea7d1aba479aa4e7.jpg)

![e13aa38b5a015f14f75ca70e93b847f58c2860f42c03fc49c06ecfd24e96d556.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/e13aa38b5a015f14f75ca70e93b847f58c2860f42c03fc49c06ecfd24e96d556.jpg)

![edf51ae49f8a1680f2e5a3c30b00b9318a10275c073782b74d2c248f57d6f60a.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/edf51ae49f8a1680f2e5a3c30b00b9318a10275c073782b74d2c248f57d6f60a.jpg)

![ef387a041431b2f0905ce6cb7d3e08683e678ce7fa92ae5dd0767a94fd693ce1.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/ef387a041431b2f0905ce6cb7d3e08683e678ce7fa92ae5dd0767a94fd693ce1.jpg)

![fa957fab297b9bb860c2776b67177e12730769e6a61db092d13e889704218778.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/fa957fab297b9bb860c2776b67177e12730769e6a61db092d13e889704218778.jpg)

![ff396b344ccc4e00539d1a513c29af2a99bc5075b094059a85dc98eac1e8aec3.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/images/ff396b344ccc4e00539d1a513c29af2a99bc5075b094059a85dc98eac1e8aec3.jpg)

### Tables

![17c602998d1f578fbfcdda57f5b3a565c9cbc435c2a7369847a76272ce3175df.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/tables/17c602998d1f578fbfcdda57f5b3a565c9cbc435c2a7369847a76272ce3175df.jpg)

![4578ad730603b9a2973e13ac2d34fe0348a5fac6408e929acdce90de0cd539bd.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/tables/4578ad730603b9a2973e13ac2d34fe0348a5fac6408e929acdce90de0cd539bd.jpg)

![6473f5a440b0461e2e995035c23caece30de7cb583ba3a927b97ba6170db1e98.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/tables/6473f5a440b0461e2e995035c23caece30de7cb583ba3a927b97ba6170db1e98.jpg)

![ab3ab0053888652bd3abc4f4174fa3207af26a143597d9abee8f29720215001f.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/tables/ab3ab0053888652bd3abc4f4174fa3207af26a143597d9abee8f29720215001f.jpg)

![b284d2c5e3846d5b8449e49e85234c429c0fbcf70302599a15115dce43226013.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/tables/b284d2c5e3846d5b8449e49e85234c429c0fbcf70302599a15115dce43226013.jpg)

![ce1a27c4838c84c6e8086735d740faaf64236f42ef3055e905e673b6c6feaee0.jpg](../icml_results/1500_Real%20Data%20Lies_%20Unveiling%20and%20Closing%20the%20Quality%20Shortcut%20in%20Generalizable%20AI-Generated%20Video%20Detec/tables/ce1a27c4838c84c6e8086735d740faaf64236f42ef3055e905e673b6c6feaee0.jpg)

<a id="paper-1480"></a>

## ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization


### Images

![0b26d0620045e4b9405ae49b2d0cbca6d242921d31e12e9178338cdec6a79c53.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/0b26d0620045e4b9405ae49b2d0cbca6d242921d31e12e9178338cdec6a79c53.jpg)

![1324c62ea7dbfcc95e1f4a0adc376509291a7f4a0ae43abff619492d2db587ab.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/1324c62ea7dbfcc95e1f4a0adc376509291a7f4a0ae43abff619492d2db587ab.jpg)

![491138c81680db36e57f8f14d5cb69ff918bb5ca33eef3e3cd19888d80903691.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/491138c81680db36e57f8f14d5cb69ff918bb5ca33eef3e3cd19888d80903691.jpg)

![671989002352735e48abe73638cfe63e74ece8fca59bd30689bb07160cbde475.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/671989002352735e48abe73638cfe63e74ece8fca59bd30689bb07160cbde475.jpg)

![7b721be5c10df58b60c21fe19aa96c3c0bc3e12bbb3b4fa68f561cc51fb8b464.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/7b721be5c10df58b60c21fe19aa96c3c0bc3e12bbb3b4fa68f561cc51fb8b464.jpg)

![a32eac385e7ee683df1adfb4f48abd387f70725b5e260ba3a8a7b6b915c2d040.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/a32eac385e7ee683df1adfb4f48abd387f70725b5e260ba3a8a7b6b915c2d040.jpg)

![b8ddb26f5bc8e3b46d33c891ce99aaf3b89daa2d6780df4958cd6b81c299f38a.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/b8ddb26f5bc8e3b46d33c891ce99aaf3b89daa2d6780df4958cd6b81c299f38a.jpg)

![be610999e5ddf1e17d385084c00b7fcf80e08c9c944bb849d4a2633b3265846b.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/be610999e5ddf1e17d385084c00b7fcf80e08c9c944bb849d4a2633b3265846b.jpg)

![d01c682f1c0850040f4b832da8215d4e0316c19aa5fcbe182f6551d566f1c91e.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/d01c682f1c0850040f4b832da8215d4e0316c19aa5fcbe182f6551d566f1c91e.jpg)

![d40370e98b09f69deb1308722b04b2d4d3da5a9cd00d8aaa6c5e95748fa82c72.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/images/d40370e98b09f69deb1308722b04b2d4d3da5a9cd00d8aaa6c5e95748fa82c72.jpg)

### Tables

![02f53e7ba63ff76274a9458c0e6be935b15b5cb2459f6b2d7e0cf512e9a564f3.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/02f53e7ba63ff76274a9458c0e6be935b15b5cb2459f6b2d7e0cf512e9a564f3.jpg)

![069b18cdfa46d004d6b099e1a10dd6883d90914afdbcd0cf4ad3f8bdeb43b43a.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/069b18cdfa46d004d6b099e1a10dd6883d90914afdbcd0cf4ad3f8bdeb43b43a.jpg)

![20e644f0d6a65b2293c3cc34b44453f70ab21b3e71d4f8c05233136f2d03ca80.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/20e644f0d6a65b2293c3cc34b44453f70ab21b3e71d4f8c05233136f2d03ca80.jpg)

![2ca39242194a1037df4be06e17bf0cbb2bb7b66e8289cbb70ce75df49d8264d5.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/2ca39242194a1037df4be06e17bf0cbb2bb7b66e8289cbb70ce75df49d8264d5.jpg)

![3c0dca29326b11b8a665412ec9eb29bc66a27b3bb911a37740469c28d9661c5c.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/3c0dca29326b11b8a665412ec9eb29bc66a27b3bb911a37740469c28d9661c5c.jpg)

![42ab8e637facc9bb3b914cc966f55140b21f2f68c50385ed0dae8a33d4c2b81a.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/42ab8e637facc9bb3b914cc966f55140b21f2f68c50385ed0dae8a33d4c2b81a.jpg)

![63159a87330f190bc43f534ead34ada1ba363570300c8a45ce730c6d38cabf5a.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/63159a87330f190bc43f534ead34ada1ba363570300c8a45ce730c6d38cabf5a.jpg)

![821fddcaef450ba8d1e4cbaba23226f2bc8a0bc0805884b1a3b1895b71a3d693.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/821fddcaef450ba8d1e4cbaba23226f2bc8a0bc0805884b1a3b1895b71a3d693.jpg)

![b6ef9c60b75069b6286c8363bc6f23643e66c855963203ed296081ffaccddb19.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/b6ef9c60b75069b6286c8363bc6f23643e66c855963203ed296081ffaccddb19.jpg)

![cfdf352f3632b5c89c781bd25582371146bd17778cd4d714112284436a3ac149.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/cfdf352f3632b5c89c781bd25582371146bd17778cd4d714112284436a3ac149.jpg)

![d0101009921b09bd329cb5908f015f3e7ffe8214a2b0e60d0b674fac9bd9a154.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/d0101009921b09bd329cb5908f015f3e7ffe8214a2b0e60d0b674fac9bd9a154.jpg)

![ddb3332de4041a642c1cfedf4455825e629c83b05860f90dad59500e380e99d1.jpg](../icml_results/1501_ASAP_%20Exploiting%20the%20Satisficing%20Generalization%20Edge%20in%20Neural%20Combinatorial%20Optimization/tables/ddb3332de4041a642c1cfedf4455825e629c83b05860f90dad59500e380e99d1.jpg)

<a id="paper-1481"></a>

## BIOARC: Discovering Optimal Neural Architectures for Biological Foundation Models


### Images

![076a1d3672d1f778cad95c44e4bd1313a9f876fb1089eee0444f874c8929a56b.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/076a1d3672d1f778cad95c44e4bd1313a9f876fb1089eee0444f874c8929a56b.jpg)

![1cb0dac8e411e8f3070588ae20df8bdcbe336a68589d6c34f4ef1e2ac50523e6.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/1cb0dac8e411e8f3070588ae20df8bdcbe336a68589d6c34f4ef1e2ac50523e6.jpg)

![26b2b956b4ef5c9e21f286c1e3c3d53124b55e15b7a3df76f2b88b663ed27148.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/26b2b956b4ef5c9e21f286c1e3c3d53124b55e15b7a3df76f2b88b663ed27148.jpg)

![378563083cafe0ac823f6063613da1d697065083b7cea005db680ca6cfb5776b.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/378563083cafe0ac823f6063613da1d697065083b7cea005db680ca6cfb5776b.jpg)

![38950f42f81e4d7e642ca45c87a5aaa62edb244cee34c662cbd6e5e2d54bb791.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/38950f42f81e4d7e642ca45c87a5aaa62edb244cee34c662cbd6e5e2d54bb791.jpg)

![3a68b42348a968f9958e0b7391ac8f2503aa39bf9b70c6a7dd85d5d824ecb48f.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/3a68b42348a968f9958e0b7391ac8f2503aa39bf9b70c6a7dd85d5d824ecb48f.jpg)

![40e69771ecc31fa8773da168b9353522efaea0e2a76f6f3b61c7694790e75a4a.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/40e69771ecc31fa8773da168b9353522efaea0e2a76f6f3b61c7694790e75a4a.jpg)

![490bb8a38b4d7ae2b28cd2b1d24baa262af6857d94c8a08e2db2cac5d496b9c0.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/490bb8a38b4d7ae2b28cd2b1d24baa262af6857d94c8a08e2db2cac5d496b9c0.jpg)

![54406772f4a4419d93fda4654679a8b78e9e095c9ce49e455d36a7ea6a02640a.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/54406772f4a4419d93fda4654679a8b78e9e095c9ce49e455d36a7ea6a02640a.jpg)

![6a2fa4b262d43d919fc748069e062e74947eb1887b27806fed5082cf26ceefa7.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/6a2fa4b262d43d919fc748069e062e74947eb1887b27806fed5082cf26ceefa7.jpg)

![6c37d3a2212d555377fd2fa5f64400a5fd30998497372c390f16797198b16491.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/6c37d3a2212d555377fd2fa5f64400a5fd30998497372c390f16797198b16491.jpg)

![b1e0548f71c0ed789057a08a504803c1dd6124abbaf98319defa5e8c22a99e12.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/b1e0548f71c0ed789057a08a504803c1dd6124abbaf98319defa5e8c22a99e12.jpg)

![c4f3d49a101343533636c797d4179df2fa630e16e6071642b29a062b4d989570.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/c4f3d49a101343533636c797d4179df2fa630e16e6071642b29a062b4d989570.jpg)

![d7e5c1dc8873ee523d6f663cd3b66ed0072514b9756452e2a2a8e22c27c7e0f8.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/d7e5c1dc8873ee523d6f663cd3b66ed0072514b9756452e2a2a8e22c27c7e0f8.jpg)

![e8389942669f1e84768c9a0c839513f63c6978f4887f05776486ee16eedb7da4.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/e8389942669f1e84768c9a0c839513f63c6978f4887f05776486ee16eedb7da4.jpg)

![e9b3d6d03f734ac490d7bd2ab4f322b24b8aad6b1163e62d9a691d3ed57b95ba.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/e9b3d6d03f734ac490d7bd2ab4f322b24b8aad6b1163e62d9a691d3ed57b95ba.jpg)

![f29d4ac68e0efb624310c03000c3d3c1ea78da3d3a69331e591da5218ab9d7cd.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/f29d4ac68e0efb624310c03000c3d3c1ea78da3d3a69331e591da5218ab9d7cd.jpg)

![f4db560305576af9d5d329976cafcc23c0b6bb5673f0b5535213967e9aa46443.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/images/f4db560305576af9d5d329976cafcc23c0b6bb5673f0b5535213967e9aa46443.jpg)

### Tables

![058ad24f482d5c5f1d6f393cf99d6d3bb20addf1f22761436d245a9b62ef91b1.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/058ad24f482d5c5f1d6f393cf99d6d3bb20addf1f22761436d245a9b62ef91b1.jpg)

![15ab6ac949975ad40eca57f9c8c211b3ecd877ecc3ab072d09879618d8a4a909.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/15ab6ac949975ad40eca57f9c8c211b3ecd877ecc3ab072d09879618d8a4a909.jpg)

![20ddbc7ffa55ad249e03f726d36916baf47cad27cf3913e1b6868a01a57b9331.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/20ddbc7ffa55ad249e03f726d36916baf47cad27cf3913e1b6868a01a57b9331.jpg)

![43abab6cff3ca0b829f29e3c3938d177ea781fd948084a7d8d61d22468e23883.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/43abab6cff3ca0b829f29e3c3938d177ea781fd948084a7d8d61d22468e23883.jpg)

![4b2467328ccd068edecc634071e361ab7766223a4124f78219040ed583a0b104.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/4b2467328ccd068edecc634071e361ab7766223a4124f78219040ed583a0b104.jpg)

![6d6c62f5f58972f19763d336e123e9c4bbbe2a242e2fc0a88bd9d1a448533140.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/6d6c62f5f58972f19763d336e123e9c4bbbe2a242e2fc0a88bd9d1a448533140.jpg)

![7f194f5fdbb31590df4dbd7d62b441764431203820c91053fc36fa2ab3ebd80e.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/7f194f5fdbb31590df4dbd7d62b441764431203820c91053fc36fa2ab3ebd80e.jpg)

![806f2569a84405d5c3c770e11568fd030c78ac9448844c501c19f092537c53d1.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/806f2569a84405d5c3c770e11568fd030c78ac9448844c501c19f092537c53d1.jpg)

![8a1e3400fc9ecccc53175b95916ef90c811c837a9d7236c440400f52737520ac.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/8a1e3400fc9ecccc53175b95916ef90c811c837a9d7236c440400f52737520ac.jpg)

![8c786bb4b213bf3cea1ed97a5cfce07dd4ea598e552cdc4d2838665402b32552.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/8c786bb4b213bf3cea1ed97a5cfce07dd4ea598e552cdc4d2838665402b32552.jpg)

![a15b4ea612eacdfc495c127675f110a792a63b6dc364bceeaec1f24f86fda36f.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/a15b4ea612eacdfc495c127675f110a792a63b6dc364bceeaec1f24f86fda36f.jpg)

![a4455e0af886e07dd655d9451136ebcd977b154133f4288eab7deb4a62a4f9b6.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/a4455e0af886e07dd655d9451136ebcd977b154133f4288eab7deb4a62a4f9b6.jpg)

![bce3951721f1142c4386f689cb22db69922fe6dcd9ba7c839a1ea6b94d405428.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/bce3951721f1142c4386f689cb22db69922fe6dcd9ba7c839a1ea6b94d405428.jpg)

![c5875cf19505ed85e0a0e65b135b23a2ba687d620ce130e388194471df662cd4.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/c5875cf19505ed85e0a0e65b135b23a2ba687d620ce130e388194471df662cd4.jpg)

![ebe29913652a924d935a777aba6cbb612f58278f8821081acd0812e783f8a71d.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/ebe29913652a924d935a777aba6cbb612f58278f8821081acd0812e783f8a71d.jpg)

![ef2d4aaa36a4f78932c867420d2683999148724a5adf7f41903c71b3bc5179e1.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/ef2d4aaa36a4f78932c867420d2683999148724a5adf7f41903c71b3bc5179e1.jpg)

![fe507dab539197e8da4d87675632f82648b241f16b5402fb89e8064d923d2a05.jpg](../icml_results/1502_BIOARC_%20Discovering%20Optimal%20Neural%20Architectures%20for%20Biological%20Foundation%20Models/tables/fe507dab539197e8da4d87675632f82648b241f16b5402fb89e8064d923d2a05.jpg)

<a id="paper-1482"></a>

## Decoupled Low-Rank Adaptation for Robust Federated Fine-Tuning


### Images

![10a975f5887bb100a12e340ee27abeb485a02b951be69efc5dbe3433fe4264c6.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/images/10a975f5887bb100a12e340ee27abeb485a02b951be69efc5dbe3433fe4264c6.jpg)

![3f891a9bfa15e8f51e0745c7838de86015b1537e211463203645105d251085d7.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/images/3f891a9bfa15e8f51e0745c7838de86015b1537e211463203645105d251085d7.jpg)

![bdd6fd2b3280ab1aaa621a56e4e3cc5c56c086f90554be8573b3909b9ab727b0.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/images/bdd6fd2b3280ab1aaa621a56e4e3cc5c56c086f90554be8573b3909b9ab727b0.jpg)

### Tables

![1b4f60ee14dbe49670faa27e753e0b345c7a1b376301b548437fef5026f096b7.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/1b4f60ee14dbe49670faa27e753e0b345c7a1b376301b548437fef5026f096b7.jpg)

![3ae70c86847ce9651f9569dce63641063502efdd01206b922afa5b9a1ec19654.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/3ae70c86847ce9651f9569dce63641063502efdd01206b922afa5b9a1ec19654.jpg)

![4f0c2c5175899c06b71cb365e3398a18ea99040a0add259c2431106299462df3.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/4f0c2c5175899c06b71cb365e3398a18ea99040a0add259c2431106299462df3.jpg)

![4f0f260f31c59c4f1812e14cf5d7d28acb32661bcf1834e37af602153eada728.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/4f0f260f31c59c4f1812e14cf5d7d28acb32661bcf1834e37af602153eada728.jpg)

![5328cded1c398289cf616982549421a9865be28523442b29605cebcf7c96f0ce.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/5328cded1c398289cf616982549421a9865be28523442b29605cebcf7c96f0ce.jpg)

![5652f2ee59fa1a32acba3e5727550424ebf85dbdd50a528aab50cb531eb0fc75.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/5652f2ee59fa1a32acba3e5727550424ebf85dbdd50a528aab50cb531eb0fc75.jpg)

![57212b16447d856fdc1aebdec9a8f00ee3ffd0161850dbfe69dfcb34ddeccedf.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/57212b16447d856fdc1aebdec9a8f00ee3ffd0161850dbfe69dfcb34ddeccedf.jpg)

![75385ebd808b845367d80d906a234ca1242314fc36135866fa258082d5e744dd.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/75385ebd808b845367d80d906a234ca1242314fc36135866fa258082d5e744dd.jpg)

![901bf264ad0bd146df119f76f830f006b89d64c0739570744ab85c01f38425c4.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/901bf264ad0bd146df119f76f830f006b89d64c0739570744ab85c01f38425c4.jpg)

![94c5dd7a7b09e6fbfeec09e8f95bb8301c7191433ced38bc51d7e3961b126093.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/94c5dd7a7b09e6fbfeec09e8f95bb8301c7191433ced38bc51d7e3961b126093.jpg)

![96112f3a6b8e58354a5bdf7b2cf7d94765bde4aaa94beaf09572046df590a74c.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/96112f3a6b8e58354a5bdf7b2cf7d94765bde4aaa94beaf09572046df590a74c.jpg)

![a1388cfdaf0842d1ff4846dd9c82deb0eba623d5f831fb5528b9b615c8d38abe.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/a1388cfdaf0842d1ff4846dd9c82deb0eba623d5f831fb5528b9b615c8d38abe.jpg)

![a68feda9eb32c10251c7ee2032139e2bd43a7fc826937f983a408c584da59466.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/a68feda9eb32c10251c7ee2032139e2bd43a7fc826937f983a408c584da59466.jpg)

![c80148329bb898bf5ecdc14519c617940f63c88a08a63d1a62a25ea1a65e3adb.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/c80148329bb898bf5ecdc14519c617940f63c88a08a63d1a62a25ea1a65e3adb.jpg)

![e759ca9e6c882ca66134c3e1e51159ebb4eac408add40ec140deb3b74b92e0f6.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/e759ca9e6c882ca66134c3e1e51159ebb4eac408add40ec140deb3b74b92e0f6.jpg)

![f4a09366a50553f9d51cc54bc1f3d7967f2e02fe158c19e08453c73e3e2e579e.jpg](../icml_results/1503_Decoupled%20Low-Rank%20Adaptation%20for%20Robust%20Federated%20Fine-Tuning/tables/f4a09366a50553f9d51cc54bc1f3d7967f2e02fe158c19e08453c73e3e2e579e.jpg)

<a id="paper-1483"></a>

## Artificial Hippocampus Networks for Efficient Long-Context Modeling


### Images

![31129b414914a0333b47339d61427a02372d98f444eef156b06cbb0ca1498d37.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/31129b414914a0333b47339d61427a02372d98f444eef156b06cbb0ca1498d37.jpg)

![3a957834c91ea0353a23c6ae21d6ac03b42891f0fe80e49fbd6745be25510e64.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/3a957834c91ea0353a23c6ae21d6ac03b42891f0fe80e49fbd6745be25510e64.jpg)

![442d435e430e04dae600c9f923f45178ffb93fd52db7aff2e96f51bc2e277427.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/442d435e430e04dae600c9f923f45178ffb93fd52db7aff2e96f51bc2e277427.jpg)

![654ada4adc97651e2951ad9e510ebdc8d7350ceb6e7276fdcdde90e55e2593bd.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/654ada4adc97651e2951ad9e510ebdc8d7350ceb6e7276fdcdde90e55e2593bd.jpg)

![80701c3e34f40d077f982852b20d3a9c73c4f181027e2ab1b4ace303a3ba5b50.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/80701c3e34f40d077f982852b20d3a9c73c4f181027e2ab1b4ace303a3ba5b50.jpg)

![a5630219d632f4aeb9ad78090d563c392e464baed69b260ac7c9445e3a0b96f5.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/a5630219d632f4aeb9ad78090d563c392e464baed69b260ac7c9445e3a0b96f5.jpg)

![ab9471f90ee44a13c52751f129c546be128b115b8d310a895c5b9927cfec40e5.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/ab9471f90ee44a13c52751f129c546be128b115b8d310a895c5b9927cfec40e5.jpg)

![fd079350ff3c3f8f79b29b0ada99d76c9aa5d74bb1b1244c7302ee51910519b8.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/images/fd079350ff3c3f8f79b29b0ada99d76c9aa5d74bb1b1244c7302ee51910519b8.jpg)

### Tables

![06da9795adf8ef1912ad97ccf8fb4e8d720ba14dda80c92d0d98364de03abde2.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/06da9795adf8ef1912ad97ccf8fb4e8d720ba14dda80c92d0d98364de03abde2.jpg)

![1df0c5705f34c74f7922e6720ac46f7182548600795cdc22e5aff4381aeb99a3.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/1df0c5705f34c74f7922e6720ac46f7182548600795cdc22e5aff4381aeb99a3.jpg)

![4794c5c2857e5139688e10f1235a9bfb0d9769a0ec501a604c7da99d348baf1d.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/4794c5c2857e5139688e10f1235a9bfb0d9769a0ec501a604c7da99d348baf1d.jpg)

![66d4746b2d26f6a6556e55a07fc78b5ed235d2765c9fd09703c86d1637cbd873.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/66d4746b2d26f6a6556e55a07fc78b5ed235d2765c9fd09703c86d1637cbd873.jpg)

![74755898b544d82d553dfb0ce8087dd84ed5c4aa9d6bf2ffb8d39c76795bfc0f.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/74755898b544d82d553dfb0ce8087dd84ed5c4aa9d6bf2ffb8d39c76795bfc0f.jpg)

![80a7bc0d494374eff7b0874a8ceafd04b66507756f7d63b15db0bae9132f56df.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/80a7bc0d494374eff7b0874a8ceafd04b66507756f7d63b15db0bae9132f56df.jpg)

![81ddac863d2bd5f2d141140502b58a698cc69c49fd7c9561f1e1324a834e9086.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/81ddac863d2bd5f2d141140502b58a698cc69c49fd7c9561f1e1324a834e9086.jpg)

![9d77f6ee6a0cd07921715abe31219e33ad1867298e807ed4e2fa29ab57fe2f42.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/9d77f6ee6a0cd07921715abe31219e33ad1867298e807ed4e2fa29ab57fe2f42.jpg)

![af84f115c08b3dfcbe2f4dce32dcdcf7f80f45a9e2036eb8dbe90977734fa816.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/af84f115c08b3dfcbe2f4dce32dcdcf7f80f45a9e2036eb8dbe90977734fa816.jpg)

![b7b4daf2cb1a8e06f20dafe26d245e90d5ec0b27a6dfbc15a53f0d67c34a3e60.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/b7b4daf2cb1a8e06f20dafe26d245e90d5ec0b27a6dfbc15a53f0d67c34a3e60.jpg)

![b981e65ebc64274f890a5132cb85398b198297066bb37d802c3999f47754a241.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/b981e65ebc64274f890a5132cb85398b198297066bb37d802c3999f47754a241.jpg)

![ca78fc4b2d1aa022d7ddd306a863a2b0fab7c5a581ba00ecb574ebf0e19455a9.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/ca78fc4b2d1aa022d7ddd306a863a2b0fab7c5a581ba00ecb574ebf0e19455a9.jpg)

![d1ecea1108f811de72338559c36b624134ca65f96ca7e7dde473bba02b16c694.jpg](../icml_results/1504_Artificial%20Hippocampus%20Networks%20for%20Efficient%20Long-Context%20Modeling/tables/d1ecea1108f811de72338559c36b624134ca65f96ca7e7dde473bba02b16c694.jpg)

<a id="paper-1484"></a>

## Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition


### Images

![2cd6a3c00020a055feb6df2a0c15fbf458228d358f8b6a5f92cc555d2d1d2f92.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/images/2cd6a3c00020a055feb6df2a0c15fbf458228d358f8b6a5f92cc555d2d1d2f92.jpg)

![a377584180202ef8f5847b4bf62f5062186a12c6500ec6d08832fd5e551fdd18.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/images/a377584180202ef8f5847b4bf62f5062186a12c6500ec6d08832fd5e551fdd18.jpg)

![a4106d1a6f9dbb0902cab5d84e9013f804a9229d033fb258823f0fbdd6339c10.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/images/a4106d1a6f9dbb0902cab5d84e9013f804a9229d033fb258823f0fbdd6339c10.jpg)

![d8591bf27432049a2a649880ddaff55b3ba08ea9b0715691a3604dea75d4dd2b.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/images/d8591bf27432049a2a649880ddaff55b3ba08ea9b0715691a3604dea75d4dd2b.jpg)

![e80b899898091a6c1415652ad6a6c13f4ce3316aa012f58da62198abc1483db9.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/images/e80b899898091a6c1415652ad6a6c13f4ce3316aa012f58da62198abc1483db9.jpg)

### Tables

![046357a26ba8a948a59f170e0a88e273581f4619a05f080b4112af7cb7e4cd5e.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/046357a26ba8a948a59f170e0a88e273581f4619a05f080b4112af7cb7e4cd5e.jpg)

![0b023f4eb4a6d60190330f658b61aa2a792675bdf14e6ea591f6a87158158fd7.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/0b023f4eb4a6d60190330f658b61aa2a792675bdf14e6ea591f6a87158158fd7.jpg)

![0db74254baae27d26d3cad434e44f1a8d9e43066ab312dfa620f2d996a48fd2c.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/0db74254baae27d26d3cad434e44f1a8d9e43066ab312dfa620f2d996a48fd2c.jpg)

![11ae6bcb6ffe191fb8faf0f94d1b0f98c1e38d91ff8922161d1908d03b1892ed.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/11ae6bcb6ffe191fb8faf0f94d1b0f98c1e38d91ff8922161d1908d03b1892ed.jpg)

![246e7341b24efc856904eb1d222d25df67437eee6d55d5ec44c2befbdff609e6.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/246e7341b24efc856904eb1d222d25df67437eee6d55d5ec44c2befbdff609e6.jpg)

![26d4cf3adb9ff5ae40f43e42c5bfb6fa487d6dc6a26aa9713fc4bd8af1443b5a.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/26d4cf3adb9ff5ae40f43e42c5bfb6fa487d6dc6a26aa9713fc4bd8af1443b5a.jpg)

![2b1abad8c32f8d731552df8e48ea1cba7f1c948ea46a3f4b4af643c576d4c111.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/2b1abad8c32f8d731552df8e48ea1cba7f1c948ea46a3f4b4af643c576d4c111.jpg)

![3cb0ea20e0f358288675a95c45072dae089234a279bf89d791b712f16a144f50.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/3cb0ea20e0f358288675a95c45072dae089234a279bf89d791b712f16a144f50.jpg)

![4820fa22d857346a6776510b9fcc5ed6717ad46a724b39350cf44cb0e81a00cd.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/4820fa22d857346a6776510b9fcc5ed6717ad46a724b39350cf44cb0e81a00cd.jpg)

![64d5f9d6ff64a9fae5da324d4ebba4f742ab3ae8930f5a81cd230ecf3d41ca83.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/64d5f9d6ff64a9fae5da324d4ebba4f742ab3ae8930f5a81cd230ecf3d41ca83.jpg)

![7d0679578f3c1f835f5df2cffeb8fb1ec6822b0af56197bbd47d3621d1b2cae7.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/7d0679578f3c1f835f5df2cffeb8fb1ec6822b0af56197bbd47d3621d1b2cae7.jpg)

![7d813569bcbd25d6534cb8aa3ebd75a34874d193cdfd14245e7589cf457ccdd0.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/7d813569bcbd25d6534cb8aa3ebd75a34874d193cdfd14245e7589cf457ccdd0.jpg)

![9245e891e2ccc403eb73d29924af31e14493e684788c50d7b897645968a719f5.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/9245e891e2ccc403eb73d29924af31e14493e684788c50d7b897645968a719f5.jpg)

![a43970b33fba17b34b7aaafde5f212d3de2ccb2585a36a2087a28ed585630d4e.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/a43970b33fba17b34b7aaafde5f212d3de2ccb2585a36a2087a28ed585630d4e.jpg)

![a9cc6357dafed40c81ae368773e4e07a734f012b14223ec8a59629eb95c860f9.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/a9cc6357dafed40c81ae368773e4e07a734f012b14223ec8a59629eb95c860f9.jpg)

![db153f0e5ac82bc921e241057fa493199c7fb915d450ed0679200eba718631da.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/db153f0e5ac82bc921e241057fa493199c7fb915d450ed0679200eba718631da.jpg)

![db8ef67ebe8ea40da1e951e3e92235a8cece95bbeb08663d6e7fe78e5344d004.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/db8ef67ebe8ea40da1e951e3e92235a8cece95bbeb08663d6e7fe78e5344d004.jpg)

![e1ebdc7acb4ce548c64aad71ddf2b01bdbbb6b4bdf3e850f48bef74d6496ddbd.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/e1ebdc7acb4ce548c64aad71ddf2b01bdbbb6b4bdf3e850f48bef74d6496ddbd.jpg)

![e7940a702b3f2fcf7e677b2630a027c32a762d642b2e8b964dfe237b9281b6e7.jpg](../icml_results/1505_Towards%20Understanding%20Modality%20Interaction%20in%20Multimodal%20Language%20Models%20via%20Partial%20Information%20Dec/tables/e7940a702b3f2fcf7e677b2630a027c32a762d642b2e8b964dfe237b9281b6e7.jpg)

<a id="paper-1485"></a>

## StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation

### Images

![4a80ee1e9595b4b38eb7ffc2aae150de7fbf9d034c5e55cb21a0c5083c1b4371.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/4a80ee1e9595b4b38eb7ffc2aae150de7fbf9d034c5e55cb21a0c5083c1b4371.jpg)

![5e3e3ca45e19ad8de37def048daf27925c60d731529f3948e172e28e3b68fc4c.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/5e3e3ca45e19ad8de37def048daf27925c60d731529f3948e172e28e3b68fc4c.jpg)

![615550ed01c8ff41a9cd524a865ad12049e73cb044aad6e34494451001d6f830.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/615550ed01c8ff41a9cd524a865ad12049e73cb044aad6e34494451001d6f830.jpg)

![6ea5f5ff599c33abff3e2a945c366014b3a2e3e12fdd03676ee62067512cf2fa.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/6ea5f5ff599c33abff3e2a945c366014b3a2e3e12fdd03676ee62067512cf2fa.jpg)

![6efb7f959a92d76fb222e68a1024f0374bd599850092fa6276946e3bc96ff09b.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/6efb7f959a92d76fb222e68a1024f0374bd599850092fa6276946e3bc96ff09b.jpg)

![7b6abe670a5ad30c178a9469af7e712c55ec3a37ac0788e96e4141c3e50adea2.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/7b6abe670a5ad30c178a9469af7e712c55ec3a37ac0788e96e4141c3e50adea2.jpg)

![8089003aad0d6fc20468471e65190265d70b587a6b8047a53aa952de80cf3329.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/8089003aad0d6fc20468471e65190265d70b587a6b8047a53aa952de80cf3329.jpg)

![aa68606d7812121dbc2f14a6e2d99b5168a4f121392023fa49335e979df9a24b.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/aa68606d7812121dbc2f14a6e2d99b5168a4f121392023fa49335e979df9a24b.jpg)

### Tables

![3b895f9cbce488ead7dc2ba6514bd8927a9820b3734972777f82994e178b602c.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/tables/3b895f9cbce488ead7dc2ba6514bd8927a9820b3734972777f82994e178b602c.jpg)

![6c33c72abaf9571a40acd3fab8209e246a676a4d59054fc2f7d5afc04e126df5.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/tables/6c33c72abaf9571a40acd3fab8209e246a676a4d59054fc2f7d5afc04e126df5.jpg)
