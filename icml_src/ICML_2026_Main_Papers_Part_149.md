# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 149 of 197

## 目录 (Table of Contents)

1. [Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures](#paper-4885)
2. [FedEBA+: Towards Fair and Effective Federated Learning via Entropy-Based Model](#paper-4886)
3. [Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](#paper-4887)
4. [Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future](#paper-4888)
5. [Steady-State Behavior of Constant-Stepsize Stochastic Approximation: Gaussian Approximation and Tail Bounds](#paper-4889)
6. [Agentic Model Predictive Questioning Control in Visual Design](#paper-4890)
7. [Reward Modeling from Natural Language Human Feedback](#paper-4891)
8. [MuCO: Generative Peptide Cyclization Empowered by Multi-stage Conformation Optimization](#paper-4892)
9. [Navigating Massive Visual Context in Retrieval-Augmented Generation via Multimodal Memory Graph](#paper-4893)
10. [Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](#paper-4894)
11. [Sparser Block-Sparse Attention via Token Permutation](#paper-4895)
12. [GradPower: Powering Gradients for Faster Language Model Pre-Training](#paper-4896)
13. [Breaking Manifold Continuity: Vector Quantized Modeling for Real-Centric Deepfake Detection](#paper-4897)
14. [WorldCompass: Reinforcement Learning for Long-Horizon World Models](#paper-4898)
15. [REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation](#paper-4899)
16. [SL-VC: A Benchmark and Automated Framework for Separation Logic Verification Condition Proving](#paper-4900)
17. [ScaleErasure: Inference-Time Minimal Intervention for Precise Concept Erasure in Next-Scale Autoregressive Image Generation](#paper-4901)
18. [Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](#paper-4902)
19. [Mixing Configurations for Downstream Prediction](#paper-4903)
20. [Beyond Sample-Level Forgetting: Improving Reliability in Multimodal Unlearning](#paper-4904)
21. [Adaptive Querying with AI Persona Priors](#paper-4905)
22. [Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning](#paper-4906)
23. [A Progressive Evidence Localization Framework Based on Wasserstein Gradient Flows for Document Visual Question Answering](#paper-4907)
24. [Keep It in Mind: User Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams](#paper-4908)
25. [RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System](#paper-4909)
26. [On the Entropy Dynamics in Reinforcement Fine-Tuning of Large Language Models](#paper-4910)
27. [Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](#paper-4911)
28. [SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning](#paper-4912)
29. [Variable Clustering via Distributionally Robust Nodewise Regression](#paper-4913)
30. [DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training](#paper-4914)
31. [Milestone-Guided Policy Learning for Long-Horizon Language Agents](#paper-4915)
32. [DR-MMSearchAgent: Deepening Reasoning in Multimodal Search Agents](#paper-4916)
33. [Optimizing Diversity and Quality through Base-Aligned Model Collaboration](#paper-4917)

---


<a id="paper-4885"></a>

## Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures

### Images

![18508e031a384749ecc84aa090fa2592aceb4184d93325759666127265e465cc.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/18508e031a384749ecc84aa090fa2592aceb4184d93325759666127265e465cc.jpg)

![54549a1bcb41eab2b4362201e390ed42ba8f75231f768a9ba68edadd103bff00.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/54549a1bcb41eab2b4362201e390ed42ba8f75231f768a9ba68edadd103bff00.jpg)

![558d6d8b1960390d0c7b655b337428f868c5b72e3594fc3a8cc07f774f92cd0e.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/558d6d8b1960390d0c7b655b337428f868c5b72e3594fc3a8cc07f774f92cd0e.jpg)

![6b4b99e02e2089d463c3f1232abf45b71620facae60d92d402633b99829f4665.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/6b4b99e02e2089d463c3f1232abf45b71620facae60d92d402633b99829f4665.jpg)

![8892843852c3cfa93949ac351e85a62f0eeda63eaab1945be51f48274ff5d7aa.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/8892843852c3cfa93949ac351e85a62f0eeda63eaab1945be51f48274ff5d7aa.jpg)

![c163953c8efd1871b9280298a9997afd622c2f1c798f255dfae20ed104ee6375.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/c163953c8efd1871b9280298a9997afd622c2f1c798f255dfae20ed104ee6375.jpg)

![fe2cac3464def0ded518bf86f8c42d9fe61c652670ccd87035c81774b48c77a7.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/images/fe2cac3464def0ded518bf86f8c42d9fe61c652670ccd87035c81774b48c77a7.jpg)

### Tables

![064aae957323327485ac57baa40a5bd3cfd361eea539d61ebc654897f296a869.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/064aae957323327485ac57baa40a5bd3cfd361eea539d61ebc654897f296a869.jpg)

![0b4a19ac3e61d7edcad47fdb6dbb377d17775937fb65760957b2770586e539c4.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/0b4a19ac3e61d7edcad47fdb6dbb377d17775937fb65760957b2770586e539c4.jpg)

![17aae16ba205c2aed7fb84097fe638456afe90f1f7f833686a5407e53b1f8af0.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/17aae16ba205c2aed7fb84097fe638456afe90f1f7f833686a5407e53b1f8af0.jpg)

![1bf4e6ddd6fe69143552b2fd4d1d4366b70012eb9eda5d2e4d4ba01419072267.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/1bf4e6ddd6fe69143552b2fd4d1d4366b70012eb9eda5d2e4d4ba01419072267.jpg)

![21ec6e0b142905b9e0b90a65bf37f2610a79411aaf57c3068fea20b4c86070c6.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/21ec6e0b142905b9e0b90a65bf37f2610a79411aaf57c3068fea20b4c86070c6.jpg)

![2332d948e96fe0f322d3355236f8d3a59f26eeb9f2ac7efdb0f254e8ff219c48.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/2332d948e96fe0f322d3355236f8d3a59f26eeb9f2ac7efdb0f254e8ff219c48.jpg)

![3313dee50d67113ec42e3c7fd55e54600a5472b35f143a17cfae3a5422ff16ee.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/3313dee50d67113ec42e3c7fd55e54600a5472b35f143a17cfae3a5422ff16ee.jpg)

![407ebba1d9bfedfb5b3a477b2e889a04fb8947ebea1eca58063007742284c03d.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/407ebba1d9bfedfb5b3a477b2e889a04fb8947ebea1eca58063007742284c03d.jpg)

![48a43ab9f55077a2e0e77d479a4527ff97fcfb027215144051417d36b381440d.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/48a43ab9f55077a2e0e77d479a4527ff97fcfb027215144051417d36b381440d.jpg)

![6a3439bb06981f3256d1536b354011b25f4f9c459831df1c5e9f8f7183a414c7.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/6a3439bb06981f3256d1536b354011b25f4f9c459831df1c5e9f8f7183a414c7.jpg)

![794d2cab16636d4cde1d76442c06a1f1f04e41d0533adf92093232703fac8308.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/794d2cab16636d4cde1d76442c06a1f1f04e41d0533adf92093232703fac8308.jpg)

![830a81880248f96d126b2edfc89ff62085e0daabdaa308b89f7c2acd6a25cc3d.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/830a81880248f96d126b2edfc89ff62085e0daabdaa308b89f7c2acd6a25cc3d.jpg)

![956cc34b034a2e733592a2696a887b9b1684583766c2f2189e1d8ca7bf6c52d0.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/956cc34b034a2e733592a2696a887b9b1684583766c2f2189e1d8ca7bf6c52d0.jpg)

![b7c2a141309bd30a78f86855a060a28a08d900aca34d9fbe7d2ae16a90f09467.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/b7c2a141309bd30a78f86855a060a28a08d900aca34d9fbe7d2ae16a90f09467.jpg)

![b9620188f9b722252788217a4a4752db5a990c5be20817d24e3e8bc53719e07d.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/b9620188f9b722252788217a4a4752db5a990c5be20817d24e3e8bc53719e07d.jpg)

![bddc8b5426a2b31bed0d45ea18df1856d0ddcc16717fe3c40547887f3372305f.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/bddc8b5426a2b31bed0d45ea18df1856d0ddcc16717fe3c40547887f3372305f.jpg)

![c01a88dacd1ca9532c231a77eefd6ee7e9611f6bc36dacdf8c282087d36ad97e.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/c01a88dacd1ca9532c231a77eefd6ee7e9611f6bc36dacdf8c282087d36ad97e.jpg)

![c1ae522dbca66a0077131b9f946cce926236180c14a0281001e619ef092d2e09.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/c1ae522dbca66a0077131b9f946cce926236180c14a0281001e619ef092d2e09.jpg)

![d83dae9ef973b5607242c55fec1a43847df09485633bbdef69910b75e2150f1c.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/d83dae9ef973b5607242c55fec1a43847df09485633bbdef69910b75e2150f1c.jpg)

![dc6631f57653d13b7de55fe854539822debd1125bd599445fd10edd58890f379.jpg](../icml_results/4932_WFR-MFM_%20One-Step%20Inference%20for%20Dynamic%20Unbalanced%20Optimal%20Transport/tables/dc6631f57653d13b7de55fe854539822debd1125bd599445fd10edd58890f379.jpg)

## Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures


### Images

![299b482b79483ac0d05f7918a7e0769d35926725d5f616e20a027e42dbe288f7.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/299b482b79483ac0d05f7918a7e0769d35926725d5f616e20a027e42dbe288f7.jpg)

![3c4bae3a4399018491cb6f7505a46aa604a1103977418b04377d18f0ceb9109e.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/3c4bae3a4399018491cb6f7505a46aa604a1103977418b04377d18f0ceb9109e.jpg)

![47066dc6d9f8e48d06ddfc3de02d1a6c3674a9659c08b6c6c5579ee756f713b8.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/47066dc6d9f8e48d06ddfc3de02d1a6c3674a9659c08b6c6c5579ee756f713b8.jpg)

![5e4d96e191b1e189fc07297251ba331160b69eae7adbcaeba444faafa0f5ada3.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/5e4d96e191b1e189fc07297251ba331160b69eae7adbcaeba444faafa0f5ada3.jpg)

![ae4fd4f9bfd9b9a58ae5f17f02f4186cc4e2d17e9e26d54adf728e66b2e0a241.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/ae4fd4f9bfd9b9a58ae5f17f02f4186cc4e2d17e9e26d54adf728e66b2e0a241.jpg)

![af9312f2bd017ca959c02bdf75883b62c079d84c1e0219eaf0246abe01e96d51.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/af9312f2bd017ca959c02bdf75883b62c079d84c1e0219eaf0246abe01e96d51.jpg)

![b1d91b70b77e1f30748c41d14937f17f8d3ddd3bb995d6d879bc24dd2264be5b.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/b1d91b70b77e1f30748c41d14937f17f8d3ddd3bb995d6d879bc24dd2264be5b.jpg)

![b30fb1fc0a613d9adc711b4803d4ca4c8a69b2804f2206887e4bb389a48e47ef.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/b30fb1fc0a613d9adc711b4803d4ca4c8a69b2804f2206887e4bb389a48e47ef.jpg)

![b91216b45a48f343a75b43d01597f4bf7c216d7bf98f21569f979757dba80066.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/b91216b45a48f343a75b43d01597f4bf7c216d7bf98f21569f979757dba80066.jpg)

![c859ad10fade552b6d510f970eb222dd2e0420f3d5457a24099e9accca1d971e.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/c859ad10fade552b6d510f970eb222dd2e0420f3d5457a24099e9accca1d971e.jpg)

![ca8d6672712799d74f26bb3d9dbfba78206f94ab5d4693e1760e409256b869e6.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/ca8d6672712799d74f26bb3d9dbfba78206f94ab5d4693e1760e409256b869e6.jpg)

![d574b734a56902c6e13b14e02fa0fa39ce44af12524e06909059f3f9f4147a2f.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/d574b734a56902c6e13b14e02fa0fa39ce44af12524e06909059f3f9f4147a2f.jpg)

![d6cc85db1ae5e4de0a5f1c89aa43ce00f57e9a2ef9371e1220b7534639358ea8.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/images/d6cc85db1ae5e4de0a5f1c89aa43ce00f57e9a2ef9371e1220b7534639358ea8.jpg)

### Tables

![48fbce37269ebce69ee304d8b2f0e7c69e3f6a3433583941f7f2c5e712a3c495.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/48fbce37269ebce69ee304d8b2f0e7c69e3f6a3433583941f7f2c5e712a3c495.jpg)

![77ccfdcf6b636b0d2dc48cc479acadc95cbf2eaeca64255265425fac7cbdd945.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/77ccfdcf6b636b0d2dc48cc479acadc95cbf2eaeca64255265425fac7cbdd945.jpg)

![9361062c95de28f66f54bd5ca2b7f81514bbd89d399aa0c52fe21da8cddb1def.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/9361062c95de28f66f54bd5ca2b7f81514bbd89d399aa0c52fe21da8cddb1def.jpg)

![982284905c22bb1c57fbe0aee2fd80c81a9af28207a6c953270ba1e41bbb7521.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/982284905c22bb1c57fbe0aee2fd80c81a9af28207a6c953270ba1e41bbb7521.jpg)

![a9c099ba7c0175c1c0b27ef9e4db650537b6a818b47be340fd54ae2552d27b4d.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/a9c099ba7c0175c1c0b27ef9e4db650537b6a818b47be340fd54ae2552d27b4d.jpg)

![eddd4ed0b786620b5b4517bb8e5cf034854df08e53bd9dda9027785c0f9c3379.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/eddd4ed0b786620b5b4517bb8e5cf034854df08e53bd9dda9027785c0f9c3379.jpg)

![fa6600bfef651473dcc7e3518230592d425267f4355dbc08590291d80ae87bb0.jpg](../icml_results/4933_Simple%20Approximation%20and%20Derivative%20Free%20Inference-Time%20Scaling%20for%20Diffusion%20Models%20via%20Sequential%20/tables/fa6600bfef651473dcc7e3518230592d425267f4355dbc08590291d80ae87bb0.jpg)

<a id="paper-4886"></a>

## FedEBA+: Towards Fair and Effective Federated Learning via Entropy-Based Model


### Images

![0fe27a23025bd8ed98be6e34d6ed37e9162ae975dc7bfdf782cf47b3c070bcf9.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/images/0fe27a23025bd8ed98be6e34d6ed37e9162ae975dc7bfdf782cf47b3c070bcf9.jpg)

![2e1056f23529e8496938864870c1c839ad754f604cd7cdcf8f904fecfd556ce1.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/images/2e1056f23529e8496938864870c1c839ad754f604cd7cdcf8f904fecfd556ce1.jpg)

![52b9a0ec6ee870b16bc14a02ecd5e560a6b727dcd346813f1ee4a9d75200a2b4.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/images/52b9a0ec6ee870b16bc14a02ecd5e560a6b727dcd346813f1ee4a9d75200a2b4.jpg)

![b0edc18a45b4c26a242d73f65a26ec662fc64c65b2cfb264bd26ab69204902b9.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/images/b0edc18a45b4c26a242d73f65a26ec662fc64c65b2cfb264bd26ab69204902b9.jpg)

![c699c6b600946dc2ef76119cf5a21532a5846fa2b367a50ecd80ce28a9a5373f.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/images/c699c6b600946dc2ef76119cf5a21532a5846fa2b367a50ecd80ce28a9a5373f.jpg)

![dd18a61a3186baa53c71d41559e710a62bb49a2a0f3b94cb75dabc8701dc8218.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/images/dd18a61a3186baa53c71d41559e710a62bb49a2a0f3b94cb75dabc8701dc8218.jpg)

### Tables

![03e1ce567483e2b921975f1411b907cdae2a866a9045fe19c1b3abf93c8605b8.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/03e1ce567483e2b921975f1411b907cdae2a866a9045fe19c1b3abf93c8605b8.jpg)

![056b56b350017bc8f10033c47037987e6419f18a6a76291f5e285d619e786c79.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/056b56b350017bc8f10033c47037987e6419f18a6a76291f5e285d619e786c79.jpg)

![0fe7ee5cceb7426d55bb16fdd63b8bc4f51d6912ea02f71f94d5da9ffdbc1d29.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/0fe7ee5cceb7426d55bb16fdd63b8bc4f51d6912ea02f71f94d5da9ffdbc1d29.jpg)

![1b05e482e449124619485955e95e258bd458a027cbd2e134f049e061f586eb9d.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/1b05e482e449124619485955e95e258bd458a027cbd2e134f049e061f586eb9d.jpg)

![258322cb8dbcdda04cbba0bc5ba33048e1e1c1e769ff5d56a7a8e6ac7ed2432d.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/258322cb8dbcdda04cbba0bc5ba33048e1e1c1e769ff5d56a7a8e6ac7ed2432d.jpg)

![26f003d1e970c6bc39f653537d3b7fa9b7e304bcc70b056ebf5277ec30db2b13.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/26f003d1e970c6bc39f653537d3b7fa9b7e304bcc70b056ebf5277ec30db2b13.jpg)

![27c1315f1757d1c05989ce0bdfc9fa20e1131987464b5bc756985ac68ab62db4.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/27c1315f1757d1c05989ce0bdfc9fa20e1131987464b5bc756985ac68ab62db4.jpg)

![309a9df9d91668dc31c110620b42e81d0181be901e5d9f87727e94b20c4adec0.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/309a9df9d91668dc31c110620b42e81d0181be901e5d9f87727e94b20c4adec0.jpg)

![32355d73ccee0cde18e2c669cd97f8b6377d036ba824bb2d268012b701225072.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/32355d73ccee0cde18e2c669cd97f8b6377d036ba824bb2d268012b701225072.jpg)

![4614dc8b487df8920a833e9affe6998ce31adf69b523ce5b40d970b44c21e6ce.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/4614dc8b487df8920a833e9affe6998ce31adf69b523ce5b40d970b44c21e6ce.jpg)

![576eaee875adccfedc453dbe3a51ca86d75f1dac7abedcbca2febd955b5c7e58.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/576eaee875adccfedc453dbe3a51ca86d75f1dac7abedcbca2febd955b5c7e58.jpg)

![589dbb43c717280b96583935a5ccf70f88b4c06a8b21827886891cd2fd93758e.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/589dbb43c717280b96583935a5ccf70f88b4c06a8b21827886891cd2fd93758e.jpg)

![a04c5a33a631f734c9ac47403ee454f5281119c1eb569124e333f212414c5be8.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/a04c5a33a631f734c9ac47403ee454f5281119c1eb569124e333f212414c5be8.jpg)

![a9eab71c4d1a0a35b755682e3b83afb84a97845d3417da79e3fdc47f002fe997.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/a9eab71c4d1a0a35b755682e3b83afb84a97845d3417da79e3fdc47f002fe997.jpg)

![aff30b8d499c2111cbc4cd9b92ec083004880f1513b2e746a3c897d4ba73dcb5.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/aff30b8d499c2111cbc4cd9b92ec083004880f1513b2e746a3c897d4ba73dcb5.jpg)

![bf0d9bd0e9c7952d620dba1e455b6290ed96b1b9b15e8c9f25e8b34aae4329ef.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/bf0d9bd0e9c7952d620dba1e455b6290ed96b1b9b15e8c9f25e8b34aae4329ef.jpg)

![ceae1e7e6a65228a93405dd89b3372dbe406f27df4b8665f3d90c9801fe1c73e.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/ceae1e7e6a65228a93405dd89b3372dbe406f27df4b8665f3d90c9801fe1c73e.jpg)

![cff6e12d03dd0159f999bca05a3f269b9572692ddb05abb8187f864d9b5642e4.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/cff6e12d03dd0159f999bca05a3f269b9572692ddb05abb8187f864d9b5642e4.jpg)

![dd409779a0d20c816bc321849511b170d853878cf849829c8c5d5609c8ea62ed.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/dd409779a0d20c816bc321849511b170d853878cf849829c8c5d5609c8ea62ed.jpg)

![ddd49c97685b1b7def0a885edec73672274779e2369903c75cd7bf6458ac3c07.jpg](../icml_results/4934_FedEBA%2B_%20Towards%20Fair%20and%20Effective%20Federated%20Learning%20via%20Entropy-Based%20Model/tables/ddd49c97685b1b7def0a885edec73672274779e2369903c75cd7bf6458ac3c07.jpg)

<a id="paper-4887"></a>

## Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs


### Images

![149f9efd1cc4db6a8e4568079067986a3138075d7f435a1bf6346b7ffbdd4ade.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/149f9efd1cc4db6a8e4568079067986a3138075d7f435a1bf6346b7ffbdd4ade.jpg)

![1be2ef778ff562eb2c49174c65fed7c3588c89eef79adcb298e767ed9021ef81.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/1be2ef778ff562eb2c49174c65fed7c3588c89eef79adcb298e767ed9021ef81.jpg)

![1f44e1ea56dc1dad876e386672226aa9d41e3af0d98ce1a3801a6f8ac654c711.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/1f44e1ea56dc1dad876e386672226aa9d41e3af0d98ce1a3801a6f8ac654c711.jpg)

![2c165bb1561f356c027d53f4750675f46edf03bb5807a0588a6b8971faf97e61.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/2c165bb1561f356c027d53f4750675f46edf03bb5807a0588a6b8971faf97e61.jpg)

![345e57a3c9402ff946c84e3c2665ccb9c3eeb62ec6f77589c75271fe06b74183.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/345e57a3c9402ff946c84e3c2665ccb9c3eeb62ec6f77589c75271fe06b74183.jpg)

![4399903640c9bd69f3f8d0f54f52d473768f0dc5bb2d6218a64cbf9feeef52b0.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/4399903640c9bd69f3f8d0f54f52d473768f0dc5bb2d6218a64cbf9feeef52b0.jpg)

![62fc10ca1c30148cad04693eb2580ed627860dcdcb822b12cf1e8a93c74a579b.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/62fc10ca1c30148cad04693eb2580ed627860dcdcb822b12cf1e8a93c74a579b.jpg)

![6ed6f2042adef6c10fe69264042cf753fdf97e572a69c60cad8f77bb40167aa0.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/6ed6f2042adef6c10fe69264042cf753fdf97e572a69c60cad8f77bb40167aa0.jpg)

![8f02b4402b4133cd221fbd87f0f866b03711b87f28f0cf7d342d1d369b86b8f9.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/8f02b4402b4133cd221fbd87f0f866b03711b87f28f0cf7d342d1d369b86b8f9.jpg)

![a50d5df11a4b1c2cad1ae50bebedc4312b3bc82e158747394403aa4993a646c2.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/a50d5df11a4b1c2cad1ae50bebedc4312b3bc82e158747394403aa4993a646c2.jpg)

![b46b9d1102ed147ba33d4dfea0df10e81f9b0b9597e529685879d0b553494475.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/b46b9d1102ed147ba33d4dfea0df10e81f9b0b9597e529685879d0b553494475.jpg)

![c4b8039a35e77635f87459c25824fd6afdc37a8d8f037409f9312a62a6de63e4.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/c4b8039a35e77635f87459c25824fd6afdc37a8d8f037409f9312a62a6de63e4.jpg)

![cd3592b001c6f878863bad0585cd293d4edb6b5ce370fab3fef88c72067f7686.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/cd3592b001c6f878863bad0585cd293d4edb6b5ce370fab3fef88c72067f7686.jpg)

![d4d6c2ba304620fdd0b155ae6f9128b3e1432e4416c33fcd73a8fd67cb4e7fc6.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/d4d6c2ba304620fdd0b155ae6f9128b3e1432e4416c33fcd73a8fd67cb4e7fc6.jpg)

![dadabcbf2212e988547bba0de482d32c0d669d287300a56c6a1b85eb2dfbf7a4.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/dadabcbf2212e988547bba0de482d32c0d669d287300a56c6a1b85eb2dfbf7a4.jpg)

![dc4c554efa2688a22e8272477a77132f6d1691d20f84246e87f71dce0c37ec45.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/dc4c554efa2688a22e8272477a77132f6d1691d20f84246e87f71dce0c37ec45.jpg)

![ed2a68370058fbb3380e296056ced84e10d60cb9206b695eba461518f47d4270.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/images/ed2a68370058fbb3380e296056ced84e10d60cb9206b695eba461518f47d4270.jpg)

### Tables

![06e8eb2e593f234b064f162d698ca48f48ee263c7b075f17800f61f372be71ea.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/06e8eb2e593f234b064f162d698ca48f48ee263c7b075f17800f61f372be71ea.jpg)

![0e90d701cfe9029db35c759621385f89e61ce6293f97aa26f1da728387345fa5.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/0e90d701cfe9029db35c759621385f89e61ce6293f97aa26f1da728387345fa5.jpg)

![18491c25cb14a17701af717765c5289389289cfaa342e0638eca6ccaf573947b.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/18491c25cb14a17701af717765c5289389289cfaa342e0638eca6ccaf573947b.jpg)

![3b34429c87e163da2ec7862bd89d6fc6730678aa35905f61929fe69632e20d52.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/3b34429c87e163da2ec7862bd89d6fc6730678aa35905f61929fe69632e20d52.jpg)

![4b1f459b309a339e9802af46f4e3d73b998d92ee26526cebba6574cd47d965bd.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/4b1f459b309a339e9802af46f4e3d73b998d92ee26526cebba6574cd47d965bd.jpg)

![5b25624b1ce98bca2d02a379bf28ab09983d5485e883cf9b8cf510a61bfe5a2e.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/5b25624b1ce98bca2d02a379bf28ab09983d5485e883cf9b8cf510a61bfe5a2e.jpg)

![6544277add065c53f9fcdf25ff191ce18b8040ac93ef6520d0f65c05192fb167.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/6544277add065c53f9fcdf25ff191ce18b8040ac93ef6520d0f65c05192fb167.jpg)

![65acb357beafcb78b36682e80d1dc31a8cb2076856929d06910d58f97b88c0d4.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/65acb357beafcb78b36682e80d1dc31a8cb2076856929d06910d58f97b88c0d4.jpg)

![678006d02eaa2e4db700b2c74fc30b86f14124e611c6e5efededdc17f5fc5480.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/678006d02eaa2e4db700b2c74fc30b86f14124e611c6e5efededdc17f5fc5480.jpg)

![789b8b3da729f01f5cbb6ae2b5c12533f32fd78592828a75f7d46a4175e13b38.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/789b8b3da729f01f5cbb6ae2b5c12533f32fd78592828a75f7d46a4175e13b38.jpg)

![7badef836e752274775ddde9c3dc51dce7cadf92a58f1ed89cef4b1a80404689.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/7badef836e752274775ddde9c3dc51dce7cadf92a58f1ed89cef4b1a80404689.jpg)

![863b6672e177caf7b8ff71e57dcfa9ef8e1ee5478737bcbfe11ff035e5845fdc.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/863b6672e177caf7b8ff71e57dcfa9ef8e1ee5478737bcbfe11ff035e5845fdc.jpg)

![95ef87d9cc53fcb22be8b4843c55bb4249a9b4f1be97e1f56073b1cf5ac6f9c2.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/95ef87d9cc53fcb22be8b4843c55bb4249a9b4f1be97e1f56073b1cf5ac6f9c2.jpg)

![c41ce77c687e07508f305e4295e4debff59ea57e5968976f761712ccd9597c5c.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/c41ce77c687e07508f305e4295e4debff59ea57e5968976f761712ccd9597c5c.jpg)

![d7783dd5a8fb5478547e60364f91de62871734bcec6fa0bcd30e584ca86acc18.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/d7783dd5a8fb5478547e60364f91de62871734bcec6fa0bcd30e584ca86acc18.jpg)

![d9fc56b6c97618161fa2a880f6a17503ef761aeac691ccdb47a6678062797b58.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/d9fc56b6c97618161fa2a880f6a17503ef761aeac691ccdb47a6678062797b58.jpg)

![ee2ea3e8e4af13c684ea00522f103fd7bcbf247009d2de3dba9f7a088c3534e0.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/ee2ea3e8e4af13c684ea00522f103fd7bcbf247009d2de3dba9f7a088c3534e0.jpg)

![f66fe478cd0947bf5216fb79593671d6b5fd3fa443fddc07aaed1a9c3b5d1ed0.jpg](../icml_results/4935_Seeing%20is%20Understanding_%20Unlocking%20Causal%20Attention%20into%20Modality-Mutual%20Attention%20for%20Multimodal%20LL/tables/f66fe478cd0947bf5216fb79593671d6b5fd3fa443fddc07aaed1a9c3b5d1ed0.jpg)

<a id="paper-4888"></a>

## Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future


### Images

![094ce7f6e79dfdaa1329c475c0b76db277cf828c0a853c9d4a891be6f98316bf.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/images/094ce7f6e79dfdaa1329c475c0b76db277cf828c0a853c9d4a891be6f98316bf.jpg)

![3c7a5165ce09772708b25822adaac00fd14731c4681b09a6282dce3d847cc399.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/images/3c7a5165ce09772708b25822adaac00fd14731c4681b09a6282dce3d847cc399.jpg)

![3f98f84bab0846ff699b4d457801ff41e61fa37c741861b9d79e8d287f225449.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/images/3f98f84bab0846ff699b4d457801ff41e61fa37c741861b9d79e8d287f225449.jpg)

![4dc10f2b0506de72763c16c3f9a40be5227d95c87cae93e4a7c2cd8f55b68ceb.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/images/4dc10f2b0506de72763c16c3f9a40be5227d95c87cae93e4a7c2cd8f55b68ceb.jpg)

### Tables

![083408de8833c604f3ca78c31e2cae4797e4780ae696493719048fa64941e092.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/083408de8833c604f3ca78c31e2cae4797e4780ae696493719048fa64941e092.jpg)

![366e3d72dfb47bf3bff70dd5cf73982513020233b14dbd32f2c249c096a7474b.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/366e3d72dfb47bf3bff70dd5cf73982513020233b14dbd32f2c249c096a7474b.jpg)

![6ee51edc14f15ad9ff7995ea1fcf166f5e450cf077afc75d3fc96fe4edf359ca.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/6ee51edc14f15ad9ff7995ea1fcf166f5e450cf077afc75d3fc96fe4edf359ca.jpg)

![90cdda575a653bdda89383c467cf09d7649998d493662197656f7ae498a5a17b.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/90cdda575a653bdda89383c467cf09d7649998d493662197656f7ae498a5a17b.jpg)

![927f8c1cb2dfe77b0fab487ce5ed0a96ecd9b8d93ef21a34f17be2401e5ced83.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/927f8c1cb2dfe77b0fab487ce5ed0a96ecd9b8d93ef21a34f17be2401e5ced83.jpg)

![9bb8403f0e95f741f951a1fda259f67df1b2784f4eee4895a71a7d6a371fb0dd.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/9bb8403f0e95f741f951a1fda259f67df1b2784f4eee4895a71a7d6a371fb0dd.jpg)

![dbdfa19ccdcd262e7d4294aca2f8371509c0cd3e4305389c410d7e1ad8f8851a.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/dbdfa19ccdcd262e7d4294aca2f8371509c0cd3e4305389c410d7e1ad8f8851a.jpg)

![e7d28b15778beea56d20cbd629e67983033cb156a626ab0b25d6b0490fbb634f.jpg](../icml_results/4936_Temporal%20Self-Rewarding%20Language%20Models_%20Decoupling%20Chosen-Rejected%20via%20Past-Future/tables/e7d28b15778beea56d20cbd629e67983033cb156a626ab0b25d6b0490fbb634f.jpg)

<a id="paper-4889"></a>

## Steady-State Behavior of Constant-Stepsize Stochastic Approximation: Gaussian Approximation and Tail Bounds


### Images

![539498ee0b178746f2918cb56c5ea43faa7ee74807a5735ea940bb4242effa83.jpg](../icml_results/4937_Steady-State%20Behavior%20of%20Constant-Stepsize%20Stochastic%20Approximation_%20Gaussian%20Approximation%20and%20Tail/images/539498ee0b178746f2918cb56c5ea43faa7ee74807a5735ea940bb4242effa83.jpg)

![9b66f0933b704217f20bf437a6aeba7a91c41e7121734909cfc19211c80827c8.jpg](../icml_results/4937_Steady-State%20Behavior%20of%20Constant-Stepsize%20Stochastic%20Approximation_%20Gaussian%20Approximation%20and%20Tail/images/9b66f0933b704217f20bf437a6aeba7a91c41e7121734909cfc19211c80827c8.jpg)

![bbb27172627c8878780b717b5c522adf61807d557455762a0607c2b5435c561c.jpg](../icml_results/4937_Steady-State%20Behavior%20of%20Constant-Stepsize%20Stochastic%20Approximation_%20Gaussian%20Approximation%20and%20Tail/images/bbb27172627c8878780b717b5c522adf61807d557455762a0607c2b5435c561c.jpg)

![d65dae22dad9cd0e0507b866601d68c9b48e31bd047cb2305ce2aaadd8e057f8.jpg](../icml_results/4937_Steady-State%20Behavior%20of%20Constant-Stepsize%20Stochastic%20Approximation_%20Gaussian%20Approximation%20and%20Tail/images/d65dae22dad9cd0e0507b866601d68c9b48e31bd047cb2305ce2aaadd8e057f8.jpg)

![e7b8b0bffd019bd1db6445cbc8b1ae1faa2f34e542009adb94ba44a031b0e800.jpg](../icml_results/4937_Steady-State%20Behavior%20of%20Constant-Stepsize%20Stochastic%20Approximation_%20Gaussian%20Approximation%20and%20Tail/images/e7b8b0bffd019bd1db6445cbc8b1ae1faa2f34e542009adb94ba44a031b0e800.jpg)

<a id="paper-4890"></a>

## Agentic Model Predictive Questioning Control in Visual Design


### Images

![2e5655be795879ba86b0230a887ac3cd9234e029d406de6acdfa48a6b1ab9bc0.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/2e5655be795879ba86b0230a887ac3cd9234e029d406de6acdfa48a6b1ab9bc0.jpg)

![4a688d242b04be427b02b43aa53e5d4a6c4f39d0ffaaaa27d1f120440111e40d.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/4a688d242b04be427b02b43aa53e5d4a6c4f39d0ffaaaa27d1f120440111e40d.jpg)

![72dcef3473289d8002f5676835c34407e665252f0538729353475b1ebda4c0d7.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/72dcef3473289d8002f5676835c34407e665252f0538729353475b1ebda4c0d7.jpg)

![74138eb608e315874e3cf91ffa0a6e5ba8f019d7ea5117fa43577cecf2fca664.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/74138eb608e315874e3cf91ffa0a6e5ba8f019d7ea5117fa43577cecf2fca664.jpg)

![965d42796ea4e0799096a7c52409a9abaab2ff31b6c735e3dc16a218fc10b94b.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/965d42796ea4e0799096a7c52409a9abaab2ff31b6c735e3dc16a218fc10b94b.jpg)

![ac159fc31e38387300053a86f16c905b30e9de0694b8489f858793a1ee3d4ad8.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/ac159fc31e38387300053a86f16c905b30e9de0694b8489f858793a1ee3d4ad8.jpg)

![b62bfe827a7c8f599a9c979325e6cefaaf35b3f50f5b013a2e5e2741a88cfba2.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/b62bfe827a7c8f599a9c979325e6cefaaf35b3f50f5b013a2e5e2741a88cfba2.jpg)

![c7d54fa2d6740e8e97cd0abe9337ecc71dafecc3d1535e68382ca06aec67d20a.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/images/c7d54fa2d6740e8e97cd0abe9337ecc71dafecc3d1535e68382ca06aec67d20a.jpg)

### Tables

![16275485464ed9289fec851f3cf3d00bae7ad3d40412011a9ae542c984dbfd5d.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/tables/16275485464ed9289fec851f3cf3d00bae7ad3d40412011a9ae542c984dbfd5d.jpg)

![287408083a2c3b937d59d7c49d681532e5d81003c4ca90d86cc0af94d57f8ede.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/tables/287408083a2c3b937d59d7c49d681532e5d81003c4ca90d86cc0af94d57f8ede.jpg)

![72548a70dc19b5a7d1049471a83e87799d1772ce5fb2ac5f191d272bc5c63a49.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/tables/72548a70dc19b5a7d1049471a83e87799d1772ce5fb2ac5f191d272bc5c63a49.jpg)

![957ca4596dffaf83fc7a5b5d850c25255f4a899fdaad4f0e53877d9f39b943f3.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/tables/957ca4596dffaf83fc7a5b5d850c25255f4a899fdaad4f0e53877d9f39b943f3.jpg)

![c720835935895017486fd5e4e400e7ccacc60562fb642614c911dfec15c0b832.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/tables/c720835935895017486fd5e4e400e7ccacc60562fb642614c911dfec15c0b832.jpg)

![e40227c74a708517301b121668017827749082a020f919f417081ddfc1100eb9.jpg](../icml_results/4938_Agentic%20Model%20Predictive%20Questioning%20Control%20in%20Visual%20Design/tables/e40227c74a708517301b121668017827749082a020f919f417081ddfc1100eb9.jpg)

<a id="paper-4891"></a>

## Reward Modeling from Natural Language Human Feedback


### Images

![0574f18bb326e95406bcec063e7e443ef3bbf518ee19b19467a674d7ea392459.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/0574f18bb326e95406bcec063e7e443ef3bbf518ee19b19467a674d7ea392459.jpg)

![268d8a2af7b534156189d7ae418f3d3fcc3795a3ec5d3258fe2a4547fca4fc59.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/268d8a2af7b534156189d7ae418f3d3fcc3795a3ec5d3258fe2a4547fca4fc59.jpg)

![29784cae7cadf4d04e78f25271c4fc43e32d2183c38c224c399de5e83d95b509.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/29784cae7cadf4d04e78f25271c4fc43e32d2183c38c224c399de5e83d95b509.jpg)

![311aa6d3415e346800566a686df80fb1184cadb077fcfb9c1498e0d8520a8e5d.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/311aa6d3415e346800566a686df80fb1184cadb077fcfb9c1498e0d8520a8e5d.jpg)

![66565694fde5bd948ebb8ad397908c330a97d19b2d5b7ae6e19fb82788b24d73.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/66565694fde5bd948ebb8ad397908c330a97d19b2d5b7ae6e19fb82788b24d73.jpg)

![6af881f011d717a126e09d39e5a0ec469cf7125ac10fadcefdca369a57861727.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/6af881f011d717a126e09d39e5a0ec469cf7125ac10fadcefdca369a57861727.jpg)

![b7322de7e5b76e2b90f252b49f3cbd96e638d7bdf43ccfd09d78ecb2ba1c4102.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/b7322de7e5b76e2b90f252b49f3cbd96e638d7bdf43ccfd09d78ecb2ba1c4102.jpg)

![cb47666701f1cbe56a03e91e6324f7251d6c304d68e05b861ad34d60f7c3a4e5.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/cb47666701f1cbe56a03e91e6324f7251d6c304d68e05b861ad34d60f7c3a4e5.jpg)

![dcd7713368970ca1a44ccae548f5d59cc16e443f79db94b243d374bbd1949c80.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/dcd7713368970ca1a44ccae548f5d59cc16e443f79db94b243d374bbd1949c80.jpg)

![fc5f3a664cf93ce38842584ffe9e371f715f82119b030e6fa34566c391c7418e.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/fc5f3a664cf93ce38842584ffe9e371f715f82119b030e6fa34566c391c7418e.jpg)

![ff70cbbd5e4060b9deac0908bd9ad6e5c9be4b030d9967d3f83fad7a58b828fc.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/images/ff70cbbd5e4060b9deac0908bd9ad6e5c9be4b030d9967d3f83fad7a58b828fc.jpg)

### Tables

![1b2d2a7b855e19f21cbbc5be6dacde12cdd9bf971435e48f0ced3282aac7b19b.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/1b2d2a7b855e19f21cbbc5be6dacde12cdd9bf971435e48f0ced3282aac7b19b.jpg)

![2443dc6ece66bc3ae8d25c4d234409e7f881c1ce0aa04d40e1bf43bdc50eec9e.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/2443dc6ece66bc3ae8d25c4d234409e7f881c1ce0aa04d40e1bf43bdc50eec9e.jpg)

![25853ce712bfde1431046b595b08250153ad0e450b071a036c2f19a304029636.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/25853ce712bfde1431046b595b08250153ad0e450b071a036c2f19a304029636.jpg)

![26d7c4d5e8d3be4f91b48554cbe7bbd390c61d1ca55aa5c76be29ff006530de3.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/26d7c4d5e8d3be4f91b48554cbe7bbd390c61d1ca55aa5c76be29ff006530de3.jpg)

![2e24241c92fce3e893f9e06b8c816de08ae9396218e15edc6863c820d12e29c2.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/2e24241c92fce3e893f9e06b8c816de08ae9396218e15edc6863c820d12e29c2.jpg)

![449840cbc69b3347423f19b5b719614399581f3426bf24aacedc85e81772eef8.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/449840cbc69b3347423f19b5b719614399581f3426bf24aacedc85e81772eef8.jpg)

![543c0def9ec11ed13b1ee6b1e92268d11d944915c32e2a88ef3477f853612bd9.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/543c0def9ec11ed13b1ee6b1e92268d11d944915c32e2a88ef3477f853612bd9.jpg)

![56d9216ed48595a6d979e4e6eff76dfb8a72d0217c16d105ccc4224135d5be59.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/56d9216ed48595a6d979e4e6eff76dfb8a72d0217c16d105ccc4224135d5be59.jpg)

![71809e8b419e45719306e8901c6cdf799dfd14502bc5590963475dd227961c08.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/71809e8b419e45719306e8901c6cdf799dfd14502bc5590963475dd227961c08.jpg)

![7e084ea38870e99d30e3da64eb80e6e1cc85f9fef9976d304e4236e7a1ae8502.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/7e084ea38870e99d30e3da64eb80e6e1cc85f9fef9976d304e4236e7a1ae8502.jpg)

![856134f85add6800f56836fc78a600fa91b33fdc29556bdfcf97fcb83927c3b3.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/856134f85add6800f56836fc78a600fa91b33fdc29556bdfcf97fcb83927c3b3.jpg)

![a6e2a0f372a5eb21af2917ef84bcee63db97d6cc4ed17357caf68efb5d8c86d8.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/a6e2a0f372a5eb21af2917ef84bcee63db97d6cc4ed17357caf68efb5d8c86d8.jpg)

![b1f85d8432abc675ab053959d7800e30ee4a2c490bc59d88d97471e38c87c3ad.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/b1f85d8432abc675ab053959d7800e30ee4a2c490bc59d88d97471e38c87c3ad.jpg)

![c3d4b42400eb842b565c49cc790cf6a129a4f1439b93e9da6e6f4e06a10098b5.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/c3d4b42400eb842b565c49cc790cf6a129a4f1439b93e9da6e6f4e06a10098b5.jpg)

![d6cb733b288fcb6e62205c6013459d5b19098dbf55014b80915c24ac39fd48b8.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/d6cb733b288fcb6e62205c6013459d5b19098dbf55014b80915c24ac39fd48b8.jpg)

![dbe14c921817e2aa16bf9ec5006d3a535ed9b8b8f89778369d5a19fdf93eca53.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/dbe14c921817e2aa16bf9ec5006d3a535ed9b8b8f89778369d5a19fdf93eca53.jpg)

![f3137c1fc07f08ae591db74323e596d87fc22ee074266764c257d69ff7990ed1.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/f3137c1fc07f08ae591db74323e596d87fc22ee074266764c257d69ff7990ed1.jpg)

![fb9950a7f395734b0a8e5c33975b51ad98cb508597dfab07f0407f967686443d.jpg](../icml_results/4939_Reward%20Modeling%20from%20Natural%20Language%20Human%20Feedback/tables/fb9950a7f395734b0a8e5c33975b51ad98cb508597dfab07f0407f967686443d.jpg)

<a id="paper-4892"></a>

## MuCO: Generative Peptide Cyclization Empowered by Multi-stage Conformation Optimization


### Images

![182afc6b15264b31fd6138889017d5ccd274827880ff66f54aab223ee61c31ea.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/182afc6b15264b31fd6138889017d5ccd274827880ff66f54aab223ee61c31ea.jpg)

![3166e94b36b1ab76a054c307ab341d498e2d43e2eb6c0b782f34cf460e5b66f5.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/3166e94b36b1ab76a054c307ab341d498e2d43e2eb6c0b782f34cf460e5b66f5.jpg)

![36ce56c9f12fc80d011d0da3bf54dc8cd4d8e0110dece8fb19524c8c18bef8d2.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/36ce56c9f12fc80d011d0da3bf54dc8cd4d8e0110dece8fb19524c8c18bef8d2.jpg)

![42d67692f3bbabe1e6aa5b3ecfdc11305c3218dc566682cfd3c0b184a6b552c3.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/42d67692f3bbabe1e6aa5b3ecfdc11305c3218dc566682cfd3c0b184a6b552c3.jpg)

![5a5fc1467ba63484bda2a2800ba529dc2bcbd5af5f7c2640a4c662f0cc8c6b8b.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/5a5fc1467ba63484bda2a2800ba529dc2bcbd5af5f7c2640a4c662f0cc8c6b8b.jpg)

![66a1d68e937219e93c39bca4f88a9ca241bbb8062c2b500c476973e5d268d115.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/66a1d68e937219e93c39bca4f88a9ca241bbb8062c2b500c476973e5d268d115.jpg)

![7653ad7e75e03a387967836842bd1a47517bce86b2e4c703d691fc7b18333b86.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/7653ad7e75e03a387967836842bd1a47517bce86b2e4c703d691fc7b18333b86.jpg)

![76d6f692b2d2a52f18288e540eaec3f7748600f914594e3b3d8617beda2f0b6c.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/76d6f692b2d2a52f18288e540eaec3f7748600f914594e3b3d8617beda2f0b6c.jpg)

![7a0c7411a2d7615211a69bb1e131015ea61f073f6041b807bd606192782ca8b9.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/7a0c7411a2d7615211a69bb1e131015ea61f073f6041b807bd606192782ca8b9.jpg)

![8d1937862dc7f3834e4b2cf06a0052c177a0eeae89d3180414ee90b280c561c4.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/8d1937862dc7f3834e4b2cf06a0052c177a0eeae89d3180414ee90b280c561c4.jpg)

![90bb7235b2016f51201445dbad820394dff5fa073588883150f47f2dce17a5cf.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/90bb7235b2016f51201445dbad820394dff5fa073588883150f47f2dce17a5cf.jpg)

![a12f43eceb39ec60df4030c311875904aa39c2f34ef391db2fb7ade8bf04dec3.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/a12f43eceb39ec60df4030c311875904aa39c2f34ef391db2fb7ade8bf04dec3.jpg)

![b13d3b17807a68aa9f3264ebac9936de7dff33d6d36315717f1d7780dc59290d.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/b13d3b17807a68aa9f3264ebac9936de7dff33d6d36315717f1d7780dc59290d.jpg)

![c44a0282426877c99856dcab4dc57fc049edd6b613b31a0d435d87f4e0073366.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/images/c44a0282426877c99856dcab4dc57fc049edd6b613b31a0d435d87f4e0073366.jpg)

### Tables

![0bddc0f18fe1669b8186d2fca41f7c0e681230c355f79167841787f7a484bf75.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/0bddc0f18fe1669b8186d2fca41f7c0e681230c355f79167841787f7a484bf75.jpg)

![0c6785180aa5ffdc1a42eb44d0b769aa355c1abb424d7cecc3de2f125b0ef3d2.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/0c6785180aa5ffdc1a42eb44d0b769aa355c1abb424d7cecc3de2f125b0ef3d2.jpg)

![259f5c1d2f8682995cdede353b4019c5f8cfd163740a0bc1a16dc7c3aba5c79c.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/259f5c1d2f8682995cdede353b4019c5f8cfd163740a0bc1a16dc7c3aba5c79c.jpg)

![2738ab99d65041860a942478ec268a7e6e20a51f92918bf48812f889303eaeda.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/2738ab99d65041860a942478ec268a7e6e20a51f92918bf48812f889303eaeda.jpg)

![397b6a91bea0b5c14dc4bf7371962b14a9ff9e0a1d3e999786fbb090e0ecfe5b.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/397b6a91bea0b5c14dc4bf7371962b14a9ff9e0a1d3e999786fbb090e0ecfe5b.jpg)

![3b304b55e620655d28d4b5a4496a16a4341bf89a6f6391e8144ed1a320ea743e.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/3b304b55e620655d28d4b5a4496a16a4341bf89a6f6391e8144ed1a320ea743e.jpg)

![4b2e1f52be82c123044495bb327075a60d695e23351d7bb657c21df431716c83.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/4b2e1f52be82c123044495bb327075a60d695e23351d7bb657c21df431716c83.jpg)

![4d21fcf4faaa03d74714bdf2c0ea96441dd1f77f55ef9b95435d4fb9113b398c.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/4d21fcf4faaa03d74714bdf2c0ea96441dd1f77f55ef9b95435d4fb9113b398c.jpg)

![635fd8001b84500cc47eb88a770e6382fcfb0e4a0e8f4205300d231400ea3d38.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/635fd8001b84500cc47eb88a770e6382fcfb0e4a0e8f4205300d231400ea3d38.jpg)

![67d3e78ae1779c1b450127c8334bb9967729c741d43116523dc831747bd183e3.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/67d3e78ae1779c1b450127c8334bb9967729c741d43116523dc831747bd183e3.jpg)

![69ba189c04bdbcde56da1aff819a17e199fade1e22dbe7c48170f1bcfa0fb05a.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/69ba189c04bdbcde56da1aff819a17e199fade1e22dbe7c48170f1bcfa0fb05a.jpg)

![70b4212a05165a05072da0ec71a325443e81c3baaf623b0f997ff42179cd8c1c.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/70b4212a05165a05072da0ec71a325443e81c3baaf623b0f997ff42179cd8c1c.jpg)

![8b6443a1fb6eac394627a030b15f39e177dd99c18ea3c319b8431c56a720c98c.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/8b6443a1fb6eac394627a030b15f39e177dd99c18ea3c319b8431c56a720c98c.jpg)

![a7c2030657c07be3416511665020dade720b02606a879246b564635b6ad0eb69.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/a7c2030657c07be3416511665020dade720b02606a879246b564635b6ad0eb69.jpg)

![aa84fdbe5ca77a08d80e5634ac52b37b6f8abf8bd3a349ebf3f78b179b07153f.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/aa84fdbe5ca77a08d80e5634ac52b37b6f8abf8bd3a349ebf3f78b179b07153f.jpg)

![aea7ced066d6444ac064e2ad3fdec3ae22c9be9515ce26d1b1febee4b6ad9035.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/aea7ced066d6444ac064e2ad3fdec3ae22c9be9515ce26d1b1febee4b6ad9035.jpg)

![b9018e384131923bf79c0be56126cfd1b60c8aa9307888bc6ad0fd2f49a914d7.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/b9018e384131923bf79c0be56126cfd1b60c8aa9307888bc6ad0fd2f49a914d7.jpg)

![d0d71730c32d66f41c27a3a66eba3dfe74711507024787745e0605189edaf8bf.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/d0d71730c32d66f41c27a3a66eba3dfe74711507024787745e0605189edaf8bf.jpg)

![d25c3c458d797d959f7967c7d084823914e1f1907d425ed39bbd46d9cc8106af.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/d25c3c458d797d959f7967c7d084823914e1f1907d425ed39bbd46d9cc8106af.jpg)

![d505c4d7c18106f697893e9b9c037f6e73ba8527e0759b0b975cb5a8cef482ad.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/d505c4d7c18106f697893e9b9c037f6e73ba8527e0759b0b975cb5a8cef482ad.jpg)

![f6e98c089d9b2edeabf094a3b2a5ad770f40c84a6b214f0e4422e5a2829b2d87.jpg](../icml_results/4940_MuCO_%20Generative%20Peptide%20Cyclization%20Empowered%20by%20Multi-stage%20Conformation%20Optimization/tables/f6e98c089d9b2edeabf094a3b2a5ad770f40c84a6b214f0e4422e5a2829b2d87.jpg)

<a id="paper-4893"></a>

## Navigating Massive Visual Context in Retrieval-Augmented Generation via Multimodal Memory Graph


### Images

![099469a081ed6586df9806e04a63d72aea870d581b703eba21f0f8295cca6131.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/099469a081ed6586df9806e04a63d72aea870d581b703eba21f0f8295cca6131.jpg)

![1e854fdf97d7856f03b49ee044b8f95cfa182167b31683f0e0a0625bf224d70a.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/1e854fdf97d7856f03b49ee044b8f95cfa182167b31683f0e0a0625bf224d70a.jpg)

![23727607130772b06f792b2ea916edc62110a1127400fd26ef0ac525859ee088.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/23727607130772b06f792b2ea916edc62110a1127400fd26ef0ac525859ee088.jpg)

![49323e945442fdefce00ca789961ffef0d8679c11a5e331dc4d2b9db0c182327.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/49323e945442fdefce00ca789961ffef0d8679c11a5e331dc4d2b9db0c182327.jpg)

![5268c31470acf23906b873ae4f04a1f72db9fdc84c9f08f273c5d1c1eb840eda.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/5268c31470acf23906b873ae4f04a1f72db9fdc84c9f08f273c5d1c1eb840eda.jpg)

![59daed547e391a0a35de087ed693648bad379cc5cf8d709c30509120207fd100.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/59daed547e391a0a35de087ed693648bad379cc5cf8d709c30509120207fd100.jpg)

![5c20186a795c3e81af5de3fc251a4d9537cb0dfc3ff73563c3cbd3398f5caff2.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/5c20186a795c3e81af5de3fc251a4d9537cb0dfc3ff73563c3cbd3398f5caff2.jpg)

![5fa5a39f183313a054ba2fd6132207cc31c02ad9176afb1af302bb57a8ac42f1.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/5fa5a39f183313a054ba2fd6132207cc31c02ad9176afb1af302bb57a8ac42f1.jpg)

![62cd235e680eddd4c499274814b92f02438fef0df02e6cf3441bc53a7dc5877f.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/62cd235e680eddd4c499274814b92f02438fef0df02e6cf3441bc53a7dc5877f.jpg)

![7e2e91c941cf2e1a9c45f27ca2ccccc14fbb2274be1eda51414457bfe1232cb0.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/7e2e91c941cf2e1a9c45f27ca2ccccc14fbb2274be1eda51414457bfe1232cb0.jpg)

![86599a42b6ad90dfe568239b3f9a3b0d76e200e609f1c1a64d8f2cbff5993f67.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/86599a42b6ad90dfe568239b3f9a3b0d76e200e609f1c1a64d8f2cbff5993f67.jpg)

![93ad443ca725f29af3ad6f56fb69efe0c4c46608cdf23262a39f4a1b30c176d4.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/93ad443ca725f29af3ad6f56fb69efe0c4c46608cdf23262a39f4a1b30c176d4.jpg)

![9445a0cd6871aac9a99e2cc9f6d54e5a6ac191d229cf00d6a85e8b4cf4076c2e.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/9445a0cd6871aac9a99e2cc9f6d54e5a6ac191d229cf00d6a85e8b4cf4076c2e.jpg)

![9ef4a32be741ee05d14d233ba22c850fb2f6676ef0ac5ba6473a8434b1b38084.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/9ef4a32be741ee05d14d233ba22c850fb2f6676ef0ac5ba6473a8434b1b38084.jpg)

![9f8afe5aafebabbd3acac43d82dac69655a6387ec68a6405b6864e9e1b34b3db.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/9f8afe5aafebabbd3acac43d82dac69655a6387ec68a6405b6864e9e1b34b3db.jpg)

![a3ea5a6003f98610d46259b5c58046aff35837f12c864e93903cfb407ed58235.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/a3ea5a6003f98610d46259b5c58046aff35837f12c864e93903cfb407ed58235.jpg)

![a66aab63bb5a2e32c1456e533f87dbb61f0acf8f4281f53d0a8ad65f2ed0dafd.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/a66aab63bb5a2e32c1456e533f87dbb61f0acf8f4281f53d0a8ad65f2ed0dafd.jpg)

![c441e4b46b9421aee9926a4ea773f72fbacfa774989a025904cacf6a4369cf01.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/c441e4b46b9421aee9926a4ea773f72fbacfa774989a025904cacf6a4369cf01.jpg)

![c45c328059500a7fe1623efd284544c7b974793344b0265fa3622672cddb1e34.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/c45c328059500a7fe1623efd284544c7b974793344b0265fa3622672cddb1e34.jpg)

![d04cc6fab24257e35d0828ffd0349482362a06bfaa864ecd0e81d64fa2cb23f5.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/d04cc6fab24257e35d0828ffd0349482362a06bfaa864ecd0e81d64fa2cb23f5.jpg)

![d2156e1cdab4f8368e410828b693d23f23e236556b5fc049334e88143b5df9b0.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/d2156e1cdab4f8368e410828b693d23f23e236556b5fc049334e88143b5df9b0.jpg)

![d3b6741811538549d2cd9eb63012db9edf5215f6037aa3a1cee816ef4b869c14.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/d3b6741811538549d2cd9eb63012db9edf5215f6037aa3a1cee816ef4b869c14.jpg)

![f854529180f57e9a412f7d89339c835a6567fa172c56c7990d3ba9ee3a197483.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/images/f854529180f57e9a412f7d89339c835a6567fa172c56c7990d3ba9ee3a197483.jpg)

### Tables

![09c2a149ecf7f1448ab1391f062b445cc24c51646619c015f1fd95c3f80195ad.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/tables/09c2a149ecf7f1448ab1391f062b445cc24c51646619c015f1fd95c3f80195ad.jpg)

![1bddd7a72d25f068a0b7b828fb516545594b36a9ddc852d15d895bcb06d0ff67.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/tables/1bddd7a72d25f068a0b7b828fb516545594b36a9ddc852d15d895bcb06d0ff67.jpg)

![261ece409c05e5a6dfe31c91da0a9f2d8182686ca0258538ca2d8110f08f0cad.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/tables/261ece409c05e5a6dfe31c91da0a9f2d8182686ca0258538ca2d8110f08f0cad.jpg)

![2b49f47f44cbbc4891e6907f8634fc6cb1a6ea040f771209f7c258817589b368.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/tables/2b49f47f44cbbc4891e6907f8634fc6cb1a6ea040f771209f7c258817589b368.jpg)

![472e6fc6212787484bbc4e2e9718f3a371ce68629bac864b3d477700e939de1a.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/tables/472e6fc6212787484bbc4e2e9718f3a371ce68629bac864b3d477700e939de1a.jpg)

![657abbf194ded6a43ef603765751a8d6e7667ff78f6bec6a7a65af16591ca845.jpg](../icml_results/4941_Navigating%20Massive%20Visual%20Context%20in%20Retrieval-Augmented%20Generation%20via%20Multimodal%20Memory%20Graph/tables/657abbf194ded6a43ef603765751a8d6e7667ff78f6bec6a7a65af16591ca845.jpg)

<a id="paper-4894"></a>

## Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions


### Images

![0fbe907f02e82caafc57b5f81e71b7c0956aee24624cd82fcaeb0b6b1e56c075.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/0fbe907f02e82caafc57b5f81e71b7c0956aee24624cd82fcaeb0b6b1e56c075.jpg)

![1d745816da44c4d60930f18d25f913b0b357768b0661769bb65fe2e34e338a43.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/1d745816da44c4d60930f18d25f913b0b357768b0661769bb65fe2e34e338a43.jpg)

![2f74619709dab1273016f64aef3eb0ca6ed9cdc7341caa386ab08cedb340f4a8.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/2f74619709dab1273016f64aef3eb0ca6ed9cdc7341caa386ab08cedb340f4a8.jpg)

![48984195a0520319a82c1fc8b62ad573940dc02e8351cb340008254a2b345bb1.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/48984195a0520319a82c1fc8b62ad573940dc02e8351cb340008254a2b345bb1.jpg)

![4f8a07d77859d423f4fec0e08ef39b64f2d480d9fc29fa95f89b637c965e2197.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/4f8a07d77859d423f4fec0e08ef39b64f2d480d9fc29fa95f89b637c965e2197.jpg)

![6d9a9d7bce683e9e26976707a1306ad9599b00754722994f32eb918f112d1a90.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/6d9a9d7bce683e9e26976707a1306ad9599b00754722994f32eb918f112d1a90.jpg)

![921862553515361afdcbe56dcac294aa40ec89dbbe9137d6098a73ce42d0620e.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/921862553515361afdcbe56dcac294aa40ec89dbbe9137d6098a73ce42d0620e.jpg)

![a317775682975ee4df4771c9c7d71bfcaedb0ed8d63df5cb42f1c578a2a98313.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/a317775682975ee4df4771c9c7d71bfcaedb0ed8d63df5cb42f1c578a2a98313.jpg)

![e24cca6692598541d25e90298c4e4c0c76e36cbef986abb5e5c7beccfa735289.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/e24cca6692598541d25e90298c4e4c0c76e36cbef986abb5e5c7beccfa735289.jpg)

![f09f9d6f5b486e395af24326db03009f1a882a14cced26efb0602ac77ada0fe1.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/f09f9d6f5b486e395af24326db03009f1a882a14cced26efb0602ac77ada0fe1.jpg)

![f324da6885f2a2ce763f547e7e934dc39f90f3c0a3da7ced283299159da46d04.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/images/f324da6885f2a2ce763f547e7e934dc39f90f3c0a3da7ced283299159da46d04.jpg)

### Tables

![0476023909596c0074e0a0f3bedb4b09db6e197047b382b98ada698432f83fa5.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/0476023909596c0074e0a0f3bedb4b09db6e197047b382b98ada698432f83fa5.jpg)

![13128b60a489355c4b4f5a8ac39efd3b4214258d0508fdcfb662c8a03dbe6a2c.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/13128b60a489355c4b4f5a8ac39efd3b4214258d0508fdcfb662c8a03dbe6a2c.jpg)

![1c8a5b2c7527b28fa230f2153782cccfd8008be73ffe2d30bfd1545fe49cb483.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/1c8a5b2c7527b28fa230f2153782cccfd8008be73ffe2d30bfd1545fe49cb483.jpg)

![24caabb0087d8b44b32386c263bfde00ca5f89fa7cf89d1333d0ca0776c47b5c.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/24caabb0087d8b44b32386c263bfde00ca5f89fa7cf89d1333d0ca0776c47b5c.jpg)

![27a5076d38d299c338e7083fa58d01a0dc227a2d5eaaeb5801347e59b66ecf44.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/27a5076d38d299c338e7083fa58d01a0dc227a2d5eaaeb5801347e59b66ecf44.jpg)

![59be3bda1dad879598d46f31204a0930be579b77780fc7cd9616995fc7e8fe86.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/59be3bda1dad879598d46f31204a0930be579b77780fc7cd9616995fc7e8fe86.jpg)

![6038cf61e4c468faa1a14d246f1bf7a0d215ab7a113c99ae1d7adf8eb2a3a646.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/6038cf61e4c468faa1a14d246f1bf7a0d215ab7a113c99ae1d7adf8eb2a3a646.jpg)

![647d18a4fde30b632d70e408994c9b33078d6b45eb599ce41f20e0a23555b5e3.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/647d18a4fde30b632d70e408994c9b33078d6b45eb599ce41f20e0a23555b5e3.jpg)

![8c0d4f6b9fa0163b4710a4973ac653183098fbbedd784d1550881fb9916679bf.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/8c0d4f6b9fa0163b4710a4973ac653183098fbbedd784d1550881fb9916679bf.jpg)

![a779dedc58b1324133392eb01763e0bc9947a2016223f8b65601d43bc65af7e7.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/a779dedc58b1324133392eb01763e0bc9947a2016223f8b65601d43bc65af7e7.jpg)

![b0286f8c1a05f1ddde95c62f32173e545018d3371e2a7954fa3659c5ca23c32b.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/b0286f8c1a05f1ddde95c62f32173e545018d3371e2a7954fa3659c5ca23c32b.jpg)

![cf64aabb5f7294a02095eea47677dab24da65d5dfe9db6d5cebc0c9c634fbd64.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/cf64aabb5f7294a02095eea47677dab24da65d5dfe9db6d5cebc0c9c634fbd64.jpg)

![d8d160a2c29a85afc7d2051b7c65cc111f3602be95a4b6a20a35b5840ebee831.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/d8d160a2c29a85afc7d2051b7c65cc111f3602be95a4b6a20a35b5840ebee831.jpg)

![e6929d2d9402f325910f8cca0987718b8e6bf98e743e6383ef183aa3e665466d.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/e6929d2d9402f325910f8cca0987718b8e6bf98e743e6383ef183aa3e665466d.jpg)

![f9d9a02c994a5d23b38057c59d7efa1e2065d6482ef147081ba280316df18591.jpg](../icml_results/4942_Physics%20from%20Video_%20Identifiability%20of%20Time-Invariant%20Second-Order%20ODEs%20under%20Minimal%20Trajectory%20Con/tables/f9d9a02c994a5d23b38057c59d7efa1e2065d6482ef147081ba280316df18591.jpg)

<a id="paper-4895"></a>

## Sparser Block-Sparse Attention via Token Permutation


### Images

![031c6b3d22032919e1ccb84cbb6fd812943cf0ee7cc3784bd60abed8a088b850.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/031c6b3d22032919e1ccb84cbb6fd812943cf0ee7cc3784bd60abed8a088b850.jpg)

![1986514f446a46b3643d2d131cbd267b215f3703dea62a7573149ec714248629.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/1986514f446a46b3643d2d131cbd267b215f3703dea62a7573149ec714248629.jpg)

![2039376d0ef0377ec6912fb50c2bca275df1afb4d6c0233d4d1ca7efa0f0890b.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/2039376d0ef0377ec6912fb50c2bca275df1afb4d6c0233d4d1ca7efa0f0890b.jpg)

![21e9afb28476bff3878f4d1f9810b65573c7c66cbbbd8a5c925fe3d14f503293.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/21e9afb28476bff3878f4d1f9810b65573c7c66cbbbd8a5c925fe3d14f503293.jpg)

![23ddf46d043d6d3b7390308e38f149fe4520e59f01be9c9692c13dd5eb9efb64.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/23ddf46d043d6d3b7390308e38f149fe4520e59f01be9c9692c13dd5eb9efb64.jpg)

![450ba57faef4540816d1cfeda7018fe8b54d3e51859a89fe826bc00d1ffcf1ed.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/450ba57faef4540816d1cfeda7018fe8b54d3e51859a89fe826bc00d1ffcf1ed.jpg)

![48988a6f2693e79339e18eb695fbb031b40df443538ebdb234f3f4703f3c0ea7.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/48988a6f2693e79339e18eb695fbb031b40df443538ebdb234f3f4703f3c0ea7.jpg)

![65310b4a973f4082f7fb69158bf02b911bfff1de0920d3423931fd9c7b06e2c2.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/65310b4a973f4082f7fb69158bf02b911bfff1de0920d3423931fd9c7b06e2c2.jpg)

![676b138e956941fbdf0eba0815b8a56f26ca2b3d7f364669a295e0302d847833.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/676b138e956941fbdf0eba0815b8a56f26ca2b3d7f364669a295e0302d847833.jpg)

![a1343b2466ad36142919a07ccf2c9b91a7d181b434d4377b23fab5bb13db8314.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/a1343b2466ad36142919a07ccf2c9b91a7d181b434d4377b23fab5bb13db8314.jpg)

![a4c3a1bfddbfce4ba2b1fcdfd8f7d6a1ef1778779881322df31647bae1893899.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/a4c3a1bfddbfce4ba2b1fcdfd8f7d6a1ef1778779881322df31647bae1893899.jpg)

![a6a9c30fe4fae23c5bf4adc93b7130b7f1dae87ac62a13328bdd2225972628ae.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/a6a9c30fe4fae23c5bf4adc93b7130b7f1dae87ac62a13328bdd2225972628ae.jpg)

![adbd93d1fef78aa48d15987fc92edfcc1f0e1587c2cb7e226c772076bb0ea025.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/adbd93d1fef78aa48d15987fc92edfcc1f0e1587c2cb7e226c772076bb0ea025.jpg)

![e232bdaf767b0dcf1176da82465c78275fece8a41c884b7c94411f3a58a7e8e1.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/e232bdaf767b0dcf1176da82465c78275fece8a41c884b7c94411f3a58a7e8e1.jpg)

![eddddbc354d8483d710e0cf8f32a43eb772622e831deea190fcdff50dc484906.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/eddddbc354d8483d710e0cf8f32a43eb772622e831deea190fcdff50dc484906.jpg)

![efcc32618e12a5609a1cf80a70f5cfcd75e669e4dafd805900ae4a94649325be.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/images/efcc32618e12a5609a1cf80a70f5cfcd75e669e4dafd805900ae4a94649325be.jpg)

### Tables

![05032ec3fb7542d0cfe5fbac08f38db4b1f2fc090d2b075bbcfbb9bf432ed1c9.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/05032ec3fb7542d0cfe5fbac08f38db4b1f2fc090d2b075bbcfbb9bf432ed1c9.jpg)

![2254d3e5b33d10977a6d15a61f4554585c9122d8c814d0c5dccd29e2f840aa84.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/2254d3e5b33d10977a6d15a61f4554585c9122d8c814d0c5dccd29e2f840aa84.jpg)

![48c966922feaf753929f0756f6ddefadf7a688f664cb3aadeae69c163d1a903a.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/48c966922feaf753929f0756f6ddefadf7a688f664cb3aadeae69c163d1a903a.jpg)

![531943aa1e7ed0a50095fc1b770876c13c8a6725a21c868dd326fe2f7312a9eb.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/531943aa1e7ed0a50095fc1b770876c13c8a6725a21c868dd326fe2f7312a9eb.jpg)

![6ba17154ff17dce28db2fa6a1aa5144a0908b80baa37fc8f11ca951ccc38f520.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/6ba17154ff17dce28db2fa6a1aa5144a0908b80baa37fc8f11ca951ccc38f520.jpg)

![800d3470edb86f147841dad42366ff9eedc60ab9487675390d71d5794e6866ae.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/800d3470edb86f147841dad42366ff9eedc60ab9487675390d71d5794e6866ae.jpg)

![86668fc12045e12149c0ae941c8d4deb608dc856e3ae56ea7d3123f867cd1f89.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/86668fc12045e12149c0ae941c8d4deb608dc856e3ae56ea7d3123f867cd1f89.jpg)

![9dce1c314d5091f65a5920f4400a521802c123cbb47af925ad9c5f6aa2349cf2.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/9dce1c314d5091f65a5920f4400a521802c123cbb47af925ad9c5f6aa2349cf2.jpg)

![d204425702dd014b4b034cf8c35192a4bc8f1e449e89b6afdbde912aa7989bdf.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/d204425702dd014b4b034cf8c35192a4bc8f1e449e89b6afdbde912aa7989bdf.jpg)

![e1a228572a89307587e24b2eca652cee2f90044d44e202967fceea7cfdc6dc4e.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/e1a228572a89307587e24b2eca652cee2f90044d44e202967fceea7cfdc6dc4e.jpg)

![f9a34e4cf6f2511837aab7138dd5030fa5754b331236eb98f9cf351bafdf44e1.jpg](../icml_results/4943_Sparser%20Block-Sparse%20Attention%20via%20Token%20Permutation/tables/f9a34e4cf6f2511837aab7138dd5030fa5754b331236eb98f9cf351bafdf44e1.jpg)

<a id="paper-4896"></a>

## GradPower: Powering Gradients for Faster Language Model Pre-Training


### Images

![2f49adfc3da30f58d4e2de3e355ed43e182fb3c8e9421f6af6ca1f49b645821c.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/2f49adfc3da30f58d4e2de3e355ed43e182fb3c8e9421f6af6ca1f49b645821c.jpg)

![300021e79536b3cbe90a54810026ad1e30aab537bacecedd5f8916d42021f693.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/300021e79536b3cbe90a54810026ad1e30aab537bacecedd5f8916d42021f693.jpg)

![4363793bf11266207ce233cfe9593a22c168ab6bff4b2e787231ee42d5221e53.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/4363793bf11266207ce233cfe9593a22c168ab6bff4b2e787231ee42d5221e53.jpg)

![6a728525af1e87ee328e132109a0f43d8a5ac46c99d01d2b78a1d9478b17b009.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/6a728525af1e87ee328e132109a0f43d8a5ac46c99d01d2b78a1d9478b17b009.jpg)

![818e3c27780c4f96488d3f0d08b6b96f4ed5fe79c3a129dad96e3d540874f7d3.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/818e3c27780c4f96488d3f0d08b6b96f4ed5fe79c3a129dad96e3d540874f7d3.jpg)

![bbe3955f60436f9f6788d4c4fb63309d75a492812edf0d0905cca2d1f1837425.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/bbe3955f60436f9f6788d4c4fb63309d75a492812edf0d0905cca2d1f1837425.jpg)

![cbc99c76cac3ff5b91e976a6a527f411639020e969341d43839fc82c22ab46e8.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/cbc99c76cac3ff5b91e976a6a527f411639020e969341d43839fc82c22ab46e8.jpg)

![f45f074e2dbabf8b74df001a15b46832f3026c49ae573a741d5287e30e550d53.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/images/f45f074e2dbabf8b74df001a15b46832f3026c49ae573a741d5287e30e550d53.jpg)

### Tables

![5e94f1c8aff0c41ccb6d6b55eeb943055aef58070e776216dc8f2eca35ff9dc1.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/tables/5e94f1c8aff0c41ccb6d6b55eeb943055aef58070e776216dc8f2eca35ff9dc1.jpg)

![80e43cfbf12238346cab2283370d298bf21fb350e44807c48a2c61d1f52904d2.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/tables/80e43cfbf12238346cab2283370d298bf21fb350e44807c48a2c61d1f52904d2.jpg)

![9f6960a5ad4f5a6702add684b8db4971c82efa31ef1256e0bf00d5f641f5df3b.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/tables/9f6960a5ad4f5a6702add684b8db4971c82efa31ef1256e0bf00d5f641f5df3b.jpg)

![edf9187e282ff2aebefa17d76993002de11b370b62031c952b36d989d1b26511.jpg](../icml_results/4944_GradPower_%20Powering%20Gradients%20for%20Faster%20Language%20Model%20Pre-Training/tables/edf9187e282ff2aebefa17d76993002de11b370b62031c952b36d989d1b26511.jpg)

<a id="paper-4897"></a>

## Breaking Manifold Continuity: Vector Quantized Modeling for Real-Centric Deepfake Detection


### Images

![3063f3337b4fa327289a9e7ad4ba75a0a79175634a902911128aaa491767120f.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/3063f3337b4fa327289a9e7ad4ba75a0a79175634a902911128aaa491767120f.jpg)

![38506c85eea9e6c422569f3fd3a4005c3e577d510c3ce05b91b77f7ceaee1ace.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/38506c85eea9e6c422569f3fd3a4005c3e577d510c3ce05b91b77f7ceaee1ace.jpg)

![3b91ecfe113b7d20c6bedc4992ff88b3b5db0ff7893e31482617bbfe9a736c58.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/3b91ecfe113b7d20c6bedc4992ff88b3b5db0ff7893e31482617bbfe9a736c58.jpg)

![3dedb0de88d49d73e0d8050c43910ba2a90e9b2f4416eaa3cad618758777c521.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/3dedb0de88d49d73e0d8050c43910ba2a90e9b2f4416eaa3cad618758777c521.jpg)

![4dff91654a7ab7aa67f5dd92e025578dee754d42889b029f53ee0255a7e5cf28.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/4dff91654a7ab7aa67f5dd92e025578dee754d42889b029f53ee0255a7e5cf28.jpg)

![58626d1461b6de747eb6e19fdba12c189d38e78b80223263ceb16cc9ab922d8a.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/58626d1461b6de747eb6e19fdba12c189d38e78b80223263ceb16cc9ab922d8a.jpg)

![6e3228143965b65a6f4db3df7e1fb82df0e502e45db015ff7a05346edc0ca2be.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/6e3228143965b65a6f4db3df7e1fb82df0e502e45db015ff7a05346edc0ca2be.jpg)

![9d63930aecf825534b89cc60c3f98b0552acf4fa290d70f70a3fc46622ae93f1.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/9d63930aecf825534b89cc60c3f98b0552acf4fa290d70f70a3fc46622ae93f1.jpg)

![a011cf4759ef817f6a5c2a06efe09297e18c9a804b40598d2038abeacab83356.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/a011cf4759ef817f6a5c2a06efe09297e18c9a804b40598d2038abeacab83356.jpg)

![d6c90a87db93fc62c027b920654a2c853e29a79b7e2dbf62a1a481acfb2d2669.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/images/d6c90a87db93fc62c027b920654a2c853e29a79b7e2dbf62a1a481acfb2d2669.jpg)

### Tables

![0854c7c1b14a7cee1c9c00884cf2633fa1d656436485d619964a79471989ef9a.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/tables/0854c7c1b14a7cee1c9c00884cf2633fa1d656436485d619964a79471989ef9a.jpg)

![40efbeb740d82e4e226d6bd00a54d97894b6b5afef937b4f604ea083ec239bef.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/tables/40efbeb740d82e4e226d6bd00a54d97894b6b5afef937b4f604ea083ec239bef.jpg)

![94aae84e68ce66c9c2df1a286584bc2e1d1784e747188cf2de628797ed02c81a.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/tables/94aae84e68ce66c9c2df1a286584bc2e1d1784e747188cf2de628797ed02c81a.jpg)

![99030de9c751a4f31e8b689bd0e8bc9f9e462b5492a15229a730b84ea5bd2361.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/tables/99030de9c751a4f31e8b689bd0e8bc9f9e462b5492a15229a730b84ea5bd2361.jpg)

![e0fa6dcc95b05c051027314ea15b2d8df16a78bba4ae91586491fa58c503356a.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/tables/e0fa6dcc95b05c051027314ea15b2d8df16a78bba4ae91586491fa58c503356a.jpg)

![ee28dc3f5cba53f968ab860f3dace92ed239de25bdf072f73b73840e5fb6ea62.jpg](../icml_results/4945_Breaking%20Manifold%20Continuity_%20Vector%20Quantized%20Modeling%20for%20Real-Centric%20Deepfake%20Detection/tables/ee28dc3f5cba53f968ab860f3dace92ed239de25bdf072f73b73840e5fb6ea62.jpg)

<a id="paper-4898"></a>

## WorldCompass: Reinforcement Learning for Long-Horizon World Models


### Images

![1686437071c361c35b49d9df11708fc7b6f5e77facb6b4fd72ee3a6519d65b51.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/1686437071c361c35b49d9df11708fc7b6f5e77facb6b4fd72ee3a6519d65b51.jpg)

![2c9d80b4d74a2cc5c57ea640b27b00a8df07b0d87893f69dd1a167dbf9494998.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/2c9d80b4d74a2cc5c57ea640b27b00a8df07b0d87893f69dd1a167dbf9494998.jpg)

![734c9515c7966000092d188966013867790ea6b835b46d55a8b591b2796b31a0.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/734c9515c7966000092d188966013867790ea6b835b46d55a8b591b2796b31a0.jpg)

![76a5f6227efe841a5c9941d6eca24ec3047d9e5823b734915e96a1393239925d.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/76a5f6227efe841a5c9941d6eca24ec3047d9e5823b734915e96a1393239925d.jpg)

![d7175b6d5fe5497a91ed9ce8273fc74da1e7075dcacde8fd2df2b7519d1f0361.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/d7175b6d5fe5497a91ed9ce8273fc74da1e7075dcacde8fd2df2b7519d1f0361.jpg)

![e5f4acd8047f61420a5c056f804307f04ba5da5aa9b01184a2115bbc842651c4.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/e5f4acd8047f61420a5c056f804307f04ba5da5aa9b01184a2115bbc842651c4.jpg)

![f0253e72a524c92624cecb6ed5111abe029647a6a1f70fba61efc5bc58738645.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/f0253e72a524c92624cecb6ed5111abe029647a6a1f70fba61efc5bc58738645.jpg)

![fa62adb02c595cb1d7c9aeb473c5e95cbfb2398c302744a7f60b3f2b338e1a40.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/fa62adb02c595cb1d7c9aeb473c5e95cbfb2398c302744a7f60b3f2b338e1a40.jpg)

![fe6152328577decfa2c2f8776958e8dc0d4f2adcaa1517481e1517a301e8ba41.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/images/fe6152328577decfa2c2f8776958e8dc0d4f2adcaa1517481e1517a301e8ba41.jpg)

### Tables

![17b6d13270eafa869e5d78ccf688e56153ded15432955414934e39a876692bc1.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/tables/17b6d13270eafa869e5d78ccf688e56153ded15432955414934e39a876692bc1.jpg)

![6a08bb87b2dac22e9304077d821a3f37b0ac874b8bb40c14914f00f6b2ba1c16.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/tables/6a08bb87b2dac22e9304077d821a3f37b0ac874b8bb40c14914f00f6b2ba1c16.jpg)

![d991ca1c80dba6a514cfff019e0ee6d059a5970c0bc8c4ee664303696dc504dd.jpg](../icml_results/4946_WorldCompass_%20Reinforcement%20Learning%20for%20Long-Horizon%20World%20Models/tables/d991ca1c80dba6a514cfff019e0ee6d059a5970c0bc8c4ee664303696dc504dd.jpg)

<a id="paper-4899"></a>

## REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation


### Images

![11b3df5ccf8f9bef27902f098c476e6d82de0e02e6f8ff60cfa1fe95a7726565.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/11b3df5ccf8f9bef27902f098c476e6d82de0e02e6f8ff60cfa1fe95a7726565.jpg)

![42023c1e37a4c515f95262b951fa7f1c07fe33f25b8d86b37a666ab2a8563608.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/42023c1e37a4c515f95262b951fa7f1c07fe33f25b8d86b37a666ab2a8563608.jpg)

![461a10fd81cf352f6d8fa048ce2551e8fd5dee5684ec35570ae031a5e0b03efc.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/461a10fd81cf352f6d8fa048ce2551e8fd5dee5684ec35570ae031a5e0b03efc.jpg)

![7209f6898286993c51ed448bb727cf1a74c43b4475bfd6ab887006d72a27ed66.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/7209f6898286993c51ed448bb727cf1a74c43b4475bfd6ab887006d72a27ed66.jpg)

![7a7e0d8d52734217bd81d859a07c90bd971692bd5adff544e6d4984b98336425.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/7a7e0d8d52734217bd81d859a07c90bd971692bd5adff544e6d4984b98336425.jpg)

![8dd5c0fc8619e1f14ab52e042d6c237f7b77f31daa0ba21c22fa2113c11a484c.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/8dd5c0fc8619e1f14ab52e042d6c237f7b77f31daa0ba21c22fa2113c11a484c.jpg)

![a00d6d357ebb47b18fa39f10fe7512126a27b490810d403def2bba8bc73c095d.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/a00d6d357ebb47b18fa39f10fe7512126a27b490810d403def2bba8bc73c095d.jpg)

![a9ca15c448dcd334e5276b489b19f805114dea3c36c83528d4be0aa6e5a116a6.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/a9ca15c448dcd334e5276b489b19f805114dea3c36c83528d4be0aa6e5a116a6.jpg)

![c620847bd13bbb2dbe1e2a3a4e69c394a3dbba036e18dbf91ff3bec7cdb0d747.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/c620847bd13bbb2dbe1e2a3a4e69c394a3dbba036e18dbf91ff3bec7cdb0d747.jpg)

![db010b3c0375972521312917ef4855d5a1f72381f5979e346e2a07e0cc0f6a85.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/db010b3c0375972521312917ef4855d5a1f72381f5979e346e2a07e0cc0f6a85.jpg)

![e95d64269d103c2b8767ec715949767a401b9a0c35324eac9217de1d07a52c0d.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/images/e95d64269d103c2b8767ec715949767a401b9a0c35324eac9217de1d07a52c0d.jpg)

### Tables

![282928821dfeeb06b39456bb8ea1a2cb3850f81aec5647c080a8056460456231.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/tables/282928821dfeeb06b39456bb8ea1a2cb3850f81aec5647c080a8056460456231.jpg)

![748e465ad21ac4fd19284cadc9958c7ff82bfb686afc84fd1b1208d0107283c0.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/tables/748e465ad21ac4fd19284cadc9958c7ff82bfb686afc84fd1b1208d0107283c0.jpg)

![8c3479d282734f3131a8542f8e60f10af88323bb9d05d8f4d8ec2e00ec83f9d3.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/tables/8c3479d282734f3131a8542f8e60f10af88323bb9d05d8f4d8ec2e00ec83f9d3.jpg)

![9f678e05fd132e49c645a76dba36413ea79d004739fbc735cce1f245d3b32be8.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/tables/9f678e05fd132e49c645a76dba36413ea79d004739fbc735cce1f245d3b32be8.jpg)

![b8192b5c5b9a38382eea70c7a021e9ab9d9f55fa5832d58ee219414175fd79a6.jpg](../icml_results/4947_REST_%20Diffusion-based%20Real-time%20End-to-end%20Streaming%20Talking%20Head%20Generation%20via%20ID-Context%20Caching%20/tables/b8192b5c5b9a38382eea70c7a021e9ab9d9f55fa5832d58ee219414175fd79a6.jpg)

<a id="paper-4900"></a>

## SL-VC: A Benchmark and Automated Framework for Separation Logic Verification Condition Proving


### Images

![2a39a0b4e66bda56de4e30f3a8bcd17f9b38965530d56cfd87c21baf3e61a85d.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/images/2a39a0b4e66bda56de4e30f3a8bcd17f9b38965530d56cfd87c21baf3e61a85d.jpg)

![736fc6dec7f2cfa3dca40f61ef8bb04581c4fe73fee923f6aba2c3ad02d6226d.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/images/736fc6dec7f2cfa3dca40f61ef8bb04581c4fe73fee923f6aba2c3ad02d6226d.jpg)

![e5c9ba889bd113d0634563ffcc379c8f08b9fd03fdf914ad8cb2485a56a0221b.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/images/e5c9ba889bd113d0634563ffcc379c8f08b9fd03fdf914ad8cb2485a56a0221b.jpg)

### Tables

![058e486cd1c7fb7e8a9abdc9319ab43d54539688fb5230254b48abaccf339e25.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/058e486cd1c7fb7e8a9abdc9319ab43d54539688fb5230254b48abaccf339e25.jpg)

![05f2714d8af5fae1f091c7f3446483f0f1ecd4dbc88519df1ef7b4f7408ed82b.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/05f2714d8af5fae1f091c7f3446483f0f1ecd4dbc88519df1ef7b4f7408ed82b.jpg)

![06d2cccb5636bf4dbb26047f1f612c0047ac8fdf9f350643c12c78853b6b96b9.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/06d2cccb5636bf4dbb26047f1f612c0047ac8fdf9f350643c12c78853b6b96b9.jpg)

![1a164d5b6b63582136f76afcd071e025359f843e70c22a72a7ab41dc611102ab.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/1a164d5b6b63582136f76afcd071e025359f843e70c22a72a7ab41dc611102ab.jpg)

![1d6d583f1d0d59f270f27afeb8eaf2443f3c10cde131d7d6c9b613a16f2294aa.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/1d6d583f1d0d59f270f27afeb8eaf2443f3c10cde131d7d6c9b613a16f2294aa.jpg)

![337dec0ee76b866f22698e70fb285e4b3ac8f5889cddadd56246de6a944b7a62.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/337dec0ee76b866f22698e70fb285e4b3ac8f5889cddadd56246de6a944b7a62.jpg)

![51f0a40b9de64eb0d73dd2e3f867ffb095d193a6cfef8549007ba6a91c45c790.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/51f0a40b9de64eb0d73dd2e3f867ffb095d193a6cfef8549007ba6a91c45c790.jpg)

![5269b76c5bd230718ff08f12633fb933c3b6f7ba6814faff66ff10dc171bc903.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/5269b76c5bd230718ff08f12633fb933c3b6f7ba6814faff66ff10dc171bc903.jpg)

![549269682ba351d6e929d8f738225db5c2e43f73e07af6179b0ecbbb1a299189.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/549269682ba351d6e929d8f738225db5c2e43f73e07af6179b0ecbbb1a299189.jpg)

![563c0a83327c1cb74129321a2f40a2390fbf39acbd469897e8ee32dc390cb575.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/563c0a83327c1cb74129321a2f40a2390fbf39acbd469897e8ee32dc390cb575.jpg)

![56c80d98fa8e2476223acd0453af995ce5eecfebf0f7bf88f7801200832524fc.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/56c80d98fa8e2476223acd0453af995ce5eecfebf0f7bf88f7801200832524fc.jpg)

![592b212cd18b477338c58be3bbf7dab96e9d28900830ff23bd03f3c8d1157d46.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/592b212cd18b477338c58be3bbf7dab96e9d28900830ff23bd03f3c8d1157d46.jpg)

![6c1888138176b9fae2389ae9488307ce6c36d0d3ae5fa854347580ac3b3ef753.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/6c1888138176b9fae2389ae9488307ce6c36d0d3ae5fa854347580ac3b3ef753.jpg)

![6c85c82c5c3e6e1d75686d888b63f5b424e22bc87d91f5256de562d7a8d5a8a6.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/6c85c82c5c3e6e1d75686d888b63f5b424e22bc87d91f5256de562d7a8d5a8a6.jpg)

![7592b4a157fd711cbd165e36496b95cae963e0372d68f3e2fa466af1fc4d696b.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/7592b4a157fd711cbd165e36496b95cae963e0372d68f3e2fa466af1fc4d696b.jpg)

![7d363cfc0ac7df4d9f65f5dc8f79fb4e85f35bca50f9ca82f3d228f13820ba48.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/7d363cfc0ac7df4d9f65f5dc8f79fb4e85f35bca50f9ca82f3d228f13820ba48.jpg)

![7fc4fac3bd8ee10ed245557662b319f2972c0d533f7125a92498059c8df3c2da.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/7fc4fac3bd8ee10ed245557662b319f2972c0d533f7125a92498059c8df3c2da.jpg)

![8e1779c513f256cc396180148f4d2a353f3d2f66aa5dd1daf59bd2b98d5da62e.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/8e1779c513f256cc396180148f4d2a353f3d2f66aa5dd1daf59bd2b98d5da62e.jpg)

![8f17cb21275daab92d54fecdb85ac66357552b2adeab969936279afe7df66327.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/8f17cb21275daab92d54fecdb85ac66357552b2adeab969936279afe7df66327.jpg)

![915ad349faaa926361705f101cabf843a034c67f9860d520b4d0c9c8ad75fa5b.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/915ad349faaa926361705f101cabf843a034c67f9860d520b4d0c9c8ad75fa5b.jpg)

![930c80d7e1b79149e13e422613718c425f503598cb8faa4c29b0e08ee903c1a3.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/930c80d7e1b79149e13e422613718c425f503598cb8faa4c29b0e08ee903c1a3.jpg)

![9a857225af5875955a8dc416aaeba5fd68119d5ef54d3a4935e57875ea912742.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/9a857225af5875955a8dc416aaeba5fd68119d5ef54d3a4935e57875ea912742.jpg)

![9ef6791c6e1602b8b93b6c7a678883e53bf35408ecda8f1caaab25877765547e.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/9ef6791c6e1602b8b93b6c7a678883e53bf35408ecda8f1caaab25877765547e.jpg)

![aaaa9e5a4c00dd34f1715cd7b738a9429de2b9c67c36ceead6e265bc81a41ed0.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/aaaa9e5a4c00dd34f1715cd7b738a9429de2b9c67c36ceead6e265bc81a41ed0.jpg)

![b08366d9c0f943c1672e91869477e85fe1a6c31ac0ff43bdc0632fa78e292505.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/b08366d9c0f943c1672e91869477e85fe1a6c31ac0ff43bdc0632fa78e292505.jpg)

![cf5b1d663aef0adbac1b9cd42b865e3cd978eede5b5091a534c928edc291ffa4.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/cf5b1d663aef0adbac1b9cd42b865e3cd978eede5b5091a534c928edc291ffa4.jpg)

![d208fbafb40881ca7ce5ca6c8173ad1e3b42ae009dd11523ac8418d670053bde.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/d208fbafb40881ca7ce5ca6c8173ad1e3b42ae009dd11523ac8418d670053bde.jpg)

![d2ebfcd1c5cdb115c92efd342198af1498acb425eca923596ea22c91509d83aa.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/d2ebfcd1c5cdb115c92efd342198af1498acb425eca923596ea22c91509d83aa.jpg)

![d4f64013c1488e62ef582b25244833f3b684f4775c240a8e0fc86a96a7961a10.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/d4f64013c1488e62ef582b25244833f3b684f4775c240a8e0fc86a96a7961a10.jpg)

![d882a6decb337a591a658aa4553241a63e782bebdd7bf91eb1727f5b3da921ab.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/d882a6decb337a591a658aa4553241a63e782bebdd7bf91eb1727f5b3da921ab.jpg)

![df061fe1c6db9f36ccb040449385fb7a3b2b89e3cd33d511a413c7f256548095.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/df061fe1c6db9f36ccb040449385fb7a3b2b89e3cd33d511a413c7f256548095.jpg)

![ee661d6cd49416c78278ac145aee2178173d210365aad6a2bead3729c973142f.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/ee661d6cd49416c78278ac145aee2178173d210365aad6a2bead3729c973142f.jpg)

![ee6ee11433c29e19d0e09b3b13a6727155bfc5c8873c0a980fc6c82a971cbce8.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/ee6ee11433c29e19d0e09b3b13a6727155bfc5c8873c0a980fc6c82a971cbce8.jpg)

![ef4247140d0dbfb2cf1006355e96443d5e3739f4da2f5675244445df12b8aa87.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/ef4247140d0dbfb2cf1006355e96443d5e3739f4da2f5675244445df12b8aa87.jpg)

![fcdc5d4aa59e60b819727fa2177f69d14163193b033d8e1a3f04c1dba5df90d4.jpg](../icml_results/4948_SL-VC_%20A%20Benchmark%20and%20Automated%20Framework%20for%20Separation%20Logic%20Verification%20Condition%20Proving/tables/fcdc5d4aa59e60b819727fa2177f69d14163193b033d8e1a3f04c1dba5df90d4.jpg)

<a id="paper-4901"></a>

## ScaleErasure: Inference-Time Minimal Intervention for Precise Concept Erasure in Next-Scale Autoregressive Image Generation


### Images

![1d2f847f4ce846de82d123a4bdaa053018aeb7a6deb502411d6e5cbddfeeb878.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/1d2f847f4ce846de82d123a4bdaa053018aeb7a6deb502411d6e5cbddfeeb878.jpg)

![26d2c43b9e2d9851d51ce2defc80fba2ddb1af9dfe091eb78fa1f404256416d7.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/26d2c43b9e2d9851d51ce2defc80fba2ddb1af9dfe091eb78fa1f404256416d7.jpg)

![438a61e5402b9899051d07ddb803da9879802a727582d13eba8192d3ea724769.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/438a61e5402b9899051d07ddb803da9879802a727582d13eba8192d3ea724769.jpg)

![6b2be157ff1571ea8e6674ca2d26add294554d6f891d775b1121b9132cf3d1f1.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/6b2be157ff1571ea8e6674ca2d26add294554d6f891d775b1121b9132cf3d1f1.jpg)

![6ca86af347a53276852a6603d5150f8102bdf891e43174992287c1d76c367bdc.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/6ca86af347a53276852a6603d5150f8102bdf891e43174992287c1d76c367bdc.jpg)

![82b2275bcaf84e0e61407fe8e5a2100d5f69247ddef6c2921265f99942130be7.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/82b2275bcaf84e0e61407fe8e5a2100d5f69247ddef6c2921265f99942130be7.jpg)

![c0cc6ff37e0b5489b9c69cc6261f760697f344b8b63b68ada7200985f662f737.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/c0cc6ff37e0b5489b9c69cc6261f760697f344b8b63b68ada7200985f662f737.jpg)

![c16e2ffb1469f3071cb5366abe6fc104f000ffbc9e993f2d25869da713d9856d.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/c16e2ffb1469f3071cb5366abe6fc104f000ffbc9e993f2d25869da713d9856d.jpg)

![e9dcf75371a67a12a138429890179f6d31bf497c899f8e8ee4eca7c3f4a98f38.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/e9dcf75371a67a12a138429890179f6d31bf497c899f8e8ee4eca7c3f4a98f38.jpg)

![ed0901bd783e0884ebe869830938f5e9742866508ab30e680d56d982bf62d032.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/images/ed0901bd783e0884ebe869830938f5e9742866508ab30e680d56d982bf62d032.jpg)

### Tables

![0a71bc42a96f742984931ef7124eac61188802eabc717d78a073717e938af916.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/0a71bc42a96f742984931ef7124eac61188802eabc717d78a073717e938af916.jpg)

![2185a1b5a1ae7725125acdfe7b192ead9a0a836a2975d06855e2c161c1eab339.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/2185a1b5a1ae7725125acdfe7b192ead9a0a836a2975d06855e2c161c1eab339.jpg)

![6cec32f1ca5981d2f8e61c6f180501c5f63d75b3ebc8758505db04316b1823c6.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/6cec32f1ca5981d2f8e61c6f180501c5f63d75b3ebc8758505db04316b1823c6.jpg)

![8892a3a38eb44f8d179933cf05b5384dfe2c0238fef0c5770e752563a191dd86.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/8892a3a38eb44f8d179933cf05b5384dfe2c0238fef0c5770e752563a191dd86.jpg)

![95a932db4ee03459db261d358ef9fd4e32a863178c9ce47677990898c2ef4715.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/95a932db4ee03459db261d358ef9fd4e32a863178c9ce47677990898c2ef4715.jpg)

![98c6545c3da62fb22f2464e74ab4c153b01a47846ef0bbe746fd142595c7d329.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/98c6545c3da62fb22f2464e74ab4c153b01a47846ef0bbe746fd142595c7d329.jpg)

![9ce602df52361a5d5234560e6cd5e672680f1fba70dd3ba2673a2da44949d763.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/9ce602df52361a5d5234560e6cd5e672680f1fba70dd3ba2673a2da44949d763.jpg)

![9f613d7e851aa8a47e47f26f08aa51c5c0d82056305ebfad7aa3aca9ee20f731.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/9f613d7e851aa8a47e47f26f08aa51c5c0d82056305ebfad7aa3aca9ee20f731.jpg)

![c631b7729830e5c1fd789317a49d4a23a07c45bcd9d5b3a3c2d4bc36d80cf89f.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/c631b7729830e5c1fd789317a49d4a23a07c45bcd9d5b3a3c2d4bc36d80cf89f.jpg)

![cf70cacea6b2417e91ada03c334b6667e85955059b1c91675f382b59ce845920.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/cf70cacea6b2417e91ada03c334b6667e85955059b1c91675f382b59ce845920.jpg)

![ddff60dd6ac1c381727e081f2b70510ff2dc5c9401b6aacdad39b3bfbbc62b85.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/ddff60dd6ac1c381727e081f2b70510ff2dc5c9401b6aacdad39b3bfbbc62b85.jpg)

![f89b63c415a20d9a34541a225840cd3decbcfd01acaac3fb132eb90664a1a139.jpg](../icml_results/4949_ScaleErasure_%20Inference-Time%20Minimal%20Intervention%20for%20Precise%20Concept%20Erasure%20in%20Next-Scale%20Autoregr/tables/f89b63c415a20d9a34541a225840cd3decbcfd01acaac3fb132eb90664a1a139.jpg)

<a id="paper-4902"></a>

## Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization


### Images

![287012b8477deefb84144a98dd87dffcb169b48f5ca9c23b71b9c7fbeb9e4b55.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/287012b8477deefb84144a98dd87dffcb169b48f5ca9c23b71b9c7fbeb9e4b55.jpg)

![4d3aeeaae7405e20983cc528971ac3ced74b17283db97b00455d89fa66eea730.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/4d3aeeaae7405e20983cc528971ac3ced74b17283db97b00455d89fa66eea730.jpg)

![6e1cb7b22c014229574025fa14bd77d4c77c94c025d0c2daf70b2d8c8de0ef3e.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/6e1cb7b22c014229574025fa14bd77d4c77c94c025d0c2daf70b2d8c8de0ef3e.jpg)

![874c09c60c0d4f747d70ace00b2dc689723d0a6fe89c7963fbf474b9059d7a1b.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/874c09c60c0d4f747d70ace00b2dc689723d0a6fe89c7963fbf474b9059d7a1b.jpg)

![a6bfcc0c652cc37d88de2f89234a4f2a8ee5891788c555266587187aaaf52d16.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/a6bfcc0c652cc37d88de2f89234a4f2a8ee5891788c555266587187aaaf52d16.jpg)

![e85451778732eb61a3181d823c0a8628a28d0d602660753092a64143116432ac.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/e85451778732eb61a3181d823c0a8628a28d0d602660753092a64143116432ac.jpg)

![f480afd511e3b44fe62e413716a9adbf5611e860efca3c6b929b4ce6f901024a.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/images/f480afd511e3b44fe62e413716a9adbf5611e860efca3c6b929b4ce6f901024a.jpg)

### Tables

![12553f7c88285d843afcf65dacff236a9f499caf6ce93c20b44977e657c0ff5e.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/12553f7c88285d843afcf65dacff236a9f499caf6ce93c20b44977e657c0ff5e.jpg)

![2d3c8ed606ae8c3b2cc2d86b10bbf5e0798644f40bae2b71a8d88c155258dca1.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/2d3c8ed606ae8c3b2cc2d86b10bbf5e0798644f40bae2b71a8d88c155258dca1.jpg)

![347fda9ecb86748191fd9d5bf52cd019baf47dbac180b117ba55cb48baba18cc.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/347fda9ecb86748191fd9d5bf52cd019baf47dbac180b117ba55cb48baba18cc.jpg)

![bd49400cf38a42eae79f74f9f8e3f7dcb859625ef68641c5ef2dea136d4ef3d4.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/bd49400cf38a42eae79f74f9f8e3f7dcb859625ef68641c5ef2dea136d4ef3d4.jpg)

![c5ffc5f42908670ecd6611220435a3d1a736f91956fec0b0d445842ade3e4123.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/c5ffc5f42908670ecd6611220435a3d1a736f91956fec0b0d445842ade3e4123.jpg)

![d13e2d8ce7e235b0881641e3d768495ee927ef91b6150bfe960de7509e4d0099.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/d13e2d8ce7e235b0881641e3d768495ee927ef91b6150bfe960de7509e4d0099.jpg)

![e9a28851e5ccbf677c539e2754f9e318d768c732721acedb142c9dd17a7cd266.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/e9a28851e5ccbf677c539e2754f9e318d768c732721acedb142c9dd17a7cd266.jpg)

![f3f43bb3fb07a8ec89724142966b10b68277da9b3c049cb2cb31711e70364895.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/f3f43bb3fb07a8ec89724142966b10b68277da9b3c049cb2cb31711e70364895.jpg)

![f874880b33ef1fc9671c3716c02c40add2e47937dcf993e6bc469564c29705a9.jpg](../icml_results/4950_Order%20within%20Chaos_%20Capturing%20Intrinsic%20Energy%20Anomalies%20for%20AI-Manipulated%20Image%20Forgery%20Localizati/tables/f874880b33ef1fc9671c3716c02c40add2e47937dcf993e6bc469564c29705a9.jpg)

<a id="paper-4903"></a>

## Mixing Configurations for Downstream Prediction


### Images

![0fd4cdfafcd31ba75b6162af0562cac79f5876ec490be55d24687f0378519a7a.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/images/0fd4cdfafcd31ba75b6162af0562cac79f5876ec490be55d24687f0378519a7a.jpg)

![9af79826cc02b1ab19f6a01cf759d2baa866733d0a4ce7256a0f2039d321e302.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/images/9af79826cc02b1ab19f6a01cf759d2baa866733d0a4ce7256a0f2039d321e302.jpg)

![9f1135294a3ee28f96550c2556e03836a6be532118e4272c4a38f97afc57164b.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/images/9f1135294a3ee28f96550c2556e03836a6be532118e4272c4a38f97afc57164b.jpg)

![b1a7de953d2c2760162f0a31e485955ae67dfb2010890d36334a6166c24b3c1f.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/images/b1a7de953d2c2760162f0a31e485955ae67dfb2010890d36334a6166c24b3c1f.jpg)

![e8830d5975afcbf9416c915d41b99451c7a4af4f5f7b3d4e1862cf86108e45a5.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/images/e8830d5975afcbf9416c915d41b99451c7a4af4f5f7b3d4e1862cf86108e45a5.jpg)

### Tables

![0668836a3f796db27de43788045790541b728d669dfef0667c370529c548eaf5.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/0668836a3f796db27de43788045790541b728d669dfef0667c370529c548eaf5.jpg)

![1de386a639ace67bf554bd3faa3a2e309a56f2a38747309c16161d3950ce6e0d.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/1de386a639ace67bf554bd3faa3a2e309a56f2a38747309c16161d3950ce6e0d.jpg)

![40299dd71eb87df2e6064679b3437ac638a31fb76420365dd8bbfc6f00802879.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/40299dd71eb87df2e6064679b3437ac638a31fb76420365dd8bbfc6f00802879.jpg)

![4862eedc798bc8c3bbcf771b44849258bdc00d6deb57ccebbae54822edb7aa16.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/4862eedc798bc8c3bbcf771b44849258bdc00d6deb57ccebbae54822edb7aa16.jpg)

![669259832f57a1d4f070304ad8f32244d74cddbf572ebb6fcdde135b0c063a7e.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/669259832f57a1d4f070304ad8f32244d74cddbf572ebb6fcdde135b0c063a7e.jpg)

![78132a817d030587764b4eba0613e0b798c0a7ad1d2b908375f012f43ee3488f.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/78132a817d030587764b4eba0613e0b798c0a7ad1d2b908375f012f43ee3488f.jpg)

![7c7723d5045cda5b68aa663bef592421c3d7d0596ef425c6d2952a5e63eb68e2.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/7c7723d5045cda5b68aa663bef592421c3d7d0596ef425c6d2952a5e63eb68e2.jpg)

![7e02cd00f9280053a8c03e1ca65b87687c3815e8f32bd35061d57bc7d0f48389.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/7e02cd00f9280053a8c03e1ca65b87687c3815e8f32bd35061d57bc7d0f48389.jpg)

![d7973a936effe5e8f40bc87e478f0bf2b48c969320487742b1b8f284aa872dfd.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/d7973a936effe5e8f40bc87e478f0bf2b48c969320487742b1b8f284aa872dfd.jpg)

![dfa236c096e87fce40b8f9220ab2b232c9c8ccfe913b0b2a9b7b613d734d9f7c.jpg](../icml_results/4951_Mixing%20Configurations%20for%20Downstream%20Prediction/tables/dfa236c096e87fce40b8f9220ab2b232c9c8ccfe913b0b2a9b7b613d734d9f7c.jpg)

<a id="paper-4904"></a>

## Beyond Sample-Level Forgetting: Improving Reliability in Multimodal Unlearning


### Images

![13ecd6abf7f633d0e641ec53e20da8b88bd0925afcf831191d3e01c517822c83.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/13ecd6abf7f633d0e641ec53e20da8b88bd0925afcf831191d3e01c517822c83.jpg)

![24414a1a6daa2217a287fdadcdf8f0029c4e29d164258dec1bc86f0e755c6c62.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/24414a1a6daa2217a287fdadcdf8f0029c4e29d164258dec1bc86f0e755c6c62.jpg)

![2cb1212c7da105b1a5a688d095ae3649606a36f94bb04f06150dae7e5e54ef66.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/2cb1212c7da105b1a5a688d095ae3649606a36f94bb04f06150dae7e5e54ef66.jpg)

![48aeb48dc35ebbf77ccac5a3f25f849f74f9dc619119fe921fe176913a876ff1.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/48aeb48dc35ebbf77ccac5a3f25f849f74f9dc619119fe921fe176913a876ff1.jpg)

![55a3f9294d6184176af813d4f993c9b81f96d997061609b6759e54356bd23e62.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/55a3f9294d6184176af813d4f993c9b81f96d997061609b6759e54356bd23e62.jpg)

![59824ad294770dd302bd942a554f74319f8692afd90f019e4bccd816697f7d9d.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/59824ad294770dd302bd942a554f74319f8692afd90f019e4bccd816697f7d9d.jpg)

![70bcc3e3b85b766845c838014f0490071b504a25655ec638066d006ff77a8b4d.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/70bcc3e3b85b766845c838014f0490071b504a25655ec638066d006ff77a8b4d.jpg)

![90efdc1cf1e5ba3b8761e045c44ac4b715dbbd5b6767213ce3e94db59efe91ed.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/90efdc1cf1e5ba3b8761e045c44ac4b715dbbd5b6767213ce3e94db59efe91ed.jpg)

![99021111a56febd8b7e285561487aa9ad2c0b18c10b5e38400e0887e04d5da36.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/99021111a56febd8b7e285561487aa9ad2c0b18c10b5e38400e0887e04d5da36.jpg)

![b57cf581ad26bcd083f8eab52d7040dbb399eb3041bd2cf230cf5aa1782b4358.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/b57cf581ad26bcd083f8eab52d7040dbb399eb3041bd2cf230cf5aa1782b4358.jpg)

![b9d97ed8c04a3fd20a22de618bb78ca717040a417835b5491785247e3f54ecfa.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/b9d97ed8c04a3fd20a22de618bb78ca717040a417835b5491785247e3f54ecfa.jpg)

![cf941b6314fb509b779c44bb583a387081ef4d73ca82ab6b93729f88914214e5.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/cf941b6314fb509b779c44bb583a387081ef4d73ca82ab6b93729f88914214e5.jpg)

![d8a69846c4e65bc9215cc5dbe7cab9ea7a819274b313ffd99dd38b18e6874969.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/images/d8a69846c4e65bc9215cc5dbe7cab9ea7a819274b313ffd99dd38b18e6874969.jpg)

### Tables

![035f9088ca679378f132bbf433afc762a6766446f759772e4c0d120e2f158645.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/035f9088ca679378f132bbf433afc762a6766446f759772e4c0d120e2f158645.jpg)

![08d4d4e3ad279aed88ee3f13ce3d365675e5a1b4a341266ac42cfd3cbdd0402e.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/08d4d4e3ad279aed88ee3f13ce3d365675e5a1b4a341266ac42cfd3cbdd0402e.jpg)

![1e70784ffbd0386764eea03ec49375c15af181ef39103e8ebe5f10022585e5e4.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/1e70784ffbd0386764eea03ec49375c15af181ef39103e8ebe5f10022585e5e4.jpg)

![402bf48de88a6b2254bb7045ba9c0b26886687b7d14fe925b7430d0b36379f93.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/402bf48de88a6b2254bb7045ba9c0b26886687b7d14fe925b7430d0b36379f93.jpg)

![4cb31cb80895af5ce758c6c55e512f89aab377d318f4e72c5d79035630afcac7.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/4cb31cb80895af5ce758c6c55e512f89aab377d318f4e72c5d79035630afcac7.jpg)

![5093380a1bfc80d8da587b7064e390d72de91524df599a27a5acfd6899f7ada5.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/5093380a1bfc80d8da587b7064e390d72de91524df599a27a5acfd6899f7ada5.jpg)

![7234d37547000e3cd7636d932ccf1464eaa59decc7039b23f4d7890579ed5a84.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/7234d37547000e3cd7636d932ccf1464eaa59decc7039b23f4d7890579ed5a84.jpg)

![c3bb585fb94a42d842d1db9bfad5742f36d9d7c82a0d53d42ed44ab4209e3d96.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/c3bb585fb94a42d842d1db9bfad5742f36d9d7c82a0d53d42ed44ab4209e3d96.jpg)

![d316739f18c04dc49d3bdd67bafc27b48350d3d196ca5f2b0105dae7af332ecf.jpg](../icml_results/4952_Beyond%20Sample-Level%20Forgetting_%20Improving%20Reliability%20in%20Multimodal%20Unlearning/tables/d316739f18c04dc49d3bdd67bafc27b48350d3d196ca5f2b0105dae7af332ecf.jpg)

<a id="paper-4905"></a>

## Adaptive Querying with AI Persona Priors


### Images

![4c038f6f75e4877183db942122369040266d146879c22a53dd5ba2da976e2e05.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/images/4c038f6f75e4877183db942122369040266d146879c22a53dd5ba2da976e2e05.jpg)

![625dc4f64f8d68a9764d4c4ef3ac72af3dd83acf844acd553b5e694dc52400f4.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/images/625dc4f64f8d68a9764d4c4ef3ac72af3dd83acf844acd553b5e694dc52400f4.jpg)

![a532e717a85e21ac963e239b1f8bb2d67ca7eb07d5d30c3bfe2b82de68377877.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/images/a532e717a85e21ac963e239b1f8bb2d67ca7eb07d5d30c3bfe2b82de68377877.jpg)

![c5fedf102ace803257a08ef5d6b7c05ae97f267d0f10bfba0547ba8e561eb243.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/images/c5fedf102ace803257a08ef5d6b7c05ae97f267d0f10bfba0547ba8e561eb243.jpg)

### Tables

![01af8cb9e909dfa94e119d9aaf57568e6518ef139f0c379a167e5ef4247a45a9.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/01af8cb9e909dfa94e119d9aaf57568e6518ef139f0c379a167e5ef4247a45a9.jpg)

![1265f1cea521bfd049504af0790ff339c12a66fcf37e59d58fb74df06088f67c.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/1265f1cea521bfd049504af0790ff339c12a66fcf37e59d58fb74df06088f67c.jpg)

![14a20ca9a13984938c0bb074cdf8f23ed5f2a58003ad6199d78826ae57becf51.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/14a20ca9a13984938c0bb074cdf8f23ed5f2a58003ad6199d78826ae57becf51.jpg)

![1bd4cc1df9df52974757897bccdcc38b3c6810efb72d297380fc14875284ed6e.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/1bd4cc1df9df52974757897bccdcc38b3c6810efb72d297380fc14875284ed6e.jpg)

![41ff7598ef4955009f793aed76ada64a258b1d6e1256768841c4fb49a72ea981.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/41ff7598ef4955009f793aed76ada64a258b1d6e1256768841c4fb49a72ea981.jpg)

![4bb563a190e60923e287e31014d897668e6981aac89345aa3fa835be664166c7.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/4bb563a190e60923e287e31014d897668e6981aac89345aa3fa835be664166c7.jpg)

![57e1e6ea88cdf7e80f28f832ab79ae61039e83e22b97942ec0aafed59f69df0e.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/57e1e6ea88cdf7e80f28f832ab79ae61039e83e22b97942ec0aafed59f69df0e.jpg)

![7e8eb4739a31b52afe9e7d9d06b767210b437feefa3870686a400173370cabf9.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/7e8eb4739a31b52afe9e7d9d06b767210b437feefa3870686a400173370cabf9.jpg)

![8dcbee3fcc035ec7f6fd65ca99c5c067dc54ab28139534b2bc422bdf7ff66a2b.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/8dcbee3fcc035ec7f6fd65ca99c5c067dc54ab28139534b2bc422bdf7ff66a2b.jpg)

![aa8e237d9c717ff5f7c49c6611ff7b3ad19cf394711baee3615294f3409d2774.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/aa8e237d9c717ff5f7c49c6611ff7b3ad19cf394711baee3615294f3409d2774.jpg)

![bdd44dd97f1c2ef911f2a3f4c72db12eb5e3c1a590f5b83d422578d8b60a0951.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/bdd44dd97f1c2ef911f2a3f4c72db12eb5e3c1a590f5b83d422578d8b60a0951.jpg)

![f05993e07a84d9f41cd06c1ab3aa10290045a15975642b6679354895d739a36b.jpg](../icml_results/4953_Adaptive%20Querying%20with%20AI%20Persona%20Priors/tables/f05993e07a84d9f41cd06c1ab3aa10290045a15975642b6679354895d739a36b.jpg)

<a id="paper-4906"></a>

## Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning


### Images

![1cdc0fa1e922f3db5103a6303a98d1297d5faeb18bbcede3dfe1d9b734856f2b.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/images/1cdc0fa1e922f3db5103a6303a98d1297d5faeb18bbcede3dfe1d9b734856f2b.jpg)

### Tables

![0ace567df37366d39991cf7be5fc335f0d3f34aca29c90a90acd6422ebe15544.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/0ace567df37366d39991cf7be5fc335f0d3f34aca29c90a90acd6422ebe15544.jpg)

![164dc76ccc0bc4b8a629b2a48748a43a3fa5b390bdc1985ac91b1d26b4363eb3.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/164dc76ccc0bc4b8a629b2a48748a43a3fa5b390bdc1985ac91b1d26b4363eb3.jpg)

![270144acf829adc08816723fef984774efe53f5241544444f7e8f1e69208b9f5.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/270144acf829adc08816723fef984774efe53f5241544444f7e8f1e69208b9f5.jpg)

![4aeaa8009f6c391b229ecf84d90e70a40143fafe55e23698be0ef450ef949a71.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/4aeaa8009f6c391b229ecf84d90e70a40143fafe55e23698be0ef450ef949a71.jpg)

![6faa2809ffcd12f6d7729862e16ad437a421a4d916f55e2db7a2638a69451b2e.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/6faa2809ffcd12f6d7729862e16ad437a421a4d916f55e2db7a2638a69451b2e.jpg)

![77314a499cbd72c9fa97fd9f193fa2a0306e6b6197181895866015853ca43f20.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/77314a499cbd72c9fa97fd9f193fa2a0306e6b6197181895866015853ca43f20.jpg)

![cfcbee641e58597e5aa196cdeaa9a002a829c3f20bce99b44af4d6fb1fbc9ff5.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/cfcbee641e58597e5aa196cdeaa9a002a829c3f20bce99b44af4d6fb1fbc9ff5.jpg)

![ef454eefa445c5c2cecb1a5421a79b03b1fb8b28480a1087d050ee8f64d5efdb.jpg](../icml_results/4954_Tracking%20Drift_%20Variation-Aware%20Entropy%20Scheduling%20for%20Non-Stationary%20Reinforcement%20Learning/tables/ef454eefa445c5c2cecb1a5421a79b03b1fb8b28480a1087d050ee8f64d5efdb.jpg)

<a id="paper-4907"></a>

## A Progressive Evidence Localization Framework Based on Wasserstein Gradient Flows for Document Visual Question Answering


### Images

![29a44d5a46acb4debf592d10d608d1f7d9331c3afd664badb8c5a3b01cd85008.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/29a44d5a46acb4debf592d10d608d1f7d9331c3afd664badb8c5a3b01cd85008.jpg)

![3e35ca4b6a39e1e6b5b39aa938cc656f0b7335331473619f03153dc61521f619.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/3e35ca4b6a39e1e6b5b39aa938cc656f0b7335331473619f03153dc61521f619.jpg)

![9289691935f59a8002d1bb5422fb845f3f9014358a6f5789adb1e1b5e3393779.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/9289691935f59a8002d1bb5422fb845f3f9014358a6f5789adb1e1b5e3393779.jpg)

![9cce0c4059d7f20600ec59e948a37c60245bed91159743a5cd67f9da5fb1f118.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/9cce0c4059d7f20600ec59e948a37c60245bed91159743a5cd67f9da5fb1f118.jpg)

![a1a7e7cab45b58838226344a4f82b20a5b62e8c39707349e3df3625de1b15aff.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/a1a7e7cab45b58838226344a4f82b20a5b62e8c39707349e3df3625de1b15aff.jpg)

![c52db57d7496dbcc7e56284cdf74b2edfd88ddba488b67c0753c32f0a1f3d186.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/c52db57d7496dbcc7e56284cdf74b2edfd88ddba488b67c0753c32f0a1f3d186.jpg)

![f58bef0fd785aec91c97e9517131ff9c619e0e34df45e0b537522c630478641d.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/f58bef0fd785aec91c97e9517131ff9c619e0e34df45e0b537522c630478641d.jpg)

![fad4df6f5c489cf0c7a19a74a6bb16f5e3afc35c814b30ab644b7561a56dd138.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/images/fad4df6f5c489cf0c7a19a74a6bb16f5e3afc35c814b30ab644b7561a56dd138.jpg)

### Tables

![612e94e07e06568966df814aa4788f129af4803049e9554ce09b9b294e672d35.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/612e94e07e06568966df814aa4788f129af4803049e9554ce09b9b294e672d35.jpg)

![6a3f39a48b524edcf682296ba36c79b441f83876c29f2b6210652f0c91db8410.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/6a3f39a48b524edcf682296ba36c79b441f83876c29f2b6210652f0c91db8410.jpg)

![835f39c16acd8197afa1288071b4bc54ad58215bcdd4a7c92a301cd0d8fcc6e2.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/835f39c16acd8197afa1288071b4bc54ad58215bcdd4a7c92a301cd0d8fcc6e2.jpg)

![85b407babfa8f1a594b6d6072c99aad54b1f5ee8db72e122ae37af3086ef9d2b.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/85b407babfa8f1a594b6d6072c99aad54b1f5ee8db72e122ae37af3086ef9d2b.jpg)

![ae0448a77f9943e4de6d356477a98ba89172dee600c9248665b788f86a29fc15.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/ae0448a77f9943e4de6d356477a98ba89172dee600c9248665b788f86a29fc15.jpg)

![bc67da7b453656d2f300c879b4345339457722cd56a5b97a0485dc1442e131d6.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/bc67da7b453656d2f300c879b4345339457722cd56a5b97a0485dc1442e131d6.jpg)

![ed83473713f6773df929f631dcd237c88a670bbf530087acbdf589f14c8356c4.jpg](../icml_results/4955_A%20Progressive%20Evidence%20Localization%20Framework%20Based%20on%20Wasserstein%20Gradient%20Flows%20for%20Document%20Visua/tables/ed83473713f6773df929f631dcd237c88a670bbf530087acbdf589f14c8356c4.jpg)

<a id="paper-4908"></a>

## Keep It in Mind: User Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams


### Images

![03a98b7a211374cdac2c83ba493b5bd9d241408647abb776fa172947eccbf538.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/03a98b7a211374cdac2c83ba493b5bd9d241408647abb776fa172947eccbf538.jpg)

![18a61a7e74fa31d01cb0c9c522ed336d3b970c072fe153433955c7c9cc6594d6.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/18a61a7e74fa31d01cb0c9c522ed336d3b970c072fe153433955c7c9cc6594d6.jpg)

![2424edf0f9eb6b8ec8c6fc287ef726ceafde00acd04d4a70e28aadd9fc0d8170.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/2424edf0f9eb6b8ec8c6fc287ef726ceafde00acd04d4a70e28aadd9fc0d8170.jpg)

![363bde1d9465dc20c93b2f01cc9ace34421e50da20f6f04b7af005d49a0a43c1.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/363bde1d9465dc20c93b2f01cc9ace34421e50da20f6f04b7af005d49a0a43c1.jpg)

![3c4c4a70ef4ec2090383060b413c566fe0dfaf7c259f5143c623d5bac4303a4b.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/3c4c4a70ef4ec2090383060b413c566fe0dfaf7c259f5143c623d5bac4303a4b.jpg)

![4094df55e59016e176d4235586ffec25a7997b0333801095565c22c47da7a687.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/4094df55e59016e176d4235586ffec25a7997b0333801095565c22c47da7a687.jpg)

![508a3a65a72c1f24d5d95a6a9935237351fc35fc966b4c73901e113014f30f51.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/508a3a65a72c1f24d5d95a6a9935237351fc35fc966b4c73901e113014f30f51.jpg)

![5553ecf1aeaef0bc8ba644d1e2ed4addfae5d6f9af3bd9f6caf899a295301865.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/5553ecf1aeaef0bc8ba644d1e2ed4addfae5d6f9af3bd9f6caf899a295301865.jpg)

![5d15a540b556c343ca2781e2f9ca9b56b8f436882f05fd656e2d0426aa813aa0.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/5d15a540b556c343ca2781e2f9ca9b56b8f436882f05fd656e2d0426aa813aa0.jpg)

![608a1518f72e6cf2122fa403440c21e0e11dc5612cb660a26773a7703e08aabd.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/608a1518f72e6cf2122fa403440c21e0e11dc5612cb660a26773a7703e08aabd.jpg)

![a6d43e4da6b64abc1035efb5b87e077e917d3ae0b18f11e14da08a003ab04700.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/a6d43e4da6b64abc1035efb5b87e077e917d3ae0b18f11e14da08a003ab04700.jpg)

![a81320632b0d85493687d96894f61611d81b05ed072a3b2b03571518a95ef9fc.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/a81320632b0d85493687d96894f61611d81b05ed072a3b2b03571518a95ef9fc.jpg)

![b066f92471c32c27eb20bf0b48df2edbeeaacf03a79c95c3d1ed6fbd9453bf3b.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/b066f92471c32c27eb20bf0b48df2edbeeaacf03a79c95c3d1ed6fbd9453bf3b.jpg)

![b268e9bacf169fed42e56b282c3a6fe97e6dfe29b0de2b749e456a1533d2293f.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/b268e9bacf169fed42e56b282c3a6fe97e6dfe29b0de2b749e456a1533d2293f.jpg)

![b7ffb77578c33842e1c3e7127a9f4646c6c3ce629705c3864ec7cecd8a75ff55.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/b7ffb77578c33842e1c3e7127a9f4646c6c3ce629705c3864ec7cecd8a75ff55.jpg)

![c449b4bb8b8aef8e36b3bb0cf24bf93d28a855ecada586db3ecfd0d305e68647.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/c449b4bb8b8aef8e36b3bb0cf24bf93d28a855ecada586db3ecfd0d305e68647.jpg)

![d3683b26923e31b7505734c5a578e8c9b3c948a4f4625ff91092317f8a0ee55d.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/d3683b26923e31b7505734c5a578e8c9b3c948a4f4625ff91092317f8a0ee55d.jpg)

![d6edd0fd340a90c7404d8dddbee8d9b8c82099a523b332c8eb5a9333c57401b8.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/d6edd0fd340a90c7404d8dddbee8d9b8c82099a523b332c8eb5a9333c57401b8.jpg)

![e2fc6287c55994757de97ed5c2c2042cd05629f92ea61f7b744ffa984c808795.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/e2fc6287c55994757de97ed5c2c2042cd05629f92ea61f7b744ffa984c808795.jpg)

![f2de7eaab27da964633abba7d9fb7b2b1939f8258281f806b62879e9eb97f97a.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/images/f2de7eaab27da964633abba7d9fb7b2b1939f8258281f806b62879e9eb97f97a.jpg)

### Tables

![124110993707c79cf6f9e15f5eeb54bc438cbaf0270ed6d6beeeb657c80b2bb9.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/124110993707c79cf6f9e15f5eeb54bc438cbaf0270ed6d6beeeb657c80b2bb9.jpg)

![494d30e31030ca8ae2bce6f3a5e2b56d8837f1053ad47272b8d2d7a4b6c5b0f0.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/494d30e31030ca8ae2bce6f3a5e2b56d8837f1053ad47272b8d2d7a4b6c5b0f0.jpg)

![4ac9c3ddcbbf062d4b79a10df1079d374c4394ca0d21fc0cd4222624ace3261b.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/4ac9c3ddcbbf062d4b79a10df1079d374c4394ca0d21fc0cd4222624ace3261b.jpg)

![853e06cb2d8ef55f11e4d48da20dee4e50db530f422b3a5d70312d8129929c5e.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/853e06cb2d8ef55f11e4d48da20dee4e50db530f422b3a5d70312d8129929c5e.jpg)

![912385d89d14a5976d49bde746edec5413c0e5d1c529f8e6c559aa007dd6798b.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/912385d89d14a5976d49bde746edec5413c0e5d1c529f8e6c559aa007dd6798b.jpg)

![b8bf82d5e6ccb1464aa2638e81f673b6704a868f1819d7431bec6d14a1262f91.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/b8bf82d5e6ccb1464aa2638e81f673b6704a868f1819d7431bec6d14a1262f91.jpg)

![cbabe020c3c6bedbf6da72cfba39a1c6fc9a708f23b607744be463d8c2ed13c5.jpg](../icml_results/4956_Keep%20It%20in%20Mind_%20User%20Centric%20Continual%20Spatial%20Intelligence%20Reasoning%20in%20Egocentric%20Video%20Streams/tables/cbabe020c3c6bedbf6da72cfba39a1c6fc9a708f23b607744be463d8c2ed13c5.jpg)

<a id="paper-4909"></a>

## RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System


### Images

![310824bb2404ed1757ff38b784f7b303b5e7da99f499e89d5065002baa1632d7.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/310824bb2404ed1757ff38b784f7b303b5e7da99f499e89d5065002baa1632d7.jpg)

![31c4da0569d61fa76a902cb8dcec3e7d06131533b83543e902c428feff0e36f4.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/31c4da0569d61fa76a902cb8dcec3e7d06131533b83543e902c428feff0e36f4.jpg)

![39e08e2abd08ced129d29b646478da6172ec949e0e513833e3bb5fad02eb2b7b.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/39e08e2abd08ced129d29b646478da6172ec949e0e513833e3bb5fad02eb2b7b.jpg)

![3b7335b85596d15b59f3b45a83dddc052bab36753a9f71d7b542b30a8275d3e2.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/3b7335b85596d15b59f3b45a83dddc052bab36753a9f71d7b542b30a8275d3e2.jpg)

![48f590afa413d402d675a4e0bf1cac96f9071fb9a150ed4bbe46e77f900ad755.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/48f590afa413d402d675a4e0bf1cac96f9071fb9a150ed4bbe46e77f900ad755.jpg)

![8f817067f24c1957e27a8c9c4344849dacdea143d7cac5fcf3985d3cd18dd769.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/8f817067f24c1957e27a8c9c4344849dacdea143d7cac5fcf3985d3cd18dd769.jpg)

![b9893075b10698beae918c58bd7dea9d3fd698f66c3c8b43613439b0864d7a7e.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/b9893075b10698beae918c58bd7dea9d3fd698f66c3c8b43613439b0864d7a7e.jpg)

![db71d582fe7a019c1b7a00372cd852e3c49ad36e83235c517a870493a8cb44b7.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/db71d582fe7a019c1b7a00372cd852e3c49ad36e83235c517a870493a8cb44b7.jpg)

![fcac0c9ac646521508f04fb08fd7a07cf82b8fd8c0cf054dee9fe8f7d0a13661.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/images/fcac0c9ac646521508f04fb08fd7a07cf82b8fd8c0cf054dee9fe8f7d0a13661.jpg)

### Tables

![7dbe400d39a0a897986549ae5a830437cb0e93f06c2a702607d5ac04e5a0dec5.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/tables/7dbe400d39a0a897986549ae5a830437cb0e93f06c2a702607d5ac04e5a0dec5.jpg)

![a0144738d3ecb0db70d62dd8265569fa37abe5343696eb83fbe0f9725af780c9.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/tables/a0144738d3ecb0db70d62dd8265569fa37abe5343696eb83fbe0f9725af780c9.jpg)

![a4ea50b613b112ea460e187363316614e7e917dfb182597deff8e8b23cb8c156.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/tables/a4ea50b613b112ea460e187363316614e7e917dfb182597deff8e8b23cb8c156.jpg)

![a6e0aa4b8b2100007aa85ffb530567e8a3a9562d01ba92de46f703f384950ca6.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/tables/a6e0aa4b8b2100007aa85ffb530567e8a3a9562d01ba92de46f703f384950ca6.jpg)

![d68e166ea9e4f95cc3c2fe42448f95ca64ebb3a90cf8682ff3f0658b84169911.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/tables/d68e166ea9e4f95cc3c2fe42448f95ca64ebb3a90cf8682ff3f0658b84169911.jpg)

![ecf403011a194e71c83a4b1279f0eae37f6137028b69db500c04d50f985ba9a2.jpg](../icml_results/4957_RLAnything_%20Forge%20Environment%2C%20Policy%2C%20and%20Reward%20Model%20in%20Completely%20Dynamic%20RL%20System/tables/ecf403011a194e71c83a4b1279f0eae37f6137028b69db500c04d50f985ba9a2.jpg)

<a id="paper-4910"></a>

## On the Entropy Dynamics in Reinforcement Fine-Tuning of Large Language Models


### Images

![2f16447457dfa84736fc65c0e1b917100b43483b13f50a92640fa1dcc8836588.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/2f16447457dfa84736fc65c0e1b917100b43483b13f50a92640fa1dcc8836588.jpg)

![72f0d3a2da61ba153cd72919aa2ab62a7fc6269abc1b70499d4d8f1d4c233460.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/72f0d3a2da61ba153cd72919aa2ab62a7fc6269abc1b70499d4d8f1d4c233460.jpg)

![a7271748813a2963544a7736bcb7c79431ee554ab99af95e05d4bd1a5e5c32e0.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/a7271748813a2963544a7736bcb7c79431ee554ab99af95e05d4bd1a5e5c32e0.jpg)

![bd79974c6b29c95ff9b233290b427ab5802242c4ac98b44889440f626afccae1.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/bd79974c6b29c95ff9b233290b427ab5802242c4ac98b44889440f626afccae1.jpg)

![d3ba05f255fe5dea8717bcc352bdf19828fce427b50c86056f8a16624dae7fcf.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/d3ba05f255fe5dea8717bcc352bdf19828fce427b50c86056f8a16624dae7fcf.jpg)

![f71ff3882cd0cd8bf48d70f16e5ee2b419db5bd6df3b55958939b5f4ee8c518c.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/f71ff3882cd0cd8bf48d70f16e5ee2b419db5bd6df3b55958939b5f4ee8c518c.jpg)

![fa210e8e0a8ce5a9164e19ed063fbb6907cc00dbac947407ff495331d069987b.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/images/fa210e8e0a8ce5a9164e19ed063fbb6907cc00dbac947407ff495331d069987b.jpg)

### Tables

![00612580f100193a072689ab0365d8fe9a6624dc343a64573bbd37bc9ad8c1c3.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/tables/00612580f100193a072689ab0365d8fe9a6624dc343a64573bbd37bc9ad8c1c3.jpg)

![0a243e7f7057bdb5aad1a450b3ed899f5132cea9c4d3beb2aecab460595dd81b.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/tables/0a243e7f7057bdb5aad1a450b3ed899f5132cea9c4d3beb2aecab460595dd81b.jpg)

![ae2679f8bfe7a307c54c138d199bb12443a26b097fa8bf18fd46ae8a82d66331.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/tables/ae2679f8bfe7a307c54c138d199bb12443a26b097fa8bf18fd46ae8a82d66331.jpg)

![b940c80e4d54972ff80d448e7a397962394da94869e46b12e376e6eb7b355fe2.jpg](../icml_results/4958_On%20the%20Entropy%20Dynamics%20in%20Reinforcement%20Fine-Tuning%20of%20Large%20Language%20Models/tables/b940c80e4d54972ff80d448e7a397962394da94869e46b12e376e6eb7b355fe2.jpg)

<a id="paper-4911"></a>

## Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning


### Images

![0236f0341bad1944b63fe9895d1c2359ff30dde6c7fb8deb8398cecba3110dd2.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/0236f0341bad1944b63fe9895d1c2359ff30dde6c7fb8deb8398cecba3110dd2.jpg)

![09363c614d3a43d9cf548c0f691b8dbab87bf82691fec2366d0fb5084b6d947d.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/09363c614d3a43d9cf548c0f691b8dbab87bf82691fec2366d0fb5084b6d947d.jpg)

![28653e71426842d0e70bf0f67210b2315265db9e7f327345e60591015e943a6f.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/28653e71426842d0e70bf0f67210b2315265db9e7f327345e60591015e943a6f.jpg)

![2f6bed77c36b76d21f5bd69b1aef063988a42c3a235a4ab532eb0879e190edc8.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/2f6bed77c36b76d21f5bd69b1aef063988a42c3a235a4ab532eb0879e190edc8.jpg)

![2fccfc59c07ed0ca9787bfd04bc08e48c231a958610b885bca3ecb0744a351fa.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/2fccfc59c07ed0ca9787bfd04bc08e48c231a958610b885bca3ecb0744a351fa.jpg)

![342d3b30425c2edb4506686beec6a05f60674009522fba2f355810f5c452c9cf.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/342d3b30425c2edb4506686beec6a05f60674009522fba2f355810f5c452c9cf.jpg)

![3e78c3fa7ead9a1f7aaa03c51b88e028a5eecbd68ac3778261bf3dc45edcabbd.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/3e78c3fa7ead9a1f7aaa03c51b88e028a5eecbd68ac3778261bf3dc45edcabbd.jpg)

![62375e839830adfb4f0d485818ec4c0fe66a9dc8ee8ef10d797bff47e2fe6a54.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/62375e839830adfb4f0d485818ec4c0fe66a9dc8ee8ef10d797bff47e2fe6a54.jpg)

![62ff6ab8a66778b955a62c8d770c17a91958fd33d1b3d97659ae17c3efb7921e.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/62ff6ab8a66778b955a62c8d770c17a91958fd33d1b3d97659ae17c3efb7921e.jpg)

![6599b3e22d9625d147768c831bf658d097261d0b306d8a7f62df164a5a8e4cc1.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/6599b3e22d9625d147768c831bf658d097261d0b306d8a7f62df164a5a8e4cc1.jpg)

![66eb417b2544c7c71e8d518e2fcbcbb02c4874724fd381ea94f40b98540a3821.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/66eb417b2544c7c71e8d518e2fcbcbb02c4874724fd381ea94f40b98540a3821.jpg)

![6aad9e4b07dcd5753018055c4906f61beb1c898bdfe28ade62722e61b029cc38.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/6aad9e4b07dcd5753018055c4906f61beb1c898bdfe28ade62722e61b029cc38.jpg)

![7f32d2cd7d6ed8fef045e084cabab0a3dda9202ae7344b162afd05ca71398e4d.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/7f32d2cd7d6ed8fef045e084cabab0a3dda9202ae7344b162afd05ca71398e4d.jpg)

![83bf101e30d57ba2c8068ea828d6fae5e7fd10fb39c046a121e532f239da5c54.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/83bf101e30d57ba2c8068ea828d6fae5e7fd10fb39c046a121e532f239da5c54.jpg)

![9ca0846136862eff8fb366769eb7f1d8e479c26657968ec17141816c0f6f5e93.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/9ca0846136862eff8fb366769eb7f1d8e479c26657968ec17141816c0f6f5e93.jpg)

![a1c7161fc0d8efbcaddd85d4bfdf08c760b2f931f5c896c0e6261e036adf447d.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/a1c7161fc0d8efbcaddd85d4bfdf08c760b2f931f5c896c0e6261e036adf447d.jpg)

![a6a8631454d6469afa30856b3f4f934f03a63e714669941c5bfea99b31a8d64c.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/a6a8631454d6469afa30856b3f4f934f03a63e714669941c5bfea99b31a8d64c.jpg)

![b45f546576251791cd531931dfbdf4d438e82aeaf35a7ae881107260e938920e.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/b45f546576251791cd531931dfbdf4d438e82aeaf35a7ae881107260e938920e.jpg)

![b4e3d47f0504cc8a9d4ad35ae60988d3dace796adb5ffd7b4eaf33bdf07b9dde.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/b4e3d47f0504cc8a9d4ad35ae60988d3dace796adb5ffd7b4eaf33bdf07b9dde.jpg)

![bd074018fff43323f030d7c4e1c64b7faf6d47f209a07106e007f4d626a10baa.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/bd074018fff43323f030d7c4e1c64b7faf6d47f209a07106e007f4d626a10baa.jpg)

![cc77b9058feaddde62dfe1bfb308321a54102d8060a7c00471635b03cb77c3bb.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/cc77b9058feaddde62dfe1bfb308321a54102d8060a7c00471635b03cb77c3bb.jpg)

![cc8763586fa6a1094ba8515fed5b7d781b7fc98a2ebb93c6a8d1eec629e9b0b2.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/cc8763586fa6a1094ba8515fed5b7d781b7fc98a2ebb93c6a8d1eec629e9b0b2.jpg)

![cd27ab7e654e7ec469f8bccc5b48dfd8aaadd871bbecd14a8952c9a35d345dc8.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/cd27ab7e654e7ec469f8bccc5b48dfd8aaadd871bbecd14a8952c9a35d345dc8.jpg)

![e32c1cd7b5b30ec0a97fd1e4392c9b6336c6d6c389140e18239b03e665cced5c.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/e32c1cd7b5b30ec0a97fd1e4392c9b6336c6d6c389140e18239b03e665cced5c.jpg)

![ecbf5801604d38be0d001e28b9da33d4b57807cc2fccd7d856573881e94f7344.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/ecbf5801604d38be0d001e28b9da33d4b57807cc2fccd7d856573881e94f7344.jpg)

![eee32cf2e6a75547364b348eeb2dfb161550dca25da4031f2296b5232107e2ad.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/eee32cf2e6a75547364b348eeb2dfb161550dca25da4031f2296b5232107e2ad.jpg)

![ef7fa3a3a69ff79e0ffa6853b9f752e6db60c804a240fc1d925d835a99093c76.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/images/ef7fa3a3a69ff79e0ffa6853b9f752e6db60c804a240fc1d925d835a99093c76.jpg)

### Tables

![02c4d277b1a9c0dcf8b9176b71cf07a72afb4665667ae6141b749f8f5e02af14.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/02c4d277b1a9c0dcf8b9176b71cf07a72afb4665667ae6141b749f8f5e02af14.jpg)

![208c82f29a8b87b064f1f057a6326ab7c1f7f44c99113a28b0950d7c93bf47c3.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/208c82f29a8b87b064f1f057a6326ab7c1f7f44c99113a28b0950d7c93bf47c3.jpg)

![2b548e0c41003720f0eb057c2d23a648fc1f29e3f94fd8f590ba52d7e7bc029f.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/2b548e0c41003720f0eb057c2d23a648fc1f29e3f94fd8f590ba52d7e7bc029f.jpg)

![492dea9d8315241df0c1cf438816eda7c51986ca2174bbe436e28c4724af6b8e.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/492dea9d8315241df0c1cf438816eda7c51986ca2174bbe436e28c4724af6b8e.jpg)

![5c3ae73bd15ae5f7f5c302687b427a138867fa89fa8fb62f7d8b07cfe91ed1b1.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/5c3ae73bd15ae5f7f5c302687b427a138867fa89fa8fb62f7d8b07cfe91ed1b1.jpg)

![680c23811b2d2cc05b0328c9d0bc7c1a6132872f32ac56b7bcaf6bd5990c75b3.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/680c23811b2d2cc05b0328c9d0bc7c1a6132872f32ac56b7bcaf6bd5990c75b3.jpg)

![68c8ae86ade791f718db0b9b2f95dbb8328adf79106681a44c4d117ea4b667ce.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/68c8ae86ade791f718db0b9b2f95dbb8328adf79106681a44c4d117ea4b667ce.jpg)

![6f1747e0c5104d54a12a3cd86aaf4b8be8186efd03b5814edad7c232feb54edc.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/6f1747e0c5104d54a12a3cd86aaf4b8be8186efd03b5814edad7c232feb54edc.jpg)

![794bd53b9e2bdc0b3bb5d8ddced458681b6f41eede99c319fd2db0d2660bc98b.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/794bd53b9e2bdc0b3bb5d8ddced458681b6f41eede99c319fd2db0d2660bc98b.jpg)

![7f9e9787da07bd8e4df1e11eacd591fc869deef398c25b3c9f69d51302d34caf.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/7f9e9787da07bd8e4df1e11eacd591fc869deef398c25b3c9f69d51302d34caf.jpg)

![84626e95706f331a12cd527969c952520e63955d589a49f7eecae8845d942e1e.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/84626e95706f331a12cd527969c952520e63955d589a49f7eecae8845d942e1e.jpg)

![922ecebd39890172daf6ed2c99443d3b64c764f9ac6ed5698b5fae83a49621e7.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/922ecebd39890172daf6ed2c99443d3b64c764f9ac6ed5698b5fae83a49621e7.jpg)

![b8f303ff888ecec1f315c023da08905a6582f598110ca49047c6d7a5cabe5920.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/b8f303ff888ecec1f315c023da08905a6582f598110ca49047c6d7a5cabe5920.jpg)

![c0616ccf914c435259d6eddff13d014f85ced7f61f309b304743952277ac8f33.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/c0616ccf914c435259d6eddff13d014f85ced7f61f309b304743952277ac8f33.jpg)

![d54c5ffe62482f109dd74bf4a0fe0cc18612299882b56d9d1d7333dfe6913295.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/d54c5ffe62482f109dd74bf4a0fe0cc18612299882b56d9d1d7333dfe6913295.jpg)

![f8097d025e571d21732fa8eed9dfedad95923699952e206200ca5ffb219d57d9.jpg](../icml_results/4959_Agent%20World%20Model_%20Infinity%20Synthetic%20Environments%20for%20Agentic%20Reinforcement%20Learning/tables/f8097d025e571d21732fa8eed9dfedad95923699952e206200ca5ffb219d57d9.jpg)

<a id="paper-4912"></a>

## SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning


### Images

![0bc9eeb2d1041985923a337c2c93499e644892a7d9c6eb1c8bec82a7b4ca5c16.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/0bc9eeb2d1041985923a337c2c93499e644892a7d9c6eb1c8bec82a7b4ca5c16.jpg)

![179f7655dd0d31e3b1a4a8ea7d321f93596c910b892757980be8a51396647c4b.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/179f7655dd0d31e3b1a4a8ea7d321f93596c910b892757980be8a51396647c4b.jpg)

![1d5d7337eeaf080af7ff8ef8d77de067568add5f71eef93a2f55b789bcc0289b.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/1d5d7337eeaf080af7ff8ef8d77de067568add5f71eef93a2f55b789bcc0289b.jpg)

![332561a9ebda5c4ea7bb2b893652ab7fa7a0cc011e960c0c1955ce4dce81732b.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/332561a9ebda5c4ea7bb2b893652ab7fa7a0cc011e960c0c1955ce4dce81732b.jpg)

![48c2ca71b9a1af0e60b599aa4af1da473055776d60d276ef08f7de7fe6f38bf2.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/48c2ca71b9a1af0e60b599aa4af1da473055776d60d276ef08f7de7fe6f38bf2.jpg)

![681c10af3268e6a166700e1c36807f08ab5b2fe8d100ef29459ef97e6585d684.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/681c10af3268e6a166700e1c36807f08ab5b2fe8d100ef29459ef97e6585d684.jpg)

![7f31bb7b916ae20598cfd743c11e1c279a63d5b69ef835b57c34e6d40309b282.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/7f31bb7b916ae20598cfd743c11e1c279a63d5b69ef835b57c34e6d40309b282.jpg)

![8aebde6c7b327d68fcd941d3eda8fe4967290dfc3fea7212e26053fa4e9beec9.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/8aebde6c7b327d68fcd941d3eda8fe4967290dfc3fea7212e26053fa4e9beec9.jpg)

![a81ab80a3a201f237658138ae4022cd1978b5f6c6c3f45a3f77469543b6e3e4c.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/a81ab80a3a201f237658138ae4022cd1978b5f6c6c3f45a3f77469543b6e3e4c.jpg)

![a9951592f481cf0c5a5937dc21afda3485cfc4622e0ac39591e35168b17e8e5c.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/a9951592f481cf0c5a5937dc21afda3485cfc4622e0ac39591e35168b17e8e5c.jpg)

![aa78a8077ca3c8264b09bea30a682aa40ff71c53867044f0e50984bdff41bc3f.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/aa78a8077ca3c8264b09bea30a682aa40ff71c53867044f0e50984bdff41bc3f.jpg)

![d855105304ee07f6b37670fb39c1049f110e032a057b9ffe95d3745bc6791bbe.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/d855105304ee07f6b37670fb39c1049f110e032a057b9ffe95d3745bc6791bbe.jpg)

![e685af778192d5143762f76ada0b2e2eafe5e75307f13eac59ae036b5da9feb3.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/e685af778192d5143762f76ada0b2e2eafe5e75307f13eac59ae036b5da9feb3.jpg)

![f2a76b555236b6eb42655d4c06003ef46c64d9f709ffd15ca191fe81f91d3053.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/f2a76b555236b6eb42655d4c06003ef46c64d9f709ffd15ca191fe81f91d3053.jpg)

![fe523d1904fb9aedfbb2b211fe7843295c81d2856f430544c2e04fd23d9e4fd0.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/images/fe523d1904fb9aedfbb2b211fe7843295c81d2856f430544c2e04fd23d9e4fd0.jpg)

### Tables

![0209619f74c812372b391368b6389e3631aacfe88997e3fd3905549ccbcaf1cf.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/0209619f74c812372b391368b6389e3631aacfe88997e3fd3905549ccbcaf1cf.jpg)

![0c677d95928a3951406cbb87c65604c55f5fea26c6e4d47a38ca90f4483c6dfe.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/0c677d95928a3951406cbb87c65604c55f5fea26c6e4d47a38ca90f4483c6dfe.jpg)

![0ef5fe95659697bb5f895a292f9e6d283c601558b33c6e623143d26a14fdfaaa.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/0ef5fe95659697bb5f895a292f9e6d283c601558b33c6e623143d26a14fdfaaa.jpg)

![38d4dfef41aec16983f4a8d23e76c159eee4aa980c2761f2d21e1346a5c3e712.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/38d4dfef41aec16983f4a8d23e76c159eee4aa980c2761f2d21e1346a5c3e712.jpg)

![4791478b5560488bda66ba025f20fe0d4b457a000fce6f7c45bfe036d4c9a0ea.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/4791478b5560488bda66ba025f20fe0d4b457a000fce6f7c45bfe036d4c9a0ea.jpg)

![634e3857611c329c1fae7c240ab6be77dbc52ad5e47e8dbc2a6c4fd55c98a541.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/634e3857611c329c1fae7c240ab6be77dbc52ad5e47e8dbc2a6c4fd55c98a541.jpg)

![6bee9ec4c3c78c415a83a9d4d7730470f587270aa59d12a0ec5793894a8be6a0.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/6bee9ec4c3c78c415a83a9d4d7730470f587270aa59d12a0ec5793894a8be6a0.jpg)

![9d7fad2b6d6603b736adc17be1177179fe43613b85e6d763708e8d08bf0cc673.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/9d7fad2b6d6603b736adc17be1177179fe43613b85e6d763708e8d08bf0cc673.jpg)

![a518c6e3c0b0b3a547a76bd9c87d6ccd5a6e7754c7f738fefe5c3375145a8308.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/a518c6e3c0b0b3a547a76bd9c87d6ccd5a6e7754c7f738fefe5c3375145a8308.jpg)

![cbc8d72d998d209967aa212012992992f3356eaec8d36dbeb6197f88f93edaa9.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/cbc8d72d998d209967aa212012992992f3356eaec8d36dbeb6197f88f93edaa9.jpg)

![d319143e04aeb79513944d02b85c4e431697b457efb85c3b44c1ea2e8565b016.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/d319143e04aeb79513944d02b85c4e431697b457efb85c3b44c1ea2e8565b016.jpg)

![d685ad11c4fc9f240f8870cb1bee711325fd8290c921cf167b8ad5eac2bbf5f7.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/d685ad11c4fc9f240f8870cb1bee711325fd8290c921cf167b8ad5eac2bbf5f7.jpg)

![f3e7eebc6e7c00d1c5ee84bf3799edd9c7f73c6599e42266fd1b74d35d5c496b.jpg](../icml_results/4960_SpecPrune-VLA_%20Accelerating%20Vision-Language-Action%20Models%20via%20Action-Aware%20Self-Speculative%20Pruning/tables/f3e7eebc6e7c00d1c5ee84bf3799edd9c7f73c6599e42266fd1b74d35d5c496b.jpg)

<a id="paper-4913"></a>

## Variable Clustering via Distributionally Robust Nodewise Regression


### Images

![3fb5efb7f528911ae3672f97df690714b489c637b94adeec8e8ab6d693ac9486.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/3fb5efb7f528911ae3672f97df690714b489c637b94adeec8e8ab6d693ac9486.jpg)

![41dc0f761f88d5e41df30eb56dba1c1bb39bd28cc05426e42ae36f6062ed94f7.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/41dc0f761f88d5e41df30eb56dba1c1bb39bd28cc05426e42ae36f6062ed94f7.jpg)

![4807ac5c76b53f1b638e8082a1d34e950eba78dfaf087b6b540bc8da408b930b.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/4807ac5c76b53f1b638e8082a1d34e950eba78dfaf087b6b540bc8da408b930b.jpg)

![51db771e7b5ef619536ac6c9658c75055bf331df0334991c01f22af51fb2b17c.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/51db771e7b5ef619536ac6c9658c75055bf331df0334991c01f22af51fb2b17c.jpg)

![60f1f3a2d93681de8ce234a645abb16e7a03f5334e3c186873dd37a2a9661519.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/60f1f3a2d93681de8ce234a645abb16e7a03f5334e3c186873dd37a2a9661519.jpg)

![64c334c1c7abc91664d7144f559e7bd7a5f4947c690e52b8137fdd226d084c84.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/64c334c1c7abc91664d7144f559e7bd7a5f4947c690e52b8137fdd226d084c84.jpg)

![73ec629a83df5edfcf251fc69e03784f9971c788b16ea1cd716d776236216a02.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/73ec629a83df5edfcf251fc69e03784f9971c788b16ea1cd716d776236216a02.jpg)

![8ed740ad7748569c4ad9ce46f128c0c9b568bb18acb0a4669779a5df1c9336aa.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/8ed740ad7748569c4ad9ce46f128c0c9b568bb18acb0a4669779a5df1c9336aa.jpg)

![915c2f4f383c89535ff4558b67b3629c3c7b64ff232dfd59dc56f7bdd7a28aba.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/915c2f4f383c89535ff4558b67b3629c3c7b64ff232dfd59dc56f7bdd7a28aba.jpg)

![9635ef28694816ef108fb70ea350610f743a28ce6aedfb64a0ec4f337dbdc08a.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/9635ef28694816ef108fb70ea350610f743a28ce6aedfb64a0ec4f337dbdc08a.jpg)

![a6826656aa46ff69f9a3daed2d407388f6c830bf14498a5c6d27d2d77f9829c4.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/a6826656aa46ff69f9a3daed2d407388f6c830bf14498a5c6d27d2d77f9829c4.jpg)

![c7737332bc29438d1a4e071325698131294956b15163308574812afc56da2ebd.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/c7737332bc29438d1a4e071325698131294956b15163308574812afc56da2ebd.jpg)

![f25c6b69037e35f02a43474dd65ade178d4db55dab67e4a6450be12750de9f08.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/images/f25c6b69037e35f02a43474dd65ade178d4db55dab67e4a6450be12750de9f08.jpg)

### Tables

![1c06589d80a517bf7227bf040356956d003a95568ea935829df8d1b75e1c3df6.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/1c06589d80a517bf7227bf040356956d003a95568ea935829df8d1b75e1c3df6.jpg)

![1ed10dac1a7bed25761102a0c4522b8400d3c6db1be9f0e4c2934d98321ef7d5.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/1ed10dac1a7bed25761102a0c4522b8400d3c6db1be9f0e4c2934d98321ef7d5.jpg)

![2eed1dc9a360069fc76acb754024eb6568a744a6e5a47c500dd10bc57617c9aa.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/2eed1dc9a360069fc76acb754024eb6568a744a6e5a47c500dd10bc57617c9aa.jpg)

![4f1a772e4aa0e5673e7f05ef5c49632dafc8022da7cdc156a7f9aa92acab5dbe.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/4f1a772e4aa0e5673e7f05ef5c49632dafc8022da7cdc156a7f9aa92acab5dbe.jpg)

![5f60e6d1abc543c401c3d1589023b61a76c84edd46a741da15c7f0aa65e81dd0.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/5f60e6d1abc543c401c3d1589023b61a76c84edd46a741da15c7f0aa65e81dd0.jpg)

![6e1c531e46eaa8222856653d015e09c07a8f9af0f6ff57d7b41d896e342d18b8.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/6e1c531e46eaa8222856653d015e09c07a8f9af0f6ff57d7b41d896e342d18b8.jpg)

![7748130843fa7e28ea079d2d6fe7412fd845d02a8a47a9d6b9cb81020ce2fbde.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/7748130843fa7e28ea079d2d6fe7412fd845d02a8a47a9d6b9cb81020ce2fbde.jpg)

![94851550f122acee58c5c22479d5d2244d7c28b5a8c3d08336957c16dc210571.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/94851550f122acee58c5c22479d5d2244d7c28b5a8c3d08336957c16dc210571.jpg)

![970edfa053ec999e8c2dd4613f031f3417f58ce96b744173ad9db9a08498d9c7.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/970edfa053ec999e8c2dd4613f031f3417f58ce96b744173ad9db9a08498d9c7.jpg)

![cd0448388c16ba04bf156deb21240652b60c2c5f4ed9375e425a7c9d866a1385.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/cd0448388c16ba04bf156deb21240652b60c2c5f4ed9375e425a7c9d866a1385.jpg)

![cd1d6bfc481557590d084382c766d2513de829f12c8156c69738e9d4d35e6d0a.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/cd1d6bfc481557590d084382c766d2513de829f12c8156c69738e9d4d35e6d0a.jpg)

![e4d75b4bb26f770b14f61f71c05682c632f1081a49df0890317d702718affc2f.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/e4d75b4bb26f770b14f61f71c05682c632f1081a49df0890317d702718affc2f.jpg)

![f91a70e2caf2f00540304d3c3168b1e714110ce05260fb843174461c27a0e134.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/f91a70e2caf2f00540304d3c3168b1e714110ce05260fb843174461c27a0e134.jpg)

![fdc940a1d68bdecd542300772bddc59956ee747b057d8572c83565df6ea7d8d4.jpg](../icml_results/4961_Variable%20Clustering%20via%20Distributionally%20Robust%20Nodewise%20Regression/tables/fdc940a1d68bdecd542300772bddc59956ee747b057d8572c83565df6ea7d8d4.jpg)

<a id="paper-4914"></a>

## DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training


### Images

![02fb15fe799822c4abacfdf0c8ffedc058976831f19dc1db0193af42f8d3a501.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/02fb15fe799822c4abacfdf0c8ffedc058976831f19dc1db0193af42f8d3a501.jpg)

![11730522c66203000c2471811fcf362282c74bee6c6839b3063621a648edc878.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/11730522c66203000c2471811fcf362282c74bee6c6839b3063621a648edc878.jpg)

![12b925a339f2c49643b3a89c5611ee56576d4bbf1aa91e1fccb997841a96ecc2.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/12b925a339f2c49643b3a89c5611ee56576d4bbf1aa91e1fccb997841a96ecc2.jpg)

![19418893eab64cb15b468276478f8cb36b1e26343ce7d5c40190bc1e3e5331b9.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/19418893eab64cb15b468276478f8cb36b1e26343ce7d5c40190bc1e3e5331b9.jpg)

![55f311c69169ebcbf49ff91d343364a203c3d2e5faa8bea2b3e92b89a267f372.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/55f311c69169ebcbf49ff91d343364a203c3d2e5faa8bea2b3e92b89a267f372.jpg)

![767f1f15600d9751ce0a3aa6857023c455588e9125152ffb6ef2468cec3e55ce.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/767f1f15600d9751ce0a3aa6857023c455588e9125152ffb6ef2468cec3e55ce.jpg)

![8aa94e170eb3760c4db2625a5c2c517e0e39c1708755c460f627d52c28f7494a.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/8aa94e170eb3760c4db2625a5c2c517e0e39c1708755c460f627d52c28f7494a.jpg)

![90ca723a3233ec3c76ac35650fcc972c792da185a0fa794624407835dc6d5e86.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/90ca723a3233ec3c76ac35650fcc972c792da185a0fa794624407835dc6d5e86.jpg)

![94ca07001fa172d648c5746d06f0e503cb52ea323401361b8570e3d8f41e5a70.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/94ca07001fa172d648c5746d06f0e503cb52ea323401361b8570e3d8f41e5a70.jpg)

![bab36526c81e0437d051cfc6ef2289392b954e653929b87eefb188cdcf8b877d.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/bab36526c81e0437d051cfc6ef2289392b954e653929b87eefb188cdcf8b877d.jpg)

![cbe51f133650b453c780e93797d0f38c8f958c25c95d97ab3781657499d2ee25.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/cbe51f133650b453c780e93797d0f38c8f958c25c95d97ab3781657499d2ee25.jpg)

![ec04cf5e0cd1a793cf9e38bbf0e404e3d9a1c31f454b568bf77526a8da7c8dea.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/images/ec04cf5e0cd1a793cf9e38bbf0e404e3d9a1c31f454b568bf77526a8da7c8dea.jpg)

### Tables

![4e319e36e187008b6de5f72869c36e94510e810fbc6c6a0a9800f845acdee628.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/tables/4e319e36e187008b6de5f72869c36e94510e810fbc6c6a0a9800f845acdee628.jpg)

![90a45f9f3828c2a3f2ab8f779b7a49a6f6bdcd197836f359bdc47ded45cf64bc.jpg](../icml_results/4962_DistFlow_%20A%20Fully%20Distributed%20RL%20Framework%20for%20Scalable%20and%20Efficient%20LLM%20Post-Training/tables/90a45f9f3828c2a3f2ab8f779b7a49a6f6bdcd197836f359bdc47ded45cf64bc.jpg)

<a id="paper-4915"></a>

## Milestone-Guided Policy Learning for Long-Horizon Language Agents


### Images

![174e0c6b4518b31c41251bfabd6915f21be18eabc85ba8bb3ddd1e54986302df.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/174e0c6b4518b31c41251bfabd6915f21be18eabc85ba8bb3ddd1e54986302df.jpg)

![591d0f9a7d99de73bc815344b4346f403fec397040da7f46f049c22f65cccdab.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/591d0f9a7d99de73bc815344b4346f403fec397040da7f46f049c22f65cccdab.jpg)

![5bf9eea770d4ba7e4484a778c3eeffe0417d211ef32a4b3e19c3f06db37f69e8.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/5bf9eea770d4ba7e4484a778c3eeffe0417d211ef32a4b3e19c3f06db37f69e8.jpg)

![82cf26d9b07f6827fe5e91b0e7cfbb378f543e7124efd427c876dc40d81f8059.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/82cf26d9b07f6827fe5e91b0e7cfbb378f543e7124efd427c876dc40d81f8059.jpg)

![afaf5b134921959d911a4958ccb6c8523eaf500aaa6617ea411d684fb7175398.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/afaf5b134921959d911a4958ccb6c8523eaf500aaa6617ea411d684fb7175398.jpg)

![bc470deb4198e2e9da9ac360a918d17b3076e8e09945e85f77827646296494c9.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/bc470deb4198e2e9da9ac360a918d17b3076e8e09945e85f77827646296494c9.jpg)

![e01479dffa007cd909ff33ff181baa6d797fbd977bac3327009856fd14078e42.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/e01479dffa007cd909ff33ff181baa6d797fbd977bac3327009856fd14078e42.jpg)

![e9717dd5f7a99ba239aa4b40063c96a17be508e7cc861c4897ffab5c6d47028f.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/images/e9717dd5f7a99ba239aa4b40063c96a17be508e7cc861c4897ffab5c6d47028f.jpg)

### Tables

![7a356a0a3cef23cdbb6017c13b96a642509ed9ed66317178f945f379901490a4.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/tables/7a356a0a3cef23cdbb6017c13b96a642509ed9ed66317178f945f379901490a4.jpg)

![a372c1f813f2cef032e1e83c8094522dbe37539c6e95e77a2b03248803a2533c.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/tables/a372c1f813f2cef032e1e83c8094522dbe37539c6e95e77a2b03248803a2533c.jpg)

![accd713c313efa447257f04a329a70938ae4c4027a4949b5970ffc9740b322e4.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/tables/accd713c313efa447257f04a329a70938ae4c4027a4949b5970ffc9740b322e4.jpg)

![c0bd199401080da40dc811b9f6721741176cae96f55643143fad7b3bf8e6c047.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/tables/c0bd199401080da40dc811b9f6721741176cae96f55643143fad7b3bf8e6c047.jpg)

![f68ebb3bd983664253702dc9cb3b0548b7bbbc2481d82e5fd45bbeb9742ba81e.jpg](../icml_results/4963_Milestone-Guided%20Policy%20Learning%20for%20Long-Horizon%20Language%20Agents/tables/f68ebb3bd983664253702dc9cb3b0548b7bbbc2481d82e5fd45bbeb9742ba81e.jpg)

<a id="paper-4916"></a>

## DR-MMSearchAgent: Deepening Reasoning in Multimodal Search Agents


### Images

![04fe02d206fd1cd75e0a4147506fdd300df025ee94d4687dc7cb8ad98f9610fe.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/04fe02d206fd1cd75e0a4147506fdd300df025ee94d4687dc7cb8ad98f9610fe.jpg)

![178efb3e3d500a50392d0bd9a5fe256e508928da177b4eb5b652a8a771fee24d.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/178efb3e3d500a50392d0bd9a5fe256e508928da177b4eb5b652a8a771fee24d.jpg)

![1a456ce35a6aec2cb643f1e68fa59c39e80251b43e733f3c942fa1cfa9bf153e.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/1a456ce35a6aec2cb643f1e68fa59c39e80251b43e733f3c942fa1cfa9bf153e.jpg)

![234383e401f696db2f3e7d6b5b5afc029b1a75945c27f9b6f13dad4bda7210d7.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/234383e401f696db2f3e7d6b5b5afc029b1a75945c27f9b6f13dad4bda7210d7.jpg)

![5889c6398493c5c60a931c5e9c5067c34b9f0735dff1e5b2dbfbf6d83f14f678.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/5889c6398493c5c60a931c5e9c5067c34b9f0735dff1e5b2dbfbf6d83f14f678.jpg)

![849a29d016026c3c899c67d9c7672860ca14b2a4410ddfb6fc99ba1257922757.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/849a29d016026c3c899c67d9c7672860ca14b2a4410ddfb6fc99ba1257922757.jpg)

![94e75e560e5551fb592a4d4782676507cd801df05688b9095ad923cf9868b725.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/94e75e560e5551fb592a4d4782676507cd801df05688b9095ad923cf9868b725.jpg)

![b4362a5e8912fa916cabc252398670074938ab6c71beb0ca5ac860f2aef6b909.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/b4362a5e8912fa916cabc252398670074938ab6c71beb0ca5ac860f2aef6b909.jpg)

![cb7904196cdc18a31f6bda283ec0631d2b0e530f939e75cbca8ea9bac8ae1572.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/cb7904196cdc18a31f6bda283ec0631d2b0e530f939e75cbca8ea9bac8ae1572.jpg)

![d13fc8b2947c7460b9e65cc025e590e4978aa0e003cab77cbb67a95bd79f66ae.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/d13fc8b2947c7460b9e65cc025e590e4978aa0e003cab77cbb67a95bd79f66ae.jpg)

![e8d82a482632b0edce72bec48c89d381e5e5ef0b64b8dab6f21903620ed9b96d.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/e8d82a482632b0edce72bec48c89d381e5e5ef0b64b8dab6f21903620ed9b96d.jpg)

![ff8e9d789f19f5aa9665ca03551e91cc79f0781ad5a63673666bd022a50259ec.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/images/ff8e9d789f19f5aa9665ca03551e91cc79f0781ad5a63673666bd022a50259ec.jpg)

### Tables

![0daa65376e3a5960db597a710374fac3cc1fc86c19141d9c09cfa006f6a41b21.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/0daa65376e3a5960db597a710374fac3cc1fc86c19141d9c09cfa006f6a41b21.jpg)

![1534aa1a46c6a41a4b5152b4a568b42c6200e7049c4ef67c8c9c0c8a8c0aeb28.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/1534aa1a46c6a41a4b5152b4a568b42c6200e7049c4ef67c8c9c0c8a8c0aeb28.jpg)

![16cedc6a5092e171a05a20c8b5c6529880b961ca9a9af20029682e71096327bc.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/16cedc6a5092e171a05a20c8b5c6529880b961ca9a9af20029682e71096327bc.jpg)

![4bfe1723ad1134ce29f7adace2860f4620ebc00f3066f082305f01c024ff2290.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/4bfe1723ad1134ce29f7adace2860f4620ebc00f3066f082305f01c024ff2290.jpg)

![4d0ce30971e42bb566e84f43104d7c57601b74bd730dea3b9249bc3101082a51.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/4d0ce30971e42bb566e84f43104d7c57601b74bd730dea3b9249bc3101082a51.jpg)

![51c0c1c44633b452c13cc2ab0c5e86f3eb98c92bfa4d129eaa0f3c92852f7e6d.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/51c0c1c44633b452c13cc2ab0c5e86f3eb98c92bfa4d129eaa0f3c92852f7e6d.jpg)

![8021e64c2077c032bd4a4fa5105fd1974039e94b3ab51023e5164518a9dc1bc7.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/8021e64c2077c032bd4a4fa5105fd1974039e94b3ab51023e5164518a9dc1bc7.jpg)

![adb616701d872f6970394f34ae6fc70ba747019384a5b2f903fa686e22d4e3b5.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/adb616701d872f6970394f34ae6fc70ba747019384a5b2f903fa686e22d4e3b5.jpg)

![c6ddf1b41f8ce183f4634b8cc8a7f64ca1f0a660f1e323655fb68ae337b97474.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/c6ddf1b41f8ce183f4634b8cc8a7f64ca1f0a660f1e323655fb68ae337b97474.jpg)

![dfe127800cee9860a65233e281bf7abb5de741dd1832ea6b293daa69369be747.jpg](../icml_results/4964_DR-MMSearchAgent_%20Deepening%20Reasoning%20in%20Multimodal%20Search%20Agents/tables/dfe127800cee9860a65233e281bf7abb5de741dd1832ea6b293daa69369be747.jpg)

<a id="paper-4917"></a>

## Optimizing Diversity and Quality through Base-Aligned Model Collaboration

### Images

![0262c54019961df9788d990e7063321fe7d15401fac699c0d0281ba60e02b88c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/0262c54019961df9788d990e7063321fe7d15401fac699c0d0281ba60e02b88c.jpg)

![1b1b64777afa81476fb89dd1a79b3f56c37da25033e09183baa6f57dd95dea57.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/1b1b64777afa81476fb89dd1a79b3f56c37da25033e09183baa6f57dd95dea57.jpg)

![2c8904a3d815a67644f951c323a3b869e05f0cd6d45a78f18c802c3be1445c77.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/2c8904a3d815a67644f951c323a3b869e05f0cd6d45a78f18c802c3be1445c77.jpg)

![30a83262e2eb19199f6632a6a2c9a7d985ca073abeb23f62a46d9b8fd4e72e2c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/30a83262e2eb19199f6632a6a2c9a7d985ca073abeb23f62a46d9b8fd4e72e2c.jpg)

![3dbdc74cef5fd950bbe1b9b50f226c39b9192532cd9aa086742fa6e42ab2fd99.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/3dbdc74cef5fd950bbe1b9b50f226c39b9192532cd9aa086742fa6e42ab2fd99.jpg)

![574613e5aead075321468bc357c908b3f5c913decc7304592b539a4b8e82a269.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/574613e5aead075321468bc357c908b3f5c913decc7304592b539a4b8e82a269.jpg)

![5dc2d3e145da6ce2ef73a63fc173a936444e4b32fda7eb42839ee8e828bfffc4.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/5dc2d3e145da6ce2ef73a63fc173a936444e4b32fda7eb42839ee8e828bfffc4.jpg)

![61149a11905734d98e4828a0b76aebb9b148c3837dd0c648a1e2affa42ab23cb.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/61149a11905734d98e4828a0b76aebb9b148c3837dd0c648a1e2affa42ab23cb.jpg)

![65c6e6c241f6d48fb96aa74e2078f576b171c13dee74104765cc21e9e541e843.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/65c6e6c241f6d48fb96aa74e2078f576b171c13dee74104765cc21e9e541e843.jpg)

![88bbb85f29750dd85eeb4badcbe1222c3df59d5201703caeace66cb2bce8b6f1.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/88bbb85f29750dd85eeb4badcbe1222c3df59d5201703caeace66cb2bce8b6f1.jpg)

![8c8ee5a5638c0c313b90b5ee51515a76178420050aa12c850aa11edb9b62aa14.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/8c8ee5a5638c0c313b90b5ee51515a76178420050aa12c850aa11edb9b62aa14.jpg)

![b65b41125a703d8de00f263546edaf27838e0d0adc34e23e36b0799f8848d48c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/b65b41125a703d8de00f263546edaf27838e0d0adc34e23e36b0799f8848d48c.jpg)

![bb050ccf67ca144dbc18f21c49e4d8e8ec0de5c4651e1e8e5f316d2f27a1caf2.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/bb050ccf67ca144dbc18f21c49e4d8e8ec0de5c4651e1e8e5f316d2f27a1caf2.jpg)

![c1231407316b80ac363ec93393f24c8fff868b696fc78576c16a9c47f1e9777d.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/c1231407316b80ac363ec93393f24c8fff868b696fc78576c16a9c47f1e9777d.jpg)

![d779af0f4436c0aaba14d5b6e79290f48aa97bb928975bc707a416dd0b5389dd.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/images/d779af0f4436c0aaba14d5b6e79290f48aa97bb928975bc707a416dd0b5389dd.jpg)

### Tables

![05199d524772c9836d1c7246d1770ce7c51f542f94f61e54db4cd4baabe2bc35.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/05199d524772c9836d1c7246d1770ce7c51f542f94f61e54db4cd4baabe2bc35.jpg)

![0c9defc8713dc2a7b44d4b735445283713ad76a7d3d9658f0c7a41d6a976c9ef.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/0c9defc8713dc2a7b44d4b735445283713ad76a7d3d9658f0c7a41d6a976c9ef.jpg)

![14ece19dff36ea8e53f8b3b65efea52fb442b0ed169bad122ba5e297dc26e445.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/14ece19dff36ea8e53f8b3b65efea52fb442b0ed169bad122ba5e297dc26e445.jpg)

![1b0e30efcf9a89742af6b76c04cb46a8f30a751409aec35e18e140b4fceca33f.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/1b0e30efcf9a89742af6b76c04cb46a8f30a751409aec35e18e140b4fceca33f.jpg)

![23bbcde4c3d2db4af9bc1e3d5a3e5fc56ff971986e668c417cd68605b05194f7.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/23bbcde4c3d2db4af9bc1e3d5a3e5fc56ff971986e668c417cd68605b05194f7.jpg)

![2830db14853b796c53c575377c66d3905b789f40d4c250a044dfa11e35ad6e7c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/2830db14853b796c53c575377c66d3905b789f40d4c250a044dfa11e35ad6e7c.jpg)

![2a128e18b17153231cd720085abba87dc08bf79c9862e9d97758a508ff18c441.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/2a128e18b17153231cd720085abba87dc08bf79c9862e9d97758a508ff18c441.jpg)

![2faf7845b14b9abcd0210af0d698157e66c3e5ba8fa73dffbdb5819390be8802.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/2faf7845b14b9abcd0210af0d698157e66c3e5ba8fa73dffbdb5819390be8802.jpg)

![34bdb897915e2d78703501fb841d952c10609dcf0a9992bfdbcee059968b8d1f.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/34bdb897915e2d78703501fb841d952c10609dcf0a9992bfdbcee059968b8d1f.jpg)

![6420519a172b26fcd93443d0a800fc5e56b9c04a95f6d1ee340d8fc2e9ebbe60.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/6420519a172b26fcd93443d0a800fc5e56b9c04a95f6d1ee340d8fc2e9ebbe60.jpg)

![6d51a197d11a465f1999804c5996acdd1408f9f1116dcdcc1f1926f5e12c7f0a.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/6d51a197d11a465f1999804c5996acdd1408f9f1116dcdcc1f1926f5e12c7f0a.jpg)

![7270f1822eb385b39adfe7a8c1992dbee31fc47938cd673ba917be70ef589ed1.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/7270f1822eb385b39adfe7a8c1992dbee31fc47938cd673ba917be70ef589ed1.jpg)

![7f81978600d001919ed0bb67a95f12f538f1265e2321cf1ff0e098ea9da4c609.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/7f81978600d001919ed0bb67a95f12f538f1265e2321cf1ff0e098ea9da4c609.jpg)

![b2a5edbba8832fdb08ccddc32b1c03077a0e8993cc50247535ed3aa5f205aac0.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/b2a5edbba8832fdb08ccddc32b1c03077a0e8993cc50247535ed3aa5f205aac0.jpg)

![bf6b2a9a8dc4c4a71d26a98be6556d18b30d7ba21bc35dfaa40b807a7fabb8e2.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/bf6b2a9a8dc4c4a71d26a98be6556d18b30d7ba21bc35dfaa40b807a7fabb8e2.jpg)

![c237df5c7fdd61f67233cf5ded140f79d93af2d02c19eb61190d8298c8e9e79a.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/c237df5c7fdd61f67233cf5ded140f79d93af2d02c19eb61190d8298c8e9e79a.jpg)

![c4f0a7e240d2a947dfa36d55483070b9628c614b80c49041e8c8cca0e03602d0.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/c4f0a7e240d2a947dfa36d55483070b9628c614b80c49041e8c8cca0e03602d0.jpg)

![cd3e613ad19889fdd32c454872a9aba65c16c15d056749c4d0a63b481109026c.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/cd3e613ad19889fdd32c454872a9aba65c16c15d056749c4d0a63b481109026c.jpg)

![d16863e8477e27da87879639a7ee68488ed6a5078911a20ee82e4b05b935d6b5.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/d16863e8477e27da87879639a7ee68488ed6a5078911a20ee82e4b05b935d6b5.jpg)

![d7cb340723362cb9087ee6392e5ac6466f5e3e2c99641d33cc83a17904638916.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/d7cb340723362cb9087ee6392e5ac6466f5e3e2c99641d33cc83a17904638916.jpg)

![e255d51d2e8cf326230078a7731df47bd3e9cff07d3530b8f35fc835056c23e5.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/e255d51d2e8cf326230078a7731df47bd3e9cff07d3530b8f35fc835056c23e5.jpg)

![e873269391baac5a80d98ff3c671edf105a1c56d8e68e51c7601b55d9864efbc.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/e873269391baac5a80d98ff3c671edf105a1c56d8e68e51c7601b55d9864efbc.jpg)

![ecc3243945a0fedc288d0350da4076cf30bedd5c2abd9c8bd4d511fa67511d32.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/ecc3243945a0fedc288d0350da4076cf30bedd5c2abd9c8bd4d511fa67511d32.jpg)

![ed596527fa250c89c3b42c9bc58198885c33fc7247838e0e39cdbe78e3562ef0.jpg](../icml_results/4965_Optimizing%20Diversity%20and%20Quality%20through%20Base-Aligned%20Model%20Collaboration/tables/ed596527fa250c89c3b42c9bc58198885c33fc7247838e0e39cdbe78e3562ef0.jpg)
